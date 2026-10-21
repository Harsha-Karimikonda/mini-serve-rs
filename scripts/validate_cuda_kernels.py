#!/usr/bin/env python3
"""Compile and validate docs/colab_cuda_kernels/mini_serve_kernels.cu on an active CUDA GPU."""
import argparse
import ctypes
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def reference_rmsnorm(torch, x, weight, eps):
    dtype = x.dtype
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = (x_f32 / torch.sqrt(variance + eps)).to(dtype)
    return x_normed * weight


def reference_paged_gqa(torch, q_batch, k_pool, v_pool, block_tables, seq_lens, scale):
    num_heads, head_dim = q_batch.shape[1], q_batch.shape[2]
    num_kv_heads = k_pool.shape[1]
    num_kv_groups = num_heads // num_kv_heads
    outputs = []
    for b, seq_len in enumerate(seq_lens):
        pages = block_tables[b, : (seq_len + 15) // 16].long()
        k_seq = k_pool[pages].permute(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim)[:, :seq_len, :].to(torch.float32)
        v_seq = v_pool[pages].permute(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim)[:, :seq_len, :].to(torch.float32)
        q_seq = q_batch[b].to(torch.float32)
        heads = []
        for kv_h in range(num_kv_heads):
            q_group = q_seq[kv_h * num_kv_groups : (kv_h + 1) * num_kv_groups]
            scores = (q_group @ k_seq[kv_h].T) * scale
            probs = torch.softmax(scores, dim=-1)
            heads.append(probs @ v_seq[kv_h])
        outputs.append(torch.cat(heads, dim=0).to(q_batch.dtype))
    return torch.stack(outputs, dim=0)


def run_kernel_suite(cu_path: Path, output_json: Path = None) -> dict:
    if not shutil.which("nvcc"):
        raise RuntimeError("nvcc unavailable; CUDA kernel validation requires CUDA toolkit")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(f"PyTorch unavailable for CUDA kernel validation: {exc}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False; active CUDA GPU required")

    device = torch.device("cuda:0")
    major, minor = torch.cuda.get_device_capability(device)
    arch = f"sm_{major}{minor}"
    gpu_name = torch.cuda.get_device_name(device)
    report = {
        "status": "failed",
        "gpu_name": gpu_name,
        "arch": arch,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "rmsnorm_cases": [],
        "paged_gqa_cases": [],
        "ragged_batch_cases": [],
    }

    with tempfile.TemporaryDirectory(prefix="rlse-kernels-") as tmp:
        so_path = Path(tmp) / "libmini_serve_kernels.so"
        cmd = [
            "nvcc",
            "-O3",
            f"-arch={arch}",
            "--shared",
            "-Xcompiler",
            "-fPIC",
            str(cu_path),
            "-o",
            str(so_path),
        ]
        subprocess.run(cmd, check=True)
        lib = ctypes.CDLL(str(so_path))

        for fn_name in ("launch_rmsnorm_f32", "launch_rmsnorm_f16"):
            fn = getattr(lib, fn_name)
            fn.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
            ]
            fn.restype = ctypes.c_int

        for fn_name in ("launch_paged_gqa_decode_tiled_f32", "launch_paged_gqa_decode_tiled_f16"):
            fn = getattr(lib, fn_name)
            fn.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
            ]
            fn.restype = ctypes.c_int

        # 1. Fail-closed invalid-argument checks (no process exit)
        assert lib.launch_rmsnorm_f32(None, None, None, 0, 1024, 1e-6) != 0
        assert lib.launch_paged_gqa_decode_tiled_f32(None, None, None, None, None, None, 0, 16, 8, 8, 1.0) != 0

        torch.manual_seed(42)
        eps = 1e-6
        tolerances = {
            "f32": (torch.float32, 1e-5, 1e-4, lib.launch_rmsnorm_f32, lib.launch_paged_gqa_decode_tiled_f32),
            "f16": (torch.float16, 5e-3, 5e-3, lib.launch_rmsnorm_f16, lib.launch_paged_gqa_decode_tiled_f16),
        }

        # 2. Multi-row Fused RMSNorm across hidden_size (1024) and Q/K head_dim (128)
        for dtype_name, (dtype, atol, rtol, rms_fn, _) in tolerances.items():
            for num_rows in (1, 4, 17, 64):
                for hidden_size in (128, 1024):
                    x = torch.randn(num_rows, hidden_size, device=device, dtype=dtype)
                    w = torch.randn(hidden_size, device=device, dtype=dtype)
                    out = torch.empty_like(x)
                    rc = rms_fn(
                        ctypes.c_void_p(x.data_ptr()),
                        ctypes.c_void_p(w.data_ptr()),
                        ctypes.c_void_p(out.data_ptr()),
                        num_rows,
                        hidden_size,
                        eps,
                    )
                    torch.cuda.synchronize()
                    assert rc == 0, f"RMSNorm {dtype_name} launch failed with {rc}"
                    assert torch.isfinite(out).all().item(), f"Non-finite RMSNorm {dtype_name} output"
                    ref = reference_rmsnorm(torch, x, w, eps)
                    max_abs = (out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
                    assert torch.allclose(out, ref, atol=atol, rtol=rtol), (
                        f"RMSNorm {dtype_name} rows={num_rows} dim={hidden_size} max_abs={max_abs}"
                    )
                    report["rmsnorm_cases"].append(
                        {"dtype": dtype_name, "rows": num_rows, "dim": hidden_size, "max_abs_err": max_abs}
                    )

        # 3. Single-sequence boundary lengths & fragmented page tables
        num_heads, num_kv_heads, head_dim = 16, 8, 128
        max_blocks = 32
        max_blocks_per_seq = 8
        scale = 1.0 / math.sqrt(head_dim)
        fragmented_Order = [29, 3, 18, 0, 25, 11, 7, 14]
        seq_lengths = (1, 15, 16, 17, 31, 32, 33, 64, 128)

        for dtype_name, (dtype, atol, rtol, _, attn_fn) in tolerances.items():
            k_pool = torch.randn(max_blocks, num_kv_heads, 16, head_dim, device=device, dtype=dtype)
            v_pool = torch.randn(max_blocks, num_kv_heads, 16, head_dim, device=device, dtype=dtype)
            for seq_len in seq_lengths:
                q = torch.randn(1, num_heads, head_dim, device=device, dtype=dtype)
                out = torch.empty_like(q)
                block_table = torch.tensor([fragmented_Order], device=device, dtype=torch.int32)
                seq_lens_t = torch.tensor([seq_len], device=device, dtype=torch.int32)
                rc = attn_fn(
                    ctypes.c_void_p(q.data_ptr()),
                    ctypes.c_void_p(k_pool.data_ptr()),
                    ctypes.c_void_p(v_pool.data_ptr()),
                    ctypes.c_void_p(block_table.data_ptr()),
                    ctypes.c_void_p(seq_lens_t.data_ptr()),
                    ctypes.c_void_p(out.data_ptr()),
                    1,
                    num_heads,
                    num_kv_heads,
                    max_blocks_per_seq,
                    scale,
                )
                torch.cuda.synchronize()
                assert rc == 0, f"PagedGQA {dtype_name} seq_len={seq_len} failed with {rc}"
                assert torch.isfinite(out).all().item(), f"Non-finite PagedGQA {dtype_name} output"
                ref = reference_paged_gqa(torch, q, k_pool, v_pool, block_table, [seq_len], scale)
                max_abs = (out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
                assert torch.allclose(out, ref, atol=atol, rtol=rtol), (
                    f"PagedGQA {dtype_name} seq_len={seq_len} max_abs={max_abs}"
                )
                report["paged_gqa_cases"].append(
                    {"dtype": dtype_name, "seq_len": seq_len, "max_abs_err": max_abs}
                )

            # 4. Ragged continuous batch with shared prefix page reuse (physical pages [7, 12] shared)
            ragged_lens = [1, 15, 16, 17, 31, 32, 33]
            batch_size = len(ragged_lens)
            ragged_tables = torch.tensor(
                [
                    [7, 0, 0, 0],
                    [7, 0, 0, 0],
                    [7, 0, 0, 0],
                    [7, 19, 0, 0],
                    [7, 12, 0, 0],
                    [7, 12, 0, 0],
                    [7, 12, 28, 0],
                ],
                device=device,
                dtype=torch.int32,
            )
            q_batch = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=dtype)
            out_batch = torch.empty_like(q_batch)
            seq_lens_batch = torch.tensor(ragged_lens, device=device, dtype=torch.int32)
            rc = attn_fn(
                ctypes.c_void_p(q_batch.data_ptr()),
                ctypes.c_void_p(k_pool.data_ptr()),
                ctypes.c_void_p(v_pool.data_ptr()),
                ctypes.c_void_p(ragged_tables.data_ptr()),
                ctypes.c_void_p(seq_lens_batch.data_ptr()),
                ctypes.c_void_p(out_batch.data_ptr()),
                batch_size,
                num_heads,
                num_kv_heads,
                4,
                scale,
            )
            torch.cuda.synchronize()
            assert rc == 0, f"Ragged PagedGQA {dtype_name} failed with {rc}"
            assert torch.isfinite(out_batch).all().item(), f"Non-finite ragged PagedGQA {dtype_name} output"
            ref_batch = reference_paged_gqa(torch, q_batch, k_pool, v_pool, ragged_tables, ragged_lens, scale)
            max_abs = (out_batch.to(torch.float32) - ref_batch.to(torch.float32)).abs().max().item()
            assert torch.allclose(out_batch, ref_batch, atol=atol, rtol=rtol), (
                f"Ragged PagedGQA {dtype_name} max_abs={max_abs}"
            )
            report["ragged_batch_cases"].append(
                {"dtype": dtype_name, "seq_lens": ragged_lens, "shared_prefix_pages": [7, 12], "max_abs_err": max_abs}
            )

    report["status"] = "passed"
    if output_json:
        output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"PASS: CUDA kernels verified on {gpu_name} ({arch}); "
        f"rmsnorm_cases={len(report['rmsnorm_cases'])}, "
        f"paged_gqa_cases={len(report['paged_gqa_cases'])}, "
        f"ragged_batch_cases={len(report['ragged_batch_cases'])}"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cu",
        type=Path,
        default=ROOT / "docs/colab_cuda_kernels/mini_serve_kernels.cu",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_kernel_suite(args.cu.resolve(), args.output.resolve() if args.output else None)


if __name__ == "__main__":
    sys.exit(main())

