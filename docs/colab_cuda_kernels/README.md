# Colab Remote GPU & Custom CUDA Kernel Reference

This folder documents the workflow for connecting a local macOS development workspace (`mini-serve-rs` / `rlse`) to a remote Google Colab NVIDIA GPU runtime (T4 `sm_75`, A100 `sm_80`, or RTX PRO 6000 Blackwell `sm_120`) for:
1. Syncing the Rust codebase onto the ephemeral Colab Linux VM (`x86_64-unknown-linux-gnu`).
2. Prototyping, compiling (`nvcc`), and verifying custom CUDA kernels (`mini_serve_kernels.cu`) against the reference math in [`src/backends/qwen2.rs`](../../src/backends/qwen2.rs) and the logical block allocator in [`src/cache/kv_cache.rs`](../../src/cache/kv_cache.rs).
3. Linking compiled CUDA shared libraries (`libmini_serve_kernels.so`) into Rust `cargo test` runs via FFI (`extern "C"`).

---

## Contents

| File | Description |
| :--- | :--- |
| [`README.md`](README.md) | Architecture overview, Local Mac ↔ Colab GPU sync workflow, and verification record (2026-10-20 on NVIDIA T4 & A100). |
| [`path_forward.md`](path_forward.md) | Staged CUDA correctness, kernel integration, hardware/model acceptance matrix, rollout criteria, and TPU feasibility plan. |
| [`cuda_kernels_and_ffi.md`](cuda_kernels_and_ffi.md) | Deep dive into the two prototype CUDA kernels (`fused_rmsnorm_kernel` and `paged_gqa_decode_attention_kernel<16, 128>`), memory layout, and Rust FFI linkage. |
| [`mini_serve_kernels.cu`](mini_serve_kernels.cu) | Standalone CUDA C++ source implementing fused RMSNorm and non-contiguous 16-token Paged GQA Decode Attention for `Qwen3-0.6B` (`num_heads=16, num_kv_heads=8, head_dim=128`). |
| [`test_colab.ipynb`](test_colab.ipynb) | Thin Colab interface for source upload, pinned artifacts, and evidence download. |

---

## 1. Why a Remote Colab Runtime Is Needed on macOS

`rlse` supports both Apple Silicon Metal (`--features metal`) and NVIDIA CUDA (`--features cuda`, via `candle-core/cuda`). Because `nvcc` and the NVIDIA CUDA Driver API (`libcuda.so`) only run on Linux/Windows hosts with physical NVIDIA GPUs, CUDA compilation and kernel execution cannot happen locally on macOS.

When connected to a Google Colab runtime via the VS Code Colab extension:
- **Local Workspace (macOS):** Edits Rust source files (`src/`, `tests/`) and `.ipynb` / `.cu` files.
- **Remote Runtime (Colab Linux VM):** Runs on a separate ephemeral VM (`/content`) attached to an NVIDIA GPU (`T4`, `A100-SXM4-40GB`, or `RTX PRO 6000 Blackwell`).

---

## 2. Repeatable Source Sync and Validation

Use the [validation workflow](validation_workflow.md) and thin
[`test_colab.ipynb`](test_colab.ipynb) notebook. Package the current selected source
with `python3 scripts/gpu_validation.py pack --output /tmp/rlse-source.tar.gz`,
then upload the archive, sidecar, and matching script. The harness verifies source
and pinned model hashes, runs CPU and explicit GPU checks, and exports evidence.
The notebook no longer embeds CUDA source, generated Rust tests, or a stale
archive, and does not kill unrelated processes.

The records below describe the earlier prototype notebook. They remain historical
setup evidence; the current notebook runs the repository's pretrained gates.

---

## 3. Verification Record (2026-10-20)

### 3.1 Hardware Environments Tested (historical prototype notebook)
- **NVIDIA A100-SXM4-40GB** (`Driver 580.82.07`, `CUDA 13.0`, 42.41 GB VRAM) — verified `nvidia-smi` and PyTorch CUDA runtime initialization in [`test_colab.ipynb`](test_colab.ipynb) Cells 1–2.
- **NVIDIA Tesla T4** (`sm_75`) — verified custom CUDA kernel compilation (`nvcc -O3 -arch=sm_75 --shared`), Rust toolchain bootstrap (`rustup` minimal profile), codebase unpack (`/content/mini-serve-rs`), and Rust `KVCache` + FFI test execution in [`test_colab.ipynb`](test_colab.ipynb) Cells 4–5.

### 3.2 Observed Results on Colab T4 (`2026-10-20`)
- **Codebase Sync:** Unpacked `['.githooks', '.github', '.gitignore', 'AGENTS.md', 'Cargo.lock', 'Cargo.toml', 'Makefile', 'README.md', 'docs', 'examples', 'rustfmt.toml', 'src', 'tests']` to `/content/mini-serve-rs`.
- **Rust Test Execution (`tests/test_cuda_paged_kernel.rs`):**
  ```text
  running 1 test
  Allocated physical KV blocks from Rust KVCache for 37 tokens: [2, 3, 4]
  test test_kv_cache_block_allocation_for_cuda_paged_attention ... ok

  test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
  ```

### 3.3 Scope and Remaining Limitations
- This Colab verification validates the **remote Linux/CUDA toolchain setup, standalone custom CUDA kernels (`mini_serve_kernels.cu`), and Rust `KVCache` block allocation / FFI linkage** on a Colab T4 GPU.
- Full `CandleBackend` end-to-end model inference with `--features cuda` on `Qwen/Qwen3-0.6B` (`tests/test_gpu_inference.rs`) and wiring `paged_gqa_decode_attention_kernel` into `src/backends/qwen2.rs` via `candle_core::CustomOp1` / `cudarc` remain open roadmap items under [Section 5.6 of `continuous_batching_and_kv_cache.md`](../continuous_batching_and_kv_cache.md#56-p2-physical-paging-and-deployment-expansion).

