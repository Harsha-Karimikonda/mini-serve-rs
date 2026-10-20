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
| [`test_colab.ipynb`](test_colab.ipynb) | Interactive Colab notebook executable directly from the VS Code Colab extension. |

---

## 1. Why a Remote Colab Runtime Is Needed on macOS

`rlse` supports both Apple Silicon Metal (`--features metal`) and NVIDIA CUDA (`--features cuda`, via `candle-core/cuda`). Because `nvcc` and the NVIDIA CUDA Driver API (`libcuda.so`) only run on Linux/Windows hosts with physical NVIDIA GPUs, CUDA compilation and kernel execution cannot happen locally on macOS.

When connected to a Google Colab runtime via the VS Code Colab extension:
- **Local Workspace (macOS):** Edits Rust source files (`src/`, `tests/`) and `.ipynb` / `.cu` files.
- **Remote Runtime (Colab Linux VM):** Runs on a separate ephemeral VM (`/content`) attached to an NVIDIA GPU (`T4`, `A100-SXM4-40GB`, or `RTX PRO 6000 Blackwell`).

---

## 2. Fast Codebase Sync Workflow (Mac → Colab VM)

Because the Colab VM has an isolated filesystem, local files are not mounted automatically. Instead of uploading ~100 Rust files individually, [`test_colab.ipynb`](test_colab.ipynb) uses a single-command compressed archive sync:

1. **Pack on macOS (`~80 KB`):**
   ```bash
   git archive --format=tar.gz HEAD | base64
   ```
2. **Unpack on Colab VM (`/content/mini-serve-rs`):**
   - Decodes the `base64` payload into `/tmp/repo.tar.gz`, extracts it to `/content/mini-serve-rs`, and deletes `/tmp/repo.tar.gz`.
   - Kills any stale background `cargo`/`rustc` processes (`pkill -f "cargo|rustc|rustup"`) from interrupted notebook cells so `.cargo/.package-cache` locks are never held.
   - **Incremental compilation cache:** Preserves `/content/mini-serve-rs/target` across cell re-runs. The initial cold build of ~260 crates takes ~2.5–3 minutes on a Colab CPU; subsequent incremental edits and `cargo test` runs finish in **< 2 seconds**.

---

## 3. Verification Record (2026-10-20)

### 3.1 Hardware Environments Tested
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

