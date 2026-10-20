# Custom CUDA Kernels & Rust FFI Architecture

This document details how [`mini_serve_kernels.cu`](mini_serve_kernels.cu) maps onto the Rust structures in [`src/cache/kv_cache.rs`](../../src/cache/kv_cache.rs) and [`src/backends/qwen2.rs`](../../src/backends/qwen2.rs).

---

## 1. Kernel 1: `fused_rmsnorm_kernel`

### Current Rust/Candle Implementation ([`src/backends/qwen2.rs`](../../src/backends/qwen2.rs))
In [`RmsNorm::forward`](../../src/backends/qwen2.rs), RMSNorm is decomposed into 6 separate primitive tensor launches so that it works on Apple Metal without custom shaders:
```rust
pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
    let norm = (variance + self.eps)?.sqrt()?;
    let x_normed = x_f32.broadcast_div(&norm)?.to_dtype(dtype)?;
    x_normed.broadcast_mul(&self.weight)
}
```
Each intermediate tensor (`x_f32`, `sqr`, `mean_keepdim`, `+ eps`, `sqrt`, `broadcast_div`, `broadcast_mul`) reads and writes global GPU memory.

### Fused CUDA Kernel Design
`fused_rmsnorm_kernel` performs the entire operation in a **single kernel launch**:
1. Each thread block (`256` threads) handles one token row (`hidden_size = 1024` for `Qwen3-0.6B`).
2. Threads accumulate partial sums of squares in registers (`sum_sq += v * v`) and reduce across `extern __shared__ float s_sum[]` in $\log_2(256) = 8$ steps.
3. Thread 0 computes `rsqrtf(s_sum[0] / hidden_size + eps)` and broadcasts `inv_rms` to the block.
4. Threads write `x_row[i] * inv_rms * weight[i]` directly to global output memory.

---

## 2. Kernel 2: `paged_gqa_decode_attention_kernel<16, 128>`

### Bridging Control-Plane `KVCache` and Tensor Execution
In the current codebase:
- [`KVCache`](../../src/cache/kv_cache.rs) allocates 16-token logical block IDs (`Vec<usize>`) per sequence from a free-block stack (`block_size = 16`).
- [`decode_attention_gqa_zero_copy`](../../src/backends/qwen2.rs) computes Grouped-Query Attention (GQA) without `repeat_kv`, but still expects a **contiguous** per-sequence Key/Value tensor (`[num_kv_heads, kv_len, head_dim]`).

`paged_gqa_decode_attention_kernel<16, 128>` bridges these two layers directly on the GPU:
- **Global Physical KV Pool Layout:**
  ```text
  k_pool / v_pool: [max_blocks, num_kv_heads, BLOCK_SIZE (16), HEAD_DIM (128)]
  ```
- **Sequence Page Table (`block_table`):**
  For a 37-token sequence (`ceil(37 / 16) = 3` blocks), `KVCache::allocate` returns 3 physical block IDs (e.g., `[2, 3, 4]` when sequential, or `[19, 3, 27]` when fragmented).
- **Non-Contiguous Indexing Formula:**
  For any logical token position `token_pos` $\in [0, \text{seq\_len})$:
  $$\text{logical\_block} = \lfloor \text{token\_pos} / 16 \rfloor, \quad \text{block\_offset} = \text{token\_pos} \bmod 16$$
  $$\text{physical\_block} = \text{block\_table}[\text{logical\_block}]$$
- **Zero-Copy GQA Mapping:**
  Each CUDA block corresponds to one Query head (`head_idx` $\in [0, 16)$) and maps to its shared KV head via integer division:
  $$\text{kv\_head\_idx} = \lfloor \text{head\_idx} / \text{num\_kv\_groups} \rfloor$$
  This preserves the zero-copy GQA property of [`decode_attention_gqa_zero_copy`](../../src/backends/qwen2.rs) while reading directly from non-contiguous 16-token physical pages.

---

## 3. Compiling & Linking with Rust (`cargo test`)

### Compilation Command (`nvcc`)
```bash
# Replace sm_75 (T4) with sm_80 (A100) or sm_120 (RTX PRO 6000 Blackwell)
nvcc -O3 -arch=sm_75 --shared -Xcompiler -fPIC \
  docs/colab_cuda_kernels/mini_serve_kernels.cu \
  -o /tmp/libmini_serve_kernels.so
```

### Rust FFI Declaration
```rust
use rlse::cache::KVCache;

#[link(name = "mini_serve_kernels")]
extern "C" {
    fn launch_mini_serve_kernels(
        d_x: *const f32,
        d_w: *const f32,
        d_rms_out: *mut f32,
        hidden_size: i32,
        eps: f32,
        d_q: *const f32,
        d_k_pool: *const f32,
        d_v_pool: *const f32,
        d_block_table: *const i32,
        d_attn_out: *mut f32,
        num_heads: i32,
        num_kv_heads: i32,
        seq_len: i32,
        scale: f32,
    );
}
```

### Running from Cargo
```bash
RUSTFLAGS="-L /tmp" LD_LIBRARY_PATH="/tmp:$LD_LIBRARY_PATH" \
  cargo test --test test_cuda_paged_kernel -- --nocapture
```

