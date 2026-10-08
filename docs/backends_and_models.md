# Model Backends & Hardware Architecture 🧠

Mini-Serve features a pluggable backend interface that isolates tensor math and hardware acceleration from request routing, scheduling, and network protocols.

---

## 1. The `ModelBackend` Trait

All model engines implement the asynchronous `ModelBackend` trait defined in [`src/backends/traits.rs`](file:///Users/hkarimkonda/Documents/mini-serve-rs/src/backends/traits.rs):

```rust
#[async_trait]
pub trait ModelBackend: Send + Sync {
    fn name(&self) -> &str;
    async fn init_sequence(&self, prompt: &str) -> Result<Vec<u32>, EngineError>;
    async fn step_batch(
        &self,
        sequences: &mut [ActiveSequence],
    ) -> Result<Vec<StepToken>, EngineError>;
}
```

- **`init_sequence`**: Encodes raw text prompt into token IDs using Hugging Face `tokenizers`.
- **`step_batch`**: Executes an iteration forward pass across all active sequences:
  - If a sequence has not completed prefill, feeds all prompt tokens.
  - If prefill is completed, feeds only the most recent token (decode pass).
  - Emits a `StepToken` struct per sequence containing `token_id`, decoded `text`, and boolean `is_eos`.

---

## 2. Supported Backends

### A. Mock Backend ([`src/backends/mock.rs`](file:///Users/hkarimkonda/Documents/mini-serve-rs/src/backends/mock.rs))
- Simulates realistic text generation and token latency without requiring model weights or GPU hardware.
- Ideal for unit testing, continuous integration (CI/CD), load testing, and UI validation.

### B. Hugging Face Candle Backend ([`src/backends/candle.rs`](file:///Users/hkarimkonda/Documents/mini-serve-rs/src/backends/candle.rs))
- Uses Hugging Face's pure-Rust `candle` framework with hardware acceleration:
  - **Apple Silicon**: Accelerated using Apple Metal shaders (`Device::Metal`).
  - **NVIDIA GPU**: Accelerated using CUDA kernels (`Device::Cuda`).
  - **CPU**: Portable fallback using multi-threaded BLAS (`Device::Cpu`).
- Memory-maps model weights from `.safetensors` files using zero-copy virtual memory.

---

## 3. Native Model Architectures: Qwen 2.5 & Qwen 3

Upstream `candle-transformers 0.8.4` lacked Metal shader kernels for custom operations (`rms_norm`, `rotary_emb`, `softmax_last_dim`) and was structured for single-sequence CLI demos.

Mini-Serve includes a unified native implementation in [`src/backends/qwen2.rs`](file:///Users/hkarimkonda/Documents/mini-serve-rs/src/backends/qwen2.rs) supporting both **Qwen 2 / 2.5** and **Qwen 3**:

### A. Primitive Metal Shader Decomposition
We replaced upstream custom ops with primitive tensor operations supported natively by Candle's Metal shaders (`reduce.metal`, `unary.metal`, `binary.metal`):

| Layer | Mathematical Operation | Implementation in `qwen2.rs` |
| :--- | :--- | :--- |
| **RMSNorm** | $\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \odot \gamma$ | `x_f32.sqr()?.mean_keepdim(D::Minus1)? + eps` followed by `.sqrt()`, `broadcast_div`, and `broadcast_mul(&self.weight)` |
| **RoPE** | $x \odot \cos + \text{rotate\_half}(x) \odot \sin$ | `rotate_half` via `narrow`, `neg`, and `cat`, followed by element-wise `broadcast_mul` and addition |
| **Softmax** | $\text{Softmax}(A)$ | Standard `candle_nn::ops::softmax(&attn_weights, D::Minus1)` |

### B. Qwen 3 Innovations Handled
1. **QK-Normalization:**
   - Qwen 3 applies RMSNorm to projected Query and Key states per-head before Rotary Positional Embeddings.
   - Handled via `q_norm` and `k_norm` layers in `Attention`:
   ```rust
   let query_states = match &self.q_norm {
       Some(norm) => norm.forward(&query_states)?,
       None => query_states,
   };
   ```
2. **Decoupled Head Dimension:**
   - In Qwen 3 (`0.6B`), `head_dim` is 128, meaning $\text{num\_heads} \times \text{head\_dim} = 16 \times 128 = 2048 \neq \text{hidden\_size} (1024)$.
   - Output attention states are reshaped to `self.num_heads * self.head_dim` before projection by `o_proj`.
3. **No-Bias Linear Attention:**
   - Automatically detects linear projections without bias (`attention_bias: false`) vs. Qwen 2's biased projections.
4. **Optional Sliding Window:**
   - Tolerates `null` sliding window configurations without serialization failures.

---

## 4. Future Backend Roadmap: Serverless Modal & `cudarc`

To scale to multi-node cloud clusters with NVIDIA H100/A100 GPUs:
1. **`cudarc` Low-Level Driver**: Rust bindings directly to the CUDA Driver and runtime API with zero C++ wrapper dependencies.
2. **Serverless Modal Containerization**: Single static binary deployed onto serverless GPU nodes that scale down to zero when idle, taking advantage of Mini-Serve's sub-second cold start.
