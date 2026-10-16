# RLSE ⚡
### **R**ust **L**LM **S**erving **E**ngine

A blazing-fast, fault-tolerant, high-throughput LLM inference engine and continuous batching control plane written in pure **Rust**.

Inspired by production serving systems (vLLM, TGI, and Together AI), **RLSE** delivers deterministic, low-latency token streaming with **zero Python GIL contention**, **zero model weight duplication**, and a lightweight native binary footprint (**4.7 MB** core / **14.0 MB** with Metal GPU acceleration).

---

## Key Capabilities

- **Zero GIL Multi-Threading**: Serves requests across lightweight worker threads sharing model weights in memory (`Arc<ModelWeights>`) without requiring out-of-process subprocesses or IPC overhead.
- **Zero-Copy Grouped-Query Attention (GQA)**: Per-KV-head 2D grouped matmul eliminates full-cache cloning (`repeat_kv`), maintaining high decode speeds (63.1 tok/s short, 38.5 tok/s long) with zero context degradation.
- **Iteration-Level Continuous Batching**: Dynamic per-step admission admits waiting requests into active batches on every forward pass, delivering **115.0 tok/s** aggregate throughput ($C=4$) on Apple Silicon Metal. Sequences emitting EOS or reaching token limits are evicted instantly with zero batch starvation.
- **Client-Disconnect Cooperative Cancellation**: When client HTTP/SSE connections close mid-stream, the cancellation signal propagates immediately to the scheduler, freeing KV-cache blocks in real time.
- **Paged KV-Cache Block Allocator**: Logical and physical block management (16 tokens/block) with on-demand allocation, fragmentation telemetry, and fast-fail capacity checks.
- **Cross-Request Prefix Caching**: 16-token chunk-hashed prefix caching using chained SHA-256 digests and LRU eviction to skip prefill passes on recurring prompts.
- **Latency-Aware Dynamic Routing**: Distributes requests across worker threads using `(queue_depth + active + 1) * latency` to eliminate backlog bottlenecks.
- **OpenAI-Compatible Gateway**: Full support for `/v1/completions` (JSON & SSE streaming with accurate `finish_reason`), `/v1/chat/completions`, and `/v1/models`.
- **Embedded Real-Time Dashboard**: Pure HTML5/CSS3 dark-mode monitoring dashboard compiled directly into the binary at `http://localhost:8000/dashboard`.
- **Hardware Acceleration Ready**: Pluggable backend architecture supporting `MockBackend` for testing, Hugging Face **Candle** (Metal/CUDA) with multi-shard safetensors loading, and extensible to serverless **Modal** + **`cudarc`**.

---

## Documentation 📚

Comprehensive technical deep-dives are located in the [`docs/`](docs/) directory:

- [**Getting Started**](docs/getting_started.md): Installation, compilation with Apple Metal / CUDA, CLI options, and tests.
- [**System Architecture**](docs/architecture.md): Systems design, Tokio multi-threaded worker model, zero-copy memory sharing, and routing.
- [**Continuous Batching & Paged KV-Cache**](docs/continuous_batching_and_kv_cache.md): Iteration-level scheduling, paged memory allocation, fragmentation telemetry, and prefix caching.
- [**Model Backends & Hardware Architecture**](docs/backends_and_models.md): Candle Metal integration, custom Qwen 2.5 and Qwen 3 implementations, and CUDA roadmap.
- [**API & Telemetry Reference**](docs/api_reference.md): OpenAI-compatible REST endpoints, SSE streaming protocol, and Prometheus metrics.
- [**Benchmarks & Performance Analysis**](docs/benchmarks_and_performance.md): Scientific A/B benchmark methodology and Apple M4 results comparing Rust vs. Python.

---

## Quickstart

### Prerequisites
* Rust 1.80+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)

### Build & Run
```bash
git clone https://github.com/Harsha-Karimikonda/mini-serve-rs.git
cd mini-serve-rs

# Run in release mode (starts on http://localhost:8000)
make run
```

### Try It Out

1. **Open the Web Dashboard**:
   Visit **`http://localhost:8000/dashboard`** (or `http://localhost:8000/`)

2. **Stream a Completion**:
   ```bash
   curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "mock", "prompt": "Explain continuous batching in Rust", "max_tokens": 50, "stream": true}'
   ```

3. **Inspect Real-Time Engine Telemetry**:
   ```bash
   curl http://localhost:8000/api/status
   ```

4. **Prometheus Metrics**:
   ```bash
   curl http://localhost:8000/metrics
   ```

5. **Run Interactive Examples & Client Use Cases**:
   Explore pre-built examples in [`examples/`](examples/README.md):
   ```bash
   # Interactive CLI terminal chat
   cargo run --example cli_chat

   # Prefix caching & RAG acceleration
   cargo run --example prefix_cache_rag

   # Continuous batching under concurrent load
   cargo run --example concurrent_batching

   # Cooperative client cancellation demo
   cargo run --example client_cancellation
   ```

---

## Testing & Verification

Run the full automated test suite:
```bash
make test
```
The default suite runs 46 unit and integration tests, including CPU tensor regressions for interleaved workers and prefix restoration, channel saturation, cancellation, API termination, configuration/request boundaries, tensor budgeting, and shared stop handling. The pretrained Qwen3 test and four manual diagnostics (chat formatting, runtime blocking, registry contention, and real-model health latency) are explicitly ignored by default. Metal-feature builds also discover a separately ignored GPU integration gate, run explicitly below. Place its tokenizer, config, and safetensors in `models/Qwen3-0.6B`, then run:

```bash
cargo test --locked --test test_qwen3 -- --ignored
# For Metal:
cargo test --locked --features metal --test test_qwen3 -- --ignored
```

Every completed change requires actual GPU inference verification. Run
`make gpu-check` with local Qwen3-0.6B weights and host Metal access. The gate
asserts an actual Metal device and checks prefix reuse, stops, EOS, shared-worker
outputs, cancellation, and memory-budget rejection/recovery. See
[GPU verification results](docs/gpu_validation.md) for the recorded run and limits.

Explicitly requested model tests fail if weights are absent. CPU fixture tests and Metal compilation do not establish pretrained-model or GPU inference correctness. See the [prioritized implementation plan](docs/continuous_batching_and_kv_cache.md#52-priority-and-delivery-order) for remaining work.

Run linters and formatters:
```bash
make lint
```

---

## Configuration

Configured via CLI arguments or environment variables:

| Argument | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `--host` | `RLSE_HOST` | `0.0.0.0` | Bind host address |
| `--port` | `RLSE_PORT` | `8000` | Bind port |
| `--model` | `RLSE_MODEL` | `mock` | Model name or Hugging Face ID |
| `--device` | `RLSE_DEVICE` | `auto` | Target device (`auto`, `metal`, `cuda`, `cpu`) |
| `--num-workers` | `RLSE_NUM_WORKERS` | `1` | Worker threads sharing backend weights |
| `--allow-mock-fallback` | `RLSE_ALLOW_MOCK_FALLBACK` | `false` | Fall back to mock if model weights fail to load |
| `--max-batch-size` | `RLSE_MAX_BATCH_SIZE` | `8` | Maximum concurrent sequences per step |
| `--kv-cache-blocks`| `RLSE_KV_CACHE_BLOCKS` | `1024` | Total 16-token blocks in cache pool |
| `--log-level` | `RLSE_LOG_LEVEL` | `info` | Logging verbosity (`debug`, `info`, `warn`) |

---

## Architecture

```text
Client / curl / Browser
         │
         ▼ (HTTP / SSE)
┌──────────────────────────────────────────────────┐
│ Axum Gateway & Embedded Dashboard                │
│ • Prometheus Exporter (/metrics)                 │
│ • Real-time Telemetry Gauge (/api/status)        │
└────────────────────────┬─────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ Latency-Aware Router                             │
│ Score = (queue_depth + active + 1) * latency     │
└────────────┬────────────────────────┬────────────┘
             │                        │
             ▼                        ▼
┌─────────────────────────┐  ┌─────────────────────┐
│ Worker 1 Scheduler Loop │  │ Worker 2 Scheduler  │
│ • Dynamic Admission     │  │ • Dynamic Admission │
│ • Disconnect Cancel     │  │ • Disconnect Cancel │
│ • Slot Reclamation      │  │ • Slot Reclamation  │
└────────────┬────────────┘  └────────┬────────────┘
             │                        │
             └───────────┬────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│ Shared In-Memory Backend (Arc<ModelWeights>)     │
│ • Paged KV-Cache Allocator (1024 blocks)         │
│ • 16-Token Chunk Prefix Cache (LRU)              │
│ • Compute: Mock / Candle (Metal / CUDA / cudarc) │
└──────────────────────────────────────────────────┘
```

---

## Roadmap

- [x] Pure Rust Axum HTTP & SSE streaming gateway
- [x] Iteration-level continuous batching scheduler
- [x] Client-disconnect cooperative cancellation
- [x] Paged KV-cache block allocator & 16-token chunk prefix caching
- [x] Latency-aware multi-worker load balancing
- [x] Embedded real-time web dashboard & Prometheus exporter
- [x] Candle Safetensors multi-shard model loader & native Metal acceleration
- [ ] Serverless Modal containerization & `cudarc` PagedAttention kernels
- [ ] Cloudflare Workers edge control plane integration
