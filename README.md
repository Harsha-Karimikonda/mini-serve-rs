# Mini-Serve (Rust Core) ⚡

A blazing-fast, fault-tolerant, high-throughput LLM inference engine and continuous batching control plane written in pure **Rust**.

Inspired by production serving systems (vLLM, TGI, and Together AI), `mini-serve-rs` delivers deterministic, low-latency token streaming with **zero Python GIL contention**, **zero model weight duplication**, and a single **4.7MB** native binary footprint.

---

## Key Capabilities

- **Zero GIL Multi-Threading**: Serves requests across lightweight worker threads sharing model weights in memory (`Arc<ModelWeights>`) without requiring out-of-process subprocesses or IPC overhead.
- **Iteration-Level Continuous Batching**: Dynamic per-step admission admits waiting requests into active batches on every forward pass; sequences emitting EOS or reaching token limits are evicted instantly with zero batch starvation.
- **Client-Disconnect Cooperative Cancellation**: When client HTTP/SSE connections close mid-stream, the cancellation signal propagates immediately to the scheduler, freeing KV-cache blocks in real time.
- **Paged KV-Cache Block Allocator**: Logical and physical block management (16 tokens/block) with fragmentation telemetry and fast-fail capacity checks.
- **Cross-Request Prefix Caching**: 16-token chunk-hashed prefix caching using chained SHA-256 digests and LRU eviction to skip prefill passes on recurring prompts.
- **Latency-Aware Dynamic Routing**: Distributes requests across worker threads using `(queue_depth + active + 1) * latency` to eliminate backlog bottlenecks.
- **OpenAI-Compatible Gateway**: Full support for `/v1/completions` (JSON & SSE streaming), `/v1/chat/completions`, and `/v1/models`.
- **Embedded Real-Time Dashboard**: Pure HTML5/CSS3 dark-mode monitoring dashboard compiled directly into the binary at `http://localhost:8000/dashboard`.
- **Hardware Acceleration Ready**: Pluggable backend architecture supporting `MockBackend` for testing, Hugging Face **Candle** (Metal/CUDA), and extensible to serverless **Modal** + **`cudarc`**.

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

---

## Testing & Verification

Run the full automated test suite:
```bash
make test
```
*Executes all 13 unit and integration tests (continuous batching, client cancellation, paged KV-cache, prefix caching, and API handlers) in ~0.15s.*

Run linters and formatters:
```bash
make lint
```

---

## Configuration

Configured via CLI arguments or environment variables:

| Argument | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `--host` | `MINI_HOST` | `0.0.0.0` | Bind host address |
| `--port` | `MINI_PORT` | `8000` | Bind port |
| `--model` | `MINI_MODEL` | `mock` | Model name or Hugging Face ID |
| `--device` | `MINI_DEVICE` | `auto` | Target device (`auto`, `metal`, `cuda`, `cpu`) |
| `--num-workers` | `MINI_NUM_WORKERS` | `2` | Worker threads sharing backend weights |
| `--max-batch-size` | `MINI_MAX_BATCH_SIZE` | `8` | Maximum concurrent sequences per step |
| `--kv-cache-blocks`| `MINI_KV_CACHE_BLOCKS` | `1024` | Total 16-token blocks in cache pool |
| `--log-level` | `MINI_LOG_LEVEL` | `info` | Logging verbosity (`debug`, `info`, `warn`) |

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
- [ ] Candle Safetensors / GGUF model loader on Apple Silicon Metal
- [ ] Serverless Modal containerization & `cudarc` PagedAttention kernels
- [ ] Cloudflare Workers edge control plane integration
