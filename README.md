# RLSE ⚡
### **R**ust **L**LM **S**erving **E**ngine

An LLM inference engine and continuous batching control plane written in **Rust**, with an Axum HTTP gateway and Candle model execution.

Inspired by serving systems such as vLLM and TGI, **RLSE** streams tokens without a Python runtime and shares model weights across workers. Native Qwen 2/2.5 and Qwen 3 execution supports CPU, Apple Metal, and NVIDIA CUDA builds. Focused pretrained Metal inference is verified; end-to-end pretrained CUDA inference remains pending.

---

## Key Capabilities

- **Shared Model Weights**: Dedicated inference threads share immutable model weights. The Candle backend mutex currently serializes worker forwards; multiple workers do not imply parallel GPU execution.
- **Grouped-Query Attention (GQA)**: Per-KV-head grouped matmul avoids expanding the full KV cache with `repeat_kv` during decode.
- **Iteration-Level Continuous Batching**: Admits waiting requests between forward steps and removes sequences after EOS, token limits, cancellation, or errors.
- **Cooperative Cancellation**: Detects disconnected clients between forward steps and releases request state, logical blocks, and tensor reservations.
- **Logical KV-Cache Blocks**: Tracks admission capacity and free-block telemetry with configurable blocks (16 tokens by default). Actual Candle KV tensors use contiguous per-sequence buffers; tensor-backed physical paging is planned.
- **Prefix Caching & Tensor Budgets**: Chained SHA-256 prefix matching and LRU eviction reuse cached activations. A separate byte reservation budget covers KV tensors, prefix snapshots, and estimated execution workspaces, excluding model weights; it does not guarantee protection from device OOM.
- **Latency-Aware Routing**: Scores eligible workers using `(queue_depth + active + 1) * latency` and excludes unhealthy workers.
- **OpenAI-Compatible Endpoints**: `/v1/completions` and `/v1/chat/completions` support JSON and SSE responses; `/v1/models` reports the loaded model identity. Chat formatting uses the model artifact's template. See the [API reference](docs/api_reference.md) for supported fields and limits.
- **Authentication & Client Limits**: Optional bearer keys, per-client request rates, and concurrent-generation limits protect generation routes. Queue capacity provides additional overload control.
- **Telemetry & Lifecycle**: Embedded dashboard, Prometheus metrics, `/health`, `/ready`, worker supervision, bounded draining, and backend quarantine after returned execution errors.
- **Hardware Backends**: Mock execution for development and Candle CPU/Metal/CUDA with sharded safetensors loading. Standalone [Colab CUDA kernel prototypes](docs/colab_cuda_kernels/README.md) are available; custom PagedAttention integration and Modal deployment remain planned.

---

## Documentation 📚

Comprehensive technical deep-dives are located in the [`docs/`](docs/) directory:

- [**Getting Started**](docs/getting_started.md): Installation, compilation with Apple Metal / CUDA, CLI options, and tests.
- [**System Architecture**](docs/architecture.md): Systems design, Tokio multi-threaded worker model, zero-copy memory sharing, and routing.
- [**Continuous Batching & Paged KV-Cache**](docs/continuous_batching_and_kv_cache.md): Iteration-level scheduling, logical block accounting, prefix caching, and the implementation roadmap.
- [**Model Backends & Hardware Architecture**](docs/backends_and_models.md): Candle Metal integration, custom Qwen 2.5 and Qwen 3 implementations, and CUDA roadmap.
- [**API & Telemetry Reference**](docs/api_reference.md): OpenAI-compatible REST endpoints, SSE streaming protocol, and Prometheus metrics.
- [**Benchmarks & Performance Analysis**](docs/benchmarks_and_performance.md): Historical Apple M4 results comparing Rust vs. Python. Published throughput and binary sizes describe those measured builds, not guarantees for the current tree.
- [**Authentication**](docs/api_authentication.md) and [**Client Limits**](docs/client_limits.md): Bearer credentials, client identity, quota behavior, and limitations.
- [**Tensor Memory**](docs/tensor_memory.md), [**Device Failures**](docs/device_failures.md), and [**Worker Lifecycle**](docs/worker_lifecycle.md): Reservation accounting, quarantine, readiness, and shutdown.
- [**Chat Correctness**](docs/chat_correctness.md) and [**GPU Validation**](docs/gpu_validation.md): Model/template identity, repeatable inference checks, recorded results, and remaining limits.
- [**Colab CUDA Kernels**](docs/colab_cuda_kernels/README.md): Remote CUDA workflow, prototype kernels, and integration roadmap.

---

## Quickstart

### Prerequisites
* A current stable Rust toolchain via [rustup](https://rustup.rs). The locked dependency graph determines the required compiler version.
* Metal builds require Apple Silicon macOS and Xcode Command Line Tools. CUDA builds require a compatible NVIDIA GPU, driver, and CUDA toolkit on the build/runtime host.

### Build & Run
```bash
git clone https://github.com/Harsha-Karimikonda/mini-serve-rs.git
cd mini-serve-rs

# Portable mock server (starts on http://localhost:8000)
cargo run --locked --release -- --model mock --host 127.0.0.1
```

The Makefile defaults to `FEATURES=metal`, so `make run`, `make test`, and
`make lint` target Apple Metal. For portable CPU builds, use Cargo without
features, as above. For pretrained inference with local model artifacts:

```bash
# Apple Silicon Metal
cargo run --locked --release --features metal -- \
  --model models/Qwen3-0.6B --device metal --host 127.0.0.1

# NVIDIA CUDA host
cargo run --locked --release --features cuda -- \
  --model models/Qwen3-0.6B --device cuda --host 127.0.0.1
```

Supply `config.json`, `tokenizer.json`, `tokenizer_config.json`, and safetensors
weights in the local model directory. Hugging Face IDs are also accepted.
Explicit `--device metal` or `--device cuda` requires the matching build feature
and available hardware. The default model is `mock`.

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

Run the CPU regression suite:
```bash
cargo test --locked --all-targets

# Apple Metal feature build; ordinary fixtures still execute on CPU
make test
```
The default suite covers CPU tensor regressions, scheduling, streaming, validation,
stop handling, artifact-defined Qwen chat formatting, and actual model identity.
Pretrained inference and artifact-dependent token-ID checks are explicitly opt-in;
normal Metal-feature tests still use CPU fixtures. See [chat correctness and
current verification](docs/chat_correctness.md) for template checks and recorded
pretrained Metal results. Place tokenizer, tokenizer config, model config, and safetensors
in `models/Qwen3-0.6B` before running pretrained tests:

```bash
cargo test --locked --test test_qwen3 -- --ignored
# For Metal:
cargo test --locked --features metal --test test_qwen3 -- --ignored
```

Every completed change requires actual GPU inference verification. Run
`make gpu-check` with local Qwen3-0.6B weights and host Metal access. The gate
asserts an actual Metal device and checks prefix reuse, stops, EOS, shared-worker
outputs, authentication, client quotas, draining, cancellation, memory-budget
rejection/recovery, and fail-closed routing after an injected execution error
following real GPU forwards. On a CUDA host, `make cuda-check` explicitly selects
CUDA for the same integration gate; CUDA pretrained results remain unverified.
For the independent pretrained output reference check on Metal:

```bash
RLSE_GPU_DEVICE=metal RLSE_GPU_MODEL_DIR=models/Qwen3-0.6B \
  cargo test --locked --features metal --test test_pretrained_quality -- --ignored --nocapture
```

See [GPU verification results](docs/gpu_validation.md) for the recorded run and limits.

Explicitly requested model tests fail if weights are absent. CPU fixture tests and Metal compilation do not establish pretrained-model or GPU inference correctness. See the [prioritized implementation plan](docs/continuous_batching_and_kv_cache.md#52-priority-and-delivery-order) for remaining work.

Run linters and formatters:
```bash
cargo clippy --locked --all-targets -- -D warnings
cargo fmt --all --check
# Metal-feature Clippy plus formatting:
make lint
```

---

## Configuration

Configured via CLI arguments or environment variables:

| Argument | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `--host` | `RLSE_HOST` | `0.0.0.0` | Bind host address |
| `--port` | `RLSE_PORT` | `8000` | Bind port |
| `--model` | `RLSE_MODEL` | `mock` | Mock, local directory, or Hugging Face ID |
| `--device` | `RLSE_DEVICE` | `auto` | Target device (`auto`, `metal`, `cuda`, `cpu`) |
| `--num-workers` | `RLSE_NUM_WORKERS` | `1` | Worker threads sharing backend weights |
| `--allow-mock-fallback` | `RLSE_ALLOW_MOCK_FALLBACK` | `false` | Fall back to mock if model weights fail to load |
| `--max-batch-size` | `RLSE_MAX_BATCH_SIZE` | `8` | Maximum concurrent sequences per step |
| `--kv-cache-blocks`| `RLSE_KV_CACHE_BLOCKS` | `1024` | Logical blocks in the cache pool |
| `--block-size` | `RLSE_BLOCK_SIZE` | `16` | Token slots per logical block and prefix chunk |
| `--max-waiting-requests` | `RLSE_MAX_WAITING_REQUESTS` | `256` | Maximum waiting requests per worker |
| `--tensor-memory-budget-bytes` | `RLSE_TENSOR_MEMORY_BUDGET_BYTES` | `8589934592` | Shared tensor reservation budget; excludes weights |
| `--execution-headroom-bytes` | `RLSE_EXECUTION_HEADROOM_BYTES` | `268435456` | Fixed headroom within the tensor budget |
| `--shutdown-timeout-ms` | `RLSE_SHUTDOWN_TIMEOUT_MS` | `30000` | Graceful shutdown deadline in milliseconds |
| `--api-key` | `RLSE_API_KEY` | unset | Optional shared bearer credential; prefer environment |
| `--api-keys` | `RLSE_API_KEYS` | empty | Comma-separated independent client credentials |
| `--client-requests-per-minute` | `RLSE_CLIENT_REQUESTS_PER_MINUTE` | `0` | Per-client attempts per fixed 60-second window; zero disables |
| `--client-max-concurrent` | `RLSE_CLIENT_MAX_CONCURRENT` | `0` | Per-client queued and active generations; zero disables |
| `--hf-token` | `HF_TOKEN` | unset | Hugging Face download credential |
| `--log-level` | `RLSE_LOG_LEVEL` | `info` | Logging verbosity (`trace`, `debug`, `info`, `warn`, `error`) |

With authentication enabled, send `Authorization: Bearer <key>` for generation,
model listing, `/api/status`, and `/metrics`. `/health` and `/ready` stay public;
the dashboard serves a public static notice. Distinct configured keys identify
independent clients. With authentication disabled, callers share an anonymous
quota. Limits are process-local and reset on restart. See [client limits](docs/client_limits.md).

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
│ Shared Backend Weights (Arc<dyn ModelBackend>)   │
│ • Logical KV Blocks + Tensor Reservations        │
│ • 16-Token Chunk Prefix Cache (LRU)              │
│ • Compute: Mock / Candle (CPU / Metal / CUDA)    │
└──────────────────────────────────────────────────┘
```

---

## Roadmap

- [x] Pure Rust Axum HTTP & SSE streaming gateway
- [x] Iteration-level continuous batching scheduler
- [x] Client-disconnect cooperative cancellation
- [x] Logical KV-cache block accounting & prefix activation caching
- [x] Latency-aware multi-worker load balancing
- [x] Embedded real-time web dashboard & Prometheus exporter
- [x] Candle Safetensors multi-shard model loader & native Metal acceleration
- [x] API-key authentication and per-client rate/concurrency limits
- [x] Tensor-memory reservations, backend quarantine, and worker draining
- [x] Focused pretrained Metal inference and independent output regressions
- [x] Standalone Colab CUDA kernel prototypes and remote workflow
- [ ] End-to-end pretrained CUDA inference verification
- [ ] Tensor-backed physical paging and custom PagedAttention integration
- [ ] Broader output-quality, numeric, device-failure, and production-load evaluation
- [ ] Serverless Modal containerization
- [ ] Cloudflare Workers edge control plane integration
