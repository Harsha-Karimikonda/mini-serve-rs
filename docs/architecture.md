# System Architecture & Design 🏗️

Mini-Serve is designed from the ground up to solve the architectural bottlenecks found in traditional Python-based LLM inference servers (such as GIL contention, multi-process weight duplication, and IPC serialization overhead).

---

## 1. High-Level Design Principles

1. **Shared Address Space (Zero IPC)**: In Python serving engines (e.g. FastAPI + PyTorch), avoiding GIL lockup requires multi-process architectures where each worker process replicates the entire PyTorch runtime and duplicate copies of the model weights in memory. In Mini-Serve, all worker threads and network handlers operate within a **single Rust memory space**.
2. **Read-Only Model Weights (`Arc<ModelWeights>`)**: Large tensor weight matrices are loaded into memory once and shared safely across all worker threads via atomic reference counters (`Arc`). Concurrent sequence streams allocate only lightweight transient KV caches.
3. **Asynchronous Non-Blocking Control Plane**: Axum and Tokio handle thousands of inbound HTTP connections, streaming chunks via bounded channels while worker threads execute GPU forward passes cooperatively.

---

## 2. Multi-Worker Concurrency Model

```
                          ┌─────────────────────────────┐
                          │   Inbound HTTP Connection   │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────┐
                          │      Axum HTTP Gateway      │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────┐
                          │    Latency-Aware Router     │
                          └──────────────┬──────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   │ Dispatch via Tokio Channel                │
                   ▼                                           ▼
      ┌─────────────────────────┐                 ┌─────────────────────────┐
      │     Worker Thread 1     │                 │     Worker Thread 2     │
      │  (Scheduler Event Loop) │                 │  (Scheduler Event Loop) │
      └────────────┬────────────┘                 └────────────┬────────────┘
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────┐
                          │ Shared Arc<ModelBackend>    │
                          │ - Zero memory duplication   │
                          │ - Metal/CUDA GPU Execution  │
                          └─────────────────────────────┘
```

### Worker Lifecycle
Each worker runs an independent continuous batching iteration loop managed by [`src/scheduler/scheduler.rs`](file:///Users/hkarimkonda/Documents/mini-serve-rs/src/scheduler/scheduler.rs):
1. **Admission Check**: Inspects the pending queue and current physical KV block availability.
2. **Forward Step**: Batches active sequences and invokes `backend.step_batch(&mut active_sequences)`.
3. **Streaming Dispatch**: Emits generated tokens over sequence-specific `tokio::sync::mpsc` channels directly to the client's HTTP response stream.
4. **Ejection & Free**: Sequence termination (EOS token, maximum token limit, or client disconnect) immediately returns physical KV-cache blocks to the global pool.

---

## 3. Dynamic Latency-Aware Routing

When requests arrive at the gateway, the [`Router`](file:///Users/hkarimkonda/Documents/mini-serve-rs/src/routing/router.rs) assigns them to the optimal worker using a load-scoring algorithm:

$$\text{Score} = (\text{queue\_depth} + \text{active\_sequences} + 1) \times \text{avg\_latency\_ms}$$

### Routing Properties
- **Dynamic Balancing**: Workers running compute-heavy long-sequence batches receive higher scores, naturally deflecting new incoming requests toward idle or faster workers.
- **Fail-Fast Health Checks**: Unhealthy workers are automatically filtered out. If all workers exceed their maximum waiting queue limit (`max_waiting_requests`), the router sheds load immediately with HTTP 429 (`Retry-After: 1`).

---

## 4. Memory Layout & Zero-Copy Subsystems

| Subsystem | Storage Mechanism | Thread Safety |
| :--- | :--- | :--- |
| **Model Weight Matrices** | Candle `Tensor` storage memory-mapped from `.safetensors` | `Arc<ModelForCausalLM>` (Read-only, shared) |
| **Physical KV-Cache Blocks** | Flat block vector (`Vec<Block>`) | `Arc<Mutex<KVCache>>` (Fast spinlock/parking_lot) |
| **Prefix Activation Cache** | LRU map of 16-token SHA-256 chunk hashes | `Arc<Mutex<PrefixCache>>` |
| **Token Streaming** | Bounded asynchronous channel (`mpsc`) | Per-request pair (`Sender` in worker, `Receiver` in Axum) |

