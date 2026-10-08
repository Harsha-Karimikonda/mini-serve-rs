# Mini-Serve Documentation 📚

Welcome to the comprehensive technical documentation for **Mini-Serve** (`mini-serve-rs`), a high-performance, fault-tolerant Large Language Model (LLM) inference engine and continuous batching control plane written in pure Rust.

---

## Documentation Index

| Guide | Description |
| :--- | :--- |
| [**1. Getting Started**](getting_started.md) | Setup instructions, prerequisites, compilation with Apple Silicon Metal or CUDA, running servers, CLI options, and tests. |
| [**2. System Architecture**](architecture.md) | High-level systems design, Tokio multi-threaded worker engine, zero-copy weight sharing, and latency-aware routing. |
| [**3. Continuous Batching & Paged KV-Cache**](continuous_batching_and_kv_cache.md) | Iteration-level scheduling algorithms, paged memory allocation, fragmentation telemetry, prefix caching, and cooperative cancellation. |
| [**4. Backends & Model Architecture**](backends_and_models.md) | The `ModelBackend` trait, Candle Metal runtime, native Qwen 2.5 and Qwen 3 implementations, Metal shader decomposition, and serverless Modal/CUDA roadmap. |
| [**5. API & Telemetry Reference**](api_reference.md) | OpenAI-compatible REST endpoints (`/v1/completions`, `/v1/chat/completions`), Server-Sent Events (SSE) streaming, Prometheus `/metrics`, and the live dashboard. |
| [**6. Performance & Benchmarks**](benchmarks_and_performance.md) | Scientific A/B benchmark methodology comparing Rust Mini-Serve vs. Python PyTorch MPS on Apple Silicon M4 with real models (`Qwen2.5-0.5B` and `Qwen3-0.6B`). |

---

## Architecture Overview

```
                      ┌──────────────────────────────────────────────┐
                      │              HTTP Client / SDK               │
                      └──────────────────────┬───────────────────────┘
                                             │ HTTP / SSE
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │        Axum Web Gateway & Dashboard          │
                      │  - POST /v1/chat/completions                 │
                      │  - POST /v1/completions                      │
                      │  - GET  /v1/models                           │
                      │  - GET  /health, /metrics, /dashboard        │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │        Latency-Aware Request Router          │
                      │  Score = (queue_depth + active + 1) * latency│
                      └──────────────┬────────────────┬──────────────┘
                                     │                │
                      ┌──────────────▼──────┐  ┌──────▼──────────────┐
                      │      Worker 1       │  │      Worker 2       │
                      │ (Continuous Batch)  │  │ (Continuous Batch)  │
                      └──────────────┬──────┘  └──────┬──────────────┘
                                     │                │
            ┌────────────────────────┴────────────────┴────────────────────────┐
            │                     Shared Memory Subsystems                     │
            │  ┌─────────────────────────┐         ┌────────────────────────┐  │
            │  │ Paged KV-Cache Allocator│         │ 16-Token Prefix Cache  │  │
            │  │ (1024 blocks, 16 tok/bl)│         │ (SHA-256 rolling hash) │  │
            │  └─────────────────────────┘         └────────────────────────┘  │
            │  ┌────────────────────────────────────────────────────────────┐  │
            │  │           Model Weights in Unified RAM (Zero-Copy)         │  │
            │  │            Candle Metal / Qwen 2.5 & Qwen 3 LM             │  │
            │  └────────────────────────────────────────────────────────────┘  │
            └──────────────────────────────────────────────────────────────────┘
```

---

## Core Tenets

1. **Zero Python Runtime Overhead:** No Python interpreter, no GIL locking, and no multi-process gRPC IPC weight duplication.
2. **Deterministic Memory Footprint:** Fixed-size paged physical KV-cache allocations protect the host OS against Out-Of-Memory crashes.
3. **Hardware-Native Shaders:** Pure standard Candle tensor operations that map directly onto Apple Silicon Metal shaders or NVIDIA CUDA PTX kernels.
4. **Resilient Serving:** Cooperative client cancellation drops dead connection processing within a single token iteration, preventing compute wastage.

