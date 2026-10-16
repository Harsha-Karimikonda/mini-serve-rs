# Production blocker reproductions

Executed on **2026-10-16** against the current working tree. These are focused reproductions, not a production load test or a throughput benchmark. Implementation work continued during the review: tensor reservations and shared stop handling were present in the code tested here.

## Results

| Claim | Observed result | Scope |
| :--- | :--- | :--- |
| **A: No tensor-byte budget** | **No longer reproduced as stated.** Five targeted memory tests passed: concurrent reservations, shared storage lifetime, prefix eviction, actual CPU KV capacity, cancellation, and backend-error cleanup. | The reservation system rejects excess estimated allocations. It does not impose a driver-level VRAM limit or include model weights and all private allocator/kernel overhead. No GPU OOM was deliberately induced. |
| **B: Synchronous inference delays HTTP servicing** | **Reproduced.** A controlled 250 ms synchronous forward delayed a ready `/health` probe by **255.31 ms**, versus **0.030 ms** idle. A real Qwen3 Metal prefill delayed the probe by **669.46 ms**, versus **0.171 ms** idle. An earlier real-model run measured **871.67 ms**. | Both probes use an in-process Axum request on a single-thread Tokio runtime. This establishes runtime-worker blocking, not dropped TCP handshakes or whole-pool starvation in the server's default multi-thread runtime. |
| **C: Generic chat formatting** | **Reproduced.** The backend received `user: Hello\nassistant: ` without model-specific role delimiters. | The test captures the actual route's prompt. Its expected Qwen-style delimiters are a fixture, not a universal chat template. It does not measure hallucinations or pretrained chat quality. |
| **C: Stops spanning tokens / matched stop leakage** | **Not reproduced in the current implementation.** Actual tiny Candle forwards emit repeated `User` tokens. Both `User User` across decoded fragments and a single `User` stop return empty visible text with `finish_reason=stop`. | Two desired-behavior tests passed and remain enabled in the normal suite. This verifies those cases, not every stop overlap, tokenizer, or pretrained model. |
| **D: Shared backend lock contention** | **Reproduced.** Holding the actual model-state mutex for 150 ms blocked an independent tiny Candle forward; completion took **158.74 ms**. Two simultaneous real-model prefills took **1.102 s** wall time; two sequential calls took **1.135 s**. Individual simultaneous-call latencies were **0.549 s** and **1.101 s**. | The real-model calls started together on separate OS threads and used the same backend. Greedy first-token IDs agreed with sequential calls and request reservations were released. The single timing comparison showed about **1.03×**, not 2×, speedup; it is not a statistically established scaling limit. |

The real-model run used local `models/Qwen3-0.6B` weights, greedy sampling, a **353-token prompt**, and an actual reported `Metal` device. Timings come from an unoptimized test build, with model loading excluded. Prefix reuse was disabled. The host GPU run succeeded; a sandboxed attempt panicked during Metal initialization before inference and contributes no inference result.

After adding the diagnostics, the normal default and Metal-feature `--all-targets` suites each passed **46 tests**, with **five explicitly ignored tests**. The ignored cases are the two known-gap assertions, two opt-in diagnostics, and the existing pretrained smoke test. The real-model diagnostic was executed separately as described above. Metal-feature Clippy across all targets with warnings denied, formatting, and documentation link/whitespace checks also passed.

## Reproduction commands

Run from the repository root with Rust installed. `--offline` assumes dependencies are already cached; omit it when fetching dependencies is needed.

### CPU fixtures and controlled probes

```bash
cargo test --locked --offline --features metal --lib production_reproductions -- \
  --include-ignored --nocapture --test-threads=1 --skip real_model
```

At the tested snapshot, this command runs five cases: **three pass and two fail**. The chat-formatting and controlled health-responsiveness tests intentionally assert desired behavior, so they fail while those gaps remain. They are explicitly ignored in the normal suite. The two stop tests are normal regression tests; the mutex probe is an opt-in diagnostic.

These fixtures run on CPU even with the Metal feature compiled. Omit `--features metal` on a CPU-only build.

### Real-model Metal probes

```bash
MINI_REPRO_DEVICE=metal cargo test --locked --offline --features metal --lib \
  production_reproductions::real_model_health_latency -- \
  --ignored --nocapture --test-threads=1
```

Requires local weights and host Metal access. This diagnostic performs the real health probe and the sequential/simultaneous prefill comparison. It checks completion consistency and cleanup; timings are printed without a hardware-independent performance threshold. Use `MINI_REPRO_DEVICE=cpu` and omit the Metal feature for a CPU measurement, which was not executed for the pretrained model in this run.

### Memory-budget behavior

```bash
cargo test --locked --offline --features metal --lib tensor_reservations -- --nocapture
cargo test --locked --offline --features metal --test test_tensor_memory -- --nocapture
```

The first command executes two Candle CPU-fixture tests, including backend-failure cleanup. The second executes three reservation/prefix-storage tests. All five passed. Their assertions distinguish reserved bytes from actual owned CPU storage where applicable; they do not verify device-wide memory safety.

## Implementation and follow-up

Diagnostics live in [`production_reproductions.rs`](../src/backends/production_reproductions.rs), compiled only under `cfg(test)` from the Candle module. They exercise existing code without fixing the production paths. Existing memory coverage lives in [`test_tensor_memory.rs`](../tests/test_tensor_memory.rs) and the Candle correctness tests.

Use these results to update the [improvement plan](continuous_batching_and_kv_cache.md#5-implementation-status--active-roadmap):

- Keep inference execution isolation and model-specific chat templates open. Once fixed, enable their desired-behavior tests in the normal suite.
- Preserve the passing stop and memory regressions; qualify memory safety claims by the documented estimation exclusions.
- Evaluate a dedicated inference owner per device with bounded request queues and shared batching. Removing the mutex alone does not establish GPU throughput gains.
- Repeat measurements under the actual multi-thread server and network load before claiming frozen health probes, lost connections, or a production latency bound.
- Validate memory-pressure behavior on supported hardware without treating driver OOM, recoverable backend errors, and process termination as equivalent outcomes.
