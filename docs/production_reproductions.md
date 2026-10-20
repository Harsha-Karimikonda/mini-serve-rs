# Production blocker investigation record

**2026-10-20 update:** The chat-format assertion below is now enabled and passes
with backend-owned artifact templates. Its old ignored-test commands and
expected failure describe the historical investigation. GPU validation of the
new change remains pending; see [current verification](chat_correctness.md).

Historical measurements executed on **2026-10-16** against the working tree at the time. These are focused reproductions, not a production load test or a throughput benchmark. Implementation work continued during the review: tensor reservations and shared stop handling were present in the code tested here.

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

## Retained regression checks

The manual latency and mutex-contention probes have been removed. The measurements above are retained as historical investigation evidence; their original commands are no longer available. They do not establish the current implementation's latency or contention behavior.

The two stop regressions now live in `src/backends/candle.rs` under `correctness_tests`. They use a zero-weight tiny Qwen CPU fixture (64 vocabulary entries, 32 hidden dimensions, two layers), with greedy sampling (`temperature=0`, `top_p=1`, `max_tokens=4`), to check single-token and spanning stops. The ignored chat-template expectation now lives in `tests/test_api.rs` with its capture backend. It still asserts the desired Qwen-style template, which the route does not yet apply.

```bash
cargo test --locked --offline --lib correctness_tests -- --nocapture
cargo test --locked --offline --test test_api --test test_stop_handling
cargo test --locked --offline --test test_api chat_route_applies_role_delimiters -- --ignored --nocapture
```

The last command is expected to fail until model-specific chat formatting is implemented. It exercises a capture backend, with no pretrained inference.

The retained pretrained GPU regression explicitly selects `metal` and asserts the actual device is Metal. It covers single-fragment, spanning, and incomplete stops through both JSON and SSE, alongside prefix reuse, shared workers, and resource cleanup:

```bash
cargo test --locked --offline --features metal --test test_gpu_inference pretrained_gpu_memory_prefix_stops_and_workers -- --ignored --nocapture --test-threads=1
```

It uses local `models/Qwen3-0.6B` pretrained weights with greedy sampling (`temperature=0`, `top_p=1`); stop checks use a 12-token limit and the prompt `The capital of France is`.

### Memory-budget behavior

```bash
cargo test --locked --offline --features metal --lib tensor_reservations -- --nocapture
cargo test --locked --offline --features metal --test test_tensor_memory -- --nocapture
```

The first command executes two Candle CPU-fixture tests, including backend-failure cleanup. The second executes three reservation/prefix-storage tests. All five passed. Their assertions distinguish reserved bytes from actual owned CPU storage where applicable; they do not verify device-wide memory safety.

## Implementation and follow-up

The former diagnostic module has been removed; retained coverage lives in the [Candle correctness tests](../src/backends/candle.rs), [API tests](../tests/test_api.rs), and [pretrained GPU regression](../tests/test_gpu_inference.rs). This cleanup changes test organization without changing serving behavior. Existing memory coverage lives in [`test_tensor_memory.rs`](../tests/test_tensor_memory.rs) and the Candle correctness tests.

Use these results to update the [improvement plan](continuous_batching_and_kv_cache.md#5-implementation-status--active-roadmap):

- Keep inference execution isolation and model-specific chat templates open. Once chat formatting is fixed, enable its desired-behavior test in the normal suite; add an execution-isolation regression when that work is implemented.
- Preserve the passing stop and memory regressions; qualify memory safety claims by the documented estimation exclusions.
- Evaluate a dedicated inference owner per device with bounded request queues and shared batching. Removing the mutex alone does not establish GPU throughput gains.
- Repeat measurements under the actual multi-thread server and network load before claiming frozen health probes, lost connections, or a production latency bound.
- Validate memory-pressure behavior on supported hardware without treating driver OOM, recoverable backend errors, and process termination as equivalent outcomes.

## Cleanup verification (2026-10-16)

The cleanup removed `src/backends/production_reproductions.rs`, moved its two enabled stop tests into Candle correctness tests, and moved its ignored chat-template test into API coverage. The manual health-latency and contention probes were removed; historical findings remain above.

Executed the exact retained-regression commands listed above:

- CPU Candle correctness tests: **9 passed**, including both moved stop regressions. Both produced empty visible text with `finish_reason=stop` on the explicit `Device::Cpu` fixture.
- API and shared-stop integration tests: **11 passed, 1 ignored** (the known chat-template gap).
- Explicit ignored chat-template run: **1 failed as expected**. Captured prompt remained `user: Hello\nassistant: ` rather than the expected Qwen delimiters. This test remains ignored; no chat-formatting fix is claimed.
- Pretrained Metal regression: **1 passed** in **44.70 s** with actual device `Metal(MetalDevice(DeviceId(1)))`, using the command above on the host. The initial sandboxed run panicked during Metal initialization before inference; it supplies no GPU evidence. The host run completed real pretrained forwards and all assertions.
- `cargo fmt --all --check` and `git diff --check`: passed.

For the GPU stop cases, the reference output was ` Paris. The capital of Italy is Rome. The capital of`. Stop ` Paris` returned empty output; spanning stop `. The` returned ` Paris`; an incomplete longer stop preserved the reference output through the 12-token limit. Both JSON and SSE agreed. Prefix comparisons, EOS suppression, shared-worker consistency, cancellation, reservation rejection/recovery, and injected-error cleanup also passed in the retained GPU test.

Local artifact identity (SHA-256):

```text
f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b  models/Qwen3-0.6B/model.safetensors
660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd  models/Qwen3-0.6B/config.json
aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4  models/Qwen3-0.6B/tokenizer.json
```

Sampling settings and prompts are specified with the commands above. Limitations: no CUDA run, no new production-network latency measurements, and no driver-level OOM claim. The pretrained test checks existing serving behavior; it does not resolve the ignored chat-template gap or revalidate the deleted timing probes.
