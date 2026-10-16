# Allocation and execution failures

The serving policy distinguishes capacity rejection before execution from a
returned backend execution failure. It does not attempt to recreate a Candle
device or retry a request in-process.

| Failure | Affected work | Subsequent admission |
| --- | --- | --- |
| Logical KV capacity or tensor-byte reservation pressure (`CachePressure`) | Explicit error; partially reserved request state and logical blocks are released. Prefix tensors may be evicted to make room. | Backend stays eligible. A smaller or later request may succeed. |
| Returned `BackendError` during reservation or forward execution | The worker stops admission. Candle publishes shared quarantine, drops all active model KV state/reservations and owned prefix snapshots, and prevents further forwards. Every active and queued request is terminated explicitly as its scheduler observes quarantine. | Every worker sharing that Candle backend is excluded from routing. Direct scheduler submission also fails. An independent healthy backend remains eligible. Restart the process to reload weights and initialize the device. |
| Client disconnect or shutdown | Request state and logical blocks are released at the next scheduler iteration. Queued work receives an explicit error on shutdown. | A shutdown worker rejects new work. |

Unknown returned Candle execution errors are conservatively treated as fatal,
including tensor allocation errors and returned device OOM errors. There is no
string matching for particular driver messages and no automatic retry that could
repeat an allocation failure or duplicate already streamed tokens. A reservation
rejection is a capacity signal; it does not prove the allocator or device failed.

The Candle execution mutex serializes health checks, forwards, and publication
of execution failure. A waiting worker checks health before starting its next
forward. An idle sibling checks shared health every 25 ms; this is a polling
interval, not an upper bound on failure delivery or cancellation. Running kernels
are not interrupted, and blocking execution can delay the scheduler.

`GET /health` remains public liveness and returns 200 while the HTTP server can
serve it. `GET /ready` returns `{"ready":true}` with 200 when at least one worker
has an eligible backend; otherwise it returns `{"ready":false}` with 503.
`/api/status` reports each worker's quarantine/shutdown state and an overall
`unavailable` state when none is eligible. `/metrics` exposes the aggregate
`mini_ready_workers` gauge. This does not yet supervise task panics, prove that a
scheduler task was started, or probe the GPU driver. Worker lifecycle supervision
and bounded shutdown remain separate roadmap work. Saturation routing is also
unchanged: eligibility here means health, not available queue capacity.

For work already admitted, JSON errors return HTTP 500 and SSE errors are error
frames on the existing HTTP 200 stream, without a successful finish reason or
`[DONE]`. Admission after quarantine returns HTTP 503 before tokenization,
queueing, or reservation. A request is never silently replayed. Underlying error
messages remain visible through the existing error contract.

`ModelBackend` implementations with shared state must implement `is_healthy`
and `mark_unhealthy` so all workers see quarantine. A custom implementation
using the default hooks only gets scheduler-local quarantine. Candle implements
the shared hooks. Wrappers must delegate them to the wrapped backend.

## Repeatable verification

```bash
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo clippy --locked --features metal --all-targets -- -D warnings
cargo test --locked --features metal
cargo test --locked --features metal --test test_gpu_inference -- --ignored --nocapture
```

The CPU regression file `tests/test_device_failures.rs` injects partial
reservation pressure, allocation failure, and forward failure after a token.
Both completion APIs and JSON/SSE modes are checked. Tests assert resource
cleanup, shared active/queued request termination, no forwards after quarantine,
503 before tokenization, liveness/readiness/status/metrics behavior, routing to
an independent backend, and successful work after recoverable reservation
pressure. Timeouts bound the tests. The tiny CPU Candle regression also triggers
an actual invalid-token forward error and verifies that subsequent direct
backend execution is rejected and its tensor reservations are empty.

The explicit pretrained [GPU gate](gpu_validation.md) runs local Qwen3-0.6B on
Metal and injects an error after actual GPU prefill/decode. Model identity,
settings, hardware, commands, observations, and accounting values are recorded
there. Normal Metal-feature tests use CPU fixtures and leave the GPU gate
explicitly ignored unless requested.

The default and Metal-feature suites execute 49 tests, with five and six
explicit ignores respectively. Formatting, Clippy with warnings denied in both
configurations, and both suites pass after these changes. The separate GPU gate
passes using host access. No validating commit or hosted CI execution is recorded.

## Remaining limits

No forced driver OOM, Metal device-loss event, CUDA run, or in-process recovery
has been verified. Deliberately exhausting this machine's unified memory was
not used as a pressure test: the controllable checks use byte-budget rejection
and returned-error injection. The reservation budget excludes weights and device
allocator overhead; see [tensor accounting](tensor_memory.md). It cannot prevent
all device OOMs. Panics, allocator aborts, a hung driver, process abort, and host
OOM termination may prevent cleanup or error delivery. Cleanup drops owned
Candle objects and guards; it does not establish driver-level memory reclamation.
Externally held prefix snapshot guards remain charged until dropped.
