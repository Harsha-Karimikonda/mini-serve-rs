# Inference workers and shutdown (2026-10-20)

Implemented with CPU regressions and focused pretrained Metal verification.
The 2026-10-20 explicit GPU gate passed draining, admission rejection, worker
join/cleanup, and concurrent HTTP health probes. See [current GPU evidence](gpu_validation.md#verified-current-working-tree-2026-10-20).
Earlier missing-weights results below are historical. The pre-commit rerun also
passed three CPU process-signal/deadline tests and real Metal HTTP streaming
Ctrl-C draining; see [commands and results](gpu_validation.md#pre-commit-verification-2026-10-20).
Driver hangs and forced GPU deadline expiry remain unverified.

Each scheduler runs its batching loop on a named OS thread with a private
current-thread Tokio runtime. Synchronous forward passes and contention on the
shared Candle mutex therefore run outside the HTTP runtime. The bounded waiting
queue and bounded token channels remain the communication boundary. HTTP chat
rendering and admission tokenization still run on the HTTP runtime; this change
does not establish a latency bound for unusually expensive tokenization.
Weights and the backend mutex remain shared; this is execution isolation, not
parallel GPU forward execution.

The returned supervisor handle owns the native thread handle and joins it after
cleanup. Main retains and awaits these handles. Unwinding forward panics stop
admission, quarantine the backend where supported, fail active/queued streams,
and release request state and logical allocations. Returned backend errors keep
the existing quarantine policy. Routing retries other eligible workers when a
selected worker's queue fills during admission, preserving the requested model.
Worker loops can be started only once. Dropping/aborting a supervisor requests
cooperative shutdown; it cannot forcibly stop the native thread.

Ctrl-C stops admission and starts draining accepted requests. Configure
`--shutdown-timeout-ms` / `RLSE_SHUTDOWN_TIMEOUT_MS` (default 30000; zero means
immediate shutdown). Workers cancel remaining requests between steps at their
deadline. Main bounds HTTP draining and worker joins; deadline expiry exits the
process with status 1. A running synchronous kernel cannot be safely interrupted.
Consequently the hard process deadline may terminate before allocation cleanup
or terminal delivery; only cooperative shutdown promises those outcomes. Slow
HTTP consumers can also cause process deadline expiry. Embedded library callers
must retain/await handles and impose their own process policy for stuck kernels.
Driver hangs, aborting panics, panics during backend cleanup/reservation, host OOM,
and signals other than Ctrl-C are outside the tested recovery contract.

## Verification

Exact environment and commands:

```bash
export CARGO_HOME=/private/tmp/rlse-cargo
export RUSTUP_HOME=/private/tmp/rlse-rustup
export PATH="$CARGO_HOME/bin:$PATH"
cargo fmt --all --check
cargo test --locked --test test_worker_lifecycle
cargo test --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --features metal --all-targets
cargo clippy --locked --features metal --all-targets -- -D warnings
RLSE_GPU_DEVICE=metal cargo test --locked --features metal --test test_gpu_inference -- --ignored --nocapture
git diff --check
```

Default suite: 60 passed, zero failed, two explicit ignores. Metal-feature
suite: 60 passed, zero failed, three explicit ignores. Both Clippy configurations
and formatting pass. Four new lifecycle tests verify:

- A deliberately synchronous 300 ms forward allows `/health` and `/api/status`
  to complete within 100 ms on a single-thread HTTP runtime. This is a CPU test
  bound, not a production latency guarantee.
- A forward panic fails active and queued streams, removes routing eligibility,
  and releases backend reservations and logical blocks.
- Shutdown rejects new requests, cancels after its deadline between steps, and
  can finish already accepted work and join normally.
- A full waiting queue routes inference to an available worker. The previous
  routing-score test now keeps queues stationary so inference cannot finish
  before its queue-depth assertion.

These tests use controllable CPU backends (mock identity `mock`, blocking identity
`blocked`), no pretrained artifact, and default sampling except explicit token
limits. They do not establish GPU inference correctness. Existing small Candle
regressions also select CPU explicitly, including in Metal builds.

The explicit Metal gate reports **zero passed, one failed**, at prerequisite
`local pretrained weights are required`. No actual device was initialized,
no pretrained artifact loaded, and no inference sampling executed. `models/`
is absent. This is a blocker, not a passed or silently skipped GPU check.

The repeatable GPU gate now additionally drains an accepted 12-token greedy
request on a dedicated worker, compares output with isolated generation, rejects
new admission, probes HTTP health with a 250 ms bound, joins the worker, and
asserts zero active KV reservations/logical blocks. Expected artifact:
`Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`,
local directory `models/Qwen3-0.6B`; explicit `RLSE_GPU_DEVICE=metal` (or `cuda`
with the corresponding feature), temperature 0, top-p 1. See
[chat verification prerequisites](chat_correctness.md#gpu-gate-and-colabcuda)
and [historical GPU evidence](gpu_validation.md). CUDA, process-signal/deadline integration, and broader load testing remain
unverified. No validating commit or hosted CI result is recorded.
