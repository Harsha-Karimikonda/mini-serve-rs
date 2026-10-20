# Per-client generation quotas (2026-10-20)

Configure `RLSE_CLIENT_REQUESTS_PER_MINUTE` / `--client-requests-per-minute`
and `RLSE_CLIENT_MAX_CONCURRENT` / `--client-max-concurrent`. Each defaults to
zero (disabled). Quotas apply to POST completion/chat attempts only; model
listing, telemetry, and health requests do not consume generation quotas.

`RLSE_API_KEY` keeps its original meaning. `RLSE_API_KEYS` / `--api-keys`
adds comma-separated independent client credentials. Each distinct key is one
client; duplicate keys share a quota. Authentication still rejects invalid keys
before any quota lookup/admission. There is no caller-selected identity header.
With authentication disabled, all callers share one anonymous quota, including
callers supplying arbitrary Authorization or forwarded-IP headers. A deployment
needing independent clients must issue distinct credentials.

Rate accounting uses a fixed 60-second window starting with limiter creation
and reset on the first attempt after expiry. An accepted attempt consumes one
rate slot even if JSON or request validation later fails. Rejected quota attempts
do not consume a slot. This is a fixed-window allowance, so bursts spanning a
window boundary can use two successive allowances. HTTP 429 includes
`Retry-After`: remaining window time rounded up for rate exhaustion, or one
second as retry guidance for concurrency exhaustion.

Concurrency counts requests under validation/tokenization, queued work, and
active generation. An Arc-owned permit transfers from HTTP middleware through
routing retries into the queue and active sequence. Only the final owner drops
the charge. Completion, validation/admission failures, backend errors,
disconnect cancellation, panic cleanup, and shutdown reclaim it. Dropping an
SSE body cannot reclaim a permit while its synchronous forward still runs;
cooperative worker cleanup must finish first. Completed generation releases its
permit even if the HTTP client has not consumed all buffered output.

Bookkeeping has at most 128 configured credential records plus one anonymous
record. Configuration rejects more than 128 input keys before loading weights;
records have server lifetime and never grow from traffic. Keys/digests are not
exposed as telemetry labels or in logs/errors/settings Debug. Changing keys or
limits requires restart; instances enforce quotas independently, with no shared
distributed quota state. Embedded direct scheduler callers bypass HTTP quotas.

## Verification

```bash
export CARGO_HOME=/private/tmp/rlse-cargo
export RUSTUP_HOME=/private/tmp/rlse-rustup
export PATH="$CARGO_HOME/bin:$PATH"
cargo test --locked --test test_client_limits --test test_device_failures
cargo test --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --features metal --all-targets
cargo clippy --locked --features metal --all-targets -- -D warnings
RLSE_GPU_DEVICE=metal RLSE_GPU_MODEL_DIR=models/Qwen3-0.6B cargo test --locked --features metal --test test_gpu_inference -- --ignored --nocapture
cargo fmt --all --check
git diff --check
```

CPU tests use a gated backend named `gated`, one-token requests, default sampling,
and five-second timeouts. They cover isolation, rejection before tokenization,
concurrent acquisition races, deterministic rate-window expiry/rounding,
validation/error/shutdown cleanup, and disconnects while forward execution is
blocked. The device-failure fixture now explicitly registers its failure wait
before publishing forward entry; its test waits for both forwards instead of
mistaking partial reservation for completed admission. This fixes a race exposed
by native worker scheduling without weakening its cleanup assertions.

The explicit GPU gate adds per-client concurrency rejection, a second client's
matching greedy output, and reuse after disconnect cleanup. Model/device/
sampling identity and observed results are recorded in [GPU validation](gpu_validation.md).
