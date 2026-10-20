# API authentication (2026-10-20)

Set `RLSE_API_KEY` (or `--api-key`) to enable a single shared bearer key. With no
key configured, local development remains unauthenticated. An explicitly empty
key or one containing whitespace/non-ASCII characters fails startup validation.
Use a high-entropy secret; the server does not generate keys. Prefer the
environment variable to avoid placing the key in process command-line arguments.

Clients send `Authorization: Bearer <key>`. Scheme matching is case-insensitive;
key matching is exact. Missing, wrong, malformed, or duplicate authorization
headers receive HTTP 401, `WWW-Authenticate: Bearer`, and a generic JSON error.
Authentication runs before JSON extraction, chat formatting, tokenization,
queue admission, telemetry request counting, and tensor reservation. Both JSON
and SSE requests authenticate before generation starts.

| Routes | Policy when authentication is enabled |
| --- | --- |
| `/v1/completions`, `/v1/chat/completions`, `/v1/models` | Bearer key required |
| `/api/status`, `/metrics` | Bearer key required |
| `/health`, `/ready` | Public liveness/readiness; existing health metadata remains public |
| `/`, `/dashboard` | Public static notice; interactive dashboard disabled |

The dashboard never embeds, collects, or stores the key. Request logging does
not include headers; authentication errors never echo credentials. `Settings`
Debug output omits both the API key and HF token, and CLI help hides their
environment values. Middleware retains only a SHA-256 digest and compares
fixed-size digests without a matching-prefix early exit.

Additional credentials and independent request/concurrency quotas are now
available through `RLSE_API_KEYS` and the client-limit settings. See
[per-client identity, quota contract, and verification](client_limits.md).
Key changes require restart. Transport encryption belongs to the deployment
proxy. Queue limits remain separate global overload protection.

## Repeatable verification

Environment used here:

```bash
export CARGO_HOME=/private/tmp/rlse-cargo
export RUSTUP_HOME=/private/tmp/rlse-rustup
export PATH="$CARGO_HOME/bin:$PATH"
cargo fmt --all --check
cargo test --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
cargo clippy --locked --features metal --all-targets -- -D warnings
cargo test --locked --features metal --all-targets
RLSE_GPU_DEVICE=metal RLSE_GPU_MODEL_DIR=models/Qwen3-0.6B cargo test --locked --features metal --test test_gpu_inference -- --ignored --nocapture
git diff --check
```

Results: default and Metal-feature suites each passed 62 tests (two/three
explicit ignores respectively). Both Clippy configurations, formatting, and
diff checks passed. The explicit pretrained Metal gate passed one test in
53.41 seconds, including the authentication comparisons below.

CPU regressions cover rejected credentials on every protected route, malformed
JSON rejected before parsing, duplicate headers, no tokenization/admission/
allocation on rejection, public endpoints, redacted settings/errors, invalid
configuration, and successful authenticated JSON/SSE on both generation routes.
They use a controllable CPU backend named `capture`, one-token limits, and
otherwise default sampling, with a five-second completion timeout. Existing
unauthenticated mock/API coverage remains enabled.

The pretrained gate adds authenticated completion/chat JSON and SSE comparisons
against direct greedy generation, rejects missing credentials before parsing
without changing reserved bytes, and verifies cleanup. Artifact identity, actual
device, exact sampling, results, and limitations are recorded in
[GPU validation](gpu_validation.md). No commit or hosted CI result is recorded.
