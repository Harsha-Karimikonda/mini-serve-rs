# Actual GPU inference verification

Completed changes must include an actual GPU inference run, in addition to CPU
regressions and feature builds. This project rule is recorded in [AGENTS.md](../AGENTS.md).
If GPU access or weights are unavailable, GPU verification stays pending.

## Root README refresh verification: 2026-10-20

Documentation-only change: aligned the root README with all 21 CLI settings,
authentication and client limits, logical KV accounting and contiguous tensor
storage, serialized shared-backend execution, portable startup, explicit GPU
commands, and the current CUDA/paging roadmap. Historical performance figures
are linked with their original measurement scope. No runtime code changed.
All local README link targets and all Settings flag names were checked against
the tree; `git diff --check` passed.

Exact regression commands:

```bash
export CARGO_HOME=/private/tmp/rlse-cargo
export RUSTUP_HOME=/private/tmp/rlse-rustup
export PATH="$CARGO_HOME/bin:$PATH"
cargo test --locked --all-targets
RLSE_GPU_DEVICE=metal RLSE_GPU_MODEL_DIR=models/Qwen3-0.6B cargo test --locked --features metal --test test_gpu_inference --test test_pretrained_quality -- --ignored --nocapture
shasum -a 256 models/Qwen3-0.6B/*
git diff --check
```

Initial bare `cargo` calls were unavailable in the shell PATH. The reruns used
the toolchain environment above and the absolute Cargo executable
`/private/tmp/rlse-cargo/bin/cargo`. Sandbox runs could not bind sockets for the
three process tests; Metal discovery panicked before establishing a device.
Both suites were rerun successfully with host access; those initial attempts
are not GPU evidence.

CPU result: **75 passed**, zero failed, two opt-in tests ignored, including all
three process shutdown tests. GPU integration: **1 passed** in **40.44 s**.
Independent pretrained reference: **1 passed** in **5.16 s**. Both GPU tests
asserted and printed actual device `Metal(MetalDevice(DeviceId(1)))` on Apple M4,
using local `models/Qwen3-0.6B`, revision
`c1899de289a04d12100db370d81485cdf75e47ca`, Candle 0.11, F16 weights,
debug/test build. All four artifact hashes match the identity table below.

Sampling: greedy, temperature 0, top-p 1; integration completion/chat and prefix/
worker probes use 12-token limits, EOS 32, cancellation 512, reservation rejection
4000, and injected-error 128. Independent reference cases allow 32 tokens each.
Integration checks passed prefix equivalence/accounting, JSON/SSE authentication,
client isolation and quota rejection/recovery, stops/EOS, shared-worker outputs,
cancellation, draining, reservation rejection/recovery, and returned-error
quarantine. Counters: 31 accepted, 27 completed, 2 errors, 2 cancellations,
279 generated tokens. Active/prefix reservations and logical blocks were released;
fixed headroom remained 33,554,432 bytes; peak reservation was 188,183,808 bytes.
Independent reference token IDs/text matched all four cases: instruction
(`Paris.`, 7 tokens), role context (`cobalt`, 7), Spanish (`Hola`, 6), and stop
(`red green blue.`, 9). Counts include thinking delimiters and EOS; the thinking
delimiters remain visible, and the stop case retains `red` after them.

Limits: this run validates documented existing behaviors, not throughput or
binary size. CUDA pretrained inference, physical PagedAttention integration,
broader quality evaluation, numeric logit tolerances, driver OOM/hang recovery,
production load, and hosted CI remain unverified.

## Pre-commit verification: 2026-10-20

This run covers the client quota, execution telemetry, SSE benchmark, prefix
accounting, worker/shutdown changes, and the independent generation reference.
The Colab notebook now lives at
[`colab_cuda_kernels/test_colab.ipynb`](colab_cuda_kernels/test_colab.ipynb);
its README links were updated and its JSON parses (six cells). Relocation does
not constitute a new CUDA execution result.

Hardware/model identity: Apple M4; explicit actual device
`Metal(MetalDevice(DeviceId(1)))`; local `models/Qwen3-0.6B`, pinned revision
`c1899de289a04d12100db370d81485cdf75e47ca`. The four artifact SHA-256 values
were rechecked and match the table below. GPU weights use F16; debug/test build,
Candle 0.11. Sampling is greedy, temperature 0, top-p 1. The main integration
probe uses the token limits described below; independent generation uses at
most 32 tokens per case. CPU fixture generation uses Transformers 4.57.1,
PyTorch 2.8.0, CPU/F32 eager attention, four threads, and no sampling.

Exact commands (CPU socket tests and GPU runs require host access):

```bash
export CARGO_HOME=/private/tmp/rlse-cargo
export RUSTUP_HOME=/private/tmp/rlse-rustup
export PATH="$CARGO_HOME/bin:$PATH"
/private/tmp/rlse-reference/bin/python tests/fixtures/generation/generate_reference.py --model models/Qwen3-0.6B
shasum -a 256 models/Qwen3-0.6B/*
cargo fmt --all --check
cargo test --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
cargo clippy --locked --features metal --all-targets -- -D warnings
RLSE_GPU_DEVICE=metal RLSE_GPU_MODEL_DIR=models/Qwen3-0.6B cargo test --locked --features metal --test test_gpu_inference --test test_pretrained_quality -- --ignored --nocapture
cargo build --locked --features metal --bin rlse --example bench
python3 scripts/validate_gpu_server.py --device metal --model models/Qwen3-0.6B --output /private/tmp/rlse-commit-server.log
git diff --check
```

CPU: **75 passed**, zero failed, two opt-in tests ignored. Formatting and both
Clippy configurations passed. The initial sandbox CPU run could not bind local
sockets for process tests; the host rerun passed all three process shutdown
regressions (idle Ctrl-C, active HTTP stream draining, and zero deadline).
The new reference test needed a concrete message-vector type annotation and its
missing generated JSON fixture; both are included in this commit.

GPU integration: **1 passed** in 39.12 seconds, including quota 429/isolation/
disconnect cleanup, prefix equivalence and truthful saved-token counters,
authenticated JSON/SSE, stops/EOS, worker draining, reservations, and injected
returned-error quarantine. Final execution counters: 31 accepted, 27 completed,
2 errors, 2 cancellations, 279 generated tokens. All active/prefix reservations
and logical blocks were released; fixed headroom remains 33,554,432 bytes,
peak reservation 188,183,808 bytes. These are accounting estimates.

Independent reference GPU test: **1 passed** in 5.14 seconds. All four cases
match independently generated CPU/F32 token IDs and text exactly: instruction
(`Paris.`, 7 tokens), role context (`cobalt`, 7), Spanish (`Hola`, 6), and stop
(`red green blue.`, 9; visible suffix `red` after the configured stop). Counts
include thinking delimiters and EOS. Default thinking delimiters remain in
visible output. This is a four-case regression, not a broad quality evaluation.
Real HTTP GPU server: three rounds each of warm/cold prefix benchmarks, one
warmup per mode, concurrency 2, and 12-token limits passed. All warm rounds
saved prefix tokens; cold rounds saved zero. After benchmarking: 14 accepted,
14 completed, 168 generated tokens, zero errors/cancellations and zero allocated
logical blocks. Ctrl-C after the first SSE token drained the 12-token request
and exited with status 0 in 1.613 seconds. The validator now explicitly sets
`RUST_LOG=rlse=info` so an inherited warning-only filter cannot hide its required
actual-device startup evidence; the initial attempt lacked that log line.

CUDA pretrained inference, numeric logit tolerances, forced driver OOM/hangs,
production load, and hosted CI remain unverified. Timings are correctness-run
durations and do not establish throughput.

## Verified current working tree: 2026-10-20

The explicit Metal gate passed after downloading the pinned local artifacts.
This supersedes the earlier missing-weights blockers in the chat/worker records.

- Hardware: Apple M4, 8 GPU cores, 16 GB unified memory, Mac16,2.
- Actual device asserted and printed: `Metal(MetalDevice(DeviceId(1)))`.
- Model: `Qwen/Qwen3-0.6B`, revision
  `c1899de289a04d12100db370d81485cdf75e47ca`, directory `models/Qwen3-0.6B`.
- Build: debug/test, Candle 0.11, `--locked --features metal`, F16 weights.
- Sampling: temperature 0, top-p 1, 12 output tokens for completion/chat,
  prefix/worker comparisons; EOS probe 32; cancellation probe 512; reservation
  rejection probe 4000; returned-error probe 128. These are token limits, not
  necessarily actual generated counts.
- Result: **1 passed, zero failed, zero ignored**, 53.41 seconds (53.374 seconds
  inside the test including initialization). Concurrent CPU checks ran during
  this correctness test; this duration is not a performance measurement.

Artifact SHA-256:

| File | SHA-256 |
| --- | --- |
| `config.json` | `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` |
| `tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `tokenizer_config.json` | `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101` |
| `model.safetensors` | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |

Exact setup/download and verification commands (weights are gitignored):

```bash
mkdir -p models/Qwen3-0.6B
for artifact in config.json tokenizer.json tokenizer_config.json model.safetensors; do
  curl -fL --retry 2 "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/$artifact" -o "models/Qwen3-0.6B/$artifact"
done
shasum -a 256 models/Qwen3-0.6B/*
system_profiler SPDisplaysDataType SPHardwareDataType
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

Network downloads and the GPU command used approved host access. Normal CPU and
Metal-feature suites each passed **62 tests**; two/three opt-in tests respectively
were ignored. Both Clippy configurations, formatting, and diff checks passed.
The explicit GPU command executes the normally ignored gate and rejects CPU
fallback; normal suite passes alone are not GPU evidence.

New observed checks:

- [Authentication](api_authentication.md): missing credentials on completion and
  chat receive 401 before parsing malformed JSON and without increasing reserved
  bytes. Valid bearer credentials work on JSON and SSE for both routes. Output
  equals direct greedy execution; terminal delivery and allocation cleanup pass.
  CPU regressions additionally cover all protected/public routes and invalid keys.
- [Chat formatting](chat_correctness.md): local template and nine-token `Hello`
  prompt match pinned Transformers reference text/token IDs. JSON/SSE chat output
  equals direct inference on that reference prompt. Output is
  `<think>\nOkay, the user just said "Hello". I`, with 9 input, 12 output, and
  21 total tokens, finish reason `length`. This model's default thinking template
  remains enabled; this is protocol/template correctness, not broad quality.
- [Worker lifecycle](worker_lifecycle.md): a dedicated inference worker drains
  accepted 12-token greedy work, matches isolated output, rejects new admission,
  serves five HTTP health probes within 250 ms each, joins, and releases active
  reservations and logical blocks.
- Existing exact/partial/missing-logit prefix comparisons, stops, EOS, concurrent
  workers, cancellation, budget rejection/recovery, and injected returned-error
  quarantine all passed again. Baseline France output remains
  ` Paris. The capital of Italy is Rome. The capital of` (5 input/12 output).

Peak reserved bytes were 188,183,808 within a 536,870,912-byte budget. Active KV,
execution, and prefix charges ended at zero, leaving 33,554,432 bytes of fixed
headroom; one reservation was rejected. These are estimates, not allocator/RSS
measurements. GPU chat is compared with direct inference using independently
pinned formatting/token IDs, not independently generated reference-model logits.
CUDA, broad output quality, numeric GPU logit tolerances, forced driver OOM,
process-signal/deadline integration, production load, hosted CI, and a validating
commit remain unverified/unrecorded. Historical runs below retain their original
scope and counts.

## Repeatable check

Prerequisites: macOS with a Metal GPU and local `config.json`, `tokenizer.json`,
`tokenizer_config.json`, and `model.safetensors` in `models/Qwen3-0.6B`.

```bash
cargo test --locked --features metal --test test_gpu_inference -- --ignored --nocapture
# Equivalent convenience target:
make gpu-check
```

The test explicitly selects `metal`, asserts the backend's actual device is
`Device::Metal`, and fails if weights are missing. It does not download weights
or fall back to CPU. It is ignored by normal test runs so a feature build cannot
be confused with executing pretrained GPU inference. In the restricted agent
sandbox, use approved host GPU access; the previous sandbox-only Metal attempt
failed during device initialization.

## Verified run: 2026-10-16

- Hardware: Apple M4, 8 GPU cores, 16 GB unified memory (Mac16,2).
- Actual backend: `Metal(MetalDevice(DeviceId(1)))`.
- Build: debug/test profile, `--locked --features metal`; Candle 0.11.
- Model: local Qwen/Qwen3-0.6B safetensors, loaded using F16 on Metal.
- Sampling: greedy, `temperature=0`, `top_p=1`. Comparison and stop requests use
  12 maximum generated tokens; the natural-EOS probe uses 32.
- Tensor reservation budget: 536,870,912 bytes (512 MiB), including 33,554,432
  bytes (32 MiB) of fixed headroom, shared by two schedulers.
- Result: **1 GPU integration test passed**, zero failures, in 23.86 seconds
  (23.80 seconds measured inside the test, including model initialization).
  This is a correctness check, not a throughput benchmark.

The local artifacts were identified by SHA-256; the original upstream commit
revision was not retained, so these hashes identify the exact validated files:

| Artifact | SHA-256 |
| --- | --- |
| `config.json` | `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` |
| `tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `model.safetensors` | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |

## Changes and observed behavior

| Completed change | GPU verification and observed result |
| --- | --- |
| Prompt/output limits and request-sized KV buffers | Real prompt/output lengths execute within model and logical limits. The standard prompt uses 5 input and 12 completion tokens, with an explicit `length` terminal state and consistent usage. |
| Prefix snapshots and replay | A prompt padded with repeated ` France` tokens to exactly 32 tokens produces identical greedy text on uncached and cached requests. Exact cached logits, exact snapshots without saved logits (primed by a longer 34-token prompt), and partial-prefix restoration all match their uncached references. |
| Shared stop matching | JSON and SSE with stop ` Paris` return empty text and `finish_reason=stop`. A stop spanning decoded fragments, `. The`, returns ` Paris` and `stop` without exposing the marker. |
| Flushing unmatched stop prefixes | A stop formed from the entire 12-token baseline plus an unmatched sentinel retains the baseline text through normal `length` termination in both JSON and SSE. |
| EOS suppression | The explicit Qwen-formatted probe asking for Paris, with an empty completed thinking block, returns `Paris.` and `stop`: 19 prompt tokens, 3 completion tokens, 22 total. No decoded EOS marker appears. This does not validate the still-open chat-route formatting work. |
| Shared-weight worker state | Two schedulers receive France and Italy prompts concurrently. Their outputs match isolated execution. The existing shared backend mutex still serializes forwards; this is not evidence of parallel GPU kernels or improved throughput. |
| Cooperative cancellation | Dropping the receiver after its first generated token releases active KV/execution reservations and all logical blocks. This verifies eventual cleanup, not a fixed cancellation-latency guarantee. |
| Budget rejection, prefix eviction, and recovery | A valid logical request for 4,000 output tokens is rejected by the tensor reservation budget, before its forward pass. JSON reports an explicit error containing `Tensor memory budget exhausted`. Prefix reservations are evicted; a subsequent ordinary GPU request reproduces the baseline successfully. |

Baseline France output:

```text
 Paris. The capital of Italy is Rome. The capital of
```

Isolated and concurrent Italy output:

```text
 Rome. The capital of France is Paris. The capital of
```

After cleanup, the reported active KV, execution, and prefix reservations are
all zero; only 33,554,432 bytes of fixed headroom remain. Peak charged memory
was 188,183,808 bytes, below the 536,870,912-byte budget. One tensor reservation
was rejected. Both scheduler tasks shut down cleanly.

These byte values are charged reservations, not measured device allocation or
process RSS. This check does not force device OOM, establish allocation-failure
recovery, compare logits numerically, validate CUDA, benchmark throughput, or
evaluate broad pretrained output quality. UTF-8 and overlapping stop behavior
have CPU regressions; this GPU run uses the specific stops recorded above.
See [tensor memory accounting](tensor_memory.md) for the estimate's exclusions.

## Companion checks

After adding this gate, `cargo fmt --check`, default and Metal-feature
`cargo clippy --locked --all-targets -- -D warnings`, and default and Metal-feature
`cargo test --locked` passed. Both normal suites execute 46 tests; the default
suite explicitly ignores five model/manual checks, and the Metal suite ignores
six because it also discovers this GPU gate. The separately invoked command
above actually executes the GPU gate and reports one pass.

## Allocation and execution-failure policy verification (2026-10-16)

The gate now wraps the real Candle backend with a test-only fault injector. With
injection disabled it repeats all comparisons above, including successful GPU
inference after a tensor reservation rejection. It then returns a synthetic
`BackendError` after the second successful real-model forward of a new streaming
request (128-token limit, prompt `The capital of France is`, greedy temperature
0, top-p 1). The stream delivers ` Paris` before an explicit error and closes
without `[DONE]`. Both schedulers sharing the backend become unavailable;
`/ready` and subsequent JSON admission return 503. Active KV, execution, and
prefix reservations and logical blocks return to zero. Both worker tasks exit.

This verifies cleanup and fail-closed routing after a *returned* execution error
with actual pretrained GPU tensors. It does not deliberately exhaust unified
memory or simulate a Metal command-buffer/driver failure. The backend remains
quarantined until process restart; no inference retry or device recovery is
attempted. See [failure policy](device_failures.md) for the error contract.

The same artifact SHA-256 identities, Apple M4 hardware, actual
`Metal(MetalDevice(DeviceId(1)))`, F16 weights, sampling, and 512 MiB budget above
apply. The first extended run passed one GPU test in 28.51 seconds (28.46 seconds
inside the test). A final rerun after the test-helper lint fix passed in 30.40
seconds (30.31 seconds inside the test), with the same observations and memory
values. Peak reservations were 188,183,808 bytes; cleanup retained only
33,554,432 bytes of fixed headroom. Exact commands:

```bash
shasum -a 256 models/Qwen3-0.6B/config.json models/Qwen3-0.6B/tokenizer.json models/Qwen3-0.6B/model.safetensors
system_profiler SPDisplaysDataType
cargo test --locked --features metal --test test_gpu_inference -- --ignored --nocapture
```

Host GPU access was used. No CPU fallback counts toward this result. This is a
working-tree validation; no validating commit or hosted CI run has been created.
