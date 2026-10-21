# Actual GPU inference verification

Completed changes must include an actual GPU inference run, in addition to CPU
regressions and feature builds. This project rule is recorded in [AGENTS.md](../AGENTS.md).
If GPU access or weights are unavailable, GPU verification stays pending.

## Commit recheck: 2026-10-21

Commands used to recheck the staged change set (run before staging):

```bash
python3 -m unittest discover -s scripts/tests -v
/Users/hkarimkonda/.cargo/bin/cargo test --locked --all-targets
/Users/hkarimkonda/.cargo/bin/cargo fmt --all --check
/Users/hkarimkonda/.cargo/bin/cargo clippy --locked --all-targets -- -D warnings
/Users/hkarimkonda/.cargo/bin/cargo clippy --locked --features metal --all-targets -- -D warnings
/usr/bin/env PATH=/Users/hkarimkonda/.cargo/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin RLSE_GPU_DEVICE=metal RLSE_GPU_MODEL_DIR=models/Qwen3-0.6B /Users/hkarimkonda/.cargo/bin/cargo test --locked --features metal --test test_gpu_inference --test test_pretrained_quality --test test_pretrained_numerics -- --ignored --nocapture
python3 scripts/validate_cuda_kernels.py --output /tmp/rlse-precommit-cuda-kernels.json
python3 scripts/gpu_validation.py pack --output /tmp/rlse-precommit-source.tar.gz
system_profiler SPDisplaysDataType
git diff --check
```

Python: **7 passed**. CPU: **80 passed, two ignored** with host localhost
access; the sandbox attempt failed three shutdown tests because socket binding
was denied. Formatting and both Clippy configurations passed. All four notebook
code cells compile as Python and contain no stored outputs. Source packaging
succeeded with 100 selected files before this entry was added.

The sandbox Metal attempt failed during device discovery before inference;
only the host-access rerun can provide GPU evidence. CUDA kernel validation
exited 1 because `nvcc` is unavailable on this Apple M4 host. CUDA compilation,
custom-kernel correctness/performance, and pretrained CUDA inference remain
**pending**; Metal evidence does not verify those paths.

Host GPU rerun: **all three pretrained gates passed**, integration in 50.94 s,
numeric in 143.48 s, and independent generation reference in 5.52 s. Actual
device: `Metal(MetalDevice(DeviceId(1)))`, Apple M4, 8 GPU cores. Local model:
`models/Qwen3-0.6B`, `Qwen/Qwen3-0.6B`, revision
`c1899de289a04d12100db370d81485cdf75e47ca`; all four artifact hashes matched
`tests/fixtures/generation/qwen3-reference.json`. Sampling: greedy, temperature
0, top-p 1; integration/reference token limits are recorded below. Numeric
comparison uses GPU/F16 and CPU/F32 with three teacher-forced decode steps.
All 32 logit comparisons passed with finite values and identical greedy argmax;
worst CPU/GPU maximum absolute error was 0.153257, RMS 0.038059. Restored-prefix
maximum/RMS were 0.109375/0.020940; batched/single were 0.121094/0.025872.
All four independent token/text references matched. Integration counters were
31 accepted, 27 completed, 2 injected errors, 2 cancellations, and 279 generated
tokens, with active/prefix reservations returned to zero. These focused checks
retain the model, context, and performance limitations documented below.

## Complete validation harness & advanced Colab workflow run: 2026-10-21

Extended `scripts/gpu_validation.py`, `docs/colab_cuda_kernels/mini_serve_kernels.cu`,
`scripts/validate_cuda_kernels.py`, `tests/test_pretrained_numerics.rs`, and
`docs/colab_cuda_kernels/test_colab.ipynb`:

- `scripts/gpu_validation.py` automatically prepends `~/.cargo/bin` and
  `/usr/local/cuda/bin` when present, supports `--cache-dir` for persistent Cargo
  build caching across unpacked Colab trees, runs `test_pretrained_numerics` in
  addition to `test_gpu_inference` and `test_pretrained_quality`, supports
  `--include-kernels` (`scripts/validate_cuda_kernels.py`) and `--include-server`
  (`scripts/validate_gpu_server.py`), and supports `--embed-notebook` /
  `clear-notebook` (`make colab-sync` / `make colab-clear`) for zero-popup VS Code
  local-to-Colab execution while preserving deterministic clean source archives.
- `docs/colab_cuda_kernels/mini_serve_kernels.cu` adds F16 fused RMSNorm with F32
  variance accumulation (`fused_rmsnorm_f16_kernel`) and ragged batched tiled
  online-softmax Paged GQA Decode Attention (`paged_gqa_decode_tiled_kernel` for
  F32 and F16 with constant 512-byte shared memory independent of sequence length)
  with non-exiting status-returning `extern "C"` launchers.
- `tests/test_pretrained_numerics.rs` now supports both `RLSE_GPU_DEVICE=metal`
  and `RLSE_GPU_DEVICE=cuda`.

Exact commands run locally:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo fmt --all
python3 -m unittest discover -s scripts/tests -v
python3 scripts/gpu_validation.py run --device metal --include-server --output /tmp/rlse-metal-validation-live
```

Observed results on Apple M4 (`macOS-15.8.1-arm64`, 16 GB unified memory, 8 GPU
cores, `rustc 1.99.1`, `cargo 1.99.1`, `Qwen/Qwen3-0.6B` at revision
`c1899de289a04d12100db370d81485cdf75e47ca` with all 4 SHA-256 hashes verified):

- Harness status: **`passed`** (`evidence: /private/tmp/rlse-metal-validation-live.tar.gz`,
  SHA-256 `9561c46d4c059bb3cc72fa4075f1f60278483c9952a6906a145c3cf802d032d4`).
- Actual device asserted and recorded across all GPU gates and server startup:
  `Metal(MetalDevice(DeviceId(1)))`.
- CPU & Python suites: `cargo fmt --all --check` passed (`0.118 s`), CPU
  `cargo test --locked --all-targets` passed all **80 CPU unit/integration tests**
  including the 5 new numeric comparison/batched-logit CPU tests (`83.856 s`),
  and Python unit tests passed **7 tests** (`1.971 s`, including
  `test_embed_and_clear_notebook_preserves_clean_archive`).
- Metal feature Clippy & test build: passed (`38.499 s` and `42.053 s`).
- Explicit GPU gate 1 (`test_gpu_inference`): **1 passed in 49.791 s** (31 accepted,
  27 completed, 2 errors, 2 cancellations, 279 generated tokens; active/prefix
  reservations returned to 0; peak reserved 188,183,808 bytes).
- Explicit GPU gate 2 (`test_pretrained_quality`): **1 passed in 5.232 s** (all 4
  cases matched independent CPU/F32 reference token IDs and text under greedy
  sampling `temperature=0, top_p=1`).
- Explicit GPU gate 3 (`test_pretrained_numerics`): **1 passed in 143.140 s**
  (prefill, 16-token restored prefix, and 3 teacher-forced single/batched decode
  steps on `Metal/F16` matched `CPU/F32` within `max_abs <= 0.153257`,
  `rms <= 0.038059`, and batched vs single `Metal/F16` within `max_abs <= 0.121094`,
  `rms <= 0.025872`, with identical greedy argmax decisions).
- Live HTTP GPU server validation (`scripts/validate_gpu_server.py --device metal`):
  **passed in 34.454 s** (3 warm prefix rounds saved 64 tokens/round, 3 cold rounds
  saved 0 tokens/round, 14 accepted/completed, 168 generated tokens, 0 errors,
  0 leaked KV blocks, and Ctrl-C after first SSE token drained the 12-token
  request cleanly in `1.623 s`).

Remaining limitations: CUDA execution of `gpu_validation.py run --device cuda
--include-kernels --include-server` and `scripts/validate_cuda_kernels.py`
requires running [`docs/colab_cuda_kernels/test_colab.ipynb`](colab_cuda_kernels/test_colab.ipynb)
on an active Colab NVIDIA GPU runtime.

## Pinned weights and pretrained Metal recheck: 2026-10-21

Downloaded and hash-verified the pinned Qwen3-0.6B artifacts with the new
harness's `download-model` command. Both pretrained gates then **passed** with
explicit Metal selection and host GPU access. This supersedes the missing-weight
blocker for local pretrained Metal checks; CUDA and the complete harness `run`
workflow on an equipped host remain unverified.

Exact commands:

```bash
python3 scripts/gpu_validation.py download-model
shasum -a 256 models/Qwen3-0.6B/config.json models/Qwen3-0.6B/tokenizer.json models/Qwen3-0.6B/tokenizer_config.json models/Qwen3-0.6B/model.safetensors
export PATH="/Users/hkarimkonda/.cargo/bin:$PATH"
RLSE_GPU_DEVICE=metal RLSE_GPU_MODEL_DIR=models/Qwen3-0.6B cargo test --locked --features metal --test test_gpu_inference --test test_pretrained_quality -- --ignored --nocapture
system_profiler SPDisplaysDataType
git diff --check
```

The first sandboxed inference attempt panicked during device discovery before
establishing a GPU. The successful host-access rerun used this exact invocation:

```bash
/usr/bin/env PATH=/Users/hkarimkonda/.cargo/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin RLSE_GPU_DEVICE=metal RLSE_GPU_MODEL_DIR=models/Qwen3-0.6B /Users/hkarimkonda/.cargo/bin/cargo test --locked --features metal --test test_gpu_inference --test test_pretrained_quality -- --ignored --nocapture
```

Hardware: Apple M4, 8 GPU cores. Both gates asserted and printed
`ACTUAL_DEVICE=Metal(MetalDevice(DeviceId(1)))`; CPU fallback did not count.
Artifact: local `models/Qwen3-0.6B`, model `Qwen/Qwen3-0.6B`, revision
`c1899de289a04d12100db370d81485cdf75e47ca`. All four SHA-256 values matched the
artifact identity table below and the independent generation fixture. Weights
remain gitignored. Build: Rust 1.99.1, Candle 0.11, F16 weights, debug/test.

Sampling: greedy, temperature 0, top-p 1. Integration limits: 12 tokens for
completion/chat/prefix/worker probes, 32 EOS, 512 cancellation, 4000 reservation
rejection, and 128 injected returned error. Independent reference limit: 32.

Observed results:

- Integration: **1 passed**, zero failed/ignored, **47.61 s**. Cached/uncached
  output equivalence, exact/partial prefix accounting, authenticated JSON/SSE
  chat/completions, per-client quota isolation/recovery, stops/EOS, shared-worker
  outputs, cancellation, bounded worker draining, reservation rejection/recovery,
  and quarantine after an injected returned error following real GPU execution
  all passed. Basic completion: ` Paris. The capital of Italy is Rome. The capital of`,
  5 prompt tokens, 12 generated tokens, `finish_reason=length`.
- Execution counters: 31 accepted, 27 completed, 2 errors, 2 cancellations,
  279 generated tokens. Active and prefix reservations returned to zero; fixed
  headroom remained 33,554,432 bytes; peak reservation was 188,183,808 bytes.
- Independent reference: **1 passed**, zero failed/ignored, **4.79 s**. Exact
  reference tokens/text matched instruction (`Paris.`, 7 tokens), roles
  (`cobalt`, 7), Spanish (`Hola`, 6), and stop (`red green blue.`, 9).
  Counts include thinking delimiters/EOS; thinking delimiters remain visible.
  Applying the configured stop retains `red` after the delimiters.
- Relevant CPU evidence from the immediately preceding installation recheck:
  **75 Rust tests passed**, two opt-in ignores; **6 Python tests passed**;
  formatting and CPU Clippy passed. Runtime source did not change between runs.

Limits: this is focused pretrained Metal correctness evidence. It does not
validate CUDA, optimized custom kernels, physical paging, the complete evidence
runner, broader generation quality, numeric logit tolerances, production load,
throughput, or driver OOM/hang recovery. Earlier blocked runs remain historical.

## Rust installation recheck: 2026-10-21

Rust is now installed at `/Users/hkarimkonda/.cargo/bin`, which was absent from
this session's PATH. Active toolchain: `stable-aarch64-apple-darwin`;
`cargo 1.99.1 (5f94df478 2026-08-27)` and
`rustc 1.99.1 (58cf8f9c3 2026-10-13)`. Clippy and rustfmt are installed.
This supersedes the earlier missing-toolchain blocker for the harness change.

Exact commands:

```bash
export PATH="/Users/hkarimkonda/.cargo/bin:$PATH"
cargo --version
rustc --version
rustup show active-toolchain
rustup component list --installed
cargo fmt --all --check
cargo clippy --locked --all-targets -- -D warnings
python3 -m unittest discover -s scripts/tests -v
cargo test --locked --all-targets
RLSE_GPU_DEVICE=metal RLSE_GPU_MODEL_DIR=models/Qwen3-0.6B cargo test --locked --features metal --test test_gpu_inference --test test_pretrained_quality -- --ignored --nocapture
```

Formatting, CPU Clippy, and all **6 Python regressions passed**. The first CPU
suite attempt encountered sandbox-denied localhost bindings in three process
shutdown tests. Rerunning `/Users/hkarimkonda/.cargo/bin/cargo test --locked
--all-targets` with host access passed: **75 passed, zero failed, two opt-in tests
ignored**. These are CPU fixtures/default-feature checks, not GPU evidence.

The explicit Metal test built successfully, then failed its local-weight
prerequisite: `local pretrained weights are required` (zero passed, one failed,
exit 101). Cargo stopped after `test_gpu_inference`; the independent reference
test did not execute. No actual execution device was initialized, no model
artifact loaded, and no inference sampling ran. Intended model/revision and
sampling remain those recorded in the harness implementation entry below.
`models/Qwen3-0.6B` is absent; `nvcc` and `nvidia-smi` are also unavailable.
**Actual GPU inference verification remains pending**, including CUDA execution.

## Validation harness implementation: 2026-10-21

Added a Python standard-library harness for deterministic source transport,
manifest verification, pinned Qwen3 artifact downloads, explicit CUDA/Metal test
execution, and evidence export. The Colab notebook now calls repository scripts
instead of embedding source, a stale archive, and generated Rust tests. Python
regressions also run in the Linux CI job; hosted execution is not recorded.
See [usage and limits](colab_cuda_kernels/validation_workflow.md).

Commands run in this workspace:

```bash
python3 -m unittest discover -s scripts/tests -v
python3 scripts/gpu_validation.py run --device cuda --output /tmp/rlse-cuda-validation-20261021
python3 scripts/gpu_validation.py run --device metal --output /tmp/rlse-metal-validation-20261021
cargo test --locked --all-targets
RLSE_GPU_DEVICE=metal RLSE_GPU_MODEL_DIR=models/Qwen3-0.6B cargo test --locked --features metal --test test_gpu_inference --test test_pretrained_quality -- --ignored --nocapture
python3 scripts/gpu_validation.py pack --output /tmp/rlse-source-20261021.tar.gz
python3 scripts/gpu_validation.py unpack --archive /tmp/rlse-source-20261021.tar.gz --sha256 <sidecar_archive_sha256> --destination /tmp/rlse-source-20261021
python3 scripts/gpu_validation.py pack --output /tmp/rlse-source-repeat-20261021.tar.gz
git diff --check
```

Observed Python result: **6 passed** on Python 3.9.6. Regressions cover modified
and untracked source inclusion, deterministic archives, excluded credentials and
weights, source tampering, extra source, symlinks, archive traversal/hash rejection,
model hash mismatch, blocked evidence export, and rejection of zero-test/CPU-only
GPU claims. These tests use temporary synthetic artifacts; they do not perform
pretrained inference or count as GPU evidence. Notebook code cells compile as
Python and have no stored outputs. Actual-tree archive extraction/verification
and repeated archive SHA-256 comparison passed; see the local archive sidecars
for exact file hashes, base commit, and dirty state. Whitespace checks passed.

Both explicit-device harness runs returned **blocked** (exit 1), exported logs,
and recorded `actual_device: null`. This workspace has no `cargo`/`rustc`, `nvcc`,
`nvidia-smi`, or local `models/Qwen3-0.6B` directory. Direct CPU and Metal Cargo
commands returned exit 127 (`cargo: command not found`). No pretrained artifact
was loaded and no inference sampling ran. Intended model: `Qwen/Qwen3-0.6B`,
revision `c1899de289a04d12100db370d81485cdf75e47ca`; exact expected hashes remain
in the independent reference fixture. Intended sampling: temperature 0, top-p 1,
with the integration/reference token limits listed in the workflow. No actual
execution device was established.

**Rust CPU regressions and actual GPU inference verification remain pending for
this change.** Downloaded/notebook CUDA execution, additional model manifests,
numeric comparisons, and performance samples are also pending. The historical
Metal results below validate earlier changes, not this new harness. No scaling
or optimized-kernel acceptance claim is made.

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
