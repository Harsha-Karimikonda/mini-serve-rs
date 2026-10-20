# Actual GPU inference verification

Completed changes must include an actual GPU inference run, in addition to CPU
regressions and feature builds. This project rule is recorded in [AGENTS.md](../AGENTS.md).
If GPU access or weights are unavailable, GPU verification stays pending.

## Current change awaiting verification (2026-10-20)

Chat formatting and model identity have changed since the recorded Metal runs.
Their GPU gate is **pending**; historical passes do not validate this change.
The gate now supports explicit Metal/CUDA selection and chat comparisons;
see [current commands and blockers](chat_correctness.md#gpu-gate-and-colabcuda).
Local runs also need `tokenizer_config.json` alongside the original artifacts.

## Repeatable check

Prerequisites: macOS with a Metal GPU and local `config.json`, `tokenizer.json`,
and `model.safetensors` in `models/Qwen3-0.6B`.

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
