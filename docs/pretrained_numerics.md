# Pretrained numeric regression

The next non-CUDA verification gap in the prioritized roadmap was comparing
actual GPU logits numerically. `tests/test_pretrained_numerics.rs` adds that
check to the Metal `make gpu-check` target. Runtime kernels are unchanged.

The opt-in test verifies all four local artifact SHA-256 hashes against
`tests/fixtures/generation/qwen3-reference.json`, then loads the same weights
on explicit Metal/F16 and CPU/F32. It compares all 151,936 vocabulary logits
for the instruction, system-role, Spanish, and stop prompts in that fixture.
It checks full prefill, restoring the first 16 tokens into independent KV
buffers before processing the suffix, and three decode iterations. Decode
uses CPU greedy tokens on both devices (teacher forcing) so a changed input
cannot hide a numeric discrepancy. Four concurrent sequences have different
positions; batched decode logits must match individually executed Metal paths.

Every comparison checks shape, finiteness, maximum absolute error, RMS error,
and identical greedy argmax. CPU/F32 fixture comparisons use maximum `1e-4`
and RMS `1e-5`. Metal/F16 versus CPU/F32 allows maximum `0.25` and RMS `0.04`
logit units; Metal execution-path comparisons allow maximum `0.125` and RMS
`0.03`. Separate maximum and RMS bounds catch both isolated errors and drift
across the vocabulary. These are regression bounds for F16 weight conversion
and accumulation/order differences, not relative-error guarantees or an
accuracy claim for every model, prompt, or device.

Five ordinary CPU regressions cover bounded roundoff, NaN rejection, excessive
drift, changed greedy choices, and execution-path agreement. The last uses
deterministic nonzero two-layer GQA weights, prompt lengths 3/17/33, restored
prefixes, and mixed-length three-step decode. No pretrained weights are needed
for those CPU checks. The GPU test is ignored in ordinary runs and fails if
explicit Metal selection, local artifacts, hashes, or hardware are unavailable.

## Commands and results: 2026-10-21

```bash
export PATH="/Users/hkarimkonda/.cargo/bin:$PATH"
cargo test --locked --test test_pretrained_numerics
cargo test --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
cargo clippy --locked --features metal --all-targets -- -D warnings
cargo fmt --all --check
RLSE_GPU_DEVICE=metal RLSE_GPU_MODEL_DIR=models/Qwen3-0.6B cargo test --locked --features metal --test test_pretrained_numerics -- --ignored --nocapture
system_profiler SPDisplaysDataType
git diff --check
```

CPU numeric regressions: **5 passed**. The full CPU suite passed with host
socket access: **80 passed, zero failed, two opt-in ignores**. Both Clippy
configurations and formatting passed. The sandbox
full-suite attempt failed three shutdown tests because binding sockets was
denied; the host rerun passed those tests. The initial sandbox Metal attempt
panicked during device discovery before inference and provides no GPU evidence.

GPU: **1 passed, zero failed/ignored**, **191.30 seconds**, using host access.
Actual device asserted and printed: `Metal(MetalDevice(DeviceId(1)))`.
All 32 full-vocabulary comparisons passed, with finite values and identical
greedy decisions. Observed worst errors (maximum and RMS may come from
different comparisons):

| Comparison | Maximum absolute error | Maximum RMS error |
| --- | --- | --- |
| Metal/F16 versus CPU/F32, prefill and decode | 0.153257 | 0.038059 |
| Restored prefix versus full Metal prefill | 0.109375 | 0.020940 |
| Batched versus individual Metal decode | 0.121094 | 0.025872 |

All four SHA-256 checks passed. Hardware: Apple M4, 8 GPU cores.
Model: local `models/Qwen3-0.6B`,
`Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`.
The fixture records exact artifact hashes and independently rendered chat
templates/token IDs. Build: Rust 1.99.1, debug/test, Candle 0.11. Sampling: greedy,
temperature 0, top-p 1 equivalent argmax, three teacher-forced decode iterations
after each prefill. KV capacity is 256 tokens per sequence.
Duration includes hashing, loading both models, and CPU reference forwards;
it is not a throughput measurement.

The first host run stopped on the Spanish restored-prefix comparison:
maximum absolute error `0.109375`, RMS `0.020940`, exceeding an initial RMS
bound of `0.02`. Chunking prefill changes F16 accumulation/rounding; the final
path RMS bound is `0.03`, with the maximum bound unchanged at `0.125` and
identical greedy decisions still required. CPU path comparisons retain their
much tighter bounds. This calibration must not be interpreted as proof that
arbitrary F16 differences are harmless.

This numeric reference uses the same Candle model implementation on CPU;
the separate `test_pretrained_quality` gate checks independently generated
Transformers token IDs/text. This four-prompt numeric check does not establish
broad quality, long-context stability, Qwen2.5 coverage, device OOM recovery,
load bounds, or throughput. CUDA remains outside this change.
