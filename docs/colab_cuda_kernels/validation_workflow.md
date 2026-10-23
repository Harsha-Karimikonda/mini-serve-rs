# Reproducible GPU validation harness

The first CUDA delivery step adds `scripts/gpu_validation.py` and replaces the
embedded notebook source/archive with calls to that script. Python 3.9+ and the
standard library are sufficient for packaging. Validation requires Cargo/Rust,
`rustfmt`, `clippy`, local pretrained weights, and the selected GPU toolchain.

From the local checkout:

```bash
python3 -m unittest discover -s scripts/tests -v
python3 scripts/gpu_validation.py pack --output /tmp/rlse-source.tar.gz
```

The archive contains selected source from `src`, `tests`, `examples`, `scripts`,
`docs`, `.github`, `.githooks`, and root build/documentation files. Selection
includes tracked changes and nonignored untracked source; deleted files are
omitted. The manifest records base commit, dirty state, file hashes, sizes, and
executable bits. The JSON sidecar records the archive SHA-256. Packing the same
source twice produces identical archive bytes. Symlinks are rejected.

Only the listed source paths/extensions are transported. `.git`, model weights,
build outputs, `.env*`, and paths named for credentials/secrets/private keys are
excluded. This is a source allowlist, not a content secret scanner: review the
manifest and selected source before sharing. Do not place generated evidence
under source directories; use `/tmp` or an `evidence` directory, which is excluded.

Open `test_colab.ipynb` on a GPU runtime and upload the archive, its JSON sidecar,
and the matching local `scripts/gpu_validation.py`. The notebook compares the
bootstrap script hash with the manifest, verifies archive integrity, and extracts
to a fresh directory. It never merges old source or kills unrelated processes.
An interrupted extraction must be retried with a fresh destination.

Equivalent remote commands, using paths appropriate to the runtime:

```bash
python3 gpu_validation.py unpack --archive rlse-source.tar.gz \
  --sha256 <archive_sha256_from_local_sidecar> --destination /content/rlse-run
cd /content/rlse-run
python3 scripts/gpu_validation.py download-model
python3 scripts/gpu_validation.py run --device cuda --output /content/rlse-evidence
```

For local Metal verification:

```bash
python3 scripts/gpu_validation.py run --device metal --output /tmp/rlse-metal-evidence
```

The model is `Qwen/Qwen3-0.6B`, revision
`c1899de289a04d12100db370d81485cdf75e47ca`. Hashes come from the existing independent
reference fixture at `tests/fixtures/generation/qwen3-reference.json`; config,
tokenizer, tokenizer config, and safetensors must all match. Downloads use the
pinned revision, verify temporary files before renaming them, and reject existing
mismatched artifacts. Other model manifests are deferred to the next CUDA step.

Each run saves a source manifest, lockfile hash, model identity, exact command
arguments, exit codes, durations, numbered logs, and `results.json`. CUDA discovery
records GPU name/UUID, compute capability, total/free VRAM, driver, `nvcc` toolkit
version, and the driver-supported CUDA version separately. Metal runs also record
host/device discovery. Runtime device identity comes from the pretrained tests,
which explicitly assert the selected GPU and print `ACTUAL_DEVICE`.

The harness runs formatting, CPU tests, Python regressions, selected-feature
Clippy/tests, then `test_gpu_inference` and `test_pretrained_quality` separately
with `--ignored --nocapture`. Both must report one executed passing test and the
matching actual GPU marker. Zero tests, CPU fallback, absent prerequisites, and
command failures cannot yield `passed`. Missing dependencies produce `blocked`;
a failing check produces `failed`. Both return a nonzero exit code and export
logs. No GPU is initialized when prerequisite checks block the run.

Build caches live under `target/gpu-validation/<identity>`; identity includes
source, lockfile, compiler/Cargo, CUDA toolkit, stable GPU identity, device, and
selected build environment variables. Reuse requires rerunning in the same tree.
Inherited `RLSE_*` configuration is cleared before setting explicit device/model
paths. The runner does not dump environment variables or credentials. Artifacts
and source should remain unchanged while a run executes.

Sampling is greedy (`temperature=0`, `top_p=1`). Integration token limits are
12 for completion/chat/prefix/worker probes, 32 for EOS, 512 for cancellation,
4000 for reservation rejection, and 128 for injected returned-error checks.
Independent reference cases allow 32 tokens. Raw logs preserve observed output,
usage, cleanup, and counters. The evidence directory is exported as `.tar.gz`
with a SHA-256/status sidecar, including on blocked or failed runs.

This step does not integrate optimized kernels, implement physical paging,
benchmark performance, validate additional models, or prove driver OOM/hang
recovery. CUDA acceptance and actual pretrained inference remain pending until
an equipped host executes the harness successfully. See
[the recorded local checks](../gpu_validation.md#validation-harness-implementation-2026-10-21).

## A100 follow-up: numerical comparison (2026-10-23)

The user-supplied Colab console output reports a successful baseline run on
`NVIDIA A100-SXM4-40GB` (`sm_80`, driver `580.82.07`), with
`ACTUAL_DEVICE=Cuda(CudaDevice(DeviceId(1)))`, pinned Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`, artifact verification,
kernel status `passed`, server validation `true`, and overall status `passed`.
This is console evidence; the complete baseline bundle has not been inspected.
It excludes the opt-in pretrained numerical gate and makes no throughput claim.

Added a final, separate notebook cell to run the existing numerical gate after a
successful baseline, without changing inference code or acceptance tolerances:

```bash
RLSE_GPU_DEVICE=cuda RLSE_GPU_MODEL_DIR=/content/rlse-models/Qwen3-0.6B \
CARGO_TARGET_DIR=<baseline-results.json-target_directory> \
cargo test --locked --features cuda --test test_pretrained_numerics -- --ignored --nocapture
```

The command runs from the previously unpacked source root. The cell retains the
baseline evidence (source manifest, model hashes, CPU checks, hardware discovery),
streams and saves `numerics.log`, and records command, working directory, relevant
environment, exit code, duration, and actual-device markers in a separate
`results.json`. Its downloadable `/content/rlse-numerics-<uuid>.tar.gz` is produced
before a test failure is raised. It requires one executed passing test and an
actual CUDA marker. The baseline result is not overwritten.

Comparison settings: CPU/F32 reference versus GPU/F16; greedy argmax decisions;
three teacher-forced decode steps; prefill, restored prefixes, and single/batched
decode. Existing bounds remain cross-device max absolute error 0.25 / RMS 0.04,
and same-device execution-path max absolute error 0.125 / RMS 0.03, with finite
logits and identical greedy decisions required.

Local validation commands:

```bash
python3 -m unittest discover -s scripts/tests -v
cargo test --locked --test test_pretrained_numerics
# Additionally compile() every notebook code cell and run git diff --check.
```

Python regressions: 8 passed. CPU numerical regressions: 5 passed (0.05 s,
following a 50.78 s build). All notebook code cells compile; `git diff --check`
passed. **CUDA numerical
verification and remote execution of the new cell remain pending**: this local
host has no NVIDIA CUDA device, and pinned local model weights are absent.
No Metal run or synthetic test is counted as CUDA evidence. Keep the existing
Colab runtime connected and run only the new final cell; its CPU reference work
may take minutes. Save the diagnostic bundle for diagnosis before kernel
integration. For future full runs, enable `--include-numerics` in the main
pipeline instead.

## Repeatable `make colab-sync` (2026-10-23)

The convenience target previously used the fixed `/tmp/rlse-source.tar.gz` path,
so a second invocation failed with `FileExistsError` because the packer opens new
archives exclusively. The target now creates a fresh directory with
`mktemp -d /tmp/rlse-colab-sync.XXXXXX` and embeds its archive into the notebook.
The printed output path identifies the archive and adjacent JSON sidecar. Prior
archives are preserved, and the packer's overwrite protection is unchanged.

Validation: two consecutive `make colab-sync` invocations succeeded using distinct
directories. Python harness regressions (`python3 -m unittest discover -s
scripts/tests -v`) passed all 8 tests; `git diff --check` passed. This is a source
transport change with no inference or sampling changes. The earlier user-supplied
A100 baseline console evidence is recorded above; GPU execution of this updated
source bundle and the numerical follow-up remain **pending** on the connected
Colab A100 with the pinned Qwen3-0.6B artifacts. This local host has no CUDA GPU.
