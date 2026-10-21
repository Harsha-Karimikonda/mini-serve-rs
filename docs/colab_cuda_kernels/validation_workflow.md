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
