# CI Python setup repair (2026-10-23)

The Linux job in run [39672755911](https://github.com/Harsha-Karimikonda/mini-serve-rs/actions/runs/39672755911)
failed before tests started: `actions/setup-python` could not find Python 3.9
for the Ubuntu 26.04 x64 runner selected by `ubuntu-latest`.
The earlier run 39670906699 failed at the same step. Rust linting and the
macOS/Metal job passed in both runs.

Changed `.github/workflows/ci.yml` to request Python 3.12, which the
[Python versions manifest](https://github.com/actions/python-versions/blob/main/versions-manifest.json)
lists for Ubuntu 26.04 x64. No inference code changed.

Local validation:

- `python3 -m unittest discover -s scripts/tests -v`: all 8 tests passed
  using Python 3.9.6. These tests mock GPU execution; their simulated
  PASSED/FAILED messages are not real inference evidence.
- `git diff --check`: passed.
- `command -v cargo` and `command -v rustc`: neither tool is installed/on PATH,
  so local Rust CPU regressions and GPU inference could not run.

Real GPU verification remains **pending**. No pretrained model/artifact was
loaded, no GPU device was selected, and no sampling settings were exercised.
A successful macOS/Metal CI build or test suite does not establish real GPU
inference with local pretrained weights. The GitHub Actions run for the repair
must separately confirm Python 3.12 setup, the Python harness regressions,
Rust CPU tests, and lint/Metal checks.
