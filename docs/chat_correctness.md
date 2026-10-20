# Chat formatting and model identity: 2026-10-20

Implementation is present; **pretrained GPU verification is pending**. This
change must not be marked GPU-verified based on the historical October 16 run.

## Behavior and interfaces

`ModelBackend::format_chat` owns chat rendering. Its default rejects chat;
custom backend implementations must explicitly provide formatting to support
`/v1/chat/completions`. Candle loads `tokenizer_config.json` from the same
resolved local directory or HF repository as its other artifacts. MiniJinja
with Python-method compatibility renders the string `chat_template`, passing
messages, BOS/EOS tokens, and `add_generation_prompt=true`. The artifact's
default thinking behavior is preserved. Rendering has a 100,000 fuel limit.

Only plain-text `system`, `user`, and `assistant` messages are supported. Empty
message lists, other roles, missing/invalid templates, and runtime template
errors return structured HTTP 400 before tokenization, queueing, or tensor
reservation. No generic real-model chat fallback occurs. Missing or malformed
chat metadata leaves ordinary completion inference available. Named-template
collections, tool calls, multimodal content, and thinking-mode controls are
outside this milestone. Mock retains deterministic generic formatting.

Routing selects a healthy worker with the requested loaded model name, then
formats and submits through that same worker. Names are case-sensitive and
must exactly match `/v1/models`; a local-path model is advertised under that
configured path. Unknown models return HTTP 400. Known quarantined models
remain listed but return HTTP 503 when requested.

Model listing deduplicates actual backend identities. JSON/SSE responses,
health/status, and the dashboard agree with those identities. Explicit mock
fallback now advertises `mock`; requests naming the model that failed to load
are rejected. Health/status additionally expose `models`; their existing
`model` field contains the first sorted identity, or null for no workers.
The CLI currently loads one model shared across workers.

This intentionally tightens compatibility: clients that previously sent any
arbitrary model string must now use a loaded name. Completion prompts are
otherwise unchanged. The health `device` field still describes configuration;
the GPU gate independently asserts the actual backend device.

## Repeatable checks and artifact identity

Pinned Qwen2.5 and Qwen3 configs and independent Transformers-rendered prompts
and token IDs are in [the fixture directory](../tests/fixtures/chat/README.md).
Cases cover system instructions, single/multiple turns, Unicode, assistant
history, and historical thinking blocks. Default CPU coverage also checks
invalid roles/configs, unavailable chat with working completions, model-name
rejection before admission, loaded identity despite stale configuration,
JSON/SSE identity, and routing among multiple model identities.

The toolchain was absent from PATH and common install locations. An isolated
Rust toolchain was installed under `/private/tmp`, without changing shell
profiles. Exact environment for this session:

```bash
export CARGO_HOME=/private/tmp/rlse-cargo
export RUSTUP_HOME=/private/tmp/rlse-rustup
export PATH="$CARGO_HOME/bin:$PATH"
```

Reference generation used:

```bash
/private/tmp/rlse-reference/bin/python tests/fixtures/chat/generate_reference.py --artifacts /private/tmp/rlse-chat-artifacts
RLSE_CHAT_ARTIFACTS=/private/tmp/rlse-chat-artifacts cargo test --locked --test test_chat_templates -- --ignored --nocapture
cargo fmt --all --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --all-targets
cargo clippy --locked --features metal --all-targets -- -D warnings
cargo test --locked --features metal --all-targets
```

Observed results on 2026-10-20:

- `cargo test --locked --all-targets`: **56 passed**, zero failed, two ignored
  (pretrained Qwen3 inference and the opt-in tokenizer-artifact check).
- `cargo test --locked --features metal --all-targets`: **56 passed**, zero
  failed, three ignored (the same checks plus the explicit GPU gate). These
  normal suites use explicit CPU model fixtures; they do not run GPU inference.
- Explicit pinned-tokenizer check: **one passed**, zero failed. All six cases
  across Qwen2.5/Qwen3 matched Transformers rendered text and exact token IDs.
  These are tokenizer-only comparisons, with no sampling or actual inference
  device. The three prompt lengths are 30/43/54 tokens for Qwen2.5 and 9/43/24
  for Qwen3.
- Default and Metal-feature Clippy with warnings denied: passed.
- `cargo fmt --all --check` and `git diff --check`: passed.
- Explicit Metal gate command below: **zero passed, one failed** at the
  prerequisite assertion `local pretrained weights are required`. No model was
  loaded and no actual GPU device or sampling was exercised. `models/` and the
  usual local Hugging Face cache directories are absent. Only tokenizer
  artifacts were downloaded for the reference comparison.
- CUDA compilation/inference: pending; this macOS host has neither `nvcc` nor
  `nvidia-smi`, and no connected Colab execution path is established here.

The initial CPU run exposed a stale failure-test request naming `mock` while
its backend was named `fault`. Corrected that fixture to request `fault`; its
original quarantine/error assertions now pass under strict model validation.
The new chat-template regression is enabled in the ordinary suite.

CPU fixtures and reference tokenization do not execute pretrained inference.
No commit or hosted CI result was created. The implementation remains awaiting
its GPU completion gate.

## GPU gate and Colab/CUDA

The existing pretrained GPU regression now supports both `metal` and `cuda`.
It rejects a non-GPU selection, asserts the actual Candle device, and uses a
local model directory without downloading or silently falling back. New checks
compare the first Qwen3 reference prompt and token IDs, then compare chat JSON
and SSE greedy output with direct completion on that reference prompt. Existing
prefix, stop, cancellation, budget, and failure-quarantine coverage remains.

Prerequisites: local Qwen3-0.6B `config.json`, `tokenizer.json`,
`tokenizer_config.json`, and `model.safetensors`. Use the pinned Qwen3 revision
above so tokenizer/chat references agree. Sampling is greedy, `temperature=0`,
`top_p=1`, with 12 generated tokens for the new chat comparisons. The existing
EOS probe uses 32; other resource probes keep their existing token limits.

```bash
# Metal host:
RLSE_GPU_MODEL_DIR=models/Qwen3-0.6B make gpu-check
# Equivalent explicit command:
RLSE_GPU_DEVICE=metal cargo test --locked --features metal --test test_gpu_inference -- --ignored --nocapture

# Inside the connected Colab runtime, after checking out this working tree,
# installing Rust, and ensuring nvcc/CUDA build tools are available:
nvidia-smi
nvcc --version
# Download pinned artifacts in the runtime, not through a CPU-fallback test:
hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca --include config.json tokenizer.json tokenizer_config.json model.safetensors --local-dir models/Qwen3-0.6B
RLSE_GPU_MODEL_DIR=models/Qwen3-0.6B make cuda-check
# Equivalent:
RLSE_GPU_DEVICE=cuda cargo test --locked --features cuda --test test_gpu_inference -- --ignored --nocapture
```

A notebook reporting `torch.cuda.is_available()` or an A100 in `nvidia-smi`
does not execute this Rust regression. Record the actual remote hardware,
artifact hashes, full command, device assertion, sampling, and test results
before claiming CUDA verification. No remote Colab connection or CUDA execution
has been established by this change; the user's connectivity experiment and
notebook are untouched. CUDA compilation is also pending on a CUDA host.
