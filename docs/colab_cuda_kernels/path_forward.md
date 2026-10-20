# CUDA and Accelerator Path Forward

## Summary

Build a repeatable GPU validation workflow, establish full-application CUDA correctness, integrate optimized kernels behind a controlled execution mode, then enable the optimized path on hardware and model combinations that pass correctness and performance gates. Investigate TPU support as a separate feasibility effort.

The current CUDA kernels are F32 prototypes and are not connected to model execution. CUDA inference uses F16, and the cache's block IDs currently support logical accounting only; they do not identify physical GPU KV pages. Existing notebook results for kernel compilation and cache allocation are useful setup evidence, but do not establish end-to-end pretrained inference.

| Hardware | Required models | Purpose |
| --- | --- | --- |
| NVIDIA T4 | Qwen3-0.6B, Qwen2.5-0.5B-Instruct | Compatibility and constrained-memory testing |
| NVIDIA L4 | Same small models | Serving correctness and performance |
| NVIDIA A100 | Small models and Qwen3-8B | Sharded weights, larger-model correctness, and load |
| RTX PRO 6000 Blackwell | Small models and Qwen3-8B | Blackwell compatibility and performance |
| TPU v5/v6 | Feasibility probe | Separate backend investigation |

The initial serving scope is single-GPU inference. Quantization and distributed inference are outside this plan. TPU production support depends on the feasibility results.

## 1. Repeatable notebook validation

- Make `test_colab.ipynb` a thin interface over repository scripts. Remove embedded source copies and ad hoc generated Rust tests.
- Package the exact current source tree, including intended uncommitted and untracked source files, with a manifest and archive hash. Exclude weights, credentials, build outputs, and `.git`; record the base commit and dirty state.
- Pin each model revision and verify artifact hashes. Keep the existing Qwen3-0.6B identity and create equivalent manifests and independent reference fixtures for Qwen2.5-0.5B-Instruct and Qwen3-8B.
- Record GPU name and UUID, compute capability, available VRAM, driver, toolkit, Rust version, lockfile identity, and actual execution device. Record separately the installed toolkit and the CUDA version reported by `nvidia-smi`.
- Start with CUDA 12.8 as the common toolkit candidate and verify compiler targets before building. Blackwell native compilation needs Blackwell-capable tooling. See [NVIDIA's Blackwell compatibility guide](https://docs.nvidia.com/cuda/archive/12.8.1/blackwell-compatibility-guide/index.html).
- Keep build caches separate by source, toolchain, and architecture. Clean up only processes launched by the harness.
- Export a downloadable evidence bundle containing manifests, commands, logs, structured results, numerical comparisons, and benchmark samples. Missing prerequisites are recorded as **blocked**, never **passed**.

## 2. Full-application CUDA correctness

Extend the existing tests and server validation. The baseline CPU and CUDA checks are:

```bash
cargo fmt --all --check
cargo test --locked --all-targets
cargo clippy --locked --features cuda --all-targets -- -D warnings
cargo test --locked --features cuda --all-targets

RLSE_GPU_DEVICE=cuda \
RLSE_GPU_MODEL_DIR=models/Qwen3-0.6B \
cargo test --locked --features cuda --test test_gpu_inference -- --ignored --nocapture

RLSE_GPU_DEVICE=cuda \
RLSE_GPU_MODEL_DIR=models/Qwen3-0.6B \
cargo test --locked --features cuda --test test_pretrained_quality -- --ignored --nocapture
```

Extend the model manifest and test utilities to support Qwen2.5-0.5B-Instruct and sharded Qwen3-8B weights. Resolve and pin independent reference fixtures before adding these models to acceptance runs.

Use greedy sampling (`temperature=0`, `top_p=1`) for comparisons. Cover completion and chat through JSON and SSE; template and tokenizer behavior; usage, EOS, stops, Unicode, and terminal delivery; ragged continuous batches; isolated versus concurrent requests; worker isolation and prefix reuse; authentication and client limits; slow consumers, cancellation, shutdown, readiness, and telemetry; reservation rejection and recovery; returned execution errors and cleanup; sharded loading and corrupt or missing shard rejection; and context boundaries.

Use existing curated token fixtures for protocol and output checks. Add teacher-forced logit comparisons against independent reference execution. Diagnose numerical differences instead of generating expected results from this implementation.

## 3. CUDA kernel and physical KV integration

Add `--cuda-execution=auto|reference|optimized` and the equivalent application configuration. Initially, `auto` selects the reference implementation. An explicit request for an unsupported optimized configuration returns an error; backend failures do not silently retry through another path.

Integrate fused RMSNorm using Candle's CUDA custom-operation interface and managed CUDA stream. Support F16 inputs and weights, F32 accumulation, arbitrary row counts, and model-derived dimensions, including Q/K normalization. Preserve reference casting behavior, return failures through Rust, and remove process exits and unconditional device-wide synchronization. Keep F32 cases for kernel validation.

Create a backend-owned physical page pool shared by workers attached to that backend. Use 16-token pages with per-layer K/V storage shaped `[pages, kv_heads, 16, head_dim]`. Keep scheduler admission accounting separate and named as logical accounting; never use its block IDs as GPU addresses. Reserve each request's maximum page requirement before execution, write prefill KV into its pages, and read page tables directly during decode. Support ragged batches, model-derived head counts, and dimensions required by all selected models.

Implement tiled attention with online softmax; avoid shared-memory logits that grow with sequence length. Preserve prefix caching initially through compact snapshots copied into newly owned pages. Shared-page prefix ownership and copy-on-write are deferred. Release physical pages only after dependent stream work completes. Cover cancellation, eviction, execution errors, and shutdown for leaks and premature reuse.

Keep the completion and chat APIs unchanged. Add execution-path identity and physical-pool allocation and occupancy fields to status and metrics.

## 4. Acceptance, benchmarking, and rollout

Kernel tests cover multiple rows, fragmented page tables, page reuse, mixed lengths, and sequence lengths 1, 15, 16, 17, 31, 32, and 33. Compare with reference math using initial tolerances of F32 `atol=1e-5, rtol=1e-4` and F16 `atol=5e-3, rtol=5e-3`; reject nonfinite outputs. Document numerical justification for any tolerance change. Run CUDA memory and race diagnostics on kernel tests.

Run actual pretrained inference through both reference and optimized paths for every required hardware/model pairing. Retain CPU regressions and actual pretrained Metal coverage for shared backend and cache changes. A feature build, kernel microtest, or CPU fixture alone does not satisfy the GPU inference gate.

Benchmark release builds with 128-, 512-, and 2048-token prompts, 128 output tokens, concurrency 1, 4, and 8, and cold and warm prefixes. Use three warmups and ten measured rounds. Record actual token counts, TTFT, inter-token latency, throughput, errors, and sampled VRAM separately from reservation estimates. Require no correctness failures, no repeatable regression above 5% in throughput or p95 TTFT for supported workloads, and a measurable improvement in at least one targeted workload per GPU.

Enable `auto` optimized execution only for hardware/model configurations that pass those gates. Keep reference selection available for diagnosis and rollback. Unvalidated configurations continue on reference execution. Update this guide and [GPU validation](../gpu_validation.md) with exact commands, model and artifact identities, actual device, sampling settings, results, and limitations.

## 5. TPU feasibility

First identify the available TPU variants, slice topology, notebook runtime, and software stack. Google documents JAX and PyTorch/XLA workflows for v6e; see [Google Cloud TPU v6e documentation](https://cloud.google.com/tpu/docs/v6e-training).

Produce a separate feasibility report that runs a pinned small pretrained model on an explicitly selected TPU and records compilation time, numerical behavior, memory, and request latency. Evaluate an out-of-process TPU worker behind the Rust backend interface, including cancellation and failure propagation. The result is a follow-on backend proposal, with no production TPU device option or support claim until that work is approved and implemented.
