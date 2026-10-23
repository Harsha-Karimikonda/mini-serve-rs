# A100 baseline → optimized inference → deployable GPU service

Updated: 2026-10-23. This document records the agreed plan; milestones below are
planned work unless explicitly identified as reported evidence.

## Direction and current evidence

Pursue performance and deployment readiness together, using Colab A100 sessions
for validation and targeting a portable single-GPU container.

The user-supplied run reports successful Qwen3-0.6B CUDA inference,
reference-output checks, standalone kernel validation, and HTTP/server checks.
Preserve this as reported evidence; inspect the complete evidence bundle before
tying those results to a specific source snapshot. The reported hardware is an
NVIDIA A100-SXM4-40GB (sm_80), with actual execution device
`Cuda(CudaDevice(DeviceId(1)))`, pinned Qwen/Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`, and greedy sampling
(`temperature=0`, `top_p=1`). The reported evidence archive is
`/content/rlse-evidence-b7e88e7a68c6488a962de535e6317110.tar.gz`.

The next milestone is **a reproducible release-build performance and reliability
baseline**. CUDA numerical verification remains pending, kernels remain separate
from model execution, and current server checks do not establish sustained-load
performance. See the [validation workflow](validation_workflow.md) and
[GPU validation record](../gpu_validation.md) for existing tooling and history.

## Next milestone: establish the baseline

### 1. Close the numerical and evidence gaps

- Inspect the existing bundle's source manifest, artifact hashes, command logs,
  and results.
- Run the existing pretrained numerical gate with explicit CUDA selection.
  Preserve its finite-logit, greedy-decision, and numerical tolerance requirements;
  investigate failures without loosening thresholds merely to pass.
- Update current verification summaries while retaining historical results and
  their limitations.

### 2. Make GPU sessions resumable

- Extend the existing validation harness with separate correctness,
  release-benchmark, and reliability stages.
- Save results after each stage and export evidence even on failure or
  interruption.
- Reuse artifacts only when source, model, toolchain, device, and build identities
  match. An interrupted stage remains incomplete.
- Keep the notebook a thin entrypoint. Budget up to two hours after setup, with a
  60-minute reliability stage; timeouts must not count as passes.

### 3. Measure release-build performance

- Extend the existing benchmark with machine-readable per-request and aggregate
  output, preserving its text output.
- Use pinned Qwen3-0.6B, explicit CUDA, one worker, batch capacity eight, greedy
  sampling, and up to 128 generated tokens.
- Test approximately 128-, 512-, and 2,048-token prompts at concurrency 1, 4, and
  8, with cold and warm prefixes; use three warmup rounds and ten measured rounds
  per case.
- Record actual prompt/output token counts, client-observed TTFT and latency
  distributions, aggregate throughput, errors, prefix counters, and sampled GPU
  memory. Label network arrival timing separately from GPU execution timing.
- Record reservation estimates separately from measured memory. Use fresh server
  processes to isolate cases and record cold-prefix nonce effects on prompt
  length.

### 4. Run a bounded reliability workload

- Run 60 minutes of mixed prompt lengths and JSON/SSE requests at concurrency
  eight, including periodic disconnects and slow readers.
- Exercise overload rejection and subsequent recovery separately from nominal
  traffic.
- Probe health/readiness throughout; verify accepted requests reconcile with
  completed, cancelled, errored, and active requests.
- Require no unexpected execution errors, malformed terminal streams, process
  crashes, or stranded request reservations. After draining, active requests,
  logical allocations, and active execution/KV reservations must return to zero;
  retained prefix memory must remain within budget.
- Finish with graceful shutdown under active generation and preserve exit status
  and drain duration.

**Deliverable:** an evidence bundle and baseline report identifying the exact
source, model, device, commands, sampling, measurements, and unresolved
limitations. No inference API changes are required for this milestone.

## Subsequent milestones, in delivery order

| Milestone | Implementation | Acceptance gate |
| --- | --- | --- |
| **Portable reference container** | Package the release CUDA server with externally mounted weights, explicit device selection, environment-based credentials, readiness checks, and signal forwarding. Preserve the existing reference execution path. | Actual GPU inference, HTTP checks, load cleanup, and shutdown pass inside the container. If Colab cannot run the container, container runtime verification remains pending until a compatible GPU host is available. |
| **Fused RMSNorm integration** | Add `--cuda-execution=auto\|reference\|optimized` and matching environment configuration. Initially keep `auto` on reference. Integrate F16 RMSNorm through Candle-managed CUDA operations with F32 accumulation and returned errors. | Kernel checks and pretrained numerical/output gates pass; release benchmarks show improvement in a targeted workload without a repeatable regression above 5% in throughput or p95 TTFT. |
| **Physical KV paging and paged attention** | Introduce a backend-owned 16-token page pool, independent of scheduler logical block IDs. Support ragged decode, stream-safe reclamation, and prefix snapshots copied into request-owned pages. Defer shared-prefix pages and copy-on-write. | Boundary lengths, fragmented tables, reuse, cancellation, eviction, errors, and shutdown pass through real pretrained inference. Repeat performance and reliability gates. |
| **Broader release qualification** | Add pinned Qwen2.5-0.5B-Instruct and sharded Qwen3-8B artifacts and independent references, then qualify additional GPUs as access becomes available. | Publish support only for validated combinations. Enable automatic optimized execution only for combinations that pass correctness and performance gates. |

Keep completion/chat interfaces stable. Add execution-path identity and
physical-pool occupancy to status and metrics when optimized execution is
introduced. Explicit unsupported optimized configurations fail clearly;
execution failures do not silently retry another path.

## Verification and defaults

- For each implementation milestone, run formatting, relevant Python regressions,
  CPU Rust tests, CUDA Clippy/tests, and actual pretrained GPU checks for the
  affected behavior.
- The complete correctness run uses:

  ```bash
  python3 scripts/gpu_validation.py run \
    --device cuda \
    --model-dir /content/rlse-models/Qwen3-0.6B \
    --include-numerics --include-kernels --include-server \
    --output /content/rlse-evidence-<unique-run-id>
  ```

  Replace `<unique-run-id>` with a fresh identifier before executing.

- Shared backend/cache changes also require pretrained Metal regression
  evidence; unavailable hardware is explicitly pending.
- Add CPU tests for benchmark aggregation, incomplete streams, stage failures,
  and evidence identity checks.
- Keep one worker as the baseline; do not remove the shared backend mutex without
  measured justification.
- Defer TPU, distributed inference, quantization, Cloudflare integration, and
  hosting-provider-specific work.
- Treat load-test measurements as the first operational baseline, not a
  production SLA or proof of driver OOM/hang recovery.

### Documentation save verification

This update saves the agreed roadmap only. Check its diff with
`git diff --check` and `git diff -- docs/colab_cuda_kernels/path_forward.md`.
No CPU regression or new GPU inference run was performed for this documentation
save. The supplied A100 results remain reported evidence; full bundle inspection,
CUDA numerical verification, and GPU verification of future implementation
changes remain pending.
