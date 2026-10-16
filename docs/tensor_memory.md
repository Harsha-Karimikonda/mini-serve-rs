# Tensor memory reservations

The logical KV block allocator and Candle tensor memory have separate limits.
Logical blocks control token admission and remain visible under `cache` in
`/api/status`. Candle uses contiguous, request-owned KV buffers, rather than a
physical tensor page pool.

Configure the shared tensor reservation budget with:

```bash
mini-serve --model Qwen/Qwen3-0.6B --device cpu \
  --tensor-memory-budget-bytes 2147483648 \
  --execution-headroom-bytes 268435456
```

The defaults are 8 GiB for the budget and 256 MiB for fixed execution headroom.
Both have environment equivalents: `MINI_TENSOR_MEMORY_BUDGET_BYTES` and
`MINI_EXECUTION_HEADROOM_BYTES`. Budget must be positive and strictly greater
than headroom; zero headroom is supported for controlled tests.

## Active requests

Before the scheduler allocates logical blocks or executes a forward pass,
Candle reserves the request's full KV capacity and estimated workspace. All
workers using a prefix cache share its byte budget. Queued requests have no
active tensor reservation. Requests that cannot reserve enough memory evict
cached prefixes in LRU order, then fail explicitly if capacity is still
insufficient. An admitted JSON request reports an error; an established SSE
stream emits an error event. No forward pass runs for the rejected request.

For `T = prompt_tokens + max_tokens`, the KV reservation is:

```text
2 * num_layers * num_kv_heads * head_dim * T * dtype_bytes
```

The request's physical contiguous cache is configured to `T` slots before first
use, rather than allocating the model's entire context for every request. All
multiplications and sums in the estimate are checked for overflow.

Execution reservations additionally cover a KV-sized staging allowance,
attention matrices, hidden/MLP activations, grouped decode buffers and logits.
For `P` prompt tokens, `H` attention heads, `D` head dimension, `B` dtype bytes,
`S` hidden size, `I` intermediate size, and `V` vocabulary size, the estimate is:

```text
KV + 8*H*P*T*max(B,4) + 16*S*P*B + 8*I*P*B
   + 8*H*T*D*B + 16*V*max(B,4)
```

Workspaces are conservatively reserved for every active request even when
prefills execute sequentially or reuse cached prefixes. Cancellation, normal
completion, logical allocation failure, and backend errors release active
reservations after dropping model state.

## Prefix storage

Each retained prefix has independent, compact K/V tensors and optional logits.
Copies use `force_contiguous().detach()` so even a contiguous narrow view cannot
pin the original request's larger buffer or computation graph. The reservation
covers each owned tensor's elements times its dtype size, before copying.
Cumulative snapshots at different chunk boundaries are separate allocations
and are counted separately. Metadata-only entries retain no tensor bytes.

Lookup returns a `PrefixSnapshot` guard. Clones share one reservation, and
eviction releases its bytes only after the last guard drops. Library callers
must keep the guard alive while using its tensor references. Admission cannot
reclaim memory still borrowed by a forward pass. A longer cumulative snapshot
remains usable after intermediate entries have been evicted because its chained
hash identifies the complete token prefix.

## Telemetry and limits

`/api/status` exposes `tensor_memory` with budget, fixed headroom, active KV,
estimated execution, retained prefixes, total reservations, available bytes,
peak reservations, and rejected reservations. Prefix storage includes snapshots
borrowed after eviction. Prefix eviction counts appear under
`prefix_cache.tensor_evictions`. Prometheus exposes corresponding
`mini_tensor_memory_*_bytes` gauges, `mini_tensor_memory_rejections_total`, and
`mini_prefix_tensor_evictions_total`.

These are reservations and estimates, not measured GPU allocation or process
RSS. Immutable model weights, rotary tables, tokenizer/host metadata, allocator
rounding and retained device pools, and kernel-private workspace are excluded
from exact accounting. Leave room for those resources outside the configured
budget; fixed headroom provides an additional allowance within it. The budget
bounds charged reservations, not total physical memory, and does not implement
PagedAttention. Mock requests do not allocate KV tensors and consume no active
KV reservation.

CPU regressions check underlying storage sizes, mixed request capacities,
concurrent budget contention, pinned snapshots, eviction, cancellation, and
backend-error cleanup. Metal-feature builds run these fixtures on CPU. The
[pretrained Metal GPU check](gpu_validation.md) now verifies prefix reuse,
shared-worker output, cancellation cleanup, budget rejection, eviction, and
subsequent inference. Device allocator measurements, forced device OOM, and
broad load validation remain open.
