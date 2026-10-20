# API & Telemetry Reference 📡

Mini-Serve exposes a fully OpenAI-compatible REST API alongside Prometheus metrics, health checks, and a live web dashboard.

---

## 1. OpenAI-Compatible Endpoints

Both generation endpoints require `max_tokens > 0`, finite `temperature` in
`[0, 2]` (`0` selects greedy decoding), and finite `top_p` in `(0, 1]`.
Stop strings must be nonempty. Invalid values return HTTP 400 before generation,
including when `stream: true`.

Stop matching is shared by both endpoints and backends. Partial stop prefixes
are withheld across decoded token boundaries. The first completed stop wins
(ties prefer the earliest start), its text is removed, and `finish_reason` is
`stop`. Unmatched buffered text is flushed on EOS or the token limit. Qwen's
`<|im_end|>` and `<|endoftext|>` markers are suppressed. Usage counts generated
token IDs, including tokens with no visible text and EOS, so visible chunks
cannot be used as an exact token count. Successful SSE token telemetry records
the terminal usage when generation finishes.

The tokenized prompt plus `max_tokens` must fit the model context (HTTP 400) and
the configured logical KV capacity (HTTP 429 with `Retry-After`). Capacity is
`kv_cache_blocks * block_size`, shared by workers; it is not a tensor-memory
budget. Concurrent requests can exhaust available blocks after admission. Such
requests terminate explicitly with an error (HTTP 500 for a collected JSON
response, or an SSE error event after streaming begins), never a successful
`length` completion. Their logical blocks and backend state are released.

### A. Completions (`POST /v1/completions`)

#### Request Body
```json
{
  "model": "Qwen/Qwen3-0.6B",
  "prompt": "The capital of France is",
  "max_tokens": 20,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": true
}
```

#### Non-Streaming Response (`stream: false`)
```json
{
  "id": "cmpl-a1b2c3d4-e5f6-7890",
  "object": "text_completion",
  "created": 1791492088,
  "model": "Qwen/Qwen3-0.6B",
  "choices": [
    {
      "text": " Paris. The capital of Italy is Rome.",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 10,
    "total_tokens": 15
  }
}
```

#### Streaming Response (`stream: true`)
Server-Sent Events (SSE) format:
```text
data: {"id":"cmpl-a1b2c3d4","object":"text_completion","created":1791492088,"model":"Qwen/Qwen3-0.6B","choices":[{"text":" Paris","index":0,"finish_reason":null}]}

data: {"id":"cmpl-a1b2c3d4","object":"text_completion","created":1791492088,"model":"Qwen/Qwen3-0.6B","choices":[{"text":".","index":0,"finish_reason":"stop"}]}

data: [DONE]
```

---

### B. Chat Completions (`POST /v1/chat/completions`)

Chat uses the loaded model's `tokenizer_config.json` string chat template.
Supported roles are `system`, `user`, and `assistant`, with text content.
Missing/unsupported templates return HTTP 400; ordinary completions remain
available. Model names on both inference endpoints must exactly match
`GET /v1/models`. Only loaded identities are listed; explicit mock fallback is
reported as `mock`. See [behavior and verification](chat_correctness.md).


#### Request Body
```json
{
  "model": "Qwen/Qwen3-0.6B",
  "messages": [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "What is the speed of light?"}
  ],
  "max_tokens": 30,
  "stream": true
}
```

#### Streaming Response (`stream: true`)
```text
data: {"id":"chatcmpl-x1y2","object":"chat.completion.chunk","created":1791492100,"model":"Qwen/Qwen3-0.6B","choices":[{"index":0,"delta":{"content":"Approximately"},"finish_reason":null}]}

data: [DONE]
```

---

### C. Models List (`GET /v1/models`)

```bash
curl http://localhost:8000/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3-0.6B",
      "object": "model",
      "created": 1791492084,
      "owned_by": "rlse"
    }
  ]
}
```

---

## 2. Health & Monitoring Endpoints

### A. Health Check (`GET /health`)
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "ok",
  "model": "Qwen/Qwen3-0.6B",
  "device": "metal",
  "workers": 2
}
```

### B. Detailed Status (`GET /api/status`)
Returns real-time engine telemetry:
```json
{
  "status": "healthy",
  "version": "0.2.0",
  "tokens_per_sec": 41.2,
  "total_tokens": 1420,
  "total_requests": 85,
  "cache": {
    "total_blocks": 1024,
    "allocated_blocks": 18,
    "free_blocks": 1006,
    "utilization": 0.0175,
    "fragmentation": 0.125
  },
  "prefix_cache": {
    "cached_chunks": 42,
    "hits": 28,
    "misses": 14,
    "tokens_saved": 448
  },
  "workers": [
    {
      "id": "worker-1",
      "active_count": 2,
      "queue_depth": 0,
      "avg_latency_ms": 24.1,
      "score": 72.3,
      "healthy": true
    }
  ]
}
```

### C. Prometheus Metrics (`GET /metrics`)
Exposes plain-text metrics formatted for Prometheus scraping:

```prometheus
# HELP mini_tokens_per_second Current token generation throughput
# TYPE mini_tokens_per_second gauge
mini_tokens_per_second 41.2

# HELP mini_tokens_total Cumulative tokens generated
# TYPE mini_tokens_total counter
mini_tokens_total 1420

# HELP mini_requests_total Cumulative inference requests
# TYPE mini_requests_total counter
mini_requests_total 85

# HELP mini_cache_utilization KV cache utilization fraction
# TYPE mini_cache_utilization gauge
mini_cache_utilization 0.0175

# HELP mini_cache_fragmentation KV cache fragmentation fraction
# TYPE mini_cache_fragmentation gauge
mini_cache_fragmentation 0.125
```

---

## 3. Real-Time Web Dashboard (`GET /dashboard`)

Mini-Serve compiles an embedded dark-mode HTML5/CSS3 control dashboard directly into the binary:
- **URL**: `http://localhost:8000/dashboard` or `http://localhost:8000/`
- **Features**:
  - Live throughput gauge (tokens/sec).
  - Paged KV-cache utilization and fragmentation progress bars.
  - Prefix cache hit rate and token savings meter.
  - Multi-worker load status and queue backlog meters.
  - Interactive prompt testing playground with real-time token streaming.


## Tensor memory telemetry

`/api/status` reports `tensor_memory` separately from logical `cache` blocks:
`budget_bytes`, `headroom_bytes`, `active_kv_reserved_bytes`,
`execution_reserved_bytes`, `prefix_retained_bytes`, `reserved_bytes`,
`available_bytes`, `peak_reserved_bytes`, and `rejections`.
`prefix_cache.tensor_evictions` counts evicted tensor entries.
The byte fields have corresponding `mini_tensor_memory_*_bytes` Prometheus
gauges; rejection and eviction counters are `mini_tensor_memory_rejections_total`
and `mini_prefix_tensor_evictions_total`. These are conservative reservations,
with model weights excluded. See [memory accounting](tensor_memory.md).


### Execution readiness and failures

`GET /health` reports HTTP liveness. `GET /ready` returns HTTP 200 with
`{"ready":true}` when an eligible execution path exists, or HTTP 503 with
`{"ready":false}` otherwise. Returned Candle execution/allocation failures
quarantine every worker sharing that backend until process restart. Already
admitted requests receive explicit JSON/SSE errors; subsequent admission returns
503. Tensor reservation pressure alone leaves the backend eligible. Worker
quarantine is reflected in `/api/status` and `mini_ready_workers` in `/metrics`.
See [allocation and execution failures](device_failures.md) for cleanup, recovery,
custom backend hooks, and the limits of readiness and device verification.
