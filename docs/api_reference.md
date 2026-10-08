# API & Telemetry Reference 📡

Mini-Serve exposes a fully OpenAI-compatible REST API alongside Prometheus metrics, health checks, and a live web dashboard.

---

## 1. OpenAI-Compatible Endpoints

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
      "owned_by": "mini-serve-rs"
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

