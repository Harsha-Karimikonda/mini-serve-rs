# Mini-Serve Examples & Use Cases ⚡

This directory contains standalone, reproducible examples and user-facing applications built on top of the **Mini-Serve** inference engine.

---

## Overview of Examples

| Example | Language | File | Description |
| :--- | :--- | :--- | :--- |
| **Interactive CLI Chat** | Rust | [`cli_chat.rs`](cli_chat.rs) | Interactive terminal chat REPL with real-time SSE token streaming, Time-To-First-Token (TTFT) metrics, and multi-turn context. |
| **RAG & Prefix Caching** | Rust | [`prefix_cache_rag.rs`](prefix_cache_rag.rs) | Demonstrates 16-token chunk prefix caching. Compares cold prefill vs. warm cached query latency to showcase prefill bypass. |
| **Continuous Batching** | Rust | [`concurrent_batching.rs`](concurrent_batching.rs) | Fires concurrent requests with variable sequence lengths to demonstrate dynamic iteration admission and zero head-of-line blocking. |
| **Client Disconnect** | Rust | [`client_cancellation.rs`](client_cancellation.rs) | Demonstrates cooperative client disconnect detection. Cancelling mid-stream reclaims KV blocks within a single iteration (<25ms). |
| **Python OpenAI SDK** | Python | [`python_openai_client.py`](python_openai_client.py) | Drop-in replacement for the official `openai` Python SDK. Shows non-streaming and streaming chat completions. |
| **Benchmark Harness** | Rust | [`bench.rs`](bench.rs) | High-throughput benchmark client measuring P50/P99 latency, TTFT, and tokens/sec under sustained load. |

---

## 1. Prerequisites

First, ensure the Mini-Serve server is running. You can run it either in `mock` mode (for fast, weightless development) or with real Hugging Face model weights on Apple Silicon Metal or CUDA.

```bash
# Option A: Fast Mock Backend (default on port 8000)
cargo run --release

# Option B: Real Model with Metal GPU Acceleration
cargo run --release --features metal -- --model Qwen/Qwen2.5-0.5B-Instruct --device metal
```

Verify that the server is up:
```bash
curl http://localhost:8000/health
```

---

## 2. Running the Examples

### 💬 Interactive Terminal Chat (`cli_chat`)

Launch an interactive chat session in your terminal with real-time token streaming and per-response latency stats:

```bash
# Against mock backend:
cargo run --example cli_chat

# Against a real model running on port 8000:
cargo run --example cli_chat -- --url http://127.0.0.1:8000 --model Qwen/Qwen2.5-0.5B-Instruct
```

**Commands inside chat**:
- Type your message and hit Enter to stream the assistant's reply.
- Type `/clear` to reset conversation context.
- Type `/exit` or `/quit` to exit.

---

### 🚀 Prefix Cache & RAG Acceleration (`prefix_cache_rag`)

Demonstrates how Mini-Serve's 16-token chained SHA-256 prefix cache accelerates Retrieval-Augmented Generation (RAG) by bypassing prefill computation on shared context:

```bash
cargo run --example prefix_cache_rag
```

**What it does**:
1. **Query 1 (Cold)**: Submits a prompt containing a multi-paragraph reference document. Mini-Serve performs the full prefill pass and stores KV activations in the prefix cache.
2. **Query 2 (Warm)**: Submits a *new* question against the *exact same* document. Mini-Serve hits the prefix cache, bypassing the prefill pass and immediately beginning token decoding.
3. Prints a side-by-side TTFT (Time-to-First-Token) comparison.

---

### 🔄 Concurrent Continuous Batching (`concurrent_batching`)

Dispatches multiple concurrent streams with heterogeneous token limits to demonstrate iteration-level scheduling:

```bash
cargo run --example concurrent_batching -- --concurrency 6
```

**Key Takeaway**:
Notice how short requests (e.g., 15 tokens) finish first and immediately exit, freeing batch slots for new requests while longer queries continue to decode. Traditional static batching would force the short queries to wait until the longest finished.

---

### 🛑 Client Disconnect Cooperative Cancellation (`client_cancellation`)

Demonstrates that abandoned client streams do not waste GPU FLOPs:

```bash
cargo run --example client_cancellation
```

**What it does**:
1. Dispatches a 200-token generation request.
2. Reads 8 tokens, then forcefully aborts the TCP connection (simulating a user closing their browser tab or navigating away).
3. Queries `/api/status` immediately to show that the scheduler evicted the request and released its allocated KV-cache blocks.

---

### 🐍 Python Drop-In (OpenAI SDK) (`python_openai_client.py`)

Mini-Serve exposes standard OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`, and `/v1/models`).

Install the OpenAI client:
```bash
pip install openai
```

Run the Python example:
```bash
python examples/python_openai_client.py
```

**Sample Code**:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="none"
)

# Streaming Chat Completion
stream = client.chat.completions.create(
    model="mock",
    messages=[{"role": "user", "content": "Explain continuous batching."}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

