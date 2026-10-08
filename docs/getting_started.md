# Getting Started with Mini-Serve 🚀

This guide walks you through compiling, configuring, running, and testing Mini-Serve on Apple Silicon (macOS) and Linux/NVIDIA environments.

---

## 1. Prerequisites

- **Rust Toolchain**: Rust 1.80+ (`rustup` recommended)
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source "$HOME/.cargo/env"
  ```
- **Operating System & Hardware**:
  - **macOS**: Apple Silicon (M1/M2/M3/M4) running macOS 13+ with Xcode Command Line Tools (`xcode-select --install`).
  - **Linux / Windows**: Linux x86_64 or aarch64, or Windows with MSVC. For NVIDIA acceleration, CUDA 11.8+ or 12.x and `nvcc` installed.
- **Hugging Face Token (Optional)**:
  Only required when downloading gated or rate-limited models from Hugging Face Hub:
  ```bash
  export HF_TOKEN="hf_your_token_here"
  ```

---

## 2. Building the Engine

Mini-Serve is configured using Cargo feature flags.

### Apple Silicon (Metal GPU Acceleration)
```bash
# Build optimized release binary with Metal shader acceleration
cargo build --release --features metal
```
The compiled self-contained binary will be created at:
`target/release/mini-serve` (~14.0 MB).

### Standard CPU (Portable)
```bash
cargo build --release
```

---

## 3. Running the Server

### A. Development / Testing Mode (Mock Backend)
The fastest way to test the API gateway, continuous batching scheduler, and dashboard is using the built-in mock backend:
```bash
./target/release/mini-serve --model mock --port 8000
```
This simulates a 3-millisecond per-token forward pass without requiring GPU memory or model downloads.

### B. Real Hugging Face Model (Metal GPU Acceleration)
To serve a real model (such as `Qwen/Qwen2.5-0.5B-Instruct` or `Qwen/Qwen3-0.6B`):
```bash
./target/release/mini-serve \
  --model Qwen/Qwen3-0.6B \
  --device metal \
  --port 8000 \
  --num-workers 2
```
If the model weights are not found locally in `models/`, Mini-Serve will automatically download the `config.json`, `tokenizer.json`, and `model.safetensors` files from Hugging Face Hub into `~/.cache/huggingface/hub/` and memory-map them directly into unified RAM.

### C. Offline / Local Directory Serving
If you already have models downloaded on disk:
```bash
./target/release/mini-serve \
  --model models/Qwen3-0.6B \
  --device metal \
  --port 8000
```

---

## 4. Configuration Reference

All settings can be specified either via command-line arguments or environment variables:

| Argument | Environment Variable | Default | Description |
| :--- | :--- | :---: | :--- |
| `--host` | `MINI_HOST` | `0.0.0.0` | IP address to bind the HTTP server to. |
| `--port` | `MINI_PORT` | `8000` | Port to bind the HTTP server to. |
| `--model` | `MINI_MODEL` | `mock` | Model identifier (`mock`, local path, or HF slug like `Qwen/Qwen3-0.6B`). |
| `--device` | `MINI_DEVICE` | `auto` | Device backend (`metal`, `cuda`, `cpu`, or `auto`). |
| `--num-workers` | `MINI_NUM_WORKERS` | `2` | Number of concurrent continuous batching worker threads. |
| `--max-batch-size` | `MINI_MAX_BATCH_SIZE` | `8` | Maximum concurrent active sequences per worker iteration. |
| `--max-waiting-requests` | `MINI_MAX_WAITING_REQUESTS` | `256` | Maximum queue depth before fast-failing with HTTP 429. |
| `--kv-cache-blocks` | `MINI_KV_CACHE_BLOCKS` | `1024` | Total physical KV-cache blocks allocated in memory. |
| `--block-size` | `MINI_BLOCK_SIZE` | `16` | Number of tokens stored per physical block. |
| `--log-level` | `MINI_LOG_LEVEL` | `info` | Logging verbosity (`trace`, `debug`, `info`, `warn`, `error`). |
| `--hf-token` | `HF_TOKEN` | `None` | Authentication token for Hugging Face Hub downloads. |

---

## 5. Verification & Testing

Mini-Serve includes an extensive automated integration test suite:

```bash
# Run all unit and integration tests with Metal features enabled
cargo test --features metal

# Run a specific integration test with console logging
cargo test --features metal --test test_qwen3 -- --nocapture

# Run clippy linter with zero warnings tolerance
cargo clippy --features metal --all-targets -- -D warnings

# Verify formatting
cargo fmt -- --check
```
