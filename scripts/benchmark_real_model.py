#!/usr/bin/env python3
"""
Scientific A/B Benchmark: Rust Mini-Serve (Candle Metal) vs. Python Mini-Inference-Engine (PyTorch MPS)
Model: Qwen/Qwen2.5-0.5B-Instruct on Apple Silicon (M4 unified memory)
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import argparse
import httpx
import psutil

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
CURRENT_MODEL_ID = DEFAULT_MODEL_ID
RUST_PORT = 8002
PYTHON_PORT = 8001
RUST_ROOT = Path("/Users/hkarimkonda/Documents/mini-serve-rs")
PYTHON_ROOT = Path("/Users/hkarimkonda/Documents/mini-inference-engine")

PROMPT_SHORT = "Explain the theory of relativity in one short sentence."

PROMPT_MEDIUM = (
    "In computer systems and distributed architectures, memory consistency models define "
    "the rules under which memory operations (reads and writes) appear to occur from the perspective "
    "of concurrent threads or nodes. Strict consistency requires that any read receives the most recent "
    "write, but this imposes severe synchronization costs across hardware cores or networks. Sequential "
    "consistency, formalized by Leslie Lamport in 1979, guarantees that the execution produces the same "
    "results as if all memory accesses were executed in some sequential order, and the operations of each "
    "individual processor appear in this sequence in the order specified by its program. Modern out-of-order "
    "processors such as Apple Silicon (ARMv8.5-A) and x86 implement relaxed memory models, where store buffers, "
    "speculative execution, and memory fences like DMB (Data Memory Barrier) dictate visibility. Please summarize "
    "how modern hardware memory models trade off sequential consistency for instruction-level parallelism."
)

PROMPT_LONG = (
    "A continuous batching inference engine is designed to maximize GPU utilization when serving Large Language Models. "
    "Traditional batching models wait for a fixed number of requests to arrive, pack them into a rectangular matrix tensor, "
    "and execute forward passes until all sequences finish. This introduces substantial inefficiencies known as the 'head-of-line blocking' "
    "and 'tail latency penalty', where shorter sequences sit idle while the longest sequence in the batch completes its generation. "
    "Continuous batching (or iteration-level scheduling), pioneered by Orca and popularized by vLLM, transforms inference into a dynamic, "
    "step-by-step iteration loop. At every decoding iteration, completed sequences that encounter an EOS token or reach their max_tokens "
    "limit are immediately evicted, free KV-cache slots are reclaimed, and newly arrived waiting requests in the scheduling queue are admitted "
    "dynamically without waiting for other streams to finish. "
    "Paged KV-cache architectures further solve GPU memory fragmentation. Standard KV-caching pre-allocates contiguous virtual memory buffers "
    "sized to the maximum possible sequence length (e.g. 2048 or 4096 tokens). In real-world workloads, actual output lengths exhibit high variance, "
    "leading to internal fragmentation where 60% to 80% of VRAM is reserved for tokens that are never generated. By segmenting the Key and Value cache "
    "into non-contiguous, fixed-size physical blocks (e.g. 16 tokens per block), the engine dynamically maps logical tokens to physical block IDs through "
    "a page table, mirroring OS virtual memory management. "
    "Prefix caching builds on this abstraction: when multiple prompts share common prefixes (such as system instructions, few-shot examples, "
    "or document contexts in Retrieval-Augmented Generation), the engine hashes 16-token chunks using cryptographic hashing (such as SHA-256) "
    "and stores the computed Key-Value activations in an LRU prefix cache. Subsequent requests sharing that prefix can bypass prefill computation "
    "entirely, jumping directly to decoding. "
    "Furthermore, in production systems, cooperative client disconnect cancellation is vital: if an HTTP client drops its TCP connection, "
    "the engine must immediately cease forward computation for that sequence, reclaim GPU blocks, and avoid wasting precious FLOPs. "
    "Please analyze the architectural synergy between continuous batching, paged memory allocation, and zero-overhead systems programming."
)


def get_process_tree_rss_mb(pid: int) -> float:
    """Sum RSS memory across parent and all descendant processes."""
    try:
        parent = psutil.Process(pid)
        total_rss = parent.memory_info().rss
        for child in parent.children(recursive=True):
            try:
                total_rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total_rss / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


async def wait_for_health(url: str, timeout: float = 45.0) -> bool:
    start = time.time()
    async with httpx.AsyncClient() as client:
        while time.time() - start < timeout:
            try:
                resp = await client.get(f"{url}/health", timeout=2.0)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.3)
    return False


async def measure_streaming_generation(client: httpx.AsyncClient, base_url: str, prompt: str, max_tokens: int):
    """Measure TTFT, inter-token latencies, and decode tok/s."""
    t0 = time.perf_counter()
    first_token_time = None
    token_times = []
    generated_text = []

    payload = {
        "model": CURRENT_MODEL_ID,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
    }

    async with client.stream("POST", f"{base_url}/v1/completions", json=payload, timeout=60.0) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                token = chunk["choices"][0].get("text", "")
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                token_times.append(now)
                generated_text.append(token)
            except Exception:
                pass

    t_end = time.perf_counter()
    total_tokens = len(token_times)
    total_latency_ms = (t_end - t0) * 1000
    ttft_ms = (first_token_time - t0) * 1000 if first_token_time else total_latency_ms
    tok_per_sec = total_tokens / (total_latency_ms / 1000.0) if total_latency_ms > 0 else 0.0

    return {
        "total_tokens": total_tokens,
        "ttft_ms": ttft_ms,
        "total_latency_ms": total_latency_ms,
        "tok_per_sec": tok_per_sec,
        "text": "".join(generated_text).strip()[:80] + "...",
    }


async def measure_pure_ttft(client: httpx.AsyncClient, base_url: str, prompt: str) -> float:
    """Measure exact Time-To-First-Token by requesting max_tokens=1."""
    t0 = time.perf_counter()
    payload = {
        "model": CURRENT_MODEL_ID,
        "prompt": prompt,
        "max_tokens": 1,
        "stream": False,
    }
    resp = await client.post(f"{base_url}/v1/completions", json=payload, timeout=60.0)
    resp.raise_for_status()
    t_end = time.perf_counter()
    return (t_end - t0) * 1000.0



async def measure_concurrency(client: httpx.AsyncClient, base_url: str, prompt: str, concurrency: int, max_tokens: int):
    """Run concurrent requests and measure aggregate throughput."""
    t0 = time.perf_counter()
    tasks = [
        measure_streaming_generation(client, base_url, prompt, max_tokens)
        for _ in range(concurrency)
    ]
    results = await asyncio.gather(*tasks)
    t_end = time.perf_counter()

    total_tokens = sum(r["total_tokens"] for r in results)
    elapsed_s = t_end - t0
    agg_throughput = total_tokens / elapsed_s if elapsed_s > 0 else 0.0
    avg_ttft = sum(r["ttft_ms"] for r in results) / len(results)

    return {
        "concurrency": concurrency,
        "total_tokens": total_tokens,
        "elapsed_s": elapsed_s,
        "agg_throughput_tok_per_sec": agg_throughput,
        "avg_ttft_ms": avg_ttft,
    }


async def benchmark_engine(name: str, base_url: str, proc: subprocess.Popen):
    print(f"\n========================================================")
    print(f"BENCHMARKING: {name} (URL: {base_url})")
    print(f"========================================================")

    idle_rss = get_process_tree_rss_mb(proc.pid)
    print(f"[*] Base Process RSS: {idle_rss:.1f} MB")

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Warmup
        print("[*] Running warmup request...")
        await measure_streaming_generation(client, base_url, "Hello!", 5)

        # 1. Short Prompt TTFT & Decode (25 tokens max)
        print("\n[1] Short Prompt (Length ~10 tokens, Max Generation: 25 tokens)")
        pure_ttft_short = await measure_pure_ttft(client, base_url, PROMPT_SHORT)
        res_short = await measure_streaming_generation(client, base_url, PROMPT_SHORT, 25)
        print(f"    - Pure Prefill TTFT (max_tokens=1): {pure_ttft_short:.2f} ms")
        print(f"    - End-to-End Latency (25 tokens): {res_short['total_latency_ms']:.2f} ms")
        print(f"    - Wall-Clock Throughput: {res_short['tok_per_sec']:.1f} tok/s")
        print(f"    - Tokens Produced: {res_short['total_tokens']}")
        print(f"    - Sample: '{res_short['text']}'")

        # 2. Medium Prompt Prefill & Decode (Prompt ~150 words / ~200 tokens, Max Generation: 50 tokens)
        print("\n[2] Medium Prompt Prefill (Length ~200 tokens, Max Generation: 50 tokens)")
        pure_ttft_med = await measure_pure_ttft(client, base_url, PROMPT_MEDIUM)
        res_med = await measure_streaming_generation(client, base_url, PROMPT_MEDIUM, 50)
        print(f"    - Pure Prefill TTFT (max_tokens=1): {pure_ttft_med:.2f} ms")
        print(f"    - End-to-End Latency (50 tokens): {res_med['total_latency_ms']:.2f} ms")
        print(f"    - Wall-Clock Throughput: {res_med['tok_per_sec']:.1f} tok/s")
        print(f"    - Tokens Produced: {res_med['total_tokens']}")

        # 3. Long Prompt Prefill & Decode (Prompt ~350 words / ~450 tokens, Max Generation: 50 tokens)
        print("\n[3] Long Prompt Prefill (Length ~450 tokens, Max Generation: 50 tokens)")
        pure_ttft_long = await measure_pure_ttft(client, base_url, PROMPT_LONG)
        res_long = await measure_streaming_generation(client, base_url, PROMPT_LONG, 50)
        print(f"    - Pure Prefill TTFT (max_tokens=1): {pure_ttft_long:.2f} ms")
        print(f"    - End-to-End Latency (50 tokens): {res_long['total_latency_ms']:.2f} ms")
        print(f"    - Wall-Clock Throughput: {res_long['tok_per_sec']:.1f} tok/s")
        print(f"    - Tokens Produced: {res_long['total_tokens']}")

        # 4. Concurrent Continuous Batching (C=4, 30 tokens each)
        print("\n[4] Concurrent Batching (4 concurrent requests, 30 tokens each)")
        res_c4 = await measure_concurrency(client, base_url, PROMPT_SHORT, concurrency=4, max_tokens=30)
        print(f"    - Total Elapsed Time: {res_c4['elapsed_s']:.2f} s")
        print(f"    - Average Latency: {res_c4['avg_ttft_ms']:.2f} ms")
        print(f"    - Aggregated Throughput: {res_c4['agg_throughput_tok_per_sec']:.1f} tok/s")
        print(f"    - Total Generated Tokens: {res_c4['total_tokens']}")

        peak_rss = get_process_tree_rss_mb(proc.pid)
        print(f"\n[*] Peak RSS during benchmark: {peak_rss:.1f} MB")

    return {
        "name": name,
        "idle_rss_mb": idle_rss,
        "peak_rss_mb": peak_rss,
        "short_ttft_ms": pure_ttft_short,
        "short_decode_tok_s": res_short["tok_per_sec"],
        "med_ttft_ms": pure_ttft_med,
        "med_decode_tok_s": res_med["tok_per_sec"],
        "long_ttft_ms": pure_ttft_long,
        "long_decode_tok_s": res_long["tok_per_sec"],
        "c4_agg_tok_s": res_c4["agg_throughput_tok_per_sec"],
        "c4_avg_ttft_ms": res_c4["avg_ttft_ms"],
    }


async def main(model_id: str = DEFAULT_MODEL_ID):
    global CURRENT_MODEL_ID
    CURRENT_MODEL_ID = model_id
    print("=" * 65)
    print("SCIENTIFIC REAL-MODEL A/B BENCHMARK ON APPLE SILICON M4")
    print(f"Model: {model_id}")
    print("=" * 65)

    # ----------------------------------------------------
    # PHASE 1: RUST MINI-SERVE (Candle Metal)
    # ----------------------------------------------------
    print(f"\n>>> Starting Rust Mini-Serve for {model_id} on port {RUST_PORT}...")
    t0_rust = time.perf_counter()
    rust_proc = subprocess.Popen(
        [
            str(RUST_ROOT / "target/release/mini-serve"),
            "--port", str(RUST_PORT),
            "--model", model_id,
            "--device", "metal",
            "--num-workers", "1",
        ],
        cwd=str(RUST_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    rust_ready = await wait_for_health(f"http://127.0.0.1:{RUST_PORT}")
    rust_cold_start = (time.perf_counter() - t0_rust) * 1000
    if not rust_ready:
        print("[!] Failed to start Rust server!")
        rust_proc.kill()
        return

    print(f"[+] Rust server ready in {rust_cold_start:.1f} ms")
    rust_results = await benchmark_engine("Rust (Candle Metal)", f"http://127.0.0.1:{RUST_PORT}", rust_proc)
    rust_results["cold_start_ms"] = rust_cold_start

    # Terminate Rust server
    rust_proc.terminate()
    try:
        rust_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        rust_proc.kill()
    time.sleep(1.0)

    # ----------------------------------------------------
    # PHASE 2: PYTHON MINI-INFERENCE-ENGINE (PyTorch MPS)
    # ----------------------------------------------------
    print(f"\n>>> Starting Python Mini-Inference-Engine for {model_id} on port {PYTHON_PORT}...")
    py_env = os.environ.copy()
    py_env["MINI_MODEL"] = model_id
    py_env["MINI_DEVICE"] = "mps"
    py_env["MINI_AUTOSCALE_ENABLED"] = "false"
    py_env["MINI_WORKER_COUNT"] = "1"
    py_env["PORT"] = str(PYTHON_PORT)

    t0_py = time.perf_counter()
    py_proc = subprocess.Popen(
        [
            str(PYTHON_ROOT / ".venv/bin/uvicorn"),
            "mini_inference_engine.api.app:app",
            "--host", "0.0.0.0",
            "--port", str(PYTHON_PORT),
        ],
        cwd=str(PYTHON_ROOT),
        env=py_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    py_ready = await wait_for_health(f"http://127.0.0.1:{PYTHON_PORT}")
    py_cold_start = (time.perf_counter() - t0_py) * 1000
    if not py_ready:
        print("[!] Failed to start Python server!")
        py_proc.kill()
        return

    print(f"[+] Python server ready in {py_cold_start:.1f} ms")
    py_results = await benchmark_engine("Python (PyTorch MPS)", f"http://127.0.0.1:{PYTHON_PORT}", py_proc)
    py_results["cold_start_ms"] = py_cold_start

    # Terminate Python server and any child workers
    py_proc.terminate()
    try:
        py_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        py_proc.kill()

    # Clean up any lingering worker processes
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(proc.info["cmdline"] or [])
            if "mini_inference_engine" in cmd and proc.pid != os.getpid():
                os.kill(proc.pid, signal.SIGKILL)
        except Exception:
            pass

    # ----------------------------------------------------
    # PHASE 3: COMPARISON SUMMARY
    # ----------------------------------------------------
    print("\n" + "=" * 65)
    print("FINAL A/B REAL-MODEL BENCHMARK RESULTS")
    print("=" * 65)

    headers = [
        "Metric",
        "Python (PyTorch MPS)",
        "Rust (Candle Metal)",
        "Delta / Improvement",
    ]
    rows = [
        ("Cold Start to Ready", f"{py_results['cold_start_ms']:.0f} ms", f"{rust_results['cold_start_ms']:.0f} ms", f"{(py_results['cold_start_ms'] / rust_results['cold_start_ms']):.1f}x faster"),
        ("Base Idle RSS (Process Tree)", f"{py_results['idle_rss_mb']:.1f} MB", f"{rust_results['idle_rss_mb']:.1f} MB", f"{(py_results['idle_rss_mb'] / rust_results['idle_rss_mb']):.1f}x smaller"),
        ("Peak Unified RSS Under Load", f"{py_results['peak_rss_mb']:.1f} MB", f"{rust_results['peak_rss_mb']:.1f} MB", f"{(py_results['peak_rss_mb'] / rust_results['peak_rss_mb']):.1f}x smaller"),
        ("TTFT (Short ~10 tokens)", f"{py_results['short_ttft_ms']:.2f} ms", f"{rust_results['short_ttft_ms']:.2f} ms", f"{(py_results['short_ttft_ms'] - rust_results['short_ttft_ms']):.1f} ms diff"),
        ("Decode Speed (Short)", f"{py_results['short_decode_tok_s']:.1f} tok/s", f"{rust_results['short_decode_tok_s']:.1f} tok/s", f"{((rust_results['short_decode_tok_s'] / py_results['short_decode_tok_s'] - 1) * 100):+.1f}%"),
        ("TTFT (Medium ~200 tokens)", f"{py_results['med_ttft_ms']:.2f} ms", f"{rust_results['med_ttft_ms']:.2f} ms", f"{(py_results['med_ttft_ms'] - rust_results['med_ttft_ms']):.1f} ms diff"),
        ("Decode Speed (Medium)", f"{py_results['med_decode_tok_s']:.1f} tok/s", f"{rust_results['med_decode_tok_s']:.1f} tok/s", f"{((rust_results['med_decode_tok_s'] / py_results['med_decode_tok_s'] - 1) * 100):+.1f}%"),
        ("TTFT (Long ~450 tokens)", f"{py_results['long_ttft_ms']:.2f} ms", f"{rust_results['long_ttft_ms']:.2f} ms", f"{(py_results['long_ttft_ms'] - rust_results['long_ttft_ms']):.1f} ms diff"),
        ("Decode Speed (Long)", f"{py_results['long_decode_tok_s']:.1f} tok/s", f"{rust_results['long_decode_tok_s']:.1f} tok/s", f"{((rust_results['long_decode_tok_s'] / py_results['long_decode_tok_s'] - 1) * 100):+.1f}%"),
        ("Continuous Batching (C=4)", f"{py_results['c4_agg_tok_s']:.1f} tok/s", f"{rust_results['c4_agg_tok_s']:.1f} tok/s", f"{((rust_results['c4_agg_tok_s'] / py_results['c4_agg_tok_s'] - 1) * 100):+.1f}%"),
        ("Continuous Batching TTFT", f"{py_results['c4_avg_ttft_ms']:.2f} ms", f"{rust_results['c4_avg_ttft_ms']:.2f} ms", f"{(py_results['c4_avg_ttft_ms'] - rust_results['c4_avg_ttft_ms']):.1f} ms diff"),
    ]

    col_widths = [30, 24, 24, 20]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_str = "|" + "|".join(f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)) + "|"

    print(sep)
    print(header_str)
    print(sep)
    for r in rows:
        line = "|" + "|".join(f" {r[i]:<{col_widths[i]}} " for i in range(len(r))) + "|"
        print(line)
    print(sep)

    # Save results to JSON
    output_path = RUST_ROOT / "target" / "real_model_benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"rust": rust_results, "python": py_results}, f, indent=2)
    print(f"\n[+] Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scientific real-model A/B benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Model ID or path (default: {DEFAULT_MODEL_ID})",
    )
    args = parser.parse_args()
    asyncio.run(main(args.model))
