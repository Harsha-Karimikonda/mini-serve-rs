"""Run repeated benchmarks and Ctrl-C draining against the real GPU HTTP server.

Build server + benchmark with the desired GPU feature first. Requires local
pinned Qwen3-0.6B weights; selects an explicit device and checks its startup log.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--server", default="target/debug/rlse")
parser.add_argument("--bench", default="target/debug/examples/bench")
parser.add_argument("--model", default="models/Qwen3-0.6B")
parser.add_argument("--device", choices=["metal", "cuda"], default="metal")
parser.add_argument("--output", type=Path)
args = parser.parse_args()
assert (Path(args.model) / "model.safetensors").is_file(), "local weights required"
with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
base = f"http://127.0.0.1:{port}"
key = secrets.token_hex(32)
env = dict(os.environ, RUST_LOG="rlse=info", RLSE_API_KEY=key, RLSE_CLIENT_REQUESTS_PER_MINUTE="0", RLSE_CLIENT_MAX_CONCURRENT="0")
env.pop("RLSE_API_KEYS", None)
lines = []

def report(line):
    lines.append(line)
    print(line, flush=True)

def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=35)

with tempfile.TemporaryFile(mode="w+") as log:
    server = subprocess.Popen([str(Path(args.server).resolve()), "--host", "127.0.0.1", "--port", str(port), "--model", args.model, "--device", args.device, "--num-workers", "1", "--shutdown-timeout-ms", "30000", "--tensor-memory-budget-bytes", "536870912", "--execution-headroom-bytes", "33554432"], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 60
        while True:
            assert server.poll() is None, "server exited before readiness"
            try:
                with request("/ready") as response:
                    assert json.load(response)["ready"]
                break
            except (OSError, AssertionError):
                assert time.monotonic() < deadline, "readiness timeout"
                time.sleep(0.1)
        log.seek(0)
        startup = log.read()
        device_line = next(line for line in startup.splitlines() if "Using device:" in line)
        assert ("Metal(" if args.device == "metal" else "Cuda(") in device_line, device_line
        report(device_line)
        report("build=debug/test; workers=1; tensor_budget=536870912; headroom=33554432; sampling=greedy temperature=0 top_p=1")
        prompt = "The capital of France is" + " France" * 27
        for mode in ("warm", "cold"):
            command = [args.bench, "--url", base, "--model", args.model, "--prefix-mode", mode, "--prompt", prompt, "--max-tokens", "12", "--warmups", "1", "--rounds", "3", "--concurrency", "2"]
            result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=180)
            assert result.returncode == 0, result.stdout + result.stderr
            report(result.stdout.rstrip())
            metrics = [line for line in result.stdout.splitlines() if "server_prefix_hits=" in line]
            assert len(metrics) == 3
            for metric in metrics:
                saved = int(metric.split("server_prefix_tokens_saved=")[1])
                assert saved == 0 if mode == "cold" else saved > 0, metric
        with request("/api/status") as response:
            status = json.load(response)
        assert status["execution"]["accepted"] == status["execution"]["completed"]
        assert status["execution"]["errors"] == status["execution"]["cancellations"] == 0
        assert status["cache"]["allocated_blocks"] == 0
        report("post_benchmark_execution=" + json.dumps(status["execution"], sort_keys=True))
        with request("/v1/completions", {"model": args.model, "prompt": "The capital of France is", "temperature": 0, "top_p": 1, "max_tokens": 12, "stream": True}) as response:
            signalled = False
            done = False
            terminal = None
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                data = raw[6:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                assert "error" not in chunk, chunk
                if not signalled:
                    started = time.monotonic()
                    server.send_signal(signal.SIGINT)
                    signalled = True
                if chunk["choices"][0]["finish_reason"]:
                    terminal = chunk
            assert signalled and done and terminal, "GPU stream did not drain"
            assert terminal["usage"]["completion_tokens"] == 12
        assert server.wait(timeout=35) == 0
        report(f"PASS: Ctrl-C after first GPU SSE token drains 12-token request and exits cleanly; signal_to_exit_seconds={time.monotonic()-started:.3f}")
    finally:
        if server.poll() is None:
            server.kill()
            server.wait(timeout=5)
        log.seek(0)
        if server.returncode:
            report(log.read())
if args.output:
    args.output.write_text("\n".join(lines) + "\n")
