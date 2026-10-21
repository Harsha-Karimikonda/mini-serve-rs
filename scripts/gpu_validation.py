#!/usr/bin/env python3
"""Reproducible source transport and explicit pretrained GPU validation (stdlib only)."""
import argparse
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import tempfile
import urllib.request

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = {"src", "tests", "examples", "scripts", "docs", ".github", ".githooks"}
SOURCE_FILES = {"Cargo.toml", "Cargo.lock", "Makefile", "README.md", "AGENTS.md", ".gitignore", "rustfmt.toml", "build.rs"}
SUFFIXES = {".rs", ".py", ".md", ".json", ".toml", ".yml", ".yaml", ".cu", ".ipynb"}
MANIFEST = "source-manifest.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def allowed(name):
    p = Path(name)
    return (not p.is_absolute() and ".." not in p.parts
            and not any(x in {".git", "target", "models", "__pycache__", "evidence", "node_modules"} for x in p.parts)
            and not any(x.startswith(".env") or any(s in x.lower() for s in ("credential", "secret", "private_key")) for x in p.parts)
            and (name in SOURCE_FILES or (p.parts[0] in SOURCE_DIRS and
                 (p.suffix in SUFFIXES or p.parts[0] == ".githooks"))))


def source_paths(root):
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                            cwd=root, check=True, stdout=subprocess.PIPE)
    return sorted({n for n in result.stdout.decode().split("\0") if n and allowed(n)
                   and (root / n).is_file()})


def snapshot(root):
    files = {}
    for name in source_paths(root):
        path = root / name
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root.parent):
            raise ValueError(f"source symlinks are unsupported: {name}")
        files[name] = {"sha256": digest(path), "size": path.stat().st_size,
                       "executable": bool(path.stat().st_mode & 0o111)}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root))
    return {"schema": 1, "base_commit": commit, "dirty": dirty, "files": files}


def verify_source(root, manifest):
    if manifest.get("schema") != 1 or not manifest.get("files"):
        raise ValueError("missing or unsupported source manifest")
    for name, entry in manifest["files"].items():
        if not allowed(name):
            raise ValueError(f"invalid source path: {name}")
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
            raise ValueError(f"unsafe source path: {name}")
        if not path.is_file() or digest(path) != entry["sha256"] or path.stat().st_size != entry["size"]:
            raise ValueError(f"source hash/size mismatch: {name}")
    # Unmanifested source could change compilation or execute in the notebook.
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {".git", "target", "models", "__pycache__", "evidence", "node_modules"}]
        for filename in files:
            path = Path(directory) / filename
            name = path.relative_to(root).as_posix()
            if allowed(name) and name not in manifest["files"]:
                raise ValueError(f"unmanifested source: {name}")


NOTEBOOK_REL = "docs/colab_cuda_kernels/test_colab.ipynb"


def set_notebook_payload(notebook_path, archive_b64="", metadata_b64="", script_b64=""):
    if not notebook_path.is_file():
        return False
    nb = json.loads(notebook_path.read_text())
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        lines = cell.get("source", [])
        new_lines = []
        for line in lines:
            if line.startswith("EMBEDDED_ARCHIVE_B64 = "):
                rep = f'EMBEDDED_ARCHIVE_B64 = "{archive_b64}"\n'
                changed = changed or (line != rep)
                new_lines.append(rep)
            elif line.startswith("EMBEDDED_METADATA_B64 = "):
                rep = f'EMBEDDED_METADATA_B64 = "{metadata_b64}"\n'
                changed = changed or (line != rep)
                new_lines.append(rep)
            elif line.startswith("EMBEDDED_SCRIPT_B64 = "):
                rep = f'EMBEDDED_SCRIPT_B64 = "{script_b64}"\n'
                changed = changed or (line != rep)
                new_lines.append(rep)
            else:
                new_lines.append(line)
        cell["source"] = new_lines
    if changed:
        notebook_path.write_text(json.dumps(nb, indent=1) + "\n")
    return changed


def pack(root, output, embed_notebook=False):
    notebook = root / NOTEBOOK_REL
    set_notebook_payload(notebook)
    manifest = snapshot(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        with tarfile.open(fileobj=gz, mode="w") as archive:
            for name, entry in manifest["files"].items():
                data = (root / name).read_bytes()
                if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                    raise ValueError(f"source changed during packaging: {name}")
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(data), 0o755 if entry["executable"] else 0o644
                archive.addfile(info, io.BytesIO(data))
            data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            info = tarfile.TarInfo(MANIFEST)
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))
    sidecar = Path(str(output) + ".json")
    write_json(sidecar, {"archive_sha256": digest(output), "source": manifest})
    if embed_notebook and notebook.is_file():
        set_notebook_payload(
            notebook,
            base64.b64encode(output.read_bytes()).decode("ascii"),
            base64.b64encode(sidecar.read_bytes()).decode("ascii"),
            base64.b64encode((root / "scripts/gpu_validation.py").read_bytes()).decode("ascii"),
        )
    print(f"{output}: sha256={digest(output)}, files={len(manifest['files'])}")


def unpack(archive, destination, expected):
    if digest(archive) != expected:
        raise ValueError("archive SHA-256 mismatch")
    # Require a fresh destination; never merge stale source or overwrite a checkout.
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as bundle:
        seen = set()
        for member in bundle.getmembers():
            if (member.name != MANIFEST and not allowed(member.name)) or not member.isfile() or member.name in seen:
                raise ValueError(f"unsafe or duplicate archive member: {member.name}")
            seen.add(member.name)
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as src, path.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            path.chmod(0o755 if member.mode & 0o111 else 0o644)
    verify_source(destination, json.loads((destination / MANIFEST).read_text()))
    print(f"Verified source in {destination}")


def model_manifest(root):
    fixture = json.loads((root / "tests/fixtures/generation/qwen3-reference.json").read_text())
    return {k: fixture[k] for k in ("model", "revision", "hashes")}


def verify_model(directory, manifest):
    for name, expected in manifest["hashes"].items():
        path = directory / name
        if not path.is_file():
            raise ValueError(f"missing pinned model artifact: {name}")
        if digest(path) != expected:
            raise ValueError(f"model artifact SHA-256 mismatch: {name}")


def download_model(root, directory):
    manifest = model_manifest(root)
    directory.mkdir(parents=True, exist_ok=True)
    items = list(manifest["hashes"].items())
    for idx, (name, expected) in enumerate(items, 1):
        path = directory / name
        if path.exists():
            print(f"[{idx}/{len(items)}] Verifying cached {name}...", flush=True)
            if digest(path) != expected:
                raise ValueError(f"refusing to overwrite mismatched artifact: {name}")
            continue
        fd, temporary_name = tempfile.mkstemp(prefix=name + ".", suffix=".partial", dir=directory)
        os.close(fd)
        temporary = Path(temporary_name)
        url = f"https://huggingface.co/{manifest['model']}/resolve/{manifest['revision']}/{name}"
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
                total = int(response.headers.get("Content-Length") or 0)
                pbar = (
                    tqdm(total=total or None, unit="B", unit_scale=True, unit_divisor=1024, desc=f"Downloading {name}")
                    if tqdm and os.environ.get("RLSE_STREAM_PROGRESS") != "1"
                    else None
                )
                downloaded = 0
                last_report = time.monotonic()
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if pbar is not None:
                        pbar.update(len(chunk))
                    elif time.monotonic() - last_report >= 0.25 or downloaded == total:
                        print(f"[DOWNLOAD] {name} {downloaded} {total}", flush=True)
                        last_report = time.monotonic()
                if pbar is not None:
                    pbar.close()
            print(f"[{idx}/{len(items)}] Verifying SHA-256 for {name}...", flush=True)
            if digest(temporary) != expected:
                raise ValueError(f"download SHA-256 mismatch: {name}")
            temporary.rename(path)
        finally:
            temporary.unlink(missing_ok=True)
    verify_model(directory, manifest)
    print(f"Verified {manifest['model']} at {manifest['revision']}", flush=True)


def augment_toolchain_path(env):
    path_entries = [p for p in env.get("PATH", "").split(os.pathsep) if p]
    for candidate in (Path.home() / ".cargo" / "bin", Path("/usr/local/cuda/bin")):
        if candidate.is_dir() and str(candidate) not in path_entries:
            path_entries.insert(0, str(candidate))
    env["PATH"] = os.pathsep.join(path_entries)


def run_validation(root, device, model, output, cache_dir=None, include_kernels=False, include_server=False, include_numerics=False):
    output.mkdir(parents=True, exist_ok=False)
    results = {"schema": 1, "status": "blocked", "requested_device": device,
               "actual_device": None, "host": platform.platform(), "commands": [],
               "sampling": {"temperature": 0, "top_p": 1, "reference_max_tokens": 32,
                            "integration_max_tokens": {"completion_chat_prefix_worker": 12, "eos": 32,
                                                       "cancellation": 512, "reservation": 4000, "injected_error": 128}},
               "limitations": ["No performance or driver OOM/hang claims; Qwen3-0.6B only."]}
    env = {k: v for k, v in os.environ.items() if not k.startswith("RLSE_")}
    augment_toolchain_path(env)
    env.update(RLSE_GPU_DEVICE=device, RLSE_GPU_MODEL_DIR=str(model))
    stream_mode = os.environ.get("RLSE_STREAM_PROGRESS") == "1"

    def command(argv, required=True):
        index = len(results["commands"])
        start = time.monotonic()
        label = " ".join(argv)
        print("Running:", label, flush=True)
        entry = {"argv": argv, "log": f"{index:02d}.log"}
        results["commands"].append(entry)
        log_path = output / entry["log"]
        stop_event = threading.Event()
        use_monitor = not hasattr(subprocess.run, "_mock_name")

        def monitor_log():
            offset = 0
            last_beat = time.monotonic()
            pbar = (
                tqdm(desc=f"[{index:02d}] {' '.join(argv[:4])}", unit="line", dynamic_ncols=True, leave=False)
                if (tqdm and not stream_mode)
                else None
            )
            try:
                while not stop_event.is_set():
                    if log_path.is_file():
                        with log_path.open("r", errors="replace") as reader:
                            reader.seek(offset)
                            chunk = reader.read()
                            offset = reader.tell()
                        if chunk:
                            for raw_line in chunk.splitlines():
                                s = raw_line.strip()
                                if not s:
                                    continue
                                if stream_mode:
                                    print(f"[LOG] {s}", flush=True)
                                elif pbar is not None:
                                    pbar.update(1)
                                    pbar.set_postfix_str(s[:65])
                            last_beat = time.monotonic()
                    now = time.monotonic()
                    if now - last_beat >= 1.0:
                        elapsed = now - start
                        if stream_mode:
                            print(f"[HEARTBEAT] elapsed={elapsed:.1f}s cmd={' '.join(argv[:4])}", flush=True)
                        elif pbar is not None:
                            pbar.set_postfix_str(f"running ({elapsed:.1f}s)")
                        last_beat = now
                    stop_event.wait(0.2)
            finally:
                if pbar is not None:
                    pbar.close()

        monitor = threading.Thread(target=monitor_log, daemon=True) if use_monitor else None
        if monitor is not None:
            monitor.start()
        with log_path.open("w") as log:
            try:
                completed = subprocess.run(argv, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                entry["exit_code"] = completed.returncode
            except FileNotFoundError:
                entry["exit_code"] = 127
                log.write(f"Executable unavailable: {argv[0]}\n")
            finally:
                stop_event.set()
                if monitor is not None:
                    monitor.join(timeout=1.0)
        entry["seconds"] = round(time.monotonic() - start, 3)
        content = log_path.read_text(errors="replace")
        if required and entry["exit_code"]:
            raise RuntimeError(f"command failed ({entry['exit_code']}): {label}; see {entry['log']}")
        return content

    try:
        manifest_path = root / MANIFEST
        source = json.loads(manifest_path.read_text()) if manifest_path.exists() else snapshot(root)
        verify_source(root, source)
        write_json(output / MANIFEST, source)
        results["lockfile_sha256"] = digest(root / "Cargo.lock")
        results["model"] = model_manifest(root)
        results["model_directory"] = str(model)
        compiler = command(["rustc", "--version", "--verbose"], required=False)
        toolchain = command(["cargo", "--version"], required=False)
        toolkit = command(["nvcc", "--version"], required=False)
        hardware = command(["nvidia-smi", "--query-gpu=name,uuid,compute_cap,memory.total,driver_version", "--format=csv"], required=False)
        command(["nvidia-smi", "--query-gpu=uuid,memory.free", "--format=csv"], required=False)
        command(["nvidia-smi"], required=False)  # Driver-supported CUDA version, distinct from nvcc toolkit.
        if device == "metal":
            hardware = command(["system_profiler", "SPHardwareDataType", "SPDisplaysDataType"], required=False)
        results["hardware_log"] = hardware
        # CPU checks remain useful even when GPU/model prerequisites are absent.
        if not shutil.which("cargo", path=env.get("PATH")) or not shutil.which("rustc", path=env.get("PATH")):
            results["cpu_status"] = "blocked"
            raise ValueError("Cargo toolchain unavailable; CPU and GPU checks blocked")
        results["build_environment"] = {k: env[k] for k in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CUDA_COMPUTE_CAP", "CUDA_ROOT", "CUDA_PATH", "CUDA_HOME", "CC", "CXX") if k in env}
        cache_key = hashlib.sha256((json.dumps(source, sort_keys=True) + toolchain + compiler + toolkit + hardware + device + json.dumps(results["build_environment"], sort_keys=True)).encode()).hexdigest()[:24]
        base_cache = Path(cache_dir).resolve() if cache_dir else (root / "target" / "gpu-validation")
        env["CARGO_TARGET_DIR"] = str(base_cache / cache_key)
        results["target_directory"] = env["CARGO_TARGET_DIR"]
        results["status"] = "failed"
        command(["cargo", "fmt", "--all", "--check"])
        command(["cargo", "test", "--locked", "--all-targets"])
        command([sys.executable, "-m", "unittest", "discover", "-s", "scripts/tests", "-v"])
        results["cpu_status"] = "passed"
        results["status"] = "blocked"
        verify_model(model, results["model"])
        results["artifacts_verified"] = True
        if device == "cuda" and (not shutil.which("nvcc", path=env.get("PATH")) or not shutil.which("nvidia-smi", path=env.get("PATH"))):
            raise ValueError("CUDA toolkit/driver tools unavailable")
        results["status"] = "failed"
        command(["cargo", "clippy", "--locked", "--features", device, "--all-targets", "--", "-D", "warnings"])
        command(["cargo", "test", "--locked", "--features", device, "--all-targets"])
        gpu_gates = ["test_gpu_inference", "test_pretrained_quality"]
        if include_numerics:
            gpu_gates.append("test_pretrained_numerics")
        for test in gpu_gates:
            log = command(["cargo", "test", "--locked", "--features", device, "--test", test, "--", "--ignored", "--nocapture"])
            marker = "Metal" if device == "metal" else "Cuda"
            actual = re.findall(r"ACTUAL_DEVICE=([^;\n]+)", log)
            if not actual or not all(x.startswith(marker + "(") for x in actual) or not re.search(r"test result: ok\. 1 passed; 0 failed; 0 ignored;", log):
                raise RuntimeError(f"{test}: missing actual {device} device evidence or executed-test result")
            results["actual_device"] = actual
        if include_kernels and device == "cuda":
            kernels_json = output / "cuda-kernels.json"
            command([sys.executable, "scripts/validate_cuda_kernels.py", "--output", str(kernels_json)])
            results["cuda_kernels"] = json.loads(kernels_json.read_text())
        if include_server:
            command(["cargo", "build", "--locked", "--features", device, "--bin", "rlse", "--example", "bench"])
            server_bin = Path(env["CARGO_TARGET_DIR"]) / "debug" / "rlse"
            bench_bin = Path(env["CARGO_TARGET_DIR"]) / "debug" / "examples" / "bench"
            server_log = output / "server-validation.log"
            command([
                sys.executable,
                "scripts/validate_gpu_server.py",
                "--server", str(server_bin),
                "--bench", str(bench_bin),
                "--device", device,
                "--model", str(model),
                "--output", str(server_log),
            ])
            results["server_validation_passed"] = True
        results["status"] = "passed"
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        results["reason"] = str(exc)
    finally:
        write_json(output / "results.json", results)
        bundle = Path(str(output) + ".tar.gz")
        with tarfile.open(bundle, "w:gz") as archive:
            archive.add(output, arcname=output.name)
        write_json(Path(str(bundle) + ".json"), {"sha256": digest(bundle), "status": results["status"]})
        print(f"{results['status'].upper()}: {results.get('reason', 'explicit GPU tests passed')}; evidence: {bundle}")
    return 0 if results["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--embed-notebook", action="store_true")
    sub.add_parser("clear-notebook")
    p = sub.add_parser("unpack")
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--sha256", required=True)
    p.add_argument("--destination", required=True, type=Path)
    p = sub.add_parser("download-model")
    p.add_argument("--model-dir", type=Path, default=Path("models/Qwen3-0.6B"))
    p = sub.add_parser("run")
    p.add_argument("--device", choices=("cuda", "metal"), required=True)
    p.add_argument("--model-dir", type=Path, default=Path("models/Qwen3-0.6B"))
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--include-kernels", action="store_true")
    p.add_argument("--include-server", action="store_true")
    p.add_argument("--include-numerics", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "pack":
        pack(root, args.output.resolve(), embed_notebook=args.embed_notebook)
    elif args.action == "clear-notebook":
        set_notebook_payload(root / NOTEBOOK_REL)
    elif args.action == "unpack":
        unpack(args.archive.resolve(), args.destination.resolve(), args.sha256)
    elif args.action == "download-model":
        download_model(root, (root / args.model_dir).resolve())
    else:
        return run_validation(
            root,
            args.device,
            (root / args.model_dir).resolve(),
            args.output.resolve(),
            cache_dir=args.cache_dir.resolve() if args.cache_dir else None,
            include_kernels=args.include_kernels,
            include_server=args.include_server,
            include_numerics=args.include_numerics,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
