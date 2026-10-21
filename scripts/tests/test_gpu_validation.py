"""Transport integrity and fail-closed evidence regressions; no GPU is simulated as verified."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("gpu_validation", Path(__file__).parents[1] / "gpu_validation.py")
gpu = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gpu)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "repo"
        self.root.mkdir()
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self.write("src/lib.rs", "// committed\n")
        self.write("Cargo.lock", "# fixture\n")
        self.write(".gitignore", "target/\nmodels/\n.env\n")
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                        "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"], cwd=self.root, check=True)
        self.write("tests/fixtures/generation/qwen3-reference.json", json.dumps({
            "model": "fixture", "revision": "pinned", "hashes": {"model.safetensors": "0" * 64}}))

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_pack_current_tree_and_reject_tampering(self):
        self.write("src/lib.rs", "// modified\n")
        self.write("src/new.rs", "// untracked\n")
        for name in (".env", "scripts/credentials.json", "models/model.safetensors", "target/debug/a.rs"):
            self.write(name, "excluded")
        a, b = self.base / "a.tar.gz", self.base / "b.tar.gz"
        gpu.pack(self.root, a)
        gpu.pack(self.root, b)
        self.assertEqual(gpu.digest(a), gpu.digest(b))
        destination = self.base / "unpacked"
        gpu.unpack(a, destination, gpu.digest(a))
        manifest = json.loads((destination / gpu.MANIFEST).read_text())
        self.assertTrue(manifest["dirty"])
        self.assertEqual((destination / "src/new.rs").read_text(), "// untracked\n")
        self.assertNotIn("scripts/credentials.json", manifest["files"])
        (destination / "src/lib.rs").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            gpu.verify_source(destination, manifest)

    def test_unpack_rejects_hash_and_traversal(self):
        archive = self.base / "malicious.tar.gz"
        with tarfile.open(archive, "w:gz") as out:
            member = tarfile.TarInfo("../escape.rs")
            member.size = 1
            out.addfile(member, io.BytesIO(b"x"))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            gpu.unpack(archive, self.base / "wrong", "0" * 64)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            gpu.unpack(archive, self.base / "traversal", gpu.digest(archive))
        self.assertFalse((self.base / "escape.rs").exists())

    def test_rejects_symlinks_and_extra_source(self):
        (self.root / "src/link.rs").symlink_to(self.base / "outside")
        (self.base / "outside").write_text("outside")
        with self.assertRaisesRegex(ValueError, "symlink"):
            gpu.snapshot(self.root)
        (self.root / "src/link.rs").unlink()
        manifest = gpu.snapshot(self.root)
        self.write("src/unexpected.rs", "extra")
        with self.assertRaisesRegex(ValueError, "unmanifested"):
            gpu.verify_source(self.root, manifest)

    def test_missing_toolchain_still_exports_blocked_evidence(self):
        output = self.base / "evidence"
        with patch.object(gpu.shutil, "which", return_value=None):
            self.assertEqual(gpu.run_validation(self.root, "cuda", self.root / "models", output), 1)
        result = json.loads((output / "results.json").read_text())
        self.assertEqual(result["status"], "blocked")
        self.assertIsNone(result["actual_device"])
        self.assertTrue(Path(str(output) + ".tar.gz").is_file())

    def test_zero_tests_or_cpu_device_cannot_pass(self):
        for log in ("test result: ok. 0 passed; 0 failed; 0 ignored;",
                    "ACTUAL_DEVICE=Cpu;\ntest result: ok. 1 passed; 0 failed; 0 ignored;"):
            output = self.base / ("result" + str(len(log)))
            def fake_run(argv, **kwargs):
                if "stdout" in kwargs:
                    kwargs["stdout"].write(log)
                    return subprocess.CompletedProcess(argv, 0)
                return subprocess.run(argv, **kwargs)
            # Capture source before substituting command execution.
            manifest = gpu.snapshot(self.root)
            with patch.object(gpu, "snapshot", return_value=manifest), \
                 patch.object(gpu, "verify_model"), \
                 patch.object(gpu.shutil, "which", return_value="/fake/tool"), \
                 patch.object(gpu.subprocess, "run", side_effect=fake_run):
                self.assertEqual(gpu.run_validation(self.root, "cuda", self.root / "models", output), 1)
            result = json.loads((output / "results.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertIn("missing actual cuda", result["reason"])
            self.assertIsNone(result["actual_device"])

    def test_numeric_gpu_gate_is_opt_in(self):
        manifest = gpu.snapshot(self.root)
        for include_numerics in (False, True):
            def fake_run(argv, **kwargs):
                kwargs["stdout"].write(
                    "ACTUAL_DEVICE=Cuda(fixture);\n"
                    "test result: ok. 1 passed; 0 failed; 0 ignored;\n"
                )
                return subprocess.CompletedProcess(argv, 0)

            output = self.base / ("numeric-gate-" + str(include_numerics))
            with patch.object(gpu, "snapshot", return_value=manifest), \
                 patch.object(gpu, "verify_model"), \
                 patch.object(gpu.shutil, "which", return_value="/fake/tool"), \
                 patch.object(gpu.subprocess, "run", side_effect=fake_run):
                self.assertEqual(gpu.run_validation(
                    self.root, "cuda", self.root / "models", output,
                    include_numerics=include_numerics), 0)
            result = json.loads((output / "results.json").read_text())
            gates = [entry["argv"][entry["argv"].index("--test") + 1]
                     for entry in result["commands"] if "--test" in entry["argv"]]
            expected = ["test_gpu_inference", "test_pretrained_quality"]
            if include_numerics:
                expected.append("test_pretrained_numerics")
            self.assertEqual(gates, expected)

    def test_model_hash_mismatch_is_rejected(self):
        self.write("models/model.safetensors", "wrong artifact")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            gpu.verify_model(self.root / "models", gpu.model_manifest(self.root))

    def test_embed_and_clear_notebook_preserves_clean_archive(self):
        self.write("scripts/gpu_validation.py", "# harness\n")
        self.write(gpu.NOTEBOOK_REL, json.dumps({
            "cells": [{"cell_type": "code", "source": [
                'EMBEDDED_ARCHIVE_B64 = ""\n',
                'EMBEDDED_METADATA_B64 = ""\n',
                'EMBEDDED_SCRIPT_B64 = ""\n',
            ]}]
        }, indent=1) + "\n")
        clean_nb = (self.root / gpu.NOTEBOOK_REL).read_text()
        out_embed = self.base / "embed.tar.gz"
        gpu.pack(self.root, out_embed, embed_notebook=True)
        embedded_nb = (self.root / gpu.NOTEBOOK_REL).read_text()
        self.assertNotEqual(clean_nb, embedded_nb)
        # Re-packing without embed_notebook strips the embedded payload first so archives stay identical.
        out_clean = self.base / "clean.tar.gz"
        gpu.pack(self.root, out_clean, embed_notebook=False)
        self.assertEqual(gpu.digest(out_embed), gpu.digest(out_clean))
        self.assertEqual((self.root / gpu.NOTEBOOK_REL).read_text(), clean_nb)


if __name__ == "__main__":
    unittest.main()
