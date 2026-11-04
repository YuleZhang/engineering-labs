"""Smoke tests exercise the same CLI that readers use."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DemoSmokeTests(unittest.TestCase):
    def check_cli(self, *args):
        done = subprocess.run(
            [sys.executable, "run.py", *args], cwd=ROOT,
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(done.returncode, 0, msg=done.stdout + done.stderr)
        self.assertIn("PASS", done.stdout)

    def test_jit_correctness(self):
        self.check_cli("jit", "--verify")

    def test_gemm_correctness(self):
        self.check_cli("gemm", "--verify")

    def test_default_entry(self):
        done = subprocess.run(
            [sys.executable, "run.py", "--size", "32", "--repeats", "2"],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(done.returncode, 0, msg=done.stdout + done.stderr)
        self.assertIn("Numba overloads:", done.stdout)

    def test_gemm_entry(self):
        done = subprocess.run(
            [sys.executable, "run.py", "gemm", "--repeats", "2"],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(done.returncode, 0, msg=done.stdout + done.stderr)
        self.assertIn("fastest observed:", done.stdout)


if __name__ == "__main__":
    unittest.main()
