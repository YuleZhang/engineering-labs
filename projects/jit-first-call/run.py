"""See CPU JIT specialization and first-call cost with one command.

    python projects/jit-first-call/run.py
"""

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import numba
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc.name}. Install once with: "
        "python -m pip install -r requirements.txt"
    ) from exc


def calculate(values):
    """Pure Python reference implementation."""
    total = 0.0
    for value in values:
        v = float(value)
        total += (v * v + 0.25) / (abs(v) + 1.0)
    return total


@numba.njit(cache=True)
def jit_calculate(values):
    total = 0.0
    for value in values:
        v = float(value)
        total += (v * v + 0.25) / (abs(v) + 1.0)
    return total


def timed(fn, values):
    start = time.perf_counter()
    result = fn(values)
    return (time.perf_counter() - start) * 1000.0, result


def verify():
    samples = [
        np.array([], dtype=np.float64),
        np.array([-2.0, -0.5, 0.0, 1.25, 2.0], dtype=np.float64),
        np.linspace(-2.0, 2.0, 131, dtype=np.float32),
        np.linspace(-2.0, 2.0, 247, dtype=np.float64),
    ]
    for data in samples:
        expected = calculate(data)
        actual = jit_calculate(data)
        np.testing.assert_allclose(actual, expected, rtol=1e-6 if data.dtype == np.float32 else 1e-12, atol=1e-9)
    print("PASS: CPU JIT matches the Python reference for empty, float32 and float64 inputs")


def experiment(n, repeats):
    x = np.linspace(-2.0, 2.0, n, dtype=np.float64)
    py_ms, expected = timed(calculate, x)

    # The first invocation can include native compilation OR loading a prior disk cache.
    before = len(jit_calculate.signatures)
    first_ms, actual = timed(jit_calculate, x)
    after_first = len(jit_calculate.signatures)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)

    warm_ms = [timed(jit_calculate, x)[0] for _ in range(repeats)]

    # Array length is not part of the default Numba array type signature.
    larger = np.linspace(-2.0, 2.0, n + 7, dtype=np.float64)
    len_change_ms, _ = timed(jit_calculate, larger)
    after_size = len(jit_calculate.signatures)

    # Array element dtype *does* change the Numba specialization signature.
    fp32 = np.linspace(-2.0, 2.0, n, dtype=np.float32)
    dtype_first_ms, dtype_result = timed(jit_calculate, fp32)
    np.testing.assert_allclose(dtype_result, calculate(fp32), rtol=1e-6, atol=1e-9)
    after_dtype = len(jit_calculate.signatures)
    if not (after_first >= before and after_size == after_first and after_dtype == after_size + 1):
        raise AssertionError("unexpected Numba specialization count")

    result = {
        "experiment": "jit-first-call",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "numpy": np.__version__,
            "numba": numba.__version__,
        },
        "parameters": {"n": n, "repeats": repeats, "disk_cache": True},
        "timings_ms": {
            "python_one_call": py_ms,
            "numba_first_call_in_process": first_ms,
            "numba_warm_median": statistics.median(warm_ms),
            "same_dtype_new_length": len_change_ms,
            "new_dtype_first_call": dtype_first_ms,
        },
        "compiled_signatures": {
            "before": before,
            "after_first": after_first,
            "after_length_change": after_size,
            "after_dtype_change": after_dtype,
        },
    }
    print("\nJIT first-call demo (CPU / Numba)")
    print(f"Python reference:               {py_ms:9.3f} ms")
    print(f"Numba first invocation:         {first_ms:9.3f} ms")
    print(f"Numba warm median ({repeats} runs): {statistics.median(warm_ms):9.3f} ms")
    print(f"Same dtype, new array length:   {len_change_ms:9.3f} ms")
    print(f"New dtype (float32):            {dtype_first_ms:9.3f} ms")
    print(f"Numba overloads: {before} → {after_first} → {after_size} → {after_dtype}")
    print("Note: first invocation may compile or load an existing on-disk cache.")
    print("      Times include Python/dispatcher overhead, not only compilation.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="check correctness only")
    parser.add_argument("--size", type=int, default=50_000)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--json", type=Path, help="save observed measurements to a JSON file")
    args = parser.parse_args()
    if args.size < 1 or args.repeats < 1:
        parser.error("--size and --repeats must be positive")
    if args.verify:
        verify()
        return
    results = experiment(args.size, args.repeats)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {args.json}")


if __name__ == "__main__":
    main()
