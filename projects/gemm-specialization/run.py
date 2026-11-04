"""Test JIT-compiled GEMM tile variants on CPU, without a GPU.

    python projects/gemm-specialization/run.py
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


@numba.njit(inline="always")
def blocked_gemm_impl(a, b, block):
    """Common algorithm, inlined into independently JIT-compiled tile variants."""
    m, k = a.shape
    n = b.shape[1]
    out = np.zeros((m, n), dtype=np.float32)
    for ii in range(0, m, block):
        for kk in range(0, k, block):
            for jj in range(0, n, block):
                for i in range(ii, min(ii + block, m)):
                    for p in range(kk, min(kk + block, k)):
                        av = a[i, p]
                        for j in range(jj, min(jj + block, n)):
                            out[i, j] += av * b[p, j]
    return out


@numba.njit(cache=True)
def gemm_tile_8(a, b):
    return blocked_gemm_impl(a, b, 8)


@numba.njit(cache=True)
def gemm_tile_16(a, b):
    return blocked_gemm_impl(a, b, 16)


@numba.njit(cache=True)
def gemm_tile_32(a, b):
    return blocked_gemm_impl(a, b, 32)


TILE_KERNELS = {8: gemm_tile_8, 16: gemm_tile_16, 32: gemm_tile_32}


def input_matrices(shape):
    m, n, k = shape
    rng = np.random.default_rng(2026 + m + n + k)
    a = rng.uniform(-0.5, 0.5, size=(m, k)).astype(np.float32)
    b = rng.uniform(-0.5, 0.5, size=(k, n)).astype(np.float32)
    return a, b


def check(a, b, block):
    got = TILE_KERNELS[block](a, b)
    expected = a @ b
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)


def verify():
    for shape in [(1, 1, 1), (8, 5, 9), (17, 19, 13), (32, 31, 25)]:
        a, b = input_matrices(shape)
        for block in (8, 16, 32):
            check(a, b, block)
    print("PASS: all CPU tile variants agree with the NumPy matrix-product reference")


def measure(fn, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def experiment(repeats, shapes, blocks):
    rows = []
    for shape in shapes:
        a, b = input_matrices(shape)
        candidates = []
        for block in blocks:
            t0 = time.perf_counter()
            check(a, b, block)  # first use includes compiler dispatch / cache load
            setup_ms = (time.perf_counter() - t0) * 1000.0
            warm_ms = measure(lambda: TILE_KERNELS[block](a, b), repeats)
            candidates.append({"block": block, "first_call_ms": setup_ms, "warm_median_ms": warm_ms})
        best = min(candidates, key=lambda x: x["warm_median_ms"])
        rows.append({"shape_mnk": list(shape), "candidates": candidates, "best_block": best["block"]})
        print(f"M,N,K={shape}: " + ", ".join(f"block={x['block']} {x['warm_median_ms']:.3f}ms" for x in candidates)
              + f"; fastest observed: block={best['block']}")

    print("These variants are all compiled with Numba JIT; this is a tile/specialization experiment,")
    print("not an AOT-vs-JIT or CPU-vs-GPU benchmark. Fastest tile may vary across machines.")
    return {
        "experiment": "gemm-specialization",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "processor": platform.processor() or platform.machine(),
                        "numpy": np.__version__, "numba": numba.__version__},
        "parameters": {"repeats": repeats, "blocks": list(blocks)},
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.verify:
        verify()
        return
    print("GEMM specialization demo (CPU / Numba)")
    results = experiment(args.repeats, [(32, 32, 32), (64, 48, 64), (96, 128, 64)], (8, 16, 32))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {args.json}")


if __name__ == "__main__":
    main()
