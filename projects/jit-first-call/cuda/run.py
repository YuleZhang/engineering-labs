"""Optional CUDA/TRITON example. The CPU demo does not require these packages.

    python projects/jit-first-call/cuda/run.py
"""

import time
import statistics

try:
    import torch
    import triton
    import triton.language as tl
except ImportError as exc:
    raise SystemExit(
        f"Missing optional CUDA package ({exc.name}). "
        "Install torch and triton using the instructions in the demo README."
    ) from exc


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n
    x = tl.load(x_ptr + offsets, mask=valid, other=0)
    y = tl.load(y_ptr + offsets, mask=valid, other=0)
    tl.store(out_ptr + offsets, x + y, mask=valid)


def run_once(x, y, out, block):
    grid = (triton.cdiv(x.numel(), block),)
    add_kernel[grid](x, y, out, x.numel(), BLOCK=block)
    torch.cuda.synchronize()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is unavailable. Run ../run.py for the CPU JIT demo.")
    n = 100_003
    x = torch.linspace(-1.0, 1.0, n, device="cuda", dtype=torch.float32)
    y = torch.ones_like(x)
    out = torch.empty_like(x)
    for block in (256, 512):
        start = time.perf_counter()
        run_once(x, y, out, block)
        first_ms = (time.perf_counter() - start) * 1000.0
        torch.testing.assert_close(out, x + y)
        warm = []
        for _ in range(15):
            start = time.perf_counter()
            run_once(x, y, out, block)
            warm.append((time.perf_counter() - start) * 1000.0)
        print(f"BLOCK={block}: first invocation={first_ms:.3f} ms, "
              f"warm median (includes launch/sync)={statistics.median(warm):.3f} ms")
    print("PASS: Triton vector addition matches torch. First invocation may use a disk cache.")


if __name__ == "__main__":
    main()
