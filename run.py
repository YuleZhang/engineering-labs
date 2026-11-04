"""One small entry point for all demos: python run.py [jit|gemm]."""

import subprocess
import sys
from pathlib import Path

DEMOS = {
    "jit": "projects/jit-first-call/run.py",
    "gemm": "projects/gemm-specialization/run.py",
}


def main():
    args = sys.argv[1:]
    if args and args[0] in ("-h", "--help", "list"):
        print("Usage: python run.py [jit|gemm] [demo options]")
        print("  jit   - observe the first-call cost of CPU JIT (default)")
        print("  gemm  - compare JIT-compiled GEMM tile variants on CPU")
        return
    name = "jit"
    if args and args[0] in DEMOS:
        name = args.pop(0)
    elif args and not args[0].startswith("-"):
        raise SystemExit(f"Unknown demo {args[0]!r}. Try: python run.py list")
    script = Path(__file__).resolve().parent / DEMOS[name]
    raise SystemExit(subprocess.call([sys.executable, str(script), *args]))


if __name__ == "__main__":
    main()
