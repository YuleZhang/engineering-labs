# Instructions for Agents editing Engineering Labs

Goal: maintain tiny, independently reproducible demos. The top-level `python run.py` must remain easy to use.

- Prefer a standalone `run.py` with a `--verify` mode and generated test data.
- Keep the default path CPU-portable when the underlying concept allows it; put GPU/device backends in an optional subdirectory.
- Avoid hidden package installation, background services, private endpoints, credential use, or unnecessarily large dependencies.
- Clearly distinguish concept demos, simulations, reference implementations, and actual hardware performance tests.
- Verify correctness before measuring. Do not assert that specialization or JIT always wins.
- Never modify an existing Demo merely to add global framework abstractions.
- Treat measured JSON under `results/` as machine-specific snapshots, not guaranteed targets.
- New code must have an independently executable path and a corresponding README, plus metadata in `demo.yaml` and `catalog.yaml`.
- Do not present unexecuted code or an unavailable GPU backend as validated.
