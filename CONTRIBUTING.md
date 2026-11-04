# Adding a Demo

The simplest working example wins. A reader should not need to learn the repo's infrastructure before learning the concept.

1. Create `projects/<short-topic>/` and copy the [minimal template](templates/demo/).
2. Make `python projects/<short-topic>/run.py` work with ordinary generated sample data, or document the exact non-Python command when genuinely necessary.
3. Implement `--verify` with a real correctness assertion. Separate benchmarks from correctness tests.
4. Describe the minimum dependencies and environment. Prefer no credentials, no services, and no opaque downloads. If specialized hardware is essential, make it an **optional path** where feasible; do not claim a simulator reproduces physical performance.
5. Add a short `README.md`: question, how to run, what to change, expected observation, and what the example *does not* prove.
6. Add one `demo.yaml` entry and update the root catalog. Avoid creating abstractions for a single Demo.
7. When publishing measured results, record Python/package versions, device/hardware when known, input sizes, warmup, repetitions, and the measurement boundary. Never fabricate measurements or enforce fixed performance thresholds in CI.

Prefer the Python standard library and only add packages that directly unlock the demonstration. Keep install instructions at most a few lines.

Run the current smoke tests before opening a PR:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```
