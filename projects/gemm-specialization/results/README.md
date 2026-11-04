# Local measurements

`example-linux-x86_64.json` 保存了一次真实 CPU tile sweep 的输出。它不代表 Triton、CUDA GPU、AOT 编译或工业级 BLAS 性能。

重新运行：

```bash
python run.py gemm --json projects/gemm-specialization/results/generated/your-run.json
```

默认每个 tile 测量多次 warm median；修改 `--repeats` 观察波动。数值正确性与 BLAS 参考单独验证。
