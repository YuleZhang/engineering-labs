# Local measurements

`example-linux-x86_64.json` 是一次真实 CPU 运行的样本数据，包含 Python / NumPy / Numba 版本和计时参数。

这些数字只用于展示输出格式，不作为性能承诺，不在 CI 中检查固定延迟阈值。不同硬件、系统负载和磁盘 JIT 缓存状态都会影响结果。

重新运行：

```bash
python run.py --json projects/jit-first-call/results/generated/your-run.json
```
