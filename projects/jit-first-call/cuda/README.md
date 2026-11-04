# Optional: Triton CUDA JIT

只有想观察真实 NVIDIA GPU kernel 编译时才需要本演示。默认的 CPU JIT Demo 没有这些依赖。

在带 CUDA 驱动的 Linux x86_64 + NVIDIA GPU 环境中，先按照 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装与你的 CUDA 环境匹配的 PyTorch，再安装 Triton：

```bash
python -m pip install triton
python projects/jit-first-call/cuda/run.py
```

脚本会运行一个带 `BLOCK: tl.constexpr` 的向量加法 kernel，对比 `BLOCK=256` 和 `BLOCK=512` 的首次调用与重复调用，并用 PyTorch 验证结果。

**计时边界**：包括 Python 调用、Kernel Launch 和 `torch.cuda.synchronize()`；首次调用还可能包含编译或已有磁盘缓存加载。它不是纯 GPU kernel 时间，也不是 Triton 与 CUDA C++ 的性能比较。

此路径**未在当前 CPU-only 测试环境中验证 GPU 实际执行**。首次贡献支持该平台的测试数据时，请注明驱动、GPU、PyTorch、Triton 版本。
