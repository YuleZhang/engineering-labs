# 为什么 GEMM 要按形状选配置？

同样是矩阵乘 `C[M,N] = A[M,K] × B[K,N]`，一个固定 tile 能覆盖不同 shape，却未必在所有 shape 上最快。我们可以在一台普通电脑上验证这一点。

## 运行

```bash
python -m pip install -r requirements.txt
python projects/gemm-specialization/run.py
```

或者 `python run.py gemm`。不需要 GPU。

验证结果与 NumPy 矩阵乘一致：

```bash
python projects/gemm-specialization/run.py --verify
```

保存本机结果：

```bash
python projects/gemm-specialization/run.py --json projects/gemm-specialization/results/generated/my-run.json
```

## 实验怎么做？

- 用 Numba JIT 编译三个共享核心算法、但 `block` 分别为 8、16、32 的函数（每个变体分别有 native code）。
- 用固定随机种子生成 float32 矩阵，和 `A @ B` 比较数值结果；每次运行都检查 correctness。
- 对 `(M,N,K)=(32,32,32)`, `(64,48,64)`, `(96,128,64)` 测量不同 block 的 warm median，打印实测最快配置。

**观察重点**：最佳配置不一定相同，且额外的编译变体有首次使用成本。你可以改变 shape、增加一个 block，或在自己的设备上比较。

## 特别注意

这里的三种实现**全部使用 Numba JIT**，所以对比的是 tile / 特化策略，而不是纯 AOT 与 JIT。更不是 CPU 和 GPU 的性能对比。`numpy.matmul` 只作为正确性参考，不作为可比的通用手写 kernel 性能基线。

对少量小矩阵来说，测量中的 Python 调用和计时噪声可能占很大比例；不要推广某个 block 在任何 CPU 上都更快。结果文件保留测试环境与样本个数。

测量样本说明：[results/README.md](results/README.md)。
