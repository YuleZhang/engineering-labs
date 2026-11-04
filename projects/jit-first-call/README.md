# JIT 第一次为什么慢？

你写了一段 Python 循环，第一次运行时 Numba 需要生成或加载机器码，后面用同一种类型再调用就快得多。那么：哪些参数变化会导致重新生成代码？

## 运行

在仓库根目录安装依赖一次：

```bash
python -m pip install -r requirements.txt
python projects/jit-first-call/run.py
```

也可以直接 `python run.py`，不需要 GPU。

查看正确性检查：

```bash
python projects/jit-first-call/run.py --verify
```

保存你机器上的测量数据：

```bash
python projects/jit-first-call/run.py --json projects/jit-first-call/results/generated/my-run.json
```

## 观察什么？

脚本会对同一个计算公式分别使用原生 Python 和 `@numba.njit(cache=True)`，打印首次调用、重复调用（median）、相同 dtype 不同数组长度，以及换成 float32 后第一次调用的耗时，同时展示已编译 overload 数量。

- 第一次调用不仅可能有编译成本，还可能有磁盘缓存加载与运行时初始化。
- 再次调用同一种数组元素类型通常复用已编译 overload；**数组长度不一定造成新编译**。Numba 默认 NumPy 数组类型签名主要取决于 dtype、维度和布局，并不会逐个固定 shape。
- 从 float64 换到 float32，会为不同输入类型创建新的特化。
- 开启 `cache=True` 后，再开一个 Python 进程也有机会从磁盘缓存恢复编译产物；它不意味着启动成本为零。

可以改 `--size 100000 --repeats 21` 观察变化。反复执行两次命令，比较第一次进程启动和第二次进程启动的首调用耗时。

## 这段实验不能证明什么？

不能通过单一 workload 推断所有 JIT 都能加速，也不能把首次调用总耗时当作纯 LLVM 编译耗时。Numba 的类型特化行为也不等于 Triton 的 `constexpr` 或整个 PyTorch `torch.compile` 图级 Guard 机制。

### 想看 GPU kernel 的 JIT？

可选：[Triton CUDA 版本](cuda/README.md)。CPU Demo 无需安装 PyTorch / Triton。

测量样本说明：[results/README.md](results/README.md)。
