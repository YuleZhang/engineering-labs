# Engineering Labs

**Small, runnable demos for AI, compilers, and systems.** 用最少的代码复现一个工程现象，再理解它为什么发生。

## 快速上手

安装 Python 3.11+，然后：

```bash
git clone https://github.com/YuleZhang/engineering-labs.git
cd engineering-labs
python -m pip install -r requirements.txt
python run.py
```

第一次运行会观察 CPU JIT 的**首次调用**、**缓存后的调用**以及**类型变化引发的新特化**。无需 CUDA、Docker、模型权重或账号。

## Demos

| Demo | 运行命令 | 你会看到什么 | 依赖 |
| --- | --- | --- | --- |
| [JIT 第一次为什么慢？](projects/jit-first-call/README.md) | `python run.py` | Numba 首次调用与 warm 调用的差异，dtype 变化引发的新编译 | CPU · Windows/Linux/macOS |
| [为什么 GEMM 要按形状选配置？](projects/gemm-specialization/README.md) | `python run.py gemm` | 同一算法的不同 tile 变体，在不同 M×N×K 上性能不同 | CPU · Windows/Linux/macOS |

验证正确性：

```bash
python run.py --verify
python run.py gemm --verify
```

有 NVIDIA GPU？[可选 Triton 版本](projects/jit-first-call/cuda/README.md)演示 GPU kernel 首次编译和 `constexpr` 特化。它**不影响**以上两个 CPU Demo。

## 添加 Demo 的原则

1. **启动简单**：优先 `python path/to/run.py`，不要求安装服务、使用容器、申请密钥或下载大型模型。复杂环境只作为可选路径。
2. **默认可复现**：尽可能让 Windows / Linux / macOS 普通 CPU 用户运行；如必须依赖 GPU 或设备，明确限制，并给出有意义的 portable reference（没有就如实说明）。
3. **能看懂也能验证**：在独立 `projects/<demo>/README.md` 中说明“观察什么、怎么运行、预期什么”；包含 `--verify` 或同等正确性检查。禁止用硬编码的性能数值当测试断言。
4. **结论有边界**：性能实验披露测试环境、计时方法、预热与同步条件；不要把算法/调度收益归因于 JIT 编译时机。
5. **依赖尽可能少**：一个 Demo 独立可跑，不把其他 Demo 的工具链当作前置条件。示例数据自己生成，不能提交私有资产。

贡献说明见 [CONTRIBUTING.md](CONTRIBUTING.md)，新项目从 [templates/demo](templates/demo) 复制即可。

## Related

- 学习文章可以单独分享；实验项目提供与文章相互链接的可执行证据。
- 这里的 CPU GEMM 是教学实现，**不是**高性能 BLAS 的替代品。
- 当前首批实测环境与测量边界见各项目 `results/`，不保证不同机器获得相同延迟。

License: [MIT](LICENSE).
