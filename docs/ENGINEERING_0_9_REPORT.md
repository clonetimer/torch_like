# torch_like 0.9.0 工程报告：最小 DDP 执行闭环

## 1. 交付结论

基于已交付的 0.8.0 源码，完成共享 Trainer 的最小 DDP 执行、精确验证分片、全局流式指标合并、rank-0 协调写盘及 rank-local checkpoint/resume。不是新增第二套 Trainer，也不是重写 PyTorch 张量/自动求导系统。

最终冻结代码下 **574 个唯一测试通过，0 失败，0 跳过**。采用 31 个独立 pytest 进程分组；不是一个 pytest 长进程一次跑完，也没有把开发阶段重跑次数叠加到总数中。其中 19 个测试分别通过真正的 torchrun 启动两个 CPU/Gloo worker，另有 11 个分布式单元测试和 4 个 checkpoint 兼容/修复测试。

**已实测范围：Linux / Python 3.13.5 / PyTorch 2.10.0+cpu / 本机双进程 Gloo。CUDA/NCCL 路径已实现但未实测；多机、Windows、其他依赖版本亦未验证。** 本轮不是 GPU 性能验收、165 个模型的全面 DDP 资格认证，也不构成业务精度基准。

基线文件：`torch_like-0.8.0-streaming-distributed-contracts-source.zip`。
基线 SHA-256：`dfd5a3cb4cf04bbaf39b54f77f27a6d7e4670869ebaef944bccc5bbb8d8cf927`。

## 2. 源码改动及分层

| 文件/位置 | 改动 |
|---|---|
| `torch_like/training/distributed.py` | 增加 DDPConfig、DistributedSession、DistributedEvalSampler，保留旧 context/sampler API |
| `torch_like/training/_ddp.py` | 私有 loss-forward 包装、跨 rank 检查、指标状态通信、schema-v3 协调 checkpoint |
| `torch_like/training/trainer.py` | 在原 epoch/optimizer 生命周期内接入 DDP；保留原生模型对象；事件主 rank 归属 |
| `torch_like/training/checkpoint.py` | v3 读取与 rank 状态恢复，兼容 v1/v2；新增 Python/NumPy/validation-loader RNG 状态 |
| `tests/test_ddp_runtime.py`、`tests/_workers/ddp_runtime_worker.py` | 真实双进程场景和分片/session 单元测试 |
| `tests/test_checkpoint_compatibility.py` | 历史 checkpoint、随机状态及 best restore 回归 |
| `tools/validate_release.py` | 全部 test node ID 唯一分组、源码冻结摘要、超时清理、JUnit/日志/总账 |
| `examples/train_ddp_tabular.py` | 可直接执行的原生 FTTransformer 双进程训练、保存和恢复示例 |
| `docs/DDP_RUNTIME.md` | 安装、使用、模型/数据合同、异常与恢复边界 |
| `.github/workflows/ddp.yml` | 新增 Linux CPU/Gloo CI 配置；本轮没有实际触发 GitHub Actions |

三个新公开符号只放在 `torch_like.training`，没有增加新顶层别名。公共 API 基线从 901 变为 **904**，模型目录仍为 **165**。模型的数值实现未在本轮批量改写。sklearn 仍返回原生 estimator，未塞入 DDP Trainer；AutoML trial 仍默认独立解释器、单进程训练。

## 3. 关键设计与正确性边界

### 3.1 DDP forward 覆盖模型的真实 loss

仅包装 `DDP(model)` 后再调用 `model.loss(...)` 会绕开预期的 DDP forward 执行入口。本次私有 `_LossModule` 注册原生模型，并把 runtime.loss、自定义 loss_fn 或 step_fn 放入它的 forward，再由 DDP 包装该模块。对用户仍保持 `trainer.model is model`。

训练步、optimizer、scheduler、early stopping 只有公共 Trainer 一套；DDP 辅助层不复制完整训练循环。未使用参数可显式设置 `DDPConfig(find_unused_parameters=True)`，不自动推断所有模型的梯度图。

### 3.2 训练分片与验证分片分开

训练使用官方 DistributedSampler，每个 epoch 调用 set_epoch。要求所有 rank 使用同长度、同顺序、同内容的数据集，训练步数相等且每步 sample_count 相同。训练 sampler 的补齐/丢尾仍遵循 PyTorch 语义，训练 loss 描述实际采样流，不冒充原始数据集无重复平均。

验证使用新 DistributedEvalSampler：7 个样本、2 rank 分为 4+3；1 个样本分为 1+0。无补齐、无丢弃，允许空 rank；整个验证集为空则拒绝。验证前广播一次 buffer，随后本地原生模型评估，最后合并全局 loss 和指标状态；模型内部不能自行执行跨 rank collective。

F1/R² 等先合并充分统计量，再计算最终指标。DDP 拒绝 buffered FunctionalMetric、任意函数式 metrics 和历史 metric_fns，单进程仍保留这些接口。自定义 MergeableMetric 的数学正确性由实现方负责。

目前控制信息、CPU 指标 state、rank RNG/buffer 通过 all_gather_object 传输；这是可信作业内的正确性优先实现，不是大规模高性能通信优化。它有 pickle 信任与通信/内存开销边界，不面向不可信 worker。

### 3.3 单文件协调 checkpoint 与精确恢复

单进程仍写 schema v2，读取兼容 v1/v2；DDP 使用 schema v3，由 rank 0 原子写入一个协调文件。v3 包含公共模型/optimizer/scheduler/history、run_id、恢复签名以及每个 rank 的随机状态、buffer、scaler、sampler epoch。

随机状态包括 Torch CPU/当前 CUDA、Python random、NumPy legacy 全局 RNG、train/validation DataLoader generator。NumPy 状态编码为基本类型/Tensor，可由 weights_only=True 读取。不保证独立 np.random.Generator、用户自建随机源或有内部状态的 adapter/dataset 可精确恢复。

恢复检查 world_size/rank/local_rank、全部 rank 状态、关键配置/采样签名、各 rank 读取的文件摘要；错位或缺失明确失败。签名不包含数据内容指纹，同长度但不同数据仍可能通过结构检查，数据版本由调用方保证。

cosine scheduler 必须维持初始总训练 horizon；不能先以 epochs=2 跑完，再改为 epochs=4，却要求与起初 horizon=4 的连续轨迹相同。启用 restore_best 的 DDP 恢复要求 save_best=True 且存在相同 run_id/best_epoch 的 best 文件；移动 checkpoint 时应一并移动 last/best。

直接调用 save_training_checkpoint 仍是 rank-local 保存，不执行集体协调；完整 DDP checkpoint 生命周期通过 Trainer 完成。

### 3.4 故障传播与所有权

rank-0-only 生命周期事件、checkpoint 写盘与全局早停决策已验证。已覆盖单 rank 非有限 loss、adapter/validation 异常、配置不一致、错误 sampler、缓冲型 metric 以及写盘失败，所有参与 rank 共同报错而不是一边失败另一边宣称成功。

这不意味着可以捕获任意原生故障。进程被杀、设备故障、backward 内部错误及模型自定义 collective 仍依赖进程组 timeout 与 torchrun 终止；Session 退出不做无条件 barrier。自建进程组由 Session 销毁，借用的组保持存活。

## 4. 本轮回归台账

开发中先执行训练/适配器/事件回归，再加入双进程等效、恢复和异常场景。一次较长执行通道截断没有计作整组通过，随后拆成有明确退出码的小组验证。全量回归发现 tests/test_registry.py 写死版本为 0.8.0，改为版本格式断言；安装态仍严格断言 0.9.0。此后重新冻结源码，完整执行以下全部测试，没有拼接冻结前后的结果。

| 最终冻结代码验收范围 | 唯一测试数 | 结果 |
|---|---:|---|
| 代表性 model/domain API | 266 | 全部通过 |
| Core / contracts / registry / API / catalog | 98 | 全部通过 |
| AutoML support / protocol / events | 57 | 全部通过 |
| Runner / task data / training | 71 | 全部通过 |
| AutoML Data / Backend / Task | 48 | 全部通过 |
| DDP：19 个真实双进程场景 + 11 个单元测试 | 30 | 全部通过 |
| Checkpoint 兼容及 restore-best 修复 | 4 | 全部通过 |
| **合计** | **574** | **574 passed / 0 failed / 0 skipped** |

`validation/release_validation.json` 记录每组命令对应的 node IDs、退出码、计数、超时状态及源码摘要；`release_*.log`/`release_*.xml` 为最终逐组输出；`release_ddp_*_rank_results.json` 是实际 worker 断言结果。

全量冻结源码摘要（算法见 validate_release.py）：`bf247fea7a581b982dbf7125ee44c79bc0179b15f080b0ffa88ffb7f14d72d7a`。
该摘要覆盖核心/runner/tests/tools 的 .py、governance JSON 和 pyproject；文档/验证输出不参与摘要。不是整个 ZIP 的 SHA，最终文件校验值另见 SHA256SUMS。

关键专项结果：

- 两 rank 参数同步，global-batch SGD 与单进程等效训练在指定浮点容差内一致；不是笼统宣称所有浮点运算逐位相同。
- CPU/Gloo、固定 world_size=2、num_workers=0，含 Dropout、Python/NumPy 扰动的连续 4 epoch 与 2+2 epoch 恢复：参数与训练历史逐项完全一致。
- 固定 horizon 的 cosine 恢复逐参数及 scheduler 状态一致；采样配置变更、world size 变更、缺少 rank 状态被拒绝。
- 全局 F1/R² 与整集计算对照；7 样本不补齐、空验证 rank、早停一致性、原生 FTTransformer 与 DLinear 路径均通过。
- 历史 schema v1/v2 可读；同时修复单进程设置 checkpoint_dir 且 save_best=False 时 restore_best 未正确恢复的问题。

## 5. 构建、安装与环境

| 检查 | 结果 |
|---|---|
| API baseline | 904 protected symbols / current |
| Model catalog | 165 models / current |
| Lazy export indexes | current |
| compileall | 通过 |
| wheel 构建 | pip wheel --no-deps --no-build-isolation，成功 |
| 安装态 import | 从 /tmp 运行，来自 /mnt/data/tl090/installed/torch_like/__init__.py |
| 安装态普通 smoke | 版本 0.9.0、165 模型、sklearn 延迟导入、原生 RF、DLinear save/load 完全一致 |
| 安装态真实双进程 | 同步参数、7 样本全局指标、schema v3、primary-only events、清理退出，通过 |
| 安装态可运行示例 | 训练 2 epoch，然后恢复到累计 4 epoch，通过 |

实际环境 Python 3.13.5、PyTorch 2.10.0+cpu、NumPy 2.3.5、sklearn 1.8.0、setuptools 82.0.1。wheel 使用独立 --target 目录安装，**复用宿主依赖，不是全新隔离虚拟环境**。包元数据原有的较宽依赖范围不是本轮全部版本矩阵通过的证明。

Ruff 当前不可用，安装尝试失败；未声称仓库 Ruff clean、coverage gate 通过或 GitHub CI 已运行。`release_gates.json` 与安装态 JSON/log 是本轮证据。源码清洁 staging 再做的小型复验单独记录，不重复计入 574。

## 6. 直接运行

在解压的源码根目录：

```bash
python -m pip install -e .
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 examples/train_ddp_tabular.py --device cpu --epochs 2 --out runs/ddp_tabular
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 examples/train_ddp_tabular.py --device cpu --epochs 4 --out runs/ddp_tabular --resume runs/ddp_tabular/last_checkpoint.pt
```

第二条命令训练，第三条恢复到**累计 4 epoch**。输出 last_checkpoint.pt、best_checkpoint.pt、推理用 model/、summary.json。本例 batch_size=4 是每 rank 大小，两 rank 全局 batch=8。示例数据是合成数据，结果不能解释为业务有效性。

完整可复跑验收：

```bash
python -m pip install pytest
python tools/validate_release.py --all
```

重点测试：

```bash
python -m pytest tests/test_ddp_runtime.py tests/test_checkpoint_compatibility.py -q
```

## 7. 未解决的范围与后续优先级

当前 DDP 限定普通 map-style DataLoader、固定 batch、num_workers=0、相等训练步数、样本平均标量 loss、单优化器、固定 world_size、epoch 边界恢复。不支持 FSDP、DDP join、SyncBatchNorm、梯度累积、多优化器交替训练、RL rollout 或弹性重分片。

CUDA/NCCL、多机、Windows、其他 PyTorch/Python 组合、所有 165 个模型的 DDP 适配、真实数据业务收益、吞吐量与扩展效率仍待验收。下一步应优先真实 GPU 资格测试、数据版本绑定及真实业务样例，而不是立即扩张到 FSDP。

## 8. 文件与证据索引

- `docs/DDP_RUNTIME.md`：完整使用/架构/异常边界。
- `examples/train_ddp_tabular.py`：可运行示例。
- `tools/validate_release.py`：可复跑全量分组门禁。
- `validation/release_validation.json`：唯一测试总账。
- `validation/release_*.xml` / `.log`：最终分组回归证据。
- `validation/release_ddp_*_rank_results.json`：真实双进程断言与退出结果。
- `validation/installed_smoke.json`、`installed_ddp_results.json`、`installed_example_resume.json`：安装态验收。
- `validation/change_manifest.json`：相对 0.8.0 的新增/修改/删除文件和 SHA-256。
- 单独的 `torch_like-0.9.0-SHA256SUMS.txt`：交付源码、wheel、报告及证据包的最终 SHA-256。

## 9. 上游依据

实现以实际测试的 PyTorch 2.10 文档为依据，并未把在线最新版本当作本轮测试版本。

- DDP 的进程组、forward 同步与数据分片责任：https://docs.pytorch.org/docs/2.10/generated/torch.nn.parallel.DistributedDataParallel.html
- DistributedSampler 的补齐/丢尾和 set_epoch 语义：https://docs.pytorch.org/docs/2.10/data.html#torch.utils.data.distributed.DistributedSampler
- 进程组生命周期及 object collectives 的可信数据/性能限制：https://docs.pytorch.org/docs/2.10/distributed.html
