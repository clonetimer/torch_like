# 0.9.0：最小 DDP 执行运行时

## 定位与验收范围

0.8.0 只提供 rank、sampler、指标合并合同；0.9.0 将这些合同接入共享 Trainer。
公开模型仍是原生 `nn.Module`。本次不是新的张量库，也不是自动适配所有训练目标的分布式平台。

本次实测环境是 Linux、Python 3.13.5、PyTorch 2.10.0+cpu、两个本机 Gloo 进程。
CUDA/NCCL 设备绑定和配置路径已经实现，但当前交付环境没有 GPU，不能把 CPU 通过视为 GPU 验收。
Windows、多机、其他 PyTorch 版本同样未在本轮实测。完整结果见 `validation/release_validation.json`。

## 安装与运行

在解压后的项目根目录执行：

```bash
python -m pip install -e .
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  examples/train_ddp_tabular.py --device cpu --epochs 2 --out runs/ddp_tabular
```

恢复到累计 4 个 epoch：

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  examples/train_ddp_tabular.py --device cpu --epochs 4 --out runs/ddp_tabular \
  --resume runs/ddp_tabular/last_checkpoint.pt
```

Windows PowerShell 请将上述命令写在一行，或将行尾 `\` 改成 PowerShell 续行符。
这只是命令语法提示，不代表本轮已经验证 Windows/Gloo。

运行产生 `last_checkpoint.pt`、`best_checkpoint.pt`、推理用 `model/` 和 `summary.json`。
`batch_size` 是**每个 rank** 的 batch 大小；本例每 rank 4、两 rank 的全局 batch 为 8。
样例是合成数据 smoke test，不是业务有效性或模型精度基准。

双 GPU 主机可将 `--device cpu` 换成 `--device cuda`，再自行执行同样的训练/恢复验收。
CUDA 模式要求一进程一可见 GPU，设备索引必须对应 `LOCAL_RANK`；本版本仅接受 CPU/Gloo 或 CUDA/NCCL 配对。

## 用户 API

新增接口只放在 `torch_like.training`，不继续扩张顶层别名：

```python
from torch_like.training import (
    DDPConfig,
    DistributedContext,
    DistributedEvalSampler,
    DistributedSamplerConfig,
    DistributedSession,
    create_distributed_sampler,
)
```

`DistributedSession` 从 torchrun 环境变量读取 rank/world_size/local_rank，拥有自己创建的进程组。
外部已经创建匹配的默认进程组时，session 仅借用，退出不销毁它。
Trainer 本身不创建或销毁进程组，不启动子进程。

给 `Trainer` 传入 `distributed_context=session.context`，同时在 `TrainerConfig.device` 中使用
`str(session.device)`。没有显式传 context 时，Trainer 读取标准 torchrun 环境变量；在多进程环境没有初始化
进程组会报错，不会悄悄启动多份互不通信的单进程训练。

原生模型不应预先由用户包装成 DDP。按需传入 `ddp_config=DDPConfig(find_unused_parameters=True)`，
可处理一次 loss 中有未使用参数的模型；默认值仍为 False，不自动猜测模型的梯度路径。

## 核心执行语义

### 1. 原生 domain loss 仍经过 DDP.forward

普通 PyTorch DDP 的前向入口不能被 `ddp.module.loss_step(...)` 绕过。
本实现以私有 `_LossModule` 注册原生模型，并将 `runtime.loss`、`loss_fn` 或 `step_fn` 的计算放进其 `forward`。
因此 `trainer.model is model` 保持成立，训练前向/反向仍由 DDP reducer 管理。

epoch loop、optimizer、scheduler、early stopping、checkpoint 调用顺序依旧只有公共 Trainer 一套。
`_ddp.py` 只处理集体通信、数据合同检查、loss 前向包装和多 rank checkpoint 协调。
AutoML 仍默认单进程 trial；本次没有给 AutoML trial 自动配置分布式资源。

### 2. 训练数据合同

需要普通 map-style `DataLoader`、固定 batch_size、标准 `DistributedSampler` 和 `num_workers=0`。
暂不支持自定义 batch sampler、IterableDataset、不同 rank 不同训练步数、状态化数据工作进程。
同一个 epoch 的每步必须有相同、正数的 sample_count。loss 必须是样本平均意义的标量 Tensor。

Trainer 每个 epoch 自动调用 `sampler.set_epoch(epoch_index)`，首次 epoch_index 为 0。
标准训练 sampler 在不能整除时可能补齐样本或按 drop_last 丢弃尾部；训练 loss 指的是实际采样流，
不声称是未重复、未丢弃的原始训练集平均值。

每 rank 必须有相同的原始数据和样本顺序。恢复签名检查数据集大小、batch 大小、sampler seed/shuffle/drop_last、
优化器类型、关键训练参数、scheduler 类型及 metric 合同，但**不对数据内容计算指纹**。
换了一份同长度数据仍可能通过这些结构检查；保证数据版本不变是调用方责任。

### 3. 验证不补样本，不平均各 rank 最终分数

验证使用 `DistributedEvalSampler`，按 `range(rank, n, world_size)` 精确分片，无 padding、无 drop_last。
7 个样本、两 rank 分为 4+3；1 个样本、两 rank 分为 1+0。空 rank 合法，整个验证集为空则失败。

验证前从 rank 0 同步一次模型 buffer，然后各 rank 调用本地原生模型，不经过训练用 DDP.forward，
最后再归约 loss 总和/样本数并合并 metric state。这允许验证步数不等。
模型内部不能自行执行跨 rank collective；SyncBatchNorm 被显式拒绝。

DDP 仅接受 `MergeableMetric` 且 `is_streaming=True` 的指标；合并充分统计量后才 compute。
`FunctionalMetric`、任意 lambda metric、历史 `metric_fns` 被 DDP 明确拒绝；单进程仍兼容。
自定义指标需要保证同配置、同数学语义、有限状态和正确的 merge；标记 streaming 不是自动正确性证明。

目前使用 `all_gather_object` 传输 CPU 指标状态、小型控制信息和 rank RNG/buffer 状态，依赖可信同一作业进程。
这不是面向不可信 worker 的安全边界，也不是大规模高吞吐通信方案；缓冲区/指标状态越大、world_size 越大，
通信与内存成本越高。没有宣称本轮实现了张量化 all_reduce 指标优化。

### 4. 全局决策与故障边界

train/validation loss 使用全局样本加权结果，scheduler、improved、early stopping 据此在各 rank 一致执行。
生命周期事件仅在 rank 0 发出，异常通过状态同步传播给其他 rank；事件回调不得自行调用全员 collective。

已覆盖单 rank 数据适配异常、非有限 loss、预测异常、配置不一致、错误 sampler 和 checkpoint 写入失败的
协同报错。训练前向、batch 准备、正常验证循环和写盘的 Python 异常不会被隐藏为成功。

这不等于任意故障都能在 Python 层同步捕获。进程被杀、设备损坏、模型自定义 collective、backward 内部故障
仍依赖 torchrun 和进程组 timeout 终止整个作业。session 退出不做无条件 barrier，防止清理再等一个已死亡 peer。
失败后应终止当前训练作业，从有效 checkpoint 重新启动，不继续复用失败状态。

## Checkpoint 与恢复

单进程继续写 schema v2；schema v1/v2 仍可读。新增加的 Python/NumPy、validation generator RNG 字段是向后兼容扩展。
DDP 写协调的 schema v3，只有 rank 0 原子替换目标文件：

```text
model / optimizer / scheduler / scaler state
history / best state metadata / trainer config
run_id / resume_signature
rank_states[rank]:
    global rank / local rank / world_size
    Torch CPU RNG / 当前 CUDA RNG
    Python random / NumPy legacy global RNG
    train & validation DataLoader generators
    rank-local model buffers / scaler / sampler epoch
```

随机状态以可由 `torch.load(weights_only=True)` 读取的基础类型/Tensor 编码，不放入 NumPy pickle 对象。
schema v3 保存的是一个协调文件，不是分别写 N 个可能跨 epoch 混配的独立文件。
`save_training_checkpoint` 单独调用仍是 rank-local 保存函数，不会替用户同步其他 rank；DDP 协调必须经过 Trainer。

恢复要求 world_size/rank/local_rank 布局不变、数据与配置签名匹配、所有 rank 读取相同文件内容。
缺少任何 rank 的状态、world_size 改变、训练 sampler 配置改变时拒绝继续。
使用 fresh model/optimizer/Trainer 后通过 `fit(..., resume_from=...)` 加载；不要拿不同已训练 optimizer 状态直接混跑。
仅在 epoch 边界恢复，不支持 mid-batch、梯度累积中点或弹性 world_size。

对于 cosine scheduler，初始的总 epochs 就是 schedule horizon，恢复时不能另改 horizon。
例如原本目标 4 epoch，在第 2 epoch checkpoint 中断后恢复仍设置 epochs=4；
不能先按 horizon=2 跑完，再改 horizon=4 却声称与连续 4 epoch 完全相同。

DDP 下恢复并 restore_best 要求 save_best=True，checkpoint_dir 中存在与 last 相同 run_id 和 best_epoch 的 best 文件。
移动 checkpoint 时需要一起移动 last/best，或关闭 restore_best。不会把其他运行遗留的 best 文件当作本轮最优。

精确恢复的测试覆盖：CPU/Gloo、固定 world_size=2、num_workers=0、固定数据/adapter、单优化器，
含 Dropout、Python/NumPy 随机扰动、train/validation generator、Adam 与 cosine scheduler 状态。
不包括独立 `np.random.Generator`、用户持有的自定义 generator、adapter/dataset 内部可变状态、第三方随机源。
训练 loss 与 validation loss 本身也会受业务的损失归约定义影响，框架不替用户决定目标函数。

## 推理发布

DDP checkpoint 与部署 artifact 仍是不同概念。训练完毕只在 rank 0 执行：

```python
if session.context.is_primary:
    tl.save_model(model, "runs/model")
```

只有符合原项目持久化合同的模型才能自动重建；本次没有承诺任意自定义 nn.Module 都可零配置部署。
不要尝试加载不可信模型/训练文件；权重限制不是对文件内容、资源消耗或业务语义的完整验证。

## 测试与限制

```bash
python -m pytest tests/test_ddp_runtime.py tests/test_checkpoint_compatibility.py -q
python tools/validate_release.py --all
```

测试 launcher 为每个双进程场景创建独立 torchrun 作业，并设置硬 timeout 与进程组清理。
验收包含单/双进程等效全局 batch、逐参数恢复、F1/R² 全局计算、空验证 rank、checkpoint 拒绝路径及事件归属。
没有用 `world_size=2` 元数据伪造多进程通过。

仍不支持 FSDP、DDP join/不等训练步数、SyncBatchNorm、梯度累积、多 optimizer 交替训练、RL rollout orchestration，
也没有把 165 个模型的目录发现成功等同于 165 个模型全部完成分布式训练资格验证。

## 上游设计依据

- PyTorch 2.10 DDP：https://docs.pytorch.org/docs/2.10/generated/torch.nn.parallel.DistributedDataParallel.html
- 分布式初始化、销毁、object collective 的边界：https://docs.pytorch.org/docs/2.10/distributed.html
- 官方 sampler：https://docs.pytorch.org/docs/2.10/data.html#torch.utils.data.distributed.DistributedSampler
- torchrun 环境与进程启动：https://docs.pytorch.org/docs/2.10/elastic/run.html
