# Engineering Refactor Report / 工程化重构报告

## 1. Goal / 目标

本次重构的目标不是重写现有模型，而是把仓库从“模型集合”收敛为可安装、可发现、可延迟加载、可兼容 sklearn、可稳定回归的 Python 库，同时尽量保留现有调用方式。

The goal is to turn the repository from a model collection into an installable and maintainable Python library without breaking established model APIs.

## 2. Problems found / 发现的问题

1. **首次模型访问触发全库加载。** 原实现的 `torch_like.models.__getattr__` 会先 bootstrap 全部 domain registry；访问一个 `PatchTST` 会连带导入 vision、graph、其他 forecasting 模型等大量无关模块。
2. **Domain `__init__.py` 责任过重。** 多个 domain 的 `__init__.py` 同时承担实现导入、catalog 定义、注册和公开 API，文件大且耦合。
3. **sklearn 不是一等公共 API。** sklearn 主要作为 AutoML 内部 backend 使用，用户缺少统一、稳定的 estimator discovery/construction 入口。
4. **模型发现与模型加载耦合。** 为了列出模型或判断模型所属 domain，需要先导入真实实现。
5. **源码树执行存在隐式安装依赖。** AutoML 读取 distribution metadata；未执行 editable install 时会抛 `PackageNotFoundError`。
6. **测试入口依赖调用方式。** `python -m pytest` 可工作，但直接 `pytest` 在原仓库中可能因 source root 未加入 import path 而 collection 失败。
7. **公共 API 基线受 Python 版本影响。** Enum 的运行时签名在 Python 3.12/3.13 存在差异，导致无真实 API 破坏时也会产生 false positive。
8. **长时间 AutoML 回归存在进程可重入风险。** 多次 trial 后，PyTorch/BLAS/sklearn 原生线程池与 Python multiprocessing 的反复 fork/spawn 会使后续 trial 出现挂起风险。

## 3. Implemented architecture / 已实施架构

### 3.1 Model domains

每个 package domain 现在采用：

```text
torch_like/models/<domain>/
  __init__.py     # lightweight lazy facade / 轻量延迟入口
  _catalog.py     # private metadata + registration / 私有目录与注册
  common.py       # shared contracts / 公共合同
  <model>.py      # implementation / 模型实现
```

普通 import 不再执行整个 domain catalog。单模型访问按字符串索引精确导入目标实现。

### 3.2 Lightweight indexes

新增：

- `torch_like/models/_domain_exports.py`
- `torch_like/models/_public_exports.py`
- `torch_like/models/_builtin_index.py`
- `tools/generate_lazy_exports.py`

这些运行时索引只保存字符串路径。开发者只维护实现和 `_catalog.py`，再由生成器同步索引。

### 3.3 Registry and discovery

新增或调整：

- `list_model_names(domain=None)`：无需导入模型实现即可列出 built-in 名称；
- `find_model_domain(name)`：无需全域 bootstrap 即可确定 built-in domain；
- `create_model(...)`：built-in 模型按精确索引解析，不再先加载全部 domain；
- `model_zoo[domain]`：只加载指定 domain；全量遍历时才加载全 catalog。

### 3.4 sklearn compatibility

新增 `torch_like.classical`。它暴露真实 sklearn estimator 类，不复制 sklearn 源码，也不把 sklearn estimator 包装成 `torch.nn.Module`。

```python
import torch_like as tl

rf = tl.create_estimator("RandomForestClassifier", n_estimators=200)
rf2 = tl.classical.RandomForestClassifier(n_estimators=200)
```

上述对象保持 sklearn 原生类型和行为。

### 3.5 Unified construction

新增 backend-aware factory：

```python
model = tl.create("DLinear", enc_in=8, seq_len=96, pred_len=24)
estimator = tl.create("RandomForestClassifier", backend="sklearn", n_estimators=200)
```

`backend="auto"` 会先解析 torch_like built-in model，再解析 sklearn compatibility catalog；对象本身仍保持各自原生类型。

### 3.6 AutoML process isolation

Trial 隔离从长期父进程中的 `multiprocessing` worker 改为独立 Python interpreter subprocess。父进程通过短生命周期 payload 文件传递 trial 输入，子进程继续使用既有 artifact JSON 作为 durable result contract。超时在 POSIX 下终止整个 process group，在 Windows 下使用 terminate/kill fallback。

这样避免把已经初始化 PyTorch/OpenMP/MKL/BLAS 线程池的父进程状态复制到后续 trial。Anomaly PyTorch DataLoader 也不再创建嵌套 worker process；CPU 并行度由 `torch.set_num_threads` 控制。

## 4. Compatibility / 兼容性

保留已有的顶层模型短入口，例如：

```python
import torch_like as tl
model_cls = tl.PatchTST
```

推荐新代码优先使用结构化入口：

```python
from torch_like.models.forecasting import PatchTST
from torch_like.classical import RandomForestClassifier
```

因此本次重构是兼容式收敛，不要求现有项目批量改写 import。

## 5. Measured import behavior / 导入行为实测

在本次容器环境的单次隔离进程测量中：

| Scenario / 场景 | Original / 原仓库 | Refactored / 重构后 |
|---|---:|---:|
| `torch_like.models*` modules after `tl.PatchTST` | 192 | 12 |
| unrelated `vision` loaded | yes | no |
| unrelated forecasting `Autoformer` loaded | yes | no |
| private forecasting catalog loaded | implicit via bootstrap | no |
| `list_model_names("forecasting")` adds implementation modules | N/A | 0 |
| plain `import torch_like` imports sklearn | no | no |

模块数量是比单次 wall-clock timing 更稳定的指标；这里的目标是减少 import side effects，而不是宣称固定的启动耗时提升比例。

## 6. Verification / 验证

最终验收包含：

- full pytest suite：`499 passed`；
- public API audit：`824 protected symbols`，baseline current；
- model catalog audit：`165 models`，audit current；
- lazy import / registry / engineering API 专项测试；
- source-tree AutoML 执行；
- sklearn native class identity；
- package build 与安装态 smoke test（见交付阶段记录）。

## 7. Recommended user-facing API / 推荐调用方式

```python
import torch_like as tl

# Lightweight discovery / 轻量发现
print(tl.list_model_names("forecasting"))
print(tl.list_estimators(kind="classifier"))

# torch_like native model / torch_like 原生模型
model = tl.create("DLinear", enc_in=8, seq_len=96, pred_len=24)

# sklearn native estimator / sklearn 原生 estimator
clf = tl.create("RandomForestClassifier", backend="sklearn", n_estimators=200)
```

对明确 domain 的业务代码，更推荐结构化 import，因为它最利于 IDE、静态分析和依赖边界管理。
