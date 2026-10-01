# Engineering Architecture / 工程化架构

中文：本文定义 `torch_like 0.7` 的稳定工程边界。目标不是把 PyTorch 与 scikit-learn 强行统一成同一种对象，而是在同一个库中提供清晰、可发现、可按需加载的两类模型入口。

English: This document defines the stable engineering boundaries for `torch_like 0.7`. The goal is not to force PyTorch and scikit-learn into one object model, but to expose both through clear, discoverable, lazy-loading entry points.

## Public Layers / 公共分层

| Namespace | Responsibility / 职责 | Object model / 对象模型 |
|---|---|---|
| `torch_like.models` | PyTorch-native model implementations and registry / PyTorch 原生模型与注册表 | `torch.nn.Module` |
| `torch_like.classical` | Curated native scikit-learn estimator access / sklearn 原生 estimator 兼容入口 | sklearn `fit/predict` |
| `torch_like.nn` | Reusable neural-network building blocks / 可复用神经网络组件 | `torch.nn.Module` |
| `torch_like.data` | Dataset/task-data contracts and reusable DataAdapters / 数据协议与 DataAdapter | Python/PyTorch data objects |
| `torch_like.evaluation` | Dataset-level metrics and evaluators / 数据集级指标与评估器 | numpy/torch predictions + targets |
| `torch_like.contracts` | Machine-readable task/model capabilities / 机器可读模型与任务能力 | immutable metadata |
| `torch_like.training` | Single-process PyTorch optimization lifecycle / 单进程 PyTorch 训练生命周期 | native `nn.Module` + optimizer |
| `torch_like.automl` | Optional orchestration and benchmarking / 可选 AutoML 编排 | backend-aware orchestration |

中文：`torch_like.classical` 返回的就是 sklearn 原类和原实例，不做 `nn.Module` 包装；因此不会破坏 sklearn 的 `fit`、`predict`、`get_params`、pipeline 和 clone 语义。

English: `torch_like.classical` returns the original sklearn classes and instances. It does not wrap them as `nn.Module`, preserving sklearn `fit`, `predict`, `get_params`, pipeline, and clone semantics.

`torch_like.training` similarly does not replace model classes. `torch_like.evaluation` computes dataset-level metrics without changing model output contracts, and `torch_like.data` owns reusable loader materialization. The training layer operates on the same native `nn.Module`, task batch contracts, and `ModelRuntime.loss()` capability used elsewhere. AutoML reuses this lifecycle for ordinary torch candidates instead of maintaining a second epoch loop.

## Recommended Imports / 推荐导入

### 1. Application code / 业务代码

```python
import torch_like as tl

model = tl.create("PatchTST", domain="forecasting", ...)
baseline = tl.create("RandomForestRegressor", backend="sklearn", n_estimators=200)
```

### 2. Explicit domain imports / 明确域导入

```python
from torch_like.models.forecasting import PatchTST
from torch_like.models.graph import EGNN
from torch_like.classical import RandomForestClassifier
```

### 3. Config-driven construction / 配置驱动构造

```python
from torch_like import create_model, create_estimator

model = create_model("EGNN", domain="graph", ...)
estimator = create_estimator("Ridge", alpha=1.0)
```

中文：旧的 `tl.PatchTST`、`tl.EGNN` 等顶层短入口仍兼容，但新工程代码优先使用 domain namespace 或 `create(...)`，避免顶层命名空间继续无限膨胀。

English: Legacy shortcuts such as `tl.PatchTST` and `tl.EGNN` remain compatible, but new application code should prefer domain namespaces or `create(...)` so the root namespace does not keep growing indefinitely.

## Lazy Import Boundary / 按需加载边界

每个模型 domain 采用三层结构：

```text
torch_like/models/<domain>/
  __init__.py       # lightweight lazy public facade / 轻量公共入口
  _catalog.py       # private metadata registration / 私有模型目录与注册
  common.py         # domain task contracts / 域级任务合同
  <model>.py        # model implementation / 单模型实现
```

中文：直接访问一个模型只导入其实现和直接依赖；不会因为 `tl.PatchTST` 自动导入 vision、audio、graph 等无关 domain，也不会自动加载该 domain 的完整 `_catalog.py`。只有需要完整元数据的 `list_models()` / `get_model_info()` 才加载对应 catalog。

English: Direct model access imports only the requested implementation and its direct dependencies. Accessing `tl.PatchTST` no longer imports unrelated vision, audio, or graph domains, and it does not load the full domain catalog. Full metadata operations such as `list_models()` and `get_model_info()` load the relevant catalog on demand.

轻量发现接口 / Lightweight discovery:

```python
import torch_like as tl

print(tl.list_domains())
print(tl.list_model_names("forecasting"))  # no model implementation import
print(tl.list_estimators("classification"))  # no sklearn import
```

## Compatibility Rules / 兼容规则

1. 中文：PyTorch-native 模型继续继承 `nn.Module`，保持 `state_dict()`、`.to()`、`.train()`、`.eval()` 等语义。  
   English: PyTorch-native models continue to inherit `nn.Module` and preserve `state_dict()`, `.to()`, `.train()`, and `.eval()` semantics.
2. 中文：sklearn estimator 保持原始 sklearn 类型，不做适配层伪装。  
   English: sklearn estimators retain their native sklearn types; no compatibility wrapper disguises them.
3. 中文：旧 import 路径在弃用周期内保持可用。  
   English: Existing import paths remain available under the formal deprecation policy.
4. 中文：新增模型不得把训练脚本、数据集专属处理或 benchmark 逻辑写进模型实现。  
   English: New models must not embed training scripts, dataset-specific parsing, or benchmark orchestration in model implementation files.
5. 中文：通用单优化器训练生命周期进入 `torch_like.training`；数据加载复用进入 `torch_like.data`，数据集级指标进入 `torch_like.evaluation`；确需多优化器/多阶段目标的算法必须保留显式专用路径，不能用错误抽象强行接入。  
   English: Generic single-optimizer lifecycle belongs in `torch_like.training`; algorithms requiring multi-optimizer or multi-phase objectives keep explicit specialized paths until a correct shared contract exists.

## Source-tree and Installed Execution / 源码与安装态运行

中文：开发态允许直接在仓库根目录运行 `pytest` 或 `python -m pytest`；AutoML 环境记录在 wheel 未安装时会回退到源码中的 `torch_like.__version__`，因此不再要求先执行 `pip install -e .` 才能完成测试。

English: Development runs support both `pytest` and `python -m pytest` from the repository root. AutoML environment capture falls back to `torch_like.__version__` when distribution metadata is unavailable, so editable installation is no longer a hidden prerequisite for source-tree tests.

发布态仍应使用 wheel 并执行隔离安装验收；源码可运行不等于可以跳过分发验证。

Release deployments should still use wheels and isolated installation verification; source-tree execution does not replace distribution validation.
## 7. Generated lazy indexes / 自动生成的延迟加载索引

新增或移动内置模型时，只修改实现文件与对应 domain 的 `_catalog.py`，随后执行：

```bash
python tools/generate_lazy_exports.py
python tools/generate_lazy_exports.py --check
```

生成器维护 domain 导出、顶层公开导出与 built-in model index。运行时索引只包含字符串路径，因此 `import torch_like` 和 `list_model_names()` 不会为了发现模型而导入全部模型实现。

When a built-in model is added or moved, edit its implementation and private `_catalog.py`, then regenerate the indexes. Runtime indexes contain string targets only, so package import and lightweight discovery do not import the full model implementation graph.

