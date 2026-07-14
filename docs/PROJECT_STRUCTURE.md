# torch_like Project Structure / 项目结构

中文：本文说明 `torch_like` 的目录分层和边界。目标是防止模型框架、实验脚本、数据处理和文档混在一起。

English: This document explains the directory layering and boundaries of `torch_like`. The goal is to keep the model framework, experiment scripts, data processing, and documentation separate.

## 目标 / Goal

中文：`torch_like` 的核心目标是成为本地 PyTorch 风格模型库，而不是训练工程集合。

English: The core goal of `torch_like` is to be a local PyTorch-style model library, not a collection of training projects.

## 分层 / Layering

### 1. Core framework / 核心框架层

Paths / 路径:

- `torch_like/models/`
- `torch_like/nn/`
- `torch_like/utils/`
- `torch_like/data/`

中文：这里放可复用 API、模型定义、任务 batch、registry 和基础工具。

English: This layer contains reusable APIs, model definitions, task batches, registry utilities, and basic helpers.

Rules / 规则:

- 中文：模型必须继承 `nn.Module` 或 `BaseModel` 的任务子类。
- English: Models must inherit from `nn.Module` or a task-specific subclass of `BaseModel`.
- 中文：不要在模型文件中写数据集专属训练流程。
- English: Do not put dataset-specific training workflows inside model files.
- 中文：新增模型要接入对应 domain 的 `__init__.py` 和 registry。
- English: New models should be exported through the domain `__init__.py` and registered in the catalog.

### 2. Auxiliary experiment layer / 辅助实验层

Paths / 路径:

- `runner/`
- `examples/`

中文：runner 用于验证模型能否训练、评估、比较，但不是核心框架 API。

English: Runners validate whether models can be trained, evaluated, and compared, but they are not the core framework API.

Rules / 规则:

- 中文：runner 可以依赖数据加载和训练循环。
- English: Runners may depend on data loading and training loops.
- 中文：runner 不应该定义模型结构。
- English: Runners should not define model architectures.
- 中文：如果某段逻辑对所有模型通用，可以抽到 `runner/common.py` 或 `torch_like/data/`。
- English: If logic is shared across models, move it to `runner/common.py` or `torch_like/data/`.

### 3. Verification layer / 验证层

Path / 路径:

- `tests/`

中文：测试关注框架合同，不追求论文指标复现。

English: Tests focus on framework contracts, not paper-score reproduction.

Minimum coverage / 最低覆盖:

- 中文：直接实例化。
- English: Direct instantiation.
- 中文：registry 构造。
- English: Registry construction.
- 中文：`predict` / `fit_step` / `score_batch` 输出形状。
- English: Output shapes for `predict`, `fit_step`, and `score_batch`.
- 中文：顶层 lazy import。
- English: Top-level lazy imports.

### 4. Documentation layer / 文档层

Path / 路径:

- `README.md`
- `docs/`

中文：所有说明文档采用中英文双语格式。

English: All user-facing documentation should be bilingual in Chinese and English.

### 5. Generated artifact layer / 生成产物层

Paths / 路径:

- `runner/results/`
- future checkpoint or benchmark outputs

中文：生成产物不应成为框架 API 的一部分。

English: Generated artifacts should not become part of the framework API.

### 6. Packaging layer / 打包分发层

Paths / 路径:

- `pyproject.toml`
- `tools/build_package.ps1`
- `tools/verify_package_install.ps1`
- `dist/` (generated and ignored / 生成并忽略)

中文：打包层把核心 `torch_like/` 包构建为可由其他 Python 环境安装的 wheel。runner、测试、示例和本地数据不进入分发包。

English: The packaging layer builds the core `torch_like/` package into a wheel installable by other Python environments. Runners, tests, examples, and local data are excluded from distributions.

## Registry Boundary / 注册边界

中文：registry 只注册可复用模型，不注册训练脚本、数据集脚本或一次性实验。

English: The registry should only register reusable models, not training scripts, dataset scripts, or one-off experiments.

Current domains / 当前域:

- `forecasting`
- `detection`
- `rul`
- `graph`
- `sequence`
- `vision`
- `multimodal`
- `generative`
- `diagnosis`
- `tabular`
- `recommendation`
- `audio`
- `reinforcement`

## Import Boundary / 导入边界

中文：用户可通过两种方式使用模型。

English: Users can access models in two ways.

```python
import torch_like as tl
model = tl.PatchTST(...)
```

```python
from torch_like.models import create_model
model = create_model("PatchTST", domain="forecasting", ...)
```

中文：新增模型要支持这两种方式。

English: New models should support both access patterns.

## Data Boundary / 数据边界

中文：`torch_like/data/` 提供通用数据协议；数据集专属解析优先放在 runner 或外部项目。

English: `torch_like/data/` provides generic data protocols. Dataset-specific parsing should stay in runners or external projects.

## Practical Design Rule / 实用设计规则

中文：如果一个文件的主要价值是“定义模型结构”，放进 `torch_like/models/`；如果主要价值是“跑某个实验”，放进 `runner/`；如果主要价值是“解释如何使用”，放进 `docs/`。

English: If a file mainly defines model architecture, put it under `torch_like/models/`. If it mainly runs an experiment, put it under `runner/`. If it explains usage, put it under `docs/`.
