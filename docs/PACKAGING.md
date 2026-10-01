# torch_like 打包安装指南 / Packaging Guide

## 目标 / Goal

中文：项目使用 PEP 517/PEP 621 `pyproject.toml` 和 setuptools 构建 wheel。wheel 只包含 `torch_like` Python 包，不包含 `runner/`、`tests/`、`examples/`、`tools/` 或本地临时数据。

English: The project uses a PEP 517/PEP 621 `pyproject.toml` and setuptools to build wheels. A wheel contains only the `torch_like` Python package, excluding `runner/`, `tests/`, `examples/`, `tools/`, and local temporary data.

中文：`torch_like/models/catalog_schema_v2.json` 属于运行时 registry 契约，必须随 wheel 分发。隔离安装验收会读取模型 schema v2 字段，防止源码环境和安装环境使用不同的 catalog 规则。

English: `torch_like/models/catalog_schema_v2.json` is part of the runtime registry contract and must ship in the wheel. Isolated installation verifies schema-v2 metadata so source checkouts and installed packages cannot silently use different catalog rules.

## 本地构建 / Local Build

中文：推荐使用项目指定解释器执行构建脚本。脚本通过当前环境中的 setuptools 和 wheel 构建，不会下载项目运行依赖。

English: Use the project interpreter to run the build script. It builds with setuptools and wheel already available in that environment and does not download runtime dependencies.

```powershell
& .\tools\build_package.ps1 -Python 'D:\Project\base\base\Scripts\python.exe'
```

Output / 输出:

```text
dist/torch_like-0.8.0-py3-none-any.whl
```

中文：需要同时生成 wheel 和 source distribution 时，先安装构建工具，再运行标准构建命令。

English: To produce both a wheel and a source distribution, install the build tools and run the standard build command.

```powershell
& 'D:\Project\base\base\Scripts\python.exe' -m pip install ".[build]"
& 'D:\Project\base\base\Scripts\python.exe' -m build
& 'D:\Project\base\base\Scripts\python.exe' -m twine check dist\*
```

## 隔离验收 / Isolated Verification

中文：验收脚本创建 `.packaging-venv`，安装刚构建的 wheel，并从项目目录之外导入 `torch_like`。为避免本地重复下载 PyTorch，验收环境复用指定解释器的第三方依赖，但 `torch_like` 本身必须来自 wheel。

English: The verification script creates `.packaging-venv`, installs the newly built wheel, and imports `torch_like` outside the project directory. To avoid downloading PyTorch again, it reuses third-party dependencies from the selected interpreter, while `torch_like` itself must come from the wheel.

```powershell
& .\tools\verify_package_install.ps1 -Python 'D:\Project\base\base\Scripts\python.exe'
```

The check validates / 验收内容:

- distribution version equals `torch_like.__version__` / 分发版本与运行时版本一致
- import path points to `.packaging-venv/Lib/site-packages` / 导入来自隔离环境
- registry can discover all domains / registry 可以发现全部领域

## 在其他环境安装 / Install in Another Environment

中文：目标环境联网时，直接安装 wheel，pip 会解析并安装缺失依赖。

English: In an online target environment, install the wheel directly and pip will resolve missing dependencies.

```powershell
& 'D:\Path\To\Python\python.exe' -m pip install D:\Project\torch_like\dist\torch_like-0.8.0-py3-none-any.whl
```

中文：也可以把 `dist/` 当作本地包源，从而使用按名称安装的形式。

English: The `dist/` directory can also act as a local package source, enabling installation by distribution name.

```powershell
& 'D:\Path\To\Python\python.exe' -m pip install --no-index --find-links D:\Project\torch_like\dist torch_like
```

中文：如果目标环境已经提供兼容版本的 PyTorch、NumPy、pandas 和 scikit-learn，可使用 `--no-deps` 避免重复安装。调用者负责保证这些依赖真实可用。

English: If the target environment already provides compatible PyTorch, NumPy, pandas, and scikit-learn versions, use `--no-deps` to avoid reinstalling them. The caller is responsible for dependency compatibility.

```powershell
& 'D:\Path\To\Python\python.exe' -m pip install --no-deps D:\Project\torch_like\dist\torch_like-0.8.0-py3-none-any.whl
```

Verification / 安装后检查:

```powershell
& 'D:\Path\To\Python\python.exe' -c "import torch_like as tl; print(tl.__version__); print(tl.list_domains())"
```

Upgrade and uninstall / 升级与卸载:

```powershell
& 'D:\Path\To\Python\python.exe' -m pip install --upgrade .\dist\torch_like-0.8.0-py3-none-any.whl
& 'D:\Path\To\Python\python.exe' -m pip uninstall torch_like
```

## 版本与发布边界 / Versioning and Publication Boundary

中文：构建版本由 `torch_like.__version__` 单一提供。发布新版本前应更新该值、运行全量质量门禁、重新构建并执行隔离验收。

English: `torch_like.__version__` is the single source of the build version. Before releasing a new version, update it, run the full quality gate, rebuild, and execute isolated verification.

中文：当前流程解决本地 wheel 分发，项目代码采用 Apache-2.0。上传 PyPI 还需要单独确认项目名称可用性、第三方来源合规、作者/维护者信息、发布凭据和版本发布策略。

English: This workflow supports local wheel distribution and project code is licensed under Apache-2.0. Publishing to PyPI additionally requires name availability, third-party provenance compliance, author/maintainer metadata, publishing credentials, and a release policy.
