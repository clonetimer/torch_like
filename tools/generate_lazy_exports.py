"""Regenerate lightweight lazy-export indexes from private model catalogs."""

from __future__ import annotations

import argparse
import ast
import importlib
import pprint
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOMAIN_CATALOGS = {
    "forecasting": "torch_like.models.forecasting._catalog",
    "diagnosis": "torch_like.models.diagnosis._catalog",
    "detection": "torch_like.models.detection._catalog",
    "graph": "torch_like.models.graph._catalog",
    "sequence": "torch_like.models.sequence._catalog",
    "vision": "torch_like.models.vision._catalog",
    "multimodal": "torch_like.models.multimodal._catalog",
    "generative": "torch_like.models.generative._catalog",
    "tabular": "torch_like.models.tabular._catalog",
    "recommendation": "torch_like.models.recommendation._catalog",
    "audio": "torch_like.models.audio._catalog",
    "reinforcement": "torch_like.models.reinforcement._catalog",
}
RUL_CATALOG = "torch_like.models.rul_catalog"
TOP_LEVEL_EXCLUDE = {"SELayer", "SEBottleneck"}


def _target(value: Any, export_name: str) -> tuple[str, str]:
    module_name = getattr(value, "__module__", None)
    if not isinstance(module_name, str) or not module_name.startswith("torch_like.models"):
        raise TypeError(f"Cannot derive lazy target for {export_name!r}: {value!r}")
    module = importlib.import_module(module_name)
    if getattr(module, export_name, object()) is value:
        return module_name, export_name
    attr_name = getattr(value, "__name__", None)
    if isinstance(attr_name, str) and getattr(module, attr_name, object()) is value:
        return module_name, attr_name
    raise AttributeError(f"Cannot locate {export_name!r} in {module_name}")


def _load_domain_exports() -> tuple[dict[str, dict[str, tuple[str, str]]], dict[str, list[str]]]:
    exports: dict[str, dict[str, tuple[str, str]]] = {}
    all_names: dict[str, list[str]] = {}
    for domain, module_name in DOMAIN_CATALOGS.items():
        module = importlib.import_module(module_name)
        names = list(module.__all__)
        all_names[domain] = names
        exports[domain] = {name: _target(getattr(module, name), name) for name in names if name != "model_zoo"}
    return exports, all_names


def _load_builtin_index() -> dict[str, tuple[str, str, str]]:
    result: dict[str, tuple[str, str, str]] = {}
    catalogs = {**DOMAIN_CATALOGS, "rul": RUL_CATALOG}
    for domain, module_name in catalogs.items():
        module = importlib.import_module(module_name)
        for name, metadata in module.model_zoo.items():
            cls = metadata["cls"]
            result[name] = (domain, cls.__module__, cls.__name__)
    return result


def _load_public_exports(
    domain_exports: dict[str, dict[str, tuple[str, str]]],
) -> dict[str, tuple[str, str]]:
    public: dict[str, tuple[str, str]] = {}
    for exports in domain_exports.values():
        for name, target in exports.items():
            if name not in TOP_LEVEL_EXCLUDE:
                public.setdefault(name, target)

    rul = importlib.import_module(RUL_CATALOG)
    for name in rul.__all__:
        if name != "model_zoo":
            public[name] = _target(getattr(rul, name), name)
    return public


def _render(domain_exports, domain_all, public_exports, builtin_index) -> dict[Path, str]:
    pf = lambda value: pprint.pformat(value, width=120, sort_dicts=False)  # noqa: E731
    return {
        ROOT / "torch_like/models/_domain_exports.py": (
            '"""Generated lightweight public export map for model domains.\n\n'
            "Do not import model implementations here; values must remain strings only.\n"
            '"""\n\n'
            f"DOMAIN_EXPORTS = {pf(domain_exports)}\n\n"
            f"DOMAIN_ALL = {pf(domain_all)}\n"
        ),
        ROOT / "torch_like/models/_public_exports.py": (
            f'"""Generated lazy export map for :mod:`torch_like.models`."""\n\nPUBLIC_EXPORTS = {pf(public_exports)}\n'
        ),
        ROOT / "torch_like/models/_builtin_index.py": (
            '"""Static built-in model index used for discovery without importing model implementations."""\n\n'
            f"BUILTIN_MODEL_INDEX = {pf(builtin_index)}\n"
        ),
    }


def generate() -> dict[Path, str]:
    domain_exports, domain_all = _load_domain_exports()
    return _render(
        domain_exports,
        domain_all,
        _load_public_exports(domain_exports),
        _load_builtin_index(),
    )


def _is_current(path: Path, content: str) -> bool:
    """Compare Python structure so formatters do not make indexes appear stale."""
    try:
        actual = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError, ValueError):
        return False
    expected = ast.parse(content)
    return ast.dump(actual, include_attributes=False) == ast.dump(expected, include_attributes=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail when generated files are stale")
    args = parser.parse_args()

    generated = generate()
    if args.check:
        stale = [path for path, content in generated.items() if not _is_current(path, content)]
        if stale:
            print("Lazy export indexes are stale:")
            for path in stale:
                print(f"- {path.relative_to(ROOT)}")
            return 1
        print("Lazy export indexes are current")
        return 0

    for path, content in generated.items():
        path.write_text(content)
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
