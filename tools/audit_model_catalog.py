"""Generate and verify the deterministic model-catalog maturity audit."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch_like  # noqa: E402
from torch_like.models.catalog_evidence import load_model_evidence_registry  # noqa: E402
from torch_like.models.catalog_schema import (  # noqa: E402
    CATALOG_SCHEMA_PATH,
    load_catalog_schema_policy,
)

DEFAULT_POLICY_PATH = CATALOG_SCHEMA_PATH
DEFAULT_JSON_PATH = ROOT / "governance" / "model_catalog_audit.json"
DEFAULT_MARKDOWN_PATH = ROOT / "docs" / "MODEL_CATALOG_AUDIT.md"

API_SIGNALS = (
    "forward",
    "predict",
    "loss_step",
    "fit_step",
    "config",
    "from_config",
    "param_count",
)


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    """Load the package-owned catalog policy used by runtime and audit code."""

    return load_catalog_schema_policy(path)


def _api_contract(cls: type[Any]) -> dict[str, bool]:
    contract = {signal: callable(getattr(cls, signal, None)) for signal in API_SIGNALS}
    contract["nn_module"] = issubclass(cls, torch.nn.Module)
    contract["task_loss"] = contract["loss_step"] or contract["fit_step"]
    return contract


def validate_evidence_registry(policy: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    """Validate evidence semantics and repository-local provenance links."""

    registry = load_model_evidence_registry()
    records = registry["records"]
    mappings = registry["model_evidence"]
    live_models = {
        f"{domain}/{name}": metadata
        for domain in torch_like.list_domains()
        for name, metadata in torch_like.list_models(domain).items()
    }

    for record_id, record in records.items():
        if record["maturity"] not in policy["maturity_levels"] or record["maturity"] == "unassessed":
            raise ValueError(f"Evidence record {record_id!r} has invalid promoted maturity")
        benchmark = record["benchmark_evidence"]
        if benchmark not in policy["benchmark_evidence_levels"] or benchmark == "unassessed":
            raise ValueError(f"Evidence record {record_id!r} has invalid benchmark level")
        for field in ("config", "report"):
            source_path = Path(record[field])
            if source_path.is_absolute() or not (root / source_path).is_file():
                raise ValueError(f"Evidence record {record_id!r} has missing repository {field}: {source_path}")
        for name in record["evaluated_models"]:
            if f"{record['domain']}/{name}" not in live_models:
                raise ValueError(f"Evidence record {record_id!r} references an unregistered model: {name!r}")

    orphaned = set(records) - set(mappings.values())
    if orphaned:
        raise ValueError(f"Evidence records are not selected by any model: {sorted(orphaned)}")
    for model_key, record_id in mappings.items():
        if model_key not in live_models:
            raise ValueError(f"Evidence mapping references an unregistered model: {model_key!r}")
        metadata = live_models[model_key]
        record = records[record_id]
        expected = (record["maturity"], record["benchmark_evidence"], record_id)
        actual = (metadata["maturity"], metadata["benchmark_evidence"], metadata["evidence_record"])
        if actual != expected:
            raise ValueError(f"Runtime metadata for {model_key!r} does not match evidence record {record_id!r}")

    return registry


def build_audit(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a deterministic audit from the live registry and validated evidence."""

    policy = policy or load_policy()
    evidence_registry = validate_evidence_registry(policy)
    models: list[dict[str, Any]] = []
    integration_counts: Counter[str] = Counter()
    domain_counts: list[dict[str, Any]] = []
    api_counts: Counter[str] = Counter()
    migration_ready = 0

    mappings = policy["legacy_integration_levels"]
    for domain in torch_like.list_domains():
        catalog = torch_like.list_models(domain)
        domain_counts.append({"domain": domain, "model_count": len(catalog)})
        for name, metadata in sorted(catalog.items()):
            legacy = metadata["integration_level"]
            integration_counts[legacy] += 1
            mapping = mappings.get(legacy)
            migration_status = "ready" if mapping is not None else "unmapped"
            migration_ready += migration_status == "ready"
            contract = _api_contract(metadata["cls"])
            api_counts.update(signal for signal, present in contract.items() if present)
            catalog_v2 = {
                "implementation_fidelity": metadata["implementation_fidelity"],
                "implementation_scope": metadata["implementation_scope"],
                "execution_profile": metadata["execution_profile"],
                "maturity": metadata["maturity"],
                "benchmark_evidence": metadata["benchmark_evidence"],
                "evidence_record": metadata["evidence_record"],
            }
            models.append(
                {
                    "domain": domain,
                    "name": name,
                    "class_path": f"{metadata['cls'].__module__}.{metadata['cls'].__qualname__}",
                    "family": metadata["family"],
                    "layer": metadata["layer"],
                    "supports": metadata["supports"],
                    "paper": metadata["paper"],
                    "description": metadata["description"],
                    "complexity": metadata["complexity"],
                    "automl_default": metadata["automl_default"],
                    "legacy_integration_level": legacy,
                    "catalog_v2": catalog_v2,
                    "api_contract": contract,
                    "migration_status": migration_status,
                }
            )

    model_count = len(models)
    return {
        "schema_version": 1,
        "policy_id": policy["policy_id"],
        "policy_version": policy["policy_version"],
        "registry_version": torch_like.__version__,
        "target_catalog_schema_version": policy["target_catalog_schema_version"],
        "summary": {
            "domain_count": len(domain_counts),
            "model_count": model_count,
            "migration_ready_count": migration_ready,
            "migration_unmapped_count": model_count - migration_ready,
            "maturity_unassessed_count": sum(model["catalog_v2"]["maturity"] == "unassessed" for model in models),
            "benchmark_evidence_unassessed_count": sum(
                model["catalog_v2"]["benchmark_evidence"] == "unassessed" for model in models
            ),
            "benchmarked_model_count": sum(model["catalog_v2"]["maturity"] == "benchmarked" for model in models),
            "evidence_record_count": len(evidence_registry["records"]),
        },
        "domains": domain_counts,
        "legacy_integration_levels": [
            {"value": value, "model_count": count} for value, count in sorted(integration_counts.items())
        ],
        "api_contract_counts": {signal: api_counts[signal] for signal in (*API_SIGNALS, "nn_module", "task_loss")},
        "evidence_records": evidence_registry["records"],
        "models": models,
    }


def render_json(audit: dict[str, Any]) -> str:
    return json.dumps(audit, ensure_ascii=False, indent=2) + "\n"


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(audit: dict[str, Any]) -> str:
    """Render the complete bilingual audit table."""

    summary = audit["summary"]
    lines = [
        "# Model Catalog Audit / 模型目录审计",
        "",
        "中文：本文件由 `tools/audit_model_catalog.py` 从实时 registry 确定性生成。接口存在性来自代码检查；成熟度和 benchmark 证据默认保持 `unassessed`，不得据此宣称论文复现或生产可用。",
        "",
        "English: This file is generated deterministically from the live registry by `tools/audit_model_catalog.py`. API presence comes from code inspection. Maturity and benchmark evidence remain `unassessed` by default and must not be treated as paper reproduction or production-readiness claims.",
        "",
        "## Summary / 摘要",
        "",
        f"- Registry version / 注册表版本: `{audit['registry_version']}`",
        f"- Domains / 领域数: {summary['domain_count']}",
        f"- Models / 模型数: {summary['model_count']}",
        f"- Schema-v2 migration ready / 可迁移: {summary['migration_ready_count']}",
        f"- Unmapped integration levels / 未映射值: {summary['migration_unmapped_count']}",
        f"- Maturity unassessed / 成熟度未评估: {summary['maturity_unassessed_count']}",
        f"- Benchmark evidence unassessed / 基准证据未评估: {summary['benchmark_evidence_unassessed_count']}",
        "",
        "## Domain Coverage / 领域覆盖",
        "",
        "| Domain / 领域 | Models / 模型数 |",
        "|---|---:|",
    ]
    lines.extend(f"| `{row['domain']}` | {row['model_count']} |" for row in audit["domains"])
    lines.extend(
        [
            "",
            "## Legacy Integration Levels / 旧集成等级",
            "",
            "| Value / 取值 | Models / 模型数 |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| `{row['value']}` | {row['model_count']} |" for row in audit["legacy_integration_levels"])
    lines.extend(
        [
            "",
            "## API Signals / API 信号",
            "",
            "中文：这里只表示类暴露了对应可调用属性，不代表其所有输入、数学性质或任务指标已经验证。",
            "",
            "English: These counts indicate callable class attributes only. They do not prove all inputs, mathematical properties, or task metrics.",
            "",
            "| Signal / 信号 | Models / 模型数 |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| `{name}` | {count} |" for name, count in audit["api_contract_counts"].items())
    lines.extend(
        [
            "",
            "## Complete Inventory / 完整清单",
            "",
            "| Domain | Model | Fidelity | Scope | Execution | Maturity | Benchmark | AutoML | Capabilities | API signals | Migration |",
            "|---|---|---|---|---|---|---|---:|---|---|---|",
        ]
    )
    for model in audit["models"]:
        proposed = model["catalog_v2"]
        signals = ", ".join(name for name, present in model["api_contract"].items() if present)
        supports = ", ".join(model["supports"]) or "none"
        lines.append(
            "| `{domain}` | `{name}` | `{fidelity}` | `{scope}` | `{execution}` | `{maturity}` | "
            "`{benchmark}` | {automl} | {supports} | {signals} | `{migration}` |".format(
                domain=_escape(model["domain"]),
                name=_escape(model["name"]),
                fidelity=_escape(proposed.get("implementation_fidelity", "unmapped")),
                scope=_escape(proposed.get("implementation_scope", "unmapped")),
                execution=_escape(proposed.get("execution_profile", "unmapped")),
                maturity=_escape(proposed["maturity"]),
                benchmark=_escape(proposed["benchmark_evidence"]),
                automl=str(model["automl_default"]).lower(),
                supports=_escape(supports),
                signals=_escape(signals),
                migration=_escape(model["migration_status"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation / 解释边界",
            "",
            "中文：`migration_status=ready` 只表示旧 `integration_level` 能无歧义拆分为 schema v2 字段。模型只有在独立证据登记后才能从 `unassessed` 晋升。",
            "",
            "English: `migration_status=ready` only means the legacy `integration_level` maps unambiguously to the draft schema-v2 fields. Promotion from `unassessed` requires separately recorded evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_committed_outputs(
    audit: dict[str, Any] | None = None,
    json_path: Path = DEFAULT_JSON_PATH,
    markdown_path: Path = DEFAULT_MARKDOWN_PATH,
) -> list[str]:
    audit = audit or build_audit()
    expected = {json_path: render_json(audit), markdown_path: render_markdown(audit)}
    errors = []
    for path, content in expected.items():
        try:
            display_path = path.relative_to(ROOT)
        except ValueError:
            display_path = path
        if not path.is_file():
            errors.append(f"Missing generated catalog audit: {display_path}")
        elif path.read_text(encoding="utf-8") != content:
            errors.append(f"Stale generated catalog audit: {display_path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write deterministic JSON and Markdown audit files")
    mode.add_argument("--check", action="store_true", help="fail when committed audit files differ from the registry")
    args = parser.parse_args()

    audit = build_audit()
    if args.write:
        DEFAULT_JSON_PATH.write_text(render_json(audit), encoding="utf-8")
        DEFAULT_MARKDOWN_PATH.write_text(render_markdown(audit), encoding="utf-8")
        print(f"Wrote {DEFAULT_JSON_PATH.relative_to(ROOT)} and {DEFAULT_MARKDOWN_PATH.relative_to(ROOT)}")
        return 0

    errors = validate_committed_outputs(audit)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print("Run: python tools/audit_model_catalog.py --write")
        return 1
    print(f"Model catalog audit is current: {audit['summary']['model_count']} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
