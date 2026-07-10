"""Generate markdown fragments from torch_like model registry metadata."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_like.models import list_models  # noqa: E402


FORECAST_LAYER_ORDER = [
    "Transformer / attention",
    "Exogenous forecasting",
    "Dynamical systems forecasting",
    "Frequency-domain forecasting",
    "MLP / decomposition / convolutional mixing",
    "Graph spatiotemporal forecasting",
    "Traffic-specialized dynamic similarity",
    "Probabilistic forecasting",
]

GRAPH_LAYER_ORDER = [
    "General graph learning",
]

SEQUENCE_LAYER_ORDER = [
    "General sequence modeling",
]

ADDITIONAL_DOMAINS = ["detection", "rul", "vision", "multimodal", "generative", "diagnosis"]


def _format_model_list(names: list[str]) -> str:
    return "\n".join(f"- `{name}`" for name in names)


def build_forecasting_layers() -> str:
    models = list_models("forecasting")
    grouped: dict[str, list[str]] = defaultdict(list)
    for name, meta in models.items():
        layer = meta.get("layer")
        if layer in FORECAST_LAYER_ORDER and meta.get("family") != "rul_adapter":
            grouped[layer].append(name)

    parts = []
    for layer in FORECAST_LAYER_ORDER:
        names = sorted(grouped.get(layer, []))
        if not names:
            continue
        parts.append(f"{layer}:\n\n{_format_model_list(names)}")
    return "\n\n".join(parts)


def build_capability_tags() -> str:
    models = list_models("forecasting")
    supports_map: dict[str, list[str]] = defaultdict(list)
    for name, meta in models.items():
        for support in meta.get("supports", []):
            supports_map[support].append(name)

    lines = [
        "- `graph_mode`: " + ", ".join(f"`{name}`" for name in sorted(supports_map.get("graph_mode", []))),
        "- `exogenous`: " + ", ".join(f"`{name}`" for name in sorted(supports_map.get("exogenous", []))),
        "- `sampling`: " + ", ".join(f"`{name}`" for name in sorted(supports_map.get("sampling", []))),
        "- `parametric_prior`: " + ", ".join(f"`{name}`" for name in sorted(supports_map.get("parametric_prior", []))),
    ]
    return "\n".join(lines)


def build_graph_layers() -> str:
    models = list_models("graph")
    grouped: dict[str, list[str]] = defaultdict(list)
    for name, meta in models.items():
        layer = meta.get("layer")
        if layer in GRAPH_LAYER_ORDER:
            grouped[layer].append(name)

    parts = []
    for layer in GRAPH_LAYER_ORDER:
        names = sorted(grouped.get(layer, []))
        if not names:
            continue
        parts.append(f"{layer}:\n\n{_format_model_list(names)}")
    return "\n\n".join(parts)


def build_graph_capability_tags() -> str:
    models = list_models("graph")
    supports_map: dict[str, list[str]] = defaultdict(list)
    for name, meta in models.items():
        for support in meta.get("supports", []):
            supports_map[support].append(name)

    ordered_supports = ["adjacency", "node_task", "graph_task", "attention", "spiking"]
    lines = []
    for support in ordered_supports:
        names = sorted(supports_map.get(support, []))
        if names:
            lines.append(f"- `{support}`: " + ", ".join(f"`{name}`" for name in names))
    return "\n".join(lines)


def build_sequence_layers() -> str:
    models = list_models("sequence")
    grouped: dict[str, list[str]] = defaultdict(list)
    for name, meta in models.items():
        layer = meta.get("layer")
        if layer in SEQUENCE_LAYER_ORDER:
            grouped[layer].append(name)

    parts = []
    for layer in SEQUENCE_LAYER_ORDER:
        names = sorted(grouped.get(layer, []))
        if not names:
            continue
        parts.append(f"{layer}:\n\n{_format_model_list(names)}")
    return "\n\n".join(parts)


def build_sequence_capability_tags() -> str:
    models = list_models("sequence")
    supports_map: dict[str, list[str]] = defaultdict(list)
    for name, meta in models.items():
        for support in meta.get("supports", []):
            supports_map[support].append(name)

    ordered_supports = ["next_token", "generation", "causal_mask", "state_space_like", "fft_mixing", "long_context"]
    lines = []
    for support in ordered_supports:
        names = sorted(supports_map.get(support, []))
        if names:
            lines.append(f"- `{support}`: " + ", ".join(f"`{name}`" for name in names))
    return "\n".join(lines)


def build_domain_layers(domain: str) -> str:
    models = list_models(domain)
    grouped: dict[str, list[str]] = defaultdict(list)
    for name, meta in models.items():
        grouped[meta.get("layer", "Uncategorized")].append(name)
    parts = []
    for layer in sorted(grouped):
        parts.append(f"{layer}:\n\n{_format_model_list(sorted(grouped[layer]))}")
    return "\n\n".join(parts)


def build_domain_capability_tags(domain: str) -> str:
    supports_map: dict[str, list[str]] = defaultdict(list)
    for name, meta in list_models(domain).items():
        for support in meta.get("supports", []):
            supports_map[support].append(name)
    return "\n".join(
        f"- `{support}`: " + ", ".join(f"`{name}`" for name in sorted(names))
        for support, names in sorted(supports_map.items())
    )


def main():
    print("## Forecasting Layers")
    print()
    print(build_forecasting_layers())
    print()
    print("## Forecasting Capability Tags")
    print()
    print(build_capability_tags())
    print()
    print("## Graph Layers")
    print()
    print(build_graph_layers())
    print()
    print("## Graph Capability Tags")
    print()
    print(build_graph_capability_tags())
    print()
    print("## Sequence Layers")
    print()
    print(build_sequence_layers())
    print()
    print("## Sequence Capability Tags")
    print()
    print(build_sequence_capability_tags())
    for domain in ADDITIONAL_DOMAINS:
        layers = build_domain_layers(domain)
        if not layers:
            continue
        title = "RUL" if domain == "rul" else domain.replace("_", " ").title()
        print()
        print(f"## {title} Layers")
        print()
        print(layers)
        capabilities = build_domain_capability_tags(domain)
        if capabilities:
            print()
            print(f"## {title} Capability Tags")
            print()
            print(capabilities)


if __name__ == "__main__":
    main()
