"""Verify pinned AAS Phase 9 algorithms against current torch_like implementations.

Optional integration verifier: explicitly pass a locally trusted AAS source
directory. Nothing from AAS is bundled or imported during normal library use.

Usage:
    python tools/interop/verify_aas_phase9.py --aas-root /path/to/AAS
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from torch_like.diagnostics import (
    CUSUMDetector,
    EWMADetector,
    PCASPEDetector,
    SignalBatch,
    project_anomaly_scores,
)

UPSTREAM_AAS_REVISION = "1e603915896e33fd9f326073c42a477718166b68"
INPUT_SEED = 2060
STAT_TOLERANCE = {"rtol": 1e-9, "atol": 1e-9}
NEURAL_TOLERANCE = {"rtol": 2e-6, "atol": 2e-6}


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(INPUT_SEED)
    reference = generator.normal(size=(30, 3))
    continuous = generator.normal(size=(12, 3))
    continuous[3:5] += [2.0, -1.0, 0.5]
    continuous[9:11] += [0.0, 3.0, -0.5]
    windows = generator.normal(size=(3, 4, 3))
    windows[0, -1, :] += [2.0, -2.0, 1.0]
    return reference, continuous, windows


def _batch(values: np.ndarray) -> SignalBatch:
    if values.ndim == 2:
        return SignalBatch.from_sequence(values, sample_rate_hz=10)
    return SignalBatch(values, sample_rate_hz=10)


def _assert_parity(name: str, actual: np.ndarray, expected: np.ndarray, **tolerance: float) -> dict[str, object]:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape mismatch {actual.shape} vs {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError(f"{name}: nonfinite scores")
    np.testing.assert_allclose(actual, expected, **tolerance, err_msg=name)
    return {"case": name, "shape": list(actual.shape), "max_abs_error": float(np.max(np.abs(actual - expected)))}


def _verify_statistical(aas_classes: tuple[type, type, type]) -> list[dict[str, object]]:
    aas_cusum, aas_ewma, aas_pca = aas_classes
    reference, continuous, windows = _fixture()
    results: list[dict[str, object]] = []
    configurations = (
        ("CUSUM", aas_cusum, CUSUMDetector, {"k": 0.6, "h": 4.5, "min_std": 1e-8}, "time_channel", "max"),
        ("EWMA", aas_ewma, EWMADetector, {"alpha": 0.3, "control_limit": 2.5, "min_std": 1e-8}, "time_channel", "max"),
        ("PCA-SPE", aas_pca, PCASPEDetector, {"n_components": 2, "contamination": 0.1, "min_std": 1e-8}, "time", None),
    )
    for name, aas_type, torch_type, params, layout, reduction in configurations:
        aas = aas_type(params)
        aas.fit(reference)
        upstream_params = {key: value for key, value in params.items() if key != "contamination"}
        upstream = torch_type(**upstream_params).fit(_batch(reference))
        if name != "PCA-SPE":
            aas.set_segments([5, 7])
        expected_time = np.asarray(aas.score(continuous), dtype=float)
        if name != "PCA-SPE":
            aas.set_segments(None)
        time_scores = []
        for start, stop in ((0, 5), (5, 12)) if name != "PCA-SPE" else ((0, 12),):
            raw = upstream.score(_batch(continuous[start:stop]))
            projected = project_anomaly_scores(
                raw, score_layout=layout, sequence_mode="continuous", channel_reduction=reduction
            )
            time_scores.extend(projected.values)
        results.append(_assert_parity(f"{name}/continuous-run-separated", time_scores, expected_time, **STAT_TOLERANCE))
        expected_windows = np.asarray(aas.score(windows), dtype=float)
        raw_windows = upstream.score(_batch(windows))
        projected_windows = project_anomaly_scores(
            raw_windows, score_layout=layout, sequence_mode="independent_windows", channel_reduction=reduction
        )
        results.append(
            _assert_parity(f"{name}/independent-windows", projected_windows.values, expected_windows, **STAT_TOLERANCE)
        )
    return results


def _remap_usad_state(state: dict[str, object]) -> dict[str, object]:
    """Strict, lossless key remapping; not an automatic AAS checkpoint loader."""
    mapped: dict[str, object] = {}
    for key, value in state.items():
        if key.startswith(("decoder_1.", "decoder_2.")):
            name, suffix = key.split(".", 1)
            new_key = f"{name}.net.{suffix}"
        else:
            new_key = key
        if new_key in mapped:
            raise ValueError(f"USAD checkpoint key collision: {new_key}")
        mapped[new_key] = value
    return mapped


def _verify_neural(aas_neural_module: object) -> list[dict[str, object]]:
    import torch

    from torch_like.models.detection.common import DetectionBatch
    from torch_like.models.detection.tranad import TranAD
    from torch_like.models.detection.usad import USAD

    torch.manual_seed(70)
    x = torch.linspace(-1, 1, 48, dtype=torch.float32).reshape(4, 4, 3)
    cases = (
        (
            "USAD",
            aas_neural_module._USADModel.create(3, 4, 12, 5, 0.0, 0.45),
            USAD(input_dim=3, seq_len=4, hidden_dim=12, latent_dim=5, dropout=0.0, alpha=0.45),
            True,
        ),
        (
            "TranAD",
            aas_neural_module._TranADModel.create(3, 4, 8, 2, 1, 16, 0.0),
            TranAD(input_dim=3, seq_len=4, d_model=8, n_heads=2, e_layers=1, d_ff=16, dropout=0.0),
            False,
        ),
    )
    results = []
    for name, aas, upstream, remap in cases:
        aas_state = aas.state_dict()
        upstream_state = _remap_usad_state(aas_state) if remap else dict(aas_state)
        expected_keys = upstream.state_dict()
        if set(upstream_state) != set(expected_keys):
            raise AssertionError(f"{name}: state_dict keys differ after explicit adapter mapping")
        for key in upstream_state:
            if tuple(upstream_state[key].shape) != tuple(expected_keys[key].shape):
                raise AssertionError(f"{name}: state_dict tensor shape mismatch for {key}")
        upstream.load_state_dict(upstream_state, strict=True)
        aas.eval()
        upstream.eval()
        with torch.inference_mode():
            aas_scores = aas.score_batch(x).detach().cpu().numpy()
            upstream_scores = upstream.score_batch(DetectionBatch(x)).detach().cpu().numpy()
            # fit_step uses model training semantics but dropout is disabled.
            aas_loss = aas.fit_step(x).item()
            upstream_loss = upstream.fit_step(DetectionBatch(x)).item()
        results.append(_assert_parity(f"{name}/mapped-weights-score", upstream_scores, aas_scores, **NEURAL_TOLERANCE))
        results.append(
            _assert_parity(f"{name}/mapped-weights-training-loss", [upstream_loss], [aas_loss], **NEURAL_TOLERANCE)
        )
    return results


def verify(aas_root: Path) -> dict[str, object]:
    root = aas_root.resolve(strict=True)
    if not (root / "detectors" / "torch_like_adapters.py").is_file():
        raise ValueError("--aas-root must point to the trusted AAS project directory containing detectors/")
    sys.path.insert(0, str(root))
    try:
        from detectors import torch_like_neural
        from detectors.torch_like_adapters import (
            TorchLikeCUSUMDetector,
            TorchLikeEWMADetector,
            TorchLikePCASPEDetector,
        )

        statistical = _verify_statistical((TorchLikeCUSUMDetector, TorchLikeEWMADetector, TorchLikePCASPEDetector))
        neural = _verify_neural(torch_like_neural)
        return {
            "protocol": "explicit_aas_phase9_numeric_and_mapped_weight_parity_v1",
            "aas_upstream_reference_revision": UPSTREAM_AAS_REVISION,
            "input_seed": INPUT_SEED,
            "passed": len(statistical) + len(neural),
            "statistical": statistical,
            "neural": neural,
            "scope": "scoring and fit-step on matched weights only; not AAS checkpoint policy or end-to-end migration",
        }
    finally:
        sys.path.remove(str(root))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aas-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify(args.aas_root)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
