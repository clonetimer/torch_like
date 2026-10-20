# torch-like-algorithms — standalone stable algorithm distribution

This is an **independently installable algorithms-and-interfaces wheel** assembled
from the `clonetimer/torch_like` source tree, not a rename of the full framework.

- Distribution: `torch-like-algorithms`, version `0.2.0` (initial stable delivery track).
- Import root: `torch_like_stable` — **does not shadow or overwrite**
  `import torch_like`. Both distributions can be installed together.
- Core runtime requirements: Python ≥3.10, NumPy, PyTorch, scikit-learn.
- Not included: AutoML, training/experiment orchestration, CLI, runner,
  data-source/provenance audits, dataset ingestion, sealed-test management,
  benchmark platform, reporting dashboards, generalization/uncertainty workflows,
  unrelated forecasting, vision, language and generative model families.

## Public algorithms and interfaces

| Capability | Classes and interfaces |
|---|---|
| Input/output contracts | `SignalBatch`, `DiagnosticResult`, `DiagnosticAlgorithmSpec`, `DiagnosticTask` |
| Statistical anomaly detection | `ZScoreDetector`, `CUSUMDetector`, `EWMADetector`, `PCAT2Detector`, `PCASPEDetector` |
| Signal features | `time_domain_features`, `fft_amplitude`, `welch_psd`, `WindowFeatureTransformer` |
| Traditional machine learning | `SklearnWindowAnomalyDetector`, `SklearnWindowFaultClassifier`, native sklearn `classical.create_estimator` |
| Native time-series fault diagnosis | `InceptionTime`, `ROCKET`, `TorchWindowFaultClassifier` |
| Health estimation | `GaussianHealthIndex`, `MahalanobisHealthIndex` |
| Transparent fault candidate mapping | `FaultSourceRulebook`, `FaultSourceEvidenceMapper` |
| Diagnostic events | `extract_anomaly_events` |
| Native RUL models | `TCNRUL`, `PatchTSTRUL`, `TransformerRUL`, `RULBatch` |
| Explicit composition | `DiagnosticStage`, `DiagnosticPipeline`, `DiagnosticBundleResult` |
| Native RUL bridge | `RULRegressor`, `RULForecast`, `create_rul_regressor`, `list_rul_models` |
| Algorithm discovery | `DiagnosticRegistry`, opt-in registration and construction functions |

The **three** RUL backbones above are the self-contained models shipped in
this initial slim wheel. The full repository has seven registered RUL entries,
including forecasting-based adapters that are intentionally omitted from
the standalone delivery to avoid bringing in unrelated domains. Do not
represent the slim wheel as containing every model of the full framework.

## Installation

The stable GitHub Actions workflow uploads a wheel as a build artifact.
After retrieving the artifact:

```sh
python -m pip install torch_like_algorithms-0.2.0-py3-none-any.whl
```

Imports work independently of the full `torch_like` package:

```python
import numpy as np
from torch_like_stable import SignalBatch, GaussianHealthIndex, ZScoreDetector
from torch_like_stable.models import TCNRUL
from torch_like_stable.diagnostics import (
    FaultLabelSchema,
    SklearnWindowFaultClassifier,
)

healthy = SignalBatch(
    values=np.ones((3, 10, 2)),
    sample_rate_hz=100,
    channel_names=("s1", "s2"),
)
incoming = SignalBatch(
    values=np.ones((1, 10, 2)) * 1.05,
    sample_rate_hz=100,
    channel_names=("s1", "s2"),
)
hi = GaussianHealthIndex().fit(healthy).score(incoming).values
anomaly = ZScoreDetector().fit(healthy).score(incoming).values
```

**v0.2.0 change:** adds native diagnostic composition and RUL adapters.
Pipeline stages do not automatically fuse scores or calibrate policies;
RUL targets, time/cycle units and ready-state declarations are caller-owned.
Full model selection is available in the original framework, while the slim
wheel still ships three self-contained RUL architectures.

The library **does not validate or certify the user's data sources**.
Reference choice, label truth, physical provenance, data cleaning and application
health/failure thresholds belong to the caller. Checks for finite numeric
inputs, shape, units and fitted metadata are algorithm interface checks only.

## Release reproducibility

```sh
python packaging/stable/prepare.py --destination build/stable
python -m pip install build
python -m build --wheel --outdir dist/stable build/stable
```

The stable profile has a strict source-file allowlist in
`packaging/stable/prepare.py` and emits `stable-manifest.json` recording
which original implementation modules were included. Namespaces are rebased
only in the staged build directory; the original algorithm implementations
are not modified.

The dedicated CI workflow validates staged file membership, builds and
installs the wheel, tests numerical model imports in an isolated environment,
and uploads the exact wheel and manifest. A passing main-repository Quality
workflow alone does **not** certify the separate stable wheel.
