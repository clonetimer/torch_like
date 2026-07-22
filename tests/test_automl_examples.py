from pathlib import Path

from tools.validate_automl_examples import validate_examples
from torch_like.automl import LeaderboardTrustLevel


def test_automl_example_configs_are_governed():
    results = validate_examples()
    errors = [(result.path, error) for result in results for error in result.errors]

    assert errors == []


def test_validated_forecasting_example_is_explicitly_trusted():
    results = {result.path.name: result for result in validate_examples()}
    result = results["forecasting_rolling_validated.yaml"]

    assert result.config is not None
    assert result.config.trust.level == LeaderboardTrustLevel.VALIDATED
    assert result.config.search.evaluation_seeds == (11, 29, 47)
    assert Path(result.path).name.startswith("forecasting_rolling")
