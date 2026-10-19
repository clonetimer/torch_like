import subprocess
import sys
import textwrap
import unittest


class LazyImportTests(unittest.TestCase):
    def run_python(self, code: str) -> str:
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def test_plain_import_does_not_eager_load_forecasting_domain(self):
        code = textwrap.dedent(
            """
            import sys
            import torch_like

            print("torch_like.models.forecasting" in sys.modules)
            _ = torch_like.PatchTST
            print("torch_like.models.forecasting" in sys.modules)
            """
        )
        output = self.run_python(code).splitlines()
        self.assertEqual(output, ["False", "True"])

    def test_lazy_symbols_import_from_torch_like_models(self):
        code = textwrap.dedent(
            """
            from torch_like.models import PatchTST, THGNN, LRU

            print(PatchTST.__name__)
            print(THGNN.__name__)
            print(LRU.__name__)
            """
        )
        output = self.run_python(code).splitlines()
        self.assertEqual(output, ["PatchTST", "THGNN", "LRU"])

    def test_model_zoo_bootstraps_on_first_mapping_access(self):
        code = textwrap.dedent(
            """
            import sys
            import torch_like

            print("torch_like.models.forecasting" in sys.modules)
            print(len(torch_like.model_zoo["forecasting"]) > 0)
            print("torch_like.models.forecasting" in sys.modules)
            """
        )
        output = self.run_python(code).splitlines()
        self.assertEqual(output, ["False", "True", "True"])

    def test_evaluation_namespace_remains_lazy_until_requested(self):
        code = textwrap.dedent(
            """
            import sys
            import torch_like as tl

            print("torch_like.evaluation" in sys.modules)
            _ = tl.evaluation.MetricEvaluator
            print("torch_like.evaluation" in sys.modules)
            """
        )
        output = self.run_python(code).splitlines()
        self.assertEqual(output, ["False", "True"])

    def test_training_and_evaluation_do_not_import_sklearn_until_metric_helpers_are_called(self):
        code = textwrap.dedent(
            """
            import sys
            import torch_like.training
            import torch_like.evaluation

            print(any(name == "sklearn" or name.startswith("sklearn.") for name in sys.modules))
            """
        )
        output = self.run_python(code).splitlines()
        self.assertEqual(output, ["False"])


if __name__ == "__main__":
    unittest.main()
