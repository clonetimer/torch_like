import subprocess
import sys
import textwrap
import unittest

import numpy as np

import torch_like as tl


class EngineeringAPIContractTests(unittest.TestCase):
    def run_python(self, code: str) -> list[str]:
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip().splitlines()

    def test_model_names_are_discoverable_without_loading_domain_implementations(self):
        output = self.run_python(
            """
            import sys
            import torch_like as tl
            print("torch_like.models.forecasting" in sys.modules)
            print("PatchTST" in tl.list_model_names("forecasting"))
            print("torch_like.models.forecasting" in sys.modules)
            """
        )
        self.assertEqual(output, ["False", "True", "False"])

    def test_direct_model_shortcut_loads_only_requested_implementation(self):
        output = self.run_python(
            """
            import sys
            import torch_like as tl
            _ = tl.PatchTST
            print("torch_like.models.forecasting.patchtst" in sys.modules)
            print("torch_like.models.forecasting.autoformer" in sys.modules)
            print("torch_like.models.forecasting._catalog" in sys.modules)
            print("torch_like.models.vision" in sys.modules)
            print(sum(name.startswith("torch_like.models.") for name in sys.modules) < 30)
            """
        )
        self.assertEqual(output, ["True", "False", "False", "False", "True"])

    def test_sklearn_compatibility_is_lazy_and_native(self):
        output = self.run_python(
            """
            import sys
            import torch_like as tl
            print("sklearn" in sys.modules)
            print(tl.get_estimator_info("RandomForestClassifier")["backend"])
            print("sklearn" in sys.modules)
            from torch_like.classical import RandomForestClassifier
            from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
            print(RandomForestClassifier is SklearnRandomForestClassifier)
            """
        )
        self.assertEqual(output, ["False", "sklearn", "False", "True"])

    def test_backend_aware_factory_preserves_native_object_models(self):
        torch_model = tl.create(
            "DLinear",
            enc_in=2,
            dec_in=2,
            c_out=2,
            seq_len=8,
            label_len=4,
            pred_len=2,
        )
        self.assertEqual(torch_model.__class__.__name__, "DLinear")

        estimator = tl.create("RandomForestClassifier", backend="sklearn", n_estimators=5, random_state=0)
        x = np.asarray([[0.0], [1.0], [0.1], [0.9]])
        y = np.asarray([0, 1, 0, 1])
        estimator.fit(x, y)
        self.assertEqual(estimator.predict([[0.05]]).shape, (1,))

    def test_find_model_domain_and_validation(self):
        self.assertEqual(tl.find_model_domain("EGNN"), "graph")
        with self.assertRaises(KeyError):
            tl.find_model_domain("DefinitelyMissingModel")
        with self.assertRaisesRegex(ValueError, "domain is only valid"):
            tl.create("Ridge", backend="sklearn", domain="forecasting")


if __name__ == "__main__":
    unittest.main()
