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


if __name__ == "__main__":
    unittest.main()
