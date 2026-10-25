"""Native C ABI v1 integration smoke with a compiled C fixture (no AAS import)."""

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from torch_like.diagnostics import SignalBatch
from torch_like.native import ABI_VERSION, NativeAnomalyDetector

ROOT = Path(__file__).resolve().parents[1]
HEADER_DIR = ROOT / "torch_like" / "native" / "include"
DEMO = ROOT / "examples" / "native" / "demo_mean_deviation.c"


def build_fixture(destination, *, bad_version=False):
    compiler = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    if compiler is None:
        raise unittest.SkipTest("no C compiler available for native smoke test")
    suffix = ".dll" if os.name == "nt" else (".dylib" if sys.platform == "darwin" else ".so")
    output = Path(destination) / ("demo_bad" if bad_version else "demo")
    output = output.with_suffix(suffix)
    args = [
        compiler, "-std=c11", "-O2", "-shared", "-I", str(HEADER_DIR),
        str(DEMO), "-o", str(output),
    ]
    if os.name != "nt":
        args.insert(3, "-fPIC")
        args.append("-lm")
    if bad_version:
        args.insert(1, "-DTL_FORCE_BAD_VERSION")
    subprocess.run(args, check=True, capture_output=True, text=True)
    return output


class NativeAbiTests(unittest.TestCase):
    def test_c_header_declares_version_and_caller_owned_buffers(self):
        text = (HEADER_DIR / "torch_like_native_anomaly_v1.h").read_text(encoding="utf-8")
        self.assertIn("TL_NATIVE_ANOMALY_ABI_VERSION 1u", text)
        self.assertIn("const double *values", text)
        self.assertIn("double *scores", text)
        self.assertIn('extern "C"', text)

    def test_compile_fit_score_restore_and_close(self):
        self.assertEqual(ABI_VERSION, 1)
        with tempfile.TemporaryDirectory() as directory:
            library = build_fixture(directory)
            reference = SignalBatch(
                np.array([[[1., 2.], [3., 4.]]]), sample_rate_hz=10,
                channel_names=("a", "b"), units=("A", "B"),
            )
            new = SignalBatch(
                np.array([[[2., 3.], [5., 1.]]]), sample_rate_hz=10,
                channel_names=("a", "b"), units=("A", "B"),
            )
            with NativeAnomalyDetector(
                library, algorithm_name="demo_mean_dev", config={"seed": 5}
            ) as model:
                with self.assertRaisesRegex(RuntimeError, "fit"):
                    model.score(new)
                self.assertIs(model.fit(reference), model)
                result = model.score(new)
                np.testing.assert_allclose(result.values, [[[0.5, 0.5], [2.5, 1.5]]])
                self.assertEqual(result.task.value, "anomaly_detection")
                snapshot = json.loads(json.dumps(model.export_state()))
                with self.assertRaisesRegex(ValueError, "metadata"):
                    model.score(SignalBatch(
                        np.ones((1, 2, 2)), sample_rate_hz=10,
                        channel_names=("b", "a"), units=("A", "B"),
                    ))
            with self.assertRaisesRegex(RuntimeError, "closed"):
                model.score(new)
            with NativeAnomalyDetector(
                library, algorithm_name="demo_mean_dev", config={"seed": 5}
            ) as restored:
                restored.import_state(snapshot)
                np.testing.assert_allclose(restored.score(new).values, result.values)
                before = restored.export_state()
                invalid = copy.deepcopy(snapshot)
                invalid["abi_version"] = 2
                with self.assertRaisesRegex(ValueError, "version"):
                    restored.import_state(invalid)
                invalid = copy.deepcopy(snapshot)
                invalid["config_json"] = "{}"
                with self.assertRaisesRegex(ValueError, "configuration"):
                    restored.import_state(invalid)
                invalid = copy.deepcopy(snapshot)
                invalid["plugin_state_base64"] = "%%%invalid%%%"
                with self.assertRaises(ValueError):
                    restored.import_state(invalid)
                self.assertEqual(restored.export_state(), before)

    def test_version_handshake_rejects_incompatible_library(self):
        with tempfile.TemporaryDirectory() as directory:
            library = build_fixture(directory, bad_version=True)
            with self.assertRaisesRegex(ValueError, "version"):
                NativeAnomalyDetector(library, algorithm_name="bad")


if __name__ == "__main__":
    unittest.main()
