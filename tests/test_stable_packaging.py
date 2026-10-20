"""Algorithm-only stable source allowlist and wheel staging regression tests."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

STAGING_SCRIPT = Path(__file__).resolve().parents[1] / "packaging" / "stable" / "prepare.py"
SPEC = importlib.util.spec_from_file_location("torch_like_stable_build_profile", STAGING_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
build_profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_profile)
FORBIDDEN = build_profile.FORBIDDEN
REPO = build_profile.REPO
STABLE_MODULES = build_profile.STABLE_MODULES
prepare = build_profile.prepare


class StableDistributionTests(unittest.TestCase):
    def test_staging_contains_only_explicit_algorithm_sources_and_api_facades(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "stable"
            metadata = prepare(stage)
            self.assertEqual(metadata["scope"], "algorithms_and_interfaces_only")
            self.assertEqual(metadata["distribution"], "torch-like-algorithms")
            self.assertEqual(metadata["import_package"], "torch_like_stable")
            self.assertFalse(metadata["dataset_source_audit"])
            self.assertFalse(metadata["automl_included"])
            self.assertFalse(metadata["runner_included"])
            self.assertFalse(metadata["evaluation_platform_included"])
            files = {p.relative_to(stage).as_posix() for p in stage.rglob("*.py")}
            self.assertEqual(files, set(metadata["python_files"]))
            self.assertEqual(len(files), len(STABLE_MODULES) + 4)
            for module in STABLE_MODULES:
                filename = "src/torch_like_stable/" + module
                self.assertIn(filename, files)
                source = (REPO / "torch_like" / module).read_text(encoding="utf-8")
                actual = (stage / filename).read_text(encoding="utf-8")
                self.assertEqual(actual, source.replace("torch_like.", "torch_like_stable."))
                self.assertNotIn("from torch_like.", actual)
            for path in FORBIDDEN:
                self.assertFalse(any(p == "src/" + path or p.startswith("src/" + path + "/") for p in files))
            self.assertTrue((stage / "README.md").exists())
            self.assertTrue((stage / "LICENSE").exists())
            manifest = json.loads((stage / "stable-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest, metadata)
            self.assertIn('name = "torch-like-algorithms"', (stage / "pyproject.toml").read_text())
            self.assertNotIn("torch-like = ", (stage / "pyproject.toml").read_text())

    def test_identical_build_manifest_and_missing_module_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "stable"
            first = prepare(stage)
            second = prepare(stage)
            self.assertEqual(first, second)
            self.assertEqual(
                (stage / "stable-manifest.json").read_text(encoding="utf-8"),
                (stage / "stable-manifest.json").read_text(encoding="utf-8"),
            )
            with patch.object(build_profile, "STABLE_MODULES", STABLE_MODULES + ("missing.py",)):
                with self.assertRaisesRegex(FileNotFoundError, "missing.py"):
                    prepare(stage)


if __name__ == "__main__":
    unittest.main()
