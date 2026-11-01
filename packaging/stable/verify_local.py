"""Locally validate the standalone algorithms wheel without GitHub Actions.

Requires preinstalled build, torch, NumPy and scikit-learn. No downloads are
performed by this script. Creates an ephemeral virtual environment that can
see system packages but not user site packages, installs only the candidate
wheel, and executes the exact installed-wheel smoke from outside the repo.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "packaging" / "stable" / "smoke_test.py"


def _call(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("stable-v0.3.0-local-report.json"))
    args = parser.parse_args()

    missing = [name for name in ("build", "torch", "numpy", "sklearn") if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit("Install required packages in the current Python environment: " + ", ".join(missing))

    with tempfile.TemporaryDirectory(prefix="torch-like-stable-") as work:
        root = Path(work)
        stage = root / "stage"
        wheels = root / "wheels"
        env = root / "venv"
        _call([sys.executable, "-m", "compileall", "-q", str(REPO / "packaging" / "stable")], cwd=root)
        _call(
            [sys.executable, str(REPO / "packaging" / "stable" / "prepare.py"), "--destination", str(stage)],
            cwd=root,
        )
        manifest = json.loads((stage / "stable-manifest.json").read_text(encoding="utf-8"))
        if manifest["version"] != "0.3.0":
            raise AssertionError("expected version 0.3.0; candidate configuration is inconsistent")
        _call([sys.executable, "-m", "compileall", "-q", str(stage / "src")], cwd=root)
        _call(
            [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(wheels), str(stage)],
            cwd=root,
        )
        wheel_files = list(wheels.glob("torch_like_algorithms-0.3.0-*.whl"))
        if len(wheel_files) != 1:
            raise AssertionError("expected exactly one v0.3.0 wheel")
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(env)
        interpreter = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        _call([str(interpreter), "-m", "pip", "install", "--no-deps", str(wheel_files[0])], cwd=root)
        _call([str(interpreter), "-I", str(SMOKE)], cwd=root)
        report = {
            "version": "0.3.0",
            "status": "pass",
            "source_modules": manifest["source_modules"],
            "verification": "staged_manifest+wheel_build+installed_wheel_smoke",
            "disclaimer": "system-site-packages reused; not a fully dependency-isolated environment",
        }

    destination = args.report.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("PASS: stable v0.3.0 staged wheel and installed smoke", destination)


if __name__ == "__main__":
    main()
