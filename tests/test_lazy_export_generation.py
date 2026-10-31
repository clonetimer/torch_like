"""Lazy index checks ignore formatting while preserving every Python AST value."""

import sys

import pytest

from tools import generate_lazy_exports

GENERATED = "PUBLIC_EXPORTS = {'FITS': ('torch_like.models.forecasting.fits', 'FITS')}\n"


@pytest.fixture
def export_file(tmp_path, monkeypatch):
    path = tmp_path / "_public_exports.py"
    monkeypatch.setattr(generate_lazy_exports, "ROOT", tmp_path)
    monkeypatch.setattr(generate_lazy_exports, "generate", lambda: {path: GENERATED})
    monkeypatch.setattr(sys, "argv", ["generate_lazy_exports.py", "--check"])
    return path


def test_check_accepts_quote_and_layout_changes_without_rewriting(export_file, capsys):
    formatted = 'PUBLIC_EXPORTS = {\n    "FITS": ("torch_like.models.forecasting.fits", "FITS"),\n}\n'
    export_file.write_text(formatted, encoding="utf-8")

    assert generate_lazy_exports.main() == 0
    assert "Lazy export indexes are current" in capsys.readouterr().out
    assert export_file.read_text(encoding="utf-8") == formatted


@pytest.mark.parametrize(
    "existing",
    [
        "PUBLIC_EXPORTS = {'FITS': ('torch_like.models.forecasting.fits', 'WrongClass')}\n",
        "PUBLIC_EXPORTS = {'FITS': (\n",
        None,
    ],
    ids=["changed-export", "syntax-error", "missing-file"],
)
def test_check_reports_changed_invalid_or_missing_indexes_as_stale(export_file, capsys, existing):
    if existing is not None:
        export_file.write_text(existing, encoding="utf-8")

    assert generate_lazy_exports.main() == 1
    assert capsys.readouterr().out == "Lazy export indexes are stale:\n- _public_exports.py\n"


def test_default_generation_preserves_rendered_output(export_file, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["generate_lazy_exports.py"])

    assert generate_lazy_exports.main() == 0
    assert export_file.read_text(encoding="utf-8") == GENERATED
    assert capsys.readouterr().out == "wrote _public_exports.py\n"
