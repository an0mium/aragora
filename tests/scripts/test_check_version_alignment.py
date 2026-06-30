"""Tests for scripts/check_version_alignment.py."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.check_version_alignment as version_alignment


def test_main_reports_runtime_aragora_version(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_version_alignment.py"])

    assert version_alignment.main() == 0

    output = capsys.readouterr().out
    canonical = version_alignment.get_canonical_version()
    assert f"import aragora: {canonical} [runtime] [OK]" in output


def test_main_fails_when_runtime_aragora_version_drifts(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_version_alignment.py"])
    monkeypatch.setattr(
        version_alignment,
        "get_runtime_aragora_version",
        lambda: "0.0.0",
        raising=False,
    )

    assert version_alignment.main() == 1

    output = capsys.readouterr().out
    assert "import aragora: 0.0.0 [runtime] [MISMATCH]" in output
