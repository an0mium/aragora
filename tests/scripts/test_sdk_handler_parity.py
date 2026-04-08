"""Tests for scripts/sdk_handler_parity.py path handling."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.sdk_handler_parity as sdk_handler_parity


def test_load_sdk_endpoints_reads_repo_sdk_roots(monkeypatch):
    seen_roots: list[Path] = []
    expected_ts_root = PROJECT_ROOT / "sdk" / "typescript" / "src"
    expected_py_root = PROJECT_ROOT / "sdk" / "python" / "aragora_sdk" / "namespaces"

    def fake_iter_files(root: Path, suffix: str):
        seen_roots.append(root)
        return []

    monkeypatch.setattr(sdk_handler_parity, "iter_files", fake_iter_files)
    monkeypatch.setattr(sdk_handler_parity, "parse_ts_sdk", lambda paths: set())
    monkeypatch.setattr(sdk_handler_parity, "parse_python_sdk", lambda paths: set())

    sdk_handler_parity.load_sdk_endpoints()

    assert seen_roots == [expected_ts_root, expected_py_root]


def test_main_writes_report_under_project_root(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sdk_handler_parity, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sdk_handler_parity, "load_sdk_endpoints", lambda: (set(), set()))

    sdk_handler_parity.main()

    report_path = tmp_path / "docs" / "SDK_HANDLER_PARITY.md"
    assert report_path.exists()
    assert "# SDK ↔ Handler Parity Report" in report_path.read_text(encoding="utf-8")
