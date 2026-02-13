"""Tests for scripts/check_sdk_parity.py strict-mode semantics."""

from __future__ import annotations

import sys
from typing import Any

import scripts.check_sdk_parity as check_sdk_parity


def _patch_report(monkeypatch, *, missing: int, py_cov: float = 100.0, ts_cov: float = 100.0) -> None:
    missing_routes = [f"/api/{chr(ord('a') + i)}" for i in range(missing)]
    report: dict[str, Any] = {
        "summary": {
            "python_sdk_coverage_pct": py_cov,
            "typescript_sdk_coverage_pct": ts_cov,
            "routes_missing_from_both_sdks": missing,
        },
        "gaps": {"missing_from_both_sdks": missing_routes},
        "handler_coverage": [],
    }
    monkeypatch.setattr(check_sdk_parity, "extract_handler_routes", lambda: {})
    monkeypatch.setattr(check_sdk_parity, "extract_sdk_paths_python", lambda: {})
    monkeypatch.setattr(check_sdk_parity, "extract_sdk_paths_typescript", lambda: {})
    monkeypatch.setattr(check_sdk_parity, "build_parity_report", lambda *_: report)
    monkeypatch.setattr(check_sdk_parity, "print_report", lambda *_: None)


def test_strict_fails_when_missing_routes_without_override(monkeypatch):
    _patch_report(monkeypatch, missing=3)
    monkeypatch.setattr(sys, "argv", ["check_sdk_parity.py", "--strict"])
    assert check_sdk_parity.main() == 1


def test_strict_allows_missing_routes_with_explicit_override(monkeypatch):
    _patch_report(monkeypatch, missing=3)
    monkeypatch.setattr(sys, "argv", ["check_sdk_parity.py", "--strict", "--allow-missing"])
    assert check_sdk_parity.main() == 0


def test_strict_threshold_still_enforced(monkeypatch):
    _patch_report(monkeypatch, missing=0, py_cov=75.0, ts_cov=88.0)
    monkeypatch.setattr(
        sys, "argv", ["check_sdk_parity.py", "--strict", "--threshold", "90"]
    )
    assert check_sdk_parity.main() == 1


def test_strict_passes_when_missing_routes_are_in_baseline(monkeypatch, tmp_path):
    _patch_report(monkeypatch, missing=2)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        '{"missing_from_both_sdks": ["/api/a", "/api/b"]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_sdk_parity.py", "--strict", "--baseline", str(baseline)],
    )
    assert check_sdk_parity.main() == 0
