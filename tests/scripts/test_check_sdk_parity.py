"""Tests for scripts/check_sdk_parity.py strict-mode semantics."""

from __future__ import annotations

import sys
from typing import Any

import scripts.check_sdk_parity as check_sdk_parity


def _patch_report(
    monkeypatch,
    *,
    missing: int,
    py_cov: float = 100.0,
    ts_cov: float = 100.0,
    stale_python: int = 0,
) -> None:
    missing_routes = [f"/api/{chr(ord('a') + i)}" for i in range(missing)]
    stale_python_routes = [f"/api/stale/{i}" for i in range(stale_python)]
    report: dict[str, Any] = {
        "summary": {
            "python_sdk_coverage_pct": py_cov,
            "typescript_sdk_coverage_pct": ts_cov,
            "routes_missing_from_both_sdks": missing,
        },
        "gaps": {
            "missing_from_both_sdks": missing_routes,
            "stale_python_sdk_paths": stale_python_routes,
        },
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


def test_strict_budget_fails_when_missing_exceeds_budget(monkeypatch, tmp_path):
    _patch_report(monkeypatch, missing=3, stale_python=1)
    budget = tmp_path / "budget.json"
    budget.write_text(
        """
{
  "start_date": "2026-01-01",
  "initial_missing_from_both_sdks": 2,
  "weekly_reduction_missing_from_both_sdks": 0,
  "initial_stale_python_sdk_paths": 1,
  "weekly_reduction_stale_python_sdk_paths": 0
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_sdk_parity.py",
            "--strict",
            "--allow-missing",
            "--budget",
            str(budget),
            "--today",
            "2026-02-13",
        ],
    )
    assert check_sdk_parity.main() == 1


def test_strict_budget_fails_when_stale_exceeds_budget(monkeypatch, tmp_path):
    _patch_report(monkeypatch, missing=0, stale_python=5)
    budget = tmp_path / "budget.json"
    budget.write_text(
        """
{
  "start_date": "2026-01-01",
  "initial_missing_from_both_sdks": 0,
  "weekly_reduction_missing_from_both_sdks": 0,
  "initial_stale_python_sdk_paths": 4,
  "weekly_reduction_stale_python_sdk_paths": 0
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_sdk_parity.py",
            "--strict",
            "--allow-missing",
            "--budget",
            str(budget),
            "--today",
            "2026-02-13",
        ],
    )
    assert check_sdk_parity.main() == 1


def test_strict_budget_passes_when_within_budget(monkeypatch, tmp_path):
    _patch_report(monkeypatch, missing=2, stale_python=10)
    budget = tmp_path / "budget.json"
    budget.write_text(
        """
{
  "start_date": "2026-01-01",
  "initial_missing_from_both_sdks": 4,
  "weekly_reduction_missing_from_both_sdks": 1,
  "initial_stale_python_sdk_paths": 20,
  "weekly_reduction_stale_python_sdk_paths": 2
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_sdk_parity.py",
            "--strict",
            "--allow-missing",
            "--budget",
            str(budget),
            "--today",
            "2026-02-13",
        ],
    )
    assert check_sdk_parity.main() == 0


def test_extract_openapi_routes_normalizes_versioned_paths(tmp_path):
    spec = tmp_path / "openapi.json"
    spec.write_text(
        """
{
  "paths": {
    "/api/v1/alpha/{id}": {"get": {"summary": "x"}},
    "/api/v1/beta": {"post": {"summary": "y"}},
    "/not-http": {"x-meta": {}}
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    routes = check_sdk_parity.extract_openapi_routes(spec)
    assert "/api/alpha/{id}" in routes
    assert "/api/beta" in routes
