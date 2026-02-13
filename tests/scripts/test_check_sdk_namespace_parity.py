"""Tests for scripts/check_sdk_namespace_parity.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.check_sdk_namespace_parity as ns_parity


def _patch_report(monkeypatch, routes: list[str]) -> None:
    report = {
        "summary": {"routes_missing_from_both_sdks": len(routes)},
        "gaps": {"missing_from_both_sdks": routes},
    }
    monkeypatch.setattr(ns_parity.sdk_parity, "extract_handler_routes", lambda: {})
    monkeypatch.setattr(ns_parity.sdk_parity, "extract_sdk_paths_python", lambda: {})
    monkeypatch.setattr(ns_parity.sdk_parity, "extract_sdk_paths_typescript", lambda: {})
    monkeypatch.setattr(ns_parity.sdk_parity, "extract_openapi_routes", lambda: set())
    monkeypatch.setattr(ns_parity.sdk_parity, "build_parity_report", lambda *_: report)


def test_strict_fails_when_namespace_regresses(monkeypatch, tmp_path: Path):
    _patch_report(monkeypatch, ["/api/audit/x", "/api/audit/y"])
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"namespaces": {"audit": 1}}) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_sdk_namespace_parity.py", "--strict", "--baseline", str(baseline)],
    )
    assert ns_parity.main() == 1


def test_strict_passes_when_no_namespace_regression(monkeypatch, tmp_path: Path):
    _patch_report(monkeypatch, ["/api/audit/x"])
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"namespaces": {"audit": 1}}) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_sdk_namespace_parity.py", "--strict", "--baseline", str(baseline)],
    )
    assert ns_parity.main() == 0
