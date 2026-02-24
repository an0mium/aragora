"""Tests for scripts/check_sdk_parity.py strict-mode semantics."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def test_strict_allows_missing_routes_with_explicit_override(monkeypatch, tmp_path):
    _patch_report(monkeypatch, missing=3)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_sdk_parity.py",
            "--strict",
            "--allow-missing",
            "--budget",
            str(tmp_path / "no-budget.json"),
        ],
    )
    assert check_sdk_parity.main() == 0


def test_strict_threshold_still_enforced(monkeypatch):
    _patch_report(monkeypatch, missing=0, py_cov=75.0, ts_cov=88.0)
    monkeypatch.setattr(sys, "argv", ["check_sdk_parity.py", "--strict", "--threshold", "90"])
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
        [
            "check_sdk_parity.py",
            "--strict",
            "--baseline",
            str(baseline),
            "--budget",
            str(tmp_path / "no-budget.json"),
        ],
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
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        '{"missing_from_both_sdks": ["/api/a", "/api/b"]}\n',
        encoding="utf-8",
    )
    budget = tmp_path / "budget.json"
    budget.write_text(
        """
{
  "start_date": "2026-02-13",
  "initial_missing_from_both_sdks": 2,
  "weekly_reduction_missing_from_both_sdks": 1,
  "initial_stale_python_sdk_paths": 10,
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
            "--baseline",
            str(baseline),
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
    assert "/api/alpha/{param}" in routes
    assert "/api/beta" in routes


def test_stale_detection_uses_documented_routes_for_dispatch_handlers():
    handler_routes = {"SomeHandler": ["/api/v1/debates"]}
    python_sdk = {"moderation": {"/api/v1/moderation/config"}}
    typescript_sdk: dict[str, set[str]] = {}
    documented_routes = {check_sdk_parity.normalize_route("/api/v1/moderation/config")}

    report_without_docs = check_sdk_parity.build_parity_report(
        handler_routes, python_sdk, typescript_sdk, documented_routes=None
    )
    report_with_docs = check_sdk_parity.build_parity_report(
        handler_routes, python_sdk, typescript_sdk, documented_routes=documented_routes
    )

    assert "/api/moderation/config" in report_without_docs["gaps"]["stale_python_sdk_paths"]
    assert "/api/moderation/config" not in report_with_docs["gaps"]["stale_python_sdk_paths"]


def test_collect_routes_includes_dynamic_and_route_map_entries():
    class DummyHandler:
        ROUTES = ["/api/v1/static", "GET /api/v1/method-route"]
        DYNAMIC_ROUTES = {
            "GET /api/v1/resources/{id}": object(),
            "POST /api/v1/resources/{id}/action": object(),
        }
        _ROUTE_MAP = {
            "DELETE /api/v1/resources/{id}": object(),
            "PATCH /api/v1/resources/{id}": object(),
        }

    routes = check_sdk_parity._collect_routes_from_handler_class(DummyHandler)

    assert "/api/v1/static" in routes
    assert "/api/v1/method-route" in routes
    assert "/api/v1/resources/{id}" in routes
    assert "/api/v1/resources/{id}/action" in routes
    assert "/api/v1/resources/{id}" in routes


def test_collect_routes_includes_can_handle_prefixes():
    class PrefixHandler:
        def can_handle(self, path: str) -> bool:
            return path.startswith(
                (
                    "/api/v1/actions",
                    "/api/v1/orchestration/canvas",
                    "/api/pipeline/transitions",
                    "/api/plans",
                )
            )

    routes = check_sdk_parity._collect_routes_from_handler_class(PrefixHandler)

    assert "/api/v1/actions" in routes
    assert "/api/v1/actions/{param}" in routes
    assert "/api/v1/orchestration/canvas" in routes
    assert "/api/v1/orchestration/canvas/{param}" in routes
    assert "/api/pipeline/transitions" in routes
    assert "/api/pipeline/transitions/{param}" in routes
    assert "/api/plans" in routes
    assert "/api/plans/{param}" in routes


def test_extract_sdk_paths_python_captures_request_variants(tmp_path, monkeypatch):
    sdk_ns = tmp_path / "sdk" / "python" / "aragora_sdk" / "namespaces"
    sdk_ns.mkdir(parents=True, exist_ok=True)
    module = sdk_ns / "sample.py"
    module.write_text(
        """
class SampleAPI:
    def sync_request(self):
        return self._client.request("GET", "/api/v1/sync/request")

    def sync_private(self, item_id: str):
        return self._client._request("POST", f"/api/v1/sync/{item_id}/private")

    async def async_request(self):
        return await self._client.request('GET', '/api/v1/async/request')

    async def async_private(self, item_id: str):
        return await self._client._request('DELETE', f"/api/v1/async/{item_id}/private")
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(check_sdk_parity, "PROJECT_ROOT", tmp_path)
    paths_by_ns = check_sdk_parity.extract_sdk_paths_python()
    assert "sample" in paths_by_ns

    sample_paths = paths_by_ns["sample"]
    assert "/api/v1/sync/request" in sample_paths
    assert "/api/v1/sync/{param}/private" in sample_paths
    assert "/api/v1/async/request" in sample_paths
    assert "/api/v1/async/{param}/private" in sample_paths
