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
    monkeypatch.setattr(
        check_sdk_parity,
        "extract_handler_routes_with_status",
        lambda: check_sdk_parity.HandlerRouteExtractionResult(routes={}, available=True),
    )
    monkeypatch.setattr(check_sdk_parity, "extract_sdk_paths_python", lambda: {})
    monkeypatch.setattr(check_sdk_parity, "extract_sdk_paths_typescript", lambda: {})
    monkeypatch.setattr(check_sdk_parity, "build_parity_report", lambda *_, **__: report)
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
  "max_missing_from_both_sdks": 2,
  "max_stale_python_sdk_paths": 1
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
  "max_missing_from_both_sdks": 0,
  "max_stale_python_sdk_paths": 4
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
  "max_missing_from_both_sdks": 2,
  "max_stale_python_sdk_paths": 10,
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


def test_stale_sdk_paths_exclude_internal_route_families():
    """Internal-family SDK paths never count as stale.

    The internal-route policy excludes those families from the public spec,
    so documented_routes can no longer vouch for them; without this carve-out
    every internal SDK method whose handler is dispatch-based (no ROUTES)
    would become stale debt against the decaying budget.
    """
    report = check_sdk_parity.build_parity_report(
        handler_routes={"PublicHandler": ["/api/v1/debates"]},
        python_sdk={
            "debates": {"/api/v1/debates"},
            "control_plane": {"/api/v1/control-plane/health/status"},
            "sme": {"/api/v1/sme/budgets/{budget_id}"},
            "stale": {"/api/v1/genuinely-stale"},
        },
        typescript_sdk={"control_plane": {"/api/v1/control-plane/stats"}},
        documented_routes={"/api/debates"},
    )

    assert report["gaps"]["stale_python_sdk_paths"] == ["/api/genuinely-stale"]
    assert report["gaps"]["stale_typescript_sdk_paths"] == []


def test_handler_coverage_excludes_internal_route_families():
    """Internal-family handler routes are never public SDK-coverage gaps.

    When no documented-route source is available, coverage enforcement falls
    back to handler ROUTES alone; the internal-route policy must still keep
    e.g. control-plane routes out of the missing-coverage buckets.
    """
    report = check_sdk_parity.build_parity_report(
        handler_routes={
            "PublicHandler": ["/api/v1/debates"],
            "ControlPlaneHandler": ["/api/v1/control-plane/health/status"],
        },
        python_sdk={"debates": {"/api/v1/debates"}},
        typescript_sdk={"debates": {"/api/v1/debates"}},
        documented_routes=None,
    )

    assert report["gaps"]["missing_from_python_sdk"] == []
    assert report["gaps"]["missing_from_typescript_sdk"] == []
    assert report["gaps"]["missing_from_both_sdks"] == []


# ---------------------------------------------------------------------------
# Budget semantics (#9086)
#
# The progressive budget used to be derived from wall-clock: an `initial` debt
# minus `weekly_reduction` per elapsed week. Two consequences, both real:
#
#   1. The required `sdk-parity` check could go red with NO code change, simply
#      because a week boundary passed. #9086 records the 54 -> 51 roll doing
#      exactly that while actual debt sat at 53.
#   2. `dt.date.today()` is local-time, so the same commit passed under
#      TZ=America/Chicago and failed under TZ=UTC — CI and a developer's laptop
#      disagreed about whether the tree was green.
#
# The enforced ceiling is now an explicit committed number (`max_*`). It changes
# when a human commits a change to it, never when the clock advances.
# ---------------------------------------------------------------------------

import datetime as dt  # noqa: E402
import json  # noqa: E402


def _budget(tmp_path, *, max_missing: int, max_stale: int, extra: str = "") -> Path:
    path = tmp_path / "budget.json"
    payload = {
        "start_date": "2026-01-01",
        "max_missing_from_both_sdks": max_missing,
        "max_stale_python_sdk_paths": max_stale,
    }
    if extra:
        payload.update(json.loads(extra))
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _run(monkeypatch, budget: Path, *, today: str | None = None) -> int:
    argv = ["check_sdk_parity.py", "--strict", "--allow-missing", "--budget", str(budget)]
    if today:
        argv += ["--today", today]
    monkeypatch.setattr(sys, "argv", argv)
    return check_sdk_parity.main()


def test_budget_verdict_is_stable_across_dates(monkeypatch, tmp_path):
    """The #9086 invariant: the calendar alone must never flip the verdict.

    Same tree, same budget file, dates a year apart — identical result. Under the
    old date-derived ceiling the later date went red with no code change.
    """
    budget = _budget(tmp_path, max_missing=0, max_stale=44)
    verdicts = []
    for today in ("2026-01-01", "2026-08-07", "2027-06-01"):
        _patch_report(monkeypatch, missing=0, stale_python=44)
        verdicts.append(_run(monkeypatch, budget, today=today))
    assert verdicts == [0, 0, 0], (
        f"verdict changed with the calendar alone: {verdicts}. A gate that reddens "
        "without a code change is untruthful (#9086)."
    )


def test_default_today_is_utc_not_local_time(monkeypatch):
    """CI and a laptop must agree on the date.

    `dt.date.today()` is local-time, which is why the same commit passed under
    TZ=America/Chicago and failed under TZ=UTC.
    """
    resolved = check_sdk_parity._resolve_today(None)
    assert resolved == dt.datetime.now(dt.timezone.utc).date(), (
        "default 'today' is not UTC-pinned; local timezone can shift it by a day"
    )


def test_budget_fails_when_actual_exceeds_committed_ceiling(monkeypatch, tmp_path):
    """A real regression — debt rose above the committed number — still fails."""
    _patch_report(monkeypatch, missing=0, stale_python=45)
    budget = _budget(tmp_path, max_missing=0, max_stale=44)
    assert _run(monkeypatch, budget, today="2026-08-07") == 1


def test_budget_passes_when_actual_within_committed_ceiling(monkeypatch, tmp_path):
    _patch_report(monkeypatch, missing=0, stale_python=44)
    budget = _budget(tmp_path, max_missing=0, max_stale=44)
    assert _run(monkeypatch, budget, today="2026-08-07") == 0


def test_budget_file_without_committed_ceiling_fails_closed(monkeypatch, tmp_path):
    """A legacy time-derived budget file must not silently resurrect the bug.

    Failing closed with an actionable message beats quietly re-enabling a ceiling
    that tightens on a timer.
    """
    _patch_report(monkeypatch, missing=0, stale_python=44)
    legacy = tmp_path / "budget.json"
    legacy.write_text(
        json.dumps(
            {
                "start_date": "2026-04-24",
                "initial_stale_python_sdk_paths": 87,
                "weekly_reduction_stale_python_sdk_paths": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert _run(monkeypatch, legacy, today="2026-08-07") == 2


def test_tighten_lowers_ceiling_to_current_actuals(monkeypatch, tmp_path):
    """`--tighten` is the explicit roll: debt paid down, ceiling committed lower."""
    _patch_report(monkeypatch, missing=0, stale_python=40)
    budget = _budget(tmp_path, max_missing=0, max_stale=44)
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_sdk_parity.py", "--budget", str(budget), "--tighten", "--today", "2026-08-07"],
    )
    assert check_sdk_parity.main() == 0
    written = json.loads(budget.read_text(encoding="utf-8"))
    assert written["max_stale_python_sdk_paths"] == 40


def test_tighten_never_raises_the_ceiling(monkeypatch, tmp_path):
    """A ratchet only turns one way — `--tighten` must not launder a regression."""
    _patch_report(monkeypatch, missing=0, stale_python=60)
    budget = _budget(tmp_path, max_missing=0, max_stale=44)
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_sdk_parity.py", "--budget", str(budget), "--tighten", "--today", "2026-08-07"],
    )
    assert check_sdk_parity.main() == 1
    written = json.loads(budget.read_text(encoding="utf-8"))
    assert written["max_stale_python_sdk_paths"] == 44, "ceiling was raised to hide a regression"
