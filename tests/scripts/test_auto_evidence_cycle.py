"""Tests for ``scripts/auto_evidence_cycle.py`` (bounded auto-evidence cycle).

All boundaries (PR listing, merge-packet probe, collect-evidence runner,
reconciler runner, clock) are injected; no test touches the network or spawns
a subprocess.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_module(script_name: str) -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cycle = _load_module("auto_evidence_cycle.py")


def _quorum_check(conclusion: str = "FAILURE") -> dict[str, Any]:
    return {
        "workflowName": "aragora-merge-quorum",
        "name": "merge-quorum",
        "conclusion": conclusion,
        "completedAt": "2026-06-10T10:00:00Z",
    }


def _pr(
    number: int,
    *,
    draft: bool = False,
    checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "number": number,
        "isDraft": draft,
        "statusCheckRollup": checks if checks is not None else [_quorum_check()],
    }


def _entry(
    pr: int,
    *,
    status: str = "needs_model_review_quorum",
    tier: int | None = 1,
    families: list[str] | None = None,
    human: bool = False,
    dissent: bool = False,
) -> dict[str, Any]:
    return {
        "pr_number": pr,
        "status": status,
        "tier": tier,
        "counted_model_families": families or [],
        "requires_human_risk_settlement": human,
        "unresolved_dissent": dissent,
    }


class CollectRecorder:
    """Injected collect-evidence runner that records calls."""

    def __init__(self, results: dict[int, dict[str, Any]] | None = None) -> None:
        self.calls: list[tuple[int, bool]] = []
        self.results = results or {}

    def __call__(self, pr: int, apply: bool) -> dict[str, Any]:
        self.calls.append((pr, apply))
        return self.results.get(
            pr,
            {
                "ok": True,
                "counting_families": ["claude", "grok"],
                "posted_families": ["claude", "grok"],
                "error": "",
            },
        )


class ReconcilerRecorder:
    def __init__(self, returncode: int = 0) -> None:
        self.calls = 0
        self.returncode = returncode

    def __call__(self) -> int:
        self.calls += 1
        return self.returncode


def _run(
    prs: list[dict[str, Any]],
    packets: dict[int, dict[str, Any]],
    *,
    apply: bool = False,
    max_prs: int = 3,
    max_scan: int = 30,
    budget_seconds: float = 1200.0,
    collect: CollectRecorder | None = None,
    reconciler: ReconcilerRecorder | None = None,
    clock: Any = None,
    breaker_threshold: int = 3,
) -> tuple[dict[str, Any], CollectRecorder, ReconcilerRecorder]:
    collect = collect or CollectRecorder()
    reconciler = reconciler or ReconcilerRecorder()
    packet_calls: list[int] = []

    def fetch_packet(pr: int) -> dict[str, Any]:
        packet_calls.append(pr)
        return packets.get(pr, {})

    ticks = iter(range(100000)) if clock is None else clock
    summary = cycle.run_cycle(
        list_prs=lambda: prs,
        fetch_packet=fetch_packet,
        run_collect=collect,
        run_reconciler=reconciler,
        apply=apply,
        max_prs=max_prs,
        max_scan=max_scan,
        budget_seconds=budget_seconds,
        breaker_threshold=breaker_threshold,
        clock=(lambda: next(ticks) * 0.001) if clock is None else clock,
    )
    summary["_packet_calls"] = packet_calls
    return summary, collect, reconciler


# --- Stage-1 selection -------------------------------------------------------


def test_drafts_are_never_candidates() -> None:
    summary, collect, _ = _run(
        [_pr(1, draft=True), _pr(2)],
        {2: _entry(2)},
    )
    assert [item["pr"] for item in summary["plan"]] == [2]
    assert 1 not in summary["_packet_calls"]
    assert collect.calls == []  # dry-run


def test_quorum_success_heads_are_skipped_without_packet_probe() -> None:
    summary, _, _ = _run(
        [_pr(1, checks=[_quorum_check("SUCCESS")]), _pr(2)],
        {2: _entry(2)},
    )
    assert summary["_packet_calls"] == [2]
    assert [item["pr"] for item in summary["plan"]] == [2]


def test_pr_without_quorum_check_is_still_probed() -> None:
    summary, _, _ = _run([_pr(3, checks=[])], {3: _entry(3)})
    assert summary["_packet_calls"] == [3]
    assert [item["pr"] for item in summary["plan"]] == [3]


def test_candidates_scanned_oldest_first() -> None:
    summary, _, _ = _run(
        [_pr(9), _pr(4), _pr(7)],
        {4: _entry(4), 7: _entry(7), 9: _entry(9)},
    )
    assert summary["_packet_calls"] == [4, 7, 9]


# --- Stage-2 selection (canonical merge-packet probe) -------------------------


def test_satisfied_packets_are_not_selected() -> None:
    summary, _, _ = _run(
        [_pr(1), _pr(2)],
        {1: _entry(1, status="satisfied", families=["claude", "grok"]), 2: _entry(2)},
    )
    assert [item["pr"] for item in summary["plan"]] == [2]


def test_tier3_and_unknown_tier_are_never_selected() -> None:
    summary, _, _ = _run(
        [_pr(1), _pr(2), _pr(3)],
        {1: _entry(1, tier=3), 2: _entry(2, tier=None), 3: _entry(3, tier=2)},
    )
    assert [item["pr"] for item in summary["plan"]] == [3]


def test_human_settlement_and_dissent_are_never_selected() -> None:
    summary, _, _ = _run(
        [_pr(1), _pr(2), _pr(3)],
        {1: _entry(1, human=True), 2: _entry(2, dissent=True), 3: _entry(3)},
    )
    assert [item["pr"] for item in summary["plan"]] == [3]


def test_packet_with_two_counted_families_is_not_selected() -> None:
    summary, _, _ = _run(
        [_pr(1)],
        {1: _entry(1, families=["claude", "grok"])},
    )
    assert summary["plan"] == []


def test_empty_packet_probe_is_skipped_failsafe() -> None:
    summary, _, _ = _run([_pr(1)], {1: {}})
    assert summary["plan"] == []


# --- Caps ---------------------------------------------------------------------


def test_max_prs_caps_the_plan() -> None:
    prs = [_pr(n) for n in (1, 2, 3, 4, 5)]
    packets = {n: _entry(n) for n in (1, 2, 3, 4, 5)}
    summary, _, _ = _run(prs, packets, max_prs=2)
    assert [item["pr"] for item in summary["plan"]] == [1, 2]
    # Probing stops once the plan is full.
    assert summary["_packet_calls"] == [1, 2]


def test_max_scan_caps_packet_probes() -> None:
    prs = [_pr(n) for n in range(1, 11)]
    packets = {n: _entry(n, status="satisfied", families=["claude", "grok"]) for n in range(1, 11)}
    summary, _, _ = _run(prs, packets, max_scan=4)
    assert len(summary["_packet_calls"]) == 4
    assert summary["plan"] == []


def test_budget_exhaustion_stops_scanning() -> None:
    ticks = iter([0.0, 0.0, 10_000.0, 10_000.0, 10_000.0, 10_000.0, 10_000.0])
    summary, collect, _ = _run(
        [_pr(1), _pr(2)],
        {1: _entry(1), 2: _entry(2)},
        budget_seconds=60.0,
        clock=lambda: next(ticks),
    )
    assert summary["budget_exhausted"] is True
    assert collect.calls == []


# --- Dry-run purity ------------------------------------------------------------


def test_dry_run_never_collects_or_reconciles_and_exits_zero() -> None:
    summary, collect, reconciler = _run([_pr(1)], {1: _entry(1)}, apply=False)
    assert collect.calls == []
    assert reconciler.calls == 0
    assert summary["exit_code"] == 0
    assert summary["mode"] == "dry-run"
    assert [item["pr"] for item in summary["plan"]] == [1]


# --- Apply path ----------------------------------------------------------------


def test_apply_runs_collect_with_apply_then_reconciler_once() -> None:
    summary, collect, reconciler = _run(
        [_pr(1), _pr(2)],
        {1: _entry(1), 2: _entry(2)},
        apply=True,
    )
    assert collect.calls == [(1, True), (2, True)]
    assert reconciler.calls == 1
    assert summary["exit_code"] == 0
    assert sorted(summary["posted_prs"]) == [1, 2]


def test_lint_fail_counts_as_failure_and_skips_reconciler() -> None:
    # collect-evidence returns ok=False when <2 families produce counting
    # evidence; the wrapper must treat that as failure and never reconcile.
    collect = CollectRecorder(
        results={
            1: {
                "ok": False,
                "counting_families": ["claude"],
                "posted_families": [],
                "error": "only 1 counting family",
            },
        }
    )
    summary, _, reconciler = _run([_pr(1)], {1: _entry(1)}, apply=True, collect=collect)
    assert reconciler.calls == 0
    assert summary["posted_prs"] == []
    assert summary["exit_code"] == 1
    assert summary["failed_prs"] == [1]


def test_reconciler_failure_fails_closed() -> None:
    reconciler = ReconcilerRecorder(returncode=1)
    summary, _, rec = _run([_pr(1)], {1: _entry(1)}, apply=True, reconciler=reconciler)
    assert rec.calls == 1
    assert summary["exit_code"] == 1


def test_identical_error_breaker_trips_after_threshold() -> None:
    err = {
        "ok": False,
        "counting_families": [],
        "posted_families": [],
        "error": "claude CLI not found on PATH",
    }
    collect = CollectRecorder(results={1: err, 2: err, 3: err, 4: err})
    summary, rec_collect, reconciler = _run(
        [_pr(n) for n in (1, 2, 3, 4)],
        {n: _entry(n) for n in (1, 2, 3, 4)},
        apply=True,
        max_prs=4,
        collect=collect,
        breaker_threshold=3,
    )
    assert len(rec_collect.calls) == 3  # stopped at the breaker, PR 4 untouched
    assert summary["breaker_tripped"] is True
    assert summary["exit_code"] == 2
    assert reconciler.calls == 0


def test_distinct_errors_do_not_trip_breaker() -> None:
    collect = CollectRecorder(
        results={
            1: {"ok": False, "counting_families": [], "posted_families": [], "error": "error A"},
            2: {"ok": False, "counting_families": [], "posted_families": [], "error": "error B"},
            3: {"ok": False, "counting_families": [], "posted_families": [], "error": "error A"},
        }
    )
    summary, rec_collect, _ = _run(
        [_pr(n) for n in (1, 2, 3)],
        {n: _entry(n) for n in (1, 2, 3)},
        apply=True,
        collect=collect,
    )
    assert len(rec_collect.calls) == 3
    assert summary["breaker_tripped"] is False
    assert summary["exit_code"] == 1


def test_success_resets_breaker_counter() -> None:
    err = {"ok": False, "counting_families": [], "posted_families": [], "error": "same"}
    collect = CollectRecorder(results={1: err, 2: err})  # 3 succeeds by default
    summary, rec_collect, _ = _run(
        [_pr(n) for n in (1, 2, 3, 4)],
        {n: _entry(n) for n in (1, 2, 3, 4)},
        apply=True,
        max_prs=4,
        collect=collect,
        breaker_threshold=3,
    )
    assert len(rec_collect.calls) == 4
    assert summary["breaker_tripped"] is False


def test_partial_post_is_a_failure() -> None:
    # One posted family is not quorum evidence; do not treat it as success.
    collect = CollectRecorder(
        results={
            1: {
                "ok": True,
                "counting_families": ["claude", "grok"],
                "posted_families": ["claude"],
                "error": "",
            },
        }
    )
    summary, _, reconciler = _run([_pr(1)], {1: _entry(1)}, apply=True, collect=collect)
    assert summary["posted_prs"] == []
    assert summary["failed_prs"] == [1]
    assert summary["exit_code"] == 1
    assert reconciler.calls == 0


# --- Secrets guard --------------------------------------------------------------


def test_sanitized_env_forces_secrets_off(monkeypatch: Any) -> None:
    # Lane V (run-20260609): collect-evidence API reviewers crash under
    # ARAGORA_SECRETS_STRICT=true with MFA-gated AWS (interactive prompt EOF).
    monkeypatch.setenv("ARAGORA_SECRETS_STRICT", "true")
    monkeypatch.setenv("ARAGORA_USE_SECRETS_MANAGER", "true")
    env = cycle._sanitized_env()
    assert env["ARAGORA_SECRETS_STRICT"] == "false"
    assert env["ARAGORA_USE_SECRETS_MANAGER"] == "false"


# --- Result parsing --------------------------------------------------------------


def test_parse_collect_output_success() -> None:
    payload = {
        "mode": "collect_evidence",
        "counting_families": ["claude", "grok"],
        "posted_families": ["claude", "grok"],
        "post_errors": [],
    }
    result = cycle.parse_collect_output(0, __import__("json").dumps(payload), "")
    assert result["ok"] is True
    assert result["posted_families"] == ["claude", "grok"]


def test_parse_collect_output_failure_uses_error_field() -> None:
    payload = {"mode": "collect_evidence", "error": "could not resolve head SHA"}
    result = cycle.parse_collect_output(1, __import__("json").dumps(payload), "")
    assert result["ok"] is False
    assert "could not resolve head SHA" in result["error"]


def test_parse_collect_output_garbage_is_failure() -> None:
    result = cycle.parse_collect_output(0, "not json", "boom")
    assert result["ok"] is False
    assert result["error"]


# --- Singleton lock (double-post guard) --------------------------------------------


def test_cycle_lock_blocks_second_acquirer(tmp_path: Any) -> None:
    lock = str(tmp_path / "cycle.lock")
    release = cycle.acquire_cycle_lock(lock)
    try:
        try:
            cycle.acquire_cycle_lock(lock)
        except cycle.CycleLockHeld:
            pass
        else:
            raise AssertionError("expected CycleLockHeld")
    finally:
        release()
    # After release the lock is acquirable again.
    cycle.acquire_cycle_lock(lock)()


def test_cycle_lock_reclaims_stale_lock(tmp_path: Any) -> None:
    lock = str(tmp_path / "cycle.lock")
    cycle.acquire_cycle_lock(lock)  # crashed invocation: never released
    fake_now = __import__("os").path.getmtime(lock) + 7201.0
    release = cycle.acquire_cycle_lock(lock, now=lambda: fake_now)
    release()


def test_cycle_lock_fresh_lock_is_not_reclaimed(tmp_path: Any) -> None:
    lock = str(tmp_path / "cycle.lock")
    cycle.acquire_cycle_lock(lock)
    fake_now = __import__("os").path.getmtime(lock) + 60.0
    try:
        cycle.acquire_cycle_lock(lock, now=lambda: fake_now)
    except cycle.CycleLockHeld:
        pass
    else:
        raise AssertionError("expected CycleLockHeld")


# --- Listing degradation ----------------------------------------------------------


def test_list_prs_falls_back_to_light_listing(monkeypatch: Any) -> None:
    # statusCheckRollup over many PRs 504s GitHub GraphQL (observed live);
    # the lister must degrade to a light listing instead of crashing.
    calls: list[tuple[str, int]] = []

    def fake_gh_pr_list(repo: str, fields: str, limit: int) -> list[dict[str, Any]] | None:
        calls.append((fields, limit))
        if "statusCheckRollup" in fields:
            return None  # heavy query 504s at every page size
        return [{"number": 5, "isDraft": False}]

    monkeypatch.setattr(cycle, "_gh_pr_list", fake_gh_pr_list)
    rows = cycle.default_list_prs("owner/repo")
    assert rows == [{"number": 5, "isDraft": False}]
    assert calls[-1] == ("number,isDraft", 200)
    assert [limit for fields, limit in calls if "statusCheckRollup" in fields] == [100, 50, 30]


def test_list_prs_fails_closed_when_all_listings_fail(monkeypatch: Any) -> None:
    monkeypatch.setattr(cycle, "_gh_pr_list", lambda repo, fields, limit: None)
    try:
        cycle.default_list_prs("owner/repo")
    except RuntimeError as exc:
        assert "gh pr list failed" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_list_prs_prefers_first_successful_heavy_page(monkeypatch: Any) -> None:
    def fake_gh_pr_list(repo: str, fields: str, limit: int) -> list[dict[str, Any]] | None:
        if "statusCheckRollup" in fields and limit == 30:
            return [{"number": 7, "isDraft": False, "statusCheckRollup": []}]
        if "statusCheckRollup" in fields:
            return None
        raise AssertionError("light listing should not be reached")

    monkeypatch.setattr(cycle, "_gh_pr_list", fake_gh_pr_list)
    rows = cycle.default_list_prs("owner/repo")
    assert rows[0]["number"] == 7


# --- Quorum-check rollup helper ---------------------------------------------------


def test_latest_quorum_conclusion_uses_latest_row() -> None:
    checks = [
        {**_quorum_check("FAILURE"), "completedAt": "2026-06-10T10:00:00Z"},
        {**_quorum_check("SUCCESS"), "completedAt": "2026-06-10T11:00:00Z"},
    ]
    assert cycle.latest_quorum_conclusion({"statusCheckRollup": checks}) == "SUCCESS"


def test_latest_quorum_conclusion_missing() -> None:
    assert cycle.latest_quorum_conclusion({"statusCheckRollup": []}) == ""
    assert cycle.latest_quorum_conclusion({}) == ""
