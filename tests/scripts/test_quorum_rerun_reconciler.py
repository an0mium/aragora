"""Tests for ``scripts/quorum_rerun_reconciler.py`` planning logic."""

from __future__ import annotations

import importlib.util
import json
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


reconciler = _load_module("quorum_rerun_reconciler.py")

NOW = "2026-06-10T12:00:00Z"
OLD = "2026-06-10T10:00:00Z"
JUST_NOW = "2026-06-10T11:58:00Z"
RUN_URL = "https://github.com/synaptent/aragora/actions/runs/987654/job/1234567"


def _quorum_check(
    *,
    conclusion: str = "FAILURE",
    completed_at: str = OLD,
    details_url: str = RUN_URL,
) -> dict[str, Any]:
    return {
        "workflowName": "aragora-merge-quorum",
        "name": "merge-quorum",
        "conclusion": conclusion,
        "status": "COMPLETED",
        "completedAt": completed_at,
        "detailsUrl": details_url,
    }


def _pr(number: int, *, draft: bool = False) -> dict[str, Any]:
    return {"number": number, "isDraft": draft}


def _pr_detail(checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {"statusCheckRollup": checks, "headRefOid": "a" * 40}


def _packet(status: str) -> dict[str, Any]:
    return {"status": status}


def _plan(
    prs: list[dict[str, Any]],
    details: dict[int, dict[str, Any]],
    packets: dict[int, dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return reconciler.build_plan(
        prs,
        fetch_pr_detail=lambda n: details[n],
        fetch_packet=lambda n: packets[n],
        now=reconciler.parse_iso(NOW),
        min_age_minutes=kwargs.pop("min_age_minutes", 10),
        max_reruns=kwargs.pop("max_reruns", 3),
    )


def test_stale_failure_with_satisfiable_packet_plans_rerun() -> None:
    plan = _plan(
        [_pr(101)],
        {101: _pr_detail([_quorum_check()])},
        {101: _packet("satisfied")},
    )
    assert len(plan) == 1
    action = plan[0]
    assert action["pr"] == 101
    assert action["run_id"] == "987654"
    assert action["command"] == ["gh", "run", "rerun", "987654"]


def test_needs_model_review_quorum_packet_also_plans_rerun() -> None:
    plan = _plan(
        [_pr(102)],
        {102: _pr_detail([_quorum_check()])},
        {102: _packet("needs_model_review_quorum")},
    )
    assert [a["pr"] for a in plan] == [102]


def test_unsatisfiable_packet_plans_nothing() -> None:
    plan = _plan(
        [_pr(103)],
        {103: _pr_detail([_quorum_check()])},
        {103: _packet("repair_or_wait")},
    )
    assert plan == []


def test_successful_quorum_check_is_skipped_without_packet_call() -> None:
    calls: list[int] = []

    def packet(n: int) -> dict[str, Any]:
        calls.append(n)
        return _packet("satisfied")

    plan = reconciler.build_plan(
        [_pr(104)],
        fetch_pr_detail=lambda n: _pr_detail([_quorum_check(conclusion="SUCCESS")]),
        fetch_packet=packet,
        now=reconciler.parse_iso(NOW),
        min_age_minutes=10,
        max_reruns=3,
    )
    assert plan == []
    assert calls == []


def test_draft_prs_are_skipped() -> None:
    plan = _plan(
        [_pr(105, draft=True)],
        {105: _pr_detail([_quorum_check()])},
        {105: _packet("satisfied")},
    )
    assert plan == []


def test_recent_failure_below_min_age_is_skipped() -> None:
    plan = _plan(
        [_pr(106)],
        {106: _pr_detail([_quorum_check(completed_at=JUST_NOW)])},
        {106: _packet("satisfied")},
    )
    assert plan == []


def test_max_reruns_caps_the_plan() -> None:
    prs = [_pr(n) for n in (201, 202, 203, 204)]
    details = {n: _pr_detail([_quorum_check()]) for n in (201, 202, 203, 204)}
    packets = {n: _packet("satisfied") for n in (201, 202, 203, 204)}
    plan = _plan(prs, details, packets, max_reruns=2)
    assert len(plan) == 2


def test_missing_run_id_in_details_url_plans_nothing() -> None:
    plan = _plan(
        [_pr(107)],
        {107: _pr_detail([_quorum_check(details_url="https://example.com/no-run")])},
        {107: _packet("satisfied")},
    )
    assert plan == []


def test_dry_run_main_never_executes(monkeypatch: Any, capsys: Any) -> None:
    executed: list[list[str]] = []
    monkeypatch.setattr(reconciler, "_run_command", lambda cmd: executed.append(cmd) or 0)
    monkeypatch.setattr(
        reconciler,
        "_fetch_open_prs",
        lambda repo: [_pr(108)],
    )
    monkeypatch.setattr(
        reconciler,
        "_fetch_pr_detail",
        lambda repo, n: _pr_detail([_quorum_check()]),
    )
    monkeypatch.setattr(
        reconciler,
        "_fetch_packet",
        lambda repo, n: _packet("satisfied"),
    )
    rc = reconciler.main(["--repo", "synaptent/aragora"])
    assert rc == 0
    assert executed == []
    out = capsys.readouterr().out
    payload = [json.loads(line) for line in out.strip().splitlines() if line.startswith("{")]
    assert any(item.get("pr") == 108 for item in payload)


def test_apply_main_executes_and_fails_closed(monkeypatch: Any) -> None:
    monkeypatch.setattr(reconciler, "_run_command", lambda cmd: 1)
    monkeypatch.setattr(reconciler, "_fetch_open_prs", lambda repo: [_pr(109)])
    monkeypatch.setattr(
        reconciler,
        "_fetch_pr_detail",
        lambda repo, n: _pr_detail([_quorum_check()]),
    )
    monkeypatch.setattr(reconciler, "_fetch_packet", lambda repo, n: _packet("satisfied"))
    rc = reconciler.main(["--repo", "synaptent/aragora", "--apply"])
    assert rc == 1


def test_circular_repair_or_wait_with_only_quorum_failing_plans_rerun() -> None:
    packet = {
        "status": "repair_or_wait",
        "counted_reviewer_ids": ["grok"],
        "unresolved_dissent": False,
    }
    plan = _plan(
        [_pr(301)],
        {301: _pr_detail([_quorum_check()])},
        {301: packet},
    )
    assert [a["pr"] for a in plan] == [301]


def test_repair_or_wait_with_other_real_failure_is_skipped() -> None:
    other_failure = {
        "workflowName": "ci",
        "name": "build",
        "conclusion": "FAILURE",
        "status": "COMPLETED",
        "completedAt": OLD,
        "detailsUrl": "https://github.com/synaptent/aragora/actions/runs/111/job/222",
    }
    packet = {
        "status": "repair_or_wait",
        "counted_reviewer_ids": ["grok"],
        "unresolved_dissent": False,
    }
    plan = _plan(
        [_pr(302)],
        {302: _pr_detail([_quorum_check(), other_failure])},
        {302: packet},
    )
    assert plan == []


def test_repair_or_wait_without_counted_reviewers_is_skipped() -> None:
    packet = {"status": "repair_or_wait", "counted_reviewer_ids": [], "unresolved_dissent": False}
    plan = _plan(
        [_pr(303)],
        {303: _pr_detail([_quorum_check()])},
        {303: packet},
    )
    assert plan == []


def test_repair_or_wait_with_unresolved_dissent_is_skipped() -> None:
    packet = {
        "status": "repair_or_wait",
        "counted_reviewer_ids": ["grok"],
        "unresolved_dissent": True,
    }
    plan = _plan(
        [_pr(304)],
        {304: _pr_detail([_quorum_check()])},
        {304: packet},
    )
    assert plan == []
