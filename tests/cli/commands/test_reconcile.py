from __future__ import annotations

import argparse

from aragora.cli.commands.reconcile import (
    add_reconcile_parser,
    build_settle_report,
    cmd_reconcile,
)


def _green_rollup() -> list[dict[str, str]]:
    return [
        {"name": "Generate & Validate", "conclusion": "SUCCESS"},
        {"name": "TypeScript SDK Type Check", "conclusion": "SUCCESS"},
        {"name": "aragora-merge-quorum", "conclusion": "SUCCESS"},
        {"name": "lint", "conclusion": "SUCCESS"},
        {"name": "sdk-parity", "conclusion": "SUCCESS"},
        {"name": "typecheck", "conclusion": "SUCCESS"},
    ]


def _view(
    number: int,
    *,
    state: str = "OPEN",
    head: str | None = None,
    is_draft: bool = False,
    mergeable: str = "MERGEABLE",
    merge_state: str = "BLOCKED",
    merged_at: str = "",
    rollup: list[dict[str, str]] | None = None,
) -> dict:
    return {
        "number": number,
        "title": f"PR {number}",
        "url": f"https://github.com/synaptent/aragora/pull/{number}",
        "state": state,
        "mergedAt": merged_at,
        "headRefName": f"branch-{number}",
        "headRefOid": head or f"{number:040d}"[-40:],
        "isDraft": is_draft,
        "mergeable": mergeable,
        "mergeStateStatus": merge_state,
        "statusCheckRollup": rollup or _green_rollup(),
    }


def _entry(number: int, *, head: str | None = None, tier: int = 2, status: str = "satisfied"):
    return {
        "pr_number": number,
        "title": f"PR {number}",
        "url": f"https://github.com/synaptent/aragora/pull/{number}",
        "head_sha": head or f"{number:040d}"[-40:],
        "tier": tier,
        "tier_name": f"tier_{tier}",
        "status": status,
        "verdict": "admin_squash_allowed" if status == "satisfied" else "not_ready",
        "admin_squash_allowed": status == "satisfied",
        "requires_human_risk_settlement": tier >= 3,
        "requires_human_preapproval": tier >= 4,
        "human_preapproval_recorded": False,
        "unresolved_dissent": False,
        "reviewer_signals": [],
        "dogfood_evidence": [],
        "counted_reviewer_ids": [],
        "counted_model_families": [],
        "reasons": [],
    }


def _packet(*entries: dict) -> dict:
    return {
        "version": "merge_authorization_packet.v1",
        "generated_at": "2026-06-28T00:00:00+00:00",
        "queue_pressure": {"current_open_prs": len(entries), "cap": 6, "active": False},
        "entries": list(entries),
        "admin_squash_order": [e["pr_number"] for e in entries if e["admin_squash_allowed"]],
        "human_risk_settlement_required": [
            e["pr_number"] for e in entries if e["requires_human_risk_settlement"]
        ],
        "not_ready": [e["pr_number"] for e in entries if e["status"] != "satisfied"],
    }


def test_reconcile_parser_registers_settle_report() -> None:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers()
    add_reconcile_parser(sub)

    ns = root.parse_args(["reconcile", "settle", "--autonomy", "report", "--pr", "8658", "--json"])

    assert ns.command == "reconcile"
    assert ns.reconcile_command == "settle"
    assert ns.autonomy == "report"
    assert ns.pr == ["8658"]
    assert ns.json is True


def test_top_level_parser_registers_reconcile_settle() -> None:
    from aragora.cli.parser import build_parser

    ns = build_parser().parse_args(
        ["reconcile", "settle", "--autonomy", "report", "--pr", "8658", "--json"]
    )

    assert ns.command == "reconcile"
    assert ns.reconcile_command == "settle"
    assert ns.autonomy == "report"
    assert ns.pr == ["8658"]
    assert ns.json is True


def test_build_settle_report_buckets_exact_head_prs() -> None:
    packet = _packet(
        _entry(1),
        _entry(2, status="needs_model_review_quorum"),
        _entry(3, tier=4, status="human_risk_settlement_required"),
    )
    views = {
        1: _view(1),
        2: _view(2),
        3: _view(3),
        4: _view(4, state="MERGED", merged_at="2026-06-28T01:02:03Z"),
    }

    report = build_settle_report(packet=packet, views=views, autonomy="report", repo=None)

    assert report["mutated"] is False
    assert report["counts"] == {
        "mergeable": 1,
        "parked": 1,
        "superseded": 1,
        "needs_human": 1,
    }
    assert report["mergeable"][0]["pr"] == 1
    assert report["parked"][0]["pr"] == 2
    assert report["needs_human"][0]["pr"] == 3
    assert report["superseded"][0]["pr"] == 4


def test_build_settle_report_uses_latest_status_rollup_entries() -> None:
    packet = _packet(_entry(1))
    stale_success = {
        "name": "lint",
        "workflowName": "Lint",
        "conclusion": "SUCCESS",
        "completedAt": "2026-06-28T00:00:00Z",
    }
    latest_failure = {
        "name": "lint",
        "workflowName": "Lint",
        "conclusion": "FAILURE",
        "completedAt": "2026-06-28T00:05:00Z",
    }
    views = {
        1: _view(
            1,
            rollup=[
                latest_failure,
                {"name": "Generate & Validate", "conclusion": "SUCCESS"},
                {"name": "TypeScript SDK Type Check", "conclusion": "SUCCESS"},
                {"name": "aragora-merge-quorum", "conclusion": "SUCCESS"},
                {"name": "sdk-parity", "conclusion": "SUCCESS"},
                {"name": "typecheck", "conclusion": "SUCCESS"},
                stale_success,
            ],
        )
    }

    report = build_settle_report(packet=packet, views=views, autonomy="report", repo=None)

    assert report["counts"]["mergeable"] == 0
    assert report["counts"]["parked"] == 1
    assert any("lint=FAILURE" in blocker for blocker in report["parked"][0]["blockers"])


def test_build_settle_report_classifies_human_gate_before_mergeable() -> None:
    entry = _entry(1)
    entry["requires_human_preapproval"] = True
    entry["human_preapproval_recorded"] = False
    packet = _packet(entry)
    views = {1: _view(1)}

    report = build_settle_report(packet=packet, views=views, autonomy="report", repo=None)

    assert report["counts"]["mergeable"] == 0
    assert report["counts"]["needs_human"] == 1
    assert report["needs_human"][0]["bucket_reason"] == "human_required"


def test_report_mode_does_not_call_merger_or_evidence_apply(monkeypatch) -> None:
    packet = _packet(_entry(1))
    calls: list[str] = []

    def fake_packet(**_kwargs):
        calls.append("packet")
        return packet

    def fake_view(ref: str, *, repo: str | None):
        calls.append(f"view:{ref}:{repo}")
        return _view(int(ref))

    def forbidden_side_effect(*_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("report autonomy must not mutate")

    monkeypatch.setattr("aragora.cli.commands.reconcile._build_merge_packet", fake_packet)
    monkeypatch.setattr("aragora.cli.commands.reconcile._fetch_pr_view", fake_view)
    monkeypatch.setattr("aragora.cli.commands.reconcile._merge_pr", forbidden_side_effect)
    monkeypatch.setattr(
        "aragora.cli.commands.reconcile._collect_quorum_evidence_apply",
        forbidden_side_effect,
    )

    args = argparse.Namespace(
        reconcile_command="settle",
        autonomy="report",
        pr=["1"],
        limit=30,
        repo=None,
        review_queue_root=None,
        execute_reviewers=False,
        ignore_own_quorum_check=False,
        json=True,
    )

    rc = cmd_reconcile(args)

    assert rc == 0
    assert calls == ["view:1:None", "packet", "view:1:None"]


def test_report_mode_rejects_execute_reviewers(monkeypatch, capsys) -> None:
    def forbidden_packet(**_kwargs):  # pragma: no cover - must not run
        raise AssertionError("read-only report must reject reviewer execution first")

    monkeypatch.setattr("aragora.cli.commands.reconcile._build_merge_packet", forbidden_packet)
    args = argparse.Namespace(
        reconcile_command="settle",
        autonomy="report",
        pr=["1"],
        limit=30,
        repo=None,
        review_queue_root=None,
        execute_reviewers=True,
        ignore_own_quorum_check=False,
        json=True,
    )

    rc = cmd_reconcile(args)

    assert rc == 2
    assert "--execute-reviewers is not allowed" in capsys.readouterr().err


def test_report_mode_rejects_ignore_own_quorum_check(monkeypatch, capsys) -> None:
    def forbidden_packet(**_kwargs):  # pragma: no cover - must not run
        raise AssertionError("read-only report must reject quorum bypass first")

    monkeypatch.setattr("aragora.cli.commands.reconcile._build_merge_packet", forbidden_packet)
    args = argparse.Namespace(
        reconcile_command="settle",
        autonomy="report",
        pr=["1"],
        limit=30,
        repo=None,
        review_queue_root=None,
        execute_reviewers=False,
        ignore_own_quorum_check=True,
        json=True,
    )

    rc = cmd_reconcile(args)

    assert rc == 2
    assert "--ignore-own-quorum-check is not allowed" in capsys.readouterr().err
