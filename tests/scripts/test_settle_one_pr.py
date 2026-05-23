from __future__ import annotations

from pathlib import Path

from scripts.settle_one_pr import (
    CONVERGENCE_SENTENCE,
    build_report,
    entry_blockers,
    head_blockers,
    owner_blockers,
    recursive_prompt,
    required_check_report,
    select_candidate,
)


def _entry(
    pr_number: int,
    *,
    tier: int = 2,
    status: str = "needs_model_review_quorum",
    verdict: str = "collect_model_quorum_before_merge",
    admin_squash_allowed: bool = False,
    requires_human_risk_settlement: bool = False,
    checks_summary: str = "10/10 green",
    reasons: list[str] | None = None,
) -> dict:
    return {
        "pr_number": pr_number,
        "title": f"PR {pr_number}",
        "head_sha": f"{pr_number:040d}",
        "checks_summary": checks_summary,
        "tier": tier,
        "status": status,
        "verdict": verdict,
        "admin_squash_allowed": admin_squash_allowed,
        "requires_human_risk_settlement": requires_human_risk_settlement,
        "unresolved_dissent": False,
        "reviewer_signals": [],
        "dogfood_evidence": [],
        "counted_reviewer_ids": [],
        "reasons": reasons or ["live automation surface", "model quorum incomplete: 0/2 signal(s)"],
    }


def _packet(*entries: dict, admin_order: list[int] | None = None) -> dict:
    return {
        "entries": list(entries),
        "admin_squash_order": admin_order or [],
        "not_ready": [entry["pr_number"] for entry in entries],
        "human_risk_settlement_required": [
            entry["pr_number"] for entry in entries if entry["requires_human_risk_settlement"]
        ],
    }


def test_select_candidate_prefers_admin_order() -> None:
    unauthorized = _entry(1001)
    authorized = _entry(
        1002,
        status="satisfied",
        verdict="admin_squash_allowed",
        admin_squash_allowed=True,
        reasons=["bounded internal code surface"],
    )

    selected, blockers = select_candidate(_packet(unauthorized, authorized, admin_order=[1002]))

    assert blockers == []
    assert selected is authorized


def test_select_candidate_reports_no_eligible_pr() -> None:
    selected, blockers = select_candidate(
        _packet(
            _entry(
                1003,
                tier=4,
                requires_human_risk_settlement=True,
                reasons=["workflow/deploy/destructive surface touched"],
            )
        )
    )

    assert selected is None
    assert blockers == ["no Tier 0-2 non-human-risk green PR needs only settlement evidence"]


def test_select_candidate_skips_repair_first_prs() -> None:
    selected, blockers = select_candidate(
        _packet(
            {
                **_entry(1007),
                "machine_recommendation": "repair_first",
            }
        )
    )

    assert selected is None
    assert blockers == ["no Tier 0-2 non-human-risk green PR needs only settlement evidence"]


def test_tier3_or_human_risk_is_report_only() -> None:
    blockers = entry_blockers(
        _entry(
            1004,
            tier=3,
            requires_human_risk_settlement=True,
            reasons=["semantic, persistence, security, API, or SDK surface touched"],
        )
    )

    assert "Tier 3 requires report-only handling" in blockers
    assert "requires_human_risk_settlement=true" in blockers


def test_owner_payload_blocks_active_owner() -> None:
    blockers = owner_blockers(
        {
            "owner": {
                "lane_id": "Q99-live-owner",
                "status": "active",
            }
        }
    )

    assert blockers == ["active owner Q99-live-owner"]


def test_head_drift_blocks_settlement() -> None:
    blockers = head_blockers(
        {"head_sha": "expected"},
        {
            "headRefOid": "actual",
            "isDraft": False,
            "mergeable": "MERGEABLE",
            "mergeStateStatus": "CLEAN",
        },
    )

    assert blockers == ["head drift: packet expected live actual"]


def test_missing_evidence_yields_ready_for_minimum_evidence() -> None:
    report = build_report(
        _packet(_entry(1005)),
        cwd=Path.cwd(),
        state_root=Path.cwd(),
        explicit_pr=1005,
        exclude_prs=set(),
        live=False,
        validate=False,
    )

    assert report["status"] == "ready_for_minimum_evidence"
    assert "model quorum incomplete: 0/2 signal(s)" in report["evidence"]["missing_model_quorum"]
    assert report["recursive_best_next_prompt"].endswith(CONVERGENCE_SENTENCE)


def test_cancelled_merge_quorum_suggests_rerun() -> None:
    report = required_check_report(
        [
            {
                "name": "aragora-merge-quorum",
                "workflow": "Aragora Merge Quorum",
                "state": "CANCELLED",
                "link": "https://github.com/synaptent/aragora/actions/runs/123456789",
            },
            {"name": "lint", "state": "SUCCESS"},
        ]
    )

    assert report["status"] == "blocked"
    assert report["blockers"] == ["aragora-merge-quorum is cancelled"]
    assert report["suggestions"] == ["gh run rerun 123456789 --failed"]


def test_recursive_prompt_always_contains_convergence_sentence() -> None:
    assert recursive_prompt({"selected_pr": None}).endswith(CONVERGENCE_SENTENCE)
    assert recursive_prompt({"selected_pr": 1006}).endswith(CONVERGENCE_SENTENCE)
