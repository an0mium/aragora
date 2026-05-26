from __future__ import annotations

from pathlib import Path

import scripts.settle_one_pr as settle_one_pr
from scripts.settle_one_pr import (
    CONVERGENCE_SENTENCE,
    build_report,
    entry_blockers,
    head_blockers,
    owner_blockers,
    recursive_prompt,
    required_check_report,
    required_check_source_report,
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


def test_app_pinned_required_check_blocks_manual_status_spoof() -> None:
    report = required_check_source_report(
        {"checks": [{"context": "aragora-merge-quorum", "app_id": 15368}]},
        {
            "statusCheckRollup": [
                {
                    "__typename": "StatusContext",
                    "context": "aragora-merge-quorum",
                    "state": "SUCCESS",
                }
            ]
        },
    )

    assert report["status"] == "blocked"
    assert report["blockers"] == [
        "aragora-merge-quorum is app-pinned to app_id 15368, but only a manual "
        "StatusContext is green"
    ]
    assert report["suggestions"] == [
        "rerun the app-sourced aragora-merge-quorum check; do not satisfy it with a manual status"
    ]


def test_app_pinned_required_check_accepts_successful_check_run() -> None:
    report = required_check_source_report(
        {"checks": [{"context": "aragora-merge-quorum", "app_id": 15368}]},
        {
            "statusCheckRollup": [
                {
                    "__typename": "StatusContext",
                    "context": "aragora-merge-quorum",
                    "state": "SUCCESS",
                },
                {
                    "__typename": "CheckRun",
                    "name": "aragora-merge-quorum",
                    "conclusion": "SUCCESS",
                    "workflowName": "Aragora Merge Quorum",
                    "checkSuite": {"app": {"databaseId": 15368}},
                },
            ]
        },
    )

    assert report == {"status": "pass", "blockers": [], "suggestions": []}


def test_app_pinned_required_check_blocks_wrong_app_check_run() -> None:
    report = required_check_source_report(
        {"checks": [{"context": "aragora-merge-quorum", "app_id": 15368}]},
        {
            "statusCheckRollup": [
                {
                    "__typename": "CheckRun",
                    "name": "aragora-merge-quorum",
                    "conclusion": "SUCCESS",
                    "checkSuite": {"app": {"databaseId": 42}},
                }
            ]
        },
    )

    assert report["status"] == "blocked"
    assert report["blockers"] == [
        "aragora-merge-quorum is app-pinned to app_id 15368, but successful "
        "CheckRun app_id(s) were [42]"
    ]
    assert report["suggestions"] == ["rerun the app-sourced aragora-merge-quorum check"]


def test_app_pinned_required_check_blocks_unverified_check_run_source() -> None:
    report = required_check_source_report(
        {"checks": [{"context": "aragora-merge-quorum", "app_id": 15368}]},
        {
            "statusCheckRollup": [
                {
                    "__typename": "CheckRun",
                    "name": "aragora-merge-quorum",
                    "conclusion": "SUCCESS",
                }
            ]
        },
    )

    assert report["status"] == "blocked"
    assert report["blockers"] == [
        "aragora-merge-quorum is app-pinned to app_id 15368, but CheckRun "
        "source app could not be verified"
    ]
    assert report["suggestions"] == ["fetch commit check-runs with app metadata before settlement"]


def test_app_pinned_required_check_accepts_rest_check_run_source() -> None:
    report = required_check_source_report(
        {"checks": [{"context": "aragora-merge-quorum", "app_id": 15368}]},
        {"statusCheckRollup": []},
        {
            "check_runs": [
                {
                    "name": "aragora-merge-quorum",
                    "status": "completed",
                    "conclusion": "success",
                    "app": {"id": 15368, "slug": "github-actions"},
                }
            ]
        },
    )

    assert report == {"status": "pass", "blockers": [], "suggestions": []}


def test_app_pinned_required_check_treats_neutral_check_run_as_passing() -> None:
    report = required_check_source_report(
        {"checks": [{"context": "aragora-merge-quorum", "app_id": 15368}]},
        {
            "statusCheckRollup": [
                {
                    "__typename": "CheckRun",
                    "name": "aragora-merge-quorum",
                    "conclusion": "NEUTRAL",
                    "checkSuite": {"app": {"databaseId": 15368}},
                }
            ]
        },
    )

    assert report == {"status": "pass", "blockers": [], "suggestions": []}


def test_unpinned_null_required_check_allows_status_context() -> None:
    report = required_check_source_report(
        {"checks": [{"context": "legacy-status", "app_id": None}]},
        {
            "statusCheckRollup": [
                {
                    "__typename": "StatusContext",
                    "context": "legacy-status",
                    "state": "SUCCESS",
                }
            ]
        },
    )

    assert report == {"status": "pass", "blockers": [], "suggestions": []}


def test_unpinned_any_source_required_check_allows_status_context() -> None:
    report = required_check_source_report(
        {"checks": [{"context": "legacy-status", "app_id": -1}]},
        {
            "statusCheckRollup": [
                {
                    "__typename": "StatusContext",
                    "context": "legacy-status",
                    "state": "SUCCESS",
                }
            ]
        },
    )

    assert report == {"status": "pass", "blockers": [], "suggestions": []}


def test_required_check_source_report_fails_closed_without_protection_json() -> None:
    report = required_check_source_report(
        None,
        {"statusCheckRollup": []},
    )

    assert report["status"] == "unknown"
    assert report["blockers"] == ["branch protection required_status_checks JSON unavailable"]
    assert report["suggestions"] == [
        "ensure gh can read branch protection required_status_checks before settlement"
    ]


def test_build_report_reads_required_checks_from_pr_base_branch(monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run_json(args: list[str], *, cwd: Path, timeout: int = 120):
        del cwd, timeout
        commands.append(args)
        if args[:3] == ["python3", "scripts/identify_lane_owner.py", "--pr"]:
            return {"status": "completed"}, {"command": "owner", "returncode": 0}
        if args[:3] == ["python3", "scripts/read_operator_steering.py", "--pr"]:
            return {"message_count": 0}, {"command": "mailbox", "returncode": 0}
        if args[:3] == ["gh", "pr", "view"]:
            return (
                {
                    "headRefOid": "0000000000000000000000000000000000001008",
                    "baseRefName": "release/2026.05",
                    "isDraft": False,
                    "mergeable": "MERGEABLE",
                    "mergeStateStatus": "CLEAN",
                    "statusCheckRollup": [
                        {
                            "__typename": "CheckRun",
                            "name": "aragora-merge-quorum",
                            "conclusion": "SUCCESS",
                        }
                    ],
                },
                {"command": "view", "returncode": 0},
            )
        if args[:2] == ["gh", "api"] and args[2].endswith("/required_status_checks"):
            return (
                {"checks": [{"context": "aragora-merge-quorum", "app_id": 15368}]},
                {"command": "protection", "returncode": 0},
            )
        if args[:2] == ["gh", "api"] and args[2].endswith("/check-runs?per_page=100"):
            return (
                {
                    "check_runs": [
                        {
                            "name": "aragora-merge-quorum",
                            "conclusion": "success",
                            "app": {"id": 15368},
                        }
                    ]
                },
                {"command": "check-runs", "returncode": 0},
            )
        if args[:3] == ["gh", "pr", "checks"]:
            return (
                [{"name": "aragora-merge-quorum", "state": "SUCCESS"}],
                {"command": "checks", "returncode": 0},
            )
        raise AssertionError(args)

    monkeypatch.setattr(settle_one_pr, "_run_json", fake_run_json)

    report = build_report(
        _packet(
            _entry(
                1008,
                status="satisfied",
                verdict="admin_squash_allowed",
                admin_squash_allowed=True,
                reasons=["bounded internal code surface"],
            ),
            admin_order=[1008],
        ),
        cwd=Path.cwd(),
        state_root=Path.cwd(),
        explicit_pr=1008,
        exclude_prs=set(),
        live=True,
        validate=False,
    )

    assert report["status"] == "packet_authorized_dry_run"
    assert any(
        cmd[:2] == ["gh", "api"]
        and cmd[2]
        == "repos/{owner}/{repo}/branches/release%2F2026.05/protection/required_status_checks"
        for cmd in commands
    )


def test_build_report_fails_closed_when_required_check_sources_unreadable(monkeypatch) -> None:
    def fake_run_json(args: list[str], *, cwd: Path, timeout: int = 120):
        del cwd, timeout
        if args[:3] == ["python3", "scripts/identify_lane_owner.py", "--pr"]:
            return {"status": "completed"}, {"command": "owner", "returncode": 0}
        if args[:3] == ["python3", "scripts/read_operator_steering.py", "--pr"]:
            return {"message_count": 0}, {"command": "mailbox", "returncode": 0}
        if args[:3] == ["gh", "pr", "view"]:
            return (
                {
                    "headRefOid": "0000000000000000000000000000000000001009",
                    "baseRefName": "main",
                    "isDraft": False,
                    "mergeable": "MERGEABLE",
                    "mergeStateStatus": "CLEAN",
                    "statusCheckRollup": [
                        {
                            "__typename": "CheckRun",
                            "name": "aragora-merge-quorum",
                            "conclusion": "SUCCESS",
                        }
                    ],
                },
                {"command": "view", "returncode": 0},
            )
        if args[:2] == ["gh", "api"] and args[2].endswith("/required_status_checks"):
            return None, {"command": "protection", "returncode": 403}
        if args[:2] == ["gh", "api"] and args[2].endswith("/check-runs?per_page=100"):
            return {"check_runs": []}, {"command": "check-runs", "returncode": 0}
        if args[:3] == ["gh", "pr", "checks"]:
            return (
                [{"name": "aragora-merge-quorum", "state": "SUCCESS"}],
                {"command": "checks", "returncode": 0},
            )
        raise AssertionError(args)

    monkeypatch.setattr(settle_one_pr, "_run_json", fake_run_json)

    report = build_report(
        _packet(
            _entry(
                1009,
                status="satisfied",
                verdict="admin_squash_allowed",
                admin_squash_allowed=True,
                reasons=["bounded internal code surface"],
            ),
            admin_order=[1009],
        ),
        cwd=Path.cwd(),
        state_root=Path.cwd(),
        explicit_pr=1009,
        exclude_prs=set(),
        live=True,
        validate=False,
    )

    assert report["status"] == "blocked"
    assert "branch protection required_status_checks JSON unavailable" in report["blockers"]


def test_recursive_prompt_always_contains_convergence_sentence() -> None:
    assert recursive_prompt({"selected_pr": None}).endswith(CONVERGENCE_SENTENCE)
    assert recursive_prompt({"selected_pr": 1006}).endswith(CONVERGENCE_SENTENCE)
