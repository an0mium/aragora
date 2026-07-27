from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from aragora.cli.commands.founder_status import (
    gather_founder_status,
    render_founder_status,
)
from aragora.cli.commands.status import cmd_status
from aragora.cli.parser import build_parser
from aragora.review.health import HealthReport, SurfaceCheck


def _health(status: str = "fresh") -> HealthReport:
    return HealthReport(
        generated_at=datetime(2026, 7, 4, tzinfo=timezone.utc),
        overall_status=status,
        surfaces=[
            SurfaceCheck(
                name="boss_metrics",
                status=status,
                count=3,
                detail="ok",
            )
        ],
    )


def test_founder_status_reports_queue_and_next_blocker(tmp_path: Path) -> None:
    state = tmp_path / ".aragora"
    brief_root = state / "overnight-brief"
    brief_root.mkdir(parents=True)
    (brief_root / "latest.md").write_text("# Morning brief\n\nTop 3.\n", encoding="utf-8")

    def merge_packet_builder(**_: object) -> dict[str, object]:
        return {
            "queue_pressure": {"current_open_prs": 2, "cap": 6, "active": False},
            "admin_squash_order": [],
            "human_risk_settlement_required": [],
            "not_ready": [8827],
            "entries": [
                {
                    "pr_number": 8827,
                    "title": "refactor(events): split security debate runner",
                    "url": "https://github.com/synaptent/aragora/pull/8827",
                    "head_sha": "abc123",
                    "tier": 2,
                    "status": "needs_model_review_quorum",
                    "verdict": "collect_model_quorum_before_merge",
                    "admin_squash_allowed": False,
                    "requires_human_risk_settlement": False,
                    "requires_human_preapproval": False,
                    "unresolved_dissent": False,
                    "checks_summary": "1 failing / 6 required",
                    "counted_model_families": [],
                    "reasons": ["model quorum incomplete: 0/2 signal(s)"],
                }
            ],
        }

    report = gather_founder_status(
        repo_root=str(tmp_path),
        limit=5,
        health_gatherer=lambda **_: _health(),
        merge_packet_builder=merge_packet_builder,
    )

    assert report["queue"]["transport_status"] == "ok"
    assert report["queue"]["not_ready"] == [8827]
    assert report["latest_brief"]["preview"].startswith("# Morning brief")
    assert report["next_action"]["kind"] == "queue_blocker"
    assert "PR #8827" in report["next_action"]["summary"]

    rendered = render_founder_status(report)
    assert "Aragora Founder Status" in rendered
    assert "Next action: Work one bounded blocker on PR #8827" in rendered


def test_founder_status_next_action_uses_not_ready_entry_beyond_top_entries(
    tmp_path: Path,
) -> None:
    def ready_entry(pr_number: int) -> dict[str, object]:
        return {
            "pr_number": pr_number,
            "title": f"ready {pr_number}",
            "url": f"https://github.com/synaptent/aragora/pull/{pr_number}",
            "head_sha": "abc123",
            "tier": 2,
            "status": "admin_squash_allowed",
            "verdict": "admin_squash_allowed",
            "admin_squash_allowed": True,
            "requires_human_risk_settlement": False,
            "requires_human_preapproval": False,
            "unresolved_dissent": False,
            "checks_summary": "6 / 6 required",
            "counted_model_families": ["claude", "openai"],
            "reasons": [],
        }

    blocked = {
        "pr_number": 8899,
        "title": "blocked outside top display rows",
        "url": "https://github.com/synaptent/aragora/pull/8899",
        "head_sha": "def456",
        "tier": 2,
        "status": "needs_model_review_quorum",
        "verdict": "collect_model_quorum_before_merge",
        "admin_squash_allowed": False,
        "requires_human_risk_settlement": False,
        "requires_human_preapproval": False,
        "unresolved_dissent": False,
        "checks_summary": "1 failing / 6 required",
        "counted_model_families": [],
        "reasons": ["model quorum incomplete: 0/2 signal(s)"],
    }

    def merge_packet_builder(**_: object) -> dict[str, object]:
        return {
            "queue_pressure": {"current_open_prs": 11, "cap": 6, "active": True},
            "admin_squash_order": [],
            "human_risk_settlement_required": [],
            "not_ready": [8899],
            "entries": [*(ready_entry(8800 + index) for index in range(10)), blocked],
        }

    report = gather_founder_status(
        repo_root=str(tmp_path),
        limit=12,
        health_gatherer=lambda **_: _health(),
        merge_packet_builder=merge_packet_builder,
    )

    assert all(entry["pr_number"] != 8899 for entry in report["queue"]["top_entries"])
    assert report["queue"]["not_ready_entries"][0]["pr_number"] == 8899
    assert report["next_action"]["kind"] == "queue_blocker"
    assert "PR #8899" in report["next_action"]["summary"]
    assert "model quorum incomplete" in report["next_action"]["detail"]


def test_founder_status_repairs_not_ready_before_human_settlement(tmp_path: Path) -> None:
    def merge_packet_builder(**_: object) -> dict[str, object]:
        return {
            "queue_pressure": {"current_open_prs": 1, "cap": 6, "active": False},
            "admin_squash_order": [],
            "human_risk_settlement_required": [8945],
            "not_ready": [8945],
            "entries": [
                {
                    "pr_number": 8945,
                    "title": "release workflow post-publish verification",
                    "url": "https://github.com/synaptent/aragora/pull/8945",
                    "head_sha": "abc123",
                    "tier": 4,
                    "status": "repair_or_wait",
                    "verdict": "not_ready_for_settlement",
                    "admin_squash_allowed": False,
                    "requires_human_risk_settlement": True,
                    "requires_human_preapproval": True,
                    "unresolved_dissent": False,
                    "checks_summary": "3 failing / 32 total",
                    "counted_model_families": [],
                    "reasons": [
                        "workflow/deploy/destructive surface touched",
                        "checks are failing; repair before settlement",
                    ],
                }
            ],
        }

    report = gather_founder_status(
        repo_root=str(tmp_path),
        health_gatherer=lambda **_: _health(),
        merge_packet_builder=merge_packet_builder,
    )

    assert report["next_action"]["kind"] == "queue_blocker"
    assert "PR #8945" in report["next_action"]["summary"]
    assert "workflow/deploy/destructive surface touched" in report["next_action"]["detail"]


def test_founder_status_keeps_human_settlement_when_ready(tmp_path: Path) -> None:
    def merge_packet_builder(**_: object) -> dict[str, object]:
        return {
            "queue_pressure": {"current_open_prs": 1, "cap": 6, "active": False},
            "admin_squash_order": [],
            "human_risk_settlement_required": [8756],
            "not_ready": [],
            "entries": [
                {
                    "pr_number": 8756,
                    "title": "operator-approved advisory settlement records",
                    "url": "https://github.com/synaptent/aragora/pull/8756",
                    "head_sha": "abc123",
                    "tier": 4,
                    "status": "needs_human_settlement",
                    "verdict": "requires_human_risk_settlement",
                    "admin_squash_allowed": False,
                    "requires_human_risk_settlement": True,
                    "requires_human_preapproval": True,
                    "unresolved_dissent": False,
                    "checks_summary": "all non-quorum required checks pass",
                    "counted_model_families": [],
                    "reasons": ["human risk settlement required"],
                }
            ],
        }

    report = gather_founder_status(
        repo_root=str(tmp_path),
        health_gatherer=lambda **_: _health(),
        merge_packet_builder=merge_packet_builder,
    )

    assert report["next_action"]["kind"] == "human_settlement"
    assert "PR #8756" in report["next_action"]["summary"]


def test_founder_status_degrades_on_merge_packet_transport_error(tmp_path: Path) -> None:
    def broken_builder(**_: object) -> dict[str, object]:
        raise RuntimeError("organization access is disabled")

    report = gather_founder_status(
        repo_root=str(tmp_path),
        health_gatherer=lambda **_: _health(),
        merge_packet_builder=broken_builder,
    )

    assert report["queue"]["transport_status"] == "blocked"
    assert "organization access is disabled" in report["queue"]["transport_error"]
    assert report["next_action"]["kind"] == "repair_transport"


def test_status_parser_accepts_founder_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "status",
            "--founder",
            "--json",
            "--repo",
            "synaptent/aragora",
            "--limit",
            "3",
            "--repo-root",
            "/tmp/repo",
        ]
    )

    assert args.command == "status"
    assert args.founder is True
    assert args.json is True
    assert args.repo == "synaptent/aragora"
    assert args.limit == 3
    assert args.repo_root == "/tmp/repo"


def test_cmd_status_delegates_founder(monkeypatch, capsys) -> None:
    seen: dict[str, bool] = {}

    def fake_gather(**_: object) -> dict[str, object]:
        seen["called"] = True
        return {
            "generated_at": "2026-07-04T00:00:00+00:00",
            "repo_root": "/tmp/repo",
            "queue": {
                "transport_status": "ok",
                "queue_pressure": {"current_open_prs": 0, "cap": 6, "active": False},
                "status_counts": {},
                "admin_squash_order": [],
                "human_risk_settlement_required": [],
                "not_ready": [],
                "top_entries": [],
            },
            "proof_loop": {"overall_status": "fresh", "surfaces": []},
            "latest_brief": {"path": None, "age_hours": None, "preview": ""},
            "next_action": {"summary": "No queue action.", "detail": ""},
        }

    monkeypatch.setattr(
        "aragora.cli.commands.founder_status.gather_founder_status",
        fake_gather,
    )

    result = cmd_status(argparse.Namespace(founder=True, json=False, limit=1))

    assert result == 0
    assert seen["called"] is True
    assert "Aragora Founder Status" in capsys.readouterr().out


def test_status_json_requires_founder(capsys) -> None:
    result = cmd_status(argparse.Namespace(founder=False, json=True))

    assert result == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "status --json requires --founder" in captured.err
