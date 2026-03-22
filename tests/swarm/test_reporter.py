"""Tests for SwarmReporter and SwarmReport."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from aragora.swarm.reporter import SwarmReport, SwarmReporter, build_integrator_view
from aragora.swarm.spec import SwarmSpec


@dataclass
class MockAssignment:
    """Mock assignment for testing."""

    subtask_title: str = "Test task"
    status: str = "completed"
    error: str = ""


@dataclass
class MockResult:
    """Mock OrchestrationResult for testing."""

    total_subtasks: int = 5
    completed_subtasks: int = 4
    failed_subtasks: int = 1
    skipped_subtasks: int = 0
    assignments: list[Any] = field(default_factory=list)
    total_cost_usd: float = 2.50


class TestSwarmReport:
    """Test SwarmReport rendering."""

    def test_plain_text_success(self):
        report = SwarmReport(
            success=True,
            summary="Everything worked great.",
            what_was_done=["Fixed the login", "Updated the tests"],
            what_failed=[],
            what_to_do_next=["Review changes"],
            duration_seconds=120.0,
            budget_spent_usd=1.50,
        )
        text = report.to_plain_text()
        assert "SUCCESS" in text
        assert "Everything worked great." in text
        assert "Fixed the login" in text
        assert "Updated the tests" in text
        assert "$1.50" in text
        assert "2m 0s" in text

    def test_plain_text_failure(self):
        report = SwarmReport(
            success=False,
            summary="Some tasks failed.",
            what_was_done=["Task A"],
            what_failed=["Task B: timeout"],
            what_to_do_next=["Retry task B"],
        )
        text = report.to_plain_text()
        assert "ISSUES" in text
        assert "Task B: timeout" in text

    def test_markdown_rendering(self):
        report = SwarmReport(
            success=True,
            summary="All good.",
            what_was_done=["Item 1"],
            what_to_do_next=["Review"],
        )
        md = report.to_markdown()
        assert "# Swarm Report" in md
        assert "- Item 1" in md

    def test_to_dict(self):
        spec = SwarmSpec(raw_goal="test")
        report = SwarmReport(
            success=True,
            summary="Test",
            spec=spec,
            duration_seconds=60.0,
        )
        data = report.to_dict()
        assert data["success"] is True
        assert data["summary"] == "Test"
        assert data["spec"]["raw_goal"] == "test"

    def test_to_dict_without_spec(self):
        report = SwarmReport(success=False, summary="No spec")
        data = report.to_dict()
        assert data["spec"] is None


class TestSwarmReporter:
    """Test SwarmReporter template-based generation."""

    @pytest.mark.asyncio
    async def test_template_report_success(self):
        spec = SwarmSpec(refined_goal="Fix all bugs")
        result = MockResult(
            total_subtasks=3,
            completed_subtasks=3,
            failed_subtasks=0,
            assignments=[
                MockAssignment(subtask_title="Fix bug A", status="completed"),
                MockAssignment(subtask_title="Fix bug B", status="completed"),
                MockAssignment(subtask_title="Fix bug C", status="completed"),
            ],
        )

        reporter = SwarmReporter()
        report = await reporter.generate(spec, result, duration_seconds=90.0)

        assert report.success is True
        assert "great news" in report.summary.lower()
        assert "3" in report.summary
        assert len(report.what_was_done) == 3

    @pytest.mark.asyncio
    async def test_template_report_partial_failure(self):
        spec = SwarmSpec(refined_goal="Improve everything")
        result = MockResult(
            total_subtasks=5,
            completed_subtasks=3,
            failed_subtasks=2,
            assignments=[
                MockAssignment(subtask_title="Task A", status="completed"),
                MockAssignment(subtask_title="Task B", status="failed", error="timeout"),
            ],
        )

        reporter = SwarmReporter()
        report = await reporter.generate(spec, result)

        assert report.success is False
        assert "3" in report.summary and "5" in report.summary

    @pytest.mark.asyncio
    async def test_template_report_total_failure(self):
        spec = SwarmSpec(refined_goal="Do stuff")
        result = MockResult(
            total_subtasks=2,
            completed_subtasks=0,
            failed_subtasks=2,
        )

        reporter = SwarmReporter()
        report = await reporter.generate(spec, result)

        assert report.success is False
        assert "wasn't able to complete" in report.summary

    @pytest.mark.asyncio
    async def test_template_report_with_skipped(self):
        spec = SwarmSpec(refined_goal="Partial work")
        result = MockResult(
            total_subtasks=4,
            completed_subtasks=2,
            failed_subtasks=0,
            skipped_subtasks=2,
        )

        reporter = SwarmReporter()
        report = await reporter.generate(spec, result)

        assert report.success is True
        assert any("skipped" in item.lower() for item in report.what_to_do_next)

    @pytest.mark.asyncio
    async def test_budget_extraction(self):
        spec = SwarmSpec(raw_goal="Budget test")
        result = MockResult(total_cost_usd=3.75)

        reporter = SwarmReporter()
        report = await reporter.generate(spec, result)

        assert report.budget_spent_usd == 3.75

    @pytest.mark.asyncio
    async def test_empty_result(self):
        spec = SwarmSpec(raw_goal="Nothing happened")
        result = MockResult(total_subtasks=0, completed_subtasks=0, failed_subtasks=0)

        reporter = SwarmReporter()
        report = await reporter.generate(spec, result)

        assert report.success is False


class TestIntegratorView:
    def test_build_integrator_view_prefers_canonical_task_lane(self):
        now = datetime(2026, 3, 18, 12, 0, tzinfo=UTC)
        payload = build_integrator_view(
            runs=[
                {
                    "run_id": "run-1",
                    "status": "active",
                    "goal": "Ship the integrator lane",
                    "work_orders": [
                        {
                            "work_order_id": "wo-1",
                            "title": "Build integrator lane",
                            "status": "completed",
                            "branch": "codex/integrator-lane",
                            "worktree_path": "/tmp/repo/.worktrees/integrator",
                            "target_agent": "codex",
                            "reviewer_agent": "claude",
                        }
                    ],
                }
            ],
            worktrees=[
                {
                    "session_id": "sess-1",
                    "path": "/tmp/repo/.worktrees/integrator",
                    "branch": "codex/integrator-lane",
                    "has_lock": True,
                    "pid_alive": True,
                    "agent": "codex",
                    "last_activity": (now - timedelta(minutes=2)).isoformat(),
                }
            ],
            claims=[{"session_id": "sess-1", "path": "aragora/swarm/reporter.py"}],
            merge_queue=[
                {
                    "id": "mq-1",
                    "branch": "codex/integrator-lane",
                    "session_id": "sess-1",
                    "status": "needs_human",
                    "metadata": {
                        "receipt_id": "rcpt-1",
                        "task_id": "wo-1",
                        "pr_url": "https://github.com/synaptent/aragora/pull/1051",
                        "pr_number": 1051,
                    },
                }
            ],
            coordination={
                "counts": {"active_leases": 1},
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-1",
                            "task_id": "wo-1",
                            "run_id": "run-1",
                            "goal": "Ship the integrator lane",
                            "title": "Build integrator lane",
                            "status": "completed",
                            "owner_agent": "codex",
                            "reviewer_agent": "claude",
                            "owner_session_id": "sess-1",
                            "branch": "codex/integrator-lane",
                            "worktree_path": "/tmp/repo/.worktrees/integrator",
                            "lease_id": "lease-1",
                            "receipt_id": "rcpt-1",
                            "allowed_paths": ["aragora/swarm/reporter.py"],
                            "updated_at": (now - timedelta(minutes=3)).isoformat(),
                        }
                    ],
                    "leases": [
                        {
                            "lease_id": "lease-1",
                            "task_id": "wo-1",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-1",
                            "branch": "codex/integrator-lane",
                            "worktree_path": "/tmp/repo/.worktrees/integrator",
                            "claimed_paths": ["aragora/swarm/reporter.py"],
                            "status": "completed",
                            "updated_at": (now - timedelta(minutes=4)).isoformat(),
                            "expires_at": (now + timedelta(hours=2)).isoformat(),
                            "metadata": {"task_key": "run-1:wo-1"},
                        }
                    ],
                    "completion_receipts": [
                        {
                            "receipt_id": "rcpt-1",
                            "lease_id": "lease-1",
                            "task_id": "wo-1",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-1",
                            "branch": "codex/integrator-lane",
                            "worktree_path": "/tmp/repo/.worktrees/integrator",
                            "base_sha": "abc123base",
                            "head_sha": "def456head",
                            "commit_shas": ["def456head"],
                            "changed_paths": [
                                "aragora/swarm/reporter.py",
                                "tests/swarm/test_reporter.py",
                            ],
                            "confidence": 0.94,
                            "outcome": "deliverable_created",
                            "tests_run": ["python -m pytest tests/swarm/test_reporter.py -q"],
                            "validations_run": [
                                "python -m pytest tests/swarm/test_reporter.py -q",
                                "ruff check aragora/swarm/reporter.py",
                            ],
                            "risks": ["Need integrator review before merge"],
                            "created_at": (now - timedelta(minutes=5)).isoformat(),
                            "artifact_hash": "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                            "metadata": {
                                "task_key": "run-1:wo-1",
                                "reviewer_agent": "claude",
                            },
                            "pr_url": "https://github.com/synaptent/aragora/pull/1051",
                            "pr_number": 1051,
                        }
                    ],
                    "integration_decisions": [
                        {
                            "decision_id": "dec-1",
                            "lease_id": "lease-1",
                            "receipt_id": "rcpt-1",
                            "decision": "pending_review",
                            "target_branch": "main",
                            "rationale": "Awaiting integrator review",
                            "chosen_commits": ["abc12345"],
                            "followups": ["check merge gate"],
                            "decided_by": "system",
                            "created_at": (now - timedelta(minutes=4)).isoformat(),
                        }
                    ],
                    "salvage_candidates": [],
                },
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert payload["summary"]["canonical_lanes"] == 1
        assert payload["summary"]["decision_lanes"] == 1
        assert payload["alerts"]["needs_decision"][0]["lane_id"] == "run-1:wo-1"
        assert lane["source"] == "task"
        assert lane["canonical_lane"] is True
        assert lane["task_key"] == "run-1:wo-1"
        assert lane["merge_readiness"] == "review"
        assert lane["lane_health"] == "healthy"
        assert lane["receipt_summary"]["status"] == "present"
        assert lane["receipt_summary"]["task_id"] == "wo-1"
        assert lane["receipt_summary"]["lease_id"] == "lease-1"
        assert lane["receipt_summary"]["agent_id"] == "codex"
        assert lane["receipt_summary"]["base_sha"] == "abc123base"
        assert lane["receipt_summary"]["head_sha"] == "def456head"
        assert lane["receipt_summary"]["changed_files"] == [
            "aragora/swarm/reporter.py",
            "tests/swarm/test_reporter.py",
        ]
        assert lane["receipt_summary"]["risks"] == ["Need integrator review before merge"]
        assert lane["receipt_summary"]["artifact_hash"]
        assert lane["integration_decision"] == "pending_review"
        assert lane["pr"]["number"] == 1051
        assert lane["available_actions"][:3] == ["merge", "cherry_pick", "request_changes"]

    def test_build_integrator_view_tracks_pending_receipt_provenance_for_active_lane(self):
        now = datetime(2026, 3, 18, 12, 0, tzinfo=UTC)
        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-active",
                            "task_id": "wo-active",
                            "run_id": "run-1",
                            "goal": "Keep the active lane visible",
                            "title": "Keep the active lane visible",
                            "status": "active",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-active",
                            "branch": "codex/active-lane",
                            "worktree_path": "/tmp/repo/.worktrees/active",
                            "lease_id": "lease-active",
                            "updated_at": (now - timedelta(minutes=2)).isoformat(),
                        }
                    ],
                    "leases": [
                        {
                            "lease_id": "lease-active",
                            "task_id": "wo-active",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-active",
                            "branch": "codex/active-lane",
                            "worktree_path": "/tmp/repo/.worktrees/active",
                            "status": "active",
                            "updated_at": (now - timedelta(minutes=2)).isoformat(),
                            "expires_at": (now + timedelta(hours=1)).isoformat(),
                            "metadata": {"base_sha": "base-active"},
                        }
                    ],
                    "completion_receipts": [],
                    "integration_decisions": [],
                    "salvage_candidates": [],
                }
            },
            merge_queue=[
                {
                    "id": "mq-active",
                    "branch": "codex/active-lane",
                    "session_id": "sess-active",
                    "status": "queued",
                    "metadata": {
                        "lease_id": "lease-active",
                        "task_id": "wo-active",
                        "base_sha": "base-active",
                        "head_sha": "head-active",
                        "changed_paths": [
                            "aragora/cli/commands/worktree.py",
                            "tests/cli/test_worktree_command.py",
                        ],
                    },
                }
            ],
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["merge_readiness"] == "in_progress"
        assert lane["receipt_summary"]["status"] == "pending"
        assert lane["receipt_summary"]["task_id"] == "wo-active"
        assert lane["receipt_summary"]["lease_id"] == "lease-active"
        assert lane["receipt_summary"]["agent_id"] == "codex"
        assert lane["receipt_summary"]["base_sha"] == "base-active"
        assert lane["receipt_summary"]["head_sha"] == "head-active"
        assert lane["receipt_summary"]["changed_files"] == [
            "aragora/cli/commands/worktree.py",
            "tests/cli/test_worktree_command.py",
        ]

    def test_build_integrator_view_marks_expired_and_superseded_lanes(self):
        now = datetime(2026, 3, 18, 12, 0, tzinfo=UTC)
        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-expired",
                            "task_id": "wo-expired",
                            "run_id": "run-1",
                            "goal": "Recover expired lane",
                            "title": "Recover expired lane",
                            "status": "timed_out",
                            "owner_session_id": "sess-expired",
                            "branch": "codex/expired",
                            "worktree_path": "/tmp/repo/.worktrees/expired",
                            "lease_id": "lease-expired",
                            "updated_at": (now - timedelta(hours=2)).isoformat(),
                        },
                        {
                            "task_key": "run-1:wo-superseded",
                            "task_id": "wo-superseded",
                            "run_id": "run-1",
                            "goal": "Replace stale lane",
                            "title": "Replace stale lane",
                            "status": "discarded",
                            "owner_session_id": "sess-old",
                            "branch": "codex/old-lane",
                            "worktree_path": "/tmp/repo/.worktrees/old",
                            "lease_id": "lease-old",
                            "receipt_id": "rcpt-old",
                            "updated_at": (now - timedelta(minutes=45)).isoformat(),
                        },
                    ],
                    "leases": [
                        {
                            "lease_id": "lease-expired",
                            "task_id": "wo-expired",
                            "owner_session_id": "sess-expired",
                            "branch": "codex/expired",
                            "worktree_path": "/tmp/repo/.worktrees/expired",
                            "status": "expired",
                            "updated_at": (now - timedelta(hours=2)).isoformat(),
                            "expires_at": (now - timedelta(hours=1)).isoformat(),
                        },
                        {
                            "lease_id": "lease-old",
                            "task_id": "wo-superseded",
                            "owner_session_id": "sess-old",
                            "branch": "codex/old-lane",
                            "worktree_path": "/tmp/repo/.worktrees/old",
                            "status": "completed",
                            "updated_at": (now - timedelta(hours=1)).isoformat(),
                            "expires_at": (now + timedelta(hours=1)).isoformat(),
                        },
                    ],
                    "completion_receipts": [
                        {
                            "receipt_id": "rcpt-old",
                            "lease_id": "lease-old",
                            "task_id": "wo-superseded",
                            "owner_session_id": "sess-old",
                            "branch": "codex/old-lane",
                            "worktree_path": "/tmp/repo/.worktrees/old",
                            "created_at": (now - timedelta(hours=1)).isoformat(),
                        }
                    ],
                    "integration_decisions": [
                        {
                            "decision_id": "dec-old",
                            "lease_id": "lease-old",
                            "receipt_id": "rcpt-old",
                            "decision": "discard",
                            "target_branch": "main",
                            "rationale": "Superseded by a cleaner replacement lane",
                            "created_at": (now - timedelta(minutes=50)).isoformat(),
                        }
                    ],
                    "salvage_candidates": [
                        {
                            "candidate_id": "salv-1",
                            "branch": "codex/expired",
                            "worktree_path": "/tmp/repo/.worktrees/expired",
                            "status": "detected",
                            "updated_at": (now - timedelta(minutes=20)).isoformat(),
                        }
                    ],
                }
            },
            now=now,
        )

        lanes = {lane["task_id"]: lane for lane in payload["lanes"]}
        expired = lanes["wo-expired"]
        superseded = lanes["wo-superseded"]

        assert payload["summary"]["expired_lanes"] == 1
        assert payload["summary"]["superseded_lanes"] == 1
        assert expired["lane_health"] == "expired"
        assert expired["available_actions"][:2] == ["salvage", "reassign"]
        assert expired["salvage_candidate_id"] == "salv-1"
        assert superseded["merge_readiness"] == "superseded"
        assert superseded["lane_health"] == "superseded"
        assert superseded["available_actions"] == ["archive"]

    def test_build_integrator_view_blocks_stale_colliding_lane_without_receipt(self):
        now = datetime(2026, 3, 18, 12, 0, tzinfo=UTC)
        payload = build_integrator_view(
            worktrees=[
                {
                    "session_id": "sess-1",
                    "path": "/tmp/repo/.worktrees/conflict-a",
                    "branch": "codex/collision",
                    "has_lock": True,
                    "pid_alive": True,
                    "agent": "codex",
                    "last_activity": (now - timedelta(minutes=45)).isoformat(),
                },
                {
                    "session_id": "sess-2",
                    "path": "/tmp/repo/.worktrees/conflict-b",
                    "branch": "codex/collision",
                    "has_lock": True,
                    "pid_alive": True,
                    "agent": "claude",
                    "last_activity": (now - timedelta(minutes=1)).isoformat(),
                },
            ],
            claims=[
                {"session_id": "sess-1", "path": "aragora/swarm/reporter.py"},
                {"session_id": "sess-2", "path": "aragora/swarm/reporter.py"},
            ],
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-collision",
                            "task_id": "wo-collision",
                            "run_id": "run-1",
                            "goal": "Resolve the colliding lane",
                            "title": "Resolve the colliding lane",
                            "status": "completed",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-1",
                            "branch": "codex/collision",
                            "worktree_path": "/tmp/repo/.worktrees/conflict-a",
                            "lease_id": "lease-collision",
                            "updated_at": (now - timedelta(minutes=45)).isoformat(),
                        }
                    ],
                    "leases": [
                        {
                            "lease_id": "lease-collision",
                            "task_id": "wo-collision",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-1",
                            "branch": "codex/collision",
                            "worktree_path": "/tmp/repo/.worktrees/conflict-a",
                            "claimed_paths": ["aragora/swarm/reporter.py"],
                            "status": "active",
                            "updated_at": (now - timedelta(minutes=45)).isoformat(),
                            "expires_at": (now + timedelta(hours=1)).isoformat(),
                        }
                    ],
                    "completion_receipts": [],
                    "integration_decisions": [],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = next(item for item in payload["lanes"] if item.get("task_id") == "wo-collision")
        assert payload["summary"]["blocked_lanes"] >= 1
        assert payload["summary"]["collision_lanes"] >= 1
        assert payload["summary"]["stale_heartbeat_lanes"] >= 1
        assert payload["summary"]["missing_receipt_lanes"] >= 1
        assert lane["merge_readiness"] == "blocked"
        assert lane["lane_health"] == "stalled"
        assert lane["lease_health"] == "stalled"
        assert lane["missing_receipt"] is True
        assert lane["receipt_summary"]["status"] == "missing"
        assert lane["stale_heartbeat"] is True
        assert lane["collisions"] == [
            "branch:codex/collision",
            "path:aragora/swarm/reporter.py",
        ]
        assert "attach_receipt" in lane["available_actions"]
        assert "supersede" in lane["available_actions"]

    def test_build_integrator_view_extracts_adopted_pr_reference(self):
        payload = build_integrator_view(
            runs=[
                {
                    "run_id": "run-1",
                    "goal": "Surface adopted PR evidence",
                    "work_orders": [
                        {
                            "work_order_id": "wo-1",
                            "title": "Carry forward the adopted PR",
                            "status": "completed",
                            "branch": "codex/adopted-pr",
                            "worktree_path": "/tmp/repo/.worktrees/adopted-pr",
                            "target_agent": "codex",
                            "adopted_pr": "#1057",
                        }
                    ],
                }
            ]
        )

        lane = payload["lanes"][0]
        assert lane["pr"] == {"url": None, "number": 1057, "reference": "#1057"}
        assert lane["missing_receipt"] is True
