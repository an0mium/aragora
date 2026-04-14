"""Tests for SwarmReporter and SwarmReport."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import pytest

from aragora.swarm.reporter import (
    SwarmReport,
    SwarmReporter,
    build_boss_payload,
    build_integrator_view,
    render_boss_text,
)
from aragora.swarm.lane_telemetry import LaneTelemetryCollector, LaneTelemetryRecord
from aragora.swarm.spec import SwarmSpec

UTC = timezone.utc


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
        now = datetime(2026, 3, 18, 12, 0, tzinfo=timezone.utc)
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
        now = datetime(2026, 3, 18, 12, 0, tzinfo=timezone.utc)
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
        now = datetime(2026, 3, 18, 12, 0, tzinfo=timezone.utc)
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

    def test_build_integrator_view_excludes_superseded_collision_from_summary(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)
        payload = build_integrator_view(
            worktrees=[
                {
                    "session_id": "sess-1",
                    "path": "/tmp/repo/.worktrees/conflict-a",
                    "branch": "codex/collision",
                    "has_lock": False,
                    "pid_alive": False,
                    "agent": "codex",
                    "last_activity": (now - timedelta(hours=2)).isoformat(),
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
                            "title": "Archived colliding lane",
                            "status": "discarded",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-1",
                            "branch": "codex/collision",
                            "worktree_path": "/tmp/repo/.worktrees/conflict-a",
                            "updated_at": (now - timedelta(hours=2)).isoformat(),
                        }
                    ],
                    "leases": [],
                    "completion_receipts": [],
                    "integration_decisions": [],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["superseded"] is True
        assert lane["merge_readiness"] == "superseded"
        collision_lane_ids = {item["lane_id"] for item in payload["alerts"]["collisions"]}
        assert lane["lane_id"] not in collision_lane_ids

    def test_build_integrator_view_treats_retired_lane_as_superseded(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)
        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-retired",
                            "task_id": "wo-retired",
                            "run_id": "run-1",
                            "title": "Retired dogfood lane",
                            "status": "failed",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-retired",
                            "branch": "codex/retired-lane",
                            "worktree_path": "/tmp/repo/.worktrees/retired",
                            "blockers": [
                                "Retired after first dogfood attempt exposed launcher/reconcile bugs fixed in later commits."
                            ],
                            "updated_at": (now - timedelta(hours=2)).isoformat(),
                        }
                    ],
                    "leases": [],
                    "completion_receipts": [],
                    "integration_decisions": [],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["superseded"] is True
        assert lane["merge_readiness"] == "superseded"
        assert lane["lane_health"] == "superseded"
        assert lane["next_action"] == "Archive the superseded lane and keep the canonical lane."
        assert payload["summary"]["stalled_lanes"] == 0
        assert payload["summary"]["superseded_lanes"] == 1

    def test_build_integrator_view_ignores_discarded_duplicate_work_order_in_branch_counts(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)
        payload = build_integrator_view(
            runs=[
                {
                    "run_id": "run-1",
                    "status": "completed",
                    "goal": "Collapse duplicate branch family",
                    "work_orders": [
                        {
                            "work_order_id": "wo-active",
                            "title": "Active branch lane",
                            "status": "completed",
                            "branch": "codex/shared-branch",
                            "commit_shas": ["abc123"],
                            "head_sha": "abc123",
                        },
                        {
                            "work_order_id": "wo-archived",
                            "title": "Archived branch sibling",
                            "status": "discarded",
                            "branch": "codex/shared-branch",
                            "commit_shas": ["abc123"],
                            "head_sha": "abc123",
                        },
                    ],
                }
            ],
            now=now,
        )

        active_lane = next(
            item for item in payload["lanes"] if item.get("work_order_id") == "wo-active"
        )
        archived_lane = next(
            item for item in payload["lanes"] if item.get("work_order_id") == "wo-archived"
        )

        assert active_lane["collisions"] == []
        assert archived_lane["merge_readiness"] == "superseded"
        assert payload["summary"]["collision_lanes"] == 0

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

    def test_build_integrator_view_includes_lane_telemetry_summary(self):
        collector = LaneTelemetryCollector(db_path=":memory:")
        now = datetime.now(UTC).timestamp()
        collector.record_lane(
            LaneTelemetryRecord(
                lane_kind="boss_dispatch",
                lane_id="boss-1",
                terminal_outcome="deliverable_created",
                deliverable_type="branch",
                receipt_id="rcpt-1",
                timestamp=now,
                false_success_candidate=False,
            )
        )
        collector.record_lane(
            LaneTelemetryRecord(
                lane_kind="supervisor_work_order",
                lane_id="run-1:wo-1",
                terminal_outcome="clean_exit_no_deliverable",
                human_intervention_required=True,
                false_success_candidate=True,
                timestamp=now,
            )
        )

        with patch("aragora.swarm.reporter._LANE_TELEMETRY", collector):
            payload = build_integrator_view()

        assert payload["telemetry"] == {
            "throughput_7d": 2,
            "success_rate_7d": 0.5,
            "false_success_candidates_7d": 1,
            "human_intervention_rate_7d": 0.5,
            "merge_yield_7d": 0.0,
            "avg_time_to_pr_seconds_7d": 0.0,
            "avg_time_to_merge_seconds_7d": 0.0,
        }

    def test_build_integrator_view_syncs_merged_lane_back_into_telemetry(self):
        collector = LaneTelemetryCollector(db_path=":memory:")
        now = datetime(2026, 3, 29, 12, 0, tzinfo=UTC)
        collector.record_lane(
            LaneTelemetryRecord(
                lane_kind="supervisor_work_order",
                lane_id="run-1:wo-1",
                run_id="run-1",
                task_id="wo-1",
                work_order_id="wo-1",
                terminal_outcome="deliverable_created",
                worker_outcome="completed",
                deliverable_type="branch",
                receipt_id="rcpt-1",
                duration_seconds=60.0,
                timestamp=now.timestamp(),
            )
        )

        with patch("aragora.swarm.reporter._LANE_TELEMETRY", collector):
            build_integrator_view(
                coordination={
                    "integrator": {
                        "developer_tasks": [
                            {
                                "task_key": "run-1:wo-1",
                                "task_id": "wo-1",
                                "run_id": "run-1",
                                "status": "completed",
                                "title": "Merge the lane",
                                "owner_agent": "codex",
                                "owner_session_id": "sess-1",
                                "branch": "codex/merged-lane",
                                "worktree_path": "/tmp/repo/.worktrees/merged",
                                "lease_id": "lease-1",
                                "receipt_id": "rcpt-1",
                                "updated_at": now.isoformat(),
                            }
                        ],
                        "leases": [
                            {
                                "lease_id": "lease-1",
                                "task_id": "wo-1",
                                "owner_agent": "codex",
                                "owner_session_id": "sess-1",
                                "branch": "codex/merged-lane",
                                "worktree_path": "/tmp/repo/.worktrees/merged",
                                "status": "completed",
                                "updated_at": now.isoformat(),
                                "expires_at": (now + timedelta(hours=1)).isoformat(),
                            }
                        ],
                        "completion_receipts": [
                            {
                                "receipt_id": "rcpt-1",
                                "lease_id": "lease-1",
                                "task_id": "wo-1",
                                "owner_agent": "codex",
                                "owner_session_id": "sess-1",
                                "branch": "codex/merged-lane",
                                "worktree_path": "/tmp/repo/.worktrees/merged",
                                "base_sha": "abc123base",
                                "head_sha": "def456head",
                                "commit_shas": ["def456head"],
                                "created_at": (now - timedelta(minutes=5)).isoformat(),
                            }
                        ],
                        "integration_decisions": [
                            {
                                "decision_id": "dec-1",
                                "lease_id": "lease-1",
                                "receipt_id": "rcpt-1",
                                "decision": "merge",
                                "target_branch": "main",
                                "chosen_commits": ["mergeabc123"],
                                "created_at": now.isoformat(),
                            }
                        ],
                        "salvage_candidates": [],
                    }
                },
                merge_queue=[
                    {
                        "id": "mq-merged",
                        "branch": "codex/merged-lane",
                        "session_id": "sess-1",
                        "status": "merged",
                        "updated_at": now.isoformat(),
                        "metadata": {
                            "receipt_id": "rcpt-1",
                            "task_id": "wo-1",
                            "pr_url": "https://github.com/synaptent/aragora/pull/1200",
                            "pr_number": 1200,
                            "merge_sha": "mergeabc123",
                        },
                    }
                ],
                now=now,
            )

        record = collector.get_lane("supervisor_work_order", "run-1:wo-1")
        assert record is not None
        assert record.merged_at == now.isoformat()
        assert record.merge_ref == "mergeabc123"
        assert record.pr_number == 1200
        assert record.time_to_merge_seconds == 300.0

    def test_build_integrator_view_never_marks_ready_without_deliverable_and_receipt(self):
        now = datetime(2026, 3, 29, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-empty",
                            "task_id": "wo-empty",
                            "run_id": "run-1",
                            "status": "completed",
                            "title": "Completed with no deliverable",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-empty",
                            "branch": "codex/empty-lane",
                            "worktree_path": "/tmp/repo/.worktrees/empty",
                            "updated_at": now.isoformat(),
                        }
                    ],
                    "leases": [],
                    "completion_receipts": [],
                    "integration_decisions": [],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["terminal_outcome"] == "clean_exit_no_deliverable"
        assert lane["deliverable_type"] is None
        assert lane["missing_receipt"] is False
        assert lane["merge_readiness"] == "blocked"
        assert lane["lane_health"] == "blocked"
        assert (
            lane["next_action"]
            == "Inspect why the lane produced no concrete deliverable before rerunning it."
        )

    def test_build_integrator_view_does_not_expect_receipt_for_pre_dispatch_needs_human_lane(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-lease-fail",
                            "task_id": "wo-lease-fail",
                            "run_id": "run-1",
                            "status": "needs_human",
                            "title": "Lease failed before dispatch",
                            "owner_agent": "codex",
                            "branch": "codex/lease-fail",
                            "worktree_path": "/tmp/repo/.worktrees/lease-fail",
                            "failure_reason": "work_order_leasing_failed",
                            "dispatch_error": "autopilot ensure failed",
                            "updated_at": now.isoformat(),
                        }
                    ],
                    "leases": [],
                    "completion_receipts": [],
                    "integration_decisions": [],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["terminal_outcome"] == "needs_human"
        assert lane["deliverable_type"] is None
        assert lane["receipt_expected"] is False
        assert lane["missing_receipt"] is False
        assert "missing_receipt" not in lane["blockers"]
        assert "attach_receipt" not in lane["available_actions"]
        assert lane["merge_readiness"] == "blocked"
        assert lane["lane_health"] == "blocked"
        assert (
            lane["next_action"]
            == "Reconcile or regenerate the managed worktree, then requeue the lane."
        )

    def test_build_integrator_view_does_not_expect_receipt_for_reaped_stale_lease_lane(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-stale",
                            "task_id": "wo-stale",
                            "run_id": "run-1",
                            "status": "needs_human",
                            "title": "Stale lease lane",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-stale",
                            "branch": "codex/stale-lane",
                            "worktree_path": "/tmp/repo/.worktrees/stale",
                            "lease_id": "lease-stale",
                            "blockers": ["stale_lease_reaped"],
                            "updated_at": now.isoformat(),
                        }
                    ],
                    "leases": [
                        {
                            "lease_id": "lease-stale",
                            "task_id": "wo-stale",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-stale",
                            "branch": "codex/stale-lane",
                            "worktree_path": "/tmp/repo/.worktrees/stale",
                            "status": "expired",
                            "updated_at": (now - timedelta(hours=2)).isoformat(),
                            "expires_at": (now - timedelta(hours=1)).isoformat(),
                        }
                    ],
                    "completion_receipts": [],
                    "integration_decisions": [],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["terminal_outcome"] == "needs_human"
        assert lane["receipt_expected"] is False
        assert lane["missing_receipt"] is False
        assert "missing_receipt" not in lane["blockers"]
        assert lane["lane_health"] == "expired"
        assert lane["next_action"] == "Salvage or reassign the expired lane before resuming work."
        assert "attach_receipt" not in lane["available_actions"]

    def test_build_integrator_view_does_not_expect_receipt_for_merge_gate_no_deliverable_lane(
        self,
    ):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-merge-gate",
                            "task_id": "wo-merge-gate",
                            "run_id": "run-1",
                            "status": "needs_human",
                            "title": "Merge gate lane",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-merge-gate",
                            "branch": "codex/merge-gate",
                            "worktree_path": "/tmp/repo/.worktrees/merge-gate",
                            "lease_id": "lease-merge-gate",
                            "failure_reason": "merge_gate_failed",
                            "blockers": [
                                "merge gate blocked: missing verification plan for code-change lane"
                            ],
                            "updated_at": now.isoformat(),
                        }
                    ],
                    "leases": [
                        {
                            "lease_id": "lease-merge-gate",
                            "task_id": "wo-merge-gate",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-merge-gate",
                            "branch": "codex/merge-gate",
                            "worktree_path": "/tmp/repo/.worktrees/merge-gate",
                            "status": "released",
                            "updated_at": now.isoformat(),
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

        lane = payload["lanes"][0]
        assert lane["terminal_outcome"] == "needs_human"
        assert lane["deliverable_type"] is None
        assert lane["receipt_expected"] is False
        assert lane["missing_receipt"] is False
        assert "missing_receipt" not in lane["blockers"]
        assert "attach_receipt" not in lane["available_actions"]
        assert lane["merge_readiness"] == "blocked"

    def test_build_integrator_view_marks_undeclared_scope_receipt_gap_as_scope_issue(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-scope-gap",
                            "task_id": "wo-scope-gap",
                            "run_id": "run-1",
                            "status": "completed",
                            "title": "Historical deliverable lane",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-scope-gap",
                            "branch": "codex/scope-gap",
                            "worktree_path": "/tmp/repo/.worktrees/scope-gap",
                            "lease_id": "lease-scope-gap",
                            "commit_shas": ["abc12345"],
                            "changed_paths": ["tests/swarm/test_reporter.py"],
                            "blockers": ["missing_receipt"],
                            "updated_at": now.isoformat(),
                            "metadata": {
                                "last_scope_violation": {
                                    "detected_at": now.isoformat(),
                                    "changed_paths": ["tests/swarm/test_reporter.py"],
                                    "violations": [
                                        {
                                            "type": "undeclared_scope",
                                            "paths": ["tests/swarm/test_reporter.py"],
                                            "message": "Lease has no declared file scope.",
                                        }
                                    ],
                                }
                            },
                        }
                    ],
                    "leases": [
                        {
                            "lease_id": "lease-scope-gap",
                            "task_id": "wo-scope-gap",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-scope-gap",
                            "branch": "codex/scope-gap",
                            "worktree_path": "/tmp/repo/.worktrees/scope-gap",
                            "status": "expired",
                            "updated_at": (now - timedelta(hours=2)).isoformat(),
                            "expires_at": (now - timedelta(hours=1)).isoformat(),
                        }
                    ],
                    "completion_receipts": [],
                    "integration_decisions": [],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["terminal_outcome"] == "deliverable_created"
        assert lane["deliverable"] == {
            "type": "branch",
            "branch": "codex/scope-gap",
            "commit_shas": ["abc12345"],
            "work_order_id": None,
        }
        assert lane["deliverable_type"] == "branch"
        assert lane["receipt_expected"] is False
        assert lane["missing_receipt"] is False
        assert "missing_receipt" not in lane["blockers"]
        assert "receipt_backfill_blocked_undeclared_scope" in lane["blockers"]
        assert "scope_violation" in lane["blockers"]
        assert lane["merge_readiness"] == "blocked"
        assert (
            lane["next_action"]
            == "Declare the intended lane scope or discard the lane before receipt backfill and merge review."
        )

    def test_build_integrator_view_preserves_branch_deliverable_from_task_metadata(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-meta-branch",
                            "task_id": "wo-meta-branch",
                            "run_id": "run-1",
                            "status": "completed",
                            "title": "Historical branch-backed lane",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-meta-branch",
                            "branch": "codex/meta-branch",
                            "worktree_path": "/tmp/repo/.worktrees/meta-branch",
                            "lease_id": "lease-meta-branch",
                            "updated_at": now.isoformat(),
                            "metadata": {
                                "head_sha": "abc12345",
                                "commit_shas": ["abc12345"],
                                "changed_paths": ["tests/swarm/test_reporter.py"],
                                "worker_outcome": "completed",
                                "failure_reason": "scope_violation",
                            },
                        }
                    ],
                    "leases": [
                        {
                            "lease_id": "lease-meta-branch",
                            "task_id": "wo-meta-branch",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-meta-branch",
                            "branch": "codex/meta-branch",
                            "worktree_path": "/tmp/repo/.worktrees/meta-branch",
                            "status": "expired",
                            "updated_at": (now - timedelta(hours=2)).isoformat(),
                            "expires_at": (now - timedelta(hours=1)).isoformat(),
                        }
                    ],
                    "completion_receipts": [],
                    "integration_decisions": [],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["terminal_outcome"] == "deliverable_created"
        assert lane["deliverable"] == {
            "type": "branch",
            "branch": "codex/meta-branch",
            "commit_shas": ["abc12345"],
            "work_order_id": None,
        }
        assert lane["deliverable_type"] == "branch"
        assert lane["merge_readiness"] == "blocked"
        assert (
            lane["next_action"]
            == "Narrow the lane scope or split ownership before it can re-enter merge review."
        )

    def test_build_integrator_view_does_not_mark_branchless_commit_lane_ready(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-branchless",
                            "task_id": "wo-branchless",
                            "run_id": "run-1",
                            "status": "completed",
                            "title": "Branchless historical lane",
                            "owner_agent": "codex",
                            "updated_at": now.isoformat(),
                            "metadata": {
                                "head_sha": "abc12345",
                                "commit_shas": ["abc12345"],
                                "changed_paths": ["tests/swarm/test_reporter.py"],
                                "worker_outcome": "completed",
                            },
                        }
                    ],
                    "leases": [],
                    "completion_receipts": [],
                    "integration_decisions": [],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["terminal_outcome"] == "clean_exit_no_deliverable"
        assert lane["deliverable"] is None
        assert lane["deliverable_type"] is None
        assert lane["receipt_expected"] is False
        assert lane["merge_readiness"] == "blocked"
        assert (
            lane["next_action"]
            == "Inspect why the lane produced no concrete deliverable before rerunning it."
        )

    def test_build_integrator_view_does_not_expect_receipt_for_scope_violation_lane(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-scope-violation",
                            "task_id": "wo-scope-violation",
                            "run_id": "run-1",
                            "status": "scope_violation",
                            "title": "Scope violation lane",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-scope-violation",
                            "branch": "codex/scope-violation",
                            "worktree_path": "/tmp/repo/.worktrees/scope-violation",
                            "lease_id": "lease-scope-violation",
                            "blockers": [
                                "worker edited files outside permitted scope: tests/swarm/test_reporter.py"
                            ],
                            "updated_at": now.isoformat(),
                        }
                    ],
                    "leases": [
                        {
                            "lease_id": "lease-scope-violation",
                            "task_id": "wo-scope-violation",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-scope-violation",
                            "branch": "codex/scope-violation",
                            "worktree_path": "/tmp/repo/.worktrees/scope-violation",
                            "status": "released",
                            "updated_at": now.isoformat(),
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

        lane = payload["lanes"][0]
        assert lane["terminal_outcome"] == "blocked"
        assert lane["receipt_expected"] is False
        assert lane["missing_receipt"] is False
        assert "missing_receipt" not in lane["blockers"]
        assert "scope_violation" in lane["blockers"]
        assert lane["merge_readiness"] == "blocked"

    def test_build_integrator_view_keeps_receipt_backed_reaped_lane_reviewable(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-reaped-receipt",
                            "task_id": "wo-reaped-receipt",
                            "run_id": "run-1",
                            "status": "needs_human",
                            "title": "Receipt-backed reaped lane",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-reaped-receipt",
                            "branch": "codex/reaped-receipt",
                            "worktree_path": "/tmp/repo/.worktrees/reaped-receipt",
                            "lease_id": "lease-reaped-receipt",
                            "blockers": ["stale_lease_reaped"],
                            "updated_at": now.isoformat(),
                        }
                    ],
                    "leases": [
                        {
                            "lease_id": "lease-reaped-receipt",
                            "task_id": "wo-reaped-receipt",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-reaped-receipt",
                            "branch": "codex/reaped-receipt",
                            "worktree_path": "/tmp/repo/.worktrees/reaped-receipt",
                            "status": "expired",
                            "updated_at": (now - timedelta(hours=2)).isoformat(),
                            "expires_at": (now - timedelta(hours=1)).isoformat(),
                        }
                    ],
                    "completion_receipts": [
                        {
                            "receipt_id": "receipt-reaped",
                            "lease_id": "lease-reaped-receipt",
                            "task_id": "wo-reaped-receipt",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-reaped-receipt",
                            "branch": "codex/reaped-receipt",
                            "worktree_path": "/tmp/repo/.worktrees/reaped-receipt",
                            "commit_shas": ["abc123"],
                            "changed_paths": ["aragora/swarm/reporter.py"],
                            "tests_run": ["python -m pytest tests/swarm/test_reporter.py -q"],
                            "validations_run": [],
                            "assumptions": [],
                            "blockers": [],
                            "confidence": 0.9,
                            "created_at": now.isoformat(),
                            "artifact_hash": "hash-reaped",
                        }
                    ],
                    "integration_decisions": [],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["terminal_outcome"] == "deliverable_created"
        assert lane["receipt_id"] == "receipt-reaped"
        assert lane["merge_readiness"] == "review"
        assert lane["lane_health"] == "healthy"
        assert "stale_lease_reaped" not in lane["blockers"]
        assert (
            lane["next_action"] == "Review the validated lane and decide whether it should merge."
        )

    def test_build_integrator_view_ignores_placeholder_blockers_on_receipt_backed_reaped_lane(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-reaped-placeholder",
                            "task_id": "wo-reaped-placeholder",
                            "run_id": "run-1",
                            "status": "failed",
                            "title": "Receipt-backed reaped lane with placeholder blocker",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-reaped-placeholder",
                            "branch": "codex/reaped-placeholder",
                            "worktree_path": "/tmp/repo/.worktrees/reaped-placeholder",
                            "lease_id": "lease-reaped-placeholder",
                            "blockers": ["None", "stale_lease_reaped"],
                            "updated_at": now.isoformat(),
                        }
                    ],
                    "leases": [
                        {
                            "lease_id": "lease-reaped-placeholder",
                            "task_id": "wo-reaped-placeholder",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-reaped-placeholder",
                            "branch": "codex/reaped-placeholder",
                            "worktree_path": "/tmp/repo/.worktrees/reaped-placeholder",
                            "status": "released",
                            "updated_at": (now - timedelta(hours=2)).isoformat(),
                            "expires_at": (now - timedelta(hours=1)).isoformat(),
                        }
                    ],
                    "completion_receipts": [
                        {
                            "receipt_id": "receipt-reaped-placeholder",
                            "lease_id": "lease-reaped-placeholder",
                            "task_id": "wo-reaped-placeholder",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-reaped-placeholder",
                            "branch": "codex/reaped-placeholder",
                            "worktree_path": "/tmp/repo/.worktrees/reaped-placeholder",
                            "commit_shas": ["abc123"],
                            "changed_paths": ["aragora/swarm/reporter.py"],
                            "created_at": now.isoformat(),
                        }
                    ],
                    "integration_decisions": [],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["receipt_id"] == "receipt-reaped-placeholder"
        assert lane["terminal_outcome"] == "deliverable_created"
        assert lane["merge_readiness"] == "review"
        assert lane["lane_health"] == "healthy"
        assert "None" not in lane["blockers"]
        assert "stale_lease_reaped" not in lane["blockers"]
        assert (
            lane["next_action"] == "Review the validated lane and decide whether it should merge."
        )

    def test_build_integrator_view_keeps_receipt_backed_released_dispatched_lane_reviewable(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-released-review",
                            "task_id": "wo-released-review",
                            "run_id": "run-1",
                            "status": "dispatched",
                            "title": "Receipt-backed released dispatched lane",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-released-review",
                            "branch": "codex/released-review",
                            "worktree_path": "/tmp/repo/.worktrees/released-review",
                            "lease_id": "lease-released-review",
                            "blockers": ["stale_lease_reaped"],
                            "updated_at": now.isoformat(),
                        }
                    ],
                    "leases": [
                        {
                            "lease_id": "lease-released-review",
                            "task_id": "wo-released-review",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-released-review",
                            "branch": "codex/released-review",
                            "worktree_path": "/tmp/repo/.worktrees/released-review",
                            "status": "released",
                            "updated_at": (now - timedelta(hours=2)).isoformat(),
                            "expires_at": (now - timedelta(hours=1)).isoformat(),
                        }
                    ],
                    "completion_receipts": [
                        {
                            "receipt_id": "receipt-released-review",
                            "lease_id": "lease-released-review",
                            "task_id": "wo-released-review",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-released-review",
                            "branch": "codex/released-review",
                            "worktree_path": "/tmp/repo/.worktrees/released-review",
                            "commit_shas": ["abc123"],
                            "changed_paths": ["aragora/swarm/reporter.py"],
                            "created_at": now.isoformat(),
                        }
                    ],
                    "integration_decisions": [],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["terminal_outcome"] == "deliverable_created"
        assert lane["receipt_id"] == "receipt-released-review"
        assert lane["merge_readiness"] == "review"
        assert lane["lane_health"] == "healthy"
        assert lane["blockers"] == []
        assert (
            lane["next_action"] == "Review the validated lane and decide whether it should merge."
        )

    def test_build_integrator_view_keeps_merge_gate_failed_receipt_lane_blocked(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-merge-gate",
                            "task_id": "wo-merge-gate",
                            "run_id": "run-1",
                            "status": "needs_human",
                            "title": "Merge-gate failed receipt lane",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-merge-gate",
                            "branch": "codex/merge-gate",
                            "worktree_path": "/tmp/repo/.worktrees/merge-gate",
                            "lease_id": "lease-merge-gate",
                            "blockers": [
                                "merge gate blocked: verification failed: pytest tests/swarm -q",
                                "merge_gate_failed",
                            ],
                            "updated_at": now.isoformat(),
                        }
                    ],
                    "leases": [
                        {
                            "lease_id": "lease-merge-gate",
                            "task_id": "wo-merge-gate",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-merge-gate",
                            "branch": "codex/merge-gate",
                            "worktree_path": "/tmp/repo/.worktrees/merge-gate",
                            "status": "released",
                            "updated_at": (now - timedelta(hours=2)).isoformat(),
                            "expires_at": (now - timedelta(hours=1)).isoformat(),
                        }
                    ],
                    "completion_receipts": [
                        {
                            "receipt_id": "receipt-merge-gate",
                            "lease_id": "lease-merge-gate",
                            "task_id": "wo-merge-gate",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-merge-gate",
                            "branch": "codex/merge-gate",
                            "worktree_path": "/tmp/repo/.worktrees/merge-gate",
                            "commit_shas": ["abc123"],
                            "changed_paths": ["aragora/swarm/reporter.py"],
                            "created_at": now.isoformat(),
                        }
                    ],
                    "integration_decisions": [
                        {
                            "decision_id": "decision-merge-gate",
                            "lease_id": "lease-merge-gate",
                            "receipt_id": "receipt-merge-gate",
                            "decision": "pending_review",
                            "created_at": now.isoformat(),
                        }
                    ],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["merge_readiness"] == "blocked"
        assert lane["lane_health"] == "blocked"
        assert (
            lane["next_action"]
            == "Fix the merge gate or verification failure before rerunning the lane."
        )

    def test_build_integrator_view_ignores_stale_reap_for_merge_gate_deliverable_lane(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-merge-gate-stale",
                            "task_id": "wo-merge-gate-stale",
                            "run_id": "run-1",
                            "status": "needs_human",
                            "title": "Stale merge-gate deliverable lane",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-merge-gate-stale",
                            "branch": "codex/merge-gate-stale",
                            "worktree_path": "/tmp/repo/.worktrees/merge-gate-stale",
                            "lease_id": "lease-merge-gate-stale",
                            "blockers": [
                                "merge gate blocked: verification failed: pytest tests/swarm -q",
                                "merge_gate_failed",
                                "stale_lease_reaped",
                            ],
                            "updated_at": now.isoformat(),
                        }
                    ],
                    "leases": [
                        {
                            "lease_id": "lease-merge-gate-stale",
                            "task_id": "wo-merge-gate-stale",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-merge-gate-stale",
                            "branch": "codex/merge-gate-stale",
                            "worktree_path": "/tmp/repo/.worktrees/merge-gate-stale",
                            "status": "expired",
                            "updated_at": (now - timedelta(hours=2)).isoformat(),
                            "expires_at": (now - timedelta(hours=1)).isoformat(),
                        }
                    ],
                    "completion_receipts": [
                        {
                            "receipt_id": "receipt-merge-gate-stale",
                            "lease_id": "lease-merge-gate-stale",
                            "task_id": "wo-merge-gate-stale",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-merge-gate-stale",
                            "branch": "codex/merge-gate-stale",
                            "worktree_path": "/tmp/repo/.worktrees/merge-gate-stale",
                            "commit_shas": ["abc123"],
                            "changed_paths": ["aragora/swarm/reporter.py"],
                            "created_at": now.isoformat(),
                        }
                    ],
                    "integration_decisions": [
                        {
                            "decision_id": "decision-merge-gate-stale",
                            "lease_id": "lease-merge-gate-stale",
                            "receipt_id": "receipt-merge-gate-stale",
                            "decision": "pending_review",
                            "created_at": now.isoformat(),
                        }
                    ],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["merge_readiness"] == "blocked"
        assert lane["lane_health"] == "blocked"
        assert "stale_lease_reaped" not in lane["blockers"]
        assert (
            lane["next_action"]
            == "Fix the merge gate or verification failure before rerunning the lane."
        )

    def test_build_integrator_view_keeps_scope_violation_receipt_lane_blocked(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-scope-receipt",
                            "task_id": "wo-scope-receipt",
                            "run_id": "run-1",
                            "status": "scope_violation",
                            "title": "Scope violation receipt lane",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-scope-receipt",
                            "branch": "codex/scope-receipt",
                            "worktree_path": "/tmp/repo/.worktrees/scope-receipt",
                            "lease_id": "lease-scope-receipt",
                            "blockers": ["scope_violation"],
                            "updated_at": now.isoformat(),
                        }
                    ],
                    "leases": [
                        {
                            "lease_id": "lease-scope-receipt",
                            "task_id": "wo-scope-receipt",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-scope-receipt",
                            "branch": "codex/scope-receipt",
                            "worktree_path": "/tmp/repo/.worktrees/scope-receipt",
                            "status": "released",
                            "updated_at": (now - timedelta(hours=2)).isoformat(),
                            "expires_at": (now - timedelta(hours=1)).isoformat(),
                        }
                    ],
                    "completion_receipts": [
                        {
                            "receipt_id": "receipt-scope-receipt",
                            "lease_id": "lease-scope-receipt",
                            "task_id": "wo-scope-receipt",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-scope-receipt",
                            "branch": "codex/scope-receipt",
                            "worktree_path": "/tmp/repo/.worktrees/scope-receipt",
                            "commit_shas": ["abc123"],
                            "changed_paths": ["aragora/swarm/reporter.py"],
                            "created_at": now.isoformat(),
                        }
                    ],
                    "integration_decisions": [
                        {
                            "decision_id": "decision-scope-receipt",
                            "lease_id": "lease-scope-receipt",
                            "receipt_id": "receipt-scope-receipt",
                            "decision": "pending_review",
                            "created_at": now.isoformat(),
                        }
                    ],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["scope_violation_detected"] is True
        assert lane["merge_readiness"] == "blocked"
        assert lane["lane_health"] == "blocked"
        assert (
            lane["next_action"]
            == "Narrow the lane scope or split ownership before it can re-enter merge review."
        )

    def test_build_integrator_view_ignores_stale_reap_for_scope_deliverable_lane(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-scope-stale",
                            "task_id": "wo-scope-stale",
                            "run_id": "run-1",
                            "status": "completed",
                            "title": "Stale scope deliverable lane",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-scope-stale",
                            "branch": "codex/scope-stale",
                            "worktree_path": "/tmp/repo/.worktrees/scope-stale",
                            "lease_id": "lease-scope-stale",
                            "commit_shas": ["abc123"],
                            "changed_paths": ["tests/swarm/test_reporter.py"],
                            "blockers": ["stale_lease_reaped"],
                            "updated_at": now.isoformat(),
                            "metadata": {
                                "last_scope_violation": {
                                    "detected_at": now.isoformat(),
                                    "changed_paths": ["tests/swarm/test_reporter.py"],
                                    "violations": [
                                        {
                                            "type": "undeclared_scope",
                                            "path": "tests/swarm/test_reporter.py",
                                        }
                                    ],
                                }
                            },
                        }
                    ],
                    "leases": [
                        {
                            "lease_id": "lease-scope-stale",
                            "task_id": "wo-scope-stale",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-scope-stale",
                            "branch": "codex/scope-stale",
                            "worktree_path": "/tmp/repo/.worktrees/scope-stale",
                            "status": "expired",
                            "updated_at": (now - timedelta(hours=2)).isoformat(),
                            "expires_at": (now - timedelta(hours=1)).isoformat(),
                        }
                    ],
                    "completion_receipts": [],
                    "integration_decisions": [
                        {
                            "decision_id": "decision-scope-stale",
                            "decision": "pending_review",
                            "created_at": now.isoformat(),
                        }
                    ],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["scope_violation_detected"] is True
        assert lane["merge_readiness"] == "blocked"
        assert lane["lane_health"] == "blocked"
        assert "stale_lease_reaped" not in lane["blockers"]
        assert (
            lane["next_action"]
            == "Declare the intended lane scope or discard the lane before receipt backfill and merge review."
        )

    def test_build_integrator_view_does_not_expect_receipt_when_stale_lease_record_is_gone(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-stale-missing-lease",
                            "task_id": "wo-stale-missing-lease",
                            "run_id": "run-1",
                            "status": "needs_human",
                            "title": "Stale lease lane without surviving lease row",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-stale",
                            "branch": "codex/stale-gone",
                            "worktree_path": "/tmp/repo/.worktrees/stale-gone",
                            "lease_id": "lease-gone",
                            "blockers": ["stale_lease_reaped"],
                            "updated_at": now.isoformat(),
                        }
                    ],
                    "leases": [],
                    "completion_receipts": [],
                    "integration_decisions": [],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["receipt_expected"] is False
        assert lane["missing_receipt"] is False
        assert "missing_receipt" not in lane["blockers"]
        assert lane["lease_health"] == "expired"
        assert lane["lane_health"] == "expired"
        assert lane["next_action"] == "Salvage or reassign the expired lane before resuming work."

    def test_build_integrator_view_does_not_mark_unleased_queued_lane_stale(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-queued",
                            "task_id": "wo-queued",
                            "run_id": "run-1",
                            "status": "queued",
                            "title": "Queued backlog lane",
                            "owner_agent": "codex",
                            "branch": "codex/queued-lane",
                            "worktree_path": "/tmp/repo/.worktrees/queued",
                            "updated_at": (now - timedelta(hours=2)).isoformat(),
                        }
                    ],
                    "leases": [],
                    "completion_receipts": [],
                    "integration_decisions": [],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["status"] == "queued"
        assert lane["stale_heartbeat"] is False
        assert "stale_heartbeat" not in lane["blockers"]
        assert lane["lane_health"] == "healthy"

    def test_build_integrator_view_does_not_cross_match_task_id_across_runs(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-a:subtask_1",
                            "task_id": "subtask_1",
                            "run_id": "run-a",
                            "status": "queued",
                            "title": "Queued lane should stay isolated",
                            "owner_agent": "codex",
                            "updated_at": now.isoformat(),
                        }
                    ],
                    "leases": [
                        {
                            "lease_id": "lease-run-b",
                            "task_id": "subtask_1",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-run-b",
                            "branch": "codex/run-b",
                            "worktree_path": "/tmp/repo/.worktrees/run-b",
                            "status": "active",
                            "updated_at": (now - timedelta(hours=2)).isoformat(),
                            "expires_at": (now + timedelta(hours=1)).isoformat(),
                            "metadata": {
                                "supervisor_run_id": "run-b",
                                "task_key": "run-b:subtask_1",
                                "work_order_id": "subtask_1",
                            },
                        }
                    ],
                    "completion_receipts": [
                        {
                            "receipt_id": "receipt-run-b",
                            "lease_id": "lease-run-b",
                            "task_id": "subtask_1",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-run-b",
                            "branch": "codex/run-b",
                            "worktree_path": "/tmp/repo/.worktrees/run-b",
                            "commit_shas": ["abc123"],
                            "changed_paths": ["aragora/swarm/reporter.py"],
                            "created_at": now.isoformat(),
                            "metadata": {
                                "supervisor_run_id": "run-b",
                                "task_key": "run-b:subtask_1",
                                "work_order_id": "subtask_1",
                            },
                        }
                    ],
                    "integration_decisions": [
                        {
                            "decision_id": "decision-run-b",
                            "receipt_id": "receipt-run-b",
                            "decision": "pending_review",
                            "created_at": now.isoformat(),
                        }
                    ],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["lane_id"] == "run-a:subtask_1"
        assert lane["receipt_id"] is None
        assert lane["lease_id"] is None
        assert lane["lease_status"] is None
        assert lane["stale_heartbeat"] is False
        assert lane["merge_readiness"] == "in_progress"
        assert lane["lane_health"] == "healthy"
        assert lane["blockers"] == []

    def test_build_integrator_view_collapses_cli_transcript_blocker(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)
        transcript = (
            "OpenAI Codex v0.107.0 (research preview)\n"
            "--------\n"
            "workdir: /tmp/repo/.worktrees/swarm-timeout\n"
            "approval: never\n"
            "sandbox: workspace-write\n"
            "session id: deadbeef\n"
            "user\n# Task: Example\n"
            "exec\n/bin/zsh -lc 'rg --files'\n"
            "worker timed out while waiting for completion\n"
        ) + ("x" * 700)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-timeout:subtask_1",
                            "task_id": "subtask_1",
                            "run_id": "run-timeout",
                            "status": "failed",
                            "title": "Timeout lane",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-timeout",
                            "branch": "codex/timeout-lane",
                            "worktree_path": "/tmp/repo/.worktrees/timeout-lane",
                            "lease_id": "lease-timeout",
                            "blockers": [transcript],
                            "updated_at": now.isoformat(),
                        }
                    ],
                    "leases": [
                        {
                            "lease_id": "lease-timeout",
                            "task_id": "subtask_1",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-timeout",
                            "branch": "codex/timeout-lane",
                            "worktree_path": "/tmp/repo/.worktrees/timeout-lane",
                            "status": "released",
                            "updated_at": (now - timedelta(hours=2)).isoformat(),
                            "expires_at": (now - timedelta(hours=1)).isoformat(),
                            "metadata": {
                                "supervisor_run_id": "run-timeout",
                                "task_key": "run-timeout:subtask_1",
                                "work_order_id": "subtask_1",
                            },
                        }
                    ],
                    "completion_receipts": [],
                    "integration_decisions": [],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["lane_health"] == "stalled"
        assert lane["terminal_outcome"] == "timeout"
        assert lane["blockers"] == ["worker_timeout_transcript_captured"]

    def test_build_integrator_view_marks_released_dispatched_lane_expired(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-released-dispatched",
                            "task_id": "wo-released-dispatched",
                            "run_id": "run-1",
                            "status": "dispatched",
                            "title": "Released dispatched lane",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-released",
                            "branch": "codex/released-lane",
                            "worktree_path": "/tmp/repo/.worktrees/released",
                            "lease_id": "lease-released",
                            "updated_at": (now - timedelta(hours=2)).isoformat(),
                        }
                    ],
                    "leases": [
                        {
                            "lease_id": "lease-released",
                            "task_id": "wo-released-dispatched",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-released",
                            "branch": "codex/released-lane",
                            "worktree_path": "/tmp/repo/.worktrees/released",
                            "status": "released",
                            "updated_at": (now - timedelta(hours=2)).isoformat(),
                            "expires_at": (now - timedelta(hours=1)).isoformat(),
                        }
                    ],
                    "completion_receipts": [],
                    "integration_decisions": [],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["status"] == "dispatched"
        assert lane["merge_readiness"] == "blocked"
        assert lane["stale_heartbeat"] is False
        assert "stale_lease_reaped" in lane["blockers"]
        assert lane["lane_health"] == "expired"
        assert lane["next_action"] == "Salvage or reassign the expired lane before resuming work."

    def test_build_integrator_view_marks_orphaned_dispatched_lane_expired(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-orphaned-dispatched",
                            "task_id": "wo-orphaned-dispatched",
                            "run_id": "run-1",
                            "status": "dispatched",
                            "title": "Orphaned dispatched lane",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-orphaned",
                            "branch": "codex/orphaned-lane",
                            "worktree_path": "/tmp/repo/.worktrees/orphaned",
                            "lease_id": "lease-orphaned",
                            "updated_at": (now - timedelta(hours=2)).isoformat(),
                        }
                    ],
                    "leases": [],
                    "completion_receipts": [],
                    "integration_decisions": [],
                    "salvage_candidates": [],
                }
            },
            worktrees=[
                {
                    "session_id": "sess-orphaned",
                    "path": "/tmp/repo/.worktrees/orphaned",
                    "branch": "codex/orphaned-lane",
                    "has_lock": False,
                    "pid_alive": False,
                    "agent": "codex",
                    "last_activity": (now - timedelta(hours=2)).isoformat(),
                }
            ],
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["status"] == "dispatched"
        assert lane["merge_readiness"] == "blocked"
        assert lane["stale_heartbeat"] is False
        assert "stale_lease_reaped" in lane["blockers"]
        assert lane["lane_health"] == "expired"
        assert lane["next_action"] == "Salvage or reassign the expired lane before resuming work."

    def test_build_integrator_view_marks_scope_violation_lane_blocked(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-scope",
                            "task_id": "wo-scope",
                            "run_id": "run-1",
                            "status": "scope_violation",
                            "title": "Scope violation lane",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-scope",
                            "branch": "codex/scope-lane",
                            "worktree_path": "/tmp/repo/.worktrees/scope",
                            "blockers": ["scope_violation"],
                            "updated_at": now.isoformat(),
                        }
                    ],
                    "leases": [],
                    "completion_receipts": [],
                    "integration_decisions": [],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["status"] == "scope_violation"
        assert lane["merge_readiness"] == "blocked"
        assert lane["lane_health"] == "blocked"
        assert lane["scope_violation_detected"] is True
        assert (
            lane["next_action"]
            == "Narrow the lane scope or split ownership before it can re-enter merge review."
        )
        assert payload["summary"]["scope_violation_lanes"] == 1

    def test_build_integrator_view_exposes_mission_and_gate_metadata(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-gated",
                            "task_id": "wo-gated",
                            "run_id": "run-1",
                            "status": "blocked",
                            "title": "Credential envelope gate",
                            "owner_agent": "codex",
                            "updated_at": now.isoformat(),
                            "metadata": {
                                "mission_id": "mission-rs-credential-envelope",
                                "stage_id": "stage-contract-aware-preflight",
                                "assertion_ids": ["ASSERT-001"],
                                "evidence_expectations": [
                                    "worker_contract",
                                    "preflight_result",
                                ],
                                "dispatch_gate": {
                                    "gate_type": "dispatch_ready",
                                    "verdict": "blocked",
                                    "mission_id": "mission-rs-credential-envelope",
                                    "stage_id": "stage-contract-aware-preflight",
                                    "assertion_ids": ["ASSERT-001"],
                                    "failure_classes": ["contract_missing"],
                                    "repair_eligible": False,
                                    "required_evidence": [
                                        "worker_contract",
                                        "preflight_result",
                                    ],
                                    "notes": "Credential envelope checksum absent",
                                },
                            },
                        }
                    ],
                    "leases": [],
                    "completion_receipts": [],
                    "integration_decisions": [],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["mission_id"] == "mission-rs-credential-envelope"
        assert lane["stage_id"] == "stage-contract-aware-preflight"
        assert lane["assertion_ids"] == ["ASSERT-001"]
        assert lane["evidence_expectations"] == ["worker_contract", "preflight_result"]
        assert lane["dispatch_gate"] == {
            "gate_type": "dispatch_ready",
            "verdict": "blocked",
            "mission_id": "mission-rs-credential-envelope",
            "stage_id": "stage-contract-aware-preflight",
            "assertion_ids": ["ASSERT-001"],
            "failure_classes": ["contract_missing"],
            "repair_eligible": False,
            "required_evidence": ["worker_contract", "preflight_result"],
            "notes": "Credential envelope checksum absent",
        }
        assert lane["gate_evaluations"] == [lane["dispatch_gate"]]
        assert lane["last_gate_type"] == "dispatch_ready"
        assert lane["last_gate_verdict"] == "blocked"
        assert lane["failure_classes"] == ["contract_missing"]

    def test_build_integrator_view_excludes_superseded_scope_violation_from_summary(self):
        now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)

        payload = build_integrator_view(
            coordination={
                "integrator": {
                    "developer_tasks": [
                        {
                            "task_key": "run-1:wo-scope",
                            "task_id": "wo-scope",
                            "run_id": "run-1",
                            "status": "discarded",
                            "title": "Archived scope violation lane",
                            "owner_agent": "codex",
                            "owner_session_id": "sess-scope",
                            "branch": "codex/scope-lane",
                            "worktree_path": "/tmp/repo/.worktrees/scope",
                            "blockers": ["scope_violation"],
                            "updated_at": now.isoformat(),
                        }
                    ],
                    "leases": [],
                    "completion_receipts": [],
                    "integration_decisions": [],
                    "salvage_candidates": [],
                }
            },
            now=now,
        )

        lane = payload["lanes"][0]
        assert lane["scope_violation_detected"] is True
        assert lane["merge_readiness"] == "superseded"
        assert lane["lane_health"] == "superseded"
        assert payload["summary"]["scope_violation_lanes"] == 0


class TestBossPayload:
    def test_build_boss_payload_redacts_transcript_shaped_blocker_evidence(self) -> None:
        transcript = """OpenAI Codex v0.1
workdir: /tmp/secret
approval: never
sandbox: workspace-write
API_KEY=shh-secret
Command: pytest tests/swarm/test_reporter.py -q
Result: timed out after 30s
"""

        payload = build_boss_payload(
            run={
                "run_id": "run-redacted",
                "status": "needs_human",
                "goal": "Repair failing lane",
                "target_branch": "main",
                "work_orders": [
                    {
                        "work_order_id": "wo-redacted",
                        "title": "Repair failing lane",
                        "status": "needs_human",
                        "failure_reason": "merge_gate_failed",
                        "metadata": {
                            "blocker_evidence": transcript,
                            "repair_journal": [
                                {
                                    "failure_reason": "merge_gate_failed",
                                    "exit_code": 1,
                                    "failing_verification": {
                                        "command": "pytest tests/swarm/test_reporter.py -q",
                                        "exit_code": 1,
                                        "stderr_tail": transcript,
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
            integrator_view={
                "lanes": [
                    {
                        "work_order_id": "wo-redacted",
                        "title": "Repair failing lane",
                        "status": "needs_human",
                        "blockers": ["merge_gate_failed"],
                        "failure_classes": ["merge_gate_failed"],
                    }
                ]
            },
        )

        lane = payload["lanes"][0]
        assert lane["blocker_evidence"] == "worker_timeout_transcript_captured"
        assert lane["repair_summary"]["evidence"] == "worker_timeout_transcript_captured"

        needs_human = payload["needs_human"][0]
        assert needs_human["blocker_evidence"] == "worker_timeout_transcript_captured"
        assert needs_human["repair_summary"]["evidence"] == "worker_timeout_transcript_captured"

    def test_build_boss_payload_surfaces_blocker_evidence_for_needs_human_lanes(self) -> None:
        payload = build_boss_payload(
            run={
                "run_id": "run-1",
                "status": "needs_human",
                "goal": "Repair failing lane",
                "target_branch": "main",
                "work_orders": [
                    {
                        "work_order_id": "wo-1",
                        "title": "Repair failing lane",
                        "status": "needs_human",
                        "target_agent": "codex",
                        "branch": "codex/test-lane",
                        "worktree_path": "/tmp/test-lane",
                        "review_status": "changes_requested",
                        "failure_reason": "merge_gate_failed",
                        "needs_human_reasons": ["verification plan missing"],
                        "metadata": {
                            "blocker_evidence": "pytest timed out in tests/swarm/test_reporter.py",
                            "repair_journal": [
                                {
                                    "failure_reason": "merge_gate_failed",
                                    "exit_code": 1,
                                    "failing_verification": {
                                        "command": "pytest tests/swarm/test_reporter.py -q",
                                        "exit_code": 1,
                                        "stderr_tail": "AssertionError: missing blocker evidence",
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
            integrator_view={
                "lanes": [
                    {
                        "work_order_id": "wo-1",
                        "title": "Repair failing lane",
                        "status": "needs_human",
                        "owner_agent": "codex",
                        "branch": "codex/test-lane",
                        "worktree_path": "/tmp/test-lane",
                        "blockers": ["merge_gate_failed"],
                        "failure_classes": ["merge_gate_failed"],
                        "next_action": "Fix verification failure before rerunning the lane.",
                    }
                ],
                "next_actions": ["Fix verification failure before rerunning the lane."],
            },
        )

        lane = payload["lanes"][0]
        assert lane["failure_classes"] == ["merge_gate_failed"]
        assert lane["blocker_evidence"] == "pytest timed out in tests/swarm/test_reporter.py"
        assert lane["repair_summary"] == {
            "failure_reason": "merge_gate_failed",
            "exit_code": 1,
            "verification_command": "pytest tests/swarm/test_reporter.py -q",
            "verification_exit_code": 1,
            "evidence": "AssertionError: missing blocker evidence",
        }

        needs_human = payload["needs_human"][0]
        assert needs_human["reasons"] == ["merge_gate_failed", "verification plan missing"]
        assert needs_human["failure_classes"] == ["merge_gate_failed"]
        assert needs_human["blocker_evidence"] == "pytest timed out in tests/swarm/test_reporter.py"
        assert needs_human["next_action"] == "Fix verification failure before rerunning the lane."

    def test_build_boss_payload_prefers_persisted_top_level_blocker_evidence(self) -> None:
        payload = build_boss_payload(
            run={
                "run_id": "run-1b",
                "status": "needs_human",
                "goal": "Repair failing lane",
                "target_branch": "main",
                "work_orders": [
                    {
                        "work_order_id": "wo-1b",
                        "title": "Repair failing lane",
                        "status": "needs_human",
                        "failure_reason": "worker_no_progress_timeout",
                        "blocker_evidence": "stalled warning",
                        "metadata": {
                            "repair_journal": [
                                {
                                    "failure_reason": "worker_no_progress_timeout",
                                    "exit_code": -1,
                                    "stderr_tail": "older stderr tail",
                                }
                            ]
                        },
                    }
                ],
            },
            integrator_view={
                "lanes": [
                    {
                        "work_order_id": "wo-1b",
                        "title": "Repair failing lane",
                        "status": "needs_human",
                        "blockers": ["worker_no_progress_timeout"],
                        "failure_classes": ["worker_no_progress_timeout"],
                    }
                ]
            },
        )

        lane = payload["lanes"][0]
        assert lane["blocker_evidence"] == "stalled warning"
        assert payload["needs_human"][0]["blocker_evidence"] == "stalled warning"

    def test_render_boss_text_includes_needs_human_blocker_evidence(self) -> None:
        payload = build_boss_payload(
            run={
                "run_id": "run-2",
                "status": "needs_human",
                "goal": "Repair failing lane",
                "target_branch": "main",
                "work_orders": [
                    {
                        "work_order_id": "wo-2",
                        "title": "Repair failing lane",
                        "status": "needs_human",
                        "failure_reason": "merge_gate_failed",
                        "metadata": {
                            "blocker_evidence": "pytest timed out in tests/swarm/test_reporter.py",
                            "repair_journal": [
                                {
                                    "failure_reason": "merge_gate_failed",
                                    "exit_code": 1,
                                    "failing_verification": {
                                        "command": "pytest tests/swarm/test_reporter.py -q",
                                        "exit_code": 1,
                                        "stderr_tail": "AssertionError: missing blocker evidence",
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
            integrator_view={
                "lanes": [
                    {
                        "work_order_id": "wo-2",
                        "title": "Repair failing lane",
                        "status": "needs_human",
                        "blockers": ["merge_gate_failed"],
                        "failure_classes": ["merge_gate_failed"],
                        "next_action": "Fix verification failure before rerunning the lane.",
                    }
                ]
            },
        )

        text = render_boss_text(payload)

        assert "needs_human: Repair failing lane -> merge_gate_failed" in text
        assert "needs_human_classes: Repair failing lane -> merge_gate_failed" in text
        assert (
            "needs_human_evidence: Repair failing lane -> "
            "pytest timed out in tests/swarm/test_reporter.py"
        ) in text
        assert "needs_human_repair: Repair failing lane -> reason=merge_gate_failed" in text
        assert (
            "needs_human_next: Repair failing lane -> "
            "Fix verification failure before rerunning the lane."
        ) in text

    def test_render_boss_text_redacts_transcript_shaped_evidence(self) -> None:
        transcript = """OpenAI Codex v0.1
workdir: /tmp/secret
approval: never
sandbox: workspace-write
API_KEY=shh-secret
Command: pytest tests/swarm/test_reporter.py -q
Result: timed out after 30s
"""

        text = render_boss_text(
            build_boss_payload(
                run={
                    "run_id": "run-3",
                    "status": "needs_human",
                    "goal": "Repair failing lane",
                    "target_branch": "main",
                    "work_orders": [
                        {
                            "work_order_id": "wo-3",
                            "title": "Repair failing lane",
                            "status": "needs_human",
                            "failure_reason": "merge_gate_failed",
                            "metadata": {
                                "blocker_evidence": transcript,
                                "repair_journal": [
                                    {
                                        "failure_reason": "merge_gate_failed",
                                        "exit_code": 1,
                                        "failing_verification": {
                                            "command": "pytest tests/swarm/test_reporter.py -q",
                                            "exit_code": 1,
                                            "stderr_tail": transcript,
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
                integrator_view={
                    "lanes": [
                        {
                            "work_order_id": "wo-3",
                            "title": "Repair failing lane",
                            "status": "needs_human",
                            "blockers": ["merge_gate_failed"],
                            "failure_classes": ["merge_gate_failed"],
                        }
                    ]
                },
            )
        )

        assert "worker_timeout_transcript_captured" in text
        assert "API_KEY=shh-secret" not in text
        assert "workdir: /tmp/secret" not in text
        assert "OpenAI Codex v0.1" not in text
