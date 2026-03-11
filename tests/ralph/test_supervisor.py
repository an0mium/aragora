"""Tests for ralph campaign supervisor state machine."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from aragora.ralph.classifier import BlockerKind
from aragora.ralph.supervisor import (
    RalphSupervisor,
    StepResult,
    SupervisorAction,
    SupervisorState,
    SupervisorStatus,
    load_supervisor_state,
    save_supervisor_state,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_manifest(path: Path, campaign_id: str = "test-campaign") -> Path:
    manifest = {
        "campaign_id": campaign_id,
        "created_at": "2026-03-10T00:00:00Z",
        "source_kind": "manual",
        "source_ref": "test",
        "worker_model": "codex",
        "review_model": "claude",
        "max_parallel_ready_projects": 1,
        "max_retries_per_project": 2,
        "budget_limit_usd": 20.0,
        "time_limit_hours": 4.0,
        "projects": [],
        "execution_state": {
            "ready_queue": [],
            "active_projects": [],
            "completed_projects": [],
            "failed_projects": [],
            "skipped_projects": [],
            "total_cost_usd": 0.0,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(manifest), encoding="utf-8")
    return path


def _write_state(path: Path, **overrides) -> SupervisorState:
    state = SupervisorState(
        supervisor_id="ralph-test123",
        campaign_manifest_path=str(overrides.pop("campaign_manifest_path", "/tmp/manifest.yaml")),
        campaign_id="test-campaign",
        **overrides,
    )
    save_supervisor_state(path, state)
    return state


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


class TestSupervisorStatePersistence:
    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        state_path = tmp_path / ".aragora" / "supervisor_state.yaml"
        state = SupervisorState(
            supervisor_id="ralph-abc",
            campaign_manifest_path="/foo/manifest.yaml",
            campaign_id="camp-1",
            status=SupervisorStatus.RUNNING.value,
            current_step=3,
            repair_attempts=1,
            budget_spent_usd=5.5,
        )
        save_supervisor_state(state_path, state)
        loaded = load_supervisor_state(state_path)

        assert loaded.supervisor_id == "ralph-abc"
        assert loaded.campaign_id == "camp-1"
        assert loaded.current_step == 3
        assert loaded.repair_attempts == 1
        assert loaded.budget_spent_usd == 5.5
        assert loaded.updated_at  # should be set by save

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_supervisor_state(tmp_path / "nonexistent.yaml")

    def test_to_dict_from_dict_round_trip(self) -> None:
        state = SupervisorState(
            supervisor_id="ralph-xyz",
            campaign_manifest_path="/p",
            campaign_id="c1",
            active_blocker="unknown",
            blocker_history=[{"step": 1, "kind": "unknown"}],
        )
        d = state.to_dict()
        restored = SupervisorState.from_dict(d)
        assert restored.supervisor_id == state.supervisor_id
        assert restored.active_blocker == "unknown"
        assert len(restored.blocker_history) == 1


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------


class TestSupervisorStart:
    def test_start_creates_state_file(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path / "manifest.yaml")
        state_path = tmp_path / ".aragora" / "supervisor_state.yaml"

        supervisor = RalphSupervisor.start(
            manifest_path=manifest_path,
            state_path=state_path,
            repo_root=tmp_path,
        )
        assert state_path.exists()
        state = load_supervisor_state(state_path)
        assert state.campaign_id == "test-campaign"
        assert state.status == SupervisorStatus.RUNNING.value
        assert state.supervisor_id.startswith("ralph-")

    def test_start_missing_manifest_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            RalphSupervisor.start(
                manifest_path=tmp_path / "nonexistent.yaml",
                state_path=tmp_path / "state.yaml",
            )


# ---------------------------------------------------------------------------
# Step: terminal states
# ---------------------------------------------------------------------------


class TestStepTerminalStates:
    def test_completed_state_returns_noop(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.yaml"
        _write_state(state_path, status=SupervisorStatus.COMPLETED.value)

        supervisor = RalphSupervisor(state_path=state_path, repo_root=tmp_path)
        result = supervisor.step()

        assert result.action == SupervisorAction.NOOP.value
        assert result.status == SupervisorStatus.COMPLETED.value

    def test_escalated_state_returns_noop(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.yaml"
        _write_state(state_path, status=SupervisorStatus.ESCALATED.value)

        supervisor = RalphSupervisor(state_path=state_path, repo_root=tmp_path)
        result = supervisor.step()

        assert result.action == SupervisorAction.NOOP.value

    def test_stopped_state_returns_noop(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.yaml"
        _write_state(state_path, status=SupervisorStatus.STOPPED.value)

        supervisor = RalphSupervisor(state_path=state_path, repo_root=tmp_path)
        result = supervisor.step()

        assert result.action == SupervisorAction.NOOP.value


# ---------------------------------------------------------------------------
# Step: campaign iteration
# ---------------------------------------------------------------------------


class TestStepCampaignIteration:
    def test_still_running_continues(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path / "manifest.yaml")
        state_path = tmp_path / "state.yaml"
        _write_state(
            state_path,
            campaign_manifest_path=str(manifest_path),
            status=SupervisorStatus.RUNNING.value,
        )

        with patch(
            "aragora.ralph.supervisor.asyncio.run",
            return_value={"stop_reason": "still_running", "dispatched_projects": ["p1"]},
        ):
            supervisor = RalphSupervisor(state_path=state_path, repo_root=tmp_path)
            result = supervisor.step()

        assert result.action == SupervisorAction.CAMPAIGN_ITERATION.value
        assert result.status == SupervisorStatus.RUNNING.value
        assert "p1" in result.detail

    def test_campaign_complete_transitions(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path / "manifest.yaml")
        state_path = tmp_path / "state.yaml"
        _write_state(
            state_path,
            campaign_manifest_path=str(manifest_path),
            status=SupervisorStatus.RUNNING.value,
        )

        with patch(
            "aragora.ralph.supervisor.asyncio.run",
            return_value={"stop_reason": "campaign_complete", "dispatched_projects": []},
        ):
            supervisor = RalphSupervisor(state_path=state_path, repo_root=tmp_path)
            result = supervisor.step()

        assert result.action == SupervisorAction.CAMPAIGN_COMPLETED.value
        assert result.status == SupervisorStatus.COMPLETED.value
        state = load_supervisor_state(state_path)
        assert state.status == SupervisorStatus.COMPLETED.value

    def test_blocked_classifies_blocker(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.yaml"
        manifest_data = {
            "campaign_id": "test",
            "created_at": "2026-01-01",
            "source_kind": "manual",
            "source_ref": "test",
            "worker_model": "codex",
            "review_model": "claude",
            "max_parallel_ready_projects": 1,
            "max_retries_per_project": 2,
            "budget_limit_usd": 20.0,
            "time_limit_hours": 4.0,
            "projects": [
                {
                    "project_id": "p1",
                    "status": "blocked",
                    "last_run_outcome": "deliverable_created",
                    "review": {"status": "changes_requested", "findings": ["bad"]},
                }
            ],
            "execution_state": {
                "ready_queue": [],
                "active_projects": [],
                "completed_projects": [],
                "failed_projects": [],
                "skipped_projects": [],
                "total_cost_usd": 3.0,
            },
        }
        manifest_path.write_text(yaml.dump(manifest_data), encoding="utf-8")
        state_path = tmp_path / "state.yaml"
        _write_state(
            state_path,
            campaign_manifest_path=str(manifest_path),
            status=SupervisorStatus.RUNNING.value,
        )

        with patch(
            "aragora.ralph.supervisor.asyncio.run",
            return_value={"stop_reason": "campaign_blocked", "dispatched_projects": []},
        ):
            supervisor = RalphSupervisor(state_path=state_path, repo_root=tmp_path)
            result = supervisor.step()

        assert result.action == SupervisorAction.BLOCKER_CLASSIFIED.value
        state = load_supervisor_state(state_path)
        assert state.active_blocker == BlockerKind.REVIEWER_MISSING_DIFF.value
        assert len(state.blocker_history) == 1

    def test_campaign_iteration_exception_classifies_as_infra(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path / "manifest.yaml")
        state_path = tmp_path / "state.yaml"
        _write_state(
            state_path,
            campaign_manifest_path=str(manifest_path),
            status=SupervisorStatus.RUNNING.value,
        )

        with patch(
            "aragora.ralph.supervisor.asyncio.run",
            side_effect=RuntimeError("connection failed"),
        ):
            supervisor = RalphSupervisor(state_path=state_path, repo_root=tmp_path)
            result = supervisor.step()

        assert result.action == SupervisorAction.BLOCKER_CLASSIFIED.value
        state = load_supervisor_state(state_path)
        assert state.active_blocker == BlockerKind.INFRA_FAILURE.value


# ---------------------------------------------------------------------------
# Step: handle blocker
# ---------------------------------------------------------------------------


class TestStepHandleBlocker:
    def test_deterministic_blocker_generates_repair(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path / "manifest.yaml")
        state_path = tmp_path / "state.yaml"
        _write_state(
            state_path,
            campaign_manifest_path=str(manifest_path),
            status=SupervisorStatus.RUNNING.value,
            active_blocker=BlockerKind.REVIEWER_MISSING_DIFF.value,
        )

        supervisor = RalphSupervisor(state_path=state_path, repo_root=tmp_path)
        result = supervisor.step()

        assert result.action == SupervisorAction.REPAIR_GENERATED.value
        assert result.status == SupervisorStatus.WAITING_FOR_PR.value
        assert result.repair_task is not None
        assert (
            "reviewer" in result.repair_task.title.lower()
            or "diff" in result.repair_task.title.lower()
        )

        state = load_supervisor_state(state_path)
        assert state.status == SupervisorStatus.WAITING_FOR_PR.value
        assert state.repair_attempts == 1

    def test_non_deterministic_blocker_escalates(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path / "manifest.yaml")
        state_path = tmp_path / "state.yaml"
        _write_state(
            state_path,
            campaign_manifest_path=str(manifest_path),
            status=SupervisorStatus.RUNNING.value,
            active_blocker=BlockerKind.BUDGET_EXHAUSTION.value,
        )

        supervisor = RalphSupervisor(state_path=state_path, repo_root=tmp_path)
        result = supervisor.step()

        assert result.action == SupervisorAction.ESCALATED.value
        assert result.status == SupervisorStatus.ESCALATED.value

    def test_max_repair_attempts_escalates(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path / "manifest.yaml")
        state_path = tmp_path / "state.yaml"
        _write_state(
            state_path,
            campaign_manifest_path=str(manifest_path),
            status=SupervisorStatus.RUNNING.value,
            active_blocker=BlockerKind.REVIEWER_MISSING_DIFF.value,
            repair_attempts=2,
            max_repair_attempts=2,
        )

        supervisor = RalphSupervisor(state_path=state_path, repo_root=tmp_path)
        result = supervisor.step()

        assert result.action == SupervisorAction.ESCALATED.value


# ---------------------------------------------------------------------------
# Step: PR and merge checking
# ---------------------------------------------------------------------------


class TestStepPRChecking:
    def test_waiting_for_pr_with_pr_set(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.yaml"
        _write_state(
            state_path,
            status=SupervisorStatus.WAITING_FOR_PR.value,
            active_repair_pr="https://github.com/org/repo/pull/1",
        )

        supervisor = RalphSupervisor(state_path=state_path, repo_root=tmp_path)
        result = supervisor.step()

        assert result.action == SupervisorAction.PR_CHECKED.value
        assert result.status == SupervisorStatus.WAITING_FOR_MERGE.value

    def test_waiting_for_pr_discovers_pr(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.yaml"
        _write_state(
            state_path,
            status=SupervisorStatus.WAITING_FOR_PR.value,
            active_repair_branch="fix/reviewer-diff",
        )

        with patch.object(
            RalphSupervisor,
            "_find_pr_for_branch",
            return_value="https://github.com/org/repo/pull/42",
        ):
            supervisor = RalphSupervisor(state_path=state_path, repo_root=tmp_path)
            result = supervisor.step()

        assert result.status == SupervisorStatus.WAITING_FOR_MERGE.value
        state = load_supervisor_state(state_path)
        assert state.active_repair_pr == "https://github.com/org/repo/pull/42"

    def test_waiting_for_pr_no_pr_found(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.yaml"
        _write_state(
            state_path,
            status=SupervisorStatus.WAITING_FOR_PR.value,
            active_repair_branch="fix/reviewer-diff",
        )

        with patch.object(RalphSupervisor, "_find_pr_for_branch", return_value=None):
            supervisor = RalphSupervisor(state_path=state_path, repo_root=tmp_path)
            result = supervisor.step()

        assert result.status == SupervisorStatus.WAITING_FOR_PR.value

    def test_waiting_for_merge_not_merged(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.yaml"
        _write_state(
            state_path,
            status=SupervisorStatus.WAITING_FOR_MERGE.value,
            active_repair_pr="https://github.com/org/repo/pull/1",
        )

        with patch.object(RalphSupervisor, "_check_pr_merged", return_value=(False, None)):
            supervisor = RalphSupervisor(state_path=state_path, repo_root=tmp_path)
            result = supervisor.step()

        assert result.status == SupervisorStatus.WAITING_FOR_MERGE.value

    def test_waiting_for_merge_merged_transitions_to_resuming(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.yaml"
        _write_state(
            state_path,
            status=SupervisorStatus.WAITING_FOR_MERGE.value,
            active_repair_pr="https://github.com/org/repo/pull/1",
        )

        with patch.object(RalphSupervisor, "_check_pr_merged", return_value=(True, "abc123")):
            supervisor = RalphSupervisor(state_path=state_path, repo_root=tmp_path)
            result = supervisor.step()

        assert result.status == SupervisorStatus.RESUMING.value
        state = load_supervisor_state(state_path)
        assert state.merge_commit_sha == "abc123"


# ---------------------------------------------------------------------------
# Step: resume
# ---------------------------------------------------------------------------


class TestStepResume:
    def test_resume_clears_blocker_and_sets_running(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.yaml"
        _write_state(
            state_path,
            status=SupervisorStatus.RESUMING.value,
            active_blocker=BlockerKind.REVIEWER_MISSING_DIFF.value,
            active_repair_pr="https://github.com/org/repo/pull/1",
            active_repair_branch="fix/branch",
            active_repair_task={"title": "fix"},
            merge_commit_sha="abc",
        )

        with patch("aragora.ralph.supervisor.subprocess.run"):
            supervisor = RalphSupervisor(state_path=state_path, repo_root=tmp_path)
            result = supervisor.step()

        assert result.action == SupervisorAction.CAMPAIGN_RESUMED.value
        assert result.status == SupervisorStatus.RUNNING.value

        state = load_supervisor_state(state_path)
        assert state.active_blocker is None
        assert state.active_repair_pr is None
        assert state.active_repair_branch is None
        assert state.active_repair_task is None
        assert state.merge_commit_sha is None
        assert state.status == SupervisorStatus.RUNNING.value


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------


class TestStop:
    def test_stop_sets_stopped(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.yaml"
        _write_state(state_path, status=SupervisorStatus.RUNNING.value)

        supervisor = RalphSupervisor(state_path=state_path, repo_root=tmp_path)
        result = supervisor.stop()

        assert result.status == SupervisorStatus.STOPPED.value
        state = load_supervisor_state(state_path)
        assert state.status == SupervisorStatus.STOPPED.value


# ---------------------------------------------------------------------------
# Full loop simulation
# ---------------------------------------------------------------------------


class TestFullLoopSimulation:
    def test_run_classify_repair_merge_resume_complete(self, tmp_path: Path) -> None:
        """Simulate: run -> blocked -> classify -> repair -> PR -> merge -> resume -> complete."""
        manifest_path = tmp_path / "manifest.yaml"
        manifest_data = {
            "campaign_id": "sim-campaign",
            "created_at": "2026-01-01",
            "source_kind": "manual",
            "source_ref": "test",
            "worker_model": "codex",
            "review_model": "claude",
            "max_parallel_ready_projects": 1,
            "max_retries_per_project": 2,
            "budget_limit_usd": 20.0,
            "time_limit_hours": 4.0,
            "projects": [
                {
                    "project_id": "p1",
                    "status": "blocked",
                    "last_run_outcome": "deliverable_created",
                    "review": {"status": "changes_requested", "findings": ["insufficient"]},
                }
            ],
            "execution_state": {
                "ready_queue": [],
                "active_projects": [],
                "completed_projects": [],
                "failed_projects": [],
                "skipped_projects": [],
                "total_cost_usd": 2.0,
            },
        }
        manifest_path.write_text(yaml.dump(manifest_data), encoding="utf-8")
        state_path = tmp_path / "state.yaml"

        supervisor = RalphSupervisor.start(
            manifest_path=manifest_path,
            state_path=state_path,
            repo_root=tmp_path,
        )

        # Step 1: campaign iteration -> blocked
        with patch(
            "aragora.ralph.supervisor.asyncio.run",
            return_value={"stop_reason": "campaign_blocked", "dispatched_projects": []},
        ):
            r1 = supervisor.step()
        assert r1.action == SupervisorAction.BLOCKER_CLASSIFIED.value
        state = load_supervisor_state(state_path)
        assert state.active_blocker == BlockerKind.REVIEWER_MISSING_DIFF.value

        # Step 2: handle blocker -> generate repair
        r2 = supervisor.step()
        assert r2.action == SupervisorAction.REPAIR_GENERATED.value
        assert r2.repair_task is not None

        # Simulate: someone opens a PR
        state = load_supervisor_state(state_path)
        state.active_repair_pr = "https://github.com/org/repo/pull/99"
        save_supervisor_state(state_path, state)

        # Step 3: check PR -> found, wait for merge
        r3 = supervisor.step()
        assert r3.status == SupervisorStatus.WAITING_FOR_MERGE.value

        # Step 4: check merge -> merged
        with patch.object(RalphSupervisor, "_check_pr_merged", return_value=(True, "merged-sha")):
            r4 = supervisor.step()
        assert r4.status == SupervisorStatus.RESUMING.value

        # Step 5: resume -> running
        with patch("aragora.ralph.supervisor.subprocess.run"):
            r5 = supervisor.step()
        assert r5.action == SupervisorAction.CAMPAIGN_RESUMED.value
        assert r5.status == SupervisorStatus.RUNNING.value

        # Step 6: campaign iteration -> complete
        with patch(
            "aragora.ralph.supervisor.asyncio.run",
            return_value={"stop_reason": "campaign_complete", "dispatched_projects": []},
        ):
            r6 = supervisor.step()
        assert r6.action == SupervisorAction.CAMPAIGN_COMPLETED.value
        assert r6.status == SupervisorStatus.COMPLETED.value

        # Verify terminal.
        r7 = supervisor.step()
        assert r7.action == SupervisorAction.NOOP.value

    def test_unknown_blocker_escalates_immediately(self, tmp_path: Path) -> None:
        """Simulate: run -> blocked with unknown cause -> escalate."""
        manifest_path = _write_manifest(tmp_path / "manifest.yaml")
        state_path = tmp_path / "state.yaml"

        supervisor = RalphSupervisor.start(
            manifest_path=manifest_path,
            state_path=state_path,
            repo_root=tmp_path,
        )

        with patch(
            "aragora.ralph.supervisor.asyncio.run",
            return_value={"stop_reason": "campaign_blocked", "dispatched_projects": []},
        ):
            r1 = supervisor.step()

        # No projects in manifest → unknown blocker
        assert r1.action == SupervisorAction.BLOCKER_CLASSIFIED.value
        state = load_supervisor_state(state_path)
        assert state.active_blocker == BlockerKind.UNKNOWN.value

        # Step 2: unknown blocker → escalate
        r2 = supervisor.step()
        assert r2.action == SupervisorAction.ESCALATED.value
        assert r2.status == SupervisorStatus.ESCALATED.value
