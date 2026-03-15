"""Tests for automatic receipt emission on terminal project transitions."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.swarm.campaign import (
    CampaignExecutor,
    CampaignManifest,
    CampaignProject,
    CampaignProjectStatus,
    CampaignReviewGate,
    CampaignReviewStatus,
    CampaignRunOutcome,
    _derive_phase,
    _failure_classification_from_outcome,
    _receipt_final_status,
    _receipt_review_verdict,
)


# ---------------------------------------------------------------------------
# Helper free-function tests
# ---------------------------------------------------------------------------


class TestDerivePhase:
    def test_phase0a(self) -> None:
        assert _derive_phase("phase0a-bootstrap-governance") == "0a"

    def test_phase0b(self) -> None:
        assert _derive_phase("phase0b-engine-hardening") == "0b"

    def test_phase1(self) -> None:
        assert _derive_phase("phase1-controlled-repair") == "1"

    def test_phase2(self) -> None:
        assert _derive_phase("phase2-broader-cleanup") == "2"

    def test_unknown_campaign_id(self) -> None:
        assert _derive_phase("campaign-dogfood-6") is None

    def test_empty_string(self) -> None:
        assert _derive_phase("") is None

    def test_case_insensitive(self) -> None:
        assert _derive_phase("Phase0A-test") == "0a"


class TestFailureClassification:
    def test_crash(self) -> None:
        assert _failure_classification_from_outcome("crash") == "worker_crash"

    def test_timeout(self) -> None:
        assert _failure_classification_from_outcome("timeout") == "timeout"

    def test_blocked(self) -> None:
        assert _failure_classification_from_outcome("blocked") == "stall"

    def test_needs_human(self) -> None:
        assert _failure_classification_from_outcome("needs_human") == "stall"

    def test_deliverable_created(self) -> None:
        assert _failure_classification_from_outcome("deliverable_created") is None

    def test_none(self) -> None:
        assert _failure_classification_from_outcome(None) is None


class TestReceiptFinalStatus:
    def test_completed(self) -> None:
        assert _receipt_final_status("completed") == "completed"

    def test_failed(self) -> None:
        assert _receipt_final_status("failed") == "failed"

    def test_skipped(self) -> None:
        assert _receipt_final_status("skipped") == "failed"

    def test_blocked(self) -> None:
        assert _receipt_final_status("blocked") == "rejected"


class TestReceiptReviewVerdict:
    def test_passed(self) -> None:
        assert _receipt_review_verdict("passed") == "passed"

    def test_changes_requested(self) -> None:
        assert _receipt_review_verdict("changes_requested") == "failed"

    def test_pending(self) -> None:
        assert _receipt_review_verdict("pending") == "skipped"


# ---------------------------------------------------------------------------
# Receipt emission integration tests
# ---------------------------------------------------------------------------


def _make_project(
    project_id: str = "test-001",
    status: str = "ready",
    *,
    branch: str = "codex/test-branch",
    estimated_cost_usd: float = 1.0,
) -> CampaignProject:
    return CampaignProject(
        project_id=project_id,
        title=f"Test project {project_id}",
        status=status,
        estimated_cost_usd=estimated_cost_usd,
        branch=branch,
        spec=MagicMock(to_dict=lambda: {}),
    )


def _make_manifest(
    campaign_id: str = "phase0a-test",
    projects: list[CampaignProject] | None = None,
) -> CampaignManifest:
    m = CampaignManifest(
        campaign_id=campaign_id,
        created_at="2026-03-10T21:00:00+00:00",
        source_kind="test",
        source_ref="test",
    )
    if projects:
        m.projects = projects
    return m


class TestEmitReceipt:
    def test_receipt_written_on_completed(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("campaign_id: phase0a-test\n")
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        project = _make_project(status="completed")
        manifest = _make_manifest(projects=[project])

        receipt_path = executor._emit_receipt(manifest, project, None)

        assert receipt_path.exists()
        content = receipt_path.read_text()
        assert "task_id: test-001" in content
        assert "campaign_id: phase0a-test" in content
        assert "phase: 0a" in content
        assert "final_status: completed" in content
        assert "rescue_required: false" in content

    def test_receipt_written_on_failed(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("campaign_id: phase0a-test\n")
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        project = _make_project(status="failed")
        project.last_run_outcome = CampaignRunOutcome.TIMEOUT.value
        manifest = _make_manifest(projects=[project])

        receipt_path = executor._emit_receipt(manifest, project, None)

        content = receipt_path.read_text()
        assert "final_status: failed" in content
        assert "failure_classification: timeout" in content

    def test_receipt_written_on_blocked(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("campaign_id: phase0a-test\n")
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        project = _make_project(status="blocked")
        project.review = CampaignReviewGate(
            status=CampaignReviewStatus.BLOCKED_NONREVIEWABLE.value,
        )
        manifest = _make_manifest(projects=[project])

        receipt_path = executor._emit_receipt(manifest, project, None)

        content = receipt_path.read_text()
        assert "final_status: rejected" in content

    def test_receipt_path_correct(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("campaign_id: my-campaign\n")
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        project = _make_project(project_id="proj-007", status="completed")
        manifest = _make_manifest(campaign_id="my-campaign", projects=[project])

        receipt_path = executor._emit_receipt(manifest, project, None)

        assert receipt_path == tmp_path / "docs" / "receipts" / "my-campaign" / "proj-007.yaml"

    def test_receipt_sets_receipt_id_on_project(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("campaign_id: phase0a-test\n")
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        project = _make_project(status="completed")
        manifest = _make_manifest(projects=[project])

        executor._emit_receipt(manifest, project, None)

        assert project.receipt_id == "docs/receipts/phase0a-test/test-001.yaml"

    def test_receipt_preserves_worker_receipt_id(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("campaign_id: phase0a-test\n")
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        project = _make_project(status="completed")
        project.worker_receipt_id = "worker-receipt-123"
        manifest = _make_manifest(projects=[project])

        receipt_path = executor._emit_receipt(manifest, project, None)

        content = receipt_path.read_text()
        assert "worker_receipt_id: worker-receipt-123" in content
        assert project.worker_receipt_id == "worker-receipt-123"

    def test_receipt_raises_on_write_failure(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("campaign_id: phase0a-test\n")
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        project = _make_project(status="completed")
        manifest = _make_manifest(projects=[project])

        with patch.object(Path, "mkdir", side_effect=PermissionError("denied")):
            with pytest.raises(RuntimeError, match="Failed to emit receipt"):
                executor._emit_receipt(manifest, project, None)

    def test_receipt_extracts_worker_info_from_run_dict(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("campaign_id: phase0a-test\n")
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        project = _make_project(status="completed", branch="")
        manifest = _make_manifest(projects=[project])
        run_dict = {
            "work_orders": [
                {
                    "branch": "codex/from-run",
                    "head_sha": "abc123",
                    "changed_paths": ["docs/test.md"],
                    "started_at": "2026-03-10T21:00:00+00:00",
                    "completed_at": "2026-03-10T21:05:00+00:00",
                }
            ]
        }

        receipt_path = executor._emit_receipt(manifest, project, run_dict)

        content = receipt_path.read_text()
        assert "codex/from-run" in content
        assert "abc123" in content
        assert "docs/test.md" in content
        assert "duration_seconds: 300" in content

    def test_receipt_always_marks_rescue_false(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("campaign_id: phase0a-test\n")
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        project = _make_project(status="completed")
        manifest = _make_manifest(projects=[project])

        receipt_path = executor._emit_receipt(manifest, project, None)

        content = receipt_path.read_text()
        assert "rescue_required: false" in content
        assert "rescue_description: null" in content


class TestApplyDispatchResultEmitsReceipt:
    def test_receipt_emitted_on_terminal_dispatch(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("campaign_id: phase0a-test\n")
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        project = _make_project(status="active")
        manifest = _make_manifest(projects=[project])
        manifest.max_retries_per_project = 0

        result = {
            "run_id": "run-001",
            "run": {},
            "deliverable": {},
            "outcome": CampaignRunOutcome.NEEDS_HUMAN.value,
        }
        executor._apply_dispatch_result(manifest, project, result)

        assert project.status == CampaignProjectStatus.BLOCKED.value
        receipt_path = tmp_path / "docs" / "receipts" / "phase0a-test" / "test-001.yaml"
        assert receipt_path.exists()

    def test_no_receipt_on_delivered(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("campaign_id: phase0a-test\n")
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        project = _make_project(status="active")
        manifest = _make_manifest(projects=[project])

        result = {
            "run_id": "run-001",
            "run": {},
            "deliverable": {},
            "outcome": CampaignRunOutcome.DELIVERABLE_CREATED.value,
        }
        executor._apply_dispatch_result(manifest, project, result)

        assert project.status == CampaignProjectStatus.DELIVERED.value
        receipt_path = tmp_path / "docs" / "receipts" / "phase0a-test" / "test-001.yaml"
        assert not receipt_path.exists()

    def test_no_receipt_on_needs_revision(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("campaign_id: phase0a-test\n")
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        project = _make_project(status="active")
        manifest = _make_manifest(projects=[project])
        manifest.max_retries_per_project = 5

        result = {
            "run_id": "run-001",
            "run": {},
            "deliverable": {},
            "outcome": CampaignRunOutcome.CLEAN_EXIT_NO_DELIVERABLE.value,
        }
        executor._apply_dispatch_result(manifest, project, result)

        assert project.status == CampaignProjectStatus.NEEDS_REVISION.value
        receipt_path = tmp_path / "docs" / "receipts" / "phase0a-test" / "test-001.yaml"
        assert not receipt_path.exists()

    def test_terminal_dispatch_preserves_worker_receipt_and_attempt_history(
        self, tmp_path: Path
    ) -> None:
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("campaign_id: phase0a-test\n")
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        project = _make_project(status="active")
        manifest = _make_manifest(projects=[project])
        manifest.max_retries_per_project = 0

        result = {
            "run_id": "run-001",
            "run": {
                "run_id": "run-001",
                "status": "needs_human",
                "work_orders": [
                    {
                        "status": "needs_human",
                        "receipt_id": "worker-receipt-123",
                        "dispatch_error": "human approval required",
                    }
                ],
            },
            "deliverable": {},
            "outcome": CampaignRunOutcome.NEEDS_HUMAN.value,
        }
        executor._apply_dispatch_result(manifest, project, result)

        assert project.worker_receipt_id == "worker-receipt-123"
        assert project.receipt_id == "docs/receipts/phase0a-test/test-001.yaml"
        assert len(project.attempt_history) == 1
        assert project.attempt_history[0]["worker_receipt_id"] == "worker-receipt-123"
        assert project.attempt_history[0]["campaign_receipt_id"] == project.receipt_id


class TestReadyProjectsEmitsReceipt:
    def test_receipt_emitted_when_retry_exhausted(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("campaign_id: phase0a-test\n")
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        project = _make_project(status="needs_revision")
        project.retry_count = 3
        manifest = _make_manifest(projects=[project])
        manifest.max_retries_per_project = 2

        executor._ready_projects(manifest)

        assert project.status == CampaignProjectStatus.SKIPPED.value
        receipt_path = tmp_path / "docs" / "receipts" / "phase0a-test" / "test-001.yaml"
        assert receipt_path.exists()
        content = receipt_path.read_text()
        assert "final_status: failed" in content  # skipped maps to failed


class TestApplyReviewResultEmitsReceipt:
    def test_receipt_emitted_on_completed_review(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("campaign_id: phase0a-test\n")
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        project = _make_project(status="delivered")
        manifest = _make_manifest(projects=[project])
        gate = CampaignReviewGate(status=CampaignReviewStatus.PASSED.value)

        executor._apply_review_result(manifest, project, gate)

        assert project.status == CampaignProjectStatus.COMPLETED.value
        receipt_path = tmp_path / "docs" / "receipts" / "phase0a-test" / "test-001.yaml"
        assert receipt_path.exists()
        content = receipt_path.read_text()
        assert "review_verdict: passed" in content
        assert "final_status: completed" in content

    def test_completed_review_receipt_includes_budget_accounting_and_truth_suite_placeholder(
        self, tmp_path: Path
    ) -> None:
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("campaign_id: phase0a-test\n")
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        project = _make_project(status="delivered", estimated_cost_usd=3.5)
        manifest = _make_manifest(projects=[project])
        gate = CampaignReviewGate(status=CampaignReviewStatus.PASSED.value)

        executor._apply_review_result(manifest, project, gate)

        receipt_path = tmp_path / "docs" / "receipts" / "phase0a-test" / "test-001.yaml"
        content = receipt_path.read_text()
        assert "cost_usd: 3.5" in content
        assert "truth_suite: null" in content

    def test_no_receipt_on_changes_requested(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("campaign_id: phase0a-test\n")
        executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

        project = _make_project(status="delivered")
        manifest = _make_manifest(projects=[project])
        gate = CampaignReviewGate(status=CampaignReviewStatus.CHANGES_REQUESTED.value)

        executor._apply_review_result(manifest, project, gate)

        assert project.status == CampaignProjectStatus.NEEDS_REVISION.value
        receipt_path = tmp_path / "docs" / "receipts" / "phase0a-test" / "test-001.yaml"
        assert not receipt_path.exists()
