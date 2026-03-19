"""Focused tests for Ralph operational receipt emission."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import yaml

from aragora.ralph.classifier import BlockerKind
from aragora.ralph.supervisor import (
    RalphSupervisor,
    SupervisorState,
    SupervisorStatus,
    load_supervisor_state,
    save_supervisor_state,
)


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
        "execution_state": {"total_cost_usd": 3.5},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(manifest), encoding="utf-8")
    return path


def _write_state(path: Path, **overrides: object) -> SupervisorState:
    state = SupervisorState(
        supervisor_id="ralph-test123",
        campaign_manifest_path=str(overrides.pop("campaign_manifest_path", "/tmp/manifest.yaml")),
        campaign_id="test-campaign",
        **overrides,
    )
    save_supervisor_state(path, state)
    return state


class TestRalphOperationalReceipts:
    def test_campaign_complete_emits_receipt(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path / "manifest.yaml")
        state_path = tmp_path / "state.yaml"
        _write_state(
            state_path,
            campaign_manifest_path=str(manifest_path),
            status=SupervisorStatus.RUNNING.value,
        )

        with (
            patch(
                "aragora.swarm.campaign.CampaignExecutor.execute_once",
                new=AsyncMock(
                    return_value={"stop_reason": "campaign_complete", "dispatched_projects": []}
                ),
            ),
            patch("aragora.receipts.provenance.emit_operational_receipt") as emit_receipt,
        ):
            supervisor = RalphSupervisor(state_path=state_path, repo_root=tmp_path)
            result = supervisor.step()

        assert result.status == SupervisorStatus.COMPLETED.value
        emit_receipt.assert_called_once()
        kwargs = emit_receipt.call_args.kwargs
        assert kwargs["source"] == "ralph"
        assert kwargs["action"] == "campaign_completed"
        assert kwargs["inputs"]["campaign_id"] == "test-campaign"
        assert kwargs["outputs"]["status"] == "completed"
        state = load_supervisor_state(state_path)
        assert state.status == SupervisorStatus.COMPLETED.value

    def test_escalation_emits_receipt(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path / "manifest.yaml")
        state_path = tmp_path / "state.yaml"
        _write_state(
            state_path,
            campaign_manifest_path=str(manifest_path),
            status=SupervisorStatus.RUNNING.value,
            active_blocker=BlockerKind.BUDGET_EXHAUSTION.value,
        )

        with patch("aragora.receipts.provenance.emit_operational_receipt") as emit_receipt:
            supervisor = RalphSupervisor(state_path=state_path, repo_root=tmp_path)
            result = supervisor.step()

        assert result.status == SupervisorStatus.ESCALATED.value
        emit_receipt.assert_called_once()
        kwargs = emit_receipt.call_args.kwargs
        assert kwargs["action"] == "escalated"
        assert kwargs["verdict"] == "escalated"
        assert kwargs["outputs"]["reason"].startswith("Non-deterministic blocker:")

    def test_receipt_failures_do_not_block_escalation(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path / "manifest.yaml")
        state_path = tmp_path / "state.yaml"
        _write_state(
            state_path,
            campaign_manifest_path=str(manifest_path),
            status=SupervisorStatus.RUNNING.value,
            active_blocker=BlockerKind.BUDGET_EXHAUSTION.value,
        )

        with patch(
            "aragora.receipts.provenance.emit_operational_receipt",
            side_effect=RuntimeError("receipt path unavailable"),
        ):
            supervisor = RalphSupervisor(state_path=state_path, repo_root=tmp_path)
            result = supervisor.step()

        assert result.status == SupervisorStatus.ESCALATED.value
