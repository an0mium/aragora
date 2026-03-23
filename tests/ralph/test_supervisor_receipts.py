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
from aragora.receipts.lane import LaneCompletionReceipt, validate_receipt


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
        # Operational receipt + lane receipt both call emit_operational_receipt
        assert emit_receipt.call_count >= 1
        # Find the ralph operational receipt call
        ralph_calls = [c for c in emit_receipt.call_args_list if c.kwargs.get("source") == "ralph"]
        assert len(ralph_calls) == 1
        kwargs = ralph_calls[0].kwargs
        assert kwargs["action"] == "campaign_completed"
        assert kwargs["inputs"]["campaign_id"] == "test-campaign"
        assert kwargs["outputs"]["status"] == "completed"
        state = load_supervisor_state(state_path)
        assert state.status == SupervisorStatus.COMPLETED.value

    def test_campaign_complete_emits_lane_receipt(self, tmp_path: Path) -> None:
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
        lane_calls = [
            c for c in emit_receipt.call_args_list if c.kwargs.get("source") == "swarm_lane"
        ]
        assert len(lane_calls) == 1
        kwargs = lane_calls[0].kwargs
        assert kwargs["action"] == "lane_completed"
        assert kwargs["inputs"]["task_id"] == "test-campaign"
        assert kwargs["inputs"]["lease_id"] == "ralph-test123"
        assert kwargs["outputs"]["outcome"] == "pass"

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
        assert emit_receipt.call_count >= 1
        ralph_calls = [c for c in emit_receipt.call_args_list if c.kwargs.get("source") == "ralph"]
        assert len(ralph_calls) == 1
        kwargs = ralph_calls[0].kwargs
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


class TestLaneReceiptMatchback:
    def test_receipt_links_to_originating_task_and_lease(self) -> None:
        receipt = LaneCompletionReceipt(
            task_id="proj-001",
            lease_id="7316c435-b4d",
            agent_id="claude-worker-1",
            outcome="pass",
            pr_url="https://github.com/org/repo/pull/42",
            pr_number=42,
            branch="codex/swarm-dc0359aa-proj-001",
        )
        d = receipt.to_dict()
        assert d["task_id"] == "proj-001"
        assert d["lease_id"] == "7316c435-b4d"
        assert d["pr_url"] == "https://github.com/org/repo/pull/42"
        assert d["branch"] == "codex/swarm-dc0359aa-proj-001"
        assert validate_receipt(receipt) == []

    def test_malformed_receipt_detected(self) -> None:
        errors = validate_receipt({"outcome": "pass"})
        assert len(errors) >= 3
        field_names = " ".join(errors)
        assert "task_id" in field_names
        assert "lease_id" in field_names
        assert "agent_id" in field_names
