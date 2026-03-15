"""Focused tests for campaign budget gating and visibility."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from aragora.swarm.campaign import (
    CampaignExecutor,
    CampaignManifest,
    CampaignProject,
    CampaignProjectStatus,
    CampaignReviewGate,
    CampaignReviewStatus,
    CampaignReviewer,
    CampaignStopReason,
    CampaignExecutionState,
    load_campaign_manifest,
    save_campaign_manifest,
)
from aragora.swarm.spec import SwarmSpec


def _bounded_spec(goal: str, scope: list[str] | None = None) -> SwarmSpec:
    return SwarmSpec(
        raw_goal=goal,
        refined_goal=goal,
        acceptance_criteria=["pytest -q tests/swarm/test_campaign_budget_enforcement.py"],
        constraints=["do not widen scope"],
        file_scope_hints=scope or ["aragora/swarm/campaign.py"],
        budget_limit_usd=5.0,
    )


@pytest.mark.asyncio
async def test_execute_once_halts_before_over_budget_dispatch(tmp_path: Path):
    manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
    manifest = CampaignManifest(
        campaign_id="campaign-budget-cap",
        created_at="2026-03-10T00:00:00+00:00",
        source_kind="source_file",
        source_ref="roadmap.md",
        budget_limit_usd=1.0,
        projects=[
            CampaignProject(
                project_id="proj-001",
                title="Too expensive",
                spec=_bounded_spec("Too expensive"),
                file_scope_hints=["aragora/swarm/campaign.py"],
                acceptance_criteria=["pytest -q tests/swarm/test_campaign.py"],
                constraints=["do not widen scope"],
                estimated_cost_usd=2.0,
            )
        ],
    )
    save_campaign_manifest(manifest_path, manifest)
    executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

    with patch("aragora.swarm.campaign.dispatch_bounded_spec", new=AsyncMock()) as mock_dispatch:
        payload = await executor.execute_once()

    reloaded = load_campaign_manifest(manifest_path)
    status_payload = executor.status()

    assert mock_dispatch.await_count == 0
    assert payload["stop_reason"] == CampaignStopReason.BUDGET_EXHAUSTED.value
    assert payload["budget_blocked_projects"] == ["proj-001"]
    assert payload["budget"]["available_budget_usd"] == 1.0
    assert reloaded.projects[0].status in {
        CampaignProjectStatus.PENDING.value,
        CampaignProjectStatus.READY.value,
    }
    assert status_payload["stop_reason"] == CampaignStopReason.BUDGET_EXHAUSTED.value
    assert status_payload["budget"]["available_budget_usd"] == 1.0
    assert status_payload["projects"][0]["estimated_cost_usd"] == 2.0


@pytest.mark.asyncio
async def test_execute_once_keeps_running_while_active_budget_is_reserved(tmp_path: Path):
    manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
    manifest = CampaignManifest(
        campaign_id="campaign-budget-reserved",
        created_at="2026-03-10T00:00:00+00:00",
        source_kind="source_file",
        source_ref="roadmap.md",
        budget_limit_usd=1.0,
        projects=[
            CampaignProject(
                project_id="proj-001",
                title="In-flight expensive task",
                spec=_bounded_spec("In-flight expensive task"),
                file_scope_hints=["aragora/swarm/campaign.py"],
                acceptance_criteria=["tests pass"],
                constraints=["stay in scope"],
                status=CampaignProjectStatus.ACTIVE.value,
                run_id="run-expensive",
                estimated_cost_usd=0.75,
            ),
            CampaignProject(
                project_id="proj-002",
                title="Ready but unaffordable",
                spec=_bounded_spec("Ready but unaffordable", ["docs/CLI_REFERENCE.md"]),
                file_scope_hints=["docs/CLI_REFERENCE.md"],
                acceptance_criteria=["tests pass"],
                constraints=["stay in scope"],
                estimated_cost_usd=0.50,
            ),
        ],
    )
    save_campaign_manifest(manifest_path, manifest)
    executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

    with (
        patch.object(
            executor,
            "_refresh_run_dict",
            return_value={"run_id": "run-expensive", "status": "running", "work_orders": []},
        ),
        patch("aragora.swarm.campaign.dispatch_bounded_spec", new=AsyncMock()) as mock_dispatch,
    ):
        payload = await executor.execute_once()

    status_payload = executor.status()

    assert mock_dispatch.await_count == 0
    assert payload["stop_reason"] == CampaignStopReason.STILL_RUNNING.value
    assert payload["budget_blocked_projects"] == ["proj-002"]
    assert payload["budget"]["reserved_cost_usd"] == 0.75
    assert payload["budget"]["available_budget_usd"] == 0.25
    assert status_payload["budget"]["reserved_cost_usd"] == 0.75
    assert status_payload["budget"]["available_budget_usd"] == 0.25


def test_status_exposes_budget_accounting_and_review_state(tmp_path: Path):
    manifest_path = tmp_path / ".aragora" / "campaign_manifest.yaml"
    manifest = CampaignManifest(
        campaign_id="campaign-budget-status",
        created_at="2026-03-10T00:00:00+00:00",
        source_kind="source_file",
        source_ref="roadmap.md",
        budget_limit_usd=3.0,
        projects=[
            CampaignProject(
                project_id="proj-001",
                title="Reviewed project",
                spec=_bounded_spec("Reviewed project"),
                file_scope_hints=["aragora/swarm/campaign.py"],
                acceptance_criteria=["pytest -q tests/swarm/test_campaign.py"],
                constraints=["do not widen scope"],
                status=CampaignProjectStatus.DELIVERED.value,
                review=CampaignReviewGate(
                    required=True,
                    review_model="claude",
                    status=CampaignReviewStatus.PENDING.value,
                ),
            )
        ],
        execution_state=CampaignExecutionState(total_cost_usd=1.25),
    )
    save_campaign_manifest(manifest_path, manifest)
    executor = CampaignExecutor(manifest_path=manifest_path, repo_root=tmp_path)

    status = executor.status()

    assert status["budget_limit_usd"] == pytest.approx(3.0)
    assert status["total_cost_usd"] == pytest.approx(1.25)
    assert status["budget"]["available_budget_usd"] == pytest.approx(1.75)
    assert status["projects"][0]["review_status"] == CampaignReviewStatus.PENDING.value


def test_review_prompt_includes_budget_context():
    project = CampaignProject(
        project_id="proj-001",
        title="Budget visible review",
        spec=_bounded_spec("Budget visible review"),
        acceptance_criteria=["review the budget context"],
        file_scope_hints=["aragora/swarm/campaign.py"],
    )
    prompt = CampaignReviewer._build_prompt(
        project,
        {"work_orders": []},
        "claude",
        budget_context={
            "budget_limit_usd": 5.0,
            "spent_cost_usd": 1.5,
            "reserved_cost_usd": 0.5,
            "available_budget_usd": 3.0,
            "project_estimated_cost_usd": 1.0,
        },
    )

    parsed = json.loads(prompt.split("\n")[-1])
    assert parsed["budget"]["budget_limit_usd"] == 5.0
    assert parsed["budget"]["available_budget_usd"] == 3.0
    assert parsed["budget"]["project_estimated_cost_usd"] == 1.0
