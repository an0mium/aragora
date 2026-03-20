from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aragora.swarm.campaign import CampaignReviewGate
from aragora.swarm.tranche import TrancheLaneArtifact, TrancheManifest
from aragora.swarm.tranche_review import (
    review_lane,
    run_verification_passed,
    select_review_tier,
)


def _make_manifest(*, lanes: list[dict[str, object]]) -> TrancheManifest:
    return TrancheManifest.from_dict(
        {
            "manifest_id": "pmf-tranche",
            "repo": {"name": "synaptent/aragora", "root": "/tmp/repo", "base_ref": "origin/main"},
            "references": {"source_refs": {}},
            "gates": {},
            "lanes": lanes,
            "objective": "Ship the tranche",
            "terminal_outcomes": {"success": {"definition": "done"}},
        }
    )


def _make_lane(*, lane_id: str, write_scope: list[str]) -> dict[str, object]:
    return {
        "lane_id": lane_id,
        "owner_role": "engineer",
        "title": f"Lane {lane_id}",
        "prompt": f"Implement lane {lane_id}",
        "target_agent": "codex",
        "review_model": "claude",
        "source_refs": ["https://github.com/synaptent/aragora/issues/1046"],
        "allowed_write_scope": write_scope,
        "verification_commands": ["pytest -q"],
    }


def _make_artifact(*, lane_id: str, status: str, run_id: str) -> TrancheLaneArtifact:
    return TrancheLaneArtifact(
        lane_id=lane_id,
        source_ref="issue_1046",
        status=status,
        run_id=run_id,
        metadata={"review_model": "claude"},
    )


def test_narrow_scope_gets_tier_1() -> None:
    tier = select_review_tier(
        write_scope=["aragora/live/package.json"],
        diff_lines=5,
        verification_passed=True,
        risk_tolerance=None,
    )
    assert tier == 1


def test_medium_scope_gets_tier_2() -> None:
    tier = select_review_tier(
        write_scope=["aragora/server/**", "aragora/api/**", "aragora/auth/**"],
        diff_lines=150,
        verification_passed=True,
        risk_tolerance=None,
    )
    assert tier == 2


def test_broad_scope_gets_tier_3() -> None:
    tier = select_review_tier(
        write_scope=[
            "aragora/server/**",
            "aragora/api/**",
            "aragora/auth/**",
            "aragora/swarm/**",
        ],
        diff_lines=500,
        verification_passed=False,
        risk_tolerance=None,
    )
    assert tier == 3


def test_explicit_risk_override() -> None:
    tier = select_review_tier(
        write_scope=["aragora/live/package.json"],
        diff_lines=2,
        verification_passed=True,
        risk_tolerance="high",
    )
    assert tier == 3


@pytest.mark.asyncio
async def test_tier_1_review_delegates_to_campaign_reviewer() -> None:
    manifest = _make_manifest(lanes=[_make_lane(lane_id="a", write_scope=["aragora/live/**"])])
    artifact = _make_artifact(lane_id="a", status="completed", run_id="run-1")

    mock_reviewer = AsyncMock()
    mock_reviewer.review.return_value = CampaignReviewGate(
        status="passed",
        findings=[],
        review_model="claude",
    )

    result = await review_lane(
        manifest=manifest,
        lane_id="a",
        artifact=artifact,
        run_dict={"run_id": "run-1", "status": "completed", "work_orders": []},
        reviewer=mock_reviewer,
        tier=1,
        repo_root=Path("/tmp/repo"),
    )

    assert result["status"] == "passed"
    mock_reviewer.review.assert_awaited_once()


@pytest.mark.asyncio
async def test_tier_2_runs_two_reviewers_and_synthesizes() -> None:
    manifest = _make_manifest(lanes=[_make_lane(lane_id="a", write_scope=["aragora/server/**"])])
    artifact = _make_artifact(lane_id="a", status="completed", run_id="run-1")
    run_dict = {"run_id": "run-1", "status": "completed", "work_orders": []}

    reviewer_1 = AsyncMock()
    reviewer_1.review.return_value = CampaignReviewGate(
        status="passed",
        findings=[],
        review_model="claude",
    )
    reviewer_2 = AsyncMock()
    reviewer_2.review.return_value = CampaignReviewGate(
        status="passed",
        findings=[],
        review_model="gpt-4",
    )

    result = await review_lane(
        manifest=manifest,
        lane_id="a",
        artifact=artifact,
        run_dict=run_dict,
        reviewers=[reviewer_1, reviewer_2],
        tier=2,
        repo_root=Path("/tmp/repo"),
    )
    assert result["status"] == "passed"
    assert result["tier"] == 2
    reviewer_1.review.assert_awaited_once()
    reviewer_2.review.assert_awaited_once()


@pytest.mark.asyncio
async def test_tier_3_retries_with_findings_as_constraints() -> None:
    manifest = _make_manifest(lanes=[_make_lane(lane_id="a", write_scope=["aragora/server/**"])])
    artifact = _make_artifact(lane_id="a", status="completed", run_id="run-1")
    run_dict = {"run_id": "run-1", "status": "completed", "work_orders": []}

    reviewer = AsyncMock()
    reviewer.review.return_value = CampaignReviewGate(
        status="changes_requested",
        findings=["Missing error handling in endpoint"],
        review_model="claude",
    )

    mock_dispatch = AsyncMock(
        return_value={
            "status": "completed",
            "outcome": "deliverable_created",
            "run_id": "retry-run-1",
            "run": {"run_id": "retry-run-1", "status": "completed"},
        }
    )

    result = await review_lane(
        manifest=manifest,
        lane_id="a",
        artifact=artifact,
        run_dict=run_dict,
        reviewer=reviewer,
        tier=3,
        dispatch_fn=mock_dispatch,
        max_retries=2,
        repo_root=Path("/tmp/repo"),
    )

    dispatch_call = mock_dispatch.call_args
    spec_arg = dispatch_call.args[0] if dispatch_call.args else dispatch_call.kwargs.get("spec")
    assert "Missing error handling in endpoint" in str(spec_arg.constraints)
    assert result["retry_count"] <= 2
    assert result["status"] in ("passed", "needs_human")


@pytest.mark.asyncio
async def test_tier_3_stops_after_max_retries() -> None:
    manifest = _make_manifest(lanes=[_make_lane(lane_id="a", write_scope=["aragora/server/**"])])
    artifact = _make_artifact(lane_id="a", status="completed", run_id="run-1")
    run_dict = {"run_id": "run-1", "status": "completed", "work_orders": []}

    reviewer = AsyncMock()
    reviewer.review.return_value = CampaignReviewGate(
        status="changes_requested",
        findings=["Still broken"],
        review_model="claude",
    )
    mock_dispatch = AsyncMock(
        return_value={
            "status": "completed",
            "outcome": "deliverable_created",
            "run_id": "retry-run",
            "run": {"run_id": "retry-run", "status": "completed"},
        }
    )

    result = await review_lane(
        manifest=manifest,
        lane_id="a",
        artifact=artifact,
        run_dict=run_dict,
        reviewer=reviewer,
        tier=3,
        dispatch_fn=mock_dispatch,
        max_retries=2,
        repo_root=Path("/tmp/repo"),
    )

    assert result["status"] == "needs_human"
    assert result["retry_count"] == 2


@pytest.mark.asyncio
async def test_tier_3_accumulates_findings_across_retries() -> None:
    manifest = _make_manifest(lanes=[_make_lane(lane_id="a", write_scope=["aragora/server/**"])])
    artifact = _make_artifact(lane_id="a", status="completed", run_id="run-1")
    run_dict = {"run_id": "run-1", "status": "completed", "work_orders": []}

    reviewer = AsyncMock()
    reviewer.review.side_effect = [
        CampaignReviewGate(
            status="changes_requested",
            findings=["Missing error handling"],
            review_model="claude",
        ),
        CampaignReviewGate(
            status="changes_requested",
            findings=["Add regression test"],
            review_model="claude",
        ),
        CampaignReviewGate(
            status="passed",
            findings=[],
            review_model="claude",
        ),
    ]
    mock_dispatch = AsyncMock(
        side_effect=[
            {
                "status": "completed",
                "outcome": "deliverable_created",
                "run_id": "retry-run-1",
                "run": {"run_id": "retry-run-1", "status": "completed"},
            },
            {
                "status": "completed",
                "outcome": "deliverable_created",
                "run_id": "retry-run-2",
                "run": {"run_id": "retry-run-2", "status": "completed"},
            },
        ]
    )

    result = await review_lane(
        manifest=manifest,
        lane_id="a",
        artifact=artifact,
        run_dict=run_dict,
        reviewer=reviewer,
        tier=3,
        dispatch_fn=mock_dispatch,
        max_retries=2,
        repo_root=Path("/tmp/repo"),
    )

    first_spec = mock_dispatch.await_args_list[0].args[0]
    second_spec = mock_dispatch.await_args_list[1].args[0]
    assert "Missing error handling" in str(first_spec.constraints)
    assert "Missing error handling" in str(second_spec.constraints)
    assert "Add regression test" in str(second_spec.constraints)
    assert result["status"] == "passed"
    assert result["findings"] == ["Missing error handling", "Add regression test"]


def test_run_verification_passed_uses_actual_results() -> None:
    run_dict = {
        "work_orders": [
            {
                "verification_results": [
                    {"command": "pytest -q", "passed": False, "exit_code": 1},
                ]
            }
        ]
    }

    assert run_verification_passed(run_dict, has_verification_commands=True) is False


def test_run_verification_passed_defaults_false_when_expected_results_missing() -> None:
    run_dict = {"work_orders": [{"status": "completed"}]}

    assert run_verification_passed(run_dict, has_verification_commands=True) is False
