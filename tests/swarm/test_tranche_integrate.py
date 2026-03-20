from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aragora.nomic.dev_coordination import IntegrationDecisionType
from aragora.swarm.pr_registry import PullRequestRegistry
from aragora.swarm.tranche_integrate import (
    assess_lane_integration,
    classify_check_results,
    discover_lane_pr,
    record_lane_integration,
    register_pr,
)


def _make_artifact(
    *, metadata: dict | None = None, urls: list[str] | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        lane_id="lane-a",
        metadata=metadata or {},
        urls=urls or [],
    )


def test_discover_pr_from_artifact_metadata():
    artifact = _make_artifact(
        metadata={"deliverable": {"pr_url": "https://github.com/org/repo/pull/42"}}
    )
    pr = discover_lane_pr(artifact)
    assert pr == "https://github.com/org/repo/pull/42"


def test_discover_pr_from_branch_lookup():
    github = MagicMock()
    github.find_pr_for_branch.return_value = "https://github.com/org/repo/pull/99"
    artifact = _make_artifact(metadata={"branch": "feat-branch"})

    pr = discover_lane_pr(artifact, github=github)

    assert pr == "https://github.com/org/repo/pull/99"
    github.find_pr_for_branch.assert_called_once_with("feat-branch")


def test_discover_and_register_pr():
    registry = MagicMock(spec=PullRequestRegistry)
    artifact = _make_artifact(
        metadata={"deliverable": {"pr_url": "https://github.com/org/repo/pull/42"}}
    )
    pr = discover_lane_pr(artifact)

    register_pr(pr, "feat-branch", registry)

    registry.register.assert_called_once_with(
        "feat-branch",
        "https://github.com/org/repo/pull/42",
        creator="tranche-integrate",
    )


def test_register_pr_skips_existing_same_url():
    registry = MagicMock(spec=PullRequestRegistry)
    registry.get.return_value = {
        "branch": "feat-branch",
        "pr_url": "https://github.com/org/repo/pull/42",
    }

    register_pr("https://github.com/org/repo/pull/42", "feat-branch", registry)

    registry.register.assert_not_called()


def test_classify_checks_all_green():
    checks = [
        {"name": "lint", "conclusion": "SUCCESS", "required": True},
        {"name": "typecheck", "conclusion": "SUCCESS", "required": True},
    ]
    result = classify_check_results(checks)
    assert result == "checks_passed"


def test_classify_checks_required_failure():
    checks = [
        {"name": "lint", "conclusion": "FAILURE", "required": True},
        {"name": "typecheck", "conclusion": "SUCCESS", "required": True},
    ]
    result = classify_check_results(checks)
    assert result == "checks_failed"


def test_classify_checks_pending_required():
    checks = [
        {"name": "lint", "status": "IN_PROGRESS", "required": True},
        {"name": "typecheck", "conclusion": "SUCCESS", "required": True},
    ]
    result = classify_check_results(checks)
    assert result == "checks_pending"


def test_classify_checks_advisory_noise():
    checks = [
        {"name": "lint", "conclusion": "SUCCESS", "required": True},
        {"name": "Self-Host Compose Smoke", "conclusion": "FAILURE", "required": False},
    ]
    result = classify_check_results(checks)
    assert result == "checks_passed"


def test_classify_checks_completed_without_conclusion_is_green() -> None:
    checks = [
        {"name": "lint", "status": "COMPLETED", "required": True},
        {"name": "typecheck", "conclusion": "SUCCESS", "required": True},
    ]

    result = classify_check_results(checks)

    assert result == "checks_passed"


def test_assess_returns_recommendation_without_merging():
    result = assess_lane_integration(
        artifact=_make_artifact(),
        checks="checks_passed",
        review_status="passed",
        merge_policy="confirm",
    )
    assert result["recommendation"] == "merge"
    assert result["executed"] is False


def test_merge_executes_with_approve_flag():
    result = assess_lane_integration(
        artifact=_make_artifact(),
        checks="checks_passed",
        review_status="passed",
        merge_policy="auto",
        approve=True,
    )
    assert result["recommendation"] == "merge"
    assert result["executed"] is True


@pytest.mark.asyncio
async def test_integrate_records_decision_in_coordination_store():
    mock_store = MagicMock()
    artifact = _make_artifact(
        metadata={"receipt_id": "receipt-xyz", "lease_id": "lease-abc"},
    )

    await record_lane_integration(
        artifact=artifact,
        decision="merge",
        rationale="checks passed, review approved",
        decided_by="tranche-integrate",
        store=mock_store,
        target_branch="main",
    )

    mock_store.record_integration_decision.assert_called_once()
    call_kwargs = mock_store.record_integration_decision.call_args.kwargs
    assert call_kwargs["receipt_id"] == "receipt-xyz"
    assert call_kwargs["lease_id"] == "lease-abc"
    assert call_kwargs["decision"] == IntegrationDecisionType.MERGE


def test_assess_checkpoint_mode_always_awaits_confirmation():
    result = assess_lane_integration(
        artifact=_make_artifact(),
        checks="checks_passed",
        review_status="passed",
        merge_policy="auto",
        autonomy_mode="checkpoint",
    )
    assert result["recommendation"] == "merge"
    assert result["executed"] is False


def test_assess_fire_and_forget_auto_merges():
    result = assess_lane_integration(
        artifact=_make_artifact(),
        checks="checks_passed",
        review_status="passed",
        merge_policy="auto",
        autonomy_mode="fire_and_forget",
    )
    assert result["recommendation"] == "merge"
    assert result["executed"] is True


def test_assess_fire_and_forget_confirm_policy_still_waits() -> None:
    result = assess_lane_integration(
        artifact=_make_artifact(),
        checks="checks_passed",
        review_status="passed",
        merge_policy="confirm",
        autonomy_mode="fire_and_forget",
    )

    assert result["recommendation"] == "merge"
    assert result["executed"] is False
