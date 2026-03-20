from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aragora.nomic.dev_coordination import IntegrationDecisionType
from aragora.swarm.pr_registry import PullRequestRegistry
from aragora.swarm.tranche import TrancheLane, TrancheLaneArtifact
from aragora.swarm.tranche_integrate import (
    assess_lane_integration,
    cascade_after_merge,
    classify_check_results,
    discover_lane_pr,
    execute_lane_merge,
    integrate_lane,
    publish_lane_deliverable,
    record_lane_integration,
    register_pr,
)
from aragora.swarm.tranche_state import (
    LANE_STATUS_NEEDS_HUMAN,
    LANE_STATUS_WAITING_FOR_MERGE,
    LaneRunState,
    TrancheRunState,
)


def _make_artifact(
    *,
    lane_id: str = "lane-a",
    status: str = "review_passed",
    metadata: dict | None = None,
    urls: list[str] | None = None,
) -> TrancheLaneArtifact:
    return TrancheLaneArtifact(
        lane_id=lane_id,
        source_ref=f"lane:{lane_id}",
        status=status,
        metadata=metadata or {},
        urls=urls or [],
    )


def _make_lane(
    lane_id: str,
    *,
    dependencies: list[str] | None = None,
    branch: dict | None = None,
    allowed_write_scope: list[str] | None = None,
    metadata: dict | None = None,
) -> TrancheLane:
    return TrancheLane(
        lane_id=lane_id,
        owner_role="critical_path_engineer",
        branch=branch or {},
        allowed_write_scope=allowed_write_scope or [],
        dependencies=dependencies or [],
        metadata=metadata or {},
    )


def _make_manifest(*lanes: TrancheLane) -> SimpleNamespace:
    return SimpleNamespace(manifest_id="m1", lanes=list(lanes))


class _FakeArtifactStore:
    def __init__(self, artifacts: dict[str, TrancheLaneArtifact]) -> None:
        self._artifacts = dict(artifacts)

    def load(self, manifest_id: str, lane_id: str) -> TrancheLaneArtifact | None:
        assert manifest_id == "m1"
        return self._artifacts.get(lane_id)


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


def test_assess_low_risk_fire_and_forget_auto_merges_for_single_lane_tier_1() -> None:
    manifest = _make_manifest(
        _make_lane(
            "lane-a",
            metadata={
                "merge_class": "low_risk",
                "merge_policy": "auto",
                "enforce_cross_model_review": True,
            },
        )
    )
    artifact = _make_artifact(
        metadata={
            "review": {
                "status": "passed",
                "tier": 1,
                "changed_files": ["aragora/live/src/app/page.tsx"],
            }
        }
    )

    result = assess_lane_integration(
        artifact=artifact,
        manifest=manifest,
        checks="checks_passed",
        review_status="passed",
        autonomy_mode="fire_and_forget",
    )

    assert result["merge_class"] == "low_risk"
    assert result["recommendation"] == "merge"
    assert result["executed"] is True
    assert result["low_risk_policy"]["eligible"] is True


def test_assess_low_risk_tier_two_fails_closed_to_needs_human() -> None:
    manifest = _make_manifest(
        _make_lane(
            "lane-a",
            metadata={"merge_class": "low_risk", "merge_policy": "auto"},
        )
    )
    artifact = _make_artifact(
        metadata={
            "review": {
                "status": "passed",
                "tier": 2,
                "changed_files": ["aragora/live/src/app/page.tsx"],
            }
        }
    )

    result = assess_lane_integration(
        artifact=artifact,
        manifest=manifest,
        checks="checks_passed",
        review_status="passed",
        autonomy_mode="fire_and_forget",
    )

    assert result["recommendation"] == "needs_human"
    assert result["executed"] is False
    assert "review tier is 2 instead of 1" in result["rationale"]


def test_assess_low_risk_protected_paths_fail_closed_to_needs_human() -> None:
    manifest = _make_manifest(
        _make_lane(
            "lane-a",
            metadata={"merge_class": "low_risk", "merge_policy": "auto"},
        )
    )
    artifact = _make_artifact(
        metadata={
            "review": {
                "status": "passed",
                "tier": 1,
                "changed_files": [".github/workflows/test.yml"],
            }
        }
    )

    result = assess_lane_integration(
        artifact=artifact,
        manifest=manifest,
        checks="checks_passed",
        review_status="passed",
        autonomy_mode="fire_and_forget",
    )

    assert result["recommendation"] == "needs_human"
    assert result["low_risk_policy"]["protected_paths"] == [".github/workflows/test.yml"]


def test_assess_low_risk_missing_changed_files_fails_closed() -> None:
    manifest = _make_manifest(
        _make_lane(
            "lane-a",
            metadata={"merge_class": "low_risk", "merge_policy": "auto"},
        )
    )
    artifact = _make_artifact(metadata={"review": {"status": "passed", "tier": 1}})

    result = assess_lane_integration(
        artifact=artifact,
        manifest=manifest,
        checks="checks_passed",
        review_status="passed",
        autonomy_mode="fire_and_forget",
    )

    assert result["recommendation"] == "needs_human"
    assert "changed files were not recorded" in result["rationale"]


def test_assess_low_risk_checks_pending_does_not_emit_merge_signal() -> None:
    manifest = _make_manifest(
        _make_lane(
            "lane-a",
            metadata={"merge_class": "low_risk", "merge_policy": "auto"},
        )
    )
    artifact = _make_artifact(
        metadata={
            "review": {
                "status": "passed",
                "tier": 1,
                "changed_files": ["aragora/live/src/app/page.tsx"],
            }
        }
    )

    result = assess_lane_integration(
        artifact=artifact,
        manifest=manifest,
        checks="checks_pending",
        review_status="passed",
        autonomy_mode="fire_and_forget",
    )

    assert result["recommendation"] == "awaiting_checks"
    assert result["executed"] is False
    assert result["low_risk_policy"]["eligible"] is True


def test_assess_low_risk_cross_model_review_disabled_blocks_auto_merge() -> None:
    manifest = _make_manifest(
        _make_lane(
            "lane-a",
            metadata={
                "merge_class": "low_risk",
                "merge_policy": "auto",
                "enforce_cross_model_review": False,
            },
        )
    )
    artifact = _make_artifact(
        metadata={
            "review": {
                "status": "passed",
                "tier": 1,
                "changed_files": ["aragora/live/src/app/page.tsx"],
            }
        }
    )

    result = assess_lane_integration(
        artifact=artifact,
        manifest=manifest,
        checks="checks_passed",
        review_status="passed",
        autonomy_mode="fire_and_forget",
    )

    assert result["recommendation"] == "needs_human"
    assert "cross-model review is disabled" in result["rationale"]


@pytest.mark.parametrize(
    ("review_status", "checks", "expected_phrase"),
    [
        ("changes_requested", "checks_passed", "passing review"),
        ("blocked_nonreviewable", "checks_passed", "passing review"),
        ("passed", "checks_failed", "green required checks"),
    ],
)
def test_assess_low_risk_failed_review_or_checks_becomes_needs_human(
    review_status: str,
    checks: str,
    expected_phrase: str,
) -> None:
    manifest = _make_manifest(
        _make_lane(
            "lane-a",
            metadata={"merge_class": "low_risk", "merge_policy": "auto"},
        )
    )
    artifact = _make_artifact(
        metadata={
            "review": {
                "status": review_status,
                "tier": 1,
                "changed_files": ["aragora/live/src/app/page.tsx"],
            }
        }
    )

    result = assess_lane_integration(
        artifact=artifact,
        manifest=manifest,
        checks=checks,
        review_status=review_status,
        autonomy_mode="fire_and_forget",
    )

    assert result["recommendation"] == "needs_human"
    assert expected_phrase in result["rationale"]


def test_execute_lane_merge_reuses_github_control_and_closes_registry() -> None:
    github = MagicMock()
    github.merge_pr.return_value = SimpleNamespace(
        to_dict=lambda: {
            "merged": True,
            "action": "merge",
            "used_admin": False,
            "detail": "merged",
        }
    )
    registry = MagicMock(spec=PullRequestRegistry)

    result = execute_lane_merge(
        "https://github.com/org/repo/pull/42",
        github=github,
        branch="feat-branch",
        registry=registry,
        required_checks_green=True,
    )

    github.merge_pr.assert_called_once_with(
        "https://github.com/org/repo/pull/42",
        required_checks_green=True,
        allow_admin=False,
    )
    registry.close.assert_called_once_with("feat-branch", outcome="merged")
    assert result["merged"] is True


def test_publish_lane_deliverable_pushes_branch_and_creates_pr() -> None:
    artifact = _make_artifact(
        status="completed",
        metadata={
            "branch": "feat-branch",
            "deliverable": {
                "type": "branch",
                "branch": "feat-branch",
                "commit_shas": ["abc123"],
            },
        },
    )
    github = MagicMock()
    github.find_pr_for_branch.side_effect = [None, None, None]
    github.create_pr_for_branch.return_value = "https://github.com/org/repo/pull/77"
    registry = MagicMock(spec=PullRequestRegistry)
    artifact_store = MagicMock()

    with patch("aragora.swarm.tranche_integrate.subprocess.run") as mock_run:
        mock_run.return_value = SimpleNamespace(returncode=0, stdout="ok\n", stderr="")
        result = publish_lane_deliverable(
            artifact,
            manifest_id="m1",
            github=github,
            registry=registry,
            repo_root=Path("/tmp/repo"),
            target_branch="main",
            artifact_store=artifact_store,
        )

    assert result["published"] is True
    assert result["action"] == "pr_created"
    assert result["pr_url"] == "https://github.com/org/repo/pull/77"
    assert artifact.metadata["deliverable"]["pr_url"] == "https://github.com/org/repo/pull/77"
    assert "https://github.com/org/repo/pull/77" in artifact.urls
    registry.register.assert_called_once_with(
        "feat-branch",
        "https://github.com/org/repo/pull/77",
        creator="tranche-integrate",
    )
    artifact_store.save.assert_called_once()


def test_publish_lane_deliverable_sanitizes_pr_create_exception() -> None:
    artifact = _make_artifact(
        status="completed",
        metadata={
            "branch": "feat-branch",
            "deliverable": {
                "type": "branch",
                "branch": "feat-branch",
                "commit_shas": ["abc123"],
            },
        },
    )
    github = MagicMock()
    github.find_pr_for_branch.side_effect = [None, None, None]
    github.create_pr_for_branch.side_effect = RuntimeError(
        "token=secret gh pr create failed from /private/tmp/path"
    )
    registry = MagicMock(spec=PullRequestRegistry)
    artifact_store = MagicMock()

    with patch("aragora.swarm.tranche_integrate.subprocess.run") as mock_run:
        mock_run.return_value = SimpleNamespace(returncode=0, stdout="ok\n", stderr="")
        result = publish_lane_deliverable(
            artifact,
            manifest_id="m1",
            github=github,
            registry=registry,
            repo_root=Path("/tmp/repo"),
            target_branch="main",
            artifact_store=artifact_store,
        )

    assert result == {
        "published": False,
        "action": "pr_create_failed",
        "branch": "feat-branch",
        "pr_url": None,
        "detail": "gh pr create failed. Check logs for detail.",
    }
    assert artifact.metadata["publish"]["detail"] == "gh pr create failed. Check logs for detail."
    artifact_store.save.assert_called_once()


@pytest.mark.asyncio
async def test_integrate_lane_returns_needs_human_when_controller_publish_fails() -> None:
    manifest = _make_manifest(_make_lane("lane-a"))
    artifact = _make_artifact(
        metadata={
            "branch": "feat-a",
            "review": {"status": "passed"},
            "merge_policy": "auto",
            "receipt_id": "receipt-a",
            "lease_id": "lease-a",
            "deliverable": {
                "type": "branch",
                "branch": "feat-a",
                "commit_shas": ["abc123"],
            },
        }
    )
    github = MagicMock()
    github.find_pr_for_branch.return_value = None

    with patch("aragora.swarm.tranche_integrate.subprocess.run") as mock_run:
        mock_run.return_value = SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="Could not resolve host: github.com",
        )
        result = await integrate_lane(
            manifest=manifest,
            artifact=artifact,
            approve=False,
            repo_root=Path("/tmp/repo"),
            github=github,
            registry=MagicMock(spec=PullRequestRegistry),
            artifact_store=MagicMock(),
            target_branch="main",
        )

    assert result["pr_url"] is None
    assert result["publish_result"]["action"] == "push_failed"
    assert result["recommendation"] == "needs_human"
    assert result["rationale"] == "git push failed. Check logs for detail."


@pytest.mark.asyncio
async def test_integrate_lane_publishes_branch_before_gate_assessment() -> None:
    manifest = _make_manifest(_make_lane("lane-a"))
    artifact = _make_artifact(
        metadata={
            "branch": "feat-a",
            "review": {"status": "passed"},
            "merge_policy": "auto",
            "deliverable": {
                "type": "branch",
                "branch": "feat-a",
                "commit_shas": ["abc123"],
            },
        }
    )
    github = MagicMock()
    github.find_pr_for_branch.side_effect = [None, None, None]
    github.create_pr_for_branch.return_value = "https://github.com/org/repo/pull/42"
    github.fetch_gate_snapshot.return_value = SimpleNamespace(
        required_checks=[{"name": "lint", "conclusion": "SUCCESS", "required": True}],
        advisory_checks=[],
        required_checks_green=True,
        to_dict=lambda: {"required_checks_green": True},
        state="OPEN",
        base_branch="main",
        merge_state_status="CLEAN",
    )

    with patch("aragora.swarm.tranche_integrate.subprocess.run") as mock_run:
        mock_run.return_value = SimpleNamespace(returncode=0, stdout="ok\n", stderr="")
        result = await integrate_lane(
            manifest=manifest,
            artifact=artifact,
            approve=False,
            repo_root=Path("/tmp/repo"),
            github=github,
            registry=MagicMock(spec=PullRequestRegistry),
            artifact_store=MagicMock(),
            target_branch="main",
        )

    assert result["pr_url"] == "https://github.com/org/repo/pull/42"
    assert result["publish_result"]["action"] == "pr_created"
    assert result["recommendation"] == "merge"


def test_cascade_retargets_open_downstream_prs_and_ignores_reference_dependencies() -> None:
    manifest = _make_manifest(
        _make_lane("lane-a"),
        _make_lane("lane-b", dependencies=["lane-a"]),
        _make_lane("lane-c", dependencies=["issue_123"]),
    )
    artifact_store = _FakeArtifactStore(
        {
            "lane-b": _make_artifact(
                lane_id="lane-b",
                metadata={
                    "branch": "feat-b",
                    "deliverable": {"pr_url": "https://github.com/org/repo/pull/99"},
                },
            )
        }
    )
    github = MagicMock()
    github.fetch_gate_snapshot.return_value = SimpleNamespace(
        state="OPEN",
        base_branch="stack-base",
        merge_state_status="CLEAN",
    )
    github.retarget_pr_base.return_value = {
        "retargeted": True,
        "pr_url": "https://github.com/org/repo/pull/99",
        "base_branch": "main",
        "action": "retarget",
    }
    registry = MagicMock(spec=PullRequestRegistry)
    run_state = TrancheRunState(
        manifest_id="m1",
        status="integrating",
        autonomy_mode="adaptive",
        lane_states={
            "lane-b": LaneRunState(lane_id="lane-b", status=LANE_STATUS_WAITING_FOR_MERGE)
        },
    )

    report = cascade_after_merge(
        manifest,
        "lane-a",
        artifact_store=artifact_store,
        github=github,
        registry=registry,
        base_branch="main",
        run_state=run_state,
    )

    assert [item["lane_id"] for item in report["downstream"]] == ["lane-b"]
    assert report["downstream"][0]["action"] == "retargeted"
    assert report["clean"] is True
    assert report["needs_human"] is False
    assert run_state.lane_states["lane-b"].status == LANE_STATUS_WAITING_FOR_MERGE
    github.retarget_pr_base.assert_called_once_with(
        "https://github.com/org/repo/pull/99",
        "main",
    )


def test_cascade_flags_closed_downstream_pr_for_restack_and_updates_run_state() -> None:
    manifest = _make_manifest(
        _make_lane("lane-a"),
        _make_lane("lane-b", dependencies=["lane-a"]),
    )
    artifact_store = _FakeArtifactStore(
        {
            "lane-b": _make_artifact(
                lane_id="lane-b",
                metadata={
                    "branch": "feat-b",
                    "deliverable": {"pr_url": "https://github.com/org/repo/pull/99"},
                },
            )
        }
    )
    github = MagicMock()
    github.fetch_gate_snapshot.return_value = SimpleNamespace(
        state="CLOSED",
        base_branch="stack-base",
        merge_state_status="CLEAN",
    )
    registry = MagicMock(spec=PullRequestRegistry)
    run_state = TrancheRunState(
        manifest_id="m1",
        status="integrating",
        autonomy_mode="adaptive",
        lane_states={"lane-b": LaneRunState(lane_id="lane-b", status="waiting_for_merge")},
    )

    report = cascade_after_merge(
        manifest,
        "lane-a",
        artifact_store=artifact_store,
        github=github,
        registry=registry,
        base_branch="main",
        run_state=run_state,
    )

    assert report["downstream"][0]["action"] == "needs_restack"
    assert report["needs_human"] is True
    assert run_state.lane_states["lane-b"].status == LANE_STATUS_NEEDS_HUMAN
    assert run_state.status == "needs_human"


def test_cascade_flags_conflicting_prs_for_restack() -> None:
    manifest = _make_manifest(
        _make_lane("lane-a"),
        _make_lane("lane-b", dependencies=["lane-a"]),
    )
    artifact_store = _FakeArtifactStore(
        {
            "lane-b": _make_artifact(
                lane_id="lane-b",
                metadata={
                    "branch": "feat-b",
                    "deliverable": {"pr_url": "https://github.com/org/repo/pull/99"},
                },
            )
        }
    )
    github = MagicMock()
    github.fetch_gate_snapshot.return_value = SimpleNamespace(
        state="OPEN",
        base_branch="main",
        merge_state_status="DIRTY",
    )

    report = cascade_after_merge(
        manifest,
        "lane-a",
        artifact_store=artifact_store,
        github=github,
        registry=MagicMock(spec=PullRequestRegistry),
        base_branch="main",
    )

    assert report["downstream"][0]["action"] == "needs_restack"
    assert report["needs_human"] is True


def test_cascade_flags_missing_pr_without_recreating() -> None:
    manifest = _make_manifest(
        _make_lane("lane-a"),
        _make_lane("lane-b", dependencies=["lane-a"]),
    )
    artifact_store = _FakeArtifactStore({"lane-b": _make_artifact(lane_id="lane-b")})
    github = MagicMock()

    report = cascade_after_merge(
        manifest,
        "lane-a",
        artifact_store=artifact_store,
        github=github,
        registry=MagicMock(spec=PullRequestRegistry),
        base_branch="main",
    )

    assert report["downstream"][0]["action"] == "missing_pr"
    github.fetch_gate_snapshot.assert_not_called()
    github.retarget_pr_base.assert_not_called()


@pytest.mark.asyncio
async def test_integrate_lane_returns_merge_and_cascade_payload() -> None:
    manifest = _make_manifest(
        _make_lane("lane-a"),
        _make_lane("lane-b", dependencies=["lane-a"]),
    )
    artifact = _make_artifact(
        metadata={
            "branch": "feat-a",
            "review": {"status": "passed"},
            "merge_policy": "auto",
            "receipt_id": "receipt-a",
            "lease_id": "lease-a",
            "deliverable": {"pr_url": "https://github.com/org/repo/pull/42"},
        }
    )
    downstream_artifact = _make_artifact(
        lane_id="lane-b",
        status="waiting_for_merge",
        metadata={
            "branch": "feat-b",
            "deliverable": {"pr_url": "https://github.com/org/repo/pull/99"},
        },
    )
    artifact_store = _FakeArtifactStore({"lane-b": downstream_artifact})
    github = MagicMock()
    github.fetch_gate_snapshot.side_effect = [
        SimpleNamespace(
            required_checks=[{"name": "lint", "conclusion": "SUCCESS", "required": True}],
            advisory_checks=[],
            required_checks_green=True,
            to_dict=lambda: {"required_checks_green": True},
            state="OPEN",
            base_branch="main",
            merge_state_status="CLEAN",
        ),
        SimpleNamespace(
            state="OPEN",
            base_branch="stack-base",
            merge_state_status="CLEAN",
        ),
    ]
    github.merge_pr.return_value = SimpleNamespace(
        to_dict=lambda: {
            "merged": True,
            "action": "merge",
            "used_admin": False,
            "detail": "merged",
        }
    )
    github.retarget_pr_base.return_value = {
        "retargeted": True,
        "pr_url": "https://github.com/org/repo/pull/99",
        "base_branch": "main",
        "action": "retarget",
        "detail": "retargeted",
    }
    registry = MagicMock(spec=PullRequestRegistry)
    store = MagicMock()
    run_state = TrancheRunState(
        manifest_id="m1",
        status="integrating",
        autonomy_mode="adaptive",
        lane_states={
            "lane-a": LaneRunState(lane_id="lane-a", status="review_passed"),
            "lane-b": LaneRunState(lane_id="lane-b", status="waiting_for_merge"),
        },
    )

    result = await integrate_lane(
        manifest=manifest,
        artifact=artifact,
        approve=True,
        github=github,
        registry=registry,
        store=store,
        artifact_store=artifact_store,
        target_branch="main",
        run_state=run_state,
    )

    assert result["recommendation"] == "merge"
    assert result["executed"] is True
    assert result["merge_result"]["merged"] is True
    assert result["cascade_report"]["downstream"][0]["action"] == "retargeted"
    store.record_integration_decision.assert_called_once()
    registry.close.assert_called_once_with("feat-a", outcome="merged")
    assert run_state.lane_states["lane-a"].status == "completed"
    assert run_state.lane_states["lane-b"].status == "waiting_for_merge"


@pytest.mark.asyncio
async def test_integrate_lane_uses_manifest_low_risk_metadata_for_auto_merge() -> None:
    manifest = _make_manifest(
        _make_lane(
            "lane-a",
            metadata={
                "merge_class": "low_risk",
                "merge_policy": "auto",
                "enforce_cross_model_review": True,
            },
        )
    )
    artifact = _make_artifact(
        metadata={
            "branch": "feat-a",
            "review": {
                "status": "passed",
                "tier": 1,
                "changed_files": ["aragora/live/src/app/page.tsx"],
            },
            "receipt_id": "receipt-a",
            "lease_id": "lease-a",
            "deliverable": {"pr_url": "https://github.com/org/repo/pull/42"},
        }
    )
    github = MagicMock()
    github.fetch_gate_snapshot.return_value = SimpleNamespace(
        required_checks=[{"name": "lint", "conclusion": "SUCCESS", "required": True}],
        advisory_checks=[],
        required_checks_green=True,
        to_dict=lambda: {"required_checks_green": True},
        state="OPEN",
        base_branch="main",
        merge_state_status="CLEAN",
    )
    github.merge_pr.return_value = SimpleNamespace(
        to_dict=lambda: {
            "merged": True,
            "action": "merge",
            "used_admin": False,
            "detail": "merged",
        }
    )

    result = await integrate_lane(
        manifest=manifest,
        artifact=artifact,
        approve=True,
        github=github,
        registry=MagicMock(spec=PullRequestRegistry),
        store=MagicMock(),
        artifact_store=_FakeArtifactStore({}),
        target_branch="main",
        autonomy_mode="fire_and_forget",
    )

    assert result["merge_class"] == "low_risk"
    assert result["recommendation"] == "merge"
    assert result["executed"] is True
    assert result["merge_result"]["merged"] is True


@pytest.mark.asyncio
async def test_integrate_lane_flat_default_skips_cascade_without_downstream_lane_deps() -> None:
    manifest = _make_manifest(
        _make_lane("lane-a"),
        _make_lane("lane-b", dependencies=["issue_123"]),
    )
    artifact = _make_artifact(
        metadata={
            "branch": "feat-a",
            "review": {"status": "passed"},
            "merge_policy": "auto",
            "receipt_id": "receipt-a",
            "lease_id": "lease-a",
            "deliverable": {"pr_url": "https://github.com/org/repo/pull/42"},
        }
    )
    github = MagicMock()
    github.fetch_gate_snapshot.return_value = SimpleNamespace(
        required_checks=[{"name": "lint", "conclusion": "SUCCESS", "required": True}],
        advisory_checks=[],
        required_checks_green=True,
        to_dict=lambda: {"required_checks_green": True},
        state="OPEN",
        base_branch="main",
        merge_state_status="CLEAN",
    )
    github.merge_pr.return_value = SimpleNamespace(
        to_dict=lambda: {
            "merged": True,
            "action": "merge",
            "used_admin": False,
            "detail": "merged",
        }
    )

    result = await integrate_lane(
        manifest=manifest,
        artifact=artifact,
        approve=True,
        github=github,
        registry=MagicMock(spec=PullRequestRegistry),
        store=MagicMock(),
        artifact_store=_FakeArtifactStore({}),
        target_branch="main",
    )

    assert result["cascade_report"] == {
        "merged_lane_id": "lane-a",
        "downstream": [],
        "clean": True,
        "needs_human": False,
    }
