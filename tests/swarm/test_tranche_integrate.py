from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from aragora.swarm.pr_registry import PullRequestRegistry
from aragora.swarm.tranche_integrate import (
    classify_check_results,
    discover_lane_pr,
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
