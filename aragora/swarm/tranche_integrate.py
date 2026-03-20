from __future__ import annotations

from pathlib import Path
from typing import Any

from aragora.nomic.dev_coordination import DevCoordinationStore, IntegrationDecisionType
from aragora.ralph.github_control import (
    GitHubCheck,
    GitHubControl,
    _check_is_green,
    _partition_checks,
)
from aragora.swarm.pr_registry import PullRequestRegistry


def discover_lane_pr(
    artifact: Any,
    *,
    github: GitHubControl | Any | None = None,
    repo_root: Path | None = None,
) -> str | None:
    metadata = getattr(artifact, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    deliverable = metadata.get("deliverable", {})
    if not isinstance(deliverable, dict):
        deliverable = {}

    candidates = [
        deliverable.get("pr_url"),
        metadata.get("pr_url"),
        *list(getattr(artifact, "urls", []) or []),
    ]
    for candidate in candidates:
        value = str(candidate or "").strip()
        if _looks_like_pr_url(value):
            return value

    branch = str(
        deliverable.get("branch") or metadata.get("branch") or getattr(artifact, "branch", "") or ""
    ).strip()
    if not branch:
        return None

    control = github or GitHubControl(repo_root=(repo_root or Path.cwd()).resolve())
    return _optional_text(control.find_pr_for_branch(branch))


def register_pr(
    pr_url: str | None,
    branch: str | None,
    registry: PullRequestRegistry,
    *,
    creator: str = "tranche-integrate",
) -> dict[str, Any] | None:
    normalized_pr_url = str(pr_url or "").strip()
    normalized_branch = str(branch or "").strip()
    if not normalized_pr_url or not normalized_branch:
        return None

    existing = registry.get(normalized_branch)
    if isinstance(existing, dict) and str(existing.get("pr_url", "")).strip() == normalized_pr_url:
        return existing

    entry = registry.register(normalized_branch, normalized_pr_url, creator=creator)
    if hasattr(entry, "__dict__"):
        return dict(entry.__dict__)
    if isinstance(entry, dict):
        return dict(entry)
    return {
        "branch": normalized_branch,
        "pr_url": normalized_pr_url,
        "creator": creator,
    }


def classify_check_results(checks: list[GitHubCheck | dict[str, Any]]) -> str:
    normalized = [_normalize_check(item) for item in checks]
    required_contexts = {item.name for item in normalized if item.required}
    required, _, _required_green = _partition_checks(
        normalized,
        required_contexts=required_contexts,
        required_known=bool(required_contexts),
    )

    if any(_check_is_failed(item) for item in required):
        return "checks_failed"
    if any(not _check_is_green(item) for item in required):
        return "checks_pending"
    return "checks_passed"


def assess_lane_integration(
    *,
    artifact: Any,
    checks: str,
    review_status: str,
    merge_policy: str = "confirm",
    autonomy_mode: str = "adaptive",
    approve: bool = False,
) -> dict[str, Any]:
    normalized_checks = str(checks or "").strip().lower()
    normalized_review = str(review_status or "").strip().lower()
    normalized_policy = str(merge_policy or "confirm").strip().lower()
    normalized_autonomy = str(autonomy_mode or "adaptive").strip().lower()

    recommendation = "request_changes"
    executed = False
    rationale = "Review requested changes."

    if normalized_review not in {"passed", "approved"}:
        recommendation = "request_changes"
        rationale = "Review must pass before integration."
    elif normalized_checks == "checks_failed":
        recommendation = "request_changes"
        rationale = "Required checks are failing."
    elif normalized_checks != "checks_passed":
        recommendation = "awaiting_confirmation"
        rationale = "Required checks are still pending."
    else:
        recommendation = "merge"
        rationale = "Review passed and required checks are green."
        if normalized_policy == "manual":
            rationale = "Manual merge policy requires a human merge."
        elif normalized_autonomy == "fire_and_forget" and normalized_policy == "auto":
            executed = True
            rationale = "Fire-and-forget mode auto-executes merge for auto merge policy."
        elif normalized_autonomy == "checkpoint" and not approve:
            rationale = "Checkpoint mode requires explicit approval before merge."
        elif approve:
            executed = normalized_policy in {"auto", "confirm"}
            rationale = "Explicit approval granted for merge execution."

    return {
        "recommendation": recommendation,
        "executed": executed,
        "checks": normalized_checks,
        "review_status": normalized_review,
        "merge_policy": normalized_policy,
        "autonomy_mode": normalized_autonomy,
        "rationale": rationale,
        "lane_id": str(getattr(artifact, "lane_id", "") or "").strip() or None,
    }


async def record_lane_integration(
    *,
    artifact: Any,
    decision: str,
    rationale: str,
    decided_by: str,
    store: DevCoordinationStore,
    target_branch: str = "main",
) -> dict[str, Any]:
    metadata = getattr(artifact, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    receipt_id = str(metadata.get("receipt_id", "") or "").strip()
    lease_id = str(metadata.get("lease_id", "") or "").strip()
    if not receipt_id:
        raise ValueError("Lane artifact is missing receipt_id metadata.")

    decision_enum = IntegrationDecisionType(str(decision or "").strip().lower())
    result = store.record_integration_decision(
        receipt_id=receipt_id,
        lease_id=lease_id or None,
        decision=decision_enum,
        decided_by=str(decided_by or "").strip() or "tranche-integrate",
        rationale=str(rationale or "").strip() or "Integration decision recorded.",
        target_branch=str(target_branch or "main").strip() or "main",
    )
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if isinstance(result, dict):
        return dict(result)
    return {
        "receipt_id": receipt_id,
        "lease_id": lease_id or None,
        "decision": decision_enum.value,
        "target_branch": str(target_branch or "main").strip() or "main",
    }


def _normalize_check(check: GitHubCheck | dict[str, Any]) -> GitHubCheck:
    if isinstance(check, GitHubCheck):
        return check
    if not isinstance(check, dict):
        raise TypeError("check must be a dict or GitHubCheck")
    return GitHubCheck(
        name=str(check.get("name", "")).strip(),
        status=str(check.get("status", "") or "").strip().upper()
        or _status_from_conclusion(check.get("conclusion")),
        conclusion=_optional_text(check.get("conclusion"), upper=True),
        required=bool(check.get("required", False)),
        details_url=_optional_text(check.get("details_url")),
    )


def _status_from_conclusion(conclusion: Any) -> str:
    normalized = _optional_text(conclusion, upper=True)
    if not normalized:
        return "UNKNOWN"
    if normalized in {"SUCCESS", "NEUTRAL", "SKIPPED"}:
        return "COMPLETED"
    if normalized in {"FAILURE", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED"}:
        return "COMPLETED"
    return normalized


def _check_is_failed(check: GitHubCheck) -> bool:
    conclusion = str(check.conclusion or "").strip().upper()
    if conclusion in {"FAILURE", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED"}:
        return True
    return check.status in {"FAILURE", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED"}


def _looks_like_pr_url(value: str) -> bool:
    return value.startswith("https://github.com/") and "/pull/" in value


def _optional_text(value: Any, *, upper: bool = False) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text.upper() if upper else text
