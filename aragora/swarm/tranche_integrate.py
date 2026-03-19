from __future__ import annotations

from pathlib import Path
from typing import Any

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
