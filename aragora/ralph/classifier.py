"""Blocker classification for campaign supervision.

Maps campaign terminal states and project outcomes to a narrow taxonomy of
known, auto-fixable blocker kinds vs escalation-required unknowns.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class BlockerKind(str, Enum):
    """Taxonomy of campaign blockers.

    DETERMINISTIC kinds have known repair templates.
    ESCALATE kinds require human judgment.
    """

    # --- deterministic, auto-fixable ---
    REVIEWER_MISSING_DIFF = "reviewer_missing_diff_context"
    SCOPE_FALSE_POSITIVE = "scope_false_positive"
    WORKER_CLEAN_EXIT_NO_EFFECT = "worker_clean_exit_no_effect"
    MANIFEST_IDENTIFIER_COLLISION = "manifest_identifier_collision"
    RUNTIME_TIMEOUT_CONFIG = "campaign_runtime_timeout_config"
    RECEIPT_EMISSION_GAP = "receipt_emission_gap"

    # --- escalate ---
    BUDGET_EXHAUSTION = "budget_exhaustion"
    INFRA_FAILURE = "infra_failure"
    UNKNOWN = "unknown"

    @property
    def is_deterministic(self) -> bool:
        return self in _DETERMINISTIC_KINDS


_DETERMINISTIC_KINDS = frozenset(
    {
        BlockerKind.REVIEWER_MISSING_DIFF,
        BlockerKind.SCOPE_FALSE_POSITIVE,
        BlockerKind.WORKER_CLEAN_EXIT_NO_EFFECT,
        BlockerKind.MANIFEST_IDENTIFIER_COLLISION,
        BlockerKind.RUNTIME_TIMEOUT_CONFIG,
        BlockerKind.RECEIPT_EMISSION_GAP,
    }
)


def classify_blocker(
    *,
    stop_reason: str,
    manifest_dict: dict[str, Any],
) -> BlockerKind | None:
    """Classify a campaign blocker from its stop reason and manifest state.

    Returns ``None`` if the campaign is not blocked (still running or complete).
    """
    if stop_reason in ("still_running", "campaign_complete"):
        return None

    if stop_reason == "budget_exhausted":
        return BlockerKind.BUDGET_EXHAUSTION

    if stop_reason == "time_limit_exceeded":
        return _classify_time_limit(manifest_dict)

    if stop_reason == "campaign_blocked":
        return _classify_campaign_blocked(manifest_dict)

    return BlockerKind.UNKNOWN


def _classify_time_limit(manifest_dict: dict[str, Any]) -> BlockerKind:
    """Distinguish runtime config timeout from real time issues."""
    exec_state = manifest_dict.get("execution_state", {})
    completed = exec_state.get("completed_projects", [])

    # If some projects completed, the timeout is likely a config issue
    # (campaign needs more time, not a product bug).
    if completed:
        return BlockerKind.RUNTIME_TIMEOUT_CONFIG

    # No progress at all — likely an infra issue.
    return BlockerKind.INFRA_FAILURE


def _classify_campaign_blocked(manifest_dict: dict[str, Any]) -> BlockerKind:
    """Inspect project-level outcomes to classify the blocker."""
    projects = manifest_dict.get("projects", [])

    blocked_projects = [p for p in projects if p.get("status") in ("blocked", "failed", "skipped")]
    if not blocked_projects:
        return BlockerKind.UNKNOWN

    # Collect last_run_outcomes across blocked/failed projects.
    outcomes: list[str] = []
    review_statuses: list[str] = []
    for proj in blocked_projects:
        outcome = proj.get("last_run_outcome")
        if outcome:
            outcomes.append(outcome)
        review = proj.get("review", {})
        if isinstance(review, dict):
            rstatus = review.get("status", "")
            if rstatus:
                review_statuses.append(rstatus)
            findings = review.get("findings", [])
            for finding in findings:
                finding_lower = str(finding).lower()
                if "scope" in finding_lower and (
                    "violation" in finding_lower or "outside" in finding_lower
                ):
                    return BlockerKind.SCOPE_FALSE_POSITIVE

    # Check for reviewer false rejection pattern: deliverable_created but
    # review blocked/changes_requested.
    has_deliverable = "deliverable_created" in outcomes
    has_review_rejection = any(
        s in ("changes_requested", "blocked_nonreviewable") for s in review_statuses
    )
    if has_deliverable and has_review_rejection:
        return BlockerKind.REVIEWER_MISSING_DIFF

    # Check for repeated clean_exit_no_deliverable on docs-only tasks.
    clean_exit_count = outcomes.count("clean_exit_no_deliverable")
    if clean_exit_count >= 2:
        return BlockerKind.WORKER_CLEAN_EXIT_NO_EFFECT

    # Check for receipt gaps: terminal projects without receipt_id.
    terminal_statuses = frozenset({"completed", "failed", "blocked", "skipped"})
    for proj in projects:
        if proj.get("status") in terminal_statuses and not proj.get("receipt_id"):
            return BlockerKind.RECEIPT_EMISSION_GAP

    # Check for manifest identifier collisions (file_scope_hints pointing to
    # already-existing paths that conflict).
    seen_hints: set[str] = set()
    for proj in projects:
        for hint in proj.get("file_scope_hints", []):
            if hint in seen_hints:
                return BlockerKind.MANIFEST_IDENTIFIER_COLLISION
            seen_hints.add(hint)

    if clean_exit_count >= 1:
        return BlockerKind.WORKER_CLEAN_EXIT_NO_EFFECT

    return BlockerKind.UNKNOWN
