"""Concrete GitHub-event resolution adapter for AGT-04 synthetic prediction markets.

Converts synthetic GitHub event payloads into claim resolutions.  The
adapter operates on *pre-fetched* event data; it never makes live API
calls, so tests run offline.

Feature flag: ``ARAGORA_PREDICTION_MARKETS_ENABLED`` (env var, default OFF).

Advances: issue #6065 (AGT-04), sub-deliverable 2 — GitHub event resolution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from aragora.prediction.stakeable_claim import (
    QuestionType,
    ResolutionStatus,
    StakeableClaim,
)

_ENV_FLAG = "ARAGORA_PREDICTION_MARKETS_ENABLED"


def _flag_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "").lower() in {"1", "true", "yes", "on"}


def _require_enabled() -> None:
    if not _flag_enabled():
        raise RuntimeError(f"Prediction markets are disabled. Set {_ENV_FLAG}=1 to enable.")


@dataclass(frozen=True)
class GitHubEventPayload:
    """Minimal representation of a GitHub webhook event payload.

    Contains only the fields needed to resolve a StakeableClaim.  The
    full webhook payload is not stored; callers extract the relevant
    fields before constructing this object.

    Attributes:
        event_type: One of ``"pull_request"``, ``"issues"``, ``"check_run"``,
            ``"workflow_run"``.
        action: Event action string (e.g. ``"closed"``, ``"completed"``).
        target_ref: ``owner/repo#number`` or ``owner/repo@branch`` — must
            match the claim's ``target_ref`` for resolution to proceed.
        merged: For ``pull_request`` events: whether the PR was merged.
        conclusion: For ``check_run``/``workflow_run`` events: the final
            conclusion (``"success"``, ``"failure"``, ``"cancelled"``, …).
        raw: Arbitrary additional fields preserved for traceability.
    """

    event_type: str
    action: str
    target_ref: str
    merged: bool = False
    conclusion: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolutionResult:
    """Outcome of resolving a StakeableClaim against a GitHubEventPayload."""

    claim_id: str
    resolved: bool
    resolution_value: bool
    evidence: str


class GitHubEventResolver:
    """Resolves StakeableClaim instances from pre-fetched GitHub event payloads.

    Concrete implementation of the interface sketched in
    :class:`~aragora.prediction.stakeable_claim.GithubResolutionAdapterStub`.
    All resolution logic is deterministic and does not call the GitHub API.

    Usage::

        from aragora.prediction import GitHubEventResolver, GitHubEventPayload

        resolver = GitHubEventResolver()
        result = resolver.resolve_from_event(claim, payload)
        if result.resolved:
            store.resolve(claim.claim_id, result.resolution_value, result.evidence)
    """

    _EVENT_TYPES: dict[QuestionType, frozenset[str]] = {
        QuestionType.PR_MERGE: frozenset({"pull_request"}),
        QuestionType.ISSUE_CLOSE: frozenset({"issues"}),
        QuestionType.CI_PASS: frozenset({"check_run", "workflow_run"}),
    }

    def can_resolve(self, claim: StakeableClaim, event: GitHubEventPayload) -> bool:
        """Return True if *event* could update *claim*'s resolution state."""
        if claim.question_type not in self._EVENT_TYPES:
            return False
        return (
            event.event_type in self._EVENT_TYPES[claim.question_type]
            and event.target_ref == claim.target_ref
        )

    def resolve_from_event(
        self, claim: StakeableClaim, event: GitHubEventPayload
    ) -> ResolutionResult:
        """Attempt to resolve *claim* from *event*.

        Returns a :class:`ResolutionResult`.  When the event is not
        applicable, ``resolved=False`` is returned instead of raising.
        """
        _require_enabled()

        if claim.resolution_status != ResolutionStatus.OPEN:
            return ResolutionResult(
                claim_id=claim.claim_id,
                resolved=False,
                resolution_value=False,
                evidence=f"Claim already {claim.resolution_status.value}; skipping.",
            )

        if not self.can_resolve(claim, event):
            return ResolutionResult(
                claim_id=claim.claim_id,
                resolved=False,
                resolution_value=False,
                evidence=(
                    f"Event {event.event_type!r}/{event.action!r} does not match "
                    f"claim type {claim.question_type.value!r} or target {claim.target_ref!r}."
                ),
            )

        if claim.question_type == QuestionType.PR_MERGE:
            return self._resolve_pr_merge(claim, event)
        if claim.question_type == QuestionType.ISSUE_CLOSE:
            return self._resolve_issue_close(claim, event)
        if claim.question_type == QuestionType.CI_PASS:
            return self._resolve_ci_pass(claim, event)

        return ResolutionResult(
            claim_id=claim.claim_id,
            resolved=False,
            resolution_value=False,
            evidence=f"Unsupported question type: {claim.question_type.value!r}.",
        )

    # ------------------------------------------------------------------
    # Per-type resolvers
    # ------------------------------------------------------------------

    def _resolve_pr_merge(
        self, claim: StakeableClaim, event: GitHubEventPayload
    ) -> ResolutionResult:
        if event.action != "closed":
            return ResolutionResult(
                claim_id=claim.claim_id,
                resolved=False,
                resolution_value=False,
                evidence=f"pull_request action {event.action!r} is not terminal; waiting.",
            )
        value = event.merged
        evidence = (
            f"PR {claim.target_ref} {'merged' if value else 'closed without merge'} "
            f"(action={event.action!r}, merged={event.merged})."
        )
        return ResolutionResult(
            claim_id=claim.claim_id,
            resolved=True,
            resolution_value=value,
            evidence=evidence,
        )

    def _resolve_issue_close(
        self, claim: StakeableClaim, event: GitHubEventPayload
    ) -> ResolutionResult:
        if event.action != "closed":
            return ResolutionResult(
                claim_id=claim.claim_id,
                resolved=False,
                resolution_value=False,
                evidence=f"issues action {event.action!r} is not terminal; waiting.",
            )
        evidence = f"Issue {claim.target_ref} closed (action={event.action!r})."
        return ResolutionResult(
            claim_id=claim.claim_id,
            resolved=True,
            resolution_value=True,
            evidence=evidence,
        )

    def _resolve_ci_pass(
        self, claim: StakeableClaim, event: GitHubEventPayload
    ) -> ResolutionResult:
        if event.action != "completed":
            return ResolutionResult(
                claim_id=claim.claim_id,
                resolved=False,
                resolution_value=False,
                evidence=(
                    f"check_run/workflow_run action {event.action!r} is not terminal; waiting."
                ),
            )
        value = event.conclusion == "success"
        evidence = (
            f"CI {event.event_type} for {claim.target_ref} completed with "
            f"conclusion={event.conclusion!r}; {'pass' if value else 'fail'}."
        )
        return ResolutionResult(
            claim_id=claim.claim_id,
            resolved=True,
            resolution_value=value,
            evidence=evidence,
        )
