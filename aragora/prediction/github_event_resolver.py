"""GitHubEventResolver — concrete event-to-claim resolution adapter.

Flag-gated behind ARAGORA_PREDICTION_MARKETS_ENABLED (default OFF).
No live API calls; operates on pre-fetched event payloads.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from aragora.prediction.stakeable_claim import (
    ClaimStore,
    ClaimType,
    ResolutionStatus,
    StakeableClaim,
)

log = logging.getLogger(__name__)

_FLAG = "ARAGORA_PREDICTION_MARKETS_ENABLED"


def _flag_enabled() -> bool:
    return os.environ.get(_FLAG, "").strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class GitHubEventPayload:
    """Normalised representation of a GitHub webhook event.

    ``event_type`` — the GitHub event name, e.g. ``pull_request``, ``issues``,
    ``check_run``, ``workflow_run``.

    ``action`` — the event action, e.g. ``closed``, ``completed``.

    ``target_ref`` — canonical claim target, typically ``owner/repo#number`` or
    a branch/SHA, matching the ``target_ref`` on the :class:`StakeableClaim`.

    ``occurred_at`` — the event's own timestamp (``updated_at`` / ``completed_at``
    from the GitHub payload, **not** the time the webhook was received).

    ``merged`` — True only for pull_request events where the PR was merged.

    ``conclusion`` — check_run / workflow_run conclusion string, e.g. ``success``,
    ``failure``.

    ``raw`` — original payload dict for fields not promoted to first-class attrs.
    CI_PASS events resolve only when ``raw["aggregate"] is True``, which means
    the caller has already reduced all required checks for ``target_ref`` to one
    verdict.
    """

    event_type: str
    action: str
    target_ref: str
    occurred_at: datetime
    merged: bool = False
    conclusion: Optional[str] = None
    raw: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw", self.raw if self.raw is not None else {})


@dataclass(frozen=True)
class ResolutionResult:
    """Outcome of a single resolution attempt."""

    claim_id: str
    resolved: bool
    resolution_value: bool
    evidence: str


class GitHubEventResolver:
    """Resolves :class:`StakeableClaim` objects from GitHub webhook events.

    Supports three claim types:
    * ``PR_MERGE``   — resolved when a pull_request/closed event has ``merged=True``.
    * ``ISSUE_CLOSE`` — resolved when an issues/closed event fires.
    * ``CI_PASS``    — resolved when the caller supplies an aggregated check_run/
      workflow_run event with ``raw["aggregate"] is True``.

    All resolution is based on **event-time** (``event.occurred_at``), not
    wall-clock time at processing.  A late-delivered or replayed event that
    occurred within the claim window resolves the claim normally.  Events that
    arrived after the grace window (default 24 h past expiry) are logged and
    dropped without mutating the claim.
    """

    def __init__(self, store: Optional[ClaimStore] = None) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_resolve(self, claim: StakeableClaim) -> bool:
        """Return True if this resolver handles *claim*'s question type."""
        return claim.question_type in (
            ClaimType.PR_MERGE,
            ClaimType.ISSUE_CLOSE,
            ClaimType.CI_PASS,
        )

    def resolve_from_event(
        self,
        claim: StakeableClaim,
        event: GitHubEventPayload,
        *,
        grace_hours: float = 24.0,
    ) -> ResolutionResult:
        """Attempt to resolve *claim* from *event*.

        The resolver checks ``event.occurred_at <= claim.expiry`` (occurrence
        model).  Late-delivered events that occurred within the claim window
        still resolve the claim.  Events whose ``occurred_at`` is after
        ``claim.expiry + grace_hours`` are silently voided and a side-output
        log record is emitted at WARNING level.

        Parameters
        ----------
        claim:
            The :class:`StakeableClaim` to resolve.
        event:
            Normalised GitHub event payload.
        grace_hours:
            Processing-time grace window (default 24 h).  Events whose
            ``occurred_at`` is within the claim window always resolve normally;
            this parameter only governs post-expiry late arrivals.

        Returns
        -------
        :class:`ResolutionResult` — ``resolved=False`` when no terminal
        decision was reached (e.g. PR closed but not merged).
        """
        if not _flag_enabled():
            return ResolutionResult(
                claim_id=claim.claim_id,
                resolved=False,
                resolution_value=False,
                evidence="Prediction markets feature flag is disabled.",
            )

        if claim.resolution_status != ResolutionStatus.OPEN:
            log.warning(
                "prediction.late_event: evidence arrived for settled claim "
                "%s (status=%s); dropping.",
                claim.claim_id,
                claim.resolution_status.value,
            )
            return ResolutionResult(
                claim_id=claim.claim_id,
                resolved=False,
                resolution_value=False,
                evidence=f"Claim already {claim.resolution_status.value}; skipping.",
            )

        result = self._dispatch(claim, event, grace_hours=grace_hours)
        if result.resolved and self._store is not None:
            self._store.resolve(claim.claim_id, result.resolution_value, result.evidence)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        claim: StakeableClaim,
        event: GitHubEventPayload,
        *,
        grace_hours: float,
    ) -> ResolutionResult:
        """Route to the appropriate per-type resolver."""
        if not self.can_resolve(claim):
            return ResolutionResult(
                claim_id=claim.claim_id,
                resolved=False,
                resolution_value=False,
                evidence=f"Unsupported question type: {claim.question_type.value!r}.",
            )

        # ---- event-time expiry check (occurrence model) ---- #
        #
        # Resolution is based on when the event *occurred*, not when the
        # webhook was delivered.  Late-delivered evidence of an in-window
        # event resolves the claim normally.  Processing-time finality is
        # bounded by the store sweeper's grace window, not by this check.
        # Post-grace arrivals are voided here and logged as side-output so
        # operators can tune the grace window from empirical data.
        grace = timedelta(hours=grace_hours)
        if claim.expiry is None:
            return ResolutionResult(
                claim_id=claim.claim_id,
                resolved=False,
                resolution_value=False,
                evidence=f"Claim expiry {claim.expiry!r} is invalid; cannot resolve safely.",
            )
        if event.occurred_at is None:
            return ResolutionResult(
                claim_id=claim.claim_id,
                resolved=False,
                resolution_value=False,
                evidence="Event timestamp is missing or invalid; cannot compare against claim expiry.",
            )

        if event.occurred_at > claim.expiry + grace:
            log.warning(
                "prediction.late_event: %s arrived at %s for claim %s "
                "(expiry=%s, grace=%sh); voided.",
                event.event_type,
                event.occurred_at.isoformat(),
                claim.claim_id,
                claim.expiry.isoformat(),
                grace_hours,
            )
            return ResolutionResult(
                claim_id=claim.claim_id,
                resolved=False,
                resolution_value=False,
                evidence=(
                    f"Event occurred_at {event.occurred_at.isoformat()} is past "
                    f"expiry + grace ({(claim.expiry + grace).isoformat()}); voided."
                ),
            )

        if claim.question_type == ClaimType.PR_MERGE:
            return self._resolve_pr_merge(claim, event)
        if claim.question_type == ClaimType.ISSUE_CLOSE:
            return self._resolve_issue_close(claim, event)
        if claim.question_type == ClaimType.CI_PASS:
            return self._resolve_ci_pass(claim, event)

        return ResolutionResult(
            claim_id=claim.claim_id,
            resolved=False,
            resolution_value=False,
            evidence=f"Unhandled question type: {claim.question_type.value!r}.",
        )

    def _resolve_pr_merge(
        self,
        claim: StakeableClaim,
        event: GitHubEventPayload,
    ) -> ResolutionResult:
        if event.event_type != "pull_request" or event.action != "closed":
            return ResolutionResult(
                claim_id=claim.claim_id,
                resolved=False,
                resolution_value=False,
                evidence=f"pull_request action {event.action!r} is not terminal; waiting.",
            )
        if not event.merged:
            return ResolutionResult(
                claim_id=claim.claim_id,
                resolved=False,
                resolution_value=False,
                evidence=(
                    f"PR {claim.target_ref} closed without merge; not a terminal resolution for this claim. "
                    "Claim will expire unless the PR is reopened before expiry."
                ),
            )
        evidence = f"PR {claim.target_ref} merged (action={event.action!r}, merged=True)."
        return ResolutionResult(
            claim_id=claim.claim_id,
            resolved=True,
            resolution_value=True,
            evidence=evidence,
        )

    def _resolve_issue_close(
        self,
        claim: StakeableClaim,
        event: GitHubEventPayload,
    ) -> ResolutionResult:
        if event.event_type != "issues" or event.action != "closed":
            return ResolutionResult(
                claim_id=claim.claim_id,
                resolved=False,
                resolution_value=False,
                evidence=f"issues action {event.action!r} is not terminal; waiting.",
            )
        evidence = (
            f"Issue {claim.target_ref} closed (action={event.action!r})."
        )
        return ResolutionResult(
            claim_id=claim.claim_id,
            resolved=True,
            resolution_value=True,
            evidence=evidence,
        )

    def _resolve_ci_pass(
        self,
        claim: StakeableClaim,
        event: GitHubEventPayload,
    ) -> ResolutionResult:
        if event.event_type not in ("check_run", "workflow_run") or event.action != "completed":
            return ResolutionResult(
                claim_id=claim.claim_id,
                resolved=False,
                resolution_value=False,
                evidence=(
                    f"{event.event_type!r} action {event.action!r} is not terminal; waiting."
                ),
            )
        aggregate = event.raw.get("aggregate")
        if aggregate is not True:
            return ResolutionResult(
                claim_id=claim.claim_id,
                resolved=False,
                resolution_value=False,
                evidence=(
                    f"aggregate signal is {aggregate!r}, not True; "
                    "caller must reduce all required checks before resolving CI_PASS."
                ),
            )
        evidence = (
            f"{event.event_type} completed for {claim.target_ref} "
            f"with aggregate=True (conclusion={event.conclusion!r})."
        )
        return ResolutionResult(
            claim_id=claim.claim_id,
            resolved=True,
            resolution_value=True,
            evidence=evidence,
        )
