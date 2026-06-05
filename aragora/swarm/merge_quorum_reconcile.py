"""Pure decision logic for merge-quorum reconciliation and settlement status.

Background
----------
The enforcing merge gate (``aragora-merge-quorum``) only re-evaluates on
``pull_request`` synchronize-type events, never on ``issue_comment``. The
heterogeneous model-review evidence, however, is posted as PR comments *after*
those events. The result is a recurring stall: the check computes once
(pre-evidence) and concludes FAILURE, and never re-reads the now-complete
evidence, so a PR sits red even though its quorum is satisfied. The safe,
deterministic recovery is to re-run that read-only evaluation once a strictly
newer *countable* evidence comment exists for the current head.

This module is **pure** (no I/O): it takes already-fetched, already-linted
records and returns decisions/summaries. The GitHub I/O lives in the thin CLIs
``scripts/reconcile_merge_quorum.py`` (A1) and ``scripts/settle_status.py``
(A2), mirroring the :mod:`aragora.swarm.unstick` pattern so this logic stays
unit-testable without network access.

Safety invariants (enforced here):

* Never recommend a re-run when the gate already concluded SUCCESS.
* Never recommend a re-run when the latest run belongs to a stale head.
* Never recommend a re-run unless a strictly newer countable comment exists.
* Never recommend a re-run when a *real* required-check failure is present
  (that is a genuine red, not a stale-quorum artifact).
* Respect a per-head cooldown and a per-head max-rerun budget.

Re-running a read-only evaluation can never pass a genuinely failing PR; it only
lets the gate re-read evidence that is already public.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Final

# Mirrors aragora/cli/commands/review_queue.py::_tier_requirement. This is a
# read-only *diagnostic* mapping used only to render the "next action" hint; it
# never gates a merge. Tuple = (required_model_signals, requires_dogfood,
# requires_human_settlement). Kept tiny and annotated so drift is obvious.
TIER_REQUIREMENTS: Final[dict[int, tuple[int, bool, bool]]] = {
    0: (1, False, False),
    1: (2, True, False),
    2: (2, True, False),
    3: (2, True, True),
    4: (2, True, True),
}

_SUCCESS: Final[str] = "SUCCESS"
# Non-terminal states that may arrive via the check-run ``state`` fallback; the
# gate has not concluded, so the right hint is "wait", not "re-run".
_NON_FINAL_STATES: Final[frozenset[str]] = frozenset(
    {"IN_PROGRESS", "QUEUED", "PENDING", "WAITING", "REQUESTED", "EXPECTED"}
)


def parse_iso8601(value: str | None) -> datetime | None:
    """Parse a GitHub ISO-8601 timestamp to an aware UTC datetime, or ``None``.

    Accepts the trailing ``Z`` form GitHub returns. Returns ``None`` for empty
    or unparseable input so callers can fail safe (treat as "unknown").
    """
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class EvidenceComment:
    """A PR comment already evaluated by ``review-queue evidence-lint``.

    Attributes:
        created_at: ISO-8601 creation timestamp of the comment.
        would_count: ``evidence-lint``'s ``would_count`` verdict.
        reviewer_id: The single counted reviewer family (e.g. ``"claude"``),
            or ``""`` when the comment does not count.
        is_dogfood: Whether the comment contributes adversarial-dogfood
            evidence (``evidence-lint`` returned non-empty ``dogfood_evidence``).
    """

    created_at: str
    would_count: bool
    reviewer_id: str = ""
    is_dogfood: bool = False


@dataclass(frozen=True)
class QuorumRun:
    """The latest ``aragora-merge-quorum`` workflow run for a PR head."""

    run_id: int
    created_at: str
    conclusion: str
    head_sha: str


@dataclass(frozen=True)
class RerunDecision:
    """Whether A1 should re-run the gate for a PR."""

    pr_number: int
    should_rerun: bool
    reason: str
    run_id: int | None = None


@dataclass(frozen=True)
class SettlementStatus:
    """A2's read-only view of where a PR's settlement actually stands."""

    pr_number: int
    head_sha: str
    tier: int | None
    counted_reviewer_ids: list[str] = field(default_factory=list)
    has_dogfood: bool = False
    human_settlement_present: bool = False
    quorum_conclusion: str = ""
    next_action: str = ""

    @property
    def signal_count(self) -> int:
        return len(self.counted_reviewer_ids)


def _newest_countable_created_at(comments: list[EvidenceComment]) -> datetime | None:
    newest: datetime | None = None
    for comment in comments:
        if not comment.would_count:
            continue
        created = parse_iso8601(comment.created_at)
        if created is None:
            continue
        if newest is None or created > newest:
            newest = created
    return newest


def counted_reviewer_ids(comments: list[EvidenceComment]) -> list[str]:
    """Distinct counted reviewer families across countable comments."""
    ids: set[str] = set()
    for comment in comments:
        if comment.would_count and comment.reviewer_id:
            ids.add(comment.reviewer_id)
    return sorted(ids)


def plan_rerun(
    *,
    pr_number: int,
    run: QuorumRun | None,
    comments: list[EvidenceComment],
    current_head_sha: str,
    now: datetime,
    last_rerun_at: datetime | None = None,
    reruns_this_head: int = 0,
    cooldown_seconds: int = 600,
    max_reruns_per_head: int = 3,
    has_real_required_failure: bool = False,
) -> RerunDecision:
    """Decide whether to re-run the gate so it re-reads current-head evidence.

    All checks fail safe (a ``False`` decision) when an input is missing or
    ambiguous. See module docstring for the safety invariants.
    """

    def decide(should: bool, reason: str) -> RerunDecision:
        return RerunDecision(
            pr_number=pr_number,
            should_rerun=should,
            reason=reason,
            run_id=run.run_id if run else None,
        )

    if run is None:
        return decide(False, "no aragora-merge-quorum run found for the current head")
    if run.conclusion.upper() == _SUCCESS:
        return decide(False, "quorum check already SUCCESS")
    if run.head_sha and current_head_sha and run.head_sha != current_head_sha:
        return decide(False, "latest run belongs to a stale head; await current-head run")
    if has_real_required_failure:
        return decide(False, "a real required-check failure is present; not a stale-quorum case")

    newest = _newest_countable_created_at(comments)
    if newest is None:
        return decide(False, "no countable evidence present yet")

    run_created = parse_iso8601(run.created_at)
    if run_created is None:
        return decide(False, "quorum run timestamp unparseable")
    if not (run_created < newest):
        return decide(False, "quorum run already postdates the newest countable evidence")

    if reruns_this_head >= max_reruns_per_head:
        return decide(False, f"max reruns reached for this head ({max_reruns_per_head})")
    if last_rerun_at is not None:
        elapsed = (now - last_rerun_at).total_seconds()
        if elapsed < cooldown_seconds:
            return decide(False, f"within cooldown ({int(elapsed)}s < {cooldown_seconds}s)")

    return decide(
        True,
        "stale quorum check predates newer countable evidence; safe read-only re-run",
    )


def summarize_settlement(
    *,
    pr_number: int,
    head_sha: str,
    tier: int | None,
    comments: list[EvidenceComment],
    human_settlement_present: bool,
    quorum_conclusion: str,
) -> SettlementStatus:
    """Compute the live settlement state directly from linted comments.

    Immune to the ``merge-packet`` short-circuit (which hides the quorum detail
    whenever the gate check is failing), because it counts evidence from the
    comment lint results rather than from the packet.
    """
    ids = counted_reviewer_ids(comments)
    has_dogfood = any(c.would_count and c.is_dogfood for c in comments)

    required_signals, requires_dogfood, requires_human = TIER_REQUIREMENTS.get(
        tier if tier is not None else -1, (2, True, True)
    )

    if quorum_conclusion.upper() == _SUCCESS:
        next_action = "none — quorum check is green; PR is ready to merge"
    elif len(ids) < required_signals:
        missing = required_signals - len(ids)
        next_action = f"collect {missing} more distinct model signal(s) on the current head"
    elif requires_dogfood and not has_dogfood:
        next_action = "post adversarial-dogfood evidence on the current head"
    elif requires_human and not human_settlement_present:
        next_action = (
            "operator: record human settlement for the current head "
            "(scripts/settle_tier4_pr.py --apply)"
        )
    elif not quorum_conclusion or quorum_conclusion.upper() in _NON_FINAL_STATES:
        next_action = "wait for the aragora-merge-quorum check to run on the current head"
    else:
        next_action = (
            "re-run aragora-merge-quorum so it re-reads the current evidence "
            "(scripts/reconcile_merge_quorum.py --apply)"
        )

    return SettlementStatus(
        pr_number=pr_number,
        head_sha=head_sha,
        tier=tier,
        counted_reviewer_ids=ids,
        has_dogfood=has_dogfood,
        human_settlement_present=human_settlement_present,
        quorum_conclusion=quorum_conclusion,
        next_action=next_action,
    )
