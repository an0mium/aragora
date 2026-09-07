"""Fusion tie-breaker for split merge-quorum reviews.

When the heterogeneous model reviewers SPLIT -- at least one supportive PASS and
at least one dissenting changes-requested, with no 2-family supportive quorum --
an OpenRouter Fusion review can be run as a DISCLOSED, ADVISORY tie-breaker.

HARD CONSTRAINT (mirrors aragora.swarm.quorum_evidence's exclusion of router
surfaces): Fusion is a multi-model *blend*, so it is NEVER a counting quorum
family -- counting it would double-count consensus. The tie-breaker comment is
marked ``is_tie_breaker`` and states explicitly that it does NOT satisfy the
2-family model-quorum minimum; it only advises the operator's settlement.

Gated by the ``enable_fusion_quorum_tiebreak`` feature flag (default OFF), and by
an injected ``fusion_review`` runner -- so this stays a pure decision module with
no network/model dependency. The real runner (openrouter/fusion via the reviewer
mechanism) is supplied by the caller once Fusion is runnable; until then the
tie-breaker simply no-ops, never blocking or falsely resolving a quorum.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

TIEBREAKER_HEADING = "## OpenRouter Fusion tie-breaker (advisory, non-counting)"
MIN_DISPLAYABLE_HEAD_SHA_LENGTH = 7
# A fusion_review runner returns the reviewer's raw text (or "" on failure).
FusionReview = Callable[[], str]


def should_run_tiebreaker(
    *,
    supportive_families: Sequence[str],
    dissenting_families: Sequence[str],
    has_supportive_quorum: bool,
    flag_enabled: bool,
) -> bool:
    """True only for a genuine, flag-enabled split with no quorum yet.

    A "split" needs at least one supportive PASS AND at least one dissenting
    changes-requested. Unanimous-pass (already a quorum), unanimous-fail (nothing
    to break -- the dissent is real and should be fixed, not overridden), and the
    flag-off case all return False.
    """
    if not flag_enabled:
        return False
    if has_supportive_quorum:
        return False  # already resolved; no tie to break
    return bool(supportive_families) and bool(dissenting_families)


def compose_tiebreaker_comment(
    *,
    verdict_text: str,
    head_sha: str,
    pr: int,
    supportive_families: Sequence[str],
    dissenting_families: Sequence[str],
) -> str:
    """Compose the disclosed, explicitly non-counting tie-breaker comment.

    ``head_sha`` is required provenance. Empty or implausibly short values are
    displayed as unknown, but ``None`` remains a caller contract violation.
    """
    short_head = _display_head_sha(head_sha)
    return (
        f"{TIEBREAKER_HEADING}\n\n"
        f"Tie evaluated via the OpenRouter Fusion council (multi-model panel + judge), "
        f"grounded on PR #{pr} head {short_head}.\n"
        f"Split: supportive={sorted(supportive_families)} "
        f"dissenting={sorted(dissenting_families)}.\n\n"
        f"{verdict_text.strip()}\n\n"
        "NOTE: This is an ADVISORY tie-breaker. Fusion is a multi-model blend, so it does "
        "NOT count as an independent quorum family and does NOT satisfy the 2-family "
        "model-quorum requirement. It only advises the operator's settlement decision; the "
        "dissenting reviewer's concern must still be resolved or explicitly accepted."
    )


def _display_head_sha(head_sha: str) -> str:
    if head_sha is None:
        raise TypeError("head_sha must be a str, not None")
    normalized = head_sha.strip()
    if len(normalized) < MIN_DISPLAYABLE_HEAD_SHA_LENGTH:
        return "unknown"
    return normalized[:12]


@dataclass(frozen=True)
class TiebreakerOutcome:
    """Result of a tie-breaker attempt. ``comment`` is set only when ``ran``."""

    ran: bool
    reason: str
    comment: str | None = None
    is_tie_breaker: bool = False  # actual tie-breaker comments opt in below


def run_tiebreaker(
    *,
    supportive_families: Sequence[str],
    dissenting_families: Sequence[str],
    has_supportive_quorum: bool,
    flag_enabled: bool,
    head_sha: str,
    pr: int,
    fusion_review: FusionReview | None,
) -> TiebreakerOutcome:
    """Run the advisory tie-breaker if (and only if) the gates pass.

    Returns a non-running outcome (never raises) when the flag is off, there is no
    genuine split, no runner is supplied (Fusion not runnable here), or Fusion
    returns nothing -- so a missing/blocked Fusion never blocks or falsely
    resolves a quorum.
    """
    if not should_run_tiebreaker(
        supportive_families=supportive_families,
        dissenting_families=dissenting_families,
        has_supportive_quorum=has_supportive_quorum,
        flag_enabled=flag_enabled,
    ):
        return TiebreakerOutcome(ran=False, reason="no split, quorum already met, or flag off")
    if fusion_review is None:
        return TiebreakerOutcome(
            ran=False, reason="no fusion_review runner supplied (Fusion not runnable here)"
        )
    try:
        text = fusion_review()
    except Exception as exc:
        return TiebreakerOutcome(ran=False, reason=f"fusion review raised {type(exc).__name__}")
    if not text or not text.strip():
        return TiebreakerOutcome(ran=False, reason="fusion review returned empty")
    comment = compose_tiebreaker_comment(
        verdict_text=text,
        head_sha=head_sha,
        pr=pr,
        supportive_families=supportive_families,
        dissenting_families=dissenting_families,
    )
    return TiebreakerOutcome(
        ran=True,
        reason="tie-breaker composed (advisory, non-counting)",
        comment=comment,
        is_tie_breaker=True,
    )
