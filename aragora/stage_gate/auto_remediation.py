"""Autonomous remediation for Stage-Gate Conductor drift issues.

The Stage-Gate Conductor writes a ``stage-gate-drift`` labeled issue when it
detects a known drift pattern (boss-ready applied to a deferred track,
duplicated epics, or stale benchmark truth).  Today a human has to pick up
that issue and decide what to do.

This module extends the flow by offering a small, deterministic dispatcher:
if the drift pattern matches one of the recognised remediations, a targeted
repair is attempted.  Everything else no-ops, leaving the human-owned issue
untouched.

Design goals
------------
* **Conservative by default** — if the drift type is unknown, the evidence is
  thin, or the feature flag is disabled, the dispatcher logs and returns a
  ``noop`` result.
* **Flag gated** — the behaviour is off unless
  :data:`AUTO_REMEDIATION_FLAG_ENV` (``ARAGORA_STAGE_GATE_AUTO_REMEDIATION``)
  is set to a truthy value in the environment.
* **Dependency injected** — the remediation functions do not talk to GitHub
  directly; callers provide a small action interface so tests can exercise
  each pattern without network I/O.
* **Auditable** — every invocation returns a :class:`RemediationResult` with
  the pattern name, actions taken, and a human-readable reason.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


AUTO_REMEDIATION_FLAG_ENV = "ARAGORA_STAGE_GATE_AUTO_REMEDIATION"

_TRUTHY_VALUES = frozenset({"1", "true", "t", "yes", "y", "on"})

KNOWN_DRIFT_TYPES: frozenset[str] = frozenset(
    {
        "boss_ready_on_deferred_track",
        "duplicate_epic",
        "doc_staleness",
    }
)


# ---------------------------------------------------------------------------
# Result / action data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemediationAction:
    """One concrete action recorded during a remediation attempt."""

    kind: str
    target: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "target": self.target, "detail": self.detail}


@dataclass(frozen=True)
class RemediationResult:
    """Outcome of a remediation dispatch."""

    applied: bool
    drift_type: str
    pattern: str
    reason: str
    actions: tuple[RemediationAction, ...] = ()
    evidence_keys: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "drift_type": self.drift_type,
            "pattern": self.pattern,
            "reason": self.reason,
            "actions": [action.to_dict() for action in self.actions],
            "evidence_keys": list(self.evidence_keys),
        }


# ---------------------------------------------------------------------------
# Action protocol
# ---------------------------------------------------------------------------


class RemediationActions(Protocol):
    """Thin interface around GitHub / CI side-effects.

    The concrete implementation is provided by the Conductor's GitHub app
    adapter.  Tests pass a stub that records calls.  Every method returns a
    boolean that the dispatcher surfaces via the audit log; a method returning
    ``False`` is treated as a skip, not a hard error.
    """

    def remove_label(self, issue_number: int, label: str) -> bool: ...

    def post_comment(self, issue_number: int, body: str) -> bool: ...

    def trigger_workflow(self, workflow: str, *, ref: str | None = None) -> bool: ...


# ---------------------------------------------------------------------------
# Feature flag helper
# ---------------------------------------------------------------------------


def auto_remediation_enabled(
    env: dict[str, str] | None = None,
    *,
    flag: str = AUTO_REMEDIATION_FLAG_ENV,
) -> bool:
    """Return ``True`` when the feature flag is truthy."""
    source = env if env is not None else os.environ
    value = str(source.get(flag, "")).strip().lower()
    return value in _TRUTHY_VALUES


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def remediate_drift(
    drift_type: str,
    evidence: dict[str, Any],
    *,
    actions: RemediationActions | None = None,
    env: dict[str, str] | None = None,
) -> RemediationResult:
    """Dispatch a remediation for ``drift_type`` if the pattern is recognised.

    Parameters
    ----------
    drift_type:
        Canonical drift label emitted by the Stage-Gate Conductor.  See
        :data:`KNOWN_DRIFT_TYPES` for the recognised values.
    evidence:
        Structured payload describing the drift.  The required keys depend on
        the remediation; missing keys trigger a ``noop`` result.
    actions:
        Concrete action interface.  When ``None`` the dispatcher returns a
        ``noop`` result — useful when the caller only wants to know whether a
        remediation *would* be attempted.
    env:
        Optional environment override used for the feature flag check.  When
        ``None`` the process environment is consulted.
    """
    normalized = str(drift_type or "").strip().lower()
    if not normalized:
        return _noop(drift_type, "missing_drift_type", "no drift_type provided")

    if not auto_remediation_enabled(env):
        return _noop(
            drift_type,
            "flag_disabled",
            f"auto-remediation disabled (set {AUTO_REMEDIATION_FLAG_ENV}=1 to enable)",
            evidence=evidence,
        )

    if normalized not in KNOWN_DRIFT_TYPES:
        return _noop(
            drift_type,
            "unrecognised_drift",
            f"no remediation registered for drift_type={drift_type!r}",
            evidence=evidence,
        )

    if actions is None:
        return _noop(
            drift_type,
            "no_actions_adapter",
            "remediation recognised but no actions adapter was provided",
            evidence=evidence,
        )

    if normalized == "boss_ready_on_deferred_track":
        return _remediate_boss_ready_deferred(evidence, actions)
    if normalized == "duplicate_epic":
        return _remediate_duplicate_epic(evidence, actions)
    if normalized == "doc_staleness":
        return _remediate_doc_staleness(evidence, actions)

    # Defensive fall-through; should not happen because of the KNOWN_DRIFT_TYPES check.
    return _noop(drift_type, "unhandled_pattern", "pattern matched but no handler ran")


# ---------------------------------------------------------------------------
# Individual remediations
# ---------------------------------------------------------------------------


def _remediate_boss_ready_deferred(
    evidence: dict[str, Any],
    actions: RemediationActions,
) -> RemediationResult:
    """Strip ``boss-ready`` from an issue carrying a deferred-track label.

    Required evidence
    -----------------
    ``issue_number``:
        Integer number of the drift-bearing issue.
    ``track``:
        Name of the deferred track (for the comment text).
    ``deferred_label`` *(optional)*:
        The specific deferred-track label detected; used in the comment.
    """
    issue_number = _coerce_int(evidence.get("issue_number"))
    track = _coerce_text(evidence.get("track") or evidence.get("deferred_label"))
    if issue_number is None or not track:
        return _noop(
            "boss_ready_on_deferred_track",
            "thin_evidence",
            "expected issue_number and track in evidence",
            evidence=evidence,
        )

    applied_actions: list[RemediationAction] = []
    comment = (
        f"Stage-Gate auto-remediation: removed `boss-ready` because this issue is "
        f"tracked on the deferred `{track}` lane. The substrate gate for that lane "
        "has not opened yet. Re-add `boss-ready` only after the gate opens."
    )

    removed = _safely(lambda: actions.remove_label(issue_number, "boss-ready"))
    if removed:
        applied_actions.append(
            RemediationAction(
                kind="remove_label",
                target=f"issue#{issue_number}",
                detail="boss-ready",
            )
        )

    commented = _safely(lambda: actions.post_comment(issue_number, comment))
    if commented:
        applied_actions.append(
            RemediationAction(
                kind="post_comment",
                target=f"issue#{issue_number}",
                detail=f"deferred_track={track}",
            )
        )

    if not applied_actions:
        return _noop(
            "boss_ready_on_deferred_track",
            "no_actions_taken",
            "actions adapter rejected all calls",
            evidence=evidence,
        )

    return RemediationResult(
        applied=True,
        drift_type="boss_ready_on_deferred_track",
        pattern="strip_boss_ready_and_comment",
        reason=f"stripped boss-ready from deferred track {track}",
        actions=tuple(applied_actions),
        evidence_keys=_evidence_keys(evidence),
    )


def _remediate_duplicate_epic(
    evidence: dict[str, Any],
    actions: RemediationActions,
) -> RemediationResult:
    """Comment on both epics linking each other; flag for human merge decision.

    Required evidence
    -----------------
    ``epic_numbers``:
        Iterable of exactly two integer issue numbers that are duplicates of
        each other.  Anything other than a pair is treated as thin evidence.
    """
    raw = evidence.get("epic_numbers") or evidence.get("issue_numbers")
    numbers = _coerce_int_sequence(raw, length=2)
    if numbers is None:
        return _noop(
            "duplicate_epic",
            "thin_evidence",
            "expected epic_numbers to be a pair of issue numbers",
            evidence=evidence,
        )

    first, second = numbers
    applied_actions: list[RemediationAction] = []

    for issue_number, sibling in ((first, second), (second, first)):
        body = (
            f"Stage-Gate auto-remediation: potential duplicate of #{sibling}. "
            "Both epics appear to track the same goal. A human must decide which "
            "to keep and which to close — not auto-merging."
        )

        def _post(n: int = issue_number, b: str = body) -> bool:
            return actions.post_comment(n, b)

        if _safely(_post):
            applied_actions.append(
                RemediationAction(
                    kind="post_comment",
                    target=f"issue#{issue_number}",
                    detail=f"links=#{sibling}",
                )
            )

    if not applied_actions:
        return _noop(
            "duplicate_epic",
            "no_actions_taken",
            "actions adapter rejected all comment calls",
            evidence=evidence,
        )

    return RemediationResult(
        applied=True,
        drift_type="duplicate_epic",
        pattern="cross_link_and_flag",
        reason=f"cross-linked duplicate epics #{first} and #{second} for human review",
        actions=tuple(applied_actions),
        evidence_keys=_evidence_keys(evidence),
    )


def _remediate_doc_staleness(
    evidence: dict[str, Any],
    actions: RemediationActions,
) -> RemediationResult:
    """Rerun the benchmark/doc publication workflow to refresh truth surfaces.

    Required evidence
    -----------------
    ``workflow`` *(optional)*:
        Workflow identifier (name or path).  Defaults to
        ``benchmark-publication``.
    ``ref`` *(optional)*:
        Git ref to run against.  Passed through to the workflow trigger;
        defaults to ``None`` so the underlying adapter can pick the sensible
        default (typically ``main``).
    ``issue_number`` *(optional)*:
        When provided, a comment is posted on the drift issue recording the
        workflow trigger so humans can audit it.
    """
    workflow = _coerce_text(evidence.get("workflow")) or "benchmark-publication"
    ref = _coerce_text(evidence.get("ref")) or None
    issue_number = _coerce_int(evidence.get("issue_number"))

    applied_actions: list[RemediationAction] = []
    triggered = _safely(lambda: actions.trigger_workflow(workflow, ref=ref))
    if triggered:
        applied_actions.append(
            RemediationAction(
                kind="trigger_workflow",
                target=workflow,
                detail=f"ref={ref or 'default'}",
            )
        )

    if issue_number is not None:
        body = (
            "Stage-Gate auto-remediation: re-triggered the "
            f"`{workflow}` workflow to refresh stale benchmark truth "
            f"(ref={ref or 'default'})."
        )
        if _safely(lambda: actions.post_comment(issue_number, body)):
            applied_actions.append(
                RemediationAction(
                    kind="post_comment",
                    target=f"issue#{issue_number}",
                    detail=f"workflow={workflow}",
                )
            )

    if not applied_actions:
        return _noop(
            "doc_staleness",
            "no_actions_taken",
            "workflow trigger failed and no comment was posted",
            evidence=evidence,
        )

    return RemediationResult(
        applied=True,
        drift_type="doc_staleness",
        pattern="rerun_publication_workflow",
        reason=f"triggered workflow {workflow!r} to refresh benchmark truth",
        actions=tuple(applied_actions),
        evidence_keys=_evidence_keys(evidence),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safely(call: Callable[[], bool]) -> bool:
    """Swallow adapter exceptions so one broken call does not crash the run."""
    try:
        return bool(call())
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("stage_gate auto-remediation action raised: %s", exc)
        return False


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _coerce_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _coerce_int_sequence(value: Any, *, length: int) -> tuple[int, ...] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return None
    numbers: list[int] = []
    for item in value:
        coerced = _coerce_int(item)
        if coerced is None:
            return None
        numbers.append(coerced)
    if len(numbers) != length:
        return None
    return tuple(numbers)


def _evidence_keys(evidence: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(str(key) for key in evidence or {}))


def _noop(
    drift_type: str,
    pattern: str,
    reason: str,
    *,
    evidence: dict[str, Any] | None = None,
) -> RemediationResult:
    logger.info(
        "stage_gate auto-remediation noop: drift=%s pattern=%s reason=%s",
        drift_type,
        pattern,
        reason,
    )
    return RemediationResult(
        applied=False,
        drift_type=str(drift_type or ""),
        pattern=pattern,
        reason=reason,
        actions=(),
        evidence_keys=_evidence_keys(evidence or {}),
    )


__all__: Sequence[str] = (
    "AUTO_REMEDIATION_FLAG_ENV",
    "KNOWN_DRIFT_TYPES",
    "RemediationAction",
    "RemediationActions",
    "RemediationResult",
    "auto_remediation_enabled",
    "remediate_drift",
)
