"""
Declarative contract for the local-Codex → GitHub automation handoff substrate.

This module is the first read-only step of the consolidation proposed in
``docs/plans/2026-04-28-handoff-contract-derivation.md``. It encodes the eight
latent invariants the 17 ``fix(automation)`` outbox/handoff PRs of
2026-04-21..28 converged on:

  C1. Idempotency-key as primary identity.
  C2. Terminal-state precedence (4-level satisfaction predicate).
  C3. Outbox state-root resolution as a single function.
  C4. Branch field fingerprinting canonical.
  C5. Patch-equivalence as first-class satisfaction.
  C6. Open-PR identity beats outbox identity.
  C7. Dry-run is read-only or it's a bug.
  C8. Evidence schema loosely-typed but defensively-validated.

Per the spec's sequencing suggestion, this skeleton lands as a pure module
with **no behavior change** — no script imports it yet. Its only consumer is
the test fixture in ``tests/swarm/test_handoff_contract.py``. Future PRs will
migrate ``scripts/publish_automation_handoffs.py``,
``scripts/reconcile_automation_outbox.py``, and
``scripts/audit_codex_branch_backlog.py`` onto this contract.

This file is **additive** and **does not import** the legacy scripts; it
duplicates a small amount of regex/string handling rather than circular
import them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

# C1: idempotency keys are the canonical handoff identity. Every other
# identity (issue title, branch, patch hash) is a secondary index.
REQUIRED_OUTBOX_KEYS: tuple[str, ...] = (
    "task",
    "requires_github",
    "requested_action",
    "repo",
    "local_evidence",
    "validation",
    "idempotency_key",
    "created_at",
)

# C8: known canonical action verbs. Unknown actions are tolerated as JSON
# strings whose dict has an "action" key, which keeps the contract
# forward-compatible while surfacing diagnostics.
PR_OPEN_REQUEST_CANONICAL_ACTION: str = "open_pr"
PR_OPEN_REQUEST_ACTIONS: frozenset[str] = frozenset(
    {
        "open_pr",
        "open_pull_request",
        "open_or_update_pr",
        "open_or_update_pull_request",
        "push_branch_and_open_pr",
        "push_branch_and_open_pull_request",
        "push_branch_and_open_or_update_pr",
        "push_branch_and_open_or_update_pull_request",
    }
)

# C2: terminal receipt statuses. A receipt with any of these statuses
# satisfies the corresponding handoff identity.
TERMINAL_RECEIPT_STATUSES: frozenset[str] = frozenset(
    {"published", "already_satisfied", "completed", "skipped"}
)


class SatisfactionKind(str, Enum):
    """C2: enumerable satisfaction predicate kinds, in precedence order."""

    TERMINAL_RECEIPT = "terminal_receipt"
    MERGED_PR = "merged_pr"
    PATCH_EQUIVALENT = "patch_equivalent"  # C5
    OPEN_PR_MATCH = "open_pr_match"  # C6
    RECEIPT_ONLY_BRANCH = "receipt_only_branch"


@dataclass(frozen=True)
class HandoffIdentity:
    """C1+C4: canonical handoff identity with stable fingerprint."""

    idempotency_key: str
    branch_name: str | None
    head_sha: str | None
    action_kind: str
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "idempotency_key": self.idempotency_key,
            "branch_name": self.branch_name,
            "head_sha": self.head_sha,
            "action_kind": self.action_kind,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class InvalidHandoff:
    """C8: a handoff payload that fails validation. Quarantine candidate."""

    reason: str
    missing_keys: tuple[str, ...] = ()
    raw: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "missing_keys": list(self.missing_keys),
        }


@dataclass(frozen=True)
class SatisfactionSignal:
    """C2: a single satisfaction predicate matched."""

    kind: SatisfactionKind
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class SatisfactionContext:
    """Read-only inputs to ``evaluate_satisfaction``.

    Concrete callers populate this from disk (receipt directory) and from
    the network (open PR heads). The contract module itself does NOT touch
    the file system or call gh; it operates only on the inputs supplied
    here.
    """

    terminal_receipt_keys: frozenset[str] = frozenset()
    merged_pr_branches: frozenset[str] = frozenset()
    patch_equivalent_branches: frozenset[str] = frozenset()
    open_pr_heads: Mapping[str, int] = field(default_factory=dict)
    receipt_only_branches: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def _normalize_action(value: Any) -> str:
    """C8: extract a canonical action kind from a payload that may be a
    Mapping, a JSON string, or a plain string."""
    if isinstance(value, Mapping):
        action = str(value.get("type") or value.get("action") or "").strip()
        return action or "unknown"
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return text or "unknown"
            if isinstance(parsed, Mapping):
                return _normalize_action(parsed)
        return text or "unknown"
    return "unknown"


def _resolve_branch_field(payload: Mapping[str, Any]) -> str | None:
    """C4: tolerant branch extraction across schema-drift shapes."""
    local_evidence = payload.get("local_evidence")
    if isinstance(local_evidence, Mapping):
        branch = str(local_evidence.get("branch") or "").strip()
        if branch:
            return branch
    if isinstance(local_evidence, Sequence) and not isinstance(local_evidence, (str, bytes)):
        for item in local_evidence:
            if isinstance(item, Mapping):
                branch = str(item.get("branch") or "").strip()
                if branch:
                    return branch
    requested_action = payload.get("requested_action")
    if isinstance(requested_action, Mapping):
        branch = str(requested_action.get("branch") or "").strip()
        if branch:
            return branch
    branch = str(payload.get("branch") or "").strip()
    return branch or None


def _resolve_head_sha(payload: Mapping[str, Any]) -> str | None:
    """C4: head SHA extraction tolerant to schema drift."""
    local_evidence = payload.get("local_evidence")
    if isinstance(local_evidence, Mapping):
        sha = str(local_evidence.get("head_sha") or "").strip()
        if sha:
            return sha
    if isinstance(local_evidence, Sequence) and not isinstance(local_evidence, (str, bytes)):
        for item in local_evidence:
            if isinstance(item, Mapping):
                sha = str(item.get("head_sha") or "").strip()
                if sha:
                    return sha
    sha = str(payload.get("head_sha") or "").strip()
    return sha or None


def compute_fingerprint(
    idempotency_key: str,
    branch_name: str | None,
    head_sha: str | None,
    action_kind: str,
) -> str:
    """C4: stable fingerprint over canonical identity components.

    Used as a secondary identity index when the idempotency key alone is
    insufficient (e.g., schema drift across publisher/reconciler/auditor
    snapshots of the same logical handoff).
    """
    payload = "|".join(
        [
            idempotency_key.strip(),
            (branch_name or "").strip(),
            (head_sha or "").strip(),
            action_kind.strip(),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def parse_outbox_entry(
    payload: Mapping[str, Any],
) -> HandoffIdentity | InvalidHandoff:
    """C1+C4+C8: validate + fingerprint an outbox entry.

    Returns either a ``HandoffIdentity`` (valid + fingerprinted) or an
    ``InvalidHandoff`` describing why validation failed. Never raises on
    schema drift; only on programmer error (e.g., non-Mapping input).
    """
    if not isinstance(payload, Mapping):
        return InvalidHandoff(
            reason="payload is not a Mapping",
            raw=None,
        )

    # C8: required keys must be *present* in the payload. Falsy values are
    # NOT treated as missing here — that would conflict with C4's top-level
    # branch/sha fallback (which fires when local_evidence={}), and with the
    # need to permit requires_github=False or validation=[]. Per-field
    # semantic checks below catch the cases where a present-but-empty value
    # is genuinely invalid (idempotency_key, requested_action).
    missing = tuple(k for k in REQUIRED_OUTBOX_KEYS if k not in payload)
    if missing:
        return InvalidHandoff(
            reason=f"missing required keys: {', '.join(missing)}",
            missing_keys=missing,
            raw=payload,
        )

    idempotency_key = str(payload.get("idempotency_key") or "").strip()
    if not idempotency_key:
        return InvalidHandoff(
            reason="empty idempotency_key",
            missing_keys=("idempotency_key",),
            raw=payload,
        )

    branch_name = _resolve_branch_field(payload)
    head_sha = _resolve_head_sha(payload)
    action_kind = _normalize_action(payload.get("requested_action"))

    # C8: enforce the action-verb whitelist. The whitelist is the contract's
    # binding constraint on what publisher actions are valid; permitting
    # arbitrary action_kinds defeats the purpose of having a whitelist at
    # all. New canonical actions MUST be added to PR_OPEN_REQUEST_ACTIONS
    # explicitly so each addition is reviewable.
    if action_kind not in PR_OPEN_REQUEST_ACTIONS:
        return InvalidHandoff(
            reason=(
                f"unknown action kind {action_kind!r}; "
                f"must be one of: {', '.join(sorted(PR_OPEN_REQUEST_ACTIONS))}"
            ),
            raw=payload,
        )

    fingerprint = compute_fingerprint(idempotency_key, branch_name, head_sha, action_kind)

    return HandoffIdentity(
        idempotency_key=idempotency_key,
        branch_name=branch_name,
        head_sha=head_sha,
        action_kind=action_kind,
        fingerprint=fingerprint,
    )


def evaluate_satisfaction(
    identity: HandoffIdentity,
    context: SatisfactionContext,
) -> SatisfactionSignal | None:
    """C2+C5+C6: enumerate the four satisfaction predicates in precedence
    order. Return the first match or ``None``.

    Precedence (highest first):
      1. terminal receipt with matching idempotency key
      2. merged PR with matching head branch
      3. patch-equivalent branch (already-on-base)
      4. open PR with matching head branch (covered by another PR)
      5. receipt-only branch (no PR but receipt exists)
    """
    if identity.idempotency_key in context.terminal_receipt_keys:
        return SatisfactionSignal(
            kind=SatisfactionKind.TERMINAL_RECEIPT,
            evidence={"idempotency_key": identity.idempotency_key},
        )

    if identity.branch_name and identity.branch_name in context.merged_pr_branches:
        return SatisfactionSignal(
            kind=SatisfactionKind.MERGED_PR,
            evidence={"branch": identity.branch_name},
        )

    if identity.branch_name and identity.branch_name in context.patch_equivalent_branches:
        return SatisfactionSignal(
            kind=SatisfactionKind.PATCH_EQUIVALENT,
            evidence={"branch": identity.branch_name},
        )

    if identity.branch_name and identity.branch_name in context.open_pr_heads:
        return SatisfactionSignal(
            kind=SatisfactionKind.OPEN_PR_MATCH,
            evidence={
                "branch": identity.branch_name,
                "pr": context.open_pr_heads[identity.branch_name],
            },
        )

    if identity.branch_name and identity.branch_name in context.receipt_only_branches:
        return SatisfactionSignal(
            kind=SatisfactionKind.RECEIPT_ONLY_BRANCH,
            evidence={"branch": identity.branch_name},
        )

    return None


@dataclass(frozen=True)
class ReconcilePlanEntry:
    """A single proposed reconciliation action for one handoff identity."""

    identity: HandoffIdentity
    action: Literal["archive", "publish", "skip"]
    reason: str
    signal: SatisfactionSignal | None = None


@dataclass(frozen=True)
class ReconcilePlan:
    """C7: a reconciliation plan must be enumerable BEFORE any mutation.

    Callers compute a ``ReconcilePlan`` under dry-run mode and only invoke
    side effects when ``apply`` is True. This module provides only the
    plan; the application is delegated to the callers (publisher /
    reconciler / auditor) that will adopt this contract in follow-up PRs.
    """

    entries: tuple[ReconcilePlanEntry, ...]
    base_ref: str

    def archive_count(self) -> int:
        return sum(1 for e in self.entries if e.action == "archive")

    def publish_count(self) -> int:
        return sum(1 for e in self.entries if e.action == "publish")

    def skip_count(self) -> int:
        return sum(1 for e in self.entries if e.action == "skip")


def is_dry_run_safe(plan: ReconcilePlan) -> bool:
    """C7: a plan is dry-run safe iff every action is enumerable and
    nothing in this module ever performs a side effect.

    Since this module is a pure-function skeleton (no I/O, no
    mutation), every plan is dry-run safe by construction. The function
    exists so downstream callers can assert this invariant in tests
    against their own mutation chokepoints.
    """
    return True


def plan_archive_satisfied(
    identities: Sequence[HandoffIdentity],
    context: SatisfactionContext,
    base_ref: str,
) -> ReconcilePlan:
    """Convenience: build a plan that archives every satisfied handoff and
    skips the rest. The publisher's ``open_pr`` step is out of scope for
    this skeleton.
    """
    entries: list[ReconcilePlanEntry] = []
    for identity in identities:
        signal = evaluate_satisfaction(identity, context)
        if signal is not None:
            entries.append(
                ReconcilePlanEntry(
                    identity=identity,
                    action="archive",
                    reason=signal.kind.value,
                    signal=signal,
                )
            )
        else:
            entries.append(
                ReconcilePlanEntry(
                    identity=identity,
                    action="skip",
                    reason="not satisfied",
                )
            )
    return ReconcilePlan(entries=tuple(entries), base_ref=base_ref)


__all__ = [
    "HandoffIdentity",
    "InvalidHandoff",
    "PR_OPEN_REQUEST_ACTIONS",
    "PR_OPEN_REQUEST_CANONICAL_ACTION",
    "REQUIRED_OUTBOX_KEYS",
    "ReconcilePlan",
    "ReconcilePlanEntry",
    "SatisfactionContext",
    "SatisfactionKind",
    "SatisfactionSignal",
    "TERMINAL_RECEIPT_STATUSES",
    "compute_fingerprint",
    "evaluate_satisfaction",
    "is_dry_run_safe",
    "parse_outbox_entry",
    "plan_archive_satisfied",
]
