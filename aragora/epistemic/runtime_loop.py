"""DIC-23 Dialectical Runtime Loop orchestrator.

Connects the decay monitor (DIC-20), quarantine policy (DIC-21), and
repair pipeline (DIC-22) into a single, observable, receipt-carrying trace:

    DecaySignal
        → apply_quarantine_policy          (DIC-21)
        → propose_repair (optional)        (DIC-22)
        → DialecticalEvent                 (this module)

Default posture is *report-only*: no state is mutated, no issues are
created, and nothing is routed to the live queue.  Crux probing
(DIC-15 / crux-finder consensus mode integration) is deferred to a
follow-on PR and will clear the ``crux_probe_skipped`` field.

Flag: ``ARAGORA_DIALECTICAL_RUNTIME_ENABLED`` (default off).
Dataclasses and ``run_dialectical_loop`` are always importable; the
flag is checked only when ``require_enabled=True`` (the default for
production callers) to avoid silent no-ops. Tests and demos should set
the flag at the process boundary; this module does not mutate
``os.environ``.

Advances: #6217 (DIC-23)
See also: ``docs/plans/2026-04-18-dialectical-runtime-synthesis.md``
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

from aragora.epistemic.decay_monitor import DecaySignal
from aragora.epistemic.quarantine_policy import (
    DEFAULT_POLICIES,
    QuarantinePolicy,
    apply_quarantine_policy,
)
from aragora.epistemic.repair import RepairSpec, propose_repair

_FLAG = "ARAGORA_DIALECTICAL_RUNTIME_ENABLED"


def dialectical_runtime_enabled() -> bool:
    """Return True if the DIC-23 runtime loop is enabled for this process."""
    raw = str(os.environ.get(_FLAG) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _event_id(unit_id: str, recommended_action: str, ts: str) -> str:
    material = f"drt|{unit_id}|{recommended_action}|{ts}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"drt_{digest}"


@dataclass(frozen=True)
class DialecticalEvent:
    """Canonical trace record for one DIC-23 orchestration pass.

    Carries the decay summary, the quarantine decision, and (optionally)
    a repair spec.  ``crux_probe_skipped`` is always ``True`` in this
    slice; crux-finder integration (DIC-15) updates it in a follow-on PR.

    ``prior_receipt_ids`` is the caller-supplied ancestry chain stored as
    an immutable tuple.  ``metadata`` is stored as a shallow-frozen mapping:
    the outer mapping is immutable but nested mutable values are not
    deep-copied.  Callers must not store references to mutable nested
    objects in metadata when receipt-chain immutability matters.
    """

    event_id: str
    code_unit_id: str
    integrity_score: float
    recommended_action: str
    quarantine_action: str
    crux_probe_skipped: bool
    repair_spec: RepairSpec | None
    prior_receipt_ids: tuple[str, ...]
    created_at: str
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "code_unit_id": self.code_unit_id,
            "integrity_score": round(self.integrity_score, 4),
            "recommended_action": self.recommended_action,
            "quarantine_action": self.quarantine_action,
            "crux_probe_skipped": self.crux_probe_skipped,
            "repair_spec": self.repair_spec.to_dict() if self.repair_spec else None,
            "prior_receipt_ids": list(self.prior_receipt_ids),
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


class DialecticalRuntimeError(RuntimeError):
    """Raised when the loop is invoked but the feature flag is off."""


def run_dialectical_loop(
    signal: DecaySignal,
    *,
    policy: QuarantinePolicy | None = None,
    code_unit_class: str = "default",
    enable_repair_proposal: bool = False,
    prior_receipt_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    require_enabled: bool = True,
) -> DialecticalEvent:
    """Run one DIC-23 orchestration pass and return a :class:`DialecticalEvent`.

    Parameters
    ----------
    signal:
        The :class:`~aragora.epistemic.decay_monitor.DecaySignal` to act on.
    policy:
        Explicit :class:`~aragora.epistemic.quarantine_policy.QuarantinePolicy`.
        When ``None``, resolves from
        :data:`~aragora.epistemic.quarantine_policy.DEFAULT_POLICIES` using
        ``code_unit_class``.
    code_unit_class:
        Policy class key used when ``policy`` is ``None``.
    enable_repair_proposal:
        When ``True`` *and* the quarantine decision is
        ``"repair_required"``, calls
        :func:`~aragora.epistemic.repair.propose_repair` to produce a
        ``report_only`` :class:`~aragora.epistemic.repair.RepairSpec`.
        Defaults to ``False`` — pure report-only trace with no spec.
    prior_receipt_ids:
        Receipt IDs to attach to the event's ancestry chain.
    metadata:
        Arbitrary pass-through payload stored in the event.
    require_enabled:
        When ``True`` (default), raises :class:`DialecticalRuntimeError`
        if ``ARAGORA_DIALECTICAL_RUNTIME_ENABLED`` is not set.  Pass
        ``False`` in unit tests to build events without touching the
        environment.

    Returns
    -------
    DialecticalEvent
        Immutable trace record.  Never mutates claim manifests, never
        creates issues, never routes to the live queue.
    """
    if require_enabled and not dialectical_runtime_enabled():
        raise DialecticalRuntimeError(
            "DIC-23 dialectical runtime is disabled; "
            "set ARAGORA_DIALECTICAL_RUNTIME_ENABLED=1 at the process boundary"
        )

    resolved_policy = policy or DEFAULT_POLICIES.get(code_unit_class, DEFAULT_POLICIES["default"])
    quarantine = apply_quarantine_policy(signal, resolved_policy)

    repair_spec: RepairSpec | None = None
    if enable_repair_proposal and quarantine.policy_action == "repair_required":
        repair_spec = propose_repair(signal)

    ts = _utc_now_iso()
    event_id = _event_id(signal.code_unit_id, signal.recommended_action, ts)

    return DialecticalEvent(
        event_id=event_id,
        code_unit_id=signal.code_unit_id,
        integrity_score=signal.integrity_score,
        recommended_action=signal.recommended_action,
        quarantine_action=quarantine.policy_action,
        crux_probe_skipped=True,
        repair_spec=repair_spec,
        prior_receipt_ids=tuple(prior_receipt_ids or []),
        created_at=ts,
        metadata=dict(metadata or {}),
    )


__all__ = [
    "DialecticalEvent",
    "DialecticalRuntimeError",
    "dialectical_runtime_enabled",
    "run_dialectical_loop",
]
