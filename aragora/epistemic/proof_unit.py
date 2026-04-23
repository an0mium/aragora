"""Proof-Carrying Code Unit schema and loader (DIC-19 / #6030).

Links a code path to the assumptions, evidence, claim IDs, verifier
commands, and decay/fallback policies that justify it.  Schema-only and
read-only: no runtime mutation, quarantine, or issue creation.

Field names for ``claims``, ``verifiers``, and ``decision_receipts`` are
aligned with the DIC-13 claim manifest schema and the AGT-01 CruxSet
receipt model so downstream tooling can join across all three.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_SCAN_FLAG = "ARAGORA_PROOF_UNIT_SCAN_ENABLED"

_DECAY_ACTIONS = frozenset({"report_only", "repair_required", "fail_closed"})
_FALLBACK_ACTIONS = frozenset({"fail_closed", "degrade", "report_only"})


@dataclass
class DecayPolicy:
    failed_claim: str = "report_only"
    stale_evidence: str = "report_only"
    unresolved_crux: str = "report_only"

    def validate(self) -> list[str]:
        return [
            f"decay_policy.{attr}: invalid action {val!r}"
            for attr, val in (
                ("failed_claim", self.failed_claim),
                ("stale_evidence", self.stale_evidence),
                ("unresolved_crux", self.unresolved_crux),
            )
            if val not in _DECAY_ACTIONS
        ]


@dataclass
class FallbackPolicy:
    default: str = "fail_closed"
    operator_message: str = ""

    def validate(self) -> list[str]:
        if self.default not in _FALLBACK_ACTIONS:
            return [f"fallback_policy.default: invalid action {self.default!r}"]
        return []


@dataclass
class ProofCarryingCodeUnit:
    """Code path annotated with the proof that justifies it.

    Compatible with DIC-13 claim manifests (``claims`` IDs) and
    AGT-01 CruxSet receipts (``linked_crux_ids``).
    """

    code_unit_id: str
    symbol: str
    source_path: str
    owner: str
    decision_receipts: list[str]
    claims: list[str]
    assumptions: list[str]
    verifiers: list[dict[str, str]]
    freshness_sla_hours: int
    decay_policy: DecayPolicy
    fallback_policy: FallbackPolicy
    linked_crux_ids: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.code_unit_id:
            errors.append("code_unit_id must not be empty")
        if not self.source_path:
            errors.append("source_path must not be empty")
        if self.freshness_sla_hours < 1:
            errors.append(f"freshness_sla_hours must be >= 1, got {self.freshness_sla_hours}")
        errors.extend(self.decay_policy.validate())
        errors.extend(self.fallback_policy.validate())
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_unit_id": self.code_unit_id,
            "symbol": self.symbol,
            "source_path": self.source_path,
            "owner": self.owner,
            "decision_receipts": self.decision_receipts,
            "claims": self.claims,
            "assumptions": self.assumptions,
            "verifiers": self.verifiers,
            "freshness_sla_hours": self.freshness_sla_hours,
            "decay_policy": {
                "failed_claim": self.decay_policy.failed_claim,
                "stale_evidence": self.decay_policy.stale_evidence,
                "unresolved_crux": self.decay_policy.unresolved_crux,
            },
            "fallback_policy": {
                "default": self.fallback_policy.default,
                "operator_message": self.fallback_policy.operator_message,
            },
            "linked_crux_ids": self.linked_crux_ids,
        }


def load_proof_unit(data: dict[str, Any]) -> ProofCarryingCodeUnit:
    """Deserialise a dict (e.g. parsed YAML) into a :class:`ProofCarryingCodeUnit`."""
    decay = data.get("decay_policy") or {}
    fallback = data.get("fallback_policy") or {}
    return ProofCarryingCodeUnit(
        code_unit_id=data.get("code_unit_id", ""),
        symbol=data.get("symbol", ""),
        source_path=data.get("source_path", ""),
        owner=data.get("owner", ""),
        decision_receipts=list(data.get("decision_receipts") or []),
        claims=list(data.get("claims") or []),
        assumptions=list(data.get("assumptions") or []),
        verifiers=list(data.get("verifiers") or []),
        freshness_sla_hours=int(data.get("freshness_sla_hours", 24)),
        decay_policy=DecayPolicy(
            failed_claim=decay.get("failed_claim", "report_only"),
            stale_evidence=decay.get("stale_evidence", "report_only"),
            unresolved_crux=decay.get("unresolved_crux", "report_only"),
        ),
        fallback_policy=FallbackPolicy(
            default=fallback.get("default", "fail_closed"),
            operator_message=fallback.get("operator_message", ""),
        ),
        linked_crux_ids=list(data.get("linked_crux_ids") or []),
    )


def load_proof_unit_from_yaml(path: Path) -> ProofCarryingCodeUnit:
    """Load and validate a :class:`ProofCarryingCodeUnit` from YAML.

    Raises :class:`ValueError` if validation fails.
    """
    with path.open() as fh:
        data = yaml.safe_load(fh)
    unit = load_proof_unit(data)
    errors = unit.validate()
    if errors:
        raise ValueError(f"Invalid proof unit at {path}: {errors}")
    return unit


# Module-level override avoids os.environ mutation (per the pattern in #6454).
# External callers may still set the env var; the override takes priority.
_scan_enabled_override: bool | None = None


def proof_unit_scan_enabled() -> bool:
    """Return True when the directory scanner should load proof-unit manifests.

    Checks the module-level override first, then
    ``ARAGORA_PROOF_UNIT_SCAN_ENABLED`` in the process environment.
    Default is *False*; dataclass construction is always safe regardless.
    """
    if _scan_enabled_override is not None:
        return _scan_enabled_override
    raw = str(os.environ.get(_SCAN_FLAG) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def enable_proof_unit_scan() -> None:
    """Enable the proof-unit directory scanner for the current process.

    Sets a module-level override rather than mutating ``os.environ``.
    Call :func:`reset_proof_unit_scan` to restore the default env-var-driven
    behaviour (useful in test teardown).
    """
    global _scan_enabled_override
    _scan_enabled_override = True


def reset_proof_unit_scan() -> None:
    """Clear the module-level override, reverting to env-var-driven behaviour."""
    global _scan_enabled_override
    _scan_enabled_override = None


def load_proof_units_from_dir(base: Path) -> list[ProofCarryingCodeUnit]:
    """Load all valid ``*.yaml`` proof-unit manifests under *base*.

    Returns an empty list when :func:`proof_unit_scan_enabled` is *False*
    (default), so callers never need to guard this call themselves.
    Expected validation errors (``ValueError``) are logged concisely and
    skipped.  Unexpected schema errors (``KeyError``, ``TypeError``) are
    logged with full traceback so operators can diagnose malformed files.
    """
    if not proof_unit_scan_enabled():
        return []
    units: list[ProofCarryingCodeUnit] = []
    for path in sorted(base.glob("*.yaml")):
        try:
            units.append(load_proof_unit_from_yaml(path))
        except ValueError as exc:
            logger.warning("skipping invalid proof unit %s: %s", path, exc)
        except (KeyError, TypeError):
            logger.warning("skipping malformed proof unit %s", path, exc_info=True)
    return units
