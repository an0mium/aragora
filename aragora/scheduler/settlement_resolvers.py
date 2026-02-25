"""Settlement resolver backends for epistemic hygiene outcomes.

Resolver tiers:
- human: requires explicit adjudication evidence
- deterministic: evaluates rule-based outcomes from structured observations
- oracle: verifies signed attestation payloads
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "correct", "confirmed"}:
            return True
        if normalized in {"false", "0", "no", "incorrect", "falsified"}:
            return False
    return None


@dataclass
class ResolverDecision:
    """Outcome returned by a resolver backend."""

    resolved: bool
    outcome: bool | None = None
    resolver_type: str = ""
    resolver_id: str = ""
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved": self.resolved,
            "outcome": self.outcome,
            "resolver_type": self.resolver_type,
            "resolver_id": self.resolver_id,
            "reason": self.reason,
            "evidence": self.evidence,
        }


class BaseSettlementResolver(ABC):
    """Abstract resolver backend."""

    resolver_type: str
    resolver_id: str

    @abstractmethod
    def resolve(
        self,
        *,
        receipt_data: dict[str, Any],
        settlement: dict[str, Any],
        now: datetime,
    ) -> ResolverDecision:
        """Resolve settlement outcome if possible."""


class HumanSettlementResolver(BaseSettlementResolver):
    resolver_type = "human"
    resolver_id = "human_adjudication"

    def resolve(
        self,
        *,
        receipt_data: dict[str, Any],
        settlement: dict[str, Any],
        now: datetime,
    ) -> ResolverDecision:
        _ = receipt_data
        explicit = _coerce_bool(settlement.get("human_outcome"))
        if explicit is None:
            explicit = _coerce_bool(settlement.get("adjudicated_outcome"))
        if explicit is None:
            return ResolverDecision(
                resolved=False,
                resolver_type=self.resolver_type,
                resolver_id=self.resolver_id,
                reason="waiting_for_human_adjudication",
                evidence={"requested_at": now.isoformat()},
            )
        return ResolverDecision(
            resolved=True,
            outcome=explicit,
            resolver_type=self.resolver_type,
            resolver_id=self.resolver_id,
            reason="human_adjudication_submitted",
            evidence={"resolved_at": now.isoformat()},
        )


class DeterministicSettlementResolver(BaseSettlementResolver):
    resolver_type = "deterministic"
    resolver_id = "deterministic_rule_engine"

    def resolve(
        self,
        *,
        receipt_data: dict[str, Any],
        settlement: dict[str, Any],
        now: datetime,
    ) -> ResolverDecision:
        _ = receipt_data
        explicit = _coerce_bool(settlement.get("deterministic_outcome"))
        if explicit is not None:
            return ResolverDecision(
                resolved=True,
                outcome=explicit,
                resolver_type=self.resolver_type,
                resolver_id=self.resolver_id,
                reason="explicit_deterministic_outcome",
                evidence={"resolved_at": now.isoformat()},
            )

        rule = settlement.get("deterministic_rule")
        if not isinstance(rule, dict):
            return ResolverDecision(
                resolved=False,
                resolver_type=self.resolver_type,
                resolver_id=self.resolver_id,
                reason="missing_deterministic_rule",
            )

        observed = rule.get("observed")
        target = rule.get("target")
        operator = str(rule.get("operator") or "").strip()
        if observed is None or target is None or not operator:
            return ResolverDecision(
                resolved=False,
                resolver_type=self.resolver_type,
                resolver_id=self.resolver_id,
                reason="incomplete_deterministic_rule",
                evidence={"rule": rule},
            )

        try:
            observed_f = float(observed)
            target_f = float(target)
        except (TypeError, ValueError):
            return ResolverDecision(
                resolved=False,
                resolver_type=self.resolver_type,
                resolver_id=self.resolver_id,
                reason="non_numeric_deterministic_rule",
                evidence={"rule": rule},
            )

        comparison_map = {
            ">": observed_f > target_f,
            ">=": observed_f >= target_f,
            "<": observed_f < target_f,
            "<=": observed_f <= target_f,
            "==": observed_f == target_f,
            "!=": observed_f != target_f,
        }
        if operator not in comparison_map:
            return ResolverDecision(
                resolved=False,
                resolver_type=self.resolver_type,
                resolver_id=self.resolver_id,
                reason="unsupported_deterministic_operator",
                evidence={"operator": operator},
            )

        return ResolverDecision(
            resolved=True,
            outcome=bool(comparison_map[operator]),
            resolver_type=self.resolver_type,
            resolver_id=self.resolver_id,
            reason="deterministic_rule_evaluated",
            evidence={
                "operator": operator,
                "observed": observed_f,
                "target": target_f,
                "resolved_at": now.isoformat(),
            },
        )


class OracleSettlementResolver(BaseSettlementResolver):
    resolver_type = "oracle"
    resolver_id = "signed_oracle_attestation"

    def resolve(
        self,
        *,
        receipt_data: dict[str, Any],
        settlement: dict[str, Any],
        now: datetime,
    ) -> ResolverDecision:
        _ = receipt_data
        attestation = settlement.get("oracle_attestation")
        if not isinstance(attestation, dict):
            return ResolverDecision(
                resolved=False,
                resolver_type=self.resolver_type,
                resolver_id=self.resolver_id,
                reason="missing_oracle_attestation",
            )

        outcome = _coerce_bool(attestation.get("outcome"))
        source = str(attestation.get("source") or "").strip()
        signed_at = str(attestation.get("signed_at") or "").strip()
        signature = str(attestation.get("signature") or "").strip()
        if outcome is None or not source or not signed_at or not signature:
            return ResolverDecision(
                resolved=False,
                resolver_type=self.resolver_type,
                resolver_id=self.resolver_id,
                reason="incomplete_oracle_attestation",
                evidence={"source": source, "signed_at": signed_at},
            )

        secret = os.environ.get("ARAGORA_ORACLE_HMAC_SECRET")
        if not secret:
            return ResolverDecision(
                resolved=False,
                resolver_type=self.resolver_type,
                resolver_id=self.resolver_id,
                reason="missing_oracle_secret",
            )

        canonical = json.dumps(
            {
                "outcome": outcome,
                "source": source,
                "signed_at": signed_at,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        expected_sig = hmac.new(
            secret.encode("utf-8"),
            canonical.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            return ResolverDecision(
                resolved=False,
                resolver_type=self.resolver_type,
                resolver_id=self.resolver_id,
                reason="invalid_oracle_signature",
                evidence={"source": source},
            )

        return ResolverDecision(
            resolved=True,
            outcome=outcome,
            resolver_type=self.resolver_type,
            resolver_id=self.resolver_id,
            reason="oracle_attestation_verified",
            evidence={"source": source, "signed_at": signed_at, "verified_at": now.isoformat()},
        )


class SettlementResolverRegistry:
    """Registry for resolver backends keyed by resolver type."""

    def __init__(self) -> None:
        self._resolvers: dict[str, BaseSettlementResolver] = {
            "human": HumanSettlementResolver(),
            "deterministic": DeterministicSettlementResolver(),
            "oracle": OracleSettlementResolver(),
        }

    def resolve(
        self,
        resolver_type: str,
        *,
        receipt_data: dict[str, Any],
        settlement: dict[str, Any],
        now: datetime,
    ) -> ResolverDecision:
        resolver = self._resolvers.get(str(resolver_type).strip().lower())
        if resolver is None:
            return ResolverDecision(
                resolved=False,
                resolver_type=str(resolver_type),
                resolver_id="unknown",
                reason="unknown_resolver_type",
            )
        return resolver.resolve(receipt_data=receipt_data, settlement=settlement, now=now)


_registry: SettlementResolverRegistry | None = None


def get_settlement_resolver_registry() -> SettlementResolverRegistry:
    global _registry
    if _registry is None:
        _registry = SettlementResolverRegistry()
    return _registry


def set_settlement_resolver_registry(registry: SettlementResolverRegistry | None) -> None:
    global _registry
    _registry = registry


__all__ = [
    "ResolverDecision",
    "BaseSettlementResolver",
    "HumanSettlementResolver",
    "DeterministicSettlementResolver",
    "OracleSettlementResolver",
    "SettlementResolverRegistry",
    "get_settlement_resolver_registry",
    "set_settlement_resolver_registry",
]
