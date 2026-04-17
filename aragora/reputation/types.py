"""Core AGT-05 reputation-flow types.

Normalized shapes for claims, resolutions, and reputation deltas that
work across domains (prediction markets, debate positions, code PRs,
KM contributions, crux resolutions). These types are the in-memory
layer; on-chain anchoring via
``aragora.blockchain.contracts.reputation.ReputationRegistry`` is
deferred.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

# Canonical domain names — matches the AGT-05 spec in
# docs/plans/SKIN_IN_THE_GAME_REPUTATION.md
DOMAIN_PREDICTION_MARKET = "prediction_market"
DOMAIN_DEBATE_POSITION = "debate_position"
DOMAIN_CODE_PR = "code_pr"
DOMAIN_KM_CONTRIBUTION = "km_contribution"
DOMAIN_CRUX_RESOLUTION = "crux_resolution"

KNOWN_DOMAINS = frozenset(
    {
        DOMAIN_PREDICTION_MARKET,
        DOMAIN_DEBATE_POSITION,
        DOMAIN_CODE_PR,
        DOMAIN_KM_CONTRIBUTION,
        DOMAIN_CRUX_RESOLUTION,
    }
)

StakePolicy = Literal["forfeit_on_loss", "scaled", "refund_on_inconclusive"]
ClaimOutcome = Literal["yes", "no", "inconclusive", "invalid"]
ScoringRule = Literal["brier_proper", "binary"]


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _content_id(prefix: str, material: dict[str, Any]) -> str:
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


@dataclass(frozen=True)
class StakeableClaim:
    """A claim the agent has committed compute-credit stake to.

    - ``claim_id``: content-addressed from agent + domain + statement + resolution_source
    - ``predicted_probability`` is meaningful only for probabilistic
      claims; binary stances set it to None and rely on ``position``.
    """

    claim_id: str
    agent_id: str
    domain: str
    statement: str
    position: str
    predicted_probability: float | None
    stake_units: int
    stake_policy: StakePolicy
    resolution_source: str
    resolution_id: str
    provenance: dict[str, Any]
    created_at: str

    def __post_init__(self) -> None:
        if self.domain not in KNOWN_DOMAINS:
            raise ValueError(
                f"unknown domain: {self.domain!r}; expected one of {sorted(KNOWN_DOMAINS)}"
            )
        if not str(self.agent_id).strip():
            raise ValueError("agent_id must be non-empty")
        if not str(self.statement).strip():
            raise ValueError("statement must be non-empty")
        if not str(self.position).strip():
            raise ValueError("position must be non-empty")
        if self.stake_units < 1:
            raise ValueError("stake_units must be >= 1")
        if self.predicted_probability is not None:
            if not (0.0 <= self.predicted_probability <= 1.0):
                raise ValueError("predicted_probability must be in [0, 1] when provided")

    @classmethod
    def create(
        cls,
        *,
        agent_id: str,
        domain: str,
        statement: str,
        position: str,
        stake_units: int,
        resolution_source: str,
        resolution_id: str,
        predicted_probability: float | None = None,
        stake_policy: StakePolicy = "forfeit_on_loss",
        provenance: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> "StakeableClaim":
        provenance_dict = dict(provenance or {})
        timestamp = created_at or _utc_now_iso()
        claim_id = _content_id(
            "clm",
            {
                "agent_id": agent_id,
                "domain": domain,
                "statement": statement,
                "position": position,
                "resolution_source": resolution_source,
                "resolution_id": resolution_id,
            },
        )
        return cls(
            claim_id=claim_id,
            agent_id=agent_id,
            domain=domain,
            statement=statement,
            position=position,
            predicted_probability=(
                float(predicted_probability) if predicted_probability is not None else None
            ),
            stake_units=int(stake_units),
            stake_policy=stake_policy,
            resolution_source=resolution_source,
            resolution_id=resolution_id,
            provenance=provenance_dict,
            created_at=timestamp,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "agent_id": self.agent_id,
            "domain": self.domain,
            "statement": self.statement,
            "position": self.position,
            "predicted_probability": self.predicted_probability,
            "stake_units": self.stake_units,
            "stake_policy": self.stake_policy,
            "resolution_source": self.resolution_source,
            "resolution_id": self.resolution_id,
            "provenance": self.provenance,
            "created_at": self.created_at,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "StakeableClaim":
        return cls(
            claim_id=str(data["claim_id"]),
            agent_id=str(data["agent_id"]),
            domain=str(data["domain"]),
            statement=str(data["statement"]),
            position=str(data["position"]),
            predicted_probability=(
                float(data["predicted_probability"])
                if data.get("predicted_probability") is not None
                else None
            ),
            stake_units=int(data["stake_units"]),
            stake_policy=data.get("stake_policy") or "forfeit_on_loss",
            resolution_source=str(data["resolution_source"]),
            resolution_id=str(data["resolution_id"]),
            provenance=dict(data.get("provenance") or {}),
            created_at=str(data["created_at"]),
        )


@dataclass(frozen=True)
class ResolvedClaim:
    """A claim's resolution event — what actually happened.

    ``outcome`` is the unified ternary shape used across domains;
    ``evidence`` carries the raw payload that decided the outcome
    (GitHub state for markets, verifier result for cruxes, etc.).
    """

    claim_id: str
    outcome: ClaimOutcome
    resolved_at: str
    resolution_source: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome not in {"yes", "no", "inconclusive", "invalid"}:
            raise ValueError(f"unsupported outcome: {self.outcome!r}")
        if not str(self.claim_id).strip():
            raise ValueError("claim_id must be non-empty")

    def to_json(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "outcome": self.outcome,
            "resolved_at": self.resolved_at,
            "resolution_source": self.resolution_source,
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ResolvedClaim":
        return cls(
            claim_id=str(data["claim_id"]),
            outcome=data["outcome"],
            resolved_at=str(data["resolved_at"]),
            resolution_source=str(data["resolution_source"]),
            evidence=dict(data.get("evidence") or {}),
        )


@dataclass(frozen=True)
class ReputationDelta:
    """The computed reputation change for one resolved claim.

    - ``delta`` is signed and stake-weighted, typically in the range
      ``[-stake_units, +stake_units]`` for proper Brier and ``±stake_units``
      for binary scoring.
    - ``decay_half_life_days`` is forward-compatible with the AGT-05
      90-day rolling window spec; consumers apply the decay on read.
    - ``reason`` carries the full settlement provenance so the delta
      can be verified or re-opened later.
    """

    delta_id: str
    agent_id: str
    domain: str
    claim_id: str
    resolution_id: str
    delta: float
    scoring_rule: ScoringRule
    applied_at: str
    decay_half_life_days: float | None
    reason: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "delta_id": self.delta_id,
            "agent_id": self.agent_id,
            "domain": self.domain,
            "claim_id": self.claim_id,
            "resolution_id": self.resolution_id,
            "delta": round(self.delta, 6),
            "scoring_rule": self.scoring_rule,
            "applied_at": self.applied_at,
            "decay_half_life_days": self.decay_half_life_days,
            "reason": dict(self.reason),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ReputationDelta":
        return cls(
            delta_id=str(data["delta_id"]),
            agent_id=str(data["agent_id"]),
            domain=str(data["domain"]),
            claim_id=str(data["claim_id"]),
            resolution_id=str(data["resolution_id"]),
            delta=float(data["delta"]),
            scoring_rule=data["scoring_rule"],
            applied_at=str(data["applied_at"]),
            decay_half_life_days=(
                float(data["decay_half_life_days"])
                if data.get("decay_half_life_days") is not None
                else None
            ),
            reason=dict(data.get("reason") or {}),
        )


__all__ = [
    "DOMAIN_CODE_PR",
    "DOMAIN_CRUX_RESOLUTION",
    "DOMAIN_DEBATE_POSITION",
    "DOMAIN_KM_CONTRIBUTION",
    "DOMAIN_PREDICTION_MARKET",
    "KNOWN_DOMAINS",
    "ClaimOutcome",
    "ReputationDelta",
    "ResolvedClaim",
    "ScoringRule",
    "StakePolicy",
    "StakeableClaim",
]
