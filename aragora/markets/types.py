"""Core types for synthetic GitHub prediction markets.

These types intentionally use plain dataclasses with JSON-serializable
fields so storage can be a flat JSONL file. The schema is the planning
contract — once AGT-05 (skin-in-the-game reputation) wires resolution
events into the ERC-8004 reputation registry, these types become the
on-disk shape that survives schema review.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

QuestionKind = Literal["pr_merge", "issue_close", "ci_pass"]
ResolutionOutcome = Literal["yes", "no", "inconclusive"]

# Per-market position cap from the AGT-04 plan
MAX_POSITION_STAKE = 100

# Stale market expiry from the AGT-04 plan
MAX_RESOLUTION_WINDOW_DAYS = 90

_REPO_PATTERN = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _validate_repo(repo: str) -> None:
    if not _REPO_PATTERN.match(repo or ""):
        raise ValueError(f"invalid repo identifier: {repo!r}; expected owner/name")


def _market_id(question_kind: QuestionKind, target: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"kind": question_kind, "target": target},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"mkt_{question_kind}_{digest}"


@dataclass(frozen=True)
class Market:
    """A synthetic prediction market over a verifiable GitHub event.

    ``market_id`` is content-addressed from ``question_kind`` + ``target``
    so identical questions deduplicate. Markets are immutable once created.
    """

    market_id: str
    question_kind: QuestionKind
    target: dict[str, Any]
    description: str
    created_at: str
    expires_at: str

    @classmethod
    def create(
        cls,
        *,
        question_kind: QuestionKind,
        target: dict[str, Any],
        description: str,
        resolution_window_days: int,
        created_at: datetime | None = None,
    ) -> "Market":
        if resolution_window_days < 1 or resolution_window_days > MAX_RESOLUTION_WINDOW_DAYS:
            raise ValueError(f"resolution_window_days must be in [1, {MAX_RESOLUTION_WINDOW_DAYS}]")
        repo = str(target.get("repo") or "")
        _validate_repo(repo)
        if question_kind in {"pr_merge", "issue_close"}:
            number = target.get("number")
            if not isinstance(number, int) or number < 1:
                raise ValueError(f"target.number must be a positive int for {question_kind}")
        elif question_kind == "ci_pass":
            ref = str(target.get("ref") or "").strip()
            if not ref:
                raise ValueError("target.ref must be a non-empty string for ci_pass")
        else:  # pragma: no cover - exhaustiveness
            raise ValueError(f"unsupported question_kind: {question_kind!r}")

        created = created_at or _utc_now()
        expires = created + timedelta(days=resolution_window_days)
        normalized_target = dict(target)
        return cls(
            market_id=_market_id(question_kind, normalized_target),
            question_kind=question_kind,
            target=normalized_target,
            description=description.strip(),
            created_at=_iso(created),
            expires_at=_iso(expires),
        )

    def is_expired(self, *, now: datetime | None = None) -> bool:
        return (now or _utc_now()) >= _from_iso(self.expires_at)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Market":
        return cls(
            market_id=str(data["market_id"]),
            question_kind=data["question_kind"],
            target=dict(data["target"]),
            description=str(data.get("description") or ""),
            created_at=str(data["created_at"]),
            expires_at=str(data["expires_at"]),
        )


@dataclass(frozen=True)
class MarketPosition:
    """An agent's prediction on a market, with stake.

    ``probability`` is the agent's predicted P(YES) in ``[0, 1]``.
    ``stake`` is in internal credits (compute-budget units) and bounded by
    :data:`MAX_POSITION_STAKE`.
    """

    position_id: str
    market_id: str
    agent_id: str
    probability: float
    stake: int
    submitted_at: str
    rationale: str = ""

    @classmethod
    def create(
        cls,
        *,
        market_id: str,
        agent_id: str,
        probability: float,
        stake: int,
        submitted_at: datetime | None = None,
        rationale: str = "",
    ) -> "MarketPosition":
        if not (0.0 <= probability <= 1.0):
            raise ValueError("probability must be in [0, 1]")
        if stake < 1 or stake > MAX_POSITION_STAKE:
            raise ValueError(f"stake must be in [1, {MAX_POSITION_STAKE}]")
        agent = (agent_id or "").strip()
        if not agent:
            raise ValueError("agent_id must be non-empty")
        market = (market_id or "").strip()
        if not market:
            raise ValueError("market_id must be non-empty")
        ts = submitted_at or _utc_now()
        material = f"{market}|{agent}|{_iso(ts)}|{probability:.6f}|{stake}"
        position_id = "pos_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
        return cls(
            position_id=position_id,
            market_id=market,
            agent_id=agent,
            probability=float(probability),
            stake=int(stake),
            submitted_at=_iso(ts),
            rationale=rationale.strip(),
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "MarketPosition":
        return cls(
            position_id=str(data["position_id"]),
            market_id=str(data["market_id"]),
            agent_id=str(data["agent_id"]),
            probability=float(data["probability"]),
            stake=int(data["stake"]),
            submitted_at=str(data["submitted_at"]),
            rationale=str(data.get("rationale") or ""),
        )


@dataclass(frozen=True)
class ResolutionEvent:
    """A resolved market: outcome plus the GitHub evidence that decided it.

    ``outcome`` follows the AGT-05 ResolutionEvent contract (see
    ``docs/plans/SKIN_IN_THE_GAME_REPUTATION.md``). ``inconclusive`` covers
    expired markets where the underlying state (e.g. PR still open at
    expiry) does not provide a binary answer; the ``evidence`` payload
    explains why.
    """

    market_id: str
    outcome: ResolutionOutcome
    resolved_at: str
    resolution_source: str
    evidence: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def yes(
        cls,
        *,
        market_id: str,
        resolution_source: str,
        evidence: dict[str, Any] | None = None,
        resolved_at: datetime | None = None,
    ) -> "ResolutionEvent":
        return cls(
            market_id=market_id,
            outcome="yes",
            resolved_at=_iso(resolved_at or _utc_now()),
            resolution_source=resolution_source,
            evidence=dict(evidence or {}),
        )

    @classmethod
    def no(
        cls,
        *,
        market_id: str,
        resolution_source: str,
        evidence: dict[str, Any] | None = None,
        resolved_at: datetime | None = None,
    ) -> "ResolutionEvent":
        return cls(
            market_id=market_id,
            outcome="no",
            resolved_at=_iso(resolved_at or _utc_now()),
            resolution_source=resolution_source,
            evidence=dict(evidence or {}),
        )

    @classmethod
    def inconclusive(
        cls,
        *,
        market_id: str,
        resolution_source: str,
        evidence: dict[str, Any] | None = None,
        resolved_at: datetime | None = None,
    ) -> "ResolutionEvent":
        return cls(
            market_id=market_id,
            outcome="inconclusive",
            resolved_at=_iso(resolved_at or _utc_now()),
            resolution_source=resolution_source,
            evidence=dict(evidence or {}),
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ResolutionEvent":
        return cls(
            market_id=str(data["market_id"]),
            outcome=data["outcome"],
            resolved_at=str(data["resolved_at"]),
            resolution_source=str(data["resolution_source"]),
            evidence=dict(data.get("evidence") or {}),
        )
