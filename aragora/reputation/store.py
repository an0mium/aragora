"""AGT-05 reputation store — per-agent ledger of ReputationDeltas.

This module is the in-memory accumulation layer that sits between the
settlement function (:func:`aragora.reputation.settlement.settle_claim`)
and downstream consumers such as the team_selector dispatch-eligibility
filter (wired in a follow-on PR).

Persistence: an optional append-only JSONL path.  When ``path`` is
supplied, each recorded delta is written as one JSON line immediately
after it is added to the in-memory index.  The store can be rebuilt from
that file at startup via :meth:`ReputationStore.load_from_file`.

Decay: when a :class:`~aragora.reputation.types.ReputationDelta` was
created with a ``decay_half_life_days`` value, :meth:`get_score` applies
exponential decay so that older deltas contribute less.  Callers that
want raw un-decayed sums can pass ``apply_decay=False``.

Gating: the store is always constructable, but :func:`store_enabled`
checks ``ARAGORA_REPUTATION_FLOW_ENABLED`` so callers that respect the
AGT-05 feature flag have a single check point.

Out of scope for this PR:
- Team-selector wiring (follow-on: ``enable_agt05_reputation_selection``
  flag on TeamSelectorConfig).
- On-chain anchoring (lives in :mod:`aragora.reputation.anchor`).
- CruxSet resolution bridge (deferred to DIC-17 landing).
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aragora.reputation.types import ReputationDelta

logger = logging.getLogger(__name__)


def store_enabled() -> bool:
    """Return True when the AGT-05 reputation-flow flag is set."""
    raw = str(os.environ.get("ARAGORA_REPUTATION_FLOW_ENABLED") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def enable_store() -> None:
    """Enable the AGT-05 reputation flow for the current process."""
    os.environ["ARAGORA_REPUTATION_FLOW_ENABLED"] = "1"


class ReputationStoreError(RuntimeError):
    """Raised when a delta cannot be recorded."""


@dataclass
class AgentScore:
    """Aggregated reputation summary for one agent."""

    agent_id: str
    running_score: float
    delta_count: int
    domains: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "running_score": round(self.running_score, 6),
            "delta_count": self.delta_count,
            "domains": self.domains,
        }


def _decay_weight(delta: ReputationDelta, now: datetime) -> float:
    """Compute the time-decay multiplier for *delta*.

    Returns 1.0 when ``decay_half_life_days`` is None (no decay).
    Uses exponential half-life: ``weight = 2 ** (-age_days / half_life)``.
    """
    if delta.decay_half_life_days is None:
        return 1.0
    try:
        applied = datetime.fromisoformat(delta.applied_at.replace("Z", "+00:00")).astimezone(UTC)
    except (ValueError, TypeError):
        return 1.0
    age_days = max(0.0, (now - applied).total_seconds() / 86_400.0)
    return math.exp(-math.log(2.0) * age_days / delta.decay_half_life_days)


class ReputationStore:
    """In-memory per-agent reputation ledger with optional JSONL persistence.

    Thread-safety: not thread-safe; wrap with a lock for concurrent use.

    Typical usage::

        store = ReputationStore()
        delta = settle_claim(claim, resolved)
        store.record_delta(delta)
        score = store.get_score("agent-id")
    """

    def __init__(self, path: Path | None = None) -> None:
        self._deltas: dict[str, list[ReputationDelta]] = defaultdict(list)
        self._path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def record_delta(self, delta: ReputationDelta) -> None:
        """Record a settled reputation delta.

        Raises :class:`ReputationStoreError` when ``delta.agent_id`` is
        empty; all other validation has already been applied by
        :func:`~aragora.reputation.settlement.settle_claim`.
        """
        if not str(delta.agent_id).strip():
            raise ReputationStoreError("delta.agent_id must be non-empty")
        if self._path is not None:
            self._append_to_file(delta)
        self._deltas[delta.agent_id].append(delta)

    def _append_to_file(self, delta: ReputationDelta) -> None:
        try:
            with self._path.open("a", encoding="utf-8") as fh:  # type: ignore[union-attr]
                fh.write(json.dumps(delta.to_json(), sort_keys=True) + "\n")
        except OSError as exc:
            raise ReputationStoreError(
                f"could not persist reputation delta to {self._path}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_score(self, agent_id: str, *, apply_decay: bool = True) -> float:
        """Return the running reputation score for *agent_id*.

        Score is the sum of all recorded deltas, optionally weighted by
        exponential time-decay.  Agents with no recorded deltas return 0.0.
        The score is unbounded; callers normalise it for dispatch-eligibility.
        """
        deltas = self._deltas.get(agent_id, [])
        if not deltas:
            return 0.0
        if not apply_decay:
            return sum(d.delta for d in deltas)
        now = datetime.now(tz=UTC)
        return sum(d.delta * _decay_weight(d, now) for d in deltas)

    def agent_ids(self) -> list[str]:
        """Return sorted list of agent IDs that have at least one delta."""
        return sorted(self._deltas.keys())

    def deltas_for(self, agent_id: str) -> list[ReputationDelta]:
        """Return a copy of the delta history for *agent_id*."""
        return list(self._deltas.get(agent_id, []))

    def agent_score(self, agent_id: str, *, apply_decay: bool = True) -> AgentScore:
        """Return a structured score summary for *agent_id*."""
        deltas = self._deltas.get(agent_id, [])
        domains = sorted({d.domain for d in deltas})
        return AgentScore(
            agent_id=agent_id,
            running_score=self.get_score(agent_id, apply_decay=apply_decay),
            delta_count=len(deltas),
            domains=domains,
        )

    def all_scores(self, *, apply_decay: bool = True) -> list[AgentScore]:
        """Return per-agent score summaries, sorted descending by score."""
        scores = [self.agent_score(aid, apply_decay=apply_decay) for aid in self._deltas]
        return sorted(scores, key=lambda s: s.running_score, reverse=True)

    def __len__(self) -> int:
        """Return the total number of recorded deltas across all agents."""
        return sum(len(v) for v in self._deltas.values())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load_from_file(cls, path: Path) -> "ReputationStore":
        """Reconstruct a store from an existing JSONL file.

        Lines that cannot be parsed are skipped with a warning so a
        corrupt tail does not discard earlier good records.
        """
        store = cls(path=path)
        if not path.exists():
            return store
        with path.open(encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                    delta = ReputationDelta.from_json(payload)
                    store._deltas[delta.agent_id].append(delta)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "ReputationStore: skipping invalid line %d in %s: %s",
                        lineno,
                        path,
                        exc,
                    )
        return store


__all__ = [
    "AgentScore",
    "ReputationStore",
    "ReputationStoreError",
    "enable_store",
    "store_enabled",
]
