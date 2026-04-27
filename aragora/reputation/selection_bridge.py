"""AGT-05 dispatch-eligibility bridge — ReputationStore → CalibrationScorer.

Adapts the in-memory :class:`~aragora.reputation.store.ReputationStore`
to the ``CalibrationScorer`` protocol expected by
:class:`~aragora.debate.team_selector.TeamSelector`, enabling reputation
scores from resolved predictions and crux outcomes to influence agent
selection.

Gating
------
The bridge is **dormant by default**.  :meth:`ReputationCalibrationBridge.get_brier_score`
returns ``neutral_brier`` (0.5) for every agent unless
``ARAGORA_REPUTATION_FLOW_ENABLED`` is set *and* the agent has at least
``min_samples`` recorded deltas in the store.

Score mapping
-------------
``running_score`` from :class:`~aragora.reputation.store.ReputationStore`
is unbounded and signed (positive = good, negative = bad).  It is mapped
to a pseudo-Brier in ``[low_clip, high_clip]`` (defaults ``[0.1, 0.9]``)
via::

    brier = neutral_brier - clamp(score / score_scale, -(neutral_brier - low_clip), high_clip - neutral_brier)

A higher running score → lower pseudo-Brier (better for team selection).
A lower running score → higher pseudo-Brier (penalised by team selection).

Domain filtering
----------------
When a ``domain`` argument is supplied to :meth:`get_brier_score`, only
deltas whose ``domain`` field matches are included in the score
computation.  Agents with fewer than ``min_samples`` matching deltas
return ``neutral_brier``.

Out of scope
------------
- Wiring the bridge into ``TeamSelectionConfig`` is the follow-on slice
  (``enable_agt05_reputation_selection: bool = False``).
- On-chain anchoring lives in :mod:`aragora.reputation.anchor`.
- Persistence and JSONL load/save live in :mod:`aragora.reputation.store`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.reputation.store import ReputationStore
    from aragora.reputation.types import ReputationDelta

__all__ = [
    "NEUTRAL_BRIER",
    "ReputationBridgeConfig",
    "ReputationCalibrationBridge",
    "reputation_flow_enabled",
]

NEUTRAL_BRIER: float = 0.5


def reputation_flow_enabled() -> bool:
    """Return True when ``ARAGORA_REPUTATION_FLOW_ENABLED`` is truthy."""
    raw = str(os.environ.get("ARAGORA_REPUTATION_FLOW_ENABLED") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


class ReputationBridgeConfig:
    """Tuning knobs for :class:`ReputationCalibrationBridge`.

    Parameters
    ----------
    min_samples:
        Minimum number of deltas (after optional domain filtering) before
        returning a non-neutral score.  Agents with fewer deltas than this
        threshold return ``neutral_brier``.
    score_scale:
        Running scores are divided by this value before being clamped.
        A scale of 100 means a score of +100 maps to ``neutral − max_shift``.
    neutral_brier:
        The Brier score returned when the bridge is disabled or an agent
        has insufficient data.  Should be 0.5 for an uninformative predictor.
    low_clip:
        Minimum pseudo-Brier returned.  Prevents perfect suppression of
        any single agent.
    high_clip:
        Maximum pseudo-Brier returned.  Prevents complete exclusion.
    apply_decay:
        When True, the store's exponential time-decay is applied before
        computing the score.
    """

    def __init__(
        self,
        *,
        min_samples: int = 5,
        score_scale: float = 100.0,
        neutral_brier: float = NEUTRAL_BRIER,
        low_clip: float = 0.1,
        high_clip: float = 0.9,
        apply_decay: bool = True,
    ) -> None:
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1; got {min_samples}")
        if score_scale <= 0:
            raise ValueError(f"score_scale must be > 0; got {score_scale}")
        if not (0.0 <= low_clip < neutral_brier < high_clip <= 1.0):
            raise ValueError(
                f"must satisfy 0 <= low_clip < neutral_brier < high_clip <= 1; "
                f"got low_clip={low_clip}, neutral_brier={neutral_brier}, "
                f"high_clip={high_clip}"
            )
        self.min_samples = min_samples
        self.score_scale = score_scale
        self.neutral_brier = neutral_brier
        self.low_clip = low_clip
        self.high_clip = high_clip
        self.apply_decay = apply_decay


class ReputationCalibrationBridge:
    """AGT-05 dispatch-eligibility adapter.

    Implements the ``CalibrationScorer`` protocol so the team_selector
    can incorporate skin-in-the-game reputation when selecting debate
    participants.

    The bridge is strictly read-only with respect to the store; it never
    records or mutates deltas.

    Parameters
    ----------
    store:
        The shared :class:`~aragora.reputation.store.ReputationStore`.
        When ``None`` the bridge is always dormant and returns
        ``config.neutral_brier``.
    config:
        Tuning parameters.  Defaults to :class:`ReputationBridgeConfig`
        with its default values.
    """

    def __init__(
        self,
        store: "ReputationStore | None" = None,
        config: ReputationBridgeConfig | None = None,
    ) -> None:
        self._store = store
        self._cfg = config or ReputationBridgeConfig()

    # ------------------------------------------------------------------
    # CalibrationScorer protocol
    # ------------------------------------------------------------------

    def get_brier_score(self, agent_name: str, domain: str | None = None) -> float:
        """Return a pseudo-Brier score for *agent_name*.

        Returns ``neutral_brier`` when:
        - ``ARAGORA_REPUTATION_FLOW_ENABLED`` is not set
        - the store is ``None``
        - the agent has fewer than ``min_samples`` deltas (after domain
          filtering)

        Otherwise maps the running score to ``[low_clip, high_clip]``.
        Lower is better (follows Brier score convention).
        """
        if not reputation_flow_enabled() or self._store is None:
            return self._cfg.neutral_brier

        deltas = self._relevant_deltas(agent_name, domain)
        if len(deltas) < self._cfg.min_samples:
            return self._cfg.neutral_brier

        score = self._score_from_deltas(deltas)
        return self._map_score_to_brier(score)

    def get_brier_scores_batch(
        self, agent_names: list[str], domain: str | None = None
    ) -> dict[str, float]:
        """Return pseudo-Brier scores for multiple agents.

        Falls back to calling :meth:`get_brier_score` individually when
        the bridge is disabled; avoids repeated store lookups when active.
        """
        if not reputation_flow_enabled() or self._store is None:
            return {name: self._cfg.neutral_brier for name in agent_names}
        return {name: self.get_brier_score(name, domain) for name in agent_names}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _relevant_deltas(self, agent_id: str, domain: str | None) -> "list[ReputationDelta]":
        if self._store is None:
            return []
        deltas = self._store.deltas_for(agent_id)
        if domain is not None:
            deltas = [d for d in deltas if d.domain == domain]
        return deltas

    def _score_from_deltas(self, deltas: "list[ReputationDelta]") -> float:
        """Sum deltas with optional exponential decay."""
        if not self._cfg.apply_decay:
            return sum(d.delta for d in deltas)

        from datetime import UTC, datetime
        import math

        now = datetime.now(tz=UTC)
        total = 0.0
        for d in deltas:
            if d.decay_half_life_days is None:
                weight = 1.0
            else:
                try:
                    applied = datetime.fromisoformat(
                        d.applied_at.replace("Z", "+00:00")
                    ).astimezone(UTC)
                    age_days = max(0.0, (now - applied).total_seconds() / 86_400.0)
                    weight = math.exp(-math.log(2.0) * age_days / d.decay_half_life_days)
                except (ValueError, TypeError):
                    weight = 1.0
            total += d.delta * weight
        return total

    def _map_score_to_brier(self, score: float) -> float:
        """Map an unbounded signed score to a pseudo-Brier in [low_clip, high_clip]."""
        cfg = self._cfg
        max_shift_up = cfg.neutral_brier - cfg.low_clip
        max_shift_down = cfg.high_clip - cfg.neutral_brier
        normalised = score / cfg.score_scale
        if normalised > 0:
            shift = min(normalised, max_shift_up)
        else:
            shift = max(normalised, -max_shift_down)
        return cfg.neutral_brier - shift
