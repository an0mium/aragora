"""Lightweight convergence detection using difflib.

Tracks how similar agent proposals become across rounds. When proposals
converge beyond a configurable threshold, the debate may be ready to end --
or the trickster may need to intervene if convergence is hollow.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field


@dataclass
class ConvergenceResult:
    """Result of a convergence check across proposals."""

    converged: bool
    similarity: float  # 0-1, average pairwise similarity
    round_num: int
    pair_similarities: dict[str, float] = field(default_factory=dict)
    # Maps "agent1:agent2" -> similarity


class ConvergenceDetector:
    """Detects when agent proposals are converging using text similarity.

    Uses ``difflib.SequenceMatcher`` for stdlib-only similarity comparison.
    Compares all pairs of proposals in each round and reports the average
    similarity.

    Args:
        threshold: Similarity threshold to consider proposals converged.
            Default 0.85.

    Example::

        detector = ConvergenceDetector(threshold=0.8)

        proposals = {
            "claude": "We should use caching with Redis...",
            "gpt": "Caching via Redis would improve...",
        }

        result = detector.check(proposals, round_num=2)
        if result.converged:
            print(f"Proposals converged at {result.similarity:.0%}")
    """

    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = threshold
        self._history: list[dict[str, str]] = []

    def check(
        self,
        proposals: dict[str, str],
        round_num: int = 0,
    ) -> ConvergenceResult:
        """Check convergence across current proposals.

        Args:
            proposals: Dict of agent name -> proposal text.
            round_num: Current debate round.

        Returns:
            ConvergenceResult with similarity scores and convergence status.
        """
        self._history.append(dict(proposals))

        if len(proposals) < 2:
            return ConvergenceResult(
                converged=False,
                similarity=0.0,
                round_num=round_num,
            )

        agents = list(proposals.keys())
        pair_sims: dict[str, float] = {}

        for i, a1 in enumerate(agents):
            for a2 in agents[i + 1 :]:
                sim = self._similarity(proposals[a1], proposals[a2])
                pair_sims[f"{a1}:{a2}"] = sim

        if not pair_sims:
            return ConvergenceResult(
                converged=False,
                similarity=0.0,
                round_num=round_num,
            )

        avg_sim = sum(pair_sims.values()) / len(pair_sims)

        return ConvergenceResult(
            converged=avg_sim >= self.threshold,
            similarity=avg_sim,
            round_num=round_num,
            pair_similarities=pair_sims,
        )

    def check_trend(
        self,
        proposals: dict[str, str],
        round_num: int = 0,
    ) -> ConvergenceResult:
        """Check convergence and compare with previous round.

        Like ``check()``, but also compares each agent's current proposal
        against their proposal from the previous round to detect
        round-over-round convergence.

        Args:
            proposals: Dict of agent name -> proposal text.
            round_num: Current debate round.

        Returns:
            ConvergenceResult with pairwise *and* round-over-round sims.
        """
        result = self.check(proposals, round_num)

        # Add round-over-round comparison
        if len(self._history) >= 2:
            prev = self._history[-2]
            for agent, text in proposals.items():
                if agent in prev:
                    sim = self._similarity(prev[agent], text)
                    result.pair_similarities[f"{agent}:prev"] = sim

        return result

    @property
    def history(self) -> list[dict[str, str]]:
        """Proposal history across rounds."""
        return list(self._history)

    def reset(self) -> None:
        """Clear history for a new debate."""
        self._history.clear()

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Compute text similarity using SequenceMatcher."""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a, b).ratio()


__all__ = [
    "ConvergenceResult",
    "ConvergenceDetector",
]
