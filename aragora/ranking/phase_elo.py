"""Phase-Tagged ELO System.

Extends agent ELO rankings with phase-level granularity.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

DEFAULT_ELO = 1500
K_FACTOR = 32


@dataclass
class PhaseRating:
    agent_name: str
    domain: str
    phase: str
    rating: float = DEFAULT_ELO
    matches: int = 0
    wins: int = 0
    losses: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.matches, 1)

    @property
    def domain_phase_key(self) -> str:
        return f"{self.domain}:{self.phase}"


@dataclass
class PhaseMatchResult:
    agent_name: str
    domain: str
    phase: str
    won: bool
    influence_score: float = 0.0
    opponent_rating: float = DEFAULT_ELO


class PhaseELOTracker:
    def __init__(self, k_factor: float = K_FACTOR) -> None:
        self._ratings: dict[str, dict[str, PhaseRating]] = defaultdict(dict)
        self._k_factor = k_factor
        self._history: list[PhaseMatchResult] = []

    def update_domain_elo(
        self,
        agent_name: str,
        domain_phase: str,
        won: bool,
        opponent_rating: float = DEFAULT_ELO,
        influence_score: float = 0.0,
    ) -> float:
        parts = domain_phase.split(":", 1)
        domain = parts[0]
        phase = parts[1] if len(parts) > 1 else "general"
        key = f"{domain}:{phase}"
        ratings = self._ratings[agent_name]
        if key not in ratings:
            ratings[key] = PhaseRating(agent_name=agent_name, domain=domain, phase=phase)
        pr = ratings[key]
        expected = 1.0 / (1.0 + math.pow(10, (opponent_rating - pr.rating) / 400))
        actual = 1.0 if won else 0.0
        pr.rating += self._k_factor * (actual - expected)
        pr.matches += 1
        if won:
            pr.wins += 1
        else:
            pr.losses += 1
        self._history.append(
            PhaseMatchResult(
                agent_name=agent_name,
                domain=domain,
                phase=phase,
                won=won,
                influence_score=influence_score,
                opponent_rating=opponent_rating,
            )
        )
        return pr.rating

    def get_rating(self, agent_name: str, domain: str, phase: str) -> PhaseRating | None:
        return self._ratings.get(agent_name, {}).get(f"{domain}:{phase}")

    def get_agent_profile(self, agent_name: str) -> dict[str, PhaseRating]:
        return dict(self._ratings.get(agent_name, {}))

    def get_best_agents_for_phase(
        self, domain: str, phase: str, limit: int = 5
    ) -> list[PhaseRating]:
        key = f"{domain}:{phase}"
        candidates = [r for a, rs in self._ratings.items() for k, r in rs.items() if k == key]
        candidates.sort(key=lambda r: r.rating, reverse=True)
        return candidates[:limit]

    def get_phase_leaderboard(self, phase: str) -> list[PhaseRating]:
        results = [pr for rs in self._ratings.values() for pr in rs.values() if pr.phase == phase]
        results.sort(key=lambda r: r.rating, reverse=True)
        return results

    def get_improvement_trend(
        self, agent_name: str, domain: str, phase: str, window: int = 10
    ) -> list[float]:
        matching = [
            h.influence_score
            for h in self._history
            if h.agent_name == agent_name and h.domain == domain and h.phase == phase
        ]
        return matching[-window:]

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for agent_name, ratings in self._ratings.items():
            result[agent_name] = {
                key: {
                    "domain": pr.domain,
                    "phase": pr.phase,
                    "rating": round(pr.rating, 1),
                    "matches": pr.matches,
                    "win_rate": round(pr.win_rate, 3),
                }
                for key, pr in ratings.items()
            }
        return result
