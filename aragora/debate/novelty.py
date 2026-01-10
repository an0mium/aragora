"""
Novelty tracking for multi-agent debate proposals.

Tracks semantic distance from prior proposals to prevent convergence to mediocrity.
When agents propose ideas too similar to what's already been said, novelty scores
drop and can trigger trickster interventions.

Uses the same similarity backends as convergence detection:
1. SentenceTransformer (best accuracy)
2. TF-IDF (good accuracy)
3. Jaccard (fallback)

Novelty score = 1 - max(similarity to any prior proposal)
- High novelty (>0.7): Fresh, divergent ideas
- Medium novelty (0.3-0.7): Building on prior ideas
- Low novelty (<0.15): Too similar, may need intervention
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from .convergence import SimilarityBackend, get_similarity_backend

logger = logging.getLogger(__name__)


@dataclass
class NoveltyScore:
    """Novelty measurement for a single proposal."""

    agent: str
    round_num: int
    novelty: float  # 1 - max_similarity to prior proposals (0-1, higher = more novel)
    max_similarity: float  # Highest similarity to any prior proposal
    most_similar_to: Optional[str] = None  # Agent whose proposal was most similar
    prior_proposals_count: int = 0

    def is_low_novelty(self, threshold: float = 0.15) -> bool:
        """Check if novelty is below threshold."""
        return self.novelty < threshold


@dataclass
class NoveltyResult:
    """Result of novelty computation for a round."""

    round_num: int
    per_agent_novelty: dict[str, float] = field(default_factory=dict)
    avg_novelty: float = 0.0
    min_novelty: float = 1.0
    max_novelty: float = 0.0
    low_novelty_agents: list[str] = field(default_factory=list)
    details: dict[str, NoveltyScore] = field(default_factory=dict)

    def has_low_novelty(self) -> bool:
        """Check if any agent has low novelty."""
        return len(self.low_novelty_agents) > 0


class NoveltyTracker:
    """
    Tracks semantic novelty of proposals across debate rounds.

    Novelty = 1 - max(similarity to any prior proposal)

    A proposal is novel if it differs significantly from ALL prior proposals.
    If any prior proposal is highly similar, novelty is low.

    Usage:
        tracker = NoveltyTracker()

        # Round 1 - first proposals are maximally novel
        result1 = tracker.compute_novelty(proposals_round1, round_num=1)
        tracker.add_to_history(proposals_round1)

        # Round 2 - compared against round 1
        result2 = tracker.compute_novelty(proposals_round2, round_num=2)
        if result2.has_low_novelty():
            # Trigger intervention
            ...
        tracker.add_to_history(proposals_round2)
    """

    def __init__(
        self,
        backend: Optional[SimilarityBackend] = None,
        low_novelty_threshold: float = 0.15,
    ):
        """
        Initialize novelty tracker.

        Args:
            backend: Similarity backend to use (default: auto-select best available)
            low_novelty_threshold: Below this triggers low novelty alert (default 0.15)
        """
        self.backend = backend or get_similarity_backend("auto")
        self.low_novelty_threshold = low_novelty_threshold

        # History of proposals by round
        # Each entry is {agent_name: proposal_text}
        self.history: list[dict[str, str]] = []

        # Scores computed for each round
        self.scores: list[NoveltyResult] = []

        logger.info(
            f"NoveltyTracker initialized with {self.backend.__class__.__name__}, "
            f"threshold={low_novelty_threshold}"
        )

    def compute_novelty(
        self,
        current_proposals: dict[str, str],
        round_num: int,
    ) -> NoveltyResult:
        """
        Compute novelty scores for current round proposals.

        Novelty for each proposal = 1 - max(similarity to any prior proposal).
        First round proposals have novelty 1.0 (maximally novel).

        Args:
            current_proposals: Agent name -> proposal text mapping
            round_num: Current round number (1-indexed)

        Returns:
            NoveltyResult with per-agent scores and aggregate metrics
        """
        # Flatten history into list of (agent, proposal) tuples
        prior_proposals: list[tuple[str, str]] = []
        for round_history in self.history:
            for agent, text in round_history.items():
                prior_proposals.append((agent, text))

        # Compute novelty for each current proposal
        details: dict[str, NoveltyScore] = {}
        per_agent_novelty: dict[str, float] = {}

        for agent, proposal in current_proposals.items():
            if not prior_proposals:
                # First round - maximally novel
                score = NoveltyScore(
                    agent=agent,
                    round_num=round_num,
                    novelty=1.0,
                    max_similarity=0.0,
                    most_similar_to=None,
                    prior_proposals_count=0,
                )
            else:
                # Compare against all prior proposals
                max_similarity = 0.0
                most_similar_to: Optional[str] = None

                for prior_agent, prior_text in prior_proposals:
                    similarity = self.backend.compute_similarity(proposal, prior_text)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_to = prior_agent

                novelty = 1.0 - max_similarity

                score = NoveltyScore(
                    agent=agent,
                    round_num=round_num,
                    novelty=novelty,
                    max_similarity=max_similarity,
                    most_similar_to=most_similar_to,
                    prior_proposals_count=len(prior_proposals),
                )

            details[agent] = score
            per_agent_novelty[agent] = score.novelty

        # Compute aggregate metrics
        novelty_values = list(per_agent_novelty.values())
        avg_novelty = sum(novelty_values) / len(novelty_values) if novelty_values else 1.0
        min_novelty = min(novelty_values) if novelty_values else 1.0
        max_novelty = max(novelty_values) if novelty_values else 1.0

        # Find agents below threshold
        low_novelty_agents = [
            agent
            for agent, novelty in per_agent_novelty.items()
            if novelty < self.low_novelty_threshold
        ]

        result = NoveltyResult(
            round_num=round_num,
            per_agent_novelty=per_agent_novelty,
            avg_novelty=avg_novelty,
            min_novelty=min_novelty,
            max_novelty=max_novelty,
            low_novelty_agents=low_novelty_agents,
            details=details,
        )

        # Store result
        self.scores.append(result)

        if low_novelty_agents:
            logger.warning(
                f"Round {round_num}: Low novelty detected for {low_novelty_agents}. "
                f"Min novelty: {min_novelty:.2f}"
            )
        else:
            logger.debug(
                f"Round {round_num}: Novelty OK. Avg={avg_novelty:.2f}, Min={min_novelty:.2f}"
            )

        return result

    def add_to_history(self, proposals: dict[str, str]) -> None:
        """
        Add proposals to history for future comparisons.

        Call this after compute_novelty() if you want these proposals
        included in future novelty calculations.

        Args:
            proposals: Agent name -> proposal text mapping
        """
        # Store a copy to prevent mutation issues
        self.history.append(dict(proposals))
        logger.debug(f"Added round {len(self.history)} to history ({len(proposals)} proposals)")

    def get_agent_novelty_trajectory(self, agent: str) -> list[float]:
        """
        Get novelty scores across rounds for a specific agent.

        Args:
            agent: Agent name

        Returns:
            List of novelty scores by round
        """
        return [
            result.per_agent_novelty.get(agent, 0.0)
            for result in self.scores
        ]

    def get_debate_novelty_summary(self) -> dict:
        """
        Get summary statistics for the entire debate.

        Returns:
            Dict with overall_avg, overall_min, rounds_with_low_novelty, etc.
        """
        if not self.scores:
            return {
                "overall_avg": 1.0,
                "overall_min": 1.0,
                "rounds_with_low_novelty": 0,
                "total_rounds": 0,
            }

        all_novelties = [
            n for result in self.scores for n in result.per_agent_novelty.values()
        ]

        return {
            "overall_avg": sum(all_novelties) / len(all_novelties) if all_novelties else 1.0,
            "overall_min": min(all_novelties) if all_novelties else 1.0,
            "rounds_with_low_novelty": sum(1 for r in self.scores if r.has_low_novelty()),
            "total_rounds": len(self.scores),
            "low_novelty_agents_by_round": {
                r.round_num: r.low_novelty_agents for r in self.scores if r.low_novelty_agents
            },
        }

    def reset(self) -> None:
        """Reset tracker state for a new debate."""
        self.history.clear()
        self.scores.clear()
        logger.debug("NoveltyTracker reset")
