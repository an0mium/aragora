"""
Metrics helper for debate orchestration.

This module extracts calibration and scoring utilities from the Arena class
to improve modularity and testability.

Classes:
- MetricsHelper: Provides calibration weights, composite scores, domain extraction
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Domain detection keywords for debate classification
DOMAIN_KEYWORDS = {
    "security": ["security", "hack", "vulnerability", "auth", "encrypt"],
    "performance": ["performance", "speed", "optimize", "cache", "latency"],
    "testing": ["test", "testing", "coverage", "regression"],
    "architecture": ["design", "architecture", "pattern", "structure"],
    "debugging": ["bug", "error", "fix", "crash", "exception"],
    "api": ["api", "endpoint", "rest", "graphql"],
    "database": ["database", "sql", "query", "schema"],
    "frontend": ["ui", "frontend", "react", "css", "layout"],
}


class MetricsHelper:
    """
    Helper class for calibration and scoring calculations.

    Provides methods used during debate execution for:
    - Vote weighting based on calibration scores
    - Judge selection scoring (ELO + calibration)
    - Domain extraction from debate topics

    Usage:
        helper = MetricsHelper(elo_system=arena.elo_system)
        weight = helper.get_calibration_weight("claude")
        score = helper.get_composite_judge_score("gemini")
        domain = helper.extract_domain("Design a secure API")
    """

    def __init__(
        self,
        elo_system: Any = None,
        *,
        elo_weight: float = 0.7,
        calibration_weight: float = 0.3,
        min_calibration_weight: float = 0.5,
        max_calibration_weight: float = 1.5,
    ):
        """
        Initialize the metrics helper.

        Args:
            elo_system: Optional ELOSystem for ratings lookup
            elo_weight: Weight for ELO in composite score (default 0.7)
            calibration_weight: Weight for calibration in composite score (default 0.3)
            min_calibration_weight: Minimum calibration weight (default 0.5)
            max_calibration_weight: Maximum calibration weight (default 1.5)
        """
        self.elo_system = elo_system
        self._elo_weight = elo_weight
        self._calibration_weight = calibration_weight
        self._min_cal_weight = min_calibration_weight
        self._max_cal_weight = max_calibration_weight

        # Cache for domain extraction (stateless, one per debate)
        self._domain_cache: dict[str, str] = {}

    def get_calibration_weight(self, agent_name: str) -> float:
        """
        Get agent weight based on calibration score.

        Uses calibration_score from ELO system to weight agent contributions.
        Agents with better calibration (more accurate confidence estimates)
        have higher weight in voting and selection decisions.

        Args:
            agent_name: Name of the agent

        Returns:
            Weight between min_calibration_weight and max_calibration_weight
            (default: 0.5 to 1.5)
        """
        if not self.elo_system:
            return 1.0

        try:
            rating = self.elo_system.get_rating(agent_name)
            # calibration_score is 0-1, with 0 for agents with < MIN_COUNT predictions
            cal_score = rating.calibration_score
            # Map 0-1 to configured range (default: 0.5-1.5)
            range_size = self._max_cal_weight - self._min_cal_weight
            return self._min_cal_weight + (cal_score * range_size)
        except Exception as e:
            logger.debug(f"Calibration weight lookup failed for {agent_name}: {e}")
            return 1.0

    def get_composite_judge_score(self, agent_name: str) -> float:
        """
        Compute composite score for judge selection (ELO + calibration).

        Combines ELO ranking with calibration score for more nuanced judge selection.
        Well-calibrated agents with high ELO make better judges.

        Args:
            agent_name: Name of the agent

        Returns:
            Composite score (higher is better)
        """
        if not self.elo_system:
            return 0.0

        try:
            rating = self.elo_system.get_rating(agent_name)

            # Normalize ELO: 1000 is baseline, 500 is typical deviation
            elo_normalized = (rating.elo - 1000) / 500  # ~-1 to 3 range typically
            elo_normalized = max(0, elo_normalized)  # Floor at 0

            # Calibration score is already 0-1
            cal_score = rating.calibration_score

            # Weighted combination
            return (elo_normalized * self._elo_weight) + (cal_score * self._calibration_weight)
        except Exception as e:
            logger.debug(f"Composite score calculation failed for {agent_name}: {e}")
            return 0.0

    def extract_domain(self, task: str) -> str:
        """
        Extract domain from a debate task for calibration tracking.

        Uses heuristics to categorize the debate topic. Results are cached
        per task string to avoid repeated analysis.

        Args:
            task: The debate task/topic string

        Returns:
            Domain category (e.g., "security", "performance", "general")
        """
        # Check cache first
        if task in self._domain_cache:
            return self._domain_cache[task]

        task_lower = task.lower()

        # Check each domain's keywords
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(keyword in task_lower for keyword in keywords):
                self._domain_cache[task] = domain
                return domain

        # Default domain
        self._domain_cache[task] = "general"
        return "general"

    def clear_cache(self) -> None:
        """Clear the domain cache (call when starting new debates)."""
        self._domain_cache.clear()

    def get_agent_rating(self, agent_name: str) -> Optional[Any]:
        """
        Get full agent rating from ELO system.

        Args:
            agent_name: Name of the agent

        Returns:
            AgentRating object or None if unavailable
        """
        if not self.elo_system:
            return None

        try:
            return self.elo_system.get_rating(agent_name)
        except Exception as e:
            logger.debug(f"Rating lookup failed for {agent_name}: {e}")
            return None

    def get_ratings_batch(self, agent_names: list[str]) -> dict[str, Any]:
        """
        Get ratings for multiple agents in one call.

        Args:
            agent_names: List of agent names

        Returns:
            Dict mapping agent names to their ratings
        """
        if not self.elo_system:
            return {}

        try:
            return self.elo_system.get_ratings_batch(agent_names)
        except Exception as e:
            logger.debug(f"Batch ratings lookup failed: {e}")
            return {}


def build_relationship_updates(
    participants: list[str],
    vote_choices: dict[str, str],
    winner: Optional[str] = None,
) -> list[dict]:
    """
    Build batch of relationship updates from debate results.

    This is a standalone function to facilitate testing and reuse.
    Used by FeedbackPhase._update_relationships().

    Args:
        participants: List of agent names in the debate
        vote_choices: Dict mapping agent name to their vote choice
        winner: Optional winner agent name

    Returns:
        List of update dicts suitable for elo_system.update_relationships_batch()
    """
    updates = []
    for i, agent_a in enumerate(participants):
        for agent_b in participants[i + 1:]:
            # Check if both agents voted and agreed
            agreed = (
                agent_a in vote_choices and
                agent_b in vote_choices and
                vote_choices[agent_a] == vote_choices[agent_b]
            )
            a_win = 1 if winner == agent_a else 0
            b_win = 1 if winner == agent_b else 0
            updates.append({
                "agent_a": agent_a,
                "agent_b": agent_b,
                "debate_increment": 1,
                "agreement_increment": 1 if agreed else 0,
                "a_win": a_win,
                "b_win": b_win,
            })
    return updates
