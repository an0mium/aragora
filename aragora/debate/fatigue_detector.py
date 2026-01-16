"""
Debate Fatigue Detection System.

Monitors agent response quality for signs of cognitive fatigue including:
- Response length decline
- Increased repetition
- Argument novelty reduction
- Engagement pattern changes

Fatigue detection helps the orchestrator make decisions about:
- When to pause a debate for break
- Which agents may need rotation
- When diminishing returns indicate debate should conclude
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class FatigueSignal:
    """Signal indicating an agent may be experiencing cognitive fatigue."""

    agent: str
    score: float  # 0.0 to 1.0, higher = more fatigued
    round: int
    recommendation: str  # "monitor", "consider_rest", "rotate_out"
    metrics: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent": self.agent,
            "score": self.score,
            "round": self.round,
            "recommendation": self.recommendation,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentBaseline:
    """Baseline metrics for an agent to compare against."""

    avg_response_length: float = 500.0
    avg_unique_word_ratio: float = 0.7
    avg_argument_count: int = 3
    samples: int = 0


class FatigueDetector:
    """Monitor agent response quality for signs of cognitive fatigue.

    The detector builds baselines from early responses and then monitors
    for deviation patterns that indicate fatigue:

    1. Response length decline (shorter responses)
    2. Vocabulary reduction (fewer unique words)
    3. Repetition increase (reusing same phrases)
    4. Argument novelty decline (same points repeated)
    5. Engagement decline (less substantive critique engagement)

    Usage:
        detector = FatigueDetector()

        # For each agent response
        signal = detector.analyze_response(
            agent="claude",
            response="...",
            round=3
        )

        if signal and signal.score > 0.8:
            print(f"Agent {signal.agent} showing fatigue: {signal.recommendation}")
    """

    def __init__(
        self,
        fatigue_threshold: float = 0.7,
        critical_threshold: float = 0.85,
        baseline_rounds: int = 2,
    ):
        """Initialize fatigue detector.

        Args:
            fatigue_threshold: Score threshold for fatigue warning (0-1)
            critical_threshold: Score threshold for critical fatigue (0-1)
            baseline_rounds: Number of rounds to establish baseline
        """
        self.fatigue_threshold = fatigue_threshold
        self.critical_threshold = critical_threshold
        self.baseline_rounds = baseline_rounds

        # Per-agent state
        self.baselines: Dict[str, AgentBaseline] = defaultdict(AgentBaseline)
        self.response_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.seen_arguments: Dict[str, Set[str]] = defaultdict(set)
        self.fatigue_signals: List[FatigueSignal] = []

    def analyze_response(
        self,
        agent: str,
        response: str,
        round: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[FatigueSignal]:
        """Analyze a response for fatigue indicators.

        Args:
            agent: Agent name
            response: The agent's response text
            round: Current debate round
            context: Optional context (previous messages, topic, etc.)

        Returns:
            FatigueSignal if fatigue detected, None otherwise
        """
        if not response or not response.strip():
            return None

        # Calculate metrics
        metrics = {
            "response_length": len(response),
            "unique_words_ratio": self._unique_words_ratio(response),
            "repetition_score": self._detect_repetition(agent, response),
            "argument_novelty": self._argument_novelty(agent, response),
            "engagement_score": self._engagement_score(response, context),
        }

        # Update baseline during early rounds
        if round <= self.baseline_rounds:
            self._update_baseline(agent, metrics)
            return None

        # Calculate fatigue score
        fatigue_score = self._calculate_fatigue_score(agent, metrics)

        # Store response history
        self.response_history[agent].append({
            "round": round,
            "length": len(response),
            "metrics": metrics,
            "fatigue_score": fatigue_score,
            "timestamp": datetime.now().isoformat(),
        })

        # Generate signal if threshold exceeded
        if fatigue_score > self.fatigue_threshold:
            recommendation = self._get_recommendation(fatigue_score)
            signal = FatigueSignal(
                agent=agent,
                score=fatigue_score,
                round=round,
                recommendation=recommendation,
                metrics=metrics,
            )
            self.fatigue_signals.append(signal)
            logger.info(
                "Fatigue detected: agent=%s, score=%.2f, recommendation=%s",
                agent,
                fatigue_score,
                recommendation,
            )
            return signal

        return None

    def _unique_words_ratio(self, response: str) -> float:
        """Calculate ratio of unique words to total words."""
        words = response.lower().split()
        if not words:
            return 0.0
        unique = set(words)
        return len(unique) / len(words)

    def _detect_repetition(self, agent: str, response: str) -> float:
        """Detect repetition from previous responses.

        Returns 0.0 (no repetition) to 1.0 (highly repetitive).
        """
        history = self.response_history.get(agent, [])
        if not history:
            return 0.0

        # Get n-grams from current response
        current_ngrams = self._extract_ngrams(response, n=3)
        if not current_ngrams:
            return 0.0

        # Compare with recent history
        recent_responses = history[-5:]  # Last 5 responses
        total_overlap = 0
        for prev in recent_responses:
            # Reconstruct previous response from stored data
            prev_text = prev.get("text", "")
            if not prev_text:
                continue
            prev_ngrams = self._extract_ngrams(prev_text, n=3)
            if prev_ngrams:
                overlap = len(current_ngrams & prev_ngrams) / len(current_ngrams)
                total_overlap += overlap

        return min(total_overlap / max(len(recent_responses), 1), 1.0)

    def _extract_ngrams(self, text: str, n: int = 3) -> Set[str]:
        """Extract n-grams from text."""
        words = text.lower().split()
        if len(words) < n:
            return set()
        return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}

    def _argument_novelty(self, agent: str, response: str) -> float:
        """Measure how novel the arguments are compared to past responses.

        Returns 0.0 (completely repetitive) to 1.0 (completely novel).
        """
        # Extract key phrases (simplified argument extraction)
        phrases = self._extract_key_phrases(response)
        if not phrases:
            return 1.0  # Empty = cannot determine novelty

        seen = self.seen_arguments[agent]
        new_phrases = phrases - seen

        novelty = len(new_phrases) / len(phrases) if phrases else 0.0

        # Update seen arguments
        self.seen_arguments[agent].update(phrases)

        return novelty

    def _extract_key_phrases(self, text: str) -> Set[str]:
        """Extract key argumentative phrases from text."""
        # Simple extraction: sentences that contain argument indicators
        indicators = [
            "because", "therefore", "however", "although", "moreover",
            "furthermore", "consequently", "thus", "hence", "suggest",
            "argue", "believe", "evidence", "conclude", "implies"
        ]

        phrases = set()
        sentences = text.replace("!", ".").replace("?", ".").split(".")

        for sentence in sentences:
            sentence = sentence.strip().lower()
            if any(ind in sentence for ind in indicators):
                # Normalize and hash the sentence
                normalized = " ".join(sentence.split())[:100]
                if len(normalized) > 20:
                    phrases.add(normalized)

        return phrases

    def _engagement_score(
        self, response: str, context: Optional[Dict[str, Any]]
    ) -> float:
        """Score how engaged the response is with prior arguments.

        Higher score = better engagement with context.
        """
        if not context:
            return 0.5  # Neutral when no context

        # Check for references to other agents' points
        other_agents = context.get("other_agents", [])
        referenced = 0
        for agent in other_agents:
            if agent.lower() in response.lower():
                referenced += 1

        # Check for quote/reference patterns
        reference_patterns = [
            "as mentioned", "building on", "in response to",
            "addressing", "regarding the point", "contrary to",
            "agrees with", "disagrees with", "earlier point"
        ]
        response_lower = response.lower()
        pattern_matches = sum(1 for p in reference_patterns if p in response_lower)

        # Score based on references and patterns
        score = min(1.0, (referenced * 0.2 + pattern_matches * 0.1) + 0.3)
        return score

    def _update_baseline(self, agent: str, metrics: Dict[str, float]) -> None:
        """Update baseline metrics for an agent."""
        baseline = self.baselines[agent]

        # Running average
        n = baseline.samples
        baseline.avg_response_length = (
            (baseline.avg_response_length * n + metrics["response_length"]) / (n + 1)
        )
        baseline.avg_unique_word_ratio = (
            (baseline.avg_unique_word_ratio * n + metrics["unique_words_ratio"]) / (n + 1)
        )
        baseline.samples += 1

    def _calculate_fatigue_score(
        self, agent: str, metrics: Dict[str, float]
    ) -> float:
        """Calculate overall fatigue score from metrics.

        Weights different indicators based on their reliability:
        - Response length decline: moderate weight (can vary)
        - Unique words decline: high weight (strong indicator)
        - Repetition increase: high weight (strong indicator)
        - Argument novelty decline: very high weight (best indicator)
        - Engagement decline: moderate weight (context dependent)
        """
        baseline = self.baselines.get(agent, AgentBaseline())
        if baseline.samples < 1:
            return 0.0

        # Calculate deviation from baseline
        length_ratio = metrics["response_length"] / max(baseline.avg_response_length, 1)
        length_decline = max(0, 1 - length_ratio)  # 0 = no decline, 1 = 100% decline

        unique_ratio = metrics["unique_words_ratio"]
        unique_decline = max(0, baseline.avg_unique_word_ratio - unique_ratio)

        # Combine metrics with weights
        weights = {
            "length_decline": 0.15,
            "unique_decline": 0.20,
            "repetition": 0.25,
            "novelty_decline": 0.30,
            "engagement_decline": 0.10,
        }

        fatigue = (
            weights["length_decline"] * length_decline +
            weights["unique_decline"] * unique_decline * 2 +  # Scale to 0-1
            weights["repetition"] * metrics["repetition_score"] +
            weights["novelty_decline"] * (1 - metrics["argument_novelty"]) +
            weights["engagement_decline"] * (1 - metrics["engagement_score"])
        )

        return min(fatigue, 1.0)

    def _get_recommendation(self, score: float) -> str:
        """Get recommendation based on fatigue score."""
        if score > self.critical_threshold:
            return "rotate_out"
        elif score > self.fatigue_threshold:
            return "consider_rest"
        return "monitor"

    def get_agent_fatigue_history(self, agent: str) -> List[Dict[str, Any]]:
        """Get fatigue score history for an agent."""
        return self.response_history.get(agent, [])

    def get_current_fatigue_levels(self) -> Dict[str, float]:
        """Get current fatigue levels for all tracked agents."""
        levels = {}
        for agent, history in self.response_history.items():
            if history:
                levels[agent] = history[-1].get("fatigue_score", 0.0)
        return levels

    def get_all_signals(self) -> List[FatigueSignal]:
        """Get all fatigue signals generated during the debate."""
        return self.fatigue_signals

    def reset(self) -> None:
        """Reset detector state for new debate."""
        self.baselines.clear()
        self.response_history.clear()
        self.seen_arguments.clear()
        self.fatigue_signals.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export detector state to dictionary."""
        return {
            "fatigue_threshold": self.fatigue_threshold,
            "critical_threshold": self.critical_threshold,
            "current_levels": self.get_current_fatigue_levels(),
            "signals": [s.to_dict() for s in self.fatigue_signals],
            "tracked_agents": list(self.response_history.keys()),
        }


# Singleton instance for global access
_default_detector: Optional[FatigueDetector] = None


def get_fatigue_detector() -> FatigueDetector:
    """Get or create the default fatigue detector instance."""
    global _default_detector
    if _default_detector is None:
        _default_detector = FatigueDetector()
    return _default_detector


def reset_fatigue_detector() -> None:
    """Reset the global fatigue detector."""
    global _default_detector
    if _default_detector is not None:
        _default_detector.reset()
