"""
Rhetorical Analysis Observer - Passive debate commentary for engagement.

Observes debates and provides commentary on rhetorical patterns:
- Pattern detection (concession, rebuttal, synthesis)
- Audience-friendly insights
- Debate dynamics tracking
- Non-interference with debate flow

Inspired by nomic loop debate consensus on audience engagement.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class RhetoricalPattern(Enum):
    """Rhetorical patterns to detect."""

    CONCESSION = "concession"
    REBUTTAL = "rebuttal"
    SYNTHESIS = "synthesis"
    APPEAL_TO_AUTHORITY = "appeal_to_authority"
    APPEAL_TO_EVIDENCE = "appeal_to_evidence"
    TECHNICAL_DEPTH = "technical_depth"
    RHETORICAL_QUESTION = "rhetorical_question"
    ANALOGY = "analogy"
    QUALIFICATION = "qualification"


@dataclass
class RhetoricalObservation:
    """A rhetorical observation about debate content."""

    pattern: RhetoricalPattern
    agent: str
    round_num: int
    confidence: float
    excerpt: str
    audience_commentary: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern.value,
            "agent": self.agent,
            "round_num": self.round_num,
            "confidence": self.confidence,
            "excerpt": self.excerpt,
            "audience_commentary": self.audience_commentary,
            "timestamp": self.timestamp,
        }


class RhetoricalAnalysisObserver:
    """
    Passive observer that detects rhetorical patterns in debates.

    Analyzes agent contributions without interfering with debate flow,
    generating audience-friendly commentary on debate dynamics.

    Usage:
        observer = RhetoricalAnalysisObserver()

        # Observe a message
        observations = observer.observe(
            agent="claude",
            content="I agree with the point about security, but...",
            round_num=2
        )

        for obs in observations:
            print(f"{obs.pattern.value}: {obs.audience_commentary}")

        # Get debate dynamics summary
        dynamics = observer.get_debate_dynamics()
    """

    # Pattern detection indicators
    PATTERN_INDICATORS = {
        RhetoricalPattern.CONCESSION: {
            "keywords": [
                "acknowledge",
                "fair point",
                "you're right",
                "i agree",
                "valid point",
                "granted",
                "i concede",
                "admittedly",
                "while true",
                "that said",
                "indeed",
            ],
            "patterns": [
                r"\bi (must )?(acknowledge|admit|concede)\b",
                r"\b(fair|valid|good) point\b",
                r"\byou('re| are) (right|correct)\b",
            ],
            "commentary": [
                "{agent} shows intellectual humility, acknowledging a valid point",
                "A moment of agreement! {agent} accepts the opposing view here",
                "{agent} concedes the point - building bridges in the debate",
            ],
        },
        RhetoricalPattern.REBUTTAL: {
            "keywords": [
                "however",
                "but",
                "on the contrary",
                "disagree",
                "actually",
                "in fact",
                "not quite",
                "that's not",
                "i would argue",
                "counter to",
                "misses",
            ],
            "patterns": [
                r"\b(however|but),?\s",
                r"\bi (would )?(disagree|argue)\b",
                r"\b(actually|in fact),?\s",
                r"\bon the contrary\b",
            ],
            "commentary": [
                "{agent} pushes back! The intellectual sparring intensifies",
                "Disagreement alert! {agent} challenges the prevailing view",
                "{agent} offers a compelling counterpoint",
            ],
        },
        RhetoricalPattern.SYNTHESIS: {
            "keywords": [
                "combining",
                "both perspectives",
                "integrate",
                "bringing together",
                "middle ground",
                "reconcile",
                "common ground",
                "bridge",
                "synthesis",
                "merge",
            ],
            "patterns": [
                r"\b(combin|integrat|reconcil|merg)(e|ing)\b",
                r"\b(both|all) (perspectives?|views?|points?)\b",
                r"\b(middle|common) ground\b",
            ],
            "commentary": [
                "{agent} attempts synthesis - weaving ideas together!",
                "Building consensus: {agent} finds common ground",
                "{agent} bridges the gap between opposing views",
            ],
        },
        RhetoricalPattern.APPEAL_TO_AUTHORITY: {
            "keywords": [
                "according to",
                "research shows",
                "experts say",
                "studies indicate",
                "as per",
                "documentation states",
                "best practices",
                "industry standard",
            ],
            "patterns": [
                r"\baccording to\b",
                r"\b(research|studies|experts?) (show|indicate|say|suggest)\b",
                r"\bbest practices?\b",
            ],
            "commentary": [
                "{agent} brings authoritative sources to the table",
                "Invoking expertise: {agent} appeals to established knowledge",
                "{agent} grounds the argument in research",
            ],
        },
        RhetoricalPattern.APPEAL_TO_EVIDENCE: {
            "keywords": [
                "for example",
                "such as",
                "specifically",
                "consider the case",
                "evidence suggests",
                "data shows",
                "looking at",
                "we can see",
                "demonstrated by",
            ],
            "patterns": [
                r"\bfor (example|instance)\b",
                r"\b(such as|specifically|e\.g\.)\b",
                r"\b(evidence|data) (shows|suggests|indicates)\b",
            ],
            "commentary": [
                "{agent} backs up claims with concrete evidence",
                "Getting specific: {agent} presents examples",
                "{agent} strengthens the argument with evidence",
            ],
        },
        RhetoricalPattern.TECHNICAL_DEPTH: {
            "keywords": [
                "implementation",
                "architecture",
                "algorithm",
                "complexity",
                "performance",
                "scalability",
                "database",
                "api",
                "protocol",
                "async",
                "threading",
            ],
            "patterns": [
                r"\bO\([n\d\s\*log]+\)\b",  # Big O notation
                r"\b(implement|architect|design)(?:ed|ing|ure)?\b",
                r"\b(async|await|threading|concurrent)\b",
            ],
            "commentary": [
                "{agent} dives into technical details",
                "Technical depth: {agent} gets into the nitty-gritty",
                "{agent} brings engineering precision to the debate",
            ],
        },
        RhetoricalPattern.RHETORICAL_QUESTION: {
            "keywords": [],
            "patterns": [
                r"\b(what if|why would|how can|isn't it|shouldn't we)\b[^?]*\?",
                r"\?\s*(right|no|isn't it)\??\s*$",
            ],
            "commentary": [
                "{agent} poses a thought-provoking question",
                "Rhetorical flourish: {agent} challenges the audience to think",
                "{agent} uses questions to make their point",
            ],
        },
        RhetoricalPattern.ANALOGY: {
            "keywords": [
                "like",
                "similar to",
                "just as",
                "analogous",
                "think of it as",
                "imagine",
                "metaphorically",
                "in the same way",
                "comparable to",
            ],
            "patterns": [
                r"\b(just )?like\s+(a|the|when)\b",
                r"\bsimilar to\b",
                r"\bjust as\b[^.]+so\b",
                r"\bthink of it (like|as)\b",
            ],
            "commentary": [
                "{agent} draws an illuminating analogy",
                "Making it relatable: {agent} uses comparison to clarify",
                "{agent} bridges concepts with an apt metaphor",
            ],
        },
        RhetoricalPattern.QUALIFICATION: {
            "keywords": [
                "depends on",
                "in some cases",
                "typically",
                "usually",
                "often",
                "sometimes",
                "it varies",
                "context-dependent",
                "nuanced",
            ],
            "patterns": [
                r"\b(depends on|in some cases)\b",
                r"\b(typically|usually|often|sometimes)\b",
                r"\bit('s| is) (nuanced|complex|complicated)\b",
            ],
            "commentary": [
                "{agent} adds important nuance to the discussion",
                "Qualification spotted: {agent} acknowledges complexity",
                "{agent} reminds us that context matters",
            ],
        },
    }

    def __init__(
        self,
        broadcast_callback: Optional[Callable[[dict], Any]] = None,
        min_confidence: float = 0.5,
    ):
        """
        Initialize the observer.

        Args:
            broadcast_callback: Optional callback for streaming observations
            min_confidence: Minimum confidence for reporting observations
        """
        self.broadcast_callback = broadcast_callback
        self.min_confidence = min_confidence
        self.observations: list[RhetoricalObservation] = []
        self.agent_patterns: dict[str, dict[str, int]] = {}  # Track patterns per agent

    def observe(
        self,
        agent: str,
        content: str,
        round_num: int = 0,
    ) -> list[RhetoricalObservation]:
        """
        Analyze content for rhetorical patterns.

        Args:
            agent: Name of the agent
            content: Text content to analyze
            round_num: Current debate round

        Returns:
            List of detected patterns with commentary
        """
        if not content or len(content) < 20:
            return []

        detected = self._detect_patterns(content)
        observations = []

        for pattern, confidence in detected:
            if confidence < self.min_confidence:
                continue

            # Find excerpt
            excerpt = self._find_excerpt(content, pattern)

            # Generate commentary
            commentary = self._generate_commentary(agent, pattern)

            observation = RhetoricalObservation(
                pattern=pattern,
                agent=agent,
                round_num=round_num,
                confidence=confidence,
                excerpt=excerpt,
                audience_commentary=commentary,
            )

            observations.append(observation)
            self.observations.append(observation)

            # Track per-agent
            if agent not in self.agent_patterns:
                self.agent_patterns[agent] = {}
            pattern_key = pattern.value
            self.agent_patterns[agent][pattern_key] = (
                self.agent_patterns[agent].get(pattern_key, 0) + 1
            )

        # Broadcast if callback set
        if self.broadcast_callback and observations:
            try:
                self.broadcast_callback(
                    {
                        "type": "rhetorical_observations",
                        "data": {
                            "agent": agent,
                            "round_num": round_num,
                            "observations": [o.to_dict() for o in observations],
                        },
                    }
                )
            except Exception as e:
                logger.warning(f"rhetorical_broadcast_failed error={e}")

        return observations

    def _detect_patterns(self, content: str) -> list[tuple[RhetoricalPattern, float]]:
        """Detect rhetorical patterns in content."""
        content_lower = content.lower()
        detected = []

        for pattern, indicators in self.PATTERN_INDICATORS.items():
            score = 0.0

            # Check keywords
            keywords = indicators.get("keywords", [])
            keyword_matches = sum(1 for kw in keywords if kw in content_lower)
            if keywords and keyword_matches:
                score += min(0.5, keyword_matches * 0.15)

            # Check regex patterns
            regex_patterns = indicators.get("patterns", [])
            for regex in regex_patterns:
                try:
                    if re.search(regex, content_lower):
                        score += 0.3
                except re.error as e:
                    logger.debug(f"Invalid regex pattern '{regex}': {e}")

            # Normalize score
            confidence = min(1.0, score)

            if confidence >= self.min_confidence:
                detected.append((pattern, confidence))

        return detected

    def _find_excerpt(self, content: str, pattern: RhetoricalPattern) -> str:
        """Find a relevant excerpt for the pattern."""
        indicators = self.PATTERN_INDICATORS.get(pattern, {})
        # Try to find a sentence containing pattern indicators
        sentences = re.split(r"[.!?]+", content)

        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Check keywords
            for kw in indicators.get("keywords", []):
                if kw in sentence_lower:
                    return sentence.strip()[:150]

            # Check patterns
            for regex in indicators.get("patterns", []):
                try:
                    if re.search(regex, sentence_lower):
                        return sentence.strip()[:150]
                except re.error as e:
                    logger.debug(f"Invalid regex pattern '{regex}': {e}")

        # Fallback to first sentence
        if sentences:
            return sentences[0].strip()[:150]

        return content[:100]

    def _generate_commentary(self, agent: str, pattern: RhetoricalPattern) -> str:
        """Generate audience commentary for a pattern."""
        indicators = self.PATTERN_INDICATORS.get(pattern, {})
        templates = indicators.get("commentary", [])

        if templates:
            import random

            template = random.choice(templates)
            return template.format(agent=agent)

        return f"{agent} employs {pattern.value} in their argument"

    def get_debate_dynamics(self) -> dict:
        """Get summary of overall debate dynamics."""
        if not self.observations:
            return {
                "total_observations": 0,
                "patterns_detected": {},
                "agent_styles": {},
                "dominant_pattern": None,
            }

        # Count patterns
        pattern_counts: dict[str, int] = {}
        for obs in self.observations:
            key = obs.pattern.value
            pattern_counts[key] = pattern_counts.get(key, 0) + 1

        # Determine dominant pattern
        dominant = max(pattern_counts.items(), key=lambda x: x[1])[0] if pattern_counts else None

        # Characterize agent styles
        agent_styles = {}
        for agent, patterns in self.agent_patterns.items():
            if patterns:
                top_pattern = max(patterns.items(), key=lambda x: x[1])[0]
                style = self._pattern_to_style(top_pattern)
                agent_styles[agent] = {
                    "dominant_pattern": top_pattern,
                    "style": style,
                    "pattern_diversity": len(patterns),
                }

        return {
            "total_observations": len(self.observations),
            "patterns_detected": pattern_counts,
            "agent_styles": agent_styles,
            "dominant_pattern": dominant,
            "debate_character": self._characterize_debate(pattern_counts),
        }

    def _pattern_to_style(self, pattern: str) -> str:
        """Map pattern to debate style label."""
        style_map = {
            "concession": "diplomatic",
            "rebuttal": "combative",
            "synthesis": "collaborative",
            "appeal_to_authority": "scholarly",
            "appeal_to_evidence": "empirical",
            "technical_depth": "technical",
            "rhetorical_question": "socratic",
            "analogy": "illustrative",
            "qualification": "nuanced",
        }
        return style_map.get(pattern, "balanced")

    def _characterize_debate(self, pattern_counts: dict) -> str:
        """Characterize overall debate based on pattern distribution."""
        if not pattern_counts:
            return "emerging"

        total = sum(pattern_counts.values())

        # Check for consensus-building (concession + synthesis)
        collaborative = pattern_counts.get("concession", 0) + pattern_counts.get("synthesis", 0)
        if collaborative > total * 0.4:
            return "collaborative"

        # Check for combative (high rebuttal)
        if pattern_counts.get("rebuttal", 0) > total * 0.4:
            return "contentious"

        # Check for technical (high technical depth)
        if pattern_counts.get("technical_depth", 0) > total * 0.3:
            return "technical"

        # Check for evidence-heavy
        evidence = pattern_counts.get("appeal_to_evidence", 0) + pattern_counts.get(
            "appeal_to_authority", 0
        )
        if evidence > total * 0.3:
            return "evidence-driven"

        return "balanced"

    def get_recent_observations(self, limit: int = 10) -> list[dict]:
        """Get recent observations."""
        return [o.to_dict() for o in self.observations[-limit:]]

    def reset(self) -> None:
        """Reset all observations."""
        self.observations = []
        self.agent_patterns = {}


# Global observer instance
_observer: Optional[RhetoricalAnalysisObserver] = None


def get_rhetorical_observer() -> RhetoricalAnalysisObserver:
    """Get the global rhetorical observer instance."""
    global _observer
    if _observer is None:
        _observer = RhetoricalAnalysisObserver()
    return _observer


def reset_rhetorical_observer() -> None:
    """Reset the global observer (for testing)."""
    global _observer
    _observer = None
