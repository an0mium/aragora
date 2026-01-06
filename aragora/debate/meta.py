"""
Meta-Critique capability.

Enables agents to analyze the debate process itself, identifying:
- Unproductive exchanges
- Talking past each other
- Circular arguments
- Missed opportunities
- Process improvements
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)

from aragora.core import Message, Critique, DebateResult
from aragora.memory.embeddings import EmbeddingProvider, cosine_similarity


@dataclass
class MetaObservation:
    """An observation about the debate process."""

    observation_type: str  # "issue", "pattern", "suggestion"
    description: str
    severity: float  # 0-1
    round_range: tuple[int, int]  # Which rounds this applies to
    agents_involved: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


@dataclass
class MetaCritique:
    """A critique of the debate process itself."""

    debate_id: str
    observations: list[MetaObservation]
    overall_quality: float  # 0-1
    productive_rounds: list[int]
    unproductive_rounds: list[int]
    recommendations: list[str]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def summary(self) -> str:
        """Generate summary of meta-critique."""
        issues = [o for o in self.observations if o.observation_type == "issue"]
        patterns = [o for o in self.observations if o.observation_type == "pattern"]

        parts = []
        parts.append(f"Debate Quality: {self.overall_quality:.0%}")
        parts.append(f"Productive Rounds: {self.productive_rounds}")
        parts.append(f"Issues Found: {len(issues)}")

        if self.recommendations:
            parts.append(f"Top Recommendation: {self.recommendations[0]}")

        return " | ".join(parts)


class MetaCritiqueAnalyzer:
    """
    Analyzes debates to provide meta-level feedback.

    Identifies process issues and suggests improvements.
    """

    # Patterns that indicate unproductive exchanges
    REPETITION_THRESHOLD = 0.6  # Similarity threshold for detecting repetition
    MIN_PROGRESS_THRESHOLD = 0.2  # Minimum expected progress per round

    def __init__(self, embedding_provider: Optional[EmbeddingProvider] = None):
        """
        Initialize the analyzer.

        Args:
            embedding_provider: Optional embedding provider for semantic similarity.
                               Falls back to word-overlap if not provided.
        """
        self._embedding_provider = embedding_provider
        self._embedding_cache: dict[str, list[float]] = {}

    def analyze(self, result: DebateResult) -> MetaCritique:
        """
        Analyze a completed debate.

        Args:
            result: The debate result to analyze

        Returns:
            MetaCritique with observations and recommendations
        """
        observations = []

        # Check for repetitive exchanges
        observations.extend(self._detect_repetition(result.messages))

        # Check for talking past each other
        observations.extend(self._detect_misalignment(result.messages, result.critiques))

        # Check for circular arguments
        observations.extend(self._detect_circular_arguments(result.messages))

        # Check for ignored critiques
        observations.extend(self._detect_ignored_critiques(result.critiques, result.messages))

        # Identify productive vs unproductive rounds
        productive, unproductive = self._classify_rounds(result)

        # Calculate overall quality
        quality = self._calculate_quality(result, observations, productive, unproductive)

        # Generate recommendations
        recommendations = self._generate_recommendations(observations, result)

        return MetaCritique(
            debate_id=result.id,
            observations=observations,
            overall_quality=quality,
            productive_rounds=productive,
            unproductive_rounds=unproductive,
            recommendations=recommendations,
        )

    def _detect_repetition(self, messages: list[Message]) -> list[MetaObservation]:
        """Detect repetitive content across messages."""
        observations = []

        # Group messages by agent
        by_agent = {}
        for msg in messages:
            if msg.agent not in by_agent:
                by_agent[msg.agent] = []
            by_agent[msg.agent].append(msg)

        for agent, agent_msgs in by_agent.items():
            if len(agent_msgs) < 2:
                continue

            # Check for high similarity between consecutive messages
            for i in range(1, len(agent_msgs)):
                similarity = self._text_similarity(
                    agent_msgs[i - 1].content,
                    agent_msgs[i].content,
                )

                if similarity > self.REPETITION_THRESHOLD:
                    observations.append(MetaObservation(
                        observation_type="issue",
                        description=f"{agent} repeated similar content across rounds",
                        severity=0.6,
                        round_range=(agent_msgs[i - 1].round, agent_msgs[i].round),
                        agents_involved=[agent],
                        evidence=[
                            f"Round {agent_msgs[i-1].round}: {agent_msgs[i-1].content[:100]}...",
                            f"Round {agent_msgs[i].round}: {agent_msgs[i].content[:100]}...",
                        ],
                    ))

        return observations

    def _detect_misalignment(
        self,
        messages: list[Message],
        critiques: list[Critique],
    ) -> list[MetaObservation]:
        """Detect when agents are talking past each other."""
        observations = []

        # Check if critiques address the actual proposal content
        for critique in critiques:
            # Simple check: do critique issues mention key terms from the proposal?
            proposal_terms = set(critique.target_content.lower().split())
            critique_terms = set(" ".join(critique.issues).lower().split())

            overlap = len(proposal_terms & critique_terms) / max(len(proposal_terms), 1)

            if overlap < 0.1:  # Very low overlap
                observations.append(MetaObservation(
                    observation_type="issue",
                    description=f"{critique.agent}'s critique may not address the actual proposal",
                    severity=0.5,
                    round_range=(0, 0),  # We don't track round in Critique
                    agents_involved=[critique.agent, critique.target_agent],
                    evidence=[f"Low term overlap: {overlap:.0%}"],
                ))

        return observations

    def _detect_circular_arguments(self, messages: list[Message]) -> list[MetaObservation]:
        """Detect circular or oscillating arguments."""
        observations = []

        # Track key positions over time
        # This is a simplified version - a full implementation would use embeddings

        positions_by_round = {}
        for msg in messages:
            if msg.round not in positions_by_round:
                positions_by_round[msg.round] = []
            positions_by_round[msg.round].append(msg.content[:200])

        # Check if later rounds return to earlier positions
        rounds = sorted(positions_by_round.keys())
        if len(rounds) >= 4:
            early_content = " ".join(positions_by_round.get(rounds[0], []))
            late_content = " ".join(positions_by_round.get(rounds[-1], []))

            if self._text_similarity(early_content, late_content) > 0.7:
                observations.append(MetaObservation(
                    observation_type="pattern",
                    description="Debate may have returned to initial positions (circular)",
                    severity=0.7,
                    round_range=(rounds[0], rounds[-1]),
                    agents_involved=[],
                    evidence=["High similarity between first and last rounds"],
                ))

        return observations

    def _detect_ignored_critiques(
        self,
        critiques: list[Critique],
        messages: list[Message],
    ) -> list[MetaObservation]:
        """Detect critiques that were ignored in subsequent responses."""
        observations = []

        for critique in critiques:
            if not critique.suggestions:
                continue

            # Find the next message from the target agent
            target_agent = critique.target_agent
            subsequent_msgs = [
                m for m in messages
                if m.agent == target_agent and m.role == "proposer"
            ]

            if not subsequent_msgs:
                continue

            # Check if any suggestion keywords appear in the response
            suggestion_text = " ".join(critique.suggestions).lower()
            response_text = subsequent_msgs[-1].content.lower()

            # Very simple check - could be improved with semantic similarity
            suggestion_words = set(suggestion_text.split())
            response_words = set(response_text.split())

            # Filter out common words
            common_words = {"the", "a", "an", "to", "and", "or", "is", "are", "be", "been"}
            suggestion_words -= common_words
            response_words -= common_words

            overlap = len(suggestion_words & response_words)

            if len(suggestion_words) > 3 and overlap < 2:
                observations.append(MetaObservation(
                    observation_type="issue",
                    description=f"Suggestions from {critique.agent} may have been ignored",
                    severity=0.4,
                    round_range=(0, 0),
                    agents_involved=[critique.agent, target_agent],
                    evidence=[f"Low overlap between suggestions and response"],
                ))

        return observations

    def _classify_rounds(self, result: DebateResult) -> tuple[list[int], list[int]]:
        """Classify rounds as productive or unproductive."""
        productive = []
        unproductive = []

        rounds = set(m.round for m in result.messages)

        for r in sorted(rounds):
            round_msgs = [m for m in result.messages if m.round == r]
            round_critiques = [c for c in result.critiques if any(
                m.round == r and m.agent == c.target_agent for m in round_msgs
            )]

            # A round is productive if:
            # 1. Critiques were raised
            # 2. Issues had reasonable severity (not too high = real problems found)
            # 3. Content changed from previous round

            has_critiques = len(round_critiques) > 0
            avg_severity = sum(c.severity for c in round_critiques) / max(len(round_critiques), 1)

            if has_critiques and avg_severity < 0.8:
                productive.append(r)
            else:
                unproductive.append(r)

        return productive, unproductive

    def _calculate_quality(
        self,
        result: DebateResult,
        observations: list[MetaObservation],
        productive: list[int],
        unproductive: list[int],
    ) -> float:
        """Calculate overall debate quality score."""
        score = 1.0

        # Penalize for issues
        for obs in observations:
            if obs.observation_type == "issue":
                score -= obs.severity * 0.1

        # Penalize for unproductive rounds
        total_rounds = len(productive) + len(unproductive)
        if total_rounds > 0:
            productive_ratio = len(productive) / total_rounds
            score *= (0.5 + 0.5 * productive_ratio)

        # Bonus for reaching consensus
        if result.consensus_reached:
            score = min(1.0, score + 0.1)

        # Factor in confidence
        score *= (0.5 + 0.5 * result.confidence)

        return max(0.0, min(1.0, score))

    def _generate_recommendations(
        self,
        observations: list[MetaObservation],
        result: DebateResult,
    ) -> list[str]:
        """Generate recommendations for improving future debates."""
        recommendations = []

        issues = [o for o in observations if o.observation_type == "issue"]

        # Repetition issues
        if any("repeated" in o.description.lower() for o in issues):
            recommendations.append(
                "Encourage agents to build on previous points rather than restating"
            )

        # Misalignment issues
        if any("not address" in o.description.lower() for o in issues):
            recommendations.append(
                "Ensure critiques directly reference specific parts of proposals"
            )

        # Ignored critiques
        if any("ignored" in o.description.lower() for o in issues):
            recommendations.append(
                "Require proposers to explicitly acknowledge each critique"
            )

        # Circular arguments
        if any("circular" in o.description.lower() for o in observations):
            recommendations.append(
                "Set a maximum round limit and force synthesis when progress stalls"
            )

        # Low confidence result
        if result.confidence < 0.6:
            recommendations.append(
                "Consider adding more rounds or a judge agent to improve consensus"
            )

        # No consensus
        if not result.consensus_reached:
            recommendations.append(
                "Try a different consensus mechanism or allow minority reports"
            )

        return recommendations[:5]  # Limit to top 5

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using embeddings if available, else word overlap.

        Uses semantic similarity via embeddings when an embedding provider is
        configured, falling back to Jaccard word overlap otherwise.
        """
        if self._embedding_provider:
            try:
                return self._semantic_similarity(text1, text2)
            except Exception as e:
                logger.debug(f"Semantic similarity failed, using word-based fallback: {e}")

        # Word-based Jaccard similarity (fallback)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using embeddings."""
        def get_embedding(text: str) -> list[float]:
            # Use cache to avoid redundant API calls
            cache_key = text[:500]  # Truncate for cache key
            if cache_key not in self._embedding_cache:
                # Run async embed in sync context
                # Use asyncio.run() for proper event loop lifecycle management
                embedding = asyncio.run(
                    self._embedding_provider.embed(text[:2000])
                )
                self._embedding_cache[cache_key] = embedding
            return self._embedding_cache[cache_key]

        emb1 = get_embedding(text1)
        emb2 = get_embedding(text2)
        return cosine_similarity(emb1, emb2)

    def generate_meta_prompt(self, result: DebateResult) -> str:
        """
        Generate a prompt for an agent to perform meta-analysis.

        This can be used with any agent to get deeper insights.
        """
        messages_summary = "\n".join([
            f"[Round {m.round}] {m.agent} ({m.role}): {m.content[:150]}..."
            for m in result.messages[:20]
        ])

        critiques_summary = "\n".join([
            f"- {c.agent} → {c.target_agent}: {', '.join(c.issues[:2])}"
            for c in result.critiques[:10]
        ])

        return f"""Analyze this debate as a meta-observer. Your job is to evaluate the debate PROCESS, not the content.

Task: {result.task}

Messages:
{messages_summary}

Critiques:
{critiques_summary}

Consensus: {"Reached" if result.consensus_reached else "Not reached"} (Confidence: {result.confidence:.0%})

Analyze:
1. PRODUCTIVE ROUNDS: Which rounds made real progress? Why?
2. ISSUES: Where did the debate go wrong or become unproductive?
3. PATTERNS: What patterns do you notice in how agents interacted?
4. RECOMMENDATIONS: How could future debates be improved?

Format your response with clear sections for each point.
"""
