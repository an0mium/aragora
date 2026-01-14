"""
Uncertainty Estimator.

Quantifies epistemic uncertainty in agent responses and debate outcomes,
providing confidence calibration and disagreement analysis.
"""

import asyncio
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from aragora.core import Agent, Message, Vote


@dataclass
class ConfidenceScore:
    """A calibrated confidence score from an agent."""

    agent_name: str
    value: float  # 0-1 confidence level
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "confidence": self.value,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DisagreementCrux:
    """A key point of disagreement between agents."""

    description: str
    divergent_agents: List[str]
    evidence_needed: str = ""
    severity: float = 0.5  # 0-1, how critical this disagreement is
    crux_id: str = ""  # Unique identifier for follow-up tracking

    def __post_init__(self):
        if not self.crux_id:
            # Generate stable ID from description hash
            self.crux_id = f"crux-{abs(hash(self.description)) % 100000:05d}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.crux_id,
            "description": self.description,
            "agents": self.divergent_agents,
            "evidence_needed": self.evidence_needed,
            "severity": self.severity,
        }


@dataclass
class FollowUpSuggestion:
    """A suggested follow-up debate to resolve a crux."""

    crux: DisagreementCrux
    suggested_task: str
    priority: float  # 0-1, how important this follow-up is
    parent_debate_id: Optional[str] = None
    suggested_agents: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crux_id": self.crux.crux_id,
            "crux_description": self.crux.description,
            "suggested_task": self.suggested_task,
            "priority": self.priority,
            "parent_debate_id": self.parent_debate_id,
            "suggested_agents": self.suggested_agents,
            "divergent_agents": self.crux.divergent_agents,
        }


@dataclass
class UncertaintyMetrics:
    """Comprehensive uncertainty quantification for a debate."""

    collective_confidence: float = 0.5
    confidence_interval: Tuple[float, float] = (0.4, 0.6)
    disagreement_type: str = (
        "none"  # "factual", "value-based", "definitional", "information-asymmetry"
    )
    cruxes: List[DisagreementCrux] = field(default_factory=list)
    calibration_quality: float = 0.5  # How well agents are calibrated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collective_confidence": self.collective_confidence,
            "confidence_interval": self.confidence_interval,
            "disagreement_type": self.disagreement_type,
            "cruxes": [crux.to_dict() for crux in self.cruxes],
            "calibration_quality": self.calibration_quality,
        }


class ConfidenceEstimator:
    """Estimates confidence in agent responses and calibrates over time."""

    def __init__(self):
        self.agent_confidences: Dict[str, List[ConfidenceScore]] = {}
        self.calibration_history: Dict[
            str, List[Tuple[float, bool]]
        ] = {}  # (confidence, was_correct)
        self.brier_scores: Dict[str, float] = {}  # Calibration quality metric
        self.disagreement_analyzer = DisagreementAnalyzer()

    async def collect_confidences(
        self, agents: List[Agent], proposals: Dict[str, str], task: str
    ) -> Dict[str, ConfidenceScore]:
        """Collect confidence scores from all agents."""
        confidence_tasks = []

        for agent in agents:
            confidence_tasks.append(self._get_agent_confidence(agent, proposals, task))

        results = await asyncio.gather(*confidence_tasks, return_exceptions=True)

        confidences = {}
        for agent, result in zip(agents, results):
            if isinstance(result, Exception):
                logger.warning(f"Error getting confidence from {agent.name}: {result}")
                # Default confidence
                confidences[agent.name] = ConfidenceScore(
                    agent.name, 0.5, "Error estimating confidence"
                )
            else:
                score: ConfidenceScore = result  # type: ignore[assignment]
                confidences[agent.name] = score
                # Store for calibration tracking
                self._store_confidence(agent.name, score)

        return confidences

    async def _get_agent_confidence(
        self, agent: Agent, proposals: Dict[str, str], task: str
    ) -> ConfidenceScore:
        """Get calibrated confidence from an agent."""
        # This would ideally be a new method on Agent, but we'll simulate with existing methods
        # In practice, this would require extending the Agent interface

        # For now, use vote confidence as proxy (assuming agents provide confidence in votes)
        try:
            vote = await agent.vote(proposals, task)
            return ConfidenceScore(
                agent_name=agent.name,
                value=vote.confidence,
                reasoning=f"Based on voting confidence: {vote.reasoning}",
            )
        except Exception as e:
            # Fallback to default confidence
            logger.warning(f"Failed to get confidence from {agent.name}: {e}")
            return ConfidenceScore(agent.name, 0.5, "Default confidence")

    def _store_confidence(self, agent_name: str, confidence: ConfidenceScore):
        """Store confidence for calibration tracking."""
        if agent_name not in self.agent_confidences:
            self.agent_confidences[agent_name] = []
        self.agent_confidences[agent_name].append(confidence)

        # Keep only recent history
        if len(self.agent_confidences[agent_name]) > 100:
            self.agent_confidences[agent_name] = self.agent_confidences[agent_name][-50:]

    def get_agent_calibration_quality(self, agent_name: str) -> float:
        """Get Brier score for agent calibration quality."""
        if agent_name not in self.calibration_history:
            return 0.5  # Default

        history = self.calibration_history[agent_name]
        if not history:
            return 0.5

        # Calculate Brier score (lower is better calibration)
        brier_sum = 0.0
        for confidence, was_correct in history:
            brier_sum += (confidence - (1 if was_correct else 0)) ** 2

        brier_score = brier_sum / len(history)

        # Convert to 0-1 scale (0 = perfect calibration, 1 = worst)
        # Brier score of 0 = perfect, 0.25 = no skill, 1 = anti-skill
        calibration_quality = 1 - (brier_score * 4)  # Scale so 0->1, 0.25->0, 1->-3-> -1->0
        return max(0, min(1, calibration_quality))

    def record_outcome(self, agent_name: str, confidence: float, was_correct: bool):
        """Record prediction outcome for calibration."""
        if agent_name not in self.calibration_history:
            self.calibration_history[agent_name] = []
        self.calibration_history[agent_name].append((confidence, was_correct))

        # Keep recent history
        if len(self.calibration_history[agent_name]) > 50:
            self.calibration_history[agent_name] = self.calibration_history[agent_name][-25:]

    def analyze_disagreement(
        self,
        messages: List[Message],
        votes: List[Vote],
        proposals: Dict[str, str],
    ) -> UncertaintyMetrics:
        """Analyze disagreement using the shared analyzer."""
        return self.disagreement_analyzer.analyze_disagreement(messages, votes, proposals)


class DisagreementAnalyzer:
    """Analyzes why agents disagree in debates."""

    def __init__(self):
        self.nlp_keywords = {
            "factual": ["fact", "evidence", "data", "proven", "false", "true", "accurate"],
            "value": ["should", "better", "prefer", "ethical", "moral", "worth"],
            "definitional": ["means", "definition", "term", "concept", "understand"],
            "asymmetry": ["don't know", "unclear", "need more", "insufficient"],
        }

    def analyze_disagreement(
        self, messages: List[Message], votes: List[Vote], proposals: Dict[str, str]
    ) -> UncertaintyMetrics:
        """Analyze disagreement patterns in debate messages."""
        metrics = UncertaintyMetrics()

        # Analyze vote distribution
        vote_choices = [v.choice for v in votes]
        vote_counts = Counter(vote_choices)

        if len(vote_counts) > 1:  # There's disagreement
            # Find majority and minority positions
            majority_choice = vote_counts.most_common(1)[0][0]
            minority_votes = [v for v in votes if v.choice != majority_choice]

            # Classify disagreement type
            metrics.disagreement_type = self._classify_disagreement_type(messages, minority_votes)

            # Identify cruxes
            metrics.cruxes = self._find_cruxes(messages, proposals, majority_choice, minority_votes)

            # Calculate collective confidence
            confidences = [v.confidence for v in votes]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                # Reduce confidence when there's disagreement
                disagreement_penalty = len(vote_counts) / len(
                    votes
                )  # More options = more disagreement
                metrics.collective_confidence = avg_confidence * (1 - disagreement_penalty * 0.3)

                # Calculate confidence interval
                variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
                std_dev = math.sqrt(variance)
                margin = 1.96 * std_dev / math.sqrt(len(confidences))  # 95% CI
                metrics.confidence_interval = (
                    max(0, metrics.collective_confidence - margin),
                    min(1, metrics.collective_confidence + margin),
                )
        else:
            # Unanimous agreement
            if votes:
                metrics.collective_confidence = sum(v.confidence for v in votes) / len(votes)
            else:
                metrics.collective_confidence = 0.5  # Default when no votes
            metrics.confidence_interval = (
                metrics.collective_confidence - 0.1,
                metrics.collective_confidence + 0.1,
            )
            metrics.disagreement_type = "none"

        return metrics

    def _classify_disagreement_type(
        self, messages: List[Message], minority_votes: List[Vote]
    ) -> str:
        """Classify the type of disagreement."""
        # Analyze messages from dissenting agents
        dissenting_agents = {v.agent for v in minority_votes}
        dissenting_messages = [
            m for m in messages if m.agent in dissenting_agents and m.role == "critic"
        ]

        keyword_counts: Counter[str] = Counter()
        for message in dissenting_messages:
            content_lower = message.content.lower()
            for category, keywords in self.nlp_keywords.items():
                for keyword in keywords:
                    if keyword in content_lower:
                        keyword_counts[category] += 1

        if keyword_counts:
            return keyword_counts.most_common(1)[0][0]
        else:
            return "general"

    def _find_cruxes(
        self,
        messages: List[Message],
        proposals: Dict[str, str],
        majority_choice: str,
        minority_votes: List[Vote],
    ) -> List[DisagreementCrux]:
        """Find key points of disagreement (cruxes)."""
        cruxes = []
        dissenting_agents = [v.agent for v in minority_votes]

        # Look for critic messages that mention disagreements
        critic_messages = [m for m in messages if m.role == "critic"]

        for message in critic_messages:
            if message.agent in dissenting_agents:
                # Simple pattern matching for crux identification
                content = message.content.lower()

                if "but" in content or "however" in content or "disagree" in content:
                    # Extract potential crux
                    crux_desc = self._extract_crux_description(message.content)

                    if crux_desc:
                        crux = DisagreementCrux(
                            description=crux_desc,
                            divergent_agents=[message.agent],
                            evidence_needed="Additional data or clarification needed",
                            severity=0.6,
                        )
                        cruxes.append(crux)

        # Merge similar cruxes
        merged_cruxes = self._merge_similar_cruxes(cruxes)

        return merged_cruxes[:3]  # Top 3 cruxes

    def _extract_crux_description(self, content: str) -> Optional[str]:
        """Extract a concise description of a crux from message content."""
        # Simple extraction - look for sentences with disagreement markers
        sentences = content.split(".")
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and any(
                word in sentence.lower() for word in ["but", "however", "disagree", "concern"]
            ):
                return sentence[:100] + "..." if len(sentence) > 100 else sentence

        return None

    def _merge_similar_cruxes(self, cruxes: List[DisagreementCrux]) -> List[DisagreementCrux]:
        """Merge cruxes that describe similar issues."""
        merged: list[DisagreementCrux] = []

        for crux in cruxes:
            # Check if similar to existing
            found_similar = False
            for existing in merged:
                if self._cruxes_similar(crux, existing):
                    # Merge agents
                    existing.divergent_agents.extend(crux.divergent_agents)
                    existing.divergent_agents = list(set(existing.divergent_agents))
                    existing.severity = max(existing.severity, crux.severity)
                    found_similar = True
                    break

            if not found_similar:
                merged.append(crux)

        return merged

    def _cruxes_similar(self, crux1: DisagreementCrux, crux2: DisagreementCrux) -> bool:
        """Check if two cruxes are similar."""
        desc1 = crux1.description.lower()
        desc2 = crux2.description.lower()

        # Simple similarity check - shared keywords
        words1 = set(desc1.split())
        words2 = set(desc2.split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if union:
            similarity = len(intersection) / len(union)
            return similarity > 0.3  # 30% word overlap

        return False

    def suggest_followups(
        self,
        cruxes: List[DisagreementCrux],
        parent_debate_id: Optional[str] = None,
        available_agents: Optional[List[str]] = None,
    ) -> List[FollowUpSuggestion]:
        """
        Generate follow-up debate suggestions from identified cruxes.

        Args:
            cruxes: List of disagreement cruxes from a debate
            parent_debate_id: ID of the parent debate for lineage tracking
            available_agents: Optional list of agents that can participate

        Returns:
            List of FollowUpSuggestion ordered by priority
        """
        suggestions = []

        for crux in cruxes:
            # Generate a focused task from the crux description
            task = self._generate_followup_task(crux)

            # Calculate priority based on severity and number of divergent agents
            priority = crux.severity * 0.6 + min(len(crux.divergent_agents) / 3, 1.0) * 0.4

            # Suggest agents: include divergent agents plus others if available
            suggested_agents = list(crux.divergent_agents)
            if available_agents:
                # Add agents not already in the list
                for agent in available_agents:
                    if agent not in suggested_agents and len(suggested_agents) < 4:
                        suggested_agents.append(agent)

            suggestion = FollowUpSuggestion(
                crux=crux,
                suggested_task=task,
                priority=priority,
                parent_debate_id=parent_debate_id,
                suggested_agents=suggested_agents,
            )
            suggestions.append(suggestion)

        # Sort by priority descending
        suggestions.sort(key=lambda s: s.priority, reverse=True)

        return suggestions

    def _generate_followup_task(self, crux: DisagreementCrux) -> str:
        """Generate a focused debate task from a crux."""
        description = crux.description.strip()

        # Clean up the description
        if description.endswith("..."):
            description = description[:-3].strip()

        # Frame as a question or investigation
        if description.lower().startswith(("but ", "however ", "i disagree")):
            # Extract the core concern
            clean = description.split(",", 1)[-1].strip() if "," in description else description
            return f"Investigate: {clean}"
        elif "?" in description:
            return f"Resolve: {description}"
        else:
            return f"Debate: Should we accept that {description.lower()}?"


class UncertaintyAggregator:
    """Aggregates uncertainty from multiple sources."""

    def __init__(
        self, confidence_estimator: ConfidenceEstimator, disagreement_analyzer: DisagreementAnalyzer
    ):
        self.confidence_estimator = confidence_estimator
        self.disagreement_analyzer = disagreement_analyzer

    async def compute_uncertainty(
        self,
        agents: List[Agent],
        messages: List[Message],
        votes: List[Vote],
        proposals: Dict[str, str],
    ) -> UncertaintyMetrics:
        """Compute comprehensive uncertainty metrics for a debate."""

        # Collect confidences (triggers calibration tracking)
        await self.confidence_estimator.collect_confidences(agents, proposals, "")

        # Analyze disagreement
        metrics = self.disagreement_analyzer.analyze_disagreement(messages, votes, proposals)

        # Incorporate calibration quality
        agent_names = [a.name for a in agents]
        calibration_scores = [
            self.confidence_estimator.get_agent_calibration_quality(name) for name in agent_names
        ]

        if calibration_scores:
            avg_calibration = sum(calibration_scores) / len(calibration_scores)
            metrics.calibration_quality = avg_calibration

            # Adjust collective confidence based on calibration
            metrics.collective_confidence *= (
                0.5 + avg_calibration * 0.5
            )  # Scale toward calibrated agents

        return metrics
