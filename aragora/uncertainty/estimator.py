"""
Uncertainty Estimator.

Quantifies epistemic uncertainty in agent responses and debate outcomes,
providing confidence calibration and disagreement analysis.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
from datetime import datetime
import math

from aragora.core import Agent, Message, Vote, DebateResult


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
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class DisagreementCrux:
    """A key point of disagreement between agents."""

    description: str
    divergent_agents: List[str]
    evidence_needed: str = ""
    severity: float = 0.5  # 0-1, how critical this disagreement is

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "agents": self.divergent_agents,
            "evidence_needed": self.evidence_needed,
            "severity": self.severity
        }


@dataclass
class UncertaintyMetrics:
    """Comprehensive uncertainty quantification for a debate."""

    collective_confidence: float = 0.5
    confidence_interval: Tuple[float, float] = (0.4, 0.6)
    disagreement_type: str = "none"  # "factual", "value-based", "definitional", "information-asymmetry"
    cruxes: List[DisagreementCrux] = field(default_factory=list)
    calibration_quality: float = 0.5  # How well agents are calibrated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collective_confidence": self.collective_confidence,
            "confidence_interval": self.confidence_interval,
            "disagreement_type": self.disagreement_type,
            "cruxes": [crux.to_dict() for crux in self.cruxes],
            "calibration_quality": self.calibration_quality
        }


class ConfidenceEstimator:
    """Estimates confidence in agent responses and calibrates over time."""

    def __init__(self):
        self.agent_confidences: Dict[str, List[ConfidenceScore]] = {}
        self.calibration_history: Dict[str, List[Tuple[float, bool]]] = {}  # (confidence, was_correct)
        self.brier_scores: Dict[str, float] = {}  # Calibration quality metric

    async def collect_confidences(
        self,
        agents: List[Agent],
        proposals: Dict[str, str],
        task: str
    ) -> Dict[str, ConfidenceScore]:
        """Collect confidence scores from all agents."""
        confidence_tasks = []

        for agent in agents:
            confidence_tasks.append(self._get_agent_confidence(agent, proposals, task))

        results = await asyncio.gather(*confidence_tasks, return_exceptions=True)

        confidences = {}
        for agent, result in zip(agents, results):
            if isinstance(result, Exception):
                print(f"Error getting confidence from {agent.name}: {result}")
                # Default confidence
                confidences[agent.name] = ConfidenceScore(agent.name, 0.5, "Error estimating confidence")
            else:
                confidences[agent.name] = result
                # Store for calibration tracking
                self._store_confidence(agent.name, result)

        return confidences

    async def _get_agent_confidence(
        self,
        agent: Agent,
        proposals: Dict[str, str],
        task: str
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
                reasoning=f"Based on voting confidence: {vote.reasoning}"
            )
        except:
            # Fallback to default confidence
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
        brier_sum = 0
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


class DisagreementAnalyzer:
    """Analyzes why agents disagree in debates."""

    def __init__(self):
        self.nlp_keywords = {
            "factual": ["fact", "evidence", "data", "proven", "false", "true", "accurate"],
            "value": ["should", "better", "prefer", "ethical", "moral", "worth"],
            "definitional": ["means", "definition", "term", "concept", "understand"],
            "asymmetry": ["don't know", "unclear", "need more", "insufficient"]
        }

    def analyze_disagreement(
        self,
        messages: List[Message],
        votes: List[Vote],
        proposals: Dict[str, str]
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
                disagreement_penalty = len(vote_counts) / len(votes)  # More options = more disagreement
                metrics.collective_confidence = avg_confidence * (1 - disagreement_penalty * 0.3)

                # Calculate confidence interval
                variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
                std_dev = math.sqrt(variance)
                margin = 1.96 * std_dev / math.sqrt(len(confidences))  # 95% CI
                metrics.confidence_interval = (
                    max(0, metrics.collective_confidence - margin),
                    min(1, metrics.collective_confidence + margin)
                )
        else:
            # Unanimous agreement
            metrics.collective_confidence = sum(v.confidence for v in votes) / len(votes)
            metrics.confidence_interval = (metrics.collective_confidence - 0.1, metrics.collective_confidence + 0.1)
            metrics.disagreement_type = "none"

        return metrics

    def _classify_disagreement_type(self, messages: List[Message], minority_votes: List[Vote]) -> str:
        """Classify the type of disagreement."""
        # Analyze messages from dissenting agents
        dissenting_agents = {v.agent for v in minority_votes}
        dissenting_messages = [m for m in messages if m.agent in dissenting_agents and m.role == "critic"]

        keyword_counts = Counter()
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
        minority_votes: List[Vote]
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
                            severity=0.6
                        )
                        cruxes.append(crux)

        # Merge similar cruxes
        merged_cruxes = self._merge_similar_cruxes(cruxes)

        return merged_cruxes[:3]  # Top 3 cruxes

    def _extract_crux_description(self, content: str) -> Optional[str]:
        """Extract a concise description of a crux from message content."""
        # Simple extraction - look for sentences with disagreement markers
        sentences = content.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and any(word in sentence.lower() for word in ["but", "however", "disagree", "concern"]):
                return sentence[:100] + "..." if len(sentence) > 100 else sentence

        return None

    def _merge_similar_cruxes(self, cruxes: List[DisagreementCrux]) -> List[DisagreementCrux]:
        """Merge cruxes that describe similar issues."""
        merged = []

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


class UncertaintyAggregator:
    """Aggregates uncertainty from multiple sources."""

    def __init__(self, confidence_estimator: ConfidenceEstimator, disagreement_analyzer: DisagreementAnalyzer):
        self.confidence_estimator = confidence_estimator
        self.disagreement_analyzer = disagreement_analyzer

    async def compute_uncertainty(
        self,
        agents: List[Agent],
        messages: List[Message],
        votes: List[Vote],
        proposals: Dict[str, str]
    ) -> UncertaintyMetrics:
        """Compute comprehensive uncertainty metrics for a debate."""

        # Collect confidences
        confidences = await self.confidence_estimator.collect_confidences(agents, proposals, "")

        # Analyze disagreement
        metrics = self.disagreement_analyzer.analyze_disagreement(messages, votes, proposals)

        # Incorporate calibration quality
        agent_names = [a.name for a in agents]
        calibration_scores = [self.confidence_estimator.get_agent_calibration_quality(name) for name in agent_names]

        if calibration_scores:
            avg_calibration = sum(calibration_scores) / len(calibration_scores)
            metrics.calibration_quality = avg_calibration

            # Adjust collective confidence based on calibration
            metrics.collective_confidence *= (0.5 + avg_calibration * 0.5)  # Scale toward calibrated agents

        return metrics