"""
Pattern extraction for debate evolution.

Extracts winning argument patterns and strategies from debate outcomes
to enable prompt evolution and agent improvement.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, TypedDict


class StrategyTemplate(TypedDict):
    """Type definition for strategy templates."""

    name: str
    description: str
    tactics: list[str]


logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """A pattern extracted from a debate."""

    pattern_type: str  # 'argument', 'structure', 'evidence', 'persuasion'
    description: str
    agent: Optional[str] = None
    frequency: int = 1
    effectiveness: float = 0.0
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "agent": self.agent,
            "frequency": self.frequency,
            "effectiveness": self.effectiveness,
            "examples": self.examples,
        }


@dataclass
class Strategy:
    """A debate strategy identified from outcomes."""

    name: str
    description: str
    success_rate: float = 0.0
    agent: Optional[str] = None
    tactics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "success_rate": self.success_rate,
            "agent": self.agent,
            "tactics": self.tactics,
        }


class PatternExtractor:
    """
    Extracts patterns from debate outcomes for evolution.

    Analyzes winning debates to identify:
    - Argument structures that are persuasive
    - Evidence usage patterns
    - Rhetorical techniques
    - Response strategies to critiques
    """

    # Pattern markers for identification
    EVIDENCE_MARKERS = [
        r"according to",
        r"research shows",
        r"studies indicate",
        r"evidence suggests",
        r"data from",
        r"statistics show",
        r"for example",
        r"as demonstrated by",
    ]

    STRUCTURE_MARKERS = [
        r"first(?:ly)?[\s,]",
        r"second(?:ly)?[\s,]",
        r"third(?:ly)?[\s,]",
        r"finally[\s,]",
        r"in conclusion",
        r"to summarize",
        r"on one hand.*on the other",
        r"while.*however",
    ]

    PERSUASION_MARKERS = [
        r"it's important to note",
        r"we must consider",
        r"the key point is",
        r"this demonstrates",
        r"clearly shows",
        r"fundamentally",
        r"critically",
    ]

    def __init__(self):
        """Initialize the pattern extractor."""
        self._compiled_patterns = {
            "evidence": [re.compile(p, re.IGNORECASE) for p in self.EVIDENCE_MARKERS],
            "structure": [re.compile(p, re.IGNORECASE) for p in self.STRUCTURE_MARKERS],
            "persuasion": [re.compile(p, re.IGNORECASE) for p in self.PERSUASION_MARKERS],
        }

    def extract(self, debate_outcome: dict) -> list[Pattern]:
        """
        Extract patterns from a debate outcome.

        Args:
            debate_outcome: Debate result containing messages, winner, etc.

        Returns:
            List of extracted patterns
        """
        patterns = []
        winner = debate_outcome.get("winner")
        messages = debate_outcome.get("messages", [])
        critiques = debate_outcome.get("critiques", [])

        # Get winning agent's messages
        winner_messages = [m.get("content", "") for m in messages if m.get("agent") == winner]

        # Extract evidence usage patterns
        evidence_patterns = self._extract_evidence_patterns(winner_messages, winner)
        patterns.extend(evidence_patterns)

        # Extract structural patterns
        structure_patterns = self._extract_structure_patterns(winner_messages, winner)
        patterns.extend(structure_patterns)

        # Extract persuasion patterns
        persuasion_patterns = self._extract_persuasion_patterns(winner_messages, winner)
        patterns.extend(persuasion_patterns)

        # Extract critique response patterns
        response_patterns = self._extract_response_patterns(critiques, messages, winner)
        patterns.extend(response_patterns)

        return patterns

    def _extract_evidence_patterns(
        self, messages: list[str], agent: Optional[str]
    ) -> list[Pattern]:
        """Extract evidence usage patterns from messages."""
        patterns: list[Pattern] = []
        evidence_count = 0
        examples: list[str] = []

        for msg in messages:
            for pattern in self._compiled_patterns["evidence"]:
                matches = pattern.findall(msg)
                if matches:
                    evidence_count += len(matches)
                    # Extract context around match
                    for match in pattern.finditer(msg):
                        start = max(0, match.start() - 50)
                        end = min(len(msg), match.end() + 100)
                        context = msg[start:end].strip()
                        if len(examples) < 3:
                            examples.append(context)

        if evidence_count > 0:
            patterns.append(
                Pattern(
                    pattern_type="evidence",
                    description="Uses evidence and citations to support arguments",
                    agent=agent,
                    frequency=evidence_count,
                    effectiveness=min(1.0, evidence_count / 3),  # Normalize
                    examples=examples,
                )
            )

        return patterns

    def _extract_structure_patterns(
        self, messages: list[str], agent: Optional[str]
    ) -> list[Pattern]:
        """Extract argument structure patterns from messages."""
        patterns: list[Pattern] = []
        structure_count = 0
        examples: list[str] = []

        for msg in messages:
            for pattern in self._compiled_patterns["structure"]:
                matches = pattern.findall(msg)
                if matches:
                    structure_count += len(matches)
                    if len(examples) < 3:
                        for match in pattern.finditer(msg):
                            start = max(0, match.start() - 20)
                            end = min(len(msg), match.end() + 80)
                            examples.append(msg[start:end].strip())

        if structure_count > 0:
            patterns.append(
                Pattern(
                    pattern_type="structure",
                    description="Uses clear argument structure with transitions",
                    agent=agent,
                    frequency=structure_count,
                    effectiveness=min(1.0, structure_count / 4),
                    examples=examples[:3],
                )
            )

        return patterns

    def _extract_persuasion_patterns(
        self, messages: list[str], agent: Optional[str]
    ) -> list[Pattern]:
        """Extract persuasion technique patterns from messages."""
        patterns: list[Pattern] = []
        persuasion_count = 0
        examples: list[str] = []

        for msg in messages:
            for pattern in self._compiled_patterns["persuasion"]:
                matches = pattern.findall(msg)
                if matches:
                    persuasion_count += len(matches)
                    if len(examples) < 3:
                        for match in pattern.finditer(msg):
                            start = max(0, match.start() - 20)
                            end = min(len(msg), match.end() + 100)
                            examples.append(msg[start:end].strip())

        if persuasion_count > 0:
            patterns.append(
                Pattern(
                    pattern_type="persuasion",
                    description="Uses emphatic language and key point highlighting",
                    agent=agent,
                    frequency=persuasion_count,
                    effectiveness=min(1.0, persuasion_count / 3),
                    examples=examples[:3],
                )
            )

        return patterns

    def _extract_response_patterns(
        self,
        critiques: list[dict],
        messages: list[dict],
        winner: Optional[str],
    ) -> list[Pattern]:
        """Extract patterns for responding to critiques."""
        patterns: list[Pattern] = []

        if not critiques or not winner:
            return patterns

        # Find critiques directed at the winner
        critiques_to_winner = [c for c in critiques if c.get("to") == winner]

        if not critiques_to_winner:
            return patterns

        # Check if winner acknowledged and addressed critiques
        winner_responses = [m.get("content", "") for m in messages if m.get("agent") == winner]

        acknowledgment_markers = [
            r"you raise a valid point",
            r"that's a fair criticism",
            r"I acknowledge",
            r"while you're correct",
            r"addressing your concern",
            r"to respond to",
        ]

        acknowledgment_count = 0
        for response in winner_responses:
            for marker in acknowledgment_markers:
                if re.search(marker, response, re.IGNORECASE):
                    acknowledgment_count += 1

        if acknowledgment_count > 0:
            patterns.append(
                Pattern(
                    pattern_type="response",
                    description="Acknowledges critiques before countering",
                    agent=winner,
                    frequency=acknowledgment_count,
                    effectiveness=min(1.0, acknowledgment_count / 2),
                    examples=[],
                )
            )

        return patterns


class StrategyIdentifier:
    """
    Identifies successful debate strategies from outcomes.

    Analyzes patterns across multiple debates to identify
    recurring successful strategies.
    """

    STRATEGY_TEMPLATES: dict[str, StrategyTemplate] = {
        "evidence_based": {
            "name": "Evidence-Based Argumentation",
            "description": "Relies heavily on evidence, citations, and data",
            "tactics": [
                "Cite specific sources",
                "Use statistical data",
                "Reference studies",
            ],
        },
        "structured": {
            "name": "Structured Reasoning",
            "description": "Uses clear logical structure with numbered points",
            "tactics": [
                "Use enumeration",
                "Provide transitions",
                "Summarize key points",
            ],
        },
        "conciliatory": {
            "name": "Conciliatory Approach",
            "description": "Acknowledges valid points while building own case",
            "tactics": [
                "Acknowledge critiques",
                "Find common ground",
                "Build bridges",
            ],
        },
        "direct": {
            "name": "Direct Challenge",
            "description": "Directly addresses and refutes opposing arguments",
            "tactics": [
                "Challenge assumptions",
                "Point out flaws",
                "Request evidence",
            ],
        },
    }

    def identify(self, debate_outcome: dict) -> list[Strategy]:
        """
        Identify strategies used in a debate outcome.

        Args:
            debate_outcome: Debate result containing messages, winner, etc.

        Returns:
            List of identified strategies
        """
        strategies: list[Strategy] = []
        winner = debate_outcome.get("winner")
        messages = debate_outcome.get("messages", [])
        consensus_reached = debate_outcome.get("consensus_reached", False)
        # Get winning agent's messages
        winner_messages = [m.get("content", "") for m in messages if m.get("agent") == winner]

        if not winner_messages:
            return strategies

        combined_text = " ".join(winner_messages).lower()

        # Check for evidence-based strategy
        evidence_score = self._score_evidence_strategy(combined_text)
        if evidence_score > 0.3:
            template = self.STRATEGY_TEMPLATES["evidence_based"]
            strategies.append(
                Strategy(
                    name=template["name"],
                    description=template["description"],
                    success_rate=evidence_score,
                    agent=winner,
                    tactics=template["tactics"],
                )
            )

        # Check for structured strategy
        structure_score = self._score_structure_strategy(combined_text)
        if structure_score > 0.3:
            template = self.STRATEGY_TEMPLATES["structured"]
            strategies.append(
                Strategy(
                    name=template["name"],
                    description=template["description"],
                    success_rate=structure_score,
                    agent=winner,
                    tactics=template["tactics"],
                )
            )

        # Check for conciliatory strategy
        conciliatory_score = self._score_conciliatory_strategy(combined_text)
        if conciliatory_score > 0.3:
            template = self.STRATEGY_TEMPLATES["conciliatory"]
            strategies.append(
                Strategy(
                    name=template["name"],
                    description=template["description"],
                    success_rate=conciliatory_score,
                    agent=winner,
                    tactics=template["tactics"],
                )
            )

        # Boost scores if consensus was reached
        if consensus_reached:
            for strategy in strategies:
                strategy.success_rate = min(1.0, strategy.success_rate * 1.2)

        return strategies

    def _score_evidence_strategy(self, text: str) -> float:
        """Score how much the text uses evidence-based argumentation."""
        markers = [
            "according to",
            "research",
            "study",
            "data",
            "evidence",
            "statistics",
            "source",
            "citation",
            "shows that",
            "demonstrates",
        ]
        count = sum(1 for m in markers if m in text)
        return min(1.0, count / 5)

    def _score_structure_strategy(self, text: str) -> float:
        """Score how structured the argumentation is."""
        markers = [
            "first",
            "second",
            "third",
            "finally",
            "in conclusion",
            "therefore",
            "thus",
            "hence",
            "consequently",
            "as a result",
        ]
        count = sum(1 for m in markers if m in text)
        return min(1.0, count / 4)

    def _score_conciliatory_strategy(self, text: str) -> float:
        """Score how conciliatory the approach is."""
        markers = [
            "you're right",
            "valid point",
            "i agree",
            "fair criticism",
            "acknowledge",
            "however",
            "while true",
            "that said",
        ]
        count = sum(1 for m in markers if m in text)
        return min(1.0, count / 3)


# Module-level convenience functions
_extractor = PatternExtractor()
_identifier = StrategyIdentifier()


def extract_patterns(debate_outcome: dict) -> dict:
    """
    Extract patterns from a debate outcome.

    Convenience function for module-level access.

    Args:
        debate_outcome: Debate result dict

    Returns:
        Dict containing winning_patterns and other extracted patterns
    """
    patterns = _extractor.extract(debate_outcome)

    return {
        "winning_patterns": [p.to_dict() for p in patterns],
        "winner": debate_outcome.get("winner"),
        "pattern_count": len(patterns),
    }


def identify_strategies(debate_outcome: dict) -> list[dict]:
    """
    Identify strategies from a debate outcome.

    Convenience function for module-level access.

    Args:
        debate_outcome: Debate result dict

    Returns:
        List of identified strategy dicts
    """
    strategies = _identifier.identify(debate_outcome)
    return [s.to_dict() for s in strategies]
