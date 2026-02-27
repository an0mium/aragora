"""Core types for the Prompt-to-Spec engine.

Defines the data flow from vague user prompt through structured intent,
clarifying questions, research, and finally a formal specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class IntentType(str, Enum):
    """Classification of what the user wants to accomplish."""

    FEATURE = "feature"
    IMPROVEMENT = "improvement"
    INVESTIGATION = "investigation"
    FIX = "fix"
    STRATEGIC = "strategic"


class ScopeEstimate(str, Enum):
    """Rough scope estimate for the intent."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    EPIC = "epic"


class InterrogationDepth(str, Enum):
    """How many clarifying questions to generate."""

    QUICK = "quick"  # 3-5 questions
    THOROUGH = "thorough"  # 10-15 questions
    EXHAUSTIVE = "exhaustive"  # 20+ questions


class AutonomyLevel(str, Enum):
    """How autonomous the system should be."""

    FULL_AUTO = "full_auto"
    PROPOSE_AND_APPROVE = "propose_and_approve"
    HUMAN_GUIDED = "human_guided"
    METRICS_DRIVEN = "metrics_driven"


class UserProfile(str, Enum):
    """User persona that determines default settings."""

    FOUNDER = "founder"
    CTO = "cto"
    BUSINESS = "business"
    TEAM = "team"


@dataclass
class Ambiguity:
    """Something in the prompt that needs clarification."""

    description: str
    impact: str  # What changes based on resolution
    options: list[str] = field(default_factory=list)
    recommended: str | None = None


@dataclass
class Assumption:
    """An implicit assumption detected in the prompt."""

    description: str
    confidence: float  # How confident we are this assumption is correct
    alternative: str | None = None  # What if this assumption is wrong


@dataclass
class PromptIntent:
    """Structured decomposition of a vague user prompt."""

    raw_prompt: str
    intent_type: IntentType
    summary: str  # One-sentence summary of what the user wants
    domains: list[str]  # Affected areas of the codebase/product
    ambiguities: list[Ambiguity] = field(default_factory=list)
    assumptions: list[Assumption] = field(default_factory=list)
    scope_estimate: ScopeEstimate = ScopeEstimate.MEDIUM
    related_knowledge: list[dict[str, Any]] = field(default_factory=list)
    decomposed_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def needs_clarification(self) -> bool:
        """Whether this intent has unresolved ambiguities."""
        return len(self.ambiguities) > 0

    @property
    def high_impact_ambiguities(self) -> list[Ambiguity]:
        """Ambiguities that should be resolved before proceeding."""
        return [a for a in self.ambiguities if a.recommended is None]


@dataclass
class QuestionOption:
    """A suggested answer for a clarifying question."""

    label: str
    description: str
    tradeoffs: str = ""


@dataclass
class ClarifyingQuestion:
    """A question to ask the user to resolve an ambiguity."""

    question: str
    why_it_matters: str
    options: list[QuestionOption] = field(default_factory=list)
    default: str | None = None
    ambiguity_ref: Ambiguity | None = None
    answer: str | None = None  # Filled when user responds

    @property
    def is_answered(self) -> bool:
        """Whether the user has answered this question."""
        return self.answer is not None


@dataclass
class EvidenceLink:
    """A link to evidence supporting a research finding."""

    source: str  # "km", "obsidian", "web", "codebase"
    title: str
    url: str | None = None
    relevance: float = 1.0
    snippet: str = ""


@dataclass
class ResearchReport:
    """Research findings about the user's intent."""

    summary: str
    current_state: str  # What exists now
    related_decisions: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[EvidenceLink] = field(default_factory=list)
    competitive_analysis: str = ""
    recommendations: list[str] = field(default_factory=list)
    researched_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SpecRisk:
    """A risk identified in the specification."""

    description: str
    likelihood: str  # low, medium, high
    impact: str  # low, medium, high
    mitigation: str


@dataclass
class SpecFile:
    """A file change described in the specification."""

    path: str
    action: str  # create, modify, delete
    description: str
    estimated_lines: int = 0


@dataclass
class SuccessCriterion:
    """A measurable criterion for success."""

    description: str
    measurement: str  # How to measure it
    target: str  # What value/state indicates success


@dataclass
class SpecProvenance:
    """Full provenance chain from original prompt to specification."""

    original_prompt: str
    intent: PromptIntent | None = None
    questions_asked: list[ClarifyingQuestion] = field(default_factory=list)
    research: ResearchReport | None = None
    debate_id: str | None = None
    prompt_hash: str = ""


@dataclass
class Specification:
    """A fully specified implementation plan derived from a vague prompt."""

    title: str
    problem_statement: str
    proposed_solution: str
    alternatives_considered: list[str] = field(default_factory=list)
    file_changes: list[SpecFile] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    risks: list[SpecRisk] = field(default_factory=list)
    success_criteria: list[SuccessCriterion] = field(default_factory=list)
    estimated_effort: str = ""
    confidence: float = 0.0  # 0-1, how confident the system is in this spec
    provenance: SpecProvenance | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_high_confidence(self) -> bool:
        """Whether the system is confident in this specification."""
        return self.confidence >= 0.8


# Default configurations per user profile
PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "founder": {
        "interrogation_depth": InterrogationDepth.QUICK,
        "auto_execute_threshold": 0.8,
        "require_approval": False,
        "show_code": True,
        "autonomy_level": AutonomyLevel.PROPOSE_AND_APPROVE,
    },
    "cto": {
        "interrogation_depth": InterrogationDepth.THOROUGH,
        "auto_execute_threshold": 0.9,
        "require_approval": True,
        "show_code": True,
        "autonomy_level": AutonomyLevel.PROPOSE_AND_APPROVE,
    },
    "business": {
        "interrogation_depth": InterrogationDepth.THOROUGH,
        "auto_execute_threshold": 0.95,
        "require_approval": True,
        "show_code": False,
        "autonomy_level": AutonomyLevel.HUMAN_GUIDED,
    },
    "team": {
        "interrogation_depth": InterrogationDepth.EXHAUSTIVE,
        "auto_execute_threshold": 1.0,
        "require_approval": True,
        "show_code": True,
        "autonomy_level": AutonomyLevel.METRICS_DRIVEN,
    },
}
