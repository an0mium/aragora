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


class AmbiguityLevel(str, Enum):
    """How ambiguous a prompt is."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


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


class SpecificationStatus(str, Enum):
    """Lifecycle status of a specification."""

    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    REJECTED = "rejected"


# Default configurations per user profile (defined before UserProfile so it can reference them)
_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "founder": {
        "interrogation_depth": "quick",
        "auto_execute_threshold": 0.8,
        "require_approval": False,
        "show_code": True,
        "autonomy_level": "propose_and_approve",
    },
    "cto": {
        "interrogation_depth": "thorough",
        "auto_execute_threshold": 0.9,
        "require_approval": True,
        "show_code": True,
        "autonomy_level": "propose_and_approve",
    },
    "business": {
        "interrogation_depth": "thorough",
        "auto_execute_threshold": 0.95,
        "require_approval": True,
        "show_code": False,
        "autonomy_level": "human_guided",
    },
    "team": {
        "interrogation_depth": "exhaustive",
        "auto_execute_threshold": 1.0,
        "require_approval": True,
        "show_code": True,
        "autonomy_level": "metrics_driven",
    },
}


class UserProfile(str, Enum):
    """User persona that determines default settings."""

    FOUNDER = "founder"
    CTO = "cto"
    BUSINESS = "business"
    TEAM = "team"

    def default_config(self) -> dict[str, Any]:
        """Return the default configuration for this profile."""
        return dict(_PROFILE_DEFAULTS[self.value])


# Legacy alias for backwards compatibility
PROFILE_DEFAULTS: dict[str, dict[str, Any]] = _PROFILE_DEFAULTS


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
    """Structured decomposition of a vague user prompt.

    Accepts both rich typed fields (Ambiguity objects, ScopeEstimate enum) and
    simple values (plain strings) for flexibility.
    """

    raw_prompt: str
    intent_type: IntentType
    domains: list[str] = field(default_factory=list)
    ambiguities: list[Any] = field(default_factory=list)  # list[str] or list[Ambiguity]
    assumptions: list[Any] = field(default_factory=list)  # list[str] or list[Assumption]
    scope_estimate: str | ScopeEstimate = ScopeEstimate.MEDIUM
    summary: str = ""
    related_knowledge: list[dict[str, Any]] = field(default_factory=list)
    decomposed_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def needs_clarification(self) -> bool:
        """Whether this intent has unresolved ambiguities."""
        return len(self.ambiguities) > 0

    @property
    def high_impact_ambiguities(self) -> list[Ambiguity]:
        """Ambiguities that should be resolved before proceeding."""
        return [a for a in self.ambiguities if isinstance(a, Ambiguity) and a.recommended is None]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        scope = self.scope_estimate
        if isinstance(scope, ScopeEstimate):
            scope = scope.value
        ambigs: list[Any] = []
        for a in self.ambiguities:
            if isinstance(a, str):
                ambigs.append(a)
            elif isinstance(a, Ambiguity):
                ambigs.append({"description": a.description, "impact": a.impact})
            else:
                ambigs.append(a)
        assumps: list[Any] = []
        for a in self.assumptions:
            if isinstance(a, str):
                assumps.append(a)
            elif isinstance(a, Assumption):
                assumps.append({"description": a.description, "confidence": a.confidence})
            else:
                assumps.append(a)
        return {
            "raw_prompt": self.raw_prompt,
            "intent_type": self.intent_type.value
            if isinstance(self.intent_type, IntentType)
            else self.intent_type,
            "domains": list(self.domains),
            "ambiguities": ambigs,
            "assumptions": assumps,
            "scope_estimate": scope,
            "summary": self.summary,
            "decomposed_at": self.decomposed_at.isoformat(),
        }


@dataclass
class QuestionOption:
    """A suggested answer for a clarifying question."""

    label: str
    description: str
    tradeoffs: str = ""


@dataclass
class ClarifyingQuestion:
    """A question to ask the user to resolve an ambiguity.

    Accepts options as list[QuestionOption] or list[dict] for flexibility.
    """

    question: str
    why_it_matters: str
    options: list[Any] = field(default_factory=list)  # list[QuestionOption] or list[dict]
    default_option: str | None = None
    default: str | None = None  # Alias for default_option (backwards compat)
    impact_level: str = "medium"
    ambiguity_ref: Ambiguity | None = None
    answer: str | None = None  # Filled when user responds

    def __post_init__(self) -> None:
        # Sync default and default_option
        if self.default_option is not None and self.default is None:
            self.default = self.default_option
        elif self.default is not None and self.default_option is None:
            self.default_option = self.default

    @property
    def is_answered(self) -> bool:
        """Whether the user has answered this question."""
        return self.answer is not None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        opts: list[dict[str, Any]] = []
        for o in self.options:
            if isinstance(o, QuestionOption):
                opts.append(
                    {"label": o.label, "description": o.description, "tradeoffs": o.tradeoffs}
                )
            elif isinstance(o, dict):
                opts.append(dict(o))
            else:
                opts.append({"value": str(o)})
        return {
            "question": self.question,
            "why_it_matters": self.why_it_matters,
            "options": opts,
            "default_option": self.default_option,
            "impact_level": self.impact_level,
            "answer": self.answer,
        }


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
    """Research findings about the user's intent.

    All fields are optional to allow creating empty reports.
    """

    summary: str = ""
    current_state: str = ""
    codebase_findings: list[dict[str, Any]] = field(default_factory=list)
    past_decisions: list[dict[str, Any]] = field(default_factory=list)
    related_decisions: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[EvidenceLink] = field(default_factory=list)
    competitive_analysis: str = ""
    recommendations: list[str] = field(default_factory=list)
    researched_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "summary": self.summary,
            "current_state": self.current_state,
            "codebase_findings": list(self.codebase_findings),
            "past_decisions": list(self.past_decisions),
            "related_decisions": list(self.related_decisions),
            "evidence": [
                {"source": e.source, "title": e.title, "url": e.url, "relevance": e.relevance}
                if isinstance(e, EvidenceLink)
                else e
                for e in self.evidence
            ],
            "competitive_analysis": self.competitive_analysis,
            "recommendations": list(self.recommendations),
            "researched_at": self.researched_at.isoformat(),
        }


@dataclass
class RiskItem:
    """A risk identified in the specification."""

    description: str
    likelihood: str  # low, medium, high
    impact: str  # low, medium, high
    mitigation: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "description": self.description,
            "likelihood": self.likelihood,
            "impact": self.impact,
            "mitigation": self.mitigation,
        }


# Backwards-compatible alias
SpecRisk = RiskItem


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "original_prompt": self.original_prompt,
            "intent": self.intent.to_dict() if self.intent else None,
            "questions_asked": [
                q.to_dict() if hasattr(q, "to_dict") else q for q in self.questions_asked
            ],
            "research": self.research.to_dict() if self.research else None,
            "debate_id": self.debate_id,
            "prompt_hash": self.prompt_hash,
        }


@dataclass
class Specification:
    """A fully specified implementation plan derived from a vague prompt.

    Accepts both rich typed fields and simple values for flexibility.
    Fields ``implementation_plan`` and ``risk_register`` are aliases that
    map to ``file_changes`` and ``risks`` respectively.
    """

    title: str
    problem_statement: str
    proposed_solution: str
    implementation_plan: list[Any] = field(default_factory=list)
    risk_register: list[Any] = field(default_factory=list)
    success_criteria: list[Any] = field(default_factory=list)  # list[str] or list[SuccessCriterion]
    estimated_effort: str = ""
    status: SpecificationStatus = SpecificationStatus.DRAFT
    alternatives_considered: list[str] = field(default_factory=list)
    file_changes: list[SpecFile] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    risks: list[RiskItem] = field(default_factory=list)
    confidence: float = 0.0  # 0-1, how confident the system is in this spec
    provenance: SpecProvenance | None = None
    provenance_chain: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_high_confidence(self) -> bool:
        """Whether the system is confident in this specification."""
        return self.confidence >= 0.8

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        criteria: list[Any] = []
        for c in self.success_criteria:
            if isinstance(c, str):
                criteria.append(c)
            elif isinstance(c, SuccessCriterion):
                criteria.append(
                    {"description": c.description, "measurement": c.measurement, "target": c.target}
                )
            else:
                criteria.append(c)
        risk_items: list[dict[str, Any]] = []
        for r in list(self.risk_register) + list(self.risks):
            if isinstance(r, RiskItem):
                risk_items.append(r.to_dict())
            elif isinstance(r, dict):
                risk_items.append(dict(r))
            else:
                risk_items.append({"value": str(r)})
        return {
            "title": self.title,
            "problem_statement": self.problem_statement,
            "proposed_solution": self.proposed_solution,
            "implementation_plan": list(self.implementation_plan),
            "risk_register": risk_items,
            "success_criteria": criteria,
            "estimated_effort": self.estimated_effort,
            "status": self.status.value
            if isinstance(self.status, SpecificationStatus)
            else self.status,
            "alternatives_considered": list(self.alternatives_considered),
            "confidence": self.confidence,
            "provenance_chain": list(self.provenance_chain),
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "created_at": self.created_at.isoformat(),
        }
