"""Core types for the Prompt-to-Spec Engine."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EngineStage(str, Enum):
    INTAKE = "intake"
    DECOMPOSE = "decompose"
    INTERROGATE = "interrogate"
    RESEARCH = "research"
    SPEC = "spec"
    VALIDATE = "validate"
    HANDOFF = "handoff"


class UserProfile(str, Enum):
    FOUNDER = "founder"
    CTO = "cto"
    BUSINESS = "business"
    TEAM = "team"


class ResearchSource(str, Enum):
    KNOWLEDGE_MOUND = "km"
    CODEBASE = "codebase"
    OBSIDIAN = "obsidian"
    WEB = "web"


PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
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


@dataclass
class QuestionOption:
    label: str
    description: str
    tradeoff: str


@dataclass
class ClarifyingQuestion:
    id: str
    question: str
    why_it_matters: str
    options: list[QuestionOption]
    default_option: str | None = None
    impact: str = "medium"


@dataclass
class PromptIntent:
    raw_prompt: str
    intent_type: str
    domains: list[str]
    ambiguities: list[str]
    assumptions: list[str]
    scope_estimate: str
    enriched_dump: Any | None = None


@dataclass
class RefinedIntent:
    intent: PromptIntent
    answers: dict[str, str]
    confidence: float


@dataclass
class ResearchReport:
    km_precedents: list[dict[str, Any]]
    codebase_context: list[dict[str, Any]]
    obsidian_notes: list[dict[str, Any]]
    web_results: list[dict[str, Any]]
    sources_used: list[ResearchSource]


@dataclass
class SessionState:
    session_id: str
    stage: EngineStage
    profile: UserProfile
    research_sources: list[ResearchSource]
    raw_prompt: str
    intent: PromptIntent | None = None
    questions: list[ClarifyingQuestion] = field(default_factory=list)
    answers: dict[str, str] = field(default_factory=dict)
    refined_intent: RefinedIntent | None = None
    research: ResearchReport | None = None
    spec: Any | None = None
    validation_result: Any | None = None
    pipeline_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        if not self.provenance_hash:
            self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        content = f"{self.session_id}:{self.stage.value}:{self.raw_prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "session_id": self.session_id,
            "stage": self.stage.value,
            "profile": self.profile.value,
            "research_sources": [s.value for s in self.research_sources],
            "raw_prompt": self.raw_prompt,
            "answers": self.answers,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "provenance_hash": self.provenance_hash,
            "pipeline_id": self.pipeline_id,
        }
        if self.intent:
            d["intent"] = {
                "raw_prompt": self.intent.raw_prompt,
                "intent_type": self.intent.intent_type,
                "domains": self.intent.domains,
                "ambiguities": self.intent.ambiguities,
                "assumptions": self.intent.assumptions,
                "scope_estimate": self.intent.scope_estimate,
            }
        if self.questions:
            d["questions"] = [
                {
                    "id": q.id,
                    "question": q.question,
                    "why_it_matters": q.why_it_matters,
                    "options": [
                        {"label": o.label, "description": o.description, "tradeoff": o.tradeoff}
                        for o in q.options
                    ],
                    "default_option": q.default_option,
                    "impact": q.impact,
                }
                for q in self.questions
            ]
        if self.refined_intent:
            d["refined_intent"] = {
                "answers": self.refined_intent.answers,
                "confidence": self.refined_intent.confidence,
            }
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionState:
        intent = None
        if "intent" in data:
            i = data["intent"]
            intent = PromptIntent(
                raw_prompt=i["raw_prompt"],
                intent_type=i["intent_type"],
                domains=i["domains"],
                ambiguities=i["ambiguities"],
                assumptions=i["assumptions"],
                scope_estimate=i["scope_estimate"],
            )
        questions = []
        if "questions" in data:
            for q in data["questions"]:
                questions.append(
                    ClarifyingQuestion(
                        id=q["id"],
                        question=q["question"],
                        why_it_matters=q["why_it_matters"],
                        options=[
                            QuestionOption(o["label"], o["description"], o["tradeoff"])
                            for o in q.get("options", [])
                        ],
                        default_option=q.get("default_option"),
                        impact=q.get("impact", "medium"),
                    )
                )
        refined_intent = None
        if "refined_intent" in data and intent:
            ri = data["refined_intent"]
            refined_intent = RefinedIntent(
                intent=intent,
                answers=ri["answers"],
                confidence=ri["confidence"],
            )
        return cls(
            session_id=data["session_id"],
            stage=EngineStage(data["stage"]),
            profile=UserProfile(data["profile"]),
            research_sources=[ResearchSource(s) for s in data["research_sources"]],
            raw_prompt=data["raw_prompt"],
            intent=intent,
            questions=questions,
            answers=data.get("answers", {}),
            refined_intent=refined_intent,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            provenance_hash=data.get("provenance_hash", ""),
            pipeline_id=data.get("pipeline_id"),
        )
