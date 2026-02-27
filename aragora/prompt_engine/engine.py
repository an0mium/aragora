"""PromptToSpecEngine: stateful orchestrator for prompt-to-spec sessions."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from aragora.prompt_engine.types import (
    ClarifyingQuestion,
    EngineStage,
    PROFILE_DEFAULTS,
    PromptIntent,
    QuestionOption,
    RefinedIntent,
    ResearchSource,
    SessionState,
    UserProfile,
)

logger = logging.getLogger(__name__)

_DEPTH_LIMITS = {"quick": 5, "thorough": 12, "exhaustive": 20}


def should_validate(
    estimated_complexity: str,
    confidence: float,
    auto_execute_threshold: float,
) -> bool:
    """Determine whether a spec needs adversarial debate validation."""
    if estimated_complexity == "high":
        return True
    return confidence < auto_execute_threshold


class PromptToSpecEngine:
    """Drives sessions through the 7-stage prompt-to-spec pipeline."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    async def start_session(
        self,
        raw_prompt: str,
        profile: UserProfile,
        research_sources: list[ResearchSource] | None = None,
    ) -> SessionState:
        session_id = str(uuid.uuid4())
        if research_sources is None:
            research_sources = [ResearchSource.KNOWLEDGE_MOUND, ResearchSource.CODEBASE]
        state = SessionState(
            session_id=session_id,
            stage=EngineStage.INTAKE,
            profile=profile,
            research_sources=research_sources,
            raw_prompt=raw_prompt,
        )
        self._sessions[session_id] = state
        return state

    def get_session(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id} not found")
        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> None:
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id} not found")
        del self._sessions[session_id]

    async def decompose(self, session_id: str) -> SessionState:
        state = self.get_session(session_id)
        if state.stage != EngineStage.INTAKE:
            raise ValueError(f"Expected stage INTAKE, got {state.stage}")

        result = await self._call_llm(
            "decompose",
            prompt=state.raw_prompt,
            profile=state.profile.value,
        )
        state.intent = PromptIntent(
            raw_prompt=state.raw_prompt,
            intent_type=result.get("intent_type", "feature"),
            domains=result.get("domains", []),
            ambiguities=result.get("ambiguities", []),
            assumptions=result.get("assumptions", []),
            scope_estimate=result.get("scope_estimate", "medium"),
        )
        state.stage = EngineStage.DECOMPOSE
        state.updated_at = datetime.now(timezone.utc)
        return state

    async def generate_questions(self, session_id: str) -> SessionState:
        state = self.get_session(session_id)
        if state.stage != EngineStage.DECOMPOSE:
            raise ValueError(f"Expected stage DECOMPOSE, got {state.stage}")

        depth = PROFILE_DEFAULTS[state.profile.value]["interrogation_depth"]
        max_questions = _DEPTH_LIMITS.get(depth, 5)

        result = await self._call_llm(
            "interrogate",
            intent=state.intent,
            max_questions=max_questions,
        )
        questions_data = result.get("questions", [])[:max_questions]
        state.questions = [
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
            for q in questions_data
        ]
        state.stage = EngineStage.INTERROGATE
        state.updated_at = datetime.now(timezone.utc)
        return state

    async def answer_question(
        self,
        session_id: str,
        question_id: str,
        answer: str,
    ) -> SessionState:
        state = self.get_session(session_id)
        state.answers[question_id] = answer
        state.updated_at = datetime.now(timezone.utc)
        return state

    async def finalize_interrogation(self, session_id: str) -> SessionState:
        state = self.get_session(session_id)
        if state.intent is None:
            raise ValueError("No intent to refine")
        confidence = min(1.0, 0.5 + 0.1 * len(state.answers))
        state.refined_intent = RefinedIntent(
            intent=state.intent,
            answers=state.answers,
            confidence=confidence,
        )
        state.stage = EngineStage.RESEARCH
        state.updated_at = datetime.now(timezone.utc)
        return state

    async def _call_llm(self, task: str, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Subclass or mock _call_llm for LLM integration")
