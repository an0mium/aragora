"""HTTP handlers for the Interrogation Engine."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

from aragora.interrogation.engine import InterrogationEngine
from aragora.rbac.decorators import require_permission

logger = logging.getLogger(__name__)

# In-memory session store (production: Redis)
_sessions: dict[str, Any] = {}


@dataclass
class _InterrogationDimension:
    name: str
    description: str
    vagueness_score: float = 0.5


@dataclass
class _InterrogationQuestion:
    text: str
    why: str
    options: list[str] = field(default_factory=list)
    context: str = ""
    priority: int = 1


@dataclass
class _InterrogationStateCompat:
    prompt: str
    dimensions: list[_InterrogationDimension]
    questions: list[_InterrogationQuestion]
    answers: dict[str, str] = field(default_factory=dict)

    @property
    def unanswered(self) -> list[_InterrogationQuestion]:
        return [q for q in self.questions if q.text not in self.answers]

    @property
    def is_complete(self) -> bool:
        return len(self.unanswered) == 0


@dataclass
class _RequirementLevelCompat:
    value: str


@dataclass
class _RequirementCompat:
    description: str
    level: _RequirementLevelCompat
    dimension: str


@dataclass
class _SpecCompat:
    problem_statement: str
    requirements: list[_RequirementCompat]
    non_requirements: list[str]
    success_criteria: list[str]
    risks: list[str]
    context_summary: str

    def to_goal_text(self) -> str:
        return self.problem_statement


class InterrogationHandler:
    """Handles HTTP requests for the Interrogation Engine."""

    def __init__(self) -> None:
        self._engine = InterrogationEngine()

    async def _start_state(self, prompt: str, sources: list[str]) -> Any:
        """Start session using engine if available, otherwise use compat fallback."""
        if hasattr(self._engine, "start"):
            return await self._engine.start(prompt, sources=sources)

        if hasattr(self._engine, "interrogate"):
            result = await self._engine.interrogate(prompt)
            dimensions: list[_InterrogationDimension] = []
            for dim in result.dimensions:
                name, _, description = dim.partition(":")
                dimensions.append(
                    _InterrogationDimension(
                        name=name.strip() or "scope",
                        description=description.strip() or dim.strip(),
                        vagueness_score=0.5,
                    )
                )

            questions = [
                _InterrogationQuestion(
                    text=q.question,
                    why=q.why_it_matters or "Clarifies implementation direction.",
                    options=q.options or ["Yes", "No"],
                    context=q.hidden_assumption,
                    priority=max(1, min(10, int(round(q.priority_score * 10)))),
                )
                for q in result.prioritized_questions
            ]

            if dimensions and questions:
                return _InterrogationStateCompat(
                    prompt=prompt,
                    dimensions=dimensions,
                    questions=questions,
                )

        return _InterrogationStateCompat(
            prompt=prompt,
            dimensions=[
                _InterrogationDimension(
                    name="scope",
                    description="Clarify desired outcome and boundaries.",
                    vagueness_score=0.7,
                )
            ],
            questions=[
                _InterrogationQuestion(
                    text="What concrete outcome should this deliver?",
                    why="Defines success and implementation direction.",
                    options=["Speed", "Reliability", "User experience"],
                    context="Assumes the objective is primarily technical.",
                    priority=1,
                )
            ],
        )

    async def _record_answer(self, state: Any, question: str, answer: str) -> None:
        """Record answer using engine if available, otherwise compat state map."""
        if hasattr(self._engine, "answer"):
            self._engine.answer(state, question, answer)
            return

        state.answers[question] = answer

    async def _crystallize_spec(self, state: Any) -> Any:
        """Crystallize via engine when available, else generate compat spec."""
        if hasattr(self._engine, "crystallize"):
            result = await self._engine.crystallize(state)
            return result.spec

        requirements = [
            _RequirementCompat(
                description=f"{question}: {answer}",
                level=_RequirementLevelCompat("must"),
                dimension="interrogation",
            )
            for question, answer in state.answers.items()
        ]
        if not requirements:
            requirements.append(
                _RequirementCompat(
                    description="Define a measurable success outcome before execution.",
                    level=_RequirementLevelCompat("must"),
                    dimension="scope",
                )
            )

        return _SpecCompat(
            problem_statement=state.prompt,
            requirements=requirements,
            non_requirements=["Unscoped enhancements without explicit requirement."],
            success_criteria=["User confirms the crystallized objective is accurate."],
            risks=["Ambiguity in requirements can lead to rework."],
            context_summary="Generated from interrogation answers.",
        )

    @require_permission("debates:create")
    async def handle_start(self, request: web.Request) -> web.Response:
        """POST /api/v1/interrogation/start

        Begin an interrogation session from a vague prompt.

        Request body:
            {"prompt": "Make it better", "sources": ["knowledge_mound"]}

        Response:
            {"data": {"session_id": "...", "prompt": "...", "dimensions": [...], "questions": [...]}}
        """
        try:
            body = await request.read()
            data = json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError):
            return web.json_response({"error": "Invalid JSON"}, status=400)

        prompt = data.get("prompt", "")
        if not prompt:
            return web.json_response({"error": "prompt is required"}, status=400)

        sources = data.get("sources", [])
        state = await self._start_state(prompt, sources=sources)

        session_id = str(uuid.uuid4())
        _sessions[session_id] = state

        return web.json_response(
            {
                "data": {
                    "session_id": session_id,
                    "prompt": state.prompt,
                    "dimensions": [
                        {
                            "name": d.name,
                            "description": d.description,
                            "vagueness_score": d.vagueness_score,
                        }
                        for d in state.dimensions
                    ],
                    "questions": [
                        {
                            "text": q.text,
                            "why": q.why,
                            "options": q.options,
                            "context": q.context,
                            "priority": q.priority,
                        }
                        for q in state.questions
                    ],
                }
            }
        )

    @require_permission("debates:create")
    async def handle_answer(self, request: web.Request) -> web.Response:
        """POST /api/v1/interrogation/answer

        Record a user answer to an interrogation question.

        Request body:
            {"session_id": "...", "question": "...", "answer": "..."}

        Response:
            {"data": {"session_id": "...", "answered": N, "remaining": N, "is_complete": bool}}
        """
        try:
            body = await request.read()
            data = json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError):
            return web.json_response({"error": "Invalid JSON"}, status=400)

        session_id = data.get("session_id", "")
        state = _sessions.get(session_id)
        if not state:
            return web.json_response({"error": "Session not found"}, status=404)

        question = data.get("question", "")
        answer = data.get("answer", "")
        if not question or not answer:
            return web.json_response({"error": "question and answer required"}, status=400)

        await self._record_answer(state, question, answer)

        return web.json_response(
            {
                "data": {
                    "session_id": session_id,
                    "answered": len(state.answers),
                    "remaining": len(state.unanswered),
                    "is_complete": state.is_complete,
                }
            }
        )

    @require_permission("debates:create")
    async def handle_crystallize(self, request: web.Request) -> web.Response:
        """POST /api/v1/interrogation/crystallize

        Crystallize the interrogation state into a structured spec.

        Request body:
            {"session_id": "..."}

        Response:
            {"data": {"session_id": "...", "spec": {...}, "goal_text": "..."}}
        """
        try:
            body = await request.read()
            data = json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError):
            return web.json_response({"error": "Invalid JSON"}, status=400)

        session_id = data.get("session_id", "")
        state = _sessions.get(session_id)
        if not state:
            return web.json_response({"error": "Session not found"}, status=404)

        spec = await self._crystallize_spec(state)

        return web.json_response(
            {
                "data": {
                    "session_id": session_id,
                    "spec": {
                        "problem_statement": spec.problem_statement,
                        "requirements": [
                            {
                                "description": r.description,
                                "level": r.level.value,
                                "dimension": r.dimension,
                            }
                            for r in spec.requirements
                        ],
                        "non_requirements": spec.non_requirements,
                        "success_criteria": spec.success_criteria,
                        "risks": spec.risks,
                        "context_summary": spec.context_summary,
                    },
                    "goal_text": spec.to_goal_text(),
                }
            }
        )

    def register_routes(self, app: web.Application) -> None:
        """Register interrogation routes on an aiohttp app."""
        app.router.add_post("/api/v1/interrogation/start", self.handle_start)
        app.router.add_post("/api/v1/interrogation/answer", self.handle_answer)
        app.router.add_post("/api/v1/interrogation/crystallize", self.handle_crystallize)
