"""HTTP handlers for the Interrogation Engine."""

from __future__ import annotations

import json
import logging
import uuid

from aiohttp import web

from aragora.interrogation.engine import InterrogationEngine, InterrogationState

logger = logging.getLogger(__name__)

# In-memory session store (production: Redis)
_sessions: dict[str, InterrogationState] = {}


class InterrogationHandler:
    """Handles HTTP requests for the Interrogation Engine."""

    def __init__(self) -> None:
        self._engine = InterrogationEngine()

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
        state = await self._engine.start(prompt, sources=sources)

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

        self._engine.answer(state, question, answer)

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

        result = await self._engine.crystallize(state)
        spec = result.spec

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
