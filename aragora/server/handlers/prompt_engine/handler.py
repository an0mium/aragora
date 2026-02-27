"""HTTP handler for prompt-to-spec engine sessions."""

from __future__ import annotations

import logging
from typing import Any

from aragora.prompt_engine.engine import PromptToSpecEngine
from aragora.prompt_engine.types import ResearchSource, UserProfile

logger = logging.getLogger(__name__)

_engine = PromptToSpecEngine()


class PromptEngineHandler:
    """REST handler for prompt engine session CRUD."""

    PREFIX = "/api/v1/prompt-engine/sessions"

    def __init__(self, server_context: dict[str, Any]) -> None:
        self.server_context = server_context

    def can_handle(self, path: str) -> bool:
        return path.startswith(self.PREFIX)

    async def handle(
        self,
        path: str,
        body: dict[str, Any],
        http_handler: Any,
    ) -> dict[str, Any]:
        method = getattr(http_handler, "command", "GET")
        session_id = self._extract_session_id(path)

        try:
            if method == "POST" and not session_id:
                return await self._create_session(body)
            elif method == "GET" and session_id:
                return self._get_session(session_id)
            elif method == "DELETE" and session_id:
                return self._delete_session(session_id)
            else:
                return {"status": 405, "error": "Method not allowed"}
        except KeyError:
            return {"status": 404, "error": "Session not found"}
        except ValueError as e:
            return {"status": 400, "error": str(e)}

    def _extract_session_id(self, path: str) -> str | None:
        suffix = path[len(self.PREFIX) :]
        if suffix.startswith("/") and len(suffix) > 1:
            return suffix[1:].split("/")[0]
        return None

    async def _create_session(self, body: dict[str, Any]) -> dict[str, Any]:
        prompt = body.get("prompt", "")
        profile_str = body.get("profile", "founder")
        sources_str = body.get("research_sources")

        profile = UserProfile(profile_str)
        sources = None
        if sources_str:
            sources = [ResearchSource(s) for s in sources_str]

        state = await _engine.start_session(prompt, profile, sources)
        return {"status": 200, "data": state.to_dict()}

    def _get_session(self, session_id: str) -> dict[str, Any]:
        state = _engine.get_session(session_id)
        return {"status": 200, "data": state.to_dict()}

    def _delete_session(self, session_id: str) -> dict[str, Any]:
        _engine.delete_session(session_id)
        return {"status": 200, "data": {"deleted": True}}
