"""
Playbook HTTP handler.

Endpoints for discovering and running decision playbooks:
- GET  /api/v1/playbooks          - List available playbooks
- GET  /api/v1/playbooks/{id}     - Get playbook details
- POST /api/v1/playbooks/{id}/run - Run a playbook
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from .base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)
from .utils.decorators import require_permission
from .utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)


class PlaybookHandler:
    """Handler for playbook API endpoints."""

    ROUTES = [
        "/api/v1/playbooks",
        "/api/v1/playbooks/*",
        "/api/v1/playbooks/*/run",
        # Legacy unversioned
        "/api/playbooks",
        "/api/playbooks/*",
        "/api/playbooks/*/run",
    ]

    MAX_BODY_SIZE = 1_048_576

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    def _read_json_body(self, handler: Any) -> dict[str, Any] | None:
        """Read and parse JSON body from request handler."""
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length <= 0:
                return {}
            if content_length > self.MAX_BODY_SIZE:
                return None
            body = handler.rfile.read(content_length)
            return json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    def can_handle(self, method: str, path: str) -> bool:
        """Check if this handler can handle the given request."""
        if path.rstrip("/").endswith("/run"):
            return method == "POST"
        if "/playbooks" in path:
            return method == "GET"
        return False

    def handle(self, method: str, path: str, handler: Any) -> HandlerResult:
        """Route requests to appropriate handler methods."""
        path_clean = path.rstrip("/")
        if path_clean.endswith("/run") and method == "POST":
            return self._handle_run_playbook(path, handler)
        if path_clean == "/api/v1/playbooks" or path_clean == "/api/playbooks":
            return self._handle_list_playbooks(handler)
        if "/playbooks/" in path and not path_clean.endswith("/run"):
            return self._handle_get_playbook(path, handler)
        return error_response("Not found", 404)

    @rate_limit(requests_per_minute=60)
    def _handle_list_playbooks(self, handler: Any) -> HandlerResult:
        """
        GET /api/v1/playbooks?category=...&tags=...

        List available playbooks.
        """
        from aragora.playbooks.registry import get_playbook_registry

        registry = get_playbook_registry()

        category = None
        tags = None
        if hasattr(handler, "parsed_url") and hasattr(handler.parsed_url, "query"):
            from urllib.parse import parse_qs

            params = parse_qs(handler.parsed_url.query)
            category = params.get("category", [None])[0]
            tags_raw = params.get("tags", [""])[0]
            if tags_raw:
                tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

        playbooks = registry.list(category=category, tags=tags)

        return json_response({
            "playbooks": [p.to_dict() for p in playbooks],
            "count": len(playbooks),
        })

    @rate_limit(requests_per_minute=60)
    def _handle_get_playbook(self, path: str, handler: Any) -> HandlerResult:
        """
        GET /api/v1/playbooks/{id}

        Get playbook details.
        """
        from aragora.playbooks.registry import get_playbook_registry

        playbook_id = self._extract_playbook_id(path)
        if not playbook_id:
            return error_response("Missing playbook ID", 400)

        registry = get_playbook_registry()
        playbook = registry.get(playbook_id)

        if not playbook:
            return error_response(f"Playbook not found: {playbook_id}", 404)

        return json_response(playbook.to_dict())

    @handle_errors("playbook execution")
    def _handle_run_playbook(self, path: str, handler: Any) -> HandlerResult:
        """
        POST /api/v1/playbooks/{id}/run

        Run a playbook. Accepts input parameters in the body.

        Body:
            input: str - The question/topic for the playbook
            context: dict - Additional context variables
        """
        from aragora.playbooks.registry import get_playbook_registry

        playbook_id = self._extract_playbook_id(path, strip_run=True)
        if not playbook_id:
            return error_response("Missing playbook ID", 400)

        registry = get_playbook_registry()
        playbook = registry.get(playbook_id)

        if not playbook:
            return error_response(f"Playbook not found: {playbook_id}", 404)

        body = self._read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        input_text = body.get("input", "")
        if not input_text:
            return error_response("input is required", 400)

        context = body.get("context", {})
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        # Build the execution plan
        execution = {
            "run_id": run_id,
            "playbook_id": playbook.id,
            "playbook_name": playbook.name,
            "input": input_text,
            "context": context,
            "status": "queued",
            "created_at": now.isoformat(),
            "steps": [s.to_dict() for s in playbook.steps],
            "config": {
                "template_name": playbook.template_name,
                "vertical_profile": playbook.vertical_profile,
                "min_agents": playbook.min_agents,
                "max_agents": playbook.max_agents,
                "max_rounds": playbook.max_rounds,
                "consensus_threshold": playbook.consensus_threshold,
            },
        }

        logger.info(
            "Playbook run queued: %s (playbook=%s, input=%s)",
            run_id,
            playbook.id,
            input_text[:100],
        )

        return json_response(execution, status=202)

    def _extract_playbook_id(self, path: str, strip_run: bool = False) -> str | None:
        """Extract playbook ID from path."""
        segments = path.strip("/").split("/")
        for i, seg in enumerate(segments):
            if seg == "playbooks" and i + 1 < len(segments):
                pid = segments[i + 1]
                if pid == "run":
                    return None
                return pid
        return None


__all__ = ["PlaybookHandler"]
