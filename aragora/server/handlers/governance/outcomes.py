"""
Outcome tracking HTTP handler.

Endpoints for recording, querying, and analyzing decision outcomes:
- POST /api/v1/decisions/{id}/outcome  - Record an outcome for a decision
- GET  /api/v1/decisions/{id}/outcomes - List outcomes for a decision
- GET  /api/v1/outcomes/search         - Search outcomes by topic/tags
- GET  /api/v1/outcomes/impact         - Impact analytics across outcomes
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

from ..base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)
from ..utils.decorators import require_permission
from ..utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# In-memory outcome storage (production would use GovernanceStore)
MAX_OUTCOMES = 5000
_outcome_store: OrderedDict[str, dict[str, Any]] = OrderedDict()

VALID_OUTCOME_TYPES = {"success", "failure", "partial", "unknown"}


def _evict_old_outcomes() -> None:
    """Evict oldest outcomes if over limit."""
    while len(_outcome_store) > MAX_OUTCOMES:
        _outcome_store.popitem(last=False)


class OutcomeHandler:
    """Handler for decision outcome tracking API endpoints."""

    ROUTES = [
        "/api/v1/decisions/*/outcome",
        "/api/v1/decisions/*/outcomes",
        "/api/v1/outcomes/search",
        "/api/v1/outcomes/impact",
        # Legacy unversioned
        "/api/decisions/*/outcome",
        "/api/decisions/*/outcomes",
        "/api/outcomes/search",
        "/api/outcomes/impact",
    ]

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    MAX_BODY_SIZE = 1_048_576  # 1 MB

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

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given request."""
        if "/outcomes/search" in path:
            return True
        if "/outcomes/impact" in path:
            return True
        if path.rstrip("/").endswith("/outcome"):
            return True
        if path.rstrip("/").endswith("/outcomes"):
            return True
        return False

    def handle(self, method: str, path: str, handler: Any) -> HandlerResult:
        """Route requests to appropriate handler methods."""
        if "/outcomes/search" in path:
            return self._handle_search_outcomes(handler)
        if "/outcomes/impact" in path:
            return self._handle_impact_analytics(handler)
        if path.rstrip("/").endswith("/outcome") and method == "POST":
            return self._handle_record_outcome(path, handler)
        if path.rstrip("/").endswith("/outcomes") and method == "GET":
            return self._handle_list_outcomes(path, handler)
        return error_response("Not found", 404)

    @handle_errors("outcome recording")
    def _handle_record_outcome(self, path: str, handler: Any) -> HandlerResult:
        """
        POST /api/v1/decisions/{id}/outcome

        Record an outcome for a decision.

        Body:
            debate_id: str (required)
            outcome_type: str - success/failure/partial/unknown (required)
            outcome_description: str (required)
            impact_score: float 0-1 (required)
            kpis_before: dict (optional)
            kpis_after: dict (optional)
            lessons_learned: str (optional)
            tags: list[str] (optional)
        """
        # Extract decision_id from path
        segments = path.strip("/").split("/")
        decision_id = None
        for i, seg in enumerate(segments):
            if seg == "decisions" and i + 1 < len(segments):
                decision_id = segments[i + 1]
                break

        if not decision_id:
            return error_response("Missing decision ID in path", 400)

        body = self._read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        # Validate required fields
        debate_id = body.get("debate_id")
        outcome_type = body.get("outcome_type")
        outcome_description = body.get("outcome_description")
        impact_score = body.get("impact_score")

        if not debate_id:
            return error_response("debate_id is required", 400)
        if not outcome_type or outcome_type not in VALID_OUTCOME_TYPES:
            return error_response(
                f"outcome_type must be one of: {', '.join(sorted(VALID_OUTCOME_TYPES))}", 400
            )
        if not outcome_description:
            return error_response("outcome_description is required", 400)
        if impact_score is None or not isinstance(impact_score, (int, float)):
            return error_response("impact_score must be a number between 0 and 1", 400)
        if not (0.0 <= float(impact_score) <= 1.0):
            return error_response("impact_score must be between 0 and 1", 400)

        outcome_id = f"out_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        outcome = {
            "outcome_id": outcome_id,
            "decision_id": decision_id,
            "debate_id": debate_id,
            "outcome_type": outcome_type,
            "outcome_description": outcome_description,
            "impact_score": float(impact_score),
            "measured_at": now.isoformat(),
            "kpis_before": body.get("kpis_before", {}),
            "kpis_after": body.get("kpis_after", {}),
            "lessons_learned": body.get("lessons_learned", ""),
            "tags": body.get("tags", []),
        }

        _outcome_store[outcome_id] = outcome
        _evict_old_outcomes()

        # Ingest into KM via outcome adapter
        try:
            from aragora.knowledge.mound.adapters.outcome_adapter import get_outcome_adapter

            adapter = get_outcome_adapter()
            adapter.ingest(outcome)
        except (ImportError, RuntimeError, ValueError, TypeError) as e:
            logger.warning("Failed to ingest outcome to KM: %s", e)

        logger.info(
            "Recorded outcome %s for decision %s (type=%s, impact=%.2f)",
            outcome_id,
            decision_id,
            outcome_type,
            float(impact_score),
        )

        return json_response(
            {
                "outcome_id": outcome_id,
                "decision_id": decision_id,
                "status": "recorded",
            },
            status=201,
        )

    @rate_limit(requests_per_minute=60)
    def _handle_list_outcomes(self, path: str, handler: Any) -> HandlerResult:
        """
        GET /api/v1/decisions/{id}/outcomes

        List outcomes for a specific decision.
        """
        # Extract decision_id from path
        segments = path.strip("/").split("/")
        decision_id = None
        for i, seg in enumerate(segments):
            if seg == "decisions" and i + 1 < len(segments):
                decision_id = segments[i + 1]
                break

        if not decision_id:
            return error_response("Missing decision ID in path", 400)

        outcomes = [
            o for o in _outcome_store.values() if o["decision_id"] == decision_id
        ]

        return json_response({
            "decision_id": decision_id,
            "outcomes": outcomes,
            "count": len(outcomes),
        })

    @rate_limit(requests_per_minute=60)
    def _handle_search_outcomes(self, handler: Any) -> HandlerResult:
        """
        GET /api/v1/outcomes/search?q=...&tags=...&type=...&limit=50

        Search outcomes by topic, tags, or outcome type.
        """
        query = ""
        tags_filter: list[str] = []
        type_filter = ""
        limit = 50

        # Parse query params from handler
        if hasattr(handler, "parsed_url") and hasattr(handler.parsed_url, "query"):
            from urllib.parse import parse_qs

            params = parse_qs(handler.parsed_url.query)
            query = params.get("q", [""])[0]
            tags_raw = params.get("tags", [""])[0]
            if tags_raw:
                tags_filter = [t.strip() for t in tags_raw.split(",") if t.strip()]
            type_filter = params.get("type", [""])[0]
            try:
                limit = min(int(params.get("limit", ["50"])[0]), 200)
            except (ValueError, TypeError):
                limit = 50

        results = []
        for outcome in _outcome_store.values():
            # Filter by type
            if type_filter and outcome.get("outcome_type") != type_filter:
                continue
            # Filter by tags
            if tags_filter:
                outcome_tags = set(outcome.get("tags", []))
                if not outcome_tags.intersection(tags_filter):
                    continue
            # Filter by query (simple substring match)
            if query:
                searchable = (
                    outcome.get("outcome_description", "")
                    + " "
                    + outcome.get("lessons_learned", "")
                ).lower()
                if query.lower() not in searchable:
                    continue
            results.append(outcome)
            if len(results) >= limit:
                break

        return json_response({
            "outcomes": results,
            "count": len(results),
            "query": query,
        })

    @rate_limit(requests_per_minute=30)
    def _handle_impact_analytics(self, handler: Any) -> HandlerResult:
        """
        GET /api/v1/outcomes/impact

        Impact analytics across all outcomes.
        Returns aggregate statistics grouped by outcome type.
        """
        if not _outcome_store:
            return json_response({
                "total_outcomes": 0,
                "by_type": {},
                "avg_impact_score": 0.0,
                "top_lessons": [],
            })

        by_type: dict[str, dict[str, Any]] = {}
        all_scores: list[float] = []
        lessons: list[tuple[float, str]] = []

        for outcome in _outcome_store.values():
            otype = outcome.get("outcome_type", "unknown")
            score = outcome.get("impact_score", 0.0)
            all_scores.append(score)

            if otype not in by_type:
                by_type[otype] = {"count": 0, "total_impact": 0.0, "avg_impact": 0.0}
            by_type[otype]["count"] += 1
            by_type[otype]["total_impact"] += score

            lesson = outcome.get("lessons_learned", "")
            if lesson:
                lessons.append((score, lesson))

        # Compute averages
        for otype, stats in by_type.items():
            if stats["count"] > 0:
                stats["avg_impact"] = round(stats["total_impact"] / stats["count"], 3)

        avg_impact = round(sum(all_scores) / len(all_scores), 3) if all_scores else 0.0

        # Top lessons by impact score
        lessons.sort(key=lambda x: x[0], reverse=True)
        top_lessons = [
            {"impact_score": score, "lesson": lesson}
            for score, lesson in lessons[:10]
        ]

        return json_response({
            "total_outcomes": len(_outcome_store),
            "by_type": by_type,
            "avg_impact_score": avg_impact,
            "top_lessons": top_lessons,
        })


__all__ = ["OutcomeHandler"]
