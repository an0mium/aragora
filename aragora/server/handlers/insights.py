"""
Insights-related endpoint handlers.

Endpoints:
- GET /api/insights/recent - Get recent insights from InsightStore
- POST /api/insights/extract-detailed - Extract detailed insights from content
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_int_param,
    handle_errors,
    safe_error_message,
)
from .utils.rate_limit import RateLimiter, get_client_ip

# Rate limiter for insights endpoints (60 requests per minute)
_insights_limiter = RateLimiter(requests_per_minute=60)

# Maximum content size for insight extraction (1MB)
MAX_CONTENT_SIZE = 1024 * 1024


def _run_async(coro):
    """Run async coroutine from sync context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


logger = logging.getLogger(__name__)


class InsightsHandler(BaseHandler):
    """Handler for insights-related endpoints."""

    # Route patterns this handler manages
    ROUTES = [
        "/api/insights/recent",
        "/api/insights/extract-detailed",
        "/api/flips/recent",
    ]

    # Endpoints that require authentication
    AUTH_REQUIRED_ENDPOINTS = [
        "/api/insights/extract-detailed",  # Computationally expensive
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path.startswith("/api/insights/") or path == "/api/flips/recent"

    def handle_get(self, path: str, query: dict, handler, ctx: dict) -> Optional[HandlerResult]:
        """Handle GET requests for insights endpoints."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _insights_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for insights endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        if path == "/api/insights/recent":
            return self._get_recent_insights(query, ctx)

        if path == "/api/flips/recent":
            return self._get_recent_flips(query, ctx)

        return None

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Handle POST requests for insights endpoints."""
        # Rate limit check (shared with GET)
        client_ip = get_client_ip(handler)
        if not _insights_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for insights POST endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        if path == "/api/insights/extract-detailed":
            # Read JSON body from request
            data = self.read_json_body(handler)
            if data is None:
                return error_response("Invalid or missing JSON body", 400)
            return self._extract_detailed_insights(data, self.ctx)

        return None

    @handle_errors("recent insights retrieval")
    def _get_recent_insights(self, query: dict, ctx: dict) -> HandlerResult:
        """Get recent insights from InsightStore.

        Query params:
            limit: Maximum number of insights to return (default: 20, max: 100)

        Returns:
            List of recent insights with id, type, title, description,
            confidence, agents_involved, and evidence preview.
        """
        insight_store = ctx.get("insight_store")
        if not insight_store:
            return json_response({"error": "Insights not configured", "insights": []})

        limit = max(1, min(get_int_param(query, "limit", 20), 100))

        insights = _run_async(insight_store.get_recent_insights(limit=limit))
        return json_response(
            {
                "insights": [
                    {
                        "id": i.id,
                        "type": i.type.value,
                        "title": i.title,
                        "description": i.description,
                        "confidence": i.confidence,
                        "agents_involved": i.agents_involved,
                        "evidence": i.evidence[:3] if i.evidence else [],
                    }
                    for i in insights
                ],
                "count": len(insights),
            }
        )

    @handle_errors("recent flips retrieval")
    def _get_recent_flips(self, query: dict, ctx: dict) -> HandlerResult:
        """Get recent position flips/reversals.

        Query params:
            limit: Maximum number of flips to return (default: 20, max: 100)

        Returns:
            List of recent position reversals with agent, previous/new positions,
            confidence change, and detection timestamp.
        """
        insight_store = ctx.get("insight_store")
        if not insight_store:
            return json_response(
                {
                    "flips": [],
                    "count": 0,
                    "message": "Position flip tracking not configured",
                }
            )

        limit = max(1, min(get_int_param(query, "limit", 20), 100))

        # Get insights and filter for position reversals
        flips = []
        try:
            insights = _run_async(insight_store.get_recent_insights(limit=limit * 2))
            for i in insights:
                if i.type.value == "position_reversal":
                    flips.append(
                        {
                            "id": i.id,
                            "agent": i.agents_involved[0] if i.agents_involved else "unknown",
                            "previous_position": i.description[:200] if i.description else "",
                            "new_position": i.title,
                            "confidence": i.confidence,
                            "detected_at": str(getattr(i, "created_at", None)),
                        }
                    )
                if len(flips) >= limit:
                    break
        except Exception as e:
            logger.warning(f"Error fetching position flips: {e}")

        return json_response(
            {
                "flips": flips,
                "count": len(flips),
            }
        )

    @handle_errors("insight extraction")
    def _extract_detailed_insights(self, data: dict, ctx: dict) -> HandlerResult:
        """Extract detailed insights from debate content.

        POST body:
            content: The debate content to analyze (required)
            debate_id: Optional debate ID for context
            extract_claims: Whether to extract claims (default: True)
            extract_evidence: Whether to extract evidence chains (default: True)
            extract_patterns: Whether to extract argumentation patterns (default: True)

        Returns detailed analysis of the debate content.
        """
        content = data.get("content", "").strip()
        if not content:
            return error_response("Missing required field: content", 400)

        if len(content) > MAX_CONTENT_SIZE:
            return error_response(
                f"Content too large. Maximum size is {MAX_CONTENT_SIZE // 1024}KB",
                413,  # Payload Too Large
            )

        debate_id = data.get("debate_id", "")
        extract_claims = data.get("extract_claims", True)
        extract_evidence = data.get("extract_evidence", True)
        extract_patterns = data.get("extract_patterns", True)

        result = {
            "debate_id": debate_id,
            "content_length": len(content),
        }

        # Extract claims if requested
        if extract_claims:
            claims = self._extract_claims_from_content(content)
            result["claims"] = claims

        # Extract evidence chains if requested
        if extract_evidence:
            evidence = self._extract_evidence_from_content(content)
            result["evidence_chains"] = evidence

        # Extract patterns if requested
        if extract_patterns:
            patterns = self._extract_patterns_from_content(content)
            result["patterns"] = patterns

        return json_response(result)

    def _extract_claims_from_content(self, content: str) -> list:
        """Extract claims from content using simple heuristics."""
        claims = []
        sentences = re.split(r"[.!?]+", content)

        # Claim indicators
        claim_patterns = [
            r"\b(therefore|thus|hence|consequently|as a result)\b",
            r"\b(I believe|we argue|it is clear|evidence shows)\b",
            r"\b(should|must|need to|ought to)\b",
            r"\b(is better|is worse|is more|is less)\b",
        ]

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            for pattern in claim_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    claims.append(
                        {
                            "text": sentence[:500],
                            "position": i,
                            "type": "argument" if "should" in sentence.lower() else "assertion",
                        }
                    )
                    break

        return claims[:20]  # Limit to 20 claims

    def _extract_evidence_from_content(self, content: str) -> list:
        """Extract evidence chains from content."""
        evidence = []

        # Evidence indicators
        evidence_patterns = [
            (r"according to ([^,.]+)", "citation"),
            (r"research shows ([^.]+)", "research"),
            (r"data indicates ([^.]+)", "data"),
            (r"for example,? ([^.]+)", "example"),
            (r"studies have shown ([^.]+)", "study"),
        ]

        for pattern, etype in evidence_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                evidence.append(
                    {
                        "text": match.group(0)[:300],
                        "type": etype,
                        "source": match.group(1)[:100] if match.groups() else None,
                    }
                )

        return evidence[:15]  # Limit to 15 evidence items

    def _extract_patterns_from_content(self, content: str) -> list[dict[str, str | int]]:
        """Extract argumentation patterns from content."""
        patterns: list[dict[str, str | int]] = []

        content_lower = content.lower()

        # Pattern detection
        if "on one hand" in content_lower and "on the other hand" in content_lower:
            patterns.append({"type": "balanced_comparison", "strength": "strong"})

        if "while" in content_lower and "however" in content_lower:
            patterns.append({"type": "concession_rebuttal", "strength": "medium"})

        if content_lower.count("first") > 0 and content_lower.count("second") > 0:
            patterns.append({"type": "enumerated_argument", "strength": "medium"})

        if "if" in content_lower and "then" in content_lower:
            patterns.append({"type": "conditional_reasoning", "strength": "medium"})

        if "because" in content_lower:
            count = content_lower.count("because")
            patterns.append(
                {
                    "type": "causal_reasoning",
                    "strength": "strong" if count > 2 else "medium",
                    "instances": count,
                }
            )

        return patterns
