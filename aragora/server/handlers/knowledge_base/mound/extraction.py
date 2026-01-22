"""
Extraction operations mixin for Knowledge Mound handler.

Provides HTTP endpoints for extracting knowledge from debates:
- POST /api/knowledge/mound/extraction/debate - Extract from a debate
- POST /api/knowledge/mound/extraction/promote - Promote extracted claims
- GET /api/knowledge/mound/extraction/stats - Get extraction statistics

Phase A2 - Knowledge Extraction & Integration
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ...base import (
    HandlerResult,
    error_response,
    json_response,
    safe_error_message,
)
from ...utils.rate_limit import rate_limit

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

logger = logging.getLogger(__name__)


class ExtractionOperationsMixin:
    """Mixin providing knowledge extraction API endpoints."""

    ctx: Dict[str, Any]

    def _get_mound(self) -> Optional["KnowledgeMound"]:
        """Get the knowledge mound instance."""
        raise NotImplementedError("Subclass must implement _get_mound")

    @rate_limit(requests_per_minute=20)
    async def extract_from_debate(
        self,
        debate_id: str,
        messages: List[Dict[str, Any]],
        consensus_text: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> HandlerResult:
        """
        Extract knowledge from a debate.

        POST /api/knowledge/mound/extraction/debate
        {
            "debate_id": "...",
            "messages": [
                {"agent_id": "...", "content": "...", "round": 1},
                ...
            ],
            "consensus_text": "...",  // optional
            "topic": "..."  // optional
        }

        Args:
            debate_id: ID of the debate
            messages: List of debate messages
            consensus_text: Optional consensus conclusion
            topic: Optional debate topic

        Returns:
            ExtractionResult with extracted claims and relationships
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not debate_id:
            return error_response("debate_id is required", status=400)

        if not messages:
            return error_response("messages list is required", status=400)

        try:
            result = await mound.extract_from_debate(
                debate_id=debate_id,
                messages=messages,
                consensus_text=consensus_text,
                topic=topic,
            )

            return json_response(result.to_dict())
        except Exception as e:
            logger.error(f"Error extracting from debate: {e}")
            return error_response(safe_error_message(e), status=500)

    @rate_limit(requests_per_minute=10)
    async def promote_extracted_knowledge(
        self,
        workspace_id: str,
        claim_ids: Optional[List[str]] = None,
        min_confidence: float = 0.6,
    ) -> HandlerResult:
        """
        Promote extracted claims to Knowledge Mound.

        POST /api/knowledge/mound/extraction/promote
        {
            "workspace_id": "...",
            "claim_ids": ["...", "..."],  // optional, promotes all if omitted
            "min_confidence": 0.6
        }

        Args:
            workspace_id: Workspace to add claims to
            claim_ids: Optional specific claim IDs to promote
            min_confidence: Minimum confidence to promote

        Returns:
            Number of claims promoted
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not workspace_id:
            return error_response("workspace_id is required", status=400)

        try:
            # If specific claim IDs provided, we need to get them
            # For now, pass None to promote all eligible claims
            claims = None  # The mixin will use stored claims

            promoted_count = await mound.promote_extracted_knowledge(
                workspace_id=workspace_id,
                claims=claims,
                min_confidence=min_confidence,
            )

            return json_response(
                {
                    "success": True,
                    "workspace_id": workspace_id,
                    "min_confidence": min_confidence,
                    "promoted_count": promoted_count,
                }
            )
        except Exception as e:
            logger.error(f"Error promoting extracted knowledge: {e}")
            return error_response(safe_error_message(e), status=500)

    @rate_limit(requests_per_minute=60)
    async def get_extraction_stats(self) -> HandlerResult:
        """
        Get extraction statistics.

        GET /api/knowledge/mound/extraction/stats

        Returns:
            Extraction statistics
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        try:
            stats = mound.get_extraction_stats()
            return json_response(stats)
        except Exception as e:
            logger.error(f"Error getting extraction stats: {e}")
            return error_response(safe_error_message(e), status=500)
