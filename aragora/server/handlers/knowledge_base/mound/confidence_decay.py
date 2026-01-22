"""
Confidence decay operations mixin for Knowledge Mound handler.

Provides HTTP endpoints for confidence management:
- POST /api/knowledge/mound/confidence/decay - Apply confidence decay
- POST /api/knowledge/mound/confidence/event - Record confidence event
- GET /api/knowledge/mound/confidence/history - Get adjustment history
- GET /api/knowledge/mound/confidence/stats - Get decay statistics

Phase A2 - Knowledge Quality Assurance
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

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


class ConfidenceDecayOperationsMixin:
    """Mixin providing confidence decay API endpoints."""

    ctx: Dict[str, Any]

    def _get_mound(self) -> Optional["KnowledgeMound"]:
        """Get the knowledge mound instance."""
        raise NotImplementedError("Subclass must implement _get_mound")

    @rate_limit(requests_per_minute=10)
    async def apply_confidence_decay_endpoint(
        self,
        workspace_id: str,
        force: bool = False,
    ) -> HandlerResult:
        """
        Apply confidence decay to workspace items.

        POST /api/knowledge/mound/confidence/decay
        {
            "workspace_id": "...",
            "force": false  // Force even if recently run
        }

        Args:
            workspace_id: Workspace to process
            force: Force decay even if recently run

        Returns:
            DecayReport with results
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not workspace_id:
            return error_response("workspace_id is required", status=400)

        try:
            report = await mound.apply_confidence_decay(  # type: ignore[call-arg]
                workspace_id=workspace_id,
                force=force,
            )

            return json_response(report.to_dict())  # type: ignore[attr-defined]
        except Exception as e:
            logger.error(f"Error applying confidence decay: {e}")
            return error_response(safe_error_message(e), status=500)

    @rate_limit(requests_per_minute=100)
    async def record_confidence_event(
        self,
        item_id: str,
        event: str,
        reason: str = "",
    ) -> HandlerResult:
        """
        Record a confidence-affecting event.

        POST /api/knowledge/mound/confidence/event
        {
            "item_id": "...",
            "event": "accessed",  // created, accessed, cited, validated, invalidated, contradicted, updated
            "reason": "..."  // optional
        }

        Args:
            item_id: Item affected
            event: Event type
            reason: Optional reason description

        Returns:
            ConfidenceAdjustment if confidence changed
        """
        from aragora.knowledge.mound.ops.confidence_decay import ConfidenceEvent

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not item_id:
            return error_response("item_id is required", status=400)

        if not event:
            return error_response("event is required", status=400)

        try:
            event_enum = ConfidenceEvent(event)
        except ValueError:
            valid_events = [e.value for e in ConfidenceEvent]
            return error_response(f"Invalid event. Valid events: {valid_events}", status=400)

        try:
            adjustment = await mound.record_confidence_event(
                item_id=item_id,
                event=event_enum,
                reason=reason,
            )

            if adjustment is None:
                return json_response(
                    {
                        "success": True,
                        "adjusted": False,
                        "message": "No confidence change required",
                    }
                )

            return json_response(
                {
                    "success": True,
                    "adjusted": True,
                    "adjustment": adjustment.to_dict(),
                }
            )
        except Exception as e:
            logger.error(f"Error recording confidence event: {e}")
            return error_response(safe_error_message(e), status=500)

    @rate_limit(requests_per_minute=30)
    async def get_confidence_history(
        self,
        item_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> HandlerResult:
        """
        Get confidence adjustment history.

        GET /api/knowledge/mound/confidence/history?item_id=...&event_type=...&limit=100

        Args:
            item_id: Filter by item ID
            event_type: Filter by event type
            limit: Maximum results

        Returns:
            List of confidence adjustments
        """
        from aragora.knowledge.mound.ops.confidence_decay import ConfidenceEvent

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        # Convert event_type string to enum if provided
        event_enum = None
        if event_type:
            try:
                event_enum = ConfidenceEvent(event_type)
            except ValueError:
                valid_events = [e.value for e in ConfidenceEvent]
                return error_response(
                    f"Invalid event_type. Valid types: {valid_events}", status=400
                )

        try:
            history = await mound.get_confidence_history(
                item_id=item_id,
                event_type=event_enum,
                limit=limit,
            )

            return json_response(
                {
                    "filters": {
                        "item_id": item_id,
                        "event_type": event_type,
                    },
                    "count": len(history),
                    "adjustments": [a.to_dict() for a in history],
                }
            )
        except Exception as e:
            logger.error(f"Error getting confidence history: {e}")
            return error_response(safe_error_message(e), status=500)

    @rate_limit(requests_per_minute=60)
    async def get_decay_stats(self) -> HandlerResult:
        """
        Get confidence decay statistics.

        GET /api/knowledge/mound/confidence/stats

        Returns:
            Decay statistics
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        try:
            stats = mound.get_decay_stats()
            return json_response(stats)
        except Exception as e:
            logger.error(f"Error getting decay stats: {e}")
            return error_response(safe_error_message(e), status=500)
