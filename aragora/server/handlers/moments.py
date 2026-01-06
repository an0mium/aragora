"""
Moments endpoint handlers.

Endpoints:
- GET /api/moments/summary - Global moments overview
- GET /api/moments/timeline - Chronological moments (limit, offset)
- GET /api/moments/by-type/{type} - Filter moments by type
- GET /api/moments/trending - Most significant recent moments
"""

import logging
import re
from typing import Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_int_param,
    validate_path_segment,
    SAFE_ID_PATTERN,
)

logger = logging.getLogger(__name__)

# Valid moment types
VALID_MOMENT_TYPES = {
    "upset_victory",
    "position_reversal",
    "calibration_vindication",
    "alliance_shift",
    "consensus_breakthrough",
    "streak_achievement",
    "domain_mastery",
}

# Lazy imports for optional dependencies
MOMENT_DETECTOR_AVAILABLE = False
MomentDetector = None
SignificantMoment = None

try:
    from aragora.agents.grounded import MomentDetector as _MD, SignificantMoment as _SM
    MomentDetector = _MD
    SignificantMoment = _SM
    MOMENT_DETECTOR_AVAILABLE = True
except ImportError:
    pass

from aragora.server.error_utils import safe_error_message as _safe_error_message


class MomentsHandler(BaseHandler):
    """Handler for moments endpoints."""

    ROUTES = [
        "/api/moments/summary",
        "/api/moments/timeline",
        "/api/moments/trending",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        # Handle dynamic route: /api/moments/by-type/{type}
        if path.startswith("/api/moments/by-type/"):
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route moments requests to appropriate methods."""
        if path == "/api/moments/summary":
            return self._get_summary()

        if path == "/api/moments/timeline":
            limit = get_int_param(query_params, 'limit', 50)
            offset = get_int_param(query_params, 'offset', 0)
            return self._get_timeline(min(limit, 200), offset)

        if path == "/api/moments/trending":
            limit = get_int_param(query_params, 'limit', 10)
            return self._get_trending(min(limit, 50))

        # Handle /api/moments/by-type/{type}
        if path.startswith("/api/moments/by-type/"):
            parts = path.split("/")
            if len(parts) >= 5:
                moment_type = parts[4]
                # Validate moment type
                is_valid, err = validate_path_segment(moment_type, "moment_type", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
                if moment_type not in VALID_MOMENT_TYPES:
                    return error_response(f"Invalid moment type: {moment_type}. Valid types: {', '.join(sorted(VALID_MOMENT_TYPES))}", 400)
                limit = get_int_param(query_params, 'limit', 50)
                return self._get_by_type(moment_type, min(limit, 200))

        return None

    def _get_moment_detector(self):
        """Get moment detector from context or return None."""
        return self.ctx.get("moment_detector")

    def _get_all_moments(self) -> list:
        """Get all moments from all agents."""
        detector = self._get_moment_detector()
        if not detector:
            return []

        all_moments = []
        # Access the internal cache to get all moments
        if hasattr(detector, '_moment_cache'):
            for agent_name, moments in detector._moment_cache.items():
                all_moments.extend(moments)

        return all_moments

    def _moment_to_dict(self, moment) -> dict:
        """Convert a SignificantMoment to a dict for JSON response."""
        return {
            "id": moment.id,
            "type": moment.moment_type,
            "agent": moment.agent_name,
            "description": moment.description,
            "significance": moment.significance_score,
            "debate_id": moment.debate_id,
            "other_agents": moment.other_agents or [],
            "metadata": moment.metadata or {},
            "created_at": moment.created_at if hasattr(moment, 'created_at') else None,
        }

    def _get_summary(self) -> HandlerResult:
        """Get global moments summary."""
        if not MOMENT_DETECTOR_AVAILABLE:
            return error_response("Moment detection not available", 503)

        detector = self._get_moment_detector()
        if not detector:
            return error_response("Moment detector not configured", 503)

        try:
            all_moments = self._get_all_moments()

            # Count by type
            by_type = {}
            for moment in all_moments:
                mt = moment.moment_type
                by_type[mt] = by_type.get(mt, 0) + 1

            # Count by agent
            by_agent = {}
            for moment in all_moments:
                agent = moment.agent_name
                by_agent[agent] = by_agent.get(agent, 0) + 1

            # Most significant moment
            most_significant = None
            if all_moments:
                sorted_moments = sorted(all_moments, key=lambda m: m.significance_score, reverse=True)
                most_significant = self._moment_to_dict(sorted_moments[0])

            # Recent moments (last 5)
            recent = sorted(all_moments, key=lambda m: getattr(m, 'created_at', '') or '', reverse=True)[:5]

            return json_response({
                "total_moments": len(all_moments),
                "by_type": by_type,
                "by_agent": by_agent,
                "most_significant": most_significant,
                "recent": [self._moment_to_dict(m) for m in recent],
            })
        except Exception as e:
            return error_response(_safe_error_message(e, "moments summary"), 500)

    def _get_timeline(self, limit: int, offset: int) -> HandlerResult:
        """Get chronological moments timeline."""
        if not MOMENT_DETECTOR_AVAILABLE:
            return error_response("Moment detection not available", 503)

        detector = self._get_moment_detector()
        if not detector:
            return error_response("Moment detector not configured", 503)

        try:
            all_moments = self._get_all_moments()

            # Sort by created_at descending (most recent first)
            sorted_moments = sorted(
                all_moments,
                key=lambda m: getattr(m, 'created_at', '') or '',
                reverse=True
            )

            # Apply pagination
            paginated = sorted_moments[offset:offset + limit]

            return json_response({
                "moments": [self._moment_to_dict(m) for m in paginated],
                "total": len(all_moments),
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < len(all_moments),
            })
        except Exception as e:
            return error_response(_safe_error_message(e, "moments timeline"), 500)

    def _get_trending(self, limit: int) -> HandlerResult:
        """Get most significant recent moments."""
        if not MOMENT_DETECTOR_AVAILABLE:
            return error_response("Moment detection not available", 503)

        detector = self._get_moment_detector()
        if not detector:
            return error_response("Moment detector not configured", 503)

        try:
            all_moments = self._get_all_moments()

            # Sort by significance descending
            sorted_moments = sorted(
                all_moments,
                key=lambda m: m.significance_score,
                reverse=True
            )[:limit]

            return json_response({
                "trending": [self._moment_to_dict(m) for m in sorted_moments],
                "count": len(sorted_moments),
            })
        except Exception as e:
            return error_response(_safe_error_message(e, "moments trending"), 500)

    def _get_by_type(self, moment_type: str, limit: int) -> HandlerResult:
        """Get moments filtered by type."""
        if not MOMENT_DETECTOR_AVAILABLE:
            return error_response("Moment detection not available", 503)

        detector = self._get_moment_detector()
        if not detector:
            return error_response("Moment detector not configured", 503)

        try:
            all_moments = self._get_all_moments()

            # Filter by type
            filtered = [m for m in all_moments if m.moment_type == moment_type]

            # Sort by significance descending
            sorted_moments = sorted(
                filtered,
                key=lambda m: m.significance_score,
                reverse=True
            )[:limit]

            return json_response({
                "type": moment_type,
                "moments": [self._moment_to_dict(m) for m in sorted_moments],
                "total": len(filtered),
                "limit": limit,
            })
        except Exception as e:
            return error_response(_safe_error_message(e, f"moments by type {moment_type}"), 500)
