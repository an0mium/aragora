"""Memory critiques mixin (MemoryCritiquesMixin).

Extracted from memory.py to reduce file size.
Contains critique store browsing operations.

Note: RBAC is handled in MemoryHandler.handle() which calls these mixin methods.
"""

from __future__ import annotations

import logging

from aragora.rbac.decorators import require_permission  # noqa: F401 - Required for RBAC consistency

# Permission constant - used by parent MemoryHandler
MEMORY_READ_PERMISSION = "memory:read"

from ..base import (
    HandlerResult,
    error_response,
    get_bounded_string_param,
    get_clamped_int_param,
    handle_errors,
    json_response,
    safe_error_message,
)
from ..utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)


class MemoryCritiquesMixin:
    """Mixin providing critique store browsing operations."""

    # These attributes are defined in the main class
    ctx: dict

    @rate_limit(requests_per_minute=60, limiter_name="memory_read")
    @handle_errors("get critiques")
    def _get_critiques(self, params: dict) -> HandlerResult:
        """Browse critique store entries.

        Query params:
            agent: Filter by agent name (optional)
            limit: Maximum results (default: 20, max: 100)
            offset: Skip first N results (default: 0)

        Note: debate_id filtering not currently supported by CritiqueStore.
        """
        from .memory import CRITIQUE_STORE_AVAILABLE, CritiqueStore
        from aragora.stores.canonical import get_critique_store

        if not CRITIQUE_STORE_AVAILABLE:
            return error_response("Critique store not available", 503)

        nomic_dir = self.ctx.get("nomic_dir")
        if not nomic_dir:
            return error_response("Critique store not configured", 503)

        agent = get_bounded_string_param(params, "agent", "", max_length=100)
        limit = get_clamped_int_param(params, "limit", 20, min_val=1, max_val=100)
        offset = get_clamped_int_param(params, "offset", 0, min_val=0, max_val=10000)

        try:
            store = get_critique_store(nomic_dir)
            if store is None and CritiqueStore is not None:
                # Fallback for tests or environments where canonical DB is not present.
                store = CritiqueStore(nomic_dir)
            if store is None:
                return error_response("Critique store not available", 503)

            # Get recent critiques - CritiqueStore only supports get_recent()
            # Fetch extra to account for filtering and offset
            fetch_limit = limit + offset + 100 if agent else limit + offset
            all_critiques = store.get_recent(limit=fetch_limit)

            # Filter by agent if specified
            if agent:
                all_critiques = [c for c in all_critiques if c.agent == agent]

            # Apply offset and limit
            critiques = all_critiques[offset : offset + limit]

            results = []
            for c in critiques:
                # Build content from issues/suggestions since Critique has no 'content' field
                content_parts = []
                if c.issues:
                    content_parts.extend(c.issues[:2])
                if c.suggestions:
                    content_parts.extend(c.suggestions[:2])
                content = "; ".join(content_parts)[:300] if content_parts else ""

                results.append(
                    {
                        "id": None,  # Critique dataclass has no id field
                        "debate_id": None,  # Not available from get_recent()
                        "agent": c.agent,
                        "target_agent": c.target_agent,
                        "critique_type": None,  # Not available
                        "content": content,
                        "severity": c.severity,
                        "accepted": None,  # Not available
                        "created_at": None,  # Not available from Critique dataclass
                    }
                )

            # Get total count for pagination
            total = len(all_critiques) if agent else len(store.get_recent(limit=10000))

            return json_response(
                {
                    "critiques": results,
                    "count": len(results),
                    "total": total,
                    "offset": offset,
                    "limit": limit,
                    "filters": {
                        "agent": agent or None,
                    },
                }
            )

        except (KeyError, ValueError, OSError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Failed to get critiques: %s", e)
            return error_response(safe_error_message(e, "get critiques"), 500)
