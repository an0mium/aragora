"""
Search operations handler mixin.

Extracted from handler.py for modularity. Provides cross-debate search
functionality with efficient SQL queries.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol

from aragora.server.validation.security import (
    validate_search_query_redos_safe,
    MAX_SEARCH_QUERY_LENGTH,
)

from ..base import (
    HandlerResult,
    error_response,
    json_response,
    require_storage,
    ttl_cache,
)
from ..utils.rate_limit import rate_limit
from .response_formatting import (
    CACHE_TTL_SEARCH,
    normalize_debate_response,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class _DebatesHandlerProtocol(Protocol):
    """Protocol defining the interface expected by SearchOperationsMixin.

    This protocol enables proper type checking for mixin classes that
    expect to be mixed into a class providing these methods/attributes.
    """

    ctx: Dict[str, Any]

    def get_storage(self) -> Optional[Any]:
        """Get debate storage instance."""
        ...


class SearchOperationsMixin:
    """Mixin providing search operations for DebatesHandler."""

    @rate_limit(rpm=30, limiter_name="debates_search")
    @require_storage
    @ttl_cache(ttl_seconds=CACHE_TTL_SEARCH, key_prefix="debates_search", skip_first=True)
    def _search_debates(
        self: _DebatesHandlerProtocol,
        query: str,
        limit: int,
        offset: int,
        org_id: Optional[str] = None,
    ) -> HandlerResult:
        """Search debates by query string, optionally filtered by organization.

        Uses efficient SQL LIKE queries instead of loading all debates into memory.
        This is optimized for large debate databases.

        Args:
            query: Search query string
            limit: Maximum results to return
            offset: Offset for pagination
            org_id: If provided, only search within this organization's debates

        Returns:
            HandlerResult with matching debates and pagination metadata
        """
        from aragora.exceptions import (
            DatabaseError,
            StorageError,
        )

        # Validate search query for ReDoS safety
        if query:
            validation_result = validate_search_query_redos_safe(
                query, max_length=MAX_SEARCH_QUERY_LENGTH
            )
            if not validation_result.is_valid:
                logger.warning("Search query validation failed: %s", validation_result.error)
                return error_response(validation_result.error or "Invalid search query", 400)

        storage = self.get_storage()
        try:
            # Use efficient SQL search if query provided
            if query:
                matching, total = storage.search(
                    query=query,
                    limit=limit,
                    offset=offset,
                    org_id=org_id,
                )
            else:
                # No query - just list recent debates
                matching = storage.list_recent(limit=limit, org_id=org_id)
                total = len(matching)  # Approximate for no-query case

            # Convert to dicts and normalize for SDK compatibility
            results = []
            for d in matching:
                if hasattr(d, "__dict__"):
                    results.append(normalize_debate_response(d.__dict__))
                elif isinstance(d, dict):
                    results.append(normalize_debate_response(d))
                else:
                    results.append({"data": str(d)})

            return json_response(
                {
                    "results": results,
                    "query": query,
                    "total": total,
                    "offset": offset,
                    "limit": limit,
                    "has_more": offset + len(results) < total,
                }
            )
        except (StorageError, DatabaseError) as e:
            logger.error(
                "Search failed for query '%s': %s: %s", query, type(e).__name__, e, exc_info=True
            )
            return error_response("Database error during search", 500)
        except ValueError as e:
            logger.warning("Invalid search query '%s': %s", query, e)
            return error_response(f"Invalid search query: {e}", 400)


__all__ = ["SearchOperationsMixin"]
