"""
KM adapter #35: Persist RLM codebase understanding summaries.

When NomicContextBuilder or CodebaseRLMContext generates a codebase
summary via TRUE RLM, this adapter stores the result in KM so it
persists across sessions and is queryable in debates.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from aragora.knowledge.mound.adapters._base import KnowledgeMoundAdapter

logger = logging.getLogger(__name__)


class RLMContextAdapter(KnowledgeMoundAdapter):
    """Persist RLM codebase understanding summaries in Knowledge Mound.

    Stores codebase summaries and per-module analyses so that RLM context
    survives across sessions and can be injected into debate prompts for
    code-related decisions.
    """

    adapter_name = "rlm_context"

    def __init__(
        self,
        store_fn: Any | None = None,
        search_fn: Any | None = None,
        event_callback: Any | None = None,
        enable_tracing: bool = True,
        enable_resilience: bool = True,
    ):
        """Initialise with optional store/search callables.

        Args:
            store_fn: Async callable ``(item_dict) -> str`` returning item ID.
            search_fn: Async callable ``(query, limit) -> list[dict]``.
            event_callback: WebSocket event callback.
            enable_tracing: Enable OpenTelemetry tracing.
            enable_resilience: Enable circuit breaker/bulkhead.
        """
        super().__init__(
            event_callback=event_callback,
            enable_tracing=enable_tracing,
            enable_resilience=enable_resilience,
        )
        self._store_fn = store_fn
        self._search_fn = search_fn

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    async def store_codebase_summary(
        self,
        summary: str,
        root_path: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a codebase understanding summary as a KM item.

        Args:
            summary: The full codebase summary text.
            root_path: Path of the codebase root.
            metadata: Additional metadata to store.

        Returns:
            Item ID, or empty string on failure.
        """
        if not self._store_fn:
            return ""

        item = {
            "content": summary,
            "source": "rlm_codebase",
            "source_id": f"codebase:{root_path}",
            "confidence": 0.8,
            "metadata": {
                "type": "codebase_summary",
                "root_path": root_path,
                "created_at": time.time(),
                **(metadata or {}),
            },
        }

        with self._timed_operation("store", source="codebase_summary"):
            try:
                item_id: str = await self._store_fn(item)
                self._emit_event(
                    "km_rlm_context_store",
                    {"item_id": item_id, "type": "codebase_summary"},
                )
                return item_id
            except (RuntimeError, ValueError, OSError, TypeError) as exc:
                logger.warning("RLM context store failed: %s", exc)
                return ""

    async def store_module_analysis(
        self,
        module_path: str,
        analysis: str,
    ) -> str:
        """Store per-module analysis for granular retrieval.

        Args:
            module_path: Python module path (e.g. "aragora.debate.orchestrator").
            analysis: The analysis text for this module.

        Returns:
            Item ID, or empty string on failure.
        """
        if not self._store_fn:
            return ""

        item = {
            "content": f"[{module_path}] {analysis}",
            "source": "rlm_module",
            "source_id": f"module:{module_path}",
            "confidence": 0.75,
            "metadata": {
                "type": "module_analysis",
                "module_path": module_path,
                "created_at": time.time(),
            },
        }

        with self._timed_operation("store", source="module_analysis"):
            try:
                item_id = await self._store_fn(item)
                return item_id
            except (RuntimeError, ValueError, OSError, TypeError) as exc:
                logger.warning("RLM module analysis store failed: %s", exc)
                return ""

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def get_codebase_context(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant codebase understanding for a debate query.

        Args:
            query: The debate task or question.
            limit: Maximum results to return.

        Returns:
            List of KM-compatible item dicts.
        """
        if not self._search_fn:
            return []

        with self._timed_operation("search", query=query):
            try:
                results = await self._search_fn(query, limit)
                return results or []
            except (RuntimeError, ValueError, OSError, TypeError) as exc:
                logger.warning("RLM context search failed: %s", exc)
                return []


__all__ = ["RLMContextAdapter"]
