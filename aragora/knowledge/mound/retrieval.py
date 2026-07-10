"""
Reusable Knowledge Mound retrieval helpers for debate/context enrichment.

This module provides a thin interface over the Knowledge Mound query surface
so callers do not need to know when to use visibility-aware, semantic, or
generic query APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any


DEFAULT_CONTEXT_HEADER = "## KNOWLEDGE MOUND CONTEXT"
DEFAULT_CONTEXT_INTRO = "Relevant knowledge from organizational memory:"
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_LIMIT = 10
MAX_DEBATE_QUERY_CHARS = 1600


def _supports_method(obj: Any, method_name: str) -> bool:
    """Return True when an object explicitly provides a method."""
    if obj is None:
        return False
    try:
        instance_vars = vars(obj)
    except TypeError:
        instance_vars = {}
    if method_name in instance_vars:
        return callable(getattr(obj, method_name))
    return callable(getattr(type(obj), method_name, None))


def build_debate_knowledge_query(task: str, supplemental_text: str = "") -> str:
    """Build a bounded KM query from the task and recent debate developments."""
    task_text = " ".join((task or "").split())
    supplemental = " ".join((supplemental_text or "").split())
    if not supplemental:
        return task_text[:MAX_DEBATE_QUERY_CHARS].rstrip()
    prefix = "\n\nRecent debate developments:\n"
    if not task_text:
        return supplemental[:MAX_DEBATE_QUERY_CHARS].rstrip()

    query = f"{task_text}{prefix}{supplemental}"
    if len(query) <= MAX_DEBATE_QUERY_CHARS:
        return query

    overflow = len(query) - MAX_DEBATE_QUERY_CHARS
    if overflow > 0 and supplemental:
        supplemental = supplemental[:-overflow].rstrip() if overflow < len(supplemental) else ""
        query = f"{task_text}{prefix}{supplemental}" if supplemental else task_text
    if len(query) <= MAX_DEBATE_QUERY_CHARS:
        return query
    return query[:MAX_DEBATE_QUERY_CHARS].rstrip()


@dataclass(slots=True)
class RetrievedKnowledgeContext:
    """Normalized KM retrieval payload for prompt enrichment."""

    query: str
    items: list[Any]
    item_ids: list[str]
    formatted_context: str


class KnowledgeMoundRetriever:
    """Wrap KM query methods behind a single debate-friendly interface."""

    def __init__(
        self,
        knowledge_mound: Any,
        *,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        default_limit: int = DEFAULT_LIMIT,
    ) -> None:
        self.knowledge_mound = knowledge_mound
        self.min_confidence = min_confidence
        self.default_limit = default_limit

    def is_available(self) -> bool:
        """Return True when the configured KM exposes at least one query API."""
        return any(
            _supports_method(self.knowledge_mound, method_name)
            for method_name in ("query_with_visibility", "query_semantic", "query")
        )

    def _semantic_query_argument_name(self) -> str:
        """Resolve whether this KM expects `text=` or legacy `query=`."""
        method = getattr(self.knowledge_mound, "query_semantic", None)
        if not callable(method):
            return "text"
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            return "text"
        if "text" in parameters:
            return "text"
        if "query" in parameters:
            return "query"
        return "text"

    async def retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
        auth_context: Any | None = None,
        workspace_id: str | None = None,
    ) -> RetrievedKnowledgeContext | None:
        """Execute a KM retrieval and normalize the result for prompt injection."""
        query_text = (query or "").strip()
        if not query_text or not self.knowledge_mound or not self.is_available():
            return None

        items = await self._query_items(
            query_text,
            limit=limit or self.default_limit,
            auth_context=auth_context,
            workspace_id=workspace_id,
            allow_initialize_retry=True,
        )
        if not items:
            return None

        item_ids: list[str] = []
        for item in items:
            item_id = getattr(item, "id", None) or getattr(item, "item_id", None)
            if isinstance(item_id, str) and item_id:
                item_ids.append(item_id)
        return RetrievedKnowledgeContext(
            query=query_text,
            items=items,
            item_ids=item_ids,
            formatted_context=self.format_for_prompt(items),
        )

    async def _query_items(
        self,
        query: str,
        *,
        limit: int,
        auth_context: Any | None,
        workspace_id: str | None,
        allow_initialize_retry: bool,
    ) -> list[Any]:
        try:
            result = await self._dispatch_query(
                query,
                limit=limit,
                auth_context=auth_context,
                workspace_id=workspace_id,
            )
        except RuntimeError as exc:
            if (
                allow_initialize_retry
                and "not initialized" in str(exc).lower()
                and _supports_method(self.knowledge_mound, "initialize")
            ):
                await self.knowledge_mound.initialize()
                return await self._query_items(
                    query,
                    limit=limit,
                    auth_context=auth_context,
                    workspace_id=workspace_id,
                    allow_initialize_retry=False,
                )
            raise
        return self._normalize_items(result)

    async def _dispatch_query(
        self,
        query: str,
        *,
        limit: int,
        auth_context: Any | None,
        workspace_id: str | None,
    ) -> Any:
        effective_workspace_id = workspace_id
        if effective_workspace_id is None and auth_context is not None:
            auth_workspace_id = getattr(auth_context, "workspace_id", "") or ""
            effective_workspace_id = str(auth_workspace_id).strip() or None

        if auth_context and _supports_method(self.knowledge_mound, "query_with_visibility"):
            actor_id = getattr(auth_context, "user_id", "") or ""
            actor_workspace_id = getattr(auth_context, "workspace_id", "") or ""
            actor_org_id = getattr(auth_context, "org_id", None)
            if actor_id and actor_workspace_id:
                kwargs: dict[str, Any] = {
                    "actor_id": actor_id,
                    "actor_workspace_id": actor_workspace_id,
                    "actor_org_id": actor_org_id,
                    "limit": limit,
                }
                if workspace_id is not None:
                    kwargs["workspace_id"] = workspace_id
                return await self.knowledge_mound.query_with_visibility(
                    query,
                    **kwargs,
                )

        if _supports_method(self.knowledge_mound, "query_semantic"):
            semantic_query_arg = self._semantic_query_argument_name()
            kwargs = {
                semantic_query_arg: query,
                "limit": limit,
                "min_confidence": self.min_confidence,
            }
            if effective_workspace_id is not None:
                kwargs["workspace_id"] = effective_workspace_id
            return await self.knowledge_mound.query_semantic(**kwargs)

        kwargs = {
            "query": query,
            "sources": ("all",),
            "limit": limit,
        }
        if effective_workspace_id is not None:
            kwargs["workspace_id"] = effective_workspace_id
        return await self.knowledge_mound.query(**kwargs)

    @staticmethod
    def _normalize_items(result: Any) -> list[Any]:
        if isinstance(result, list):
            return result
        if hasattr(result, "items"):
            items = getattr(result, "items")
            if isinstance(items, list):
                return items
        return []

    @staticmethod
    def _confidence_to_float(value: Any) -> float:
        confidence_map = {
            "verified": 0.95,
            "high": 0.8,
            "medium": 0.6,
            "low": 0.3,
            "unverified": 0.2,
        }
        if isinstance(value, (int, float)):
            return float(value)
        if hasattr(value, "value"):
            value = value.value
        if isinstance(value, str):
            return confidence_map.get(value.lower(), 0.5)
        return 0.5

    def format_for_prompt(
        self,
        items: list[Any],
        *,
        header: str = DEFAULT_CONTEXT_HEADER,
        intro: str = DEFAULT_CONTEXT_INTRO,
        max_items: int | None = None,
        max_content_chars: int = 300,
    ) -> str:
        """Format retrieved knowledge into a prompt-safe evidence block."""
        if not items:
            return ""

        lines = [header, f"{intro}\n"]
        for item in items[: max_items or len(items)]:
            source = getattr(item, "source", "unknown")
            confidence = self._confidence_to_float(getattr(item, "confidence", 0.0))
            content = (getattr(item, "content", str(item)) or "")[:max_content_chars]
            lines.append(f"**[{source}]** (confidence: {confidence:.0%})")
            lines.append(content)
            lines.append("")
        return "\n".join(lines)


__all__ = [
    "KnowledgeMoundRetriever",
    "RetrievedKnowledgeContext",
    "build_debate_knowledge_query",
]
