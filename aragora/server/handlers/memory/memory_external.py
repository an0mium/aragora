"""Memory external adapters mixin (MemoryExternalMixin).

Extracted from memory.py to reduce file size.
Contains external memory adapter operations (supermemory, claude-mem).

Note: RBAC is handled in MemoryHandler.handle() which calls these mixin methods.
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.rbac.decorators import require_permission  # noqa: F401 - Required for RBAC consistency
from aragora.utils.async_utils import run_async

logger = logging.getLogger(__name__)

# Permission constant - used by parent MemoryHandler
MEMORY_READ_PERMISSION = "memory:read"


class MemoryExternalMixin:
    """Mixin providing external memory adapter operations."""

    def _get_supermemory_adapter(self) -> Any | None:
        """Get or create supermemory adapter client."""
        if hasattr(self, "_supermemory_client"):
            return getattr(self, "_supermemory_client")
        try:
            from aragora.connectors.supermemory import SupermemoryConfig, get_client
        except ImportError:
            return None

        config = SupermemoryConfig.from_env()
        if not config:
            return None

        try:
            client = get_client(config)
        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as exc:  # pragma: no cover - external dependency
            logger.debug(f"Supermemory client init failed: {exc}")
            return None

        setattr(self, "_supermemory_client", client)
        setattr(self, "_supermemory_config", config)
        return client

    def _search_supermemory(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search supermemory for relevant memories."""
        client = self._get_supermemory_adapter()
        if not client:
            return []

        config = getattr(self, "_supermemory_config", None)
        container_tag = getattr(config, "container_tag", None)

        try:
            response = run_async(
                client.search(query=query, limit=limit, container_tag=container_tag)
            )
        except Exception as exc:  # pragma: no cover - external dependency
            logger.debug(f"Supermemory search failed: {exc}")
            return []

        results: list[dict[str, Any]] = []
        for idx, item in enumerate(getattr(response, "results", []) or []):
            content = getattr(item, "content", "") or ""
            preview = content[:220].rstrip()
            if len(content) > 220:
                preview = f"{preview}..."
            results.append(
                {
                    "id": getattr(item, "memory_id", None) or f"super_{idx}",
                    "source": "supermemory",
                    "preview": preview,
                    "score": round(float(getattr(item, "similarity", 0.0)), 4),
                    "token_estimate": self._estimate_tokens(content),
                    "metadata": getattr(item, "metadata", {}) or {},
                    "container_tag": getattr(item, "container_tag", None),
                }
            )
        return results

    def _search_claude_mem(
        self, query: str, limit: int = 10, project: str | None = None
    ) -> list[dict[str, Any]]:
        """Search claude-mem for relevant memories."""
        try:
            from aragora.connectors import ClaudeMemConnector, ClaudeMemConfig
        except ImportError:
            return []

        connector = ClaudeMemConnector(ClaudeMemConfig.from_env())
        try:
            evidence = run_async(connector.search(query, limit=limit, project=project))
        except Exception as exc:  # pragma: no cover - external dependency
            logger.debug(f"Claude-mem search failed: {exc}")
            return []

        results: list[dict[str, Any]] = []
        for item in evidence:
            content = getattr(item, "content", "") or ""
            preview = content[:220].rstrip()
            if len(content) > 220:
                preview = f"{preview}..."
            results.append(
                {
                    "id": getattr(item, "id", None),
                    "source": "claude-mem",
                    "preview": preview,
                    "token_estimate": self._estimate_tokens(content),
                    "metadata": getattr(item, "metadata", {}) or {},
                    "created_at": getattr(item, "created_at", None),
                }
            )
        return results

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count for text."""
        import math

        if not text:
            return 0
        return max(1, int(math.ceil(len(text) / 4)))
