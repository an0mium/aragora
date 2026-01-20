"""
Knowledge Mound adapter for RLM.

Extracted from bridge.py for maintainability.
Provides KnowledgeMoundAdapter for integrating Knowledge Mound with RLM.
"""

import logging
from typing import Any, Callable, Optional

from .types import AbstractionLevel, AbstractionNode, RLMContext

logger = logging.getLogger(__name__)


class KnowledgeMoundAdapter:
    """
    Adapter for integrating Knowledge Mound with RLM.

    Provides hierarchical access to knowledge nodes through RLM REPL.
    """

    def __init__(self, mound: Any):
        """
        Initialize with Knowledge Mound instance.

        Args:
            mound: KnowledgeMound instance from aragora.knowledge.mound
        """
        self.mound = mound

    async def to_rlm_context(
        self,
        workspace_id: str,
        query: Optional[str] = None,
        max_nodes: int = 100,
    ) -> RLMContext:
        """
        Convert Knowledge Mound contents to RLM context.

        Args:
            workspace_id: Workspace to query
            query: Optional query to filter relevant nodes
            max_nodes: Maximum nodes to include

        Returns:
            RLMContext with hierarchical representation
        """
        # Query relevant nodes
        if query:
            nodes = await self.mound.query_semantic(
                text=query,
                limit=max_nodes,
                workspace_id=workspace_id,
            )
        else:
            nodes = await self.mound.get_recent_nodes(
                workspace_id=workspace_id,
                limit=max_nodes,
            )

        # Build content from nodes
        content_parts = []
        for node in nodes:
            content_parts.append(f"[{node.id}] {node.content}")

        full_content = "\n\n".join(content_parts)

        # Create basic context
        context = RLMContext(
            original_content=full_content,
            original_tokens=len(full_content) // 4,
            source_type="knowledge",
        )

        # Group nodes by type for hierarchical representation
        nodes_by_type: dict[str, list] = {}
        for node in nodes:
            node_type = getattr(node, "node_type", "unknown")
            nodes_by_type.setdefault(node_type, []).append(node)

        # Create abstraction nodes per type
        abstraction_nodes = []
        for node_type, type_nodes in nodes_by_type.items():
            summary_content = f"**{node_type.upper()}** ({len(type_nodes)} items):\n"
            summary_content += "\n".join(
                f"- {n.content[:100]}..." if len(n.content) > 100 else f"- {n.content}"
                for n in type_nodes[:10]
            )

            abstraction_nodes.append(AbstractionNode(
                id=f"type_{node_type}",
                level=AbstractionLevel.SUMMARY,
                content=summary_content,
                token_count=len(summary_content) // 4,
                child_ids=[n.id for n in type_nodes],
            ))

        context.levels[AbstractionLevel.SUMMARY] = abstraction_nodes
        for node in abstraction_nodes:
            context.nodes_by_id[node.id] = node

        return context

    def get_repl_helpers(self) -> dict[str, Callable]:
        """
        Get helper functions for REPL access to Knowledge Mound.

        Returns dict of functions that can be injected into REPL namespace.
        """
        async def search_mound(query: str, limit: int = 10) -> list[dict]:
            nodes = await self.mound.query_semantic(query=query, limit=limit)
            return [
                {
                    "id": n.id,
                    "type": getattr(n, "node_type", "unknown"),
                    "content": n.content[:200],
                    "confidence": getattr(n, "confidence", 1.0),
                }
                for n in nodes
            ]

        async def get_mound_node(node_id: str) -> Optional[dict]:
            node = await self.mound.get_node(node_id)
            if not node:
                return None
            return {
                "id": node.id,
                "type": getattr(node, "node_type", "unknown"),
                "content": node.content,
                "confidence": getattr(node, "confidence", 1.0),
                "relationships": getattr(node, "relationships", {}),
            }

        return {
            "search_mound": search_mound,
            "get_mound_node": get_mound_node,
        }


__all__ = ["KnowledgeMoundAdapter"]
