"""
Query Operations Mixin for Knowledge Mound.

Provides query and search operations:
- query: Multi-source knowledge query
- get_recent_nodes: Recent node retrieval
- query_semantic: Vector similarity search
- query_graph: Graph traversal
- export_graph_d3: D3.js visualization export
- export_graph_graphml: GraphML export
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Sequence

if TYPE_CHECKING:
    from aragora.knowledge.mound.types import (
        GraphQueryResult,
        KnowledgeItem,
        KnowledgeLink,
        KnowledgeSource,
        MoundConfig,
        QueryFilters,
        QueryResult,
        RelationshipType,
        SourceFilter,
    )

logger = logging.getLogger(__name__)


class QueryProtocol(Protocol):
    """Protocol defining expected interface for Query mixin."""

    config: "MoundConfig"
    workspace_id: str
    _cache: Optional[Any]
    _vector_store: Optional[Any]
    _semantic_store: Optional[Any]
    _continuum: Optional[Any]
    _consensus: Optional[Any]
    _facts: Optional[Any]
    _evidence: Optional[Any]
    _critique: Optional[Any]
    _meta_store: Optional[Any]
    _initialized: bool

    def _ensure_initialized(self) -> None: ...
    async def get(self, node_id: str) -> Optional["KnowledgeItem"]: ...
    async def _query_local(
        self, query: str, filters: Optional["QueryFilters"], limit: int, workspace_id: str
    ) -> List["KnowledgeItem"]: ...
    async def _query_continuum(
        self, query: str, filters: Optional["QueryFilters"], limit: int
    ) -> List["KnowledgeItem"]: ...
    async def _query_consensus(
        self, query: str, filters: Optional["QueryFilters"], limit: int
    ) -> List["KnowledgeItem"]: ...
    async def _query_facts(
        self, query: str, filters: Optional["QueryFilters"], limit: int, workspace_id: str
    ) -> List["KnowledgeItem"]: ...
    async def _query_evidence(
        self, query: str, filters: Optional["QueryFilters"], limit: int, workspace_id: str
    ) -> List["KnowledgeItem"]: ...
    async def _query_critique(
        self, query: str, filters: Optional["QueryFilters"], limit: int
    ) -> List["KnowledgeItem"]: ...
    async def _get_relationships(
        self, node_id: str, types: Optional[List["RelationshipType"]] = None
    ) -> List["KnowledgeLink"]: ...
    def _node_to_item(self, node: Any) -> "KnowledgeItem": ...
    def _vector_result_to_item(self, result: Any) -> "KnowledgeItem": ...
    # Self-referential methods used by other methods in mixin
    async def query(
        self,
        query: str,
        sources: Sequence["SourceFilter"] = ("all",),
        filters: Optional["QueryFilters"] = None,
        limit: int = 20,
        workspace_id: Optional[str] = None,
    ) -> "QueryResult": ...
    async def query_graph(
        self,
        start_id: str,
        relationship_types: Optional[List["RelationshipType"]] = None,
        depth: int = 2,
        max_nodes: int = 50,
    ) -> "GraphQueryResult": ...


class QueryOperationsMixin:
    """Mixin providing query operations for KnowledgeMound."""

    async def query(
        self: QueryProtocol,
        query: str,
        sources: Sequence["SourceFilter"] = ("all",),
        filters: Optional["QueryFilters"] = None,
        limit: int = 20,
        workspace_id: Optional[str] = None,
    ) -> "QueryResult":
        """
        Query across all configured knowledge sources.

        Args:
            query: Natural language query string
            sources: Which sources to query ("all" or specific sources)
            filters: Optional filters to apply
            limit: Maximum number of results
            workspace_id: Workspace to query (defaults to self.workspace_id)

        Returns:
            QueryResult with items from all queried sources
        """
        from aragora.knowledge.mound.types import KnowledgeSource, QueryResult

        self._ensure_initialized()

        start_time = time.time()
        ws_id = workspace_id or self.workspace_id
        limit = min(limit, self.config.max_query_limit)

        # Check cache first
        cache_key = f"{ws_id}:{query}:{limit}:{sources}"
        if self._cache:
            cached = await self._cache.get_query(cache_key)
            if cached:
                return cached

        # Query local mound
        items = await self._query_local(query, filters, limit, ws_id)

        # Query connected memory systems in parallel
        if self.config.parallel_queries:
            tasks = []
            if "all" in sources or "continuum" in sources:
                tasks.append(self._query_continuum(query, filters, limit))
            if "all" in sources or "consensus" in sources:
                tasks.append(self._query_consensus(query, filters, limit))
            if "all" in sources or "fact" in sources:
                tasks.append(self._query_facts(query, filters, limit, ws_id))
            if "all" in sources or "evidence" in sources:
                tasks.append(self._query_evidence(query, filters, limit, ws_id))
            if "all" in sources or "critique" in sources:
                tasks.append(self._query_critique(query, filters, limit))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, list):
                        items.extend(result)
                    elif isinstance(result, Exception):
                        logger.warning(f"Query source failed: {result}")

        # Sort by importance/relevance and limit
        items.sort(key=lambda x: x.importance or 0, reverse=True)
        items = items[:limit]

        execution_time = (time.time() - start_time) * 1000

        result = QueryResult(
            items=items,
            total_count=len(items),
            query=query,
            filters=filters,
            execution_time_ms=execution_time,
            sources_queried=[KnowledgeSource(s) for s in sources if s != "all"],
        )

        # Cache result
        if self._cache:
            await self._cache.set_query(cache_key, result)

        return result

    async def get_recent_nodes(
        self: QueryProtocol,
        workspace_id: Optional[str] = None,
        limit: int = 50,
    ) -> List["KnowledgeItem"]:
        """
        Get most recently updated knowledge nodes.

        Args:
            workspace_id: Workspace to query (defaults to self.workspace_id)
            limit: Maximum number of nodes to return

        Returns:
            List of KnowledgeItems sorted by update time (newest first)
        """
        self._ensure_initialized()

        ws_id = workspace_id or self.workspace_id

        # Query the meta store for recent nodes
        if hasattr(self._meta_store, "get_recent_nodes_async"):
            return await self._meta_store.get_recent_nodes_async(ws_id, limit)
        else:
            # SQLite fallback - query nodes ordered by updated_at
            nodes = self._meta_store.query_nodes(
                workspace_id=ws_id,
                limit=limit,
            )
            # Sort by updated_at if available, else created_at
            sorted_nodes = sorted(
                nodes,
                key=lambda n: getattr(n, 'updated_at', None) or getattr(n, 'created_at', None) or '',
                reverse=True,
            )
            return [self._node_to_item(n) for n in sorted_nodes[:limit]]

    async def query_semantic(
        self: QueryProtocol,
        text: str,
        limit: int = 10,
        min_confidence: float = 0.0,
        workspace_id: Optional[str] = None,
    ) -> List["KnowledgeItem"]:
        """Semantic similarity search using vector embeddings."""
        self._ensure_initialized()

        ws_id = workspace_id or self.workspace_id

        # Try Weaviate first (production vector store)
        if self._vector_store:
            try:
                results = await self._vector_store.search(
                    query=text,
                    limit=limit,
                    filters={"workspace_id": ws_id},
                )
                return [self._vector_result_to_item(r) for r in results]
            except Exception as e:
                logger.warning(f"Weaviate search failed: {e}, falling back")

        # Try local semantic store (embeddings in SQLite)
        if self._semantic_store:
            try:
                results = await self._semantic_store.search_similar(
                    query=text,
                    tenant_id=ws_id,
                    limit=limit,
                    min_similarity=min_confidence,
                )
                # Convert semantic results to KnowledgeItems
                items = []
                for sr in results:
                    node = await self.get(sr.source_id)
                    if node:
                        items.append(node)
                return items
            except Exception as e:
                logger.warning(f"Semantic store search failed: {e}, falling back")

        # Fall back to keyword search
        result = await self.query(text, limit=limit, workspace_id=workspace_id)
        return result.items

    async def query_graph(
        self: QueryProtocol,
        start_id: str,
        relationship_types: Optional[List["RelationshipType"]] = None,
        depth: int = 2,
        max_nodes: int = 50,
    ) -> "GraphQueryResult":
        """Traverse knowledge graph from a starting node."""
        from aragora.knowledge.mound.types import GraphQueryResult, KnowledgeItem, KnowledgeLink

        self._ensure_initialized()

        nodes: Dict[str, KnowledgeItem] = {}
        edges: List[KnowledgeLink] = []
        visited: set = set()

        async def traverse(node_id: str, current_depth: int) -> None:
            if current_depth > depth or node_id in visited or len(nodes) >= max_nodes:
                return

            visited.add(node_id)
            node = await self.get(node_id)
            if node:
                nodes[node_id] = node

                if current_depth < depth:
                    relationships = await self._get_relationships(node_id, relationship_types)
                    for rel in relationships:
                        edges.append(rel)
                        target = rel.target_id if rel.source_id == node_id else rel.source_id
                        if target not in visited:
                            await traverse(target, current_depth + 1)

        await traverse(start_id, 0)

        return GraphQueryResult(
            nodes=list(nodes.values()),
            edges=edges,
            root_id=start_id,
            depth=depth,
            total_nodes=len(nodes),
            total_edges=len(edges),
        )

    async def export_graph_d3(
        self: QueryProtocol,
        start_node_id: Optional[str] = None,
        depth: int = 3,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Export graph in D3.js-compatible format for visualization.

        Args:
            start_node_id: Starting node for traversal (None for all nodes)
            depth: Maximum traversal depth
            limit: Maximum number of nodes

        Returns:
            Dict with 'nodes' and 'links' arrays for D3 force-directed graph
        """
        self._ensure_initialized()

        nodes: List[Dict[str, Any]] = []
        links: List[Dict[str, Any]] = []
        node_ids: set = set()

        if start_node_id:
            # Traverse from starting node
            result = await self.query_graph(start_node_id, depth=depth, max_nodes=limit)
            for node in result.nodes:
                if node.id not in node_ids:
                    node_ids.add(node.id)
                    source = getattr(node, 'source', None) or getattr(node, 'source_type', None)
                    source_str = source.value if hasattr(source, 'value') else str(source) if source else 'unknown'
                    confidence = getattr(node, 'confidence', 0.0)
                    if hasattr(confidence, 'value'):
                        confidence = confidence.value
                    nodes.append({
                        "id": node.id,
                        "label": (node.content[:100] if node.content else "")[:100],
                        "type": source_str,
                        "confidence": confidence,
                    })
            for edge in result.edges:
                rel_type = getattr(edge, 'relationship', None) or getattr(edge, 'relationship_type', None)
                rel_type_str = rel_type.value if hasattr(rel_type, 'value') else str(rel_type) if rel_type else 'related'
                links.append({
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": rel_type_str,
                    "strength": getattr(edge, 'strength', 0.5) or getattr(edge, 'confidence', 0.5) or 0.5,
                })
        else:
            # Get all nodes up to limit using local query
            all_items = await self._query_local("", None, limit, self.workspace_id)
            for item in all_items[:limit]:
                node_ids.add(item.id)
                source = getattr(item, 'source', None) or getattr(item, 'source_type', None)
                source_str = source.value if hasattr(source, 'value') else str(source) if source else 'unknown'
                confidence = getattr(item, 'confidence', 0.0)
                if hasattr(confidence, 'value'):
                    confidence = confidence.value
                nodes.append({
                    "id": item.id,
                    "label": (item.content[:100] if item.content else "")[:100],
                    "type": source_str,
                    "confidence": confidence,
                })

            # Get relationships between collected nodes
            for node_id in list(node_ids)[:50]:
                rels = await self._get_relationships(node_id)
                for rel in rels:
                    target = rel.target_id if rel.source_id == node_id else rel.source_id
                    if target in node_ids:
                        rel_type = getattr(rel, 'relationship', None) or getattr(rel, 'relationship_type', None)
                        rel_type_str = rel_type.value if hasattr(rel_type, 'value') else str(rel_type) if rel_type else 'related'
                        links.append({
                            "source": rel.source_id,
                            "target": rel.target_id,
                            "type": rel_type_str,
                            "strength": getattr(rel, 'strength', 0.5) or getattr(rel, 'confidence', 0.5) or 0.5,
                        })

        return {"nodes": nodes, "links": links}

    async def export_graph_graphml(
        self: QueryProtocol,
        start_node_id: Optional[str] = None,
        depth: int = 3,
        limit: int = 100,
    ) -> str:
        """
        Export graph in GraphML format for external tools.

        Args:
            start_node_id: Starting node for traversal (None for all nodes)
            depth: Maximum traversal depth
            limit: Maximum number of nodes

        Returns:
            GraphML XML string
        """
        d3_data = await self.export_graph_d3(start_node_id, depth, limit)

        # Build GraphML XML
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
            '  <key id="label" for="node" attr.name="label" attr.type="string"/>',
            '  <key id="type" for="node" attr.name="type" attr.type="string"/>',
            '  <key id="confidence" for="node" attr.name="confidence" attr.type="double"/>',
            '  <key id="rel_type" for="edge" attr.name="type" attr.type="string"/>',
            '  <key id="strength" for="edge" attr.name="strength" attr.type="double"/>',
            '  <graph id="knowledge_graph" edgedefault="directed">',
        ]

        # Add nodes
        for node in d3_data["nodes"]:
            label = (node.get("label", "") or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
            lines.append(f'    <node id="{node["id"]}">')
            lines.append(f'      <data key="label">{label}</data>')
            lines.append(f'      <data key="type">{node.get("type", "unknown")}</data>')
            lines.append(f'      <data key="confidence">{node.get("confidence", 0.0)}</data>')
            lines.append('    </node>')

        # Add edges
        for i, link in enumerate(d3_data["links"]):
            lines.append(f'    <edge id="e{i}" source="{link["source"]}" target="{link["target"]}">')
            lines.append(f'      <data key="rel_type">{link.get("type", "related")}</data>')
            lines.append(f'      <data key="strength">{link.get("strength", 0.5)}</data>')
            lines.append('    </edge>')

        lines.append('  </graph>')
        lines.append('</graphml>')

        return '\n'.join(lines)
