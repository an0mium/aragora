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

# Distributed tracing support
try:
    from aragora.server.middleware.tracing import trace_context

    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False

    # Mock span for when tracing is not available
    class _MockSpan:
        def set_tag(self, key: str, value: Any) -> None:
            pass

        def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
            pass

        def set_error(self, error: Exception) -> None:
            pass

    from contextlib import contextmanager

    @contextmanager
    def trace_context(operation: str, **kwargs: Any):  # type: ignore[misc]
        yield _MockSpan()


from aragora.knowledge.mound.validation import (
    validate_graph_params,
    validate_id,
    validate_pagination,
    validate_query,
    validate_workspace_id,
)

if TYPE_CHECKING:
    from aragora.knowledge.mound.types import (
        GraphQueryResult,
        KnowledgeItem,
        KnowledgeLink,
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
        offset: int = 0,
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
        offset: int = 0,
        workspace_id: Optional[str] = None,
    ) -> "QueryResult":
        """
        Query across all configured knowledge sources.

        Args:
            query: Natural language query string
            sources: Which sources to query ("all" or specific sources)
            filters: Optional filters to apply
            limit: Maximum number of results
            offset: Number of results to skip (for pagination)
            workspace_id: Workspace to query (defaults to self.workspace_id)

        Returns:
            QueryResult with items from all queried sources
        """
        from aragora.knowledge.mound.types import KnowledgeSource, QueryResult

        self._ensure_initialized()

        with trace_context("km.query") as span:
            span.set_tag("query", query[:100])  # Truncate for tag size limits
            span.set_tag("sources", list(sources))
            span.set_tag("limit", limit)
            span.set_tag("offset", offset)

            # Validate inputs
            validate_query(query)
            ws_id = workspace_id or self.workspace_id
            if ws_id:
                validate_workspace_id(ws_id)
                span.set_tag("workspace_id", ws_id)
            limit, offset = validate_pagination(limit, offset)

            start_time = time.time()
            limit = min(limit, self.config.max_query_limit)

            # Check cache first (include offset in cache key)
            cache_key = f"{ws_id}:{query}:{limit}:{offset}:{sources}"
            if self._cache:
                cached = await self._cache.get_query(cache_key)
                if cached:
                    span.set_tag("cache_hit", True)
                    span.add_event("cache_hit")
                    return cached

            span.set_tag("cache_hit", False)
            span.add_event("cache_miss")

            # Query local mound
            items = await self._query_local(query, filters, limit, ws_id)
            span.add_event("local_query_complete", {"count": len(items)})

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
                    span.add_event("parallel_queries_start", {"count": len(tasks)})
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, list):
                            items.extend(result)
                        elif isinstance(result, Exception):
                            logger.warning(f"Query source failed: {result}")
                    span.add_event("parallel_queries_complete")

            # Sort by importance/relevance, apply offset, and limit
            items.sort(key=lambda x: x.importance or 0, reverse=True)
            if offset > 0:
                items = items[offset:]
            items = items[:limit]

            execution_time = (time.time() - start_time) * 1000
            span.set_tag("execution_time_ms", execution_time)
            span.set_tag("result_count", len(items))

            # Check and record SLO compliance
            try:
                from aragora.observability.metrics.slo import check_and_record_slo

                check_and_record_slo("km_query", execution_time)
            except ImportError:
                pass  # Metrics not available

            result = QueryResult(  # type: ignore[assignment]
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
                span.add_event("result_cached")

            return result  # type: ignore[return-value]

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
                key=lambda n: getattr(n, "updated_at", None)
                or getattr(n, "created_at", None)
                or "",
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
        """Traverse knowledge graph from a starting node.

        Args:
            start_id: Node ID to start traversal from
            relationship_types: Filter to specific relationship types
            depth: Maximum traversal depth (capped at 5)
            max_nodes: Maximum nodes to return (capped at 1000)

        Returns:
            GraphQueryResult with nodes and edges

        Raises:
            ValidationError: If parameters exceed limits
        """
        from aragora.knowledge.mound.types import GraphQueryResult

        self._ensure_initialized()

        # Validate inputs
        validate_id(start_id, field_name="start_id")
        depth, max_nodes = validate_graph_params(depth, max_nodes)

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
                    source = getattr(node, "source", None) or getattr(node, "source_type", None)
                    source_str = (
                        source.value
                        if hasattr(source, "value")
                        else str(source)
                        if source
                        else "unknown"
                    )
                    confidence = getattr(node, "confidence", 0.0)
                    if hasattr(confidence, "value"):
                        confidence = confidence.value
                    nodes.append(
                        {
                            "id": node.id,
                            "label": (node.content[:100] if node.content else "")[:100],
                            "type": source_str,
                            "confidence": confidence,
                        }
                    )
            for edge in result.edges:
                rel_type = getattr(edge, "relationship", None) or getattr(
                    edge, "relationship_type", None
                )
                rel_type_str = (
                    rel_type.value
                    if hasattr(rel_type, "value")
                    else str(rel_type)
                    if rel_type
                    else "related"
                )
                links.append(
                    {
                        "source": edge.source_id,
                        "target": edge.target_id,
                        "type": rel_type_str,
                        "strength": getattr(edge, "strength", 0.5)
                        or getattr(edge, "confidence", 0.5)
                        or 0.5,
                    }
                )
        else:
            # Get all nodes up to limit using local query
            all_items = await self._query_local("", None, limit, self.workspace_id)
            for item in all_items[:limit]:
                node_ids.add(item.id)
                source = getattr(item, "source", None) or getattr(item, "source_type", None)
                source_str = (
                    source.value
                    if hasattr(source, "value")
                    else str(source)
                    if source
                    else "unknown"
                )
                confidence = getattr(item, "confidence", 0.0)
                if hasattr(confidence, "value"):
                    confidence = confidence.value
                nodes.append(
                    {
                        "id": item.id,
                        "label": (item.content[:100] if item.content else "")[:100],
                        "type": source_str,
                        "confidence": confidence,
                    }
                )

            # Get relationships between collected nodes
            for node_id in list(node_ids)[:50]:
                rels = await self._get_relationships(node_id)
                for rel in rels:
                    target = rel.target_id if rel.source_id == node_id else rel.source_id
                    if target in node_ids:
                        rel_type = getattr(rel, "relationship", None) or getattr(
                            rel, "relationship_type", None
                        )
                        rel_type_str = (
                            rel_type.value
                            if hasattr(rel_type, "value")
                            else str(rel_type)
                            if rel_type
                            else "related"
                        )
                        links.append(
                            {
                                "source": rel.source_id,
                                "target": rel.target_id,
                                "type": rel_type_str,
                                "strength": getattr(rel, "strength", 0.5)
                                or getattr(rel, "confidence", 0.5)
                                or 0.5,
                            }
                        )

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
        d3_data = await self.export_graph_d3(start_node_id, depth, limit)  # type: ignore[attr-defined]

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
            label = (
                (node.get("label", "") or "")
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
            )
            lines.append(f'    <node id="{node["id"]}">')
            lines.append(f'      <data key="label">{label}</data>')
            lines.append(f'      <data key="type">{node.get("type", "unknown")}</data>')
            lines.append(f'      <data key="confidence">{node.get("confidence", 0.0)}</data>')
            lines.append("    </node>")

        # Add edges
        for i, link in enumerate(d3_data["links"]):
            lines.append(
                f'    <edge id="e{i}" source="{link["source"]}" target="{link["target"]}">'
            )
            lines.append(f'      <data key="rel_type">{link.get("type", "related")}</data>')
            lines.append(f'      <data key="strength">{link.get("strength", 0.5)}</data>')
            lines.append("    </edge>")

        lines.append("  </graph>")
        lines.append("</graphml>")

        return "\n".join(lines)

    # =========================================================================
    # Visibility-Aware Query Operations (Phase 2)
    # =========================================================================

    async def query_with_visibility(
        self: QueryProtocol,
        query: str,
        actor_id: str,
        actor_workspace_id: str,
        actor_org_id: Optional[str] = None,
        sources: Sequence["SourceFilter"] = ("all",),
        filters: Optional["QueryFilters"] = None,
        limit: int = 20,
        workspace_id: Optional[str] = None,
    ) -> "QueryResult":
        """
        Query across knowledge sources with visibility filtering.

        This method respects visibility levels:
        - PUBLIC/SYSTEM: Visible to everyone
        - WORKSPACE: Visible to workspace members
        - ORGANIZATION: Visible to organization members
        - PRIVATE: Only visible with explicit access grant

        Args:
            query: Natural language query string
            actor_id: ID of the user making the query
            actor_workspace_id: Workspace ID of the querying user
            actor_org_id: Organization ID of the querying user (optional)
            sources: Which sources to query ("all" or specific sources)
            filters: Optional filters to apply
            limit: Maximum number of results
            workspace_id: Workspace to query (defaults to actor's workspace)

        Returns:
            QueryResult with visibility-filtered items
        """
        from aragora.knowledge.mound.types import QueryResult

        self._ensure_initialized()

        start_time = time.time()
        ws_id = workspace_id or actor_workspace_id
        limit = min(limit, self.config.max_query_limit)

        # First, get regular query results
        result = await self.query(
            query, sources, filters, limit=limit * 2, workspace_id=ws_id
        )  # Get extra for filtering

        # Filter by visibility
        filtered_items = await self._filter_by_visibility(  # type: ignore[attr-defined]
            items=result.items,
            actor_id=actor_id,
            actor_workspace_id=actor_workspace_id,
            actor_org_id=actor_org_id,
        )

        execution_time = (time.time() - start_time) * 1000

        return QueryResult(
            items=filtered_items[:limit],
            total_count=len(filtered_items),
            query=query,
            filters=filters,
            execution_time_ms=execution_time,
            sources_queried=result.sources_queried,
        )

    async def _filter_by_visibility(
        self: QueryProtocol,
        items: List["KnowledgeItem"],
        actor_id: str,
        actor_workspace_id: str,
        actor_org_id: Optional[str] = None,
    ) -> List["KnowledgeItem"]:
        """
        Filter items based on visibility and access grants.

        Args:
            items: List of items to filter
            actor_id: ID of the user requesting access
            actor_workspace_id: Workspace ID of the requesting user
            actor_org_id: Organization ID of the requesting user

        Returns:
            Filtered list of visible items
        """
        from aragora.knowledge.mound.types import VisibilityLevel

        result = []

        for item in items:
            # Get visibility from item metadata or default to WORKSPACE
            vis_str = (item.metadata or {}).get("visibility", "workspace")
            try:
                vis = VisibilityLevel(vis_str)
            except ValueError:
                vis = VisibilityLevel.WORKSPACE

            item_workspace = (item.metadata or {}).get("workspace_id", "")
            item_org = (item.metadata or {}).get("org_id", "")

            # Check visibility rules
            if vis == VisibilityLevel.PUBLIC or vis == VisibilityLevel.SYSTEM:
                # Public and system items are always visible
                result.append(item)
            elif vis == VisibilityLevel.WORKSPACE:
                # Workspace items visible to workspace members
                if item_workspace == actor_workspace_id:
                    result.append(item)
            elif vis == VisibilityLevel.ORGANIZATION:
                # Organization items visible to org members
                if actor_org_id and item_org == actor_org_id:
                    result.append(item)
            elif vis == VisibilityLevel.PRIVATE:
                # Private items require explicit grant - check access_grants
                grants = (item.metadata or {}).get("access_grants", [])
                if self._has_access_grant(grants, actor_id, actor_workspace_id, actor_org_id):  # type: ignore[attr-defined]
                    result.append(item)

        return result

    def _has_access_grant(
        self: QueryProtocol,
        grants: List[dict],
        actor_id: str,
        actor_workspace_id: str,
        actor_org_id: Optional[str],
    ) -> bool:
        """
        Check if actor has an access grant in the grants list.

        Args:
            grants: List of grant dictionaries
            actor_id: User ID to check
            actor_workspace_id: Workspace ID to check
            actor_org_id: Organization ID to check

        Returns:
            True if actor has valid access grant
        """
        from datetime import datetime

        for grant in grants:
            # Check if grant is expired
            expires_at = grant.get("expires_at")
            if expires_at:
                try:
                    if datetime.fromisoformat(expires_at.replace("Z", "+00:00")) < datetime.now():
                        continue
                except (ValueError, TypeError):
                    pass

            grantee_type = grant.get("grantee_type", "")
            grantee_id = grant.get("grantee_id", "")

            if grantee_type == "user" and grantee_id == actor_id:
                return True
            elif grantee_type == "workspace" and grantee_id == actor_workspace_id:
                return True
            elif grantee_type == "organization" and actor_org_id and grantee_id == actor_org_id:
                return True

        return False

    async def get_shared_items(
        self: QueryProtocol,
        actor_id: str,
        actor_workspace_id: str,
        limit: int = 50,
    ) -> List["KnowledgeItem"]:
        """
        Get items that have been explicitly shared with the actor.

        This retrieves PRIVATE items where the actor has an access grant,
        from ANY workspace.

        Args:
            actor_id: ID of the user
            actor_workspace_id: Workspace ID of the user
            limit: Maximum number of items to return

        Returns:
            List of shared items
        """
        self._ensure_initialized()

        # If the postgres store has the method, use it
        if hasattr(self._meta_store, "get_grants_for_grantee_async"):
            grants = await self._meta_store.get_grants_for_grantee_async(actor_id)
            workspace_grants = await self._meta_store.get_grants_for_grantee_async(
                actor_workspace_id, grantee_type="workspace"
            )

            all_item_ids = set()
            for grant in grants + workspace_grants:
                if not grant.is_expired():
                    all_item_ids.add(grant.item_id)

            # Fetch items
            items = []
            for item_id in list(all_item_ids)[:limit]:
                item = await self.get(item_id)
                if item:
                    items.append(item)

            return items

        # Fallback: return empty list if store doesn't support grants
        return []
