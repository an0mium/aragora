"""
GraphRAG: Hybrid Vector + Graph Retrieval.

Based on: Microsoft GraphRAG and related research on hybrid retrieval.

Combines vector-based semantic retrieval with graph traversal for:
- Multi-hop reasoning across connected knowledge
- Community-based summarization
- Relationship-aware relevance scoring

Key insight: Pure vector retrieval misses semantic relationships.
GraphRAG uses knowledge graph structure to find contextually related
information that may not have high vector similarity.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol
from enum import Enum
import logging
import time
from collections import defaultdict


logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    """Types of relationships in the knowledge graph."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    ELABORATES = "elaborates"
    SUPERSEDES = "supersedes"
    RELATED = "related"
    CAUSED_BY = "caused_by"
    RESULTS_IN = "results_in"
    PART_OF = "part_of"


@dataclass
class GraphNode:
    """A node in the knowledge graph."""

    id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source_type: str = "unknown"


@dataclass
class GraphEdge:
    """An edge connecting two nodes."""

    source_id: str
    target_id: str
    relationship: RelationshipType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval."""

    node_id: str
    content: str
    score: float  # Combined relevance score
    vector_score: float  # Raw vector similarity
    graph_score: float  # Graph-based relevance
    hop_distance: int  # Distance from seed nodes
    path: list[str]  # Path from nearest seed
    relationships: list[str]  # Relationship types in path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CommunityResult:
    """A detected community of related knowledge."""

    community_id: str
    node_ids: list[str]
    summary: str
    central_theme: str
    coherence_score: float
    relationship_density: float


@dataclass
class GraphRAGResult:
    """Full result from GraphRAG retrieval."""

    query: str
    results: list[RetrievalResult]
    communities: list[CommunityResult]
    total_nodes_explored: int
    total_edges_traversed: int
    retrieval_time_ms: float
    vector_retrieval_time_ms: float
    graph_expansion_time_ms: float


@dataclass
class GraphRAGConfig:
    """Configuration for GraphRAG retrieval."""

    # Vector retrieval settings
    vector_top_k: int = 10  # Initial vector retrieval count
    vector_threshold: float = 0.5  # Minimum similarity threshold

    # Graph expansion settings
    max_hops: int = 2  # Maximum graph traversal depth
    max_neighbors_per_hop: int = 5  # Neighbors to explore per node
    graph_weight: float = 0.3  # Weight for graph-based score

    # Relationship weights
    relationship_weights: dict[str, float] = field(
        default_factory=lambda: {
            "supports": 1.0,
            "elaborates": 0.9,
            "related": 0.7,
            "part_of": 0.8,
            "caused_by": 0.85,
            "results_in": 0.85,
            "supersedes": 0.6,
            "contradicts": 0.4,  # Still relevant but lower weight
        }
    )

    # Community detection
    enable_community_detection: bool = True
    min_community_size: int = 3
    community_coherence_threshold: float = 0.6

    # Final output
    final_top_k: int = 20  # Final results to return


class VectorStoreProtocol(Protocol):
    """Protocol for vector store interaction."""

    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
        threshold: float,
    ) -> list[tuple[str, float]]:
        """Search by embedding, return (node_id, score) pairs."""
        ...

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding for text."""
        ...


class GraphStoreProtocol(Protocol):
    """Protocol for graph store interaction."""

    async def get_neighbors(
        self,
        node_id: str,
        relationship_types: list[RelationshipType] | None = None,
    ) -> list[tuple[str, RelationshipType, float]]:
        """Get neighboring nodes with relationship type and weight."""
        ...

    async def get_node(self, node_id: str) -> GraphNode | None:
        """Get a node by ID."""
        ...


class GraphRAGRetriever:
    """
    Hybrid retriever combining vector search with graph expansion.

    Algorithm:
    1. Vector retrieval: Find top-k similar nodes by embedding
    2. Graph expansion: Traverse relationships from seed nodes
    3. Score combination: Blend vector similarity with graph relevance
    4. Community detection: Identify clusters of related knowledge
    5. Re-ranking: Return final sorted results

    Example:
        retriever = GraphRAGRetriever(vector_store, graph_store)

        result = await retriever.retrieve(
            query="What caused the 2008 financial crisis?",
            context={"domain": "economics"},
        )

        for r in result.results[:5]:
            print(f"{r.content[:100]}... (score={r.score:.3f}, hops={r.hop_distance})")
    """

    def __init__(
        self,
        vector_store: VectorStoreProtocol,
        graph_store: GraphStoreProtocol,
        config: GraphRAGConfig | None = None,
    ):
        """Initialize the retriever.

        Args:
            vector_store: Vector store for embedding-based retrieval
            graph_store: Graph store for relationship traversal
            config: Configuration options
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.config = config or GraphRAGConfig()
        self._retrieval_history: list[GraphRAGResult] = []

    async def retrieve(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        override_config: dict[str, Any] | None = None,
    ) -> GraphRAGResult:
        """
        Perform hybrid vector + graph retrieval.

        Args:
            query: The query text
            context: Optional context for filtering/boosting
            override_config: Override specific config values

        Returns:
            GraphRAGResult with ranked results and communities
        """
        start_time = time.time()

        # Apply config overrides
        config = self.config
        if override_config:
            config = GraphRAGConfig(
                **{
                    **vars(self.config),
                    **override_config,
                }
            )

        # Step 1: Vector retrieval
        vector_start = time.time()
        query_embedding = await self.vector_store.get_embedding(query)
        seed_results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=config.vector_top_k,
            threshold=config.vector_threshold,
        )
        vector_time_ms = (time.time() - vector_start) * 1000

        # Convert to dict for easy lookup
        vector_scores: dict[str, float] = {node_id: score for node_id, score in seed_results}
        seed_node_ids = list(vector_scores.keys())

        logger.debug(
            "vector_retrieval query_len=%d seeds=%d time_ms=%.1f",
            len(query),
            len(seed_node_ids),
            vector_time_ms,
        )

        # Step 2: Graph expansion
        graph_start = time.time()
        expanded_nodes, edge_count = await self._expand_graph(
            seed_node_ids=seed_node_ids,
            max_hops=config.max_hops,
            max_neighbors=config.max_neighbors_per_hop,
        )
        graph_time_ms = (time.time() - graph_start) * 1000

        logger.debug(
            "graph_expansion seeds=%d expanded=%d edges=%d time_ms=%.1f",
            len(seed_node_ids),
            len(expanded_nodes),
            edge_count,
            graph_time_ms,
        )

        # Step 3: Score combination
        results = await self._score_and_rank(
            query_embedding=query_embedding,
            expanded_nodes=expanded_nodes,
            vector_scores=vector_scores,
            config=config,
        )

        # Step 4: Community detection (optional)
        communities: list[CommunityResult] = []
        if config.enable_community_detection and len(results) >= config.min_community_size:
            communities = await self._detect_communities(
                nodes=[r.node_id for r in results],
                config=config,
            )

        # Step 5: Final ranking and limiting
        results = sorted(results, key=lambda r: r.score, reverse=True)
        results = results[: config.final_top_k]

        total_time_ms = (time.time() - start_time) * 1000

        graph_rag_result = GraphRAGResult(
            query=query,
            results=results,
            communities=communities,
            total_nodes_explored=len(expanded_nodes),
            total_edges_traversed=edge_count,
            retrieval_time_ms=total_time_ms,
            vector_retrieval_time_ms=vector_time_ms,
            graph_expansion_time_ms=graph_time_ms,
        )

        self._retrieval_history.append(graph_rag_result)

        logger.info(
            "graphrag_retrieval query_len=%d results=%d communities=%d "
            "nodes_explored=%d time_ms=%.1f",
            len(query),
            len(results),
            len(communities),
            len(expanded_nodes),
            total_time_ms,
        )

        return graph_rag_result

    async def _expand_graph(
        self,
        seed_node_ids: list[str],
        max_hops: int,
        max_neighbors: int,
    ) -> tuple[dict[str, dict[str, Any]], int]:
        """
        Expand from seed nodes through the graph.

        Returns:
            Tuple of (expanded_nodes dict, total edges traversed)
        """
        expanded: dict[str, dict[str, Any]] = {}
        edge_count = 0
        frontier: list[tuple[str, int, list[str], list[str]]] = [(node_id, 0, [node_id], []) for node_id in seed_node_ids]
        visited: set[str] = set(seed_node_ids)

        # Initialize seeds
        for node_id in seed_node_ids:
            expanded[node_id] = {
                "hop_distance": 0,
                "path": [node_id],
                "relationships": [],
                "graph_score": 1.0,  # Seeds get full score
            }

        while frontier:
            current_id, current_hop, path, relationships = frontier.pop(0)

            if current_hop >= max_hops:
                continue

            # Get neighbors
            try:
                neighbors = await self.graph_store.get_neighbors(current_id)
            except Exception as e:
                logger.warning("neighbor_fetch_error node=%s error=%s", current_id, e)
                continue

            # Sort by weight and limit
            neighbors = sorted(neighbors, key=lambda x: x[2], reverse=True)
            neighbors = neighbors[:max_neighbors]

            for neighbor_id, rel_type, weight in neighbors:
                edge_count += 1

                if neighbor_id in visited:
                    continue

                visited.add(neighbor_id)
                new_path = path + [neighbor_id]
                new_relationships = relationships + [rel_type.value]

                # Calculate graph score based on path
                rel_weight = self.config.relationship_weights.get(rel_type.value, 0.5)
                parent_score = expanded[current_id]["graph_score"]
                new_score = parent_score * rel_weight * (0.8 ** (current_hop + 1))

                expanded[neighbor_id] = {
                    "hop_distance": current_hop + 1,
                    "path": new_path,
                    "relationships": new_relationships,
                    "graph_score": new_score,
                }

                frontier.append((neighbor_id, current_hop + 1, new_path, new_relationships))

        return expanded, edge_count

    async def _score_and_rank(
        self,
        query_embedding: list[float],
        expanded_nodes: dict[str, dict[str, Any]],
        vector_scores: dict[str, float],
        config: GraphRAGConfig,
    ) -> list[RetrievalResult]:
        """
        Combine vector and graph scores for final ranking.
        """
        results: list[RetrievalResult] = []

        for node_id, node_info in expanded_nodes.items():
            # Get or compute vector score
            if node_id in vector_scores:
                v_score = vector_scores[node_id]
            else:
                # For non-seed nodes, we don't have vector score
                # Could compute it here, but it's expensive
                v_score = 0.0

            g_score = node_info["graph_score"]

            # Combine scores
            # For seed nodes: weight vector score higher
            # For expanded nodes: graph score matters more
            if node_info["hop_distance"] == 0:
                combined = 0.8 * v_score + 0.2 * g_score
            else:
                combined = (1.0 - config.graph_weight) * v_score + config.graph_weight * g_score

            # Get node content
            try:
                node = await self.graph_store.get_node(node_id)
                content = node.content if node else ""
                metadata = node.metadata if node else {}
            except Exception:
                logger.warning("Failed to retrieve node %s from graph store", node_id, exc_info=True)
                content = ""
                metadata = {}

            results.append(
                RetrievalResult(
                    node_id=node_id,
                    content=content,
                    score=combined,
                    vector_score=v_score,
                    graph_score=g_score,
                    hop_distance=node_info["hop_distance"],
                    path=node_info["path"],
                    relationships=node_info["relationships"],
                    metadata=metadata,
                )
            )

        return results

    async def _detect_communities(
        self,
        nodes: list[str],
        config: GraphRAGConfig,
    ) -> list[CommunityResult]:
        """
        Detect communities of related knowledge using simple clustering.

        Uses connected component analysis with relationship-based weighting.
        """
        # Build adjacency for these nodes
        adjacency: dict[str, set[str]] = defaultdict(set)

        for node_id in nodes:
            try:
                neighbors = await self.graph_store.get_neighbors(node_id)
                for neighbor_id, rel_type, weight in neighbors:
                    if neighbor_id in nodes:
                        # Only consider strong relationships
                        rel_weight = config.relationship_weights.get(rel_type.value, 0.5)
                        if rel_weight >= config.community_coherence_threshold:
                            adjacency[node_id].add(neighbor_id)
                            adjacency[neighbor_id].add(node_id)
            except Exception:
                logger.debug("Failed to process relationships for node %s", node_id, exc_info=True)
                continue

        # Find connected components
        visited: set[str] = set()
        communities: list[CommunityResult] = []
        community_count = 0

        for node_id in nodes:
            if node_id in visited:
                continue

            # BFS to find component
            component: list[str] = []
            queue = [node_id]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)

                for neighbor in adjacency.get(current, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)

            if len(component) >= config.min_community_size:
                # Calculate community metrics
                total_edges = sum(len(adjacency.get(n, set())) for n in component)
                max_edges = len(component) * (len(component) - 1)
                density = total_edges / max_edges if max_edges > 0 else 0.0

                communities.append(
                    CommunityResult(
                        community_id=f"community_{community_count}",
                        node_ids=component,
                        summary="",  # Would need LLM to generate
                        central_theme="",  # Would need LLM to generate
                        coherence_score=density,
                        relationship_density=density,
                    )
                )
                community_count += 1

        return communities

    def reset(self) -> None:
        """Reset retriever state."""
        self._retrieval_history.clear()

    def get_metrics(self) -> dict[str, Any]:
        """Get retriever metrics for telemetry."""
        if not self._retrieval_history:
            return {
                "total_retrievals": 0,
                "avg_results": 0,
                "avg_nodes_explored": 0,
                "avg_time_ms": 0,
            }

        total = len(self._retrieval_history)

        return {
            "total_retrievals": total,
            "avg_results": sum(len(r.results) for r in self._retrieval_history) / total,
            "avg_nodes_explored": sum(r.total_nodes_explored for r in self._retrieval_history)
            / total,
            "avg_time_ms": sum(r.retrieval_time_ms for r in self._retrieval_history) / total,
            "avg_communities": sum(len(r.communities) for r in self._retrieval_history) / total,
        }


# Convenience functions


def create_graph_rag_retriever(
    vector_store: VectorStoreProtocol,
    graph_store: GraphStoreProtocol,
    max_hops: int = 2,
    **kwargs: Any,
) -> GraphRAGRetriever:
    """Create a GraphRAG retriever with common configuration.

    Args:
        vector_store: Vector store for embeddings
        graph_store: Graph store for relationships
        max_hops: Maximum traversal depth
        **kwargs: Additional config options

    Returns:
        Configured GraphRAGRetriever
    """
    config = GraphRAGConfig(max_hops=max_hops, **kwargs)
    return GraphRAGRetriever(vector_store, graph_store, config)
