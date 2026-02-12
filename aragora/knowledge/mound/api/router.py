"""LaRA-style router for Knowledge Mound retrieval paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from collections.abc import Sequence


RouteType = Literal["keyword", "semantic", "graph", "rlm", "long_context"]


@dataclass(frozen=True)
class QueryFeatures:
    """Lightweight query features for routing."""

    query: str
    length: int
    token_count: int
    has_graph_hint: bool
    graph_start_id: str | None
    has_rlm_hint: bool
    has_temporal_hint: bool


@dataclass(frozen=True)
class DocumentFeatures:
    """Lightweight corpus features for routing decisions."""

    total_nodes: int | None = None


@dataclass(frozen=True)
class RoutingDecision:
    """Routing decision with justification."""

    route: RouteType
    reason: str
    start_id: str | None = None


class LaRARouter:
    """Heuristic LaRA-style router (RAG vs Graph vs RLM)."""

    def __init__(
        self,
        min_nodes_for_routing: int = 200,
        query_length_threshold: int = 120,
        long_context_node_threshold: int = 1000,
        graph_hint_prefixes: Sequence[str] = ("graph:", "node:", "id:"),
        rlm_hint_keywords: Sequence[str] = (
            "summarize",
            "overview",
            "explain",
            "history",
            "context",
            "compare",
            "tradeoff",
            "why",
        ),
        temporal_hint_keywords: Sequence[str] = ("timeline", "evolution", "over time"),
    ) -> None:
        self.min_nodes_for_routing = min_nodes_for_routing
        self.query_length_threshold = query_length_threshold
        self.long_context_node_threshold = long_context_node_threshold
        self.graph_hint_prefixes = tuple(s.lower() for s in graph_hint_prefixes)
        self.rlm_hint_keywords = tuple(s.lower() for s in rlm_hint_keywords)
        self.temporal_hint_keywords = tuple(s.lower() for s in temporal_hint_keywords)

    def analyze_query(self, query: str) -> QueryFeatures:
        query_text = query.strip()
        lower = query_text.lower()
        tokens = [t for t in query_text.split() if t]
        graph_start_id = self._extract_graph_start_id(query_text)
        has_graph_hint = graph_start_id is not None
        has_rlm_hint = any(keyword in lower for keyword in self.rlm_hint_keywords)
        has_temporal_hint = any(keyword in lower for keyword in self.temporal_hint_keywords)
        return QueryFeatures(
            query=query_text,
            length=len(query_text),
            token_count=len(tokens),
            has_graph_hint=has_graph_hint,
            graph_start_id=graph_start_id,
            has_rlm_hint=has_rlm_hint,
            has_temporal_hint=has_temporal_hint,
        )

    def route(
        self,
        query: str,
        document_features: DocumentFeatures,
        *,
        supports_rlm: bool = False,
        force_route: RouteType | None = None,
    ) -> RoutingDecision:
        if force_route is not None:
            return RoutingDecision(route=force_route, reason="forced route override")

        features = self.analyze_query(query)
        total_nodes = document_features.total_nodes or 0

        if features.has_graph_hint and features.graph_start_id:
            return RoutingDecision(
                route="graph",
                reason="explicit graph hint detected",
                start_id=features.graph_start_id,
            )

        if supports_rlm and (
            features.has_rlm_hint or features.length >= self.query_length_threshold
        ):
            if total_nodes >= self.min_nodes_for_routing:
                return RoutingDecision(
                    route="rlm",
                    reason="complex query with sufficient context",
                )

        if total_nodes >= self.long_context_node_threshold and features.length >= 40:
            return RoutingDecision(
                route="long_context",
                reason="large corpus suggests long-context retrieval",
            )

        if features.token_count <= 4 and total_nodes < self.min_nodes_for_routing:
            return RoutingDecision(
                route="keyword",
                reason="short query and small corpus",
            )

        return RoutingDecision(route="semantic", reason="default semantic retrieval")

    def _extract_graph_start_id(self, query: str) -> str | None:
        stripped = query.strip()
        lower = stripped.lower()
        for prefix in self.graph_hint_prefixes:
            if lower.startswith(prefix):
                return stripped[len(prefix) :].strip() or None
        return None
