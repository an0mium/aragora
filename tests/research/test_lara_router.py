"""Tests for LaRARouter."""

import pytest
from aragora.routing.lara_router import (
    LaRARouter,
    LaRAConfig,
    QueryFeatures,
    RetrievalMode,
    RoutingDecision,
    create_lara_router,
    quick_route,
)


class TestLaRARouter:
    """Test suite for LaRARouter."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        router = LaRARouter()
        assert router.config.long_context_min_tokens == 50000
        assert router.config.min_confidence_for_primary == 0.6

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = LaRAConfig(
            long_context_min_tokens=30000,
            min_confidence_for_primary=0.7,
        )
        router = LaRARouter(config)
        assert router.config.long_context_min_tokens == 30000

    def test_factual_query_routes_to_rag(self) -> None:
        """Test that factual queries prefer RAG mode."""
        router = LaRARouter()

        decision = router.route(
            query="What is the capital of France?",
            doc_tokens=10000,
        )

        assert decision.selected_mode == RetrievalMode.RAG
        assert decision.query_features.is_factual is True

    def test_analytical_query_routes_to_rlm(self) -> None:
        """Test that analytical queries prefer RLM mode."""
        router = LaRARouter()

        decision = router.route(
            query="Why did the Roman Empire fall and how did it impact Europe?",
            doc_tokens=10000,
        )

        assert decision.selected_mode == RetrievalMode.RLM
        assert decision.query_features.is_analytical is True

    def test_multi_hop_query_routes_to_graph(self) -> None:
        """Test that multi-hop queries prefer Graph mode."""
        router = LaRARouter()

        decision = router.route(
            query="What is the relationship between climate change and biodiversity loss?",
            doc_tokens=10000,
        )

        assert decision.selected_mode == RetrievalMode.GRAPH
        assert decision.query_features.is_multi_hop is True

    def test_large_doc_routes_to_long_context(self) -> None:
        """Test that large documents with suitable queries use long context."""
        router = LaRARouter()

        decision = router.route(
            query="Summarize all the key points in this document",
            doc_tokens=75000,  # Within long context range
            available_modes={RetrievalMode.LONG_CONTEXT, RetrievalMode.RAG},
        )

        assert decision.selected_mode == RetrievalMode.LONG_CONTEXT

    def test_override_mode(self) -> None:
        """Test that override_mode forces specific mode."""
        router = LaRARouter()

        decision = router.route(
            query="What is the capital of France?",
            doc_tokens=10000,
            override_mode=RetrievalMode.GRAPH,
        )

        assert decision.selected_mode == RetrievalMode.GRAPH
        assert decision.confidence == 1.0
        assert "override" in decision.reasoning.lower()

    def test_fallback_mode_selected(self) -> None:
        """Test that fallback mode is provided."""
        router = LaRARouter()

        decision = router.route(
            query="Explain the causes of World War I",
            doc_tokens=10000,
        )

        # Should have a fallback
        assert decision.fallback_mode is not None
        assert decision.fallback_mode != decision.selected_mode

    def test_available_modes_respected(self) -> None:
        """Test that only available modes are considered."""
        router = LaRARouter()

        # Only allow RAG
        decision = router.route(
            query="Why did the Roman Empire fall?",  # Would normally prefer RLM
            doc_tokens=10000,
            available_modes={RetrievalMode.RAG},
        )

        assert decision.selected_mode == RetrievalMode.RAG

    def test_decision_has_duration(self) -> None:
        """Test that routing decision includes duration."""
        router = LaRARouter()

        decision = router.route(
            query="Test query",
            doc_tokens=10000,
        )

        assert decision.duration_ms >= 0

    def test_query_features_extraction(self) -> None:
        """Test query feature extraction."""
        router = LaRARouter()

        # Test factual query
        decision1 = router.route(
            query="When was the Eiffel Tower built?",
            doc_tokens=10000,
        )
        assert decision1.query_features.is_factual is True
        assert decision1.query_features.has_temporal_markers is True

        # Test analytical query
        decision2 = router.route(
            query="Compare and contrast democracy and authoritarianism",
            doc_tokens=10000,
        )
        assert decision2.query_features.is_analytical is True

    def test_complexity_score_increases_with_features(self) -> None:
        """Test that complexity score increases with query complexity."""
        router = LaRARouter()

        # Simple query
        simple_decision = router.route(
            query="What is AI?",
            doc_tokens=10000,
        )

        # Complex query
        complex_decision = router.route(
            query="Analyze the relationship between artificial intelligence and employment, "
            "comparing how different industries are affected and evaluating long-term implications",
            doc_tokens=10000,
        )

        assert (
            complex_decision.query_features.complexity_score
            > simple_decision.query_features.complexity_score
        )

    def test_reset_clears_history(self) -> None:
        """Test that reset clears decision history."""
        router = LaRARouter()

        # Make some decisions
        router.route("Test 1", 10000)
        router.route("Test 2", 10000)

        metrics = router.get_metrics()
        assert metrics["total_decisions"] == 2

        router.reset()

        metrics = router.get_metrics()
        assert metrics["total_decisions"] == 0

    def test_get_metrics(self) -> None:
        """Test metrics retrieval."""
        router = LaRARouter()

        # Make several decisions
        router.route("What is X?", 10000)
        router.route("Why did Y happen?", 10000)
        router.route("Compare A and B", 10000)

        metrics = router.get_metrics()

        assert metrics["total_decisions"] == 3
        assert "mode_distribution" in metrics
        assert "avg_confidence" in metrics
        assert metrics["avg_confidence"] > 0

    def test_reasoning_explains_decision(self) -> None:
        """Test that reasoning field explains the decision."""
        router = LaRARouter()

        decision = router.route(
            query="What is the population of Tokyo?",
            doc_tokens=10000,
        )

        assert decision.reasoning  # Not empty
        assert len(decision.reasoning) > 10  # Some meaningful content


class TestQueryFeatures:
    """Test QueryFeatures dataclass."""

    def test_query_features_defaults(self) -> None:
        """Test QueryFeatures can be created with all fields."""
        features = QueryFeatures(
            is_factual=True,
            is_analytical=False,
            is_multi_hop=False,
            requires_aggregation=False,
            length_tokens=10,
            has_temporal_markers=True,
            entity_count=1,
            complexity_score=0.2,
        )

        assert features.is_factual is True
        assert features.complexity_score == 0.2


class TestRoutingDecision:
    """Test RoutingDecision dataclass."""

    def test_routing_decision_creation(self) -> None:
        """Test RoutingDecision can be created."""
        features = QueryFeatures(
            is_factual=True,
            is_analytical=False,
            is_multi_hop=False,
            requires_aggregation=False,
            length_tokens=10,
            has_temporal_markers=False,
            entity_count=0,
            complexity_score=0.1,
        )

        decision = RoutingDecision(
            selected_mode=RetrievalMode.RAG,
            confidence=0.85,
            fallback_mode=RetrievalMode.RLM,
            reasoning="Test reasoning",
            query_features=features,
            doc_tokens=10000,
            duration_ms=5.0,
        )

        assert decision.selected_mode == RetrievalMode.RAG
        assert decision.confidence == 0.85


class TestRetrievalMode:
    """Test RetrievalMode enum."""

    def test_all_modes_have_values(self) -> None:
        """Test that all modes have string values."""
        assert RetrievalMode.RAG.value == "rag"
        assert RetrievalMode.RLM.value == "rlm"
        assert RetrievalMode.LONG_CONTEXT.value == "long_context"
        assert RetrievalMode.GRAPH.value == "graph"
        assert RetrievalMode.HYBRID.value == "hybrid"


class TestCreateLaraRouter:
    """Test the factory function."""

    def test_creates_router_with_defaults(self) -> None:
        """Test factory creates router with defaults."""
        router = create_lara_router()

        assert isinstance(router, LaRARouter)

    def test_creates_router_with_custom_threshold(self) -> None:
        """Test factory accepts custom configuration."""
        router = create_lara_router(long_context_threshold=30000)

        assert router.config.long_context_min_tokens == 30000


class TestQuickRoute:
    """Test the quick_route convenience function."""

    def test_quick_route_returns_mode(self) -> None:
        """Test quick_route returns a RetrievalMode."""
        mode = quick_route("What is AI?")

        assert isinstance(mode, RetrievalMode)

    def test_quick_route_factual_query(self) -> None:
        """Test quick_route with factual query."""
        mode = quick_route("What is the speed of light?")

        assert mode == RetrievalMode.RAG

    def test_quick_route_analytical_query(self) -> None:
        """Test quick_route with analytical query."""
        mode = quick_route("Analyze why the stock market crashed in 2008")

        assert mode == RetrievalMode.RLM
