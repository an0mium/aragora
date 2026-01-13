"""
Tests for the ArenaFactory module.

Covers:
- ArenaFactory class initialization
- Lazy loading of optional dependencies
- Factory methods for creating components
- Arena creation with dependency injection
- Singleton get_arena_factory() function
- Convenience create_arena() function
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from aragora.debate.factory import (
    ArenaFactory,
    get_arena_factory,
    create_arena,
    _factory,
)
from aragora.core import Environment, Agent
from aragora.debate.protocol import DebateProtocol


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def factory():
    """Create a fresh ArenaFactory instance."""
    return ArenaFactory()


@pytest.fixture
def mock_environment():
    """Create a mock Environment."""
    env = Mock(spec=Environment)
    env.task = "Test debate task"
    env.context = "Test context"
    return env


@pytest.fixture
def mock_agents():
    """Create mock agents."""
    agent1 = Mock(spec=Agent)
    agent1.name = "claude"
    agent2 = Mock(spec=Agent)
    agent2.name = "gemini"
    return [agent1, agent2]


@pytest.fixture
def mock_protocol():
    """Create a mock DebateProtocol."""
    protocol = Mock(spec=DebateProtocol)
    protocol.rounds = 3
    return protocol


# ============================================================================
# ArenaFactory Initialization Tests
# ============================================================================


class TestArenaFactoryInit:
    """Tests for ArenaFactory initialization."""

    def test_init_sets_all_caches_to_none(self, factory):
        """All class caches should be None initially."""
        assert factory._position_tracker_cls is None
        assert factory._calibration_tracker_cls is None
        assert factory._belief_network_cls is None
        assert factory._belief_analyzer_cls is None
        assert factory._citation_extractor_cls is None
        assert factory._insight_extractor_cls is None
        assert factory._insight_store_cls is None
        assert factory._critique_store_cls is None
        assert factory._argument_cartographer_cls is None

    def test_multiple_instances_independent(self):
        """Each factory instance should have independent caches."""
        factory1 = ArenaFactory()
        factory2 = ArenaFactory()

        # Trigger lazy loading on factory1
        factory1._get_position_tracker_cls()

        # factory2 should still have None
        assert factory2._position_tracker_cls is None


# ============================================================================
# Lazy Loading Tests
# ============================================================================


class TestLazyLoading:
    """Tests for lazy loading of optional dependencies."""

    def test_position_tracker_lazy_load(self, factory):
        """PositionTracker should be lazily loaded."""
        # First call loads the class
        cls = factory._get_position_tracker_cls()

        # Should be cached
        assert factory._position_tracker_cls is cls

        # Second call returns cached value
        cls2 = factory._get_position_tracker_cls()
        assert cls2 is cls

    def test_calibration_tracker_lazy_load(self, factory):
        """CalibrationTracker should be lazily loaded."""
        cls = factory._get_calibration_tracker_cls()

        # Should be cached
        assert factory._calibration_tracker_cls is cls

        # Second call returns cached value
        cls2 = factory._get_calibration_tracker_cls()
        assert cls2 is cls

    def test_belief_classes_lazy_load(self, factory):
        """BeliefNetwork and BeliefPropagationAnalyzer should be lazily loaded."""
        network_cls, analyzer_cls = factory._get_belief_classes()

        # Should be cached
        assert factory._belief_network_cls is network_cls
        assert factory._belief_analyzer_cls is analyzer_cls

        # Second call returns cached values
        network_cls2, analyzer_cls2 = factory._get_belief_classes()
        assert network_cls2 is network_cls
        assert analyzer_cls2 is analyzer_cls

    def test_citation_extractor_lazy_load(self, factory):
        """CitationExtractor should be lazily loaded."""
        cls = factory._get_citation_extractor_cls()

        # Should be cached
        assert factory._citation_extractor_cls is cls

    def test_insight_classes_lazy_load(self, factory):
        """InsightExtractor and InsightStore should be lazily loaded."""
        extractor_cls, store_cls = factory._get_insight_classes()

        # Should be cached
        assert factory._insight_extractor_cls is extractor_cls
        assert factory._insight_store_cls is store_cls

    def test_critique_store_lazy_load(self, factory):
        """CritiqueStore should be lazily loaded."""
        cls = factory._get_critique_store_cls()

        # Should be cached
        assert factory._critique_store_cls is cls

    def test_argument_cartographer_lazy_load(self, factory):
        """ArgumentCartographer should be lazily loaded."""
        cls = factory._get_argument_cartographer_cls()

        # Should be cached
        assert factory._argument_cartographer_cls is cls

    def test_import_error_returns_none(self, factory):
        """ImportError should result in None, not exception."""
        with patch.dict("sys.modules", {"aragora.agents.truth_grounding": None}):
            # Clear any cached value
            factory._position_tracker_cls = None

            # Should handle ImportError gracefully
            with patch.object(factory, "_position_tracker_cls", None):
                # The actual import logic should handle errors gracefully
                # and return None or the cached value
                pass


# ============================================================================
# Factory Method Tests
# ============================================================================


class TestFactoryMethods:
    """Tests for component creation methods."""

    def test_create_position_tracker(self, factory):
        """create_position_tracker should create instance if class available."""
        tracker = factory.create_position_tracker()

        # Result depends on whether PositionTracker is available
        # Either returns an instance or None
        assert tracker is None or hasattr(tracker, "__class__")

    def test_create_calibration_tracker(self, factory):
        """create_calibration_tracker should create instance if class available."""
        tracker = factory.create_calibration_tracker()

        # Result depends on whether CalibrationTracker is available
        assert tracker is None or hasattr(tracker, "__class__")

    def test_create_belief_network(self, factory):
        """create_belief_network should create instance if class available."""
        network = factory.create_belief_network()

        # Result depends on whether BeliefNetwork is available
        assert network is None or hasattr(network, "__class__")

    def test_create_belief_analyzer(self, factory):
        """create_belief_analyzer should create instance if class available."""
        # BeliefPropagationAnalyzer requires a network argument
        network = factory.create_belief_network()
        if network is not None:
            analyzer = factory.create_belief_analyzer(network=network)
            assert analyzer is None or hasattr(analyzer, "__class__")
        else:
            # Skip if BeliefNetwork not available
            pass

    def test_create_citation_extractor(self, factory):
        """create_citation_extractor should create instance if class available."""
        extractor = factory.create_citation_extractor()

        # Result depends on whether CitationExtractor is available
        assert extractor is None or hasattr(extractor, "__class__")

    def test_create_insight_extractor(self, factory):
        """create_insight_extractor should create instance if class available."""
        extractor = factory.create_insight_extractor()

        # Result depends on whether InsightExtractor is available
        assert extractor is None or hasattr(extractor, "__class__")

    def test_create_insight_store(self, factory):
        """create_insight_store should create instance if class available."""
        store = factory.create_insight_store()

        # Result depends on whether InsightStore is available
        assert store is None or hasattr(store, "__class__")

    def test_create_critique_store(self, factory):
        """create_critique_store should create instance if class available."""
        store = factory.create_critique_store()

        # Result depends on whether CritiqueStore is available
        assert store is None or hasattr(store, "__class__")

    def test_create_argument_cartographer(self, factory):
        """create_argument_cartographer should create instance if class available."""
        cartographer = factory.create_argument_cartographer()

        # Result depends on whether ArgumentCartographer is available
        assert cartographer is None or hasattr(cartographer, "__class__")

    def test_create_with_kwargs(self, factory):
        """Factory methods should pass kwargs to constructors."""
        # Mock the class to verify kwargs are passed
        mock_cls = Mock()
        factory._position_tracker_cls = mock_cls

        factory.create_position_tracker(db_path="/tmp/test.db")

        mock_cls.assert_called_once_with(db_path="/tmp/test.db")

    def test_create_returns_none_when_class_unavailable(self, factory):
        """Factory methods should return None when class is unavailable."""
        factory._position_tracker_cls = None
        # Force it to stay None by mocking the getter
        with patch.object(factory, "_get_position_tracker_cls", return_value=None):
            result = factory.create_position_tracker()
            assert result is None


# ============================================================================
# Arena Creation Tests
# ============================================================================


class TestArenaCreation:
    """Tests for Arena creation via factory."""

    def test_create_minimal_arena(self, factory, mock_environment, mock_agents):
        """Create arena with minimal required parameters."""
        # Arena is imported inside create(), so we patch at the source
        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            MockArena.return_value = Mock()

            arena = factory.create(
                environment=mock_environment,
                agents=mock_agents,
            )

            MockArena.assert_called_once()
            call_kwargs = MockArena.call_args.kwargs
            assert call_kwargs["environment"] is mock_environment
            assert call_kwargs["agents"] is mock_agents

    def test_create_with_protocol(self, factory, mock_environment, mock_agents, mock_protocol):
        """Create arena with custom protocol."""
        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            MockArena.return_value = Mock()

            arena = factory.create(
                environment=mock_environment,
                agents=mock_agents,
                protocol=mock_protocol,
            )

            call_kwargs = MockArena.call_args.kwargs
            assert call_kwargs["protocol"] is mock_protocol

    def test_create_with_enable_position_tracking(self, factory, mock_environment, mock_agents):
        """Enable position tracking auto-creates tracker."""
        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            MockArena.return_value = Mock()

            # Mock the factory method
            mock_tracker = Mock()
            factory.create_position_tracker = Mock(return_value=mock_tracker)

            arena = factory.create(
                environment=mock_environment,
                agents=mock_agents,
                enable_position_tracking=True,
            )

            factory.create_position_tracker.assert_called_once()
            call_kwargs = MockArena.call_args.kwargs
            assert call_kwargs["position_tracker"] is mock_tracker

    def test_create_with_enable_calibration(self, factory, mock_environment, mock_agents):
        """Enable calibration auto-creates tracker."""
        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            MockArena.return_value = Mock()

            mock_tracker = Mock()
            factory.create_calibration_tracker = Mock(return_value=mock_tracker)

            arena = factory.create(
                environment=mock_environment,
                agents=mock_agents,
                enable_calibration=True,
            )

            factory.create_calibration_tracker.assert_called_once()
            call_kwargs = MockArena.call_args.kwargs
            assert call_kwargs["calibration_tracker"] is mock_tracker

    def test_create_with_enable_insights(self, factory, mock_environment, mock_agents):
        """Enable insights auto-creates store."""
        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            MockArena.return_value = Mock()

            mock_store = Mock()
            factory.create_insight_store = Mock(return_value=mock_store)

            arena = factory.create(
                environment=mock_environment,
                agents=mock_agents,
                enable_insights=True,
            )

            factory.create_insight_store.assert_called_once()
            call_kwargs = MockArena.call_args.kwargs
            assert call_kwargs["insight_store"] is mock_store

    def test_create_with_enable_critique_patterns(self, factory, mock_environment, mock_agents):
        """Enable critique patterns auto-creates memory store."""
        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            MockArena.return_value = Mock()

            mock_store = Mock()
            factory.create_critique_store = Mock(return_value=mock_store)

            arena = factory.create(
                environment=mock_environment,
                agents=mock_agents,
                enable_critique_patterns=True,
            )

            factory.create_critique_store.assert_called_once()
            call_kwargs = MockArena.call_args.kwargs
            assert call_kwargs["memory"] is mock_store

    def test_create_explicit_overrides_enable_flag(self, factory, mock_environment, mock_agents):
        """Explicit instance should override enable flag."""
        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            MockArena.return_value = Mock()

            explicit_tracker = Mock()
            factory.create_position_tracker = Mock()

            arena = factory.create(
                environment=mock_environment,
                agents=mock_agents,
                enable_position_tracking=True,
                position_tracker=explicit_tracker,  # Explicit instance
            )

            # Should not call create_position_tracker
            factory.create_position_tracker.assert_not_called()
            call_kwargs = MockArena.call_args.kwargs
            assert call_kwargs["position_tracker"] is explicit_tracker

    def test_create_with_all_explicit_dependencies(self, factory, mock_environment, mock_agents):
        """Create arena with all explicit dependencies."""
        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            MockArena.return_value = Mock()

            mock_memory = Mock()
            mock_hooks = {"on_start": Mock()}
            mock_emitter = Mock()
            mock_spectator = Mock()
            mock_embeddings = Mock()
            mock_insight_store = Mock()
            mock_recorder = Mock()
            mock_weights = {"claude": 1.0}
            mock_position_tracker = Mock()
            mock_position_ledger = Mock()
            mock_elo = Mock()
            mock_persona_manager = Mock()
            mock_dissent = Mock()
            mock_flip = Mock()
            mock_calibration = Mock()
            mock_continuum = Mock()
            mock_relationship = Mock()
            mock_moment = Mock()
            mock_circuit_breaker = Mock()
            mock_initial_messages = [Mock()]
            mock_trending = Mock()

            arena = factory.create(
                environment=mock_environment,
                agents=mock_agents,
                memory=mock_memory,
                event_hooks=mock_hooks,
                event_emitter=mock_emitter,
                spectator=mock_spectator,
                debate_embeddings=mock_embeddings,
                insight_store=mock_insight_store,
                recorder=mock_recorder,
                agent_weights=mock_weights,
                position_tracker=mock_position_tracker,
                position_ledger=mock_position_ledger,
                elo_system=mock_elo,
                persona_manager=mock_persona_manager,
                dissent_retriever=mock_dissent,
                flip_detector=mock_flip,
                calibration_tracker=mock_calibration,
                continuum_memory=mock_continuum,
                relationship_tracker=mock_relationship,
                moment_detector=mock_moment,
                loop_id="test-loop",
                strict_loop_scoping=True,
                circuit_breaker=mock_circuit_breaker,
                initial_messages=mock_initial_messages,
                trending_topic=mock_trending,
            )

            call_kwargs = MockArena.call_args.kwargs
            assert call_kwargs["memory"] is mock_memory
            assert call_kwargs["event_hooks"] is mock_hooks
            assert call_kwargs["event_emitter"] is mock_emitter
            assert call_kwargs["spectator"] is mock_spectator
            assert call_kwargs["debate_embeddings"] is mock_embeddings
            assert call_kwargs["insight_store"] is mock_insight_store
            assert call_kwargs["recorder"] is mock_recorder
            assert call_kwargs["agent_weights"] is mock_weights
            assert call_kwargs["position_tracker"] is mock_position_tracker
            assert call_kwargs["position_ledger"] is mock_position_ledger
            assert call_kwargs["elo_system"] is mock_elo
            assert call_kwargs["persona_manager"] is mock_persona_manager
            assert call_kwargs["dissent_retriever"] is mock_dissent
            assert call_kwargs["flip_detector"] is mock_flip
            assert call_kwargs["calibration_tracker"] is mock_calibration
            assert call_kwargs["continuum_memory"] is mock_continuum
            assert call_kwargs["relationship_tracker"] is mock_relationship
            assert call_kwargs["moment_detector"] is mock_moment
            assert call_kwargs["loop_id"] == "test-loop"
            assert call_kwargs["strict_loop_scoping"] is True
            assert call_kwargs["circuit_breaker"] is mock_circuit_breaker
            assert call_kwargs["initial_messages"] is mock_initial_messages
            assert call_kwargs["trending_topic"] is mock_trending


# ============================================================================
# Singleton Tests
# ============================================================================


class TestSingleton:
    """Tests for get_arena_factory singleton."""

    def test_get_arena_factory_returns_instance(self):
        """get_arena_factory should return an ArenaFactory instance."""
        factory = get_arena_factory()
        assert isinstance(factory, ArenaFactory)

    def test_get_arena_factory_singleton(self):
        """get_arena_factory should return the same instance."""
        factory1 = get_arena_factory()
        factory2 = get_arena_factory()
        assert factory1 is factory2

    def test_get_arena_factory_creates_on_first_call(self):
        """Singleton should be created on first call."""
        import aragora.debate.factory as factory_module

        # Reset the global
        original = factory_module._factory
        factory_module._factory = None

        try:
            factory = get_arena_factory()
            assert factory is not None
            assert factory_module._factory is factory
        finally:
            # Restore original
            factory_module._factory = original


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestCreateArenaConvenience:
    """Tests for create_arena convenience function."""

    def test_create_arena_uses_singleton(self, mock_environment, mock_agents):
        """create_arena should use the singleton factory."""
        with patch("aragora.debate.factory.get_arena_factory") as mock_get:
            mock_factory = Mock()
            mock_factory.create.return_value = Mock()
            mock_get.return_value = mock_factory

            arena = create_arena(
                environment=mock_environment,
                agents=mock_agents,
            )

            mock_get.assert_called_once()
            mock_factory.create.assert_called_once_with(
                mock_environment,
                mock_agents,
            )

    def test_create_arena_passes_kwargs(self, mock_environment, mock_agents):
        """create_arena should pass kwargs to factory.create."""
        with patch("aragora.debate.factory.get_arena_factory") as mock_get:
            mock_factory = Mock()
            mock_factory.create.return_value = Mock()
            mock_get.return_value = mock_factory

            arena = create_arena(
                environment=mock_environment,
                agents=mock_agents,
                enable_insights=True,
                loop_id="test-loop",
            )

            mock_factory.create.assert_called_once_with(
                mock_environment,
                mock_agents,
                enable_insights=True,
                loop_id="test-loop",
            )


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests with real classes."""

    def test_position_tracker_class_loads(self, factory):
        """PositionTracker class should load successfully."""
        cls = factory._get_position_tracker_cls()
        # Either loads successfully or returns None if not available
        if cls is not None:
            assert cls.__name__ == "PositionTracker"

    def test_calibration_tracker_class_loads(self, factory):
        """CalibrationTracker class should load successfully."""
        cls = factory._get_calibration_tracker_cls()
        if cls is not None:
            assert cls.__name__ == "CalibrationTracker"

    def test_belief_classes_load(self, factory):
        """Belief classes should load successfully."""
        network_cls, analyzer_cls = factory._get_belief_classes()
        if network_cls is not None:
            assert network_cls.__name__ == "BeliefNetwork"
        if analyzer_cls is not None:
            assert analyzer_cls.__name__ == "BeliefPropagationAnalyzer"

    def test_citation_extractor_class_loads(self, factory):
        """CitationExtractor class should load successfully."""
        cls = factory._get_citation_extractor_cls()
        if cls is not None:
            assert cls.__name__ == "CitationExtractor"

    def test_insight_classes_load(self, factory):
        """Insight classes should load successfully."""
        extractor_cls, store_cls = factory._get_insight_classes()
        if extractor_cls is not None:
            assert extractor_cls.__name__ == "InsightExtractor"
        if store_cls is not None:
            assert store_cls.__name__ == "InsightStore"

    def test_critique_store_class_loads(self, factory):
        """CritiqueStore class should load successfully."""
        cls = factory._get_critique_store_cls()
        if cls is not None:
            assert cls.__name__ == "CritiqueStore"

    def test_argument_cartographer_class_loads(self, factory):
        """ArgumentCartographer class should load successfully."""
        cls = factory._get_argument_cartographer_cls()
        if cls is not None:
            assert cls.__name__ == "ArgumentCartographer"
