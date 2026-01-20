"""
Tests for AdapterFactory - auto-creating KM adapters from Arena subsystems.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from aragora.knowledge.mound.adapters import (
    AdapterFactory,
    AdapterSpec,
    ADAPTER_SPECS,
    BeliefAdapter,
    EloAdapter,
)
from aragora.knowledge.mound.bidirectional_coordinator import BidirectionalCoordinator


class TestAdapterFactory:
    """Tests for AdapterFactory."""

    def test_factory_creation(self):
        """Test factory can be created."""
        factory = AdapterFactory()
        assert factory is not None

    def test_factory_with_event_callback(self):
        """Test factory with event callback."""
        callback = MagicMock()
        factory = AdapterFactory(event_callback=callback)
        assert factory._event_callback is callback

    def test_get_available_specs(self):
        """Test getting available adapter specifications."""
        factory = AdapterFactory()
        specs = factory.get_available_adapter_specs()

        # Should have all expected adapters
        assert "continuum" in specs
        assert "consensus" in specs
        assert "critique" in specs
        assert "evidence" in specs
        assert "belief" in specs
        assert "insights" in specs
        assert "elo" in specs
        assert "pulse" in specs
        assert "cost" in specs

    def test_spec_properties(self):
        """Test adapter spec properties."""
        spec = ADAPTER_SPECS["elo"]

        assert spec.name == "elo"
        assert spec.forward_method == "store_match"
        assert spec.reverse_method == "sync_km_to_elo"
        assert spec.priority == 40

    def test_create_belief_adapter_no_deps(self):
        """Test creating belief adapter which has no required deps."""
        factory = AdapterFactory()
        adapters = factory.create_from_subsystems()

        # Belief adapter should be created (no deps required)
        assert "belief" in adapters
        assert isinstance(adapters["belief"].adapter, BeliefAdapter)

    def test_create_elo_adapter_with_mock(self):
        """Test creating ELO adapter with mock ELO system."""
        mock_elo = MagicMock()
        mock_elo.get_rating = MagicMock(return_value=1500)

        factory = AdapterFactory()
        adapters = factory.create_from_subsystems(elo_system=mock_elo)

        # ELO adapter should be created
        assert "elo" in adapters
        assert isinstance(adapters["elo"].adapter, EloAdapter)
        assert adapters["elo"].deps_used["elo_system"] is mock_elo

    def test_missing_deps_skips_adapter(self):
        """Test that adapters with missing deps are skipped."""
        factory = AdapterFactory()
        # No deps provided, so adapters with required deps should be skipped
        adapters = factory.create_from_subsystems()

        # These should not be created without their deps
        assert "continuum" not in adapters
        assert "consensus" not in adapters
        assert "critique" not in adapters
        assert "evidence" not in adapters
        assert "insights" not in adapters
        # elo requires elo_system
        assert "elo" not in adapters
        assert "pulse" not in adapters
        assert "cost" not in adapters

        # Only belief should be created (no deps)
        assert "belief" in adapters

    def test_created_adapter_metadata(self):
        """Test created adapter has correct metadata."""
        mock_elo = MagicMock()
        factory = AdapterFactory()
        adapters = factory.create_from_subsystems(elo_system=mock_elo)

        created = adapters["elo"]
        assert created.name == "elo"
        assert created.spec.name == "elo"
        assert "elo_system" in created.deps_used

    def test_event_callback_passed_to_adapters(self):
        """Test event callback is passed to created adapters."""
        events = []

        def track_event(event_type, data):
            events.append((event_type, data))

        factory = AdapterFactory(event_callback=track_event)
        adapters = factory.create_from_subsystems()

        # Belief adapter should have the callback
        belief_adapter = adapters["belief"].adapter
        assert belief_adapter._event_callback is track_event

        # Trigger an event
        belief_adapter._emit_event("test_event", {"key": "value"})
        assert len(events) == 1
        assert events[0][0] == "test_event"


class TestAdapterFactoryWithCoordinator:
    """Tests for factory integration with BidirectionalCoordinator."""

    def test_register_with_coordinator(self):
        """Test registering adapters with coordinator."""
        mock_elo = MagicMock()
        factory = AdapterFactory()
        adapters = factory.create_from_subsystems(elo_system=mock_elo)

        coordinator = BidirectionalCoordinator()
        registered = factory.register_with_coordinator(coordinator, adapters)

        # At least belief and elo should be registered
        assert registered >= 2
        assert "belief" in coordinator.get_registered_adapters()
        assert "elo" in coordinator.get_registered_adapters()

    def test_coordinator_adapter_config(self):
        """Test coordinator has correct adapter configuration."""
        mock_elo = MagicMock()
        factory = AdapterFactory()
        adapters = factory.create_from_subsystems(elo_system=mock_elo)

        coordinator = BidirectionalCoordinator()
        factory.register_with_coordinator(coordinator, adapters)

        # Check status
        status = coordinator.get_status()
        assert status["total_adapters"] >= 2

        # Check adapter details
        adapter_status = status["adapters"]
        assert "elo" in adapter_status
        assert adapter_status["elo"]["enabled"] is True
        assert adapter_status["elo"]["has_reverse"] is True

    def test_cost_adapter_disabled_by_default(self):
        """Test cost adapter is disabled by default (opt-in)."""
        mock_tracker = MagicMock()
        factory = AdapterFactory()
        adapters = factory.create_from_subsystems(cost_tracker=mock_tracker)

        coordinator = BidirectionalCoordinator()
        factory.register_with_coordinator(coordinator, adapters)

        # Cost adapter should be registered but disabled
        assert "cost" in coordinator.get_registered_adapters()
        status = coordinator.get_status()
        assert status["adapters"]["cost"]["enabled"] is False


class TestAdapterFactoryFromConfig:
    """Tests for creating adapters from ArenaConfig."""

    def test_create_from_arena_config_explicit_adapter(self):
        """Test using explicit adapter from config."""
        # Create a mock ArenaConfig with explicit adapter
        mock_config = MagicMock()
        mock_config.km_belief_adapter = MagicMock()
        mock_config.continuum_memory = None
        mock_config.consensus_memory = None
        mock_config.memory = None
        mock_config.insight_store = None
        mock_config.elo_system = None
        mock_config.pulse_manager = None
        mock_config.usage_tracker = None
        mock_config.flip_detector = None

        factory = AdapterFactory()
        adapters = factory.create_from_arena_config(mock_config)

        # Should use explicit belief adapter
        assert "belief" in adapters
        assert adapters["belief"].adapter is mock_config.km_belief_adapter

    def test_create_from_arena_config_with_subsystems(self):
        """Test creating adapters from config with additional subsystems."""
        mock_config = MagicMock()
        # No explicit adapters
        for attr in ["km_belief_adapter", "km_elo_bridge", "km_continuum_adapter"]:
            setattr(mock_config, attr, None)
        mock_config.continuum_memory = None
        mock_config.consensus_memory = None
        mock_config.memory = None
        mock_config.insight_store = None
        mock_config.elo_system = MagicMock()  # Has ELO system
        mock_config.pulse_manager = None
        mock_config.usage_tracker = None
        mock_config.flip_detector = None

        factory = AdapterFactory()
        adapters = factory.create_from_arena_config(mock_config)

        # ELO adapter should be auto-created from config.elo_system
        assert "elo" in adapters


class TestAdapterSpecRegistry:
    """Tests for adapter spec registry."""

    def test_all_specs_have_required_fields(self):
        """Test all specs have required fields."""
        for name, spec in ADAPTER_SPECS.items():
            assert spec.name == name
            assert spec.adapter_class is not None
            assert spec.forward_method
            assert isinstance(spec.required_deps, list)
            assert isinstance(spec.priority, int)

    def test_specs_ordered_by_priority(self):
        """Test that high-priority adapters come first when sorted."""
        specs = sorted(ADAPTER_SPECS.values(), key=lambda s: s.priority, reverse=True)

        # Continuum should be highest priority (100)
        assert specs[0].name == "continuum"

        # Cost should be lowest (10)
        assert specs[-1].name == "cost"
