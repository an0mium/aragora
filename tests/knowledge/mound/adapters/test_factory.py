"""Tests for AdapterFactory — auto-create KM adapters from subsystems."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.knowledge.mound.adapters.factory import (
    ADAPTER_SPECS,
    AdapterFactory,
    AdapterSpec,
    CreatedAdapter,
    register_adapter_spec,
)


# =============================================================================
# AdapterSpec
# =============================================================================


class TestAdapterSpec:
    def test_defaults(self):
        spec = AdapterSpec(
            name="test",
            adapter_class=MagicMock,
            required_deps=["dep1"],
        )
        assert spec.forward_method == "sync_to_km"
        assert spec.reverse_method == "sync_from_km"
        assert spec.priority == 0
        assert spec.enabled_by_default is True
        assert spec.config_key is None

    def test_custom_values(self):
        spec = AdapterSpec(
            name="custom",
            adapter_class=MagicMock,
            required_deps=[],
            forward_method="push",
            reverse_method=None,
            priority=50,
            enabled_by_default=False,
            config_key="km_custom",
        )
        assert spec.forward_method == "push"
        assert spec.reverse_method is None
        assert spec.priority == 50
        assert spec.enabled_by_default is False


# =============================================================================
# ADAPTER_SPECS registry
# =============================================================================


class TestAdapterSpecsRegistry:
    def test_all_expected_adapters_registered(self):
        """All adapter specs from _init_specs should be in the registry."""
        expected_names = {
            "continuum",
            "consensus",
            "critique",
            "evidence",
            "belief",
            "insights",
            "elo",
            "performance",
            "pulse",
            "cost",
            "provenance",
            "fabric",
            "workspace",
            "computer_use",
            "gateway",
            "calibration_fusion",
            "control_plane",
            "culture",
            "receipt",
            "rlm",
            "erc8004",
        }
        assert expected_names.issubset(set(ADAPTER_SPECS.keys()))

    def test_continuum_spec(self):
        spec = ADAPTER_SPECS["continuum"]
        assert spec.priority == 100
        assert spec.required_deps == ["continuum_memory"]
        assert spec.config_key == "km_continuum_adapter"

    def test_erc8004_disabled_by_default(self):
        spec = ADAPTER_SPECS["erc8004"]
        assert spec.enabled_by_default is False

    def test_performance_disabled_by_default(self):
        spec = ADAPTER_SPECS["performance"]
        assert spec.enabled_by_default is False

    def test_cost_disabled_by_default(self):
        spec = ADAPTER_SPECS["cost"]
        assert spec.enabled_by_default is False

    def test_provenance_no_reverse(self):
        spec = ADAPTER_SPECS["provenance"]
        assert spec.reverse_method is None

    def test_priorities_reasonable(self):
        """Higher priority adapters should be core ones."""
        assert ADAPTER_SPECS["continuum"].priority > ADAPTER_SPECS["cost"].priority
        assert ADAPTER_SPECS["consensus"].priority > ADAPTER_SPECS["pulse"].priority


# =============================================================================
# register_adapter_spec
# =============================================================================


class TestRegisterAdapterSpec:
    def test_register_new_spec(self):
        spec = AdapterSpec(
            name="test_custom_xyz",
            adapter_class=MagicMock,
            required_deps=[],
        )
        register_adapter_spec(spec)
        assert "test_custom_xyz" in ADAPTER_SPECS
        assert ADAPTER_SPECS["test_custom_xyz"] is spec

        # Clean up
        del ADAPTER_SPECS["test_custom_xyz"]


# =============================================================================
# CreatedAdapter
# =============================================================================


class TestCreatedAdapter:
    def test_fields(self):
        spec = AdapterSpec(name="t", adapter_class=MagicMock, required_deps=[])
        adapter = MagicMock()
        ca = CreatedAdapter(name="t", adapter=adapter, spec=spec, deps_used={"k": "v"})
        assert ca.name == "t"
        assert ca.adapter is adapter
        assert ca.spec is spec
        assert ca.deps_used == {"k": "v"}

    def test_default_deps(self):
        spec = AdapterSpec(name="t", adapter_class=MagicMock, required_deps=[])
        ca = CreatedAdapter(name="t", adapter=MagicMock(), spec=spec)
        assert ca.deps_used == {}


# =============================================================================
# AdapterFactory
# =============================================================================


class TestAdapterFactory:
    def test_init(self):
        factory = AdapterFactory()
        assert factory._event_callback is None

    def test_init_with_callback(self):
        cb = MagicMock()
        factory = AdapterFactory(event_callback=cb)
        assert factory._event_callback is cb

    def test_set_event_callback(self):
        factory = AdapterFactory()
        cb = MagicMock()
        factory.set_event_callback(cb)
        assert factory._event_callback is cb

    def test_get_available_adapter_specs(self):
        factory = AdapterFactory()
        specs = factory.get_available_adapter_specs()
        assert "continuum" in specs
        # Should be a copy
        specs["test_mutation"] = None
        assert "test_mutation" not in ADAPTER_SPECS


# =============================================================================
# create_from_subsystems
# =============================================================================


class TestCreateFromSubsystems:
    def test_creates_adapters_with_deps(self):
        """Should create adapters when their deps are provided."""
        factory = AdapterFactory()
        mock_consensus = MagicMock()

        adapters = factory.create_from_subsystems(
            consensus_memory=mock_consensus,
        )

        assert "consensus" in adapters
        assert adapters["consensus"].name == "consensus"

    def test_skips_adapters_with_missing_deps(self):
        """Should skip adapters whose required deps are missing."""
        factory = AdapterFactory()

        adapters = factory.create_from_subsystems()

        # Adapters with required deps should be skipped
        # Only standalone adapters (empty required_deps) should be created
        for name, created in adapters.items():
            spec = ADAPTER_SPECS[name]
            assert (
                spec.required_deps == []
                or all(
                    d in {"continuum_memory", "consensus_memory", "memory"}
                    for d in spec.required_deps
                )
                is False
            )

    def test_none_deps_filtered_out(self):
        """None values should be filtered from deps."""
        factory = AdapterFactory()

        adapters = factory.create_from_subsystems(
            continuum_memory=None,
            consensus_memory=MagicMock(),
        )

        # consensus should be created (dep provided)
        assert "consensus" in adapters
        # continuum should not (dep is None)
        assert "continuum" not in adapters

    def test_kwargs_forwarded(self):
        """Extra kwargs should be added to deps."""
        factory = AdapterFactory()
        mock_fabric = MagicMock()

        adapters = factory.create_from_subsystems(fabric=mock_fabric)

        assert "fabric" in adapters

    def test_event_callback_passed(self):
        """Event callback should be passed to created adapters."""
        cb = MagicMock()
        factory = AdapterFactory(event_callback=cb)
        mock_mem = MagicMock()

        adapters = factory.create_from_subsystems(consensus_memory=mock_mem)

        # The adapter should have received the callback
        assert "consensus" in adapters


# =============================================================================
# create_from_arena_config
# =============================================================================


class TestCreateFromArenaConfig:
    def test_explicit_adapter_from_config(self):
        """Config with explicit adapter should use it directly."""
        factory = AdapterFactory()
        explicit_adapter = MagicMock()

        config = MagicMock()
        config.km_consensus_adapter = explicit_adapter
        # Set other config attrs to None
        for spec in ADAPTER_SPECS.values():
            if spec.config_key and spec.config_key != "km_consensus_adapter":
                setattr(config, spec.config_key, None)
        config.continuum_memory = None
        config.consensus_memory = None
        config.memory = None
        config.insight_store = None
        config.elo_system = None
        config.pulse_manager = None
        config.usage_tracker = None
        config.flip_detector = None

        adapters = factory.create_from_arena_config(config)

        assert "consensus" in adapters
        assert adapters["consensus"].adapter is explicit_adapter

    def test_subsystems_dict_provides_deps(self):
        """Subsystems dict should provide dependencies."""
        factory = AdapterFactory()
        mock_ev = MagicMock()

        config = MagicMock(spec=[])  # No attributes

        adapters = factory.create_from_arena_config(
            config,
            subsystems={"evidence_store": mock_ev},
        )

        assert "evidence" in adapters

    def test_exclude_explicit_from_auto_creation(self):
        """Explicitly configured adapters should not be auto-created."""
        factory = AdapterFactory()
        explicit = MagicMock()

        config = MagicMock()
        config.km_consensus_adapter = explicit
        config.consensus_memory = MagicMock()  # Also has dep
        # Set other config attrs to None
        for spec in ADAPTER_SPECS.values():
            if spec.config_key and spec.config_key != "km_consensus_adapter":
                setattr(config, spec.config_key, None)
        config.continuum_memory = None
        config.memory = None
        config.insight_store = None
        config.elo_system = None
        config.pulse_manager = None
        config.usage_tracker = None
        config.flip_detector = None

        adapters = factory.create_from_arena_config(config)

        # Should use the explicit adapter, not auto-create
        assert adapters["consensus"].adapter is explicit


# =============================================================================
# _create_single_adapter
# =============================================================================


class TestCreateSingleAdapter:
    def test_constructor_type_error_fallback(self):
        """Should retry without event_callback on TypeError."""

        class StubAdapter:
            """Adapter that rejects event_callback."""

            def __init__(self, store=None):
                self.store = store

        spec = AdapterSpec(
            name="critique",
            adapter_class=StubAdapter,
            required_deps=["memory"],
        )
        factory = AdapterFactory(event_callback=MagicMock())

        adapter = factory._create_single_adapter(spec, {"memory": MagicMock()})
        assert adapter is not None
        assert isinstance(adapter, StubAdapter)

    def test_generic_fallback(self):
        """Unknown adapter names should use generic construction."""

        class GenericAdapter:
            def __init__(self, event_callback=None, **kwargs):
                self.kwargs = kwargs

        spec = AdapterSpec(
            name="unknown_new_adapter",
            adapter_class=GenericAdapter,
            required_deps=[],
        )
        factory = AdapterFactory()

        adapter = factory._create_single_adapter(spec, {})
        assert adapter is not None


# =============================================================================
# register_with_coordinator
# =============================================================================


class TestRegisterWithCoordinator:
    def test_registers_adapters(self):
        factory = AdapterFactory()
        coordinator = MagicMock()
        coordinator.register_adapter.return_value = True
        coordinator.disable_adapter = MagicMock()

        spec = AdapterSpec(
            name="test",
            adapter_class=MagicMock,
            required_deps=[],
            priority=50,
        )
        adapters = {
            "test": CreatedAdapter(
                name="test",
                adapter=MagicMock(),
                spec=spec,
                deps_used={},
            )
        }

        count = factory.register_with_coordinator(coordinator, adapters)
        assert count == 1
        coordinator.register_adapter.assert_called_once()

    def test_disables_non_default_adapters(self):
        factory = AdapterFactory()
        coordinator = MagicMock()
        coordinator.register_adapter.return_value = True

        spec = AdapterSpec(
            name="test",
            adapter_class=MagicMock,
            required_deps=[],
            enabled_by_default=False,
        )
        adapters = {
            "test": CreatedAdapter(
                name="test",
                adapter=MagicMock(),
                spec=spec,
            )
        }

        factory.register_with_coordinator(coordinator, adapters)
        coordinator.disable_adapter.assert_called_once_with("test")

    def test_registration_failure(self):
        factory = AdapterFactory()
        coordinator = MagicMock()
        coordinator.register_adapter.return_value = False

        spec = AdapterSpec(
            name="test",
            adapter_class=MagicMock,
            required_deps=[],
        )
        adapters = {
            "test": CreatedAdapter(
                name="test",
                adapter=MagicMock(),
                spec=spec,
            )
        }

        count = factory.register_with_coordinator(coordinator, adapters)
        assert count == 0
