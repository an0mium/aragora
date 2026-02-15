"""Tests for KM adapter auto-wiring into TeamSelector via SubsystemCoordinator.

Verifies that PerformanceAdapter and RankingAdapter are injected into
TeamSelector when KnowledgeMound is available, and that missing KM is
handled gracefully.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.subsystem_coordinator import SubsystemCoordinator


class TestKMAdapterAutoWiring:
    """Test KM adapter auto-wiring into TeamSelector."""

    def test_adapters_injected_when_km_available(self):
        """PerformanceAdapter and RankingAdapter should be wired when KM is available."""
        mock_km = MagicMock()
        mock_team_selector = MagicMock()
        mock_team_selector.performance_adapter = None
        mock_team_selector.ranking_adapter = None

        mock_adapter = MagicMock()
        mock_created = MagicMock()
        mock_created.adapter = mock_adapter

        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_subsystems"
        ):
            coord = SubsystemCoordinator.__new__(SubsystemCoordinator)
            coord.knowledge_mound = mock_km
            coord.team_selector = mock_team_selector
            coord.elo_system = MagicMock()

        with patch("aragora.knowledge.mound.adapters.factory.AdapterFactory") as MockFactory:
            factory_instance = MockFactory.return_value
            factory_instance.create_from_subsystems.return_value = {
                "performance": mock_created,
            }

            coord._auto_wire_km_adapters_to_team_selector()

        assert mock_team_selector.performance_adapter == mock_adapter
        assert mock_team_selector.ranking_adapter == mock_adapter

    def test_missing_km_handled_gracefully(self):
        """No error when KnowledgeMound is None."""
        mock_team_selector = MagicMock()
        mock_team_selector.performance_adapter = None
        mock_team_selector.ranking_adapter = None

        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_subsystems"
        ):
            coord = SubsystemCoordinator.__new__(SubsystemCoordinator)
            coord.knowledge_mound = None
            coord.team_selector = mock_team_selector

        # Should be a no-op, no error
        coord._auto_wire_km_adapters_to_team_selector()

        assert mock_team_selector.performance_adapter is None
        assert mock_team_selector.ranking_adapter is None

    def test_import_error_handled_gracefully(self):
        """No error when adapter factory module is not importable."""
        mock_km = MagicMock()
        mock_team_selector = MagicMock()
        mock_team_selector.performance_adapter = None
        mock_team_selector.ranking_adapter = None

        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_subsystems"
        ):
            coord = SubsystemCoordinator.__new__(SubsystemCoordinator)
            coord.knowledge_mound = mock_km
            coord.team_selector = mock_team_selector
            coord.elo_system = None

        with patch(
            "aragora.knowledge.mound.adapters.factory.AdapterFactory",
            side_effect=ImportError("no module"),
        ):
            # Should not raise
            coord._auto_wire_km_adapters_to_team_selector()

        assert mock_team_selector.performance_adapter is None

    def test_existing_adapters_not_overwritten(self):
        """Pre-configured adapters should not be overwritten."""
        mock_km = MagicMock()
        existing_perf = MagicMock(name="existing_perf")
        existing_rank = MagicMock(name="existing_rank")
        mock_team_selector = MagicMock()
        mock_team_selector.performance_adapter = existing_perf
        mock_team_selector.ranking_adapter = existing_rank

        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_subsystems"
        ):
            coord = SubsystemCoordinator.__new__(SubsystemCoordinator)
            coord.knowledge_mound = mock_km
            coord.team_selector = mock_team_selector
            coord.elo_system = MagicMock()

        new_adapter = MagicMock(name="new_adapter")
        mock_created = MagicMock()
        mock_created.adapter = new_adapter

        with patch("aragora.knowledge.mound.adapters.factory.AdapterFactory") as MockFactory:
            factory_instance = MockFactory.return_value
            factory_instance.create_from_subsystems.return_value = {
                "performance": mock_created,
            }

            coord._auto_wire_km_adapters_to_team_selector()

        # Should keep the existing adapters
        assert mock_team_selector.performance_adapter is existing_perf
        assert mock_team_selector.ranking_adapter is existing_rank

    def test_no_performance_adapter_created(self):
        """Graceful when factory returns no performance adapter."""
        mock_km = MagicMock()
        mock_team_selector = MagicMock()
        mock_team_selector.performance_adapter = None
        mock_team_selector.ranking_adapter = None

        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_subsystems"
        ):
            coord = SubsystemCoordinator.__new__(SubsystemCoordinator)
            coord.knowledge_mound = mock_km
            coord.team_selector = mock_team_selector
            coord.elo_system = None

        with patch("aragora.knowledge.mound.adapters.factory.AdapterFactory") as MockFactory:
            factory_instance = MockFactory.return_value
            factory_instance.create_from_subsystems.return_value = {}

            coord._auto_wire_km_adapters_to_team_selector()

        assert mock_team_selector.performance_adapter is None
        assert mock_team_selector.ranking_adapter is None

    def test_missing_team_selector_handled(self):
        """No error when TeamSelector is None."""
        mock_km = MagicMock()

        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_subsystems"
        ):
            coord = SubsystemCoordinator.__new__(SubsystemCoordinator)
            coord.knowledge_mound = mock_km
            coord.team_selector = None

        # Should be a no-op, no error
        coord._auto_wire_km_adapters_to_team_selector()
