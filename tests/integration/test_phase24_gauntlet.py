"""Integration tests for Phase 24: Gauntlet Consolidation.

Tests the gauntlet package imports.
"""

import pytest


class TestGauntletImports:
    """Test canonical gauntlet imports work correctly."""

    def test_import_gauntlet_types(self):
        """Core types should be importable from aragora.gauntlet."""
        from aragora.gauntlet import (
            InputType,
            Verdict,
            SeverityLevel,
            GauntletPhase,
        )

        assert InputType is not None
        assert Verdict is not None
        assert SeverityLevel is not None

    def test_import_gauntlet_runner(self):
        """GauntletRunner should be importable."""
        from aragora.gauntlet import GauntletRunner, GauntletConfig

        assert GauntletRunner is not None
        assert GauntletConfig is not None

    def test_import_gauntlet_orchestrator(self):
        """GauntletOrchestrator should be importable from canonical location."""
        from aragora.gauntlet import (
            GauntletOrchestrator,
            OrchestratorConfig,
            GauntletProgress,
        )

        assert GauntletOrchestrator is not None
        assert OrchestratorConfig is not None
        assert GauntletProgress is not None

    def test_import_gauntlet_output_formats(self):
        """Output format classes should be importable."""
        from aragora.gauntlet import DecisionReceipt, RiskHeatmap, HeatmapCell

        assert DecisionReceipt is not None
        assert RiskHeatmap is not None
        assert HeatmapCell is not None


class TestGauntletFunctionality:
    """Test gauntlet functionality works after consolidation."""

    def test_input_type_enum_values(self):
        """InputType enum should have expected values."""
        from aragora.gauntlet import InputType

        assert hasattr(InputType, "SPEC")
        assert hasattr(InputType, "CODE")
        assert hasattr(InputType, "POLICY")
        assert hasattr(InputType, "ARCHITECTURE")

    def test_verdict_enum_values(self):
        """Verdict enum should have expected values."""
        from aragora.gauntlet import Verdict

        assert hasattr(Verdict, "APPROVED")
        assert hasattr(Verdict, "NEEDS_REVIEW")
        assert hasattr(Verdict, "REJECTED")

    def test_gauntlet_config_creation(self):
        """GauntletConfig should be creatable with defaults."""
        from aragora.gauntlet import GauntletConfig

        config = GauntletConfig()
        assert config is not None

    def test_orchestrator_config_creation(self):
        """OrchestratorConfig should be creatable with defaults."""
        from aragora.gauntlet import OrchestratorConfig, InputType

        config = OrchestratorConfig(
            input_type=InputType.SPEC,
            input_content="Test spec content",
        )
        assert config is not None
        assert config.input_content == "Test spec content"
