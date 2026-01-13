"""Integration tests for Phase 24: Gauntlet Consolidation.

Tests the gauntlet package imports and deprecation warnings.
"""

import pytest
import warnings


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


class TestGauntletDeprecation:
    """Test deprecation warnings for old import locations."""

    def test_modes_gauntlet_shows_deprecation_warning(self):
        """Importing from modes.gauntlet should show deprecation warning."""
        import sys
        import importlib

        # Remove from cache to allow reimport
        modules_to_remove = [k for k in sys.modules if k.startswith("aragora.modes.gauntlet")]
        for mod in modules_to_remove:
            del sys.modules[mod]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Force reimport to trigger deprecation
            import aragora.modes.gauntlet as gauntlet_mod  # noqa: F401

            importlib.reload(gauntlet_mod)

            # Check that a deprecation warning was issued
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1, f"Expected deprecation warning, got: {w}"

            # Check the message mentions the new location
            msg = str(deprecation_warnings[0].message)
            assert "aragora.gauntlet" in msg


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
