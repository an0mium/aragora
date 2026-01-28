"""
Tests for Gauntlet Orchestrator.

Tests the orchestrator module including:
- GauntletOrchestrator initialization
- Phase execution
- Callback handling
- Result synthesis
- Template handling
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.gauntlet.config import (
    GauntletConfig,
    GauntletPhase,
    GauntletResult,
    GauntletSeverity,
    PhaseResult,
)
from aragora.gauntlet.orchestrator import (
    GauntletOrchestrator,
    run_gauntlet,
)
from aragora.gauntlet.templates import GauntletTemplate


# =============================================================================
# GauntletOrchestrator Initialization Tests
# =============================================================================


class TestGauntletOrchestratorInit:
    """Test GauntletOrchestrator initialization."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        orchestrator = GauntletOrchestrator()

        assert orchestrator.agents == []
        assert orchestrator.nomic_dir == Path(".nomic")
        assert orchestrator.on_phase_complete is None
        assert orchestrator.on_finding is None

    def test_init_with_agents(self):
        """Test initialization with agents."""
        mock_agents = [MagicMock(name="agent1"), MagicMock(name="agent2")]
        orchestrator = GauntletOrchestrator(agents=mock_agents)

        assert len(orchestrator.agents) == 2

    def test_init_with_nomic_dir(self):
        """Test initialization with custom nomic directory."""
        orchestrator = GauntletOrchestrator(nomic_dir=Path("/custom/dir"))

        assert orchestrator.nomic_dir == Path("/custom/dir")

    def test_init_with_callbacks(self):
        """Test initialization with callbacks."""
        phase_callback = MagicMock()
        finding_callback = MagicMock()

        orchestrator = GauntletOrchestrator(
            on_phase_complete=phase_callback,
            on_finding=finding_callback,
        )

        assert orchestrator.on_phase_complete == phase_callback
        assert orchestrator.on_finding == finding_callback

    def test_init_with_custom_run_agent_fn(self):
        """Test initialization with custom agent runner."""
        custom_runner = AsyncMock(return_value="test response")
        orchestrator = GauntletOrchestrator(run_agent_fn=custom_runner)

        assert orchestrator.run_agent_fn == custom_runner


# =============================================================================
# GauntletOrchestrator Run Tests
# =============================================================================


class TestGauntletOrchestratorRun:
    """Test GauntletOrchestrator run method."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing."""
        return GauntletOrchestrator()

    @pytest.fixture
    def minimal_config(self):
        """Create minimal config for fast tests."""
        return GauntletConfig(
            name="Test Gauntlet",
            enable_scenario_analysis=False,
            enable_adversarial_probing=False,
            enable_formal_verification=False,
            enable_deep_audit=False,
            timeout_seconds=30,
        )

    @pytest.mark.asyncio
    async def test_run_with_minimal_config(self, orchestrator, minimal_config):
        """Test run with minimal configuration."""
        result = await orchestrator.run(
            input_text="Test input",
            config=minimal_config,
        )

        assert isinstance(result, GauntletResult)
        assert result.input_text == "Test input"
        assert result.current_phase in (GauntletPhase.COMPLETE, GauntletPhase.FAILED)

    @pytest.mark.asyncio
    async def test_run_tracks_timing(self, orchestrator, minimal_config):
        """Test run tracks execution timing."""
        result = await orchestrator.run(
            input_text="Test input",
            config=minimal_config,
        )

        assert result.total_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_run_uses_config_agents(self, orchestrator):
        """Test run uses agents from config."""
        config = GauntletConfig(
            agents=["agent-1", "agent-2", "agent-3"],
            max_agents=2,
            enable_scenario_analysis=False,
            enable_adversarial_probing=False,
            enable_formal_verification=False,
            enable_deep_audit=False,
        )

        result = await orchestrator.run(
            input_text="Test",
            config=config,
        )

        assert len(result.agents_used) == 2

    @pytest.mark.asyncio
    async def test_run_with_template(self, orchestrator):
        """Test run with template."""
        result = await orchestrator.run(
            input_text="Test",
            template=GauntletTemplate.QUICK_SANITY,
        )

        assert isinstance(result, GauntletResult)


# =============================================================================
# Phase Callback Tests
# =============================================================================


class TestOrchestratorCallbacks:
    """Test orchestrator callback handling."""

    @pytest.mark.asyncio
    async def test_phase_complete_callback_called(self):
        """Test phase complete callback is called."""
        callback = MagicMock()
        orchestrator = GauntletOrchestrator(on_phase_complete=callback)

        config = GauntletConfig(
            enable_scenario_analysis=False,
            enable_adversarial_probing=False,
            enable_formal_verification=False,
            enable_deep_audit=False,
        )

        await orchestrator.run(input_text="Test", config=config)

        # Should be called at least once (for risk assessment)
        assert callback.call_count >= 1

    @pytest.mark.asyncio
    async def test_phase_callback_receives_correct_args(self):
        """Test phase callback receives correct arguments."""
        callback = MagicMock()
        orchestrator = GauntletOrchestrator(on_phase_complete=callback)

        config = GauntletConfig(
            enable_scenario_analysis=False,
            enable_adversarial_probing=False,
            enable_formal_verification=False,
            enable_deep_audit=False,
        )

        await orchestrator.run(input_text="Test", config=config)

        # Check callback was called with phase and result
        if callback.call_count > 0:
            args = callback.call_args[0]
            assert isinstance(args[0], GauntletPhase)
            assert isinstance(args[1], PhaseResult)

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_fail_run(self):
        """Test callback exception doesn't crash the run."""
        callback = MagicMock(side_effect=Exception("Callback error"))
        orchestrator = GauntletOrchestrator(on_phase_complete=callback)

        config = GauntletConfig(
            enable_scenario_analysis=False,
            enable_adversarial_probing=False,
            enable_formal_verification=False,
            enable_deep_audit=False,
        )

        # Should not raise
        result = await orchestrator.run(input_text="Test", config=config)
        assert result is not None


# =============================================================================
# Result Synthesis Tests
# =============================================================================


class TestResultSynthesis:
    """Test result synthesis."""

    @pytest.mark.asyncio
    async def test_evaluate_pass_fail(self):
        """Test pass/fail evaluation."""
        orchestrator = GauntletOrchestrator()
        config = GauntletConfig(
            enable_scenario_analysis=False,
            enable_adversarial_probing=False,
            enable_formal_verification=False,
            enable_deep_audit=False,
        )

        result = await orchestrator.run(input_text="Test", config=config)

        # Result should have pass/fail evaluated
        assert result.passed is not None or result.current_phase == GauntletPhase.FAILED

    @pytest.mark.asyncio
    async def test_confidence_calculated(self):
        """Test confidence is calculated."""
        orchestrator = GauntletOrchestrator()
        config = GauntletConfig(
            enable_scenario_analysis=False,
            enable_adversarial_probing=False,
            enable_formal_verification=False,
            enable_deep_audit=False,
        )

        result = await orchestrator.run(input_text="Test", config=config)

        # Confidence should be between 0 and 1
        assert 0 <= result.confidence <= 1


# =============================================================================
# Severity Conversion Tests
# =============================================================================


class TestSeverityConversion:
    """Test severity float to enum conversion."""

    def test_severity_critical(self):
        """Test critical severity conversion."""
        orchestrator = GauntletOrchestrator()
        assert orchestrator._severity_float_to_enum(0.95) == GauntletSeverity.CRITICAL
        assert orchestrator._severity_float_to_enum(0.9) == GauntletSeverity.CRITICAL

    def test_severity_high(self):
        """Test high severity conversion."""
        orchestrator = GauntletOrchestrator()
        assert orchestrator._severity_float_to_enum(0.8) == GauntletSeverity.HIGH
        assert orchestrator._severity_float_to_enum(0.7) == GauntletSeverity.HIGH

    def test_severity_medium(self):
        """Test medium severity conversion."""
        orchestrator = GauntletOrchestrator()
        assert orchestrator._severity_float_to_enum(0.5) == GauntletSeverity.MEDIUM
        assert orchestrator._severity_float_to_enum(0.4) == GauntletSeverity.MEDIUM

    def test_severity_low(self):
        """Test low severity conversion."""
        orchestrator = GauntletOrchestrator()
        assert orchestrator._severity_float_to_enum(0.2) == GauntletSeverity.LOW
        assert orchestrator._severity_float_to_enum(0.1) == GauntletSeverity.LOW

    def test_severity_info(self):
        """Test info severity conversion."""
        orchestrator = GauntletOrchestrator()
        assert orchestrator._severity_float_to_enum(0.0) == GauntletSeverity.INFO


# =============================================================================
# Claim Extraction Tests
# =============================================================================


class TestClaimExtraction:
    """Test claim extraction from text."""

    def test_extract_claims_with_must(self):
        """Test extracting claims with 'must'."""
        orchestrator = GauntletOrchestrator()
        text = "The system must validate all inputs. Users can enter anything."

        claims = orchestrator._extract_claims(text)

        assert len(claims) >= 1
        assert any("must" in c.lower() for c in claims)

    def test_extract_claims_with_always(self):
        """Test extracting claims with 'always'."""
        orchestrator = GauntletOrchestrator()
        text = "The cache is always consistent. Data may vary."

        claims = orchestrator._extract_claims(text)

        assert len(claims) >= 1
        assert any("always" in c.lower() for c in claims)

    def test_extract_claims_with_never(self):
        """Test extracting claims with 'never'."""
        orchestrator = GauntletOrchestrator()
        text = "The system never exposes credentials. Sometimes it logs events."

        claims = orchestrator._extract_claims(text)

        assert any("never" in c.lower() for c in claims)

    def test_extract_claims_limits_results(self):
        """Test claim extraction limits results."""
        orchestrator = GauntletOrchestrator()
        text = ". ".join([f"Statement {i} must be true" for i in range(20)])

        claims = orchestrator._extract_claims(text)

        assert len(claims) <= 10  # Should be limited

    def test_extract_claims_filters_short(self):
        """Test claim extraction filters short sentences."""
        orchestrator = GauntletOrchestrator()
        text = "Must be. This is a longer statement that must be evaluated properly."

        claims = orchestrator._extract_claims(text)

        # Short "Must be." should be filtered
        assert all(len(c) > 20 for c in claims)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestRunGauntletConvenience:
    """Test run_gauntlet convenience function."""

    @pytest.mark.asyncio
    async def test_run_gauntlet_with_defaults(self):
        """Test run_gauntlet with default template."""
        result = await run_gauntlet("Test input text")

        assert isinstance(result, GauntletResult)
        assert result.input_text == "Test input text"

    @pytest.mark.asyncio
    async def test_run_gauntlet_with_template(self):
        """Test run_gauntlet with specific template."""
        result = await run_gauntlet(
            "Test input",
            template=GauntletTemplate.QUICK_SANITY,
        )

        assert isinstance(result, GauntletResult)
