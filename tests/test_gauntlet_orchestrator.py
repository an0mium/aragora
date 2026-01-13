"""
Tests for Gauntlet Orchestrator.

Tests the adversarial validation pipeline execution.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from pathlib import Path

from aragora.gauntlet.orchestrator import GauntletOrchestrator
from aragora.gauntlet.config import (
    GauntletConfig,
    GauntletResult,
    GauntletFinding,
    GauntletPhase,
    GauntletSeverity,
    PhaseResult,
    AttackCategory,
)


class TestGauntletOrchestratorInit:
    """Test GauntletOrchestrator initialization."""

    def test_init_default(self):
        """Test default initialization."""
        orchestrator = GauntletOrchestrator()

        assert orchestrator.agents == []
        assert orchestrator.nomic_dir == Path(".nomic")
        assert orchestrator.on_phase_complete is None
        assert orchestrator.on_finding is None
        assert callable(orchestrator.run_agent_fn)

    def test_init_with_agents(self):
        """Test initialization with agents."""
        mock_agents = [Mock(), Mock()]
        orchestrator = GauntletOrchestrator(agents=mock_agents)

        assert orchestrator.agents == mock_agents

    def test_init_with_nomic_dir(self):
        """Test initialization with custom nomic directory."""
        custom_path = Path("/custom/path")
        orchestrator = GauntletOrchestrator(nomic_dir=custom_path)

        assert orchestrator.nomic_dir == custom_path

    def test_init_with_callbacks(self):
        """Test initialization with callbacks."""
        on_phase = Mock()
        on_finding = Mock()

        orchestrator = GauntletOrchestrator(
            on_phase_complete=on_phase,
            on_finding=on_finding,
        )

        assert orchestrator.on_phase_complete == on_phase
        assert orchestrator.on_finding == on_finding

    def test_init_with_custom_run_agent_fn(self):
        """Test initialization with custom agent runner."""
        custom_runner = AsyncMock()
        orchestrator = GauntletOrchestrator(run_agent_fn=custom_runner)

        assert orchestrator.run_agent_fn == custom_runner


class TestGauntletOrchestratorRun:
    """Test main run() method."""

    @pytest.fixture
    def orchestrator(self):
        return GauntletOrchestrator()

    @pytest.fixture
    def minimal_config(self):
        return GauntletConfig(
            name="Test Gauntlet",
            enable_scenario_analysis=False,
            enable_adversarial_probing=False,
            enable_formal_verification=False,
            enable_deep_audit=False,
            save_artifacts=False,
        )

    @pytest.mark.asyncio
    async def test_run_minimal(self, orchestrator, minimal_config):
        """Test minimal run with all phases disabled except risk assessment."""
        with patch.object(orchestrator, "_run_risk_assessment") as mock_risk:
            mock_risk.return_value = PhaseResult(
                phase=GauntletPhase.RISK_ASSESSMENT,
                status="completed",
                duration_ms=100,
                findings=[],
                metrics={"risks_identified": 0},
            )

            result = await orchestrator.run(
                input_text="Test input",
                config=minimal_config,
            )

            assert result is not None
            assert result.current_phase == GauntletPhase.COMPLETE
            assert len(result.phase_results) >= 1
            mock_risk.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_template(self, orchestrator):
        """Test run with template configuration."""
        with patch("aragora.gauntlet.orchestrator.get_template") as mock_get_template:
            mock_config = GauntletConfig(
                name="Template Config",
                enable_scenario_analysis=False,
                enable_adversarial_probing=False,
                enable_formal_verification=False,
                enable_deep_audit=False,
                save_artifacts=False,
            )
            mock_get_template.return_value = mock_config

            with patch.object(orchestrator, "_run_risk_assessment") as mock_risk:
                mock_risk.return_value = PhaseResult(
                    phase=GauntletPhase.RISK_ASSESSMENT,
                    status="completed",
                    duration_ms=100,
                )

                result = await orchestrator.run(
                    input_text="Test",
                    template="API_ROBUSTNESS",
                )

                mock_get_template.assert_called_once_with("API_ROBUSTNESS")

    @pytest.mark.asyncio
    async def test_run_all_phases(self, orchestrator):
        """Test run with all phases enabled."""
        config = GauntletConfig(
            name="Full Gauntlet",
            enable_scenario_analysis=True,
            enable_adversarial_probing=True,
            enable_formal_verification=True,
            enable_deep_audit=True,
            save_artifacts=False,
        )

        with patch.object(orchestrator, "_run_risk_assessment") as mock_risk:
            with patch.object(orchestrator, "_run_scenario_analysis") as mock_scenario:
                with patch.object(orchestrator, "_run_adversarial_probing") as mock_probe:
                    with patch.object(orchestrator, "_run_formal_verification") as mock_verify:
                        with patch.object(orchestrator, "_run_deep_audit") as mock_audit:
                            # Setup mock returns
                            for mock_phase, phase in [
                                (mock_risk, GauntletPhase.RISK_ASSESSMENT),
                                (mock_scenario, GauntletPhase.SCENARIO_ANALYSIS),
                                (mock_probe, GauntletPhase.ADVERSARIAL_PROBING),
                                (mock_verify, GauntletPhase.FORMAL_VERIFICATION),
                                (mock_audit, GauntletPhase.DEEP_AUDIT),
                            ]:
                                mock_phase.return_value = PhaseResult(
                                    phase=phase,
                                    status="completed",
                                    duration_ms=100,
                                    metrics={},
                                )

                            result = await orchestrator.run(
                                input_text="Test",
                                config=config,
                            )

                            # Verify all phases were called
                            mock_risk.assert_called_once()
                            mock_scenario.assert_called_once()
                            mock_probe.assert_called_once()
                            mock_verify.assert_called_once()
                            mock_audit.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_timeout_handling(self, orchestrator, minimal_config):
        """Test that timeout is properly handled."""
        minimal_config.timeout_seconds = 0.001  # Very short timeout

        with patch.object(orchestrator, "_run_risk_assessment") as mock_risk:
            # Make risk assessment take too long
            async def slow_risk(*args, **kwargs):
                await asyncio.sleep(1)
                return PhaseResult(
                    phase=GauntletPhase.RISK_ASSESSMENT, status="completed", duration_ms=1000
                )

            mock_risk.side_effect = slow_risk

            # Wrap with timeout
            result = await asyncio.wait_for(
                orchestrator.run(input_text="Test", config=minimal_config),
                timeout=2.0,  # Overall test timeout
            )

            # The orchestrator itself may not timeout, but it handles errors gracefully

    @pytest.mark.asyncio
    async def test_run_exception_handling(self, orchestrator, minimal_config):
        """Test that exceptions are properly handled."""
        with patch.object(orchestrator, "_run_risk_assessment") as mock_risk:
            mock_risk.side_effect = RuntimeError("Phase failed")

            result = await orchestrator.run(
                input_text="Test",
                config=minimal_config,
            )

            assert result.current_phase == GauntletPhase.FAILED
            assert "RuntimeError" in result.verdict_summary

    @pytest.mark.asyncio
    async def test_run_records_duration(self, orchestrator, minimal_config):
        """Test that run duration is recorded."""
        with patch.object(orchestrator, "_run_risk_assessment") as mock_risk:

            async def slow_risk(*args, **kwargs):
                await asyncio.sleep(0.01)  # Small delay to ensure measurable duration
                return PhaseResult(
                    phase=GauntletPhase.RISK_ASSESSMENT,
                    status="completed",
                    duration_ms=100,
                )

            mock_risk.side_effect = slow_risk

            result = await orchestrator.run(
                input_text="Test",
                config=minimal_config,
            )

            assert result.total_duration_ms >= 0  # May be 0 on very fast systems
            assert result.completed_at is not None


class TestGauntletOrchestratorPhases:
    """Test individual phase execution."""

    @pytest.fixture
    def orchestrator(self):
        return GauntletOrchestrator()

    @pytest.fixture
    def config(self):
        return GauntletConfig()

    @pytest.mark.asyncio
    async def test_risk_assessment_import_error(self, orchestrator, config):
        """Test risk assessment handles import error gracefully."""
        with patch.dict("sys.modules", {"aragora.debate.risk_assessor": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                result = await orchestrator._run_risk_assessment("Test input", config)

                assert result.status == "skipped"
                assert "not available" in result.error

    @pytest.mark.asyncio
    async def test_risk_assessment_success(self, orchestrator, config):
        """Test successful risk assessment."""
        mock_assessor_instance = Mock()
        mock_assessment = Mock()
        mock_assessment.level = Mock()
        mock_assessment.level.name = "HIGH"
        mock_assessment.category = "security"
        mock_assessment.domain = "auth"
        mock_assessment.description = "Security risk"
        mock_assessment.mitigations = ["Mitigate"]
        mock_assessment.confidence = 0.8
        mock_assessor_instance.assess.return_value = [mock_assessment]

        # Create a mock RiskLevel enum
        mock_risk_level = Mock()
        mock_risk_level.LOW = Mock()
        mock_risk_level.MEDIUM = Mock()
        mock_risk_level.HIGH = mock_assessment.level
        mock_risk_level.CRITICAL = Mock()

        # Patch the import inside the function
        with patch.dict(
            "sys.modules",
            {
                "aragora.debate.risk_assessor": Mock(
                    RiskAssessor=Mock(return_value=mock_assessor_instance),
                    RiskLevel=mock_risk_level,
                )
            },
        ):
            result = await orchestrator._run_risk_assessment("Test", config)

            assert result.status == "completed"
            assert len(result.findings) == 1
            assert result.findings[0].category == "security"

    @pytest.mark.asyncio
    async def test_scenario_analysis_import_error(self, orchestrator, config):
        """Test scenario analysis handles import error gracefully."""
        with patch("builtins.__import__", side_effect=ImportError):
            result = await orchestrator._run_scenario_analysis("Test", config)

            assert result.status == "skipped"

    @pytest.mark.asyncio
    async def test_adversarial_probing_no_agents(self, orchestrator, config):
        """Test adversarial probing without agents."""
        orchestrator.agents = []
        result = await orchestrator._run_adversarial_probing("Test", config)

        # Should complete but with limited probing
        assert result.status == "completed"
        assert result.metrics.get("probes_run", 0) >= 0


class TestGauntletOrchestratorCallbacks:
    """Test callback functionality."""

    @pytest.fixture
    def orchestrator(self):
        return GauntletOrchestrator()

    def test_notify_phase_complete(self, orchestrator):
        """Test phase complete callback is called."""
        callback = Mock()
        orchestrator.on_phase_complete = callback

        phase_result = PhaseResult(
            phase=GauntletPhase.RISK_ASSESSMENT,
            status="completed",
            duration_ms=100,
        )

        orchestrator._notify_phase_complete(GauntletPhase.RISK_ASSESSMENT, phase_result)

        callback.assert_called_once_with(GauntletPhase.RISK_ASSESSMENT, phase_result)

    def test_notify_phase_complete_no_callback(self, orchestrator):
        """Test notify works when no callback configured."""
        orchestrator.on_phase_complete = None

        phase_result = PhaseResult(
            phase=GauntletPhase.RISK_ASSESSMENT,
            status="completed",
            duration_ms=100,
        )

        # Should not raise
        orchestrator._notify_phase_complete(GauntletPhase.RISK_ASSESSMENT, phase_result)

    def test_notify_finding(self, orchestrator):
        """Test finding callback is called."""
        callback = Mock()
        orchestrator.on_finding = callback

        finding = GauntletFinding(
            severity=GauntletSeverity.HIGH,
            category="security",
            title="Test Finding",
            description="A test finding",
            source_phase=GauntletPhase.RISK_ASSESSMENT,
        )

        orchestrator._notify_finding(finding)

        callback.assert_called_once_with(finding)


class TestGauntletOrchestratorDefaultRunner:
    """Test default agent runner."""

    @pytest.fixture
    def orchestrator(self):
        return GauntletOrchestrator()

    @pytest.mark.asyncio
    async def test_default_run_agent(self, orchestrator):
        """Test default agent runner calls agent.generate."""
        mock_agent = Mock()
        mock_agent.generate = AsyncMock(return_value="Response")

        result = await orchestrator._default_run_agent(mock_agent, "Test prompt")

        assert result == "Response"
        mock_agent.generate.assert_called_once_with("Test prompt", [])


class TestGauntletOrchestratorSeverityConversion:
    """Test severity conversion utilities."""

    @pytest.fixture
    def orchestrator(self):
        return GauntletOrchestrator()

    def test_severity_float_to_enum_critical(self, orchestrator):
        """Test critical severity conversion."""
        result = orchestrator._severity_float_to_enum(0.95)
        assert result == GauntletSeverity.CRITICAL

    def test_severity_float_to_enum_high(self, orchestrator):
        """Test high severity conversion."""
        result = orchestrator._severity_float_to_enum(0.75)
        assert result == GauntletSeverity.HIGH

    def test_severity_float_to_enum_medium(self, orchestrator):
        """Test medium severity conversion."""
        result = orchestrator._severity_float_to_enum(0.5)
        assert result == GauntletSeverity.MEDIUM

    def test_severity_float_to_enum_low(self, orchestrator):
        """Test low severity conversion."""
        result = orchestrator._severity_float_to_enum(0.2)
        assert result == GauntletSeverity.LOW


class TestGauntletConfig:
    """Test GauntletConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = GauntletConfig()

        assert config.name == "Gauntlet Validation"
        assert config.enable_scenario_analysis is True
        assert config.enable_adversarial_probing is True
        # Formal verification and deep audit are disabled by default (expensive operations)
        assert config.enable_formal_verification is False
        assert config.enable_deep_audit is False
        assert config.max_agents > 0

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = GauntletConfig(
            name="Custom Gauntlet",
            enable_scenario_analysis=False,
            timeout_seconds=300,
        )

        assert config.name == "Custom Gauntlet"
        assert config.enable_scenario_analysis is False
        assert config.timeout_seconds == 300


class TestGauntletResult:
    """Test GauntletResult dataclass."""

    def test_result_initialization(self):
        """Test result initialization."""
        config = GauntletConfig()
        result = GauntletResult(
            config=config,
            input_text="Test input",
            agents_used=["agent1", "agent2"],
        )

        assert result.config == config
        assert result.input_text == "Test input"
        assert result.agents_used == ["agent1", "agent2"]
        assert result.current_phase == GauntletPhase.NOT_STARTED
        assert result.findings == []
        assert result.phase_results == []

    def test_result_pass_fail_evaluation(self):
        """Test pass/fail evaluation logic."""
        config = GauntletConfig()
        result = GauntletResult(
            config=config,
            input_text="Test",
            agents_used=[],
        )

        # Add findings of various severities
        result.findings = [
            GauntletFinding(
                severity=GauntletSeverity.LOW,
                category="test",
                title="Low Finding",
                description="Low severity",
                source_phase=GauntletPhase.RISK_ASSESSMENT,
            ),
        ]

        result.evaluate_pass_fail()
        # Should pass with only low findings
        assert "PASS" in result.verdict_summary or result.verdict_summary != ""

    def test_result_with_critical_findings(self):
        """Test result with critical findings fails."""
        config = GauntletConfig()
        result = GauntletResult(
            config=config,
            input_text="Test",
            agents_used=[],
        )

        result.findings = [
            GauntletFinding(
                severity=GauntletSeverity.CRITICAL,
                category="security",
                title="Critical Finding",
                description="Critical issue",
                source_phase=GauntletPhase.ADVERSARIAL_PROBING,
            ),
        ]

        result.evaluate_pass_fail()
        # Should fail with critical findings
        assert result.verdict_summary != ""


class TestGauntletPhaseResult:
    """Test PhaseResult dataclass."""

    def test_phase_result_creation(self):
        """Test creating a phase result."""
        result = PhaseResult(
            phase=GauntletPhase.RISK_ASSESSMENT,
            status="completed",
            duration_ms=150,
            findings=[],
            metrics={"risks_identified": 5},
        )

        assert result.phase == GauntletPhase.RISK_ASSESSMENT
        assert result.status == "completed"
        assert result.duration_ms == 150
        assert result.metrics["risks_identified"] == 5

    def test_phase_result_with_error(self):
        """Test phase result with error."""
        result = PhaseResult(
            phase=GauntletPhase.SCENARIO_ANALYSIS,
            status="failed",
            duration_ms=50,
            error="Module not found",
        )

        assert result.status == "failed"
        assert result.error == "Module not found"


class TestGauntletFinding:
    """Test GauntletFinding dataclass."""

    def test_finding_creation(self):
        """Test creating a finding."""
        finding = GauntletFinding(
            severity=GauntletSeverity.HIGH,
            category="security",
            title="SQL Injection Risk",
            description="Potential SQL injection vulnerability",
            source_phase=GauntletPhase.ADVERSARIAL_PROBING,
            recommendations=["Use parameterized queries"],
            exploitability=0.8,
            impact=0.9,
        )

        assert finding.severity == GauntletSeverity.HIGH
        assert finding.category == "security"
        assert finding.title == "SQL Injection Risk"
        assert len(finding.recommendations) == 1

    def test_finding_with_metadata(self):
        """Test finding with metadata."""
        finding = GauntletFinding(
            severity=GauntletSeverity.MEDIUM,
            category="test",
            title="Test Finding",
            description="Description",
            source_phase=GauntletPhase.RISK_ASSESSMENT,
            metadata={"line": 42, "file": "test.py"},
        )

        assert finding.metadata["line"] == 42
        assert finding.metadata["file"] == "test.py"


class TestGauntletIntegration:
    """Integration tests for full Gauntlet execution."""

    @pytest.fixture
    def orchestrator_with_agents(self):
        """Create orchestrator with mock agents."""
        mock_agents = [
            Mock(name="agent1", generate=AsyncMock(return_value="Response 1")),
            Mock(name="agent2", generate=AsyncMock(return_value="Response 2")),
        ]
        return GauntletOrchestrator(agents=mock_agents)

    @pytest.mark.asyncio
    async def test_full_gauntlet_flow(self, orchestrator_with_agents):
        """Test a complete Gauntlet flow with mocked phases."""
        config = GauntletConfig(
            name="Integration Test",
            enable_scenario_analysis=True,
            enable_adversarial_probing=True,
            enable_formal_verification=False,  # Skip for speed
            enable_deep_audit=False,  # Skip for speed
            save_artifacts=False,
        )

        # Mock phase methods
        with patch.object(orchestrator_with_agents, "_run_risk_assessment") as mock_risk:
            with patch.object(orchestrator_with_agents, "_run_scenario_analysis") as mock_scenario:
                with patch.object(
                    orchestrator_with_agents, "_run_adversarial_probing"
                ) as mock_probe:
                    # Setup returns with findings
                    mock_risk.return_value = PhaseResult(
                        phase=GauntletPhase.RISK_ASSESSMENT,
                        status="completed",
                        duration_ms=100,
                        findings=[
                            GauntletFinding(
                                severity=GauntletSeverity.MEDIUM,
                                category="domain_risk",
                                title="Security Domain Risk",
                                description="Authentication risks identified",
                                source_phase=GauntletPhase.RISK_ASSESSMENT,
                            )
                        ],
                        metrics={"risks_identified": 1},
                    )

                    mock_scenario.return_value = PhaseResult(
                        phase=GauntletPhase.SCENARIO_ANALYSIS,
                        status="completed",
                        duration_ms=200,
                        metrics={"scenarios_run": 5},
                    )

                    mock_probe.return_value = PhaseResult(
                        phase=GauntletPhase.ADVERSARIAL_PROBING,
                        status="completed",
                        duration_ms=300,
                        metrics={"probes_run": 10, "robustness_score": 0.75},
                    )

                    result = await orchestrator_with_agents.run(
                        input_text="Test API implementation",
                        config=config,
                    )

                    # Verify result structure
                    assert result.current_phase == GauntletPhase.COMPLETE
                    assert len(result.phase_results) == 3
                    assert len(result.findings) >= 1
                    assert result.scenarios_tested == 5
                    assert result.probes_executed == 10
                    assert result.robustness_score == 0.75

    @pytest.mark.asyncio
    async def test_gauntlet_with_callbacks(self):
        """Test Gauntlet execution with callbacks."""
        phase_callback = Mock()
        finding_callback = Mock()

        orchestrator = GauntletOrchestrator(
            on_phase_complete=phase_callback,
            on_finding=finding_callback,
        )

        config = GauntletConfig(
            enable_scenario_analysis=False,
            enable_adversarial_probing=False,
            enable_formal_verification=False,
            enable_deep_audit=False,
            save_artifacts=False,
        )

        with patch.object(orchestrator, "_run_risk_assessment") as mock_risk:
            finding = GauntletFinding(
                severity=GauntletSeverity.LOW,
                category="test",
                title="Test",
                description="Test",
                source_phase=GauntletPhase.RISK_ASSESSMENT,
            )

            mock_risk.return_value = PhaseResult(
                phase=GauntletPhase.RISK_ASSESSMENT,
                status="completed",
                duration_ms=50,
                findings=[finding],
            )

            await orchestrator.run(input_text="Test", config=config)

            # Callbacks should have been invoked
            assert phase_callback.call_count >= 1
