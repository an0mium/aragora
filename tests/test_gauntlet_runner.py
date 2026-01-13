"""
Tests for aragora.gauntlet.runner module.

Tests GauntletRunner execution, progress callbacks, result generation,
and vulnerability conversion methods.
"""

import asyncio
import hashlib
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.gauntlet.runner import GauntletRunner, run_gauntlet
from aragora.gauntlet.config import (
    GauntletConfig,
    AttackCategory,
    ProbeCategory,
)
from aragora.gauntlet.result import (
    GauntletResult,
    Vulnerability,
    SeverityLevel,
    AttackSummary,
    ProbeSummary,
    ScenarioSummary,
)


class TestGauntletRunnerInit:
    """Tests for GauntletRunner initialization."""

    def test_default_config(self):
        """Runner initializes with default config."""
        runner = GauntletRunner()
        assert runner.config is not None
        assert isinstance(runner.config, GauntletConfig)
        assert runner.agent_factory is None
        assert runner.run_agent_fn is None
        assert runner._vulnerability_counter == 0

    def test_custom_config(self):
        """Runner accepts custom config."""
        config = GauntletConfig(
            name="Custom Test",
            attack_categories=[AttackCategory.SECURITY],
            agents=["test-agent"],
        )
        runner = GauntletRunner(config=config)
        assert runner.config.name == "Custom Test"
        assert runner.config.attack_categories == [AttackCategory.SECURITY]
        assert runner.config.agents == ["test-agent"]

    def test_agent_factory(self):
        """Runner accepts agent factory function."""
        factory = MagicMock(return_value="agent")
        runner = GauntletRunner(agent_factory=factory)
        assert runner.agent_factory is factory

    def test_run_agent_fn(self):
        """Runner accepts run_agent function."""
        run_fn = AsyncMock(return_value="response")
        runner = GauntletRunner(run_agent_fn=run_fn)
        assert runner.run_agent_fn is run_fn


class TestGauntletRunnerRun:
    """Tests for GauntletRunner.run() method."""

    @pytest.mark.asyncio
    async def test_run_returns_gauntlet_result(self):
        """Run returns a GauntletResult with correct structure."""
        runner = GauntletRunner()
        result = await runner.run("Test input content")

        assert isinstance(result, GauntletResult)
        assert result.gauntlet_id.startswith("gauntlet-")
        assert len(result.gauntlet_id) == 21  # gauntlet- (9) + 12 hex chars

    @pytest.mark.asyncio
    async def test_run_computes_input_hash(self):
        """Run computes SHA-256 hash of input."""
        runner = GauntletRunner()
        input_content = "Test input for hashing"
        expected_hash = hashlib.sha256(input_content.encode()).hexdigest()

        result = await runner.run(input_content)

        assert result.input_hash == expected_hash

    @pytest.mark.asyncio
    async def test_run_truncates_input_summary(self):
        """Run truncates input summary to 500 chars."""
        runner = GauntletRunner()
        long_input = "A" * 1000

        result = await runner.run(long_input)

        assert len(result.input_summary) == 500
        assert result.input_summary == "A" * 500

    @pytest.mark.asyncio
    async def test_run_stores_config_used(self):
        """Run stores the config used in result."""
        config = GauntletConfig(name="Test Config")
        runner = GauntletRunner(config=config)

        result = await runner.run("Test input")

        assert result.config_used["name"] == "Test Config"
        assert result.agents_used == config.agents

    @pytest.mark.asyncio
    async def test_run_sets_timing(self):
        """Run sets started_at, completed_at, and duration."""
        runner = GauntletRunner()

        result = await runner.run("Test input")

        assert result.started_at != ""
        assert result.completed_at != ""
        assert result.duration_seconds >= 0

        # Verify timestamps are valid ISO format
        datetime.fromisoformat(result.started_at)
        datetime.fromisoformat(result.completed_at)

    @pytest.mark.asyncio
    async def test_run_progress_callback(self):
        """Run calls progress callback with phase and percent."""
        progress_updates = []

        def on_progress(phase: str, percent: float):
            progress_updates.append((phase, percent))

        # Use quick config to reduce phases
        config = GauntletConfig.quick()
        runner = GauntletRunner(config=config)
        await runner.run("Test input", on_progress=on_progress)

        # Should have progress for red_team, probes at minimum
        phase_names = [p[0] for p in progress_updates]
        assert "red_team" in phase_names
        assert "probes" in phase_names

        # Each phase should have start (0.0) and end (1.0) even if errors occur
        red_team_updates = [p for p in progress_updates if p[0] == "red_team"]
        assert any(r[1] == 0.0 for r in red_team_updates)
        assert any(r[1] == 1.0 for r in red_team_updates)

    @pytest.mark.asyncio
    async def test_run_with_context(self):
        """Run accepts context parameter."""
        runner = GauntletRunner()
        result = await runner.run("Test input", context="Additional context")

        assert isinstance(result, GauntletResult)

    @pytest.mark.asyncio
    async def test_run_skips_scenarios_when_disabled(self):
        """Run skips scenario matrix when run_scenario_matrix=False."""
        progress_updates = []

        def on_progress(phase: str, percent: float):
            progress_updates.append((phase, percent))

        config = GauntletConfig(run_scenario_matrix=False)
        runner = GauntletRunner(config=config)
        await runner.run("Test input", on_progress=on_progress)

        phase_names = [p[0] for p in progress_updates]
        assert "scenarios" not in phase_names

    @pytest.mark.asyncio
    async def test_run_calculates_verdict(self):
        """Run calculates verdict at the end."""
        runner = GauntletRunner()
        result = await runner.run("Test input")

        # Verdict should be set (PASS with no findings)
        assert result.verdict is not None
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_run_handles_exceptions_gracefully(self):
        """Run handles exceptions and sets error reasoning."""
        runner = GauntletRunner()

        # Patch _run_red_team to raise an exception
        with patch.object(runner, "_run_red_team", side_effect=Exception("Test error")):
            result = await runner.run("Test input")

        assert "Error during validation" in result.verdict_reasoning
        assert result.completed_at != ""  # Should still complete


class TestRunRedTeam:
    """Tests for GauntletRunner._run_red_team() method."""

    @pytest.mark.asyncio
    async def test_run_red_team_returns_attack_summary(self):
        """_run_red_team returns AttackSummary."""
        runner = GauntletRunner()
        result = GauntletResult(
            gauntlet_id="test",
            input_hash="abc",
            input_summary="test",
            started_at=datetime.now().isoformat(),
        )

        summary = await runner._run_red_team("Test input", "", result, lambda *a: None)

        assert isinstance(summary, AttackSummary)

    @pytest.mark.asyncio
    async def test_run_red_team_without_agents(self):
        """_run_red_team returns empty summary without agents."""
        runner = GauntletRunner()
        result = GauntletResult(
            gauntlet_id="test",
            input_hash="abc",
            input_summary="test",
            started_at=datetime.now().isoformat(),
        )

        summary = await runner._run_red_team("Test input", "", result, lambda *a: None)

        assert summary.total_attacks == 0
        assert summary.successful_attacks == 0


class TestRunProbes:
    """Tests for GauntletRunner._run_probes() method."""

    @pytest.mark.asyncio
    async def test_run_probes_returns_probe_summary(self):
        """_run_probes returns ProbeSummary."""
        runner = GauntletRunner()
        result = GauntletResult(
            gauntlet_id="test",
            input_hash="abc",
            input_summary="test",
            started_at=datetime.now().isoformat(),
        )

        summary = await runner._run_probes("Test input", "", result, lambda *a: None)

        assert isinstance(summary, ProbeSummary)

    @pytest.mark.asyncio
    async def test_run_probes_without_agent_factory(self):
        """_run_probes returns empty summary without agent factory."""
        runner = GauntletRunner()
        result = GauntletResult(
            gauntlet_id="test",
            input_hash="abc",
            input_summary="test",
            started_at=datetime.now().isoformat(),
        )

        summary = await runner._run_probes("Test input", "", result, lambda *a: None)

        # Without agent_factory, no probes can run
        assert summary.probes_run == 0
        assert summary.vulnerabilities_found == 0

    @pytest.mark.asyncio
    async def test_run_probes_with_empty_categories(self):
        """_run_probes returns empty when no probe categories configured."""
        runner = GauntletRunner()
        runner.config.probe_categories = []
        result = GauntletResult(
            gauntlet_id="test",
            input_hash="abc",
            input_summary="test",
            started_at=datetime.now().isoformat(),
        )

        summary = await runner._run_probes("Test input", "", result, lambda *a: None)

        assert summary.probes_run == 0
        assert summary.vulnerabilities_found == 0


class TestRunScenarios:
    """Tests for GauntletRunner._run_scenarios() method."""

    @pytest.mark.asyncio
    async def test_run_scenarios_returns_scenario_summary(self):
        """_run_scenarios returns ScenarioSummary."""
        runner = GauntletRunner()
        result = GauntletResult(
            gauntlet_id="test",
            input_hash="abc",
            input_summary="test",
            started_at=datetime.now().isoformat(),
        )

        summary = await runner._run_scenarios("Test input", "", result, lambda *a: None)

        assert isinstance(summary, ScenarioSummary)


class TestVulnerabilityConversion:
    """Tests for vulnerability conversion methods."""

    def test_add_attack_as_vulnerability_critical(self):
        """_add_attack_as_vulnerability converts critical severity correctly."""
        runner = GauntletRunner()
        result = GauntletResult(
            gauntlet_id="test",
            input_hash="abc",
            input_summary="test",
            started_at=datetime.now().isoformat(),
        )

        # Create mock attack with high severity
        attack = MagicMock()
        attack.severity = 0.95
        attack.attack_type = MagicMock()
        attack.attack_type.value = "security"
        attack.attack_description = "Critical security vulnerability found"
        attack.evidence = "Evidence here"
        attack.mitigation = "Fix it"
        attack.exploitability = 0.9
        attack.attacker = "test-agent"

        runner._add_attack_as_vulnerability(attack, result)

        assert len(result.vulnerabilities) == 1
        vuln = result.vulnerabilities[0]
        assert vuln.severity == SeverityLevel.CRITICAL
        assert vuln.id == "vuln-0001"
        assert vuln.source == "red_team"

    def test_add_attack_as_vulnerability_high(self):
        """_add_attack_as_vulnerability converts high severity correctly."""
        runner = GauntletRunner()
        result = GauntletResult(
            gauntlet_id="test",
            input_hash="abc",
            input_summary="test",
            started_at=datetime.now().isoformat(),
        )

        attack = MagicMock()
        attack.severity = 0.75
        attack.attack_type = MagicMock()
        attack.attack_type.value = "logic"
        attack.attack_description = "High severity issue"
        attack.evidence = ""
        attack.mitigation = None
        attack.exploitability = 0.7
        attack.attacker = "test-agent"

        runner._add_attack_as_vulnerability(attack, result)

        vuln = result.vulnerabilities[0]
        assert vuln.severity == SeverityLevel.HIGH

    def test_add_attack_as_vulnerability_medium(self):
        """_add_attack_as_vulnerability converts medium severity correctly."""
        runner = GauntletRunner()
        result = GauntletResult(
            gauntlet_id="test",
            input_hash="abc",
            input_summary="test",
            started_at=datetime.now().isoformat(),
        )

        attack = MagicMock()
        attack.severity = 0.5
        attack.attack_type = MagicMock()
        attack.attack_type.value = "edge_case"
        attack.attack_description = "Medium severity issue"
        attack.evidence = ""
        attack.mitigation = None
        attack.exploitability = 0.5
        attack.attacker = "test-agent"

        runner._add_attack_as_vulnerability(attack, result)

        vuln = result.vulnerabilities[0]
        assert vuln.severity == SeverityLevel.MEDIUM

    def test_add_attack_as_vulnerability_low(self):
        """_add_attack_as_vulnerability converts low severity correctly."""
        runner = GauntletRunner()
        result = GauntletResult(
            gauntlet_id="test",
            input_hash="abc",
            input_summary="test",
            started_at=datetime.now().isoformat(),
        )

        attack = MagicMock()
        attack.severity = 0.2
        attack.attack_type = MagicMock()
        attack.attack_type.value = "info"
        attack.attack_description = "Low severity issue"
        attack.evidence = ""
        attack.mitigation = None
        attack.exploitability = 0.2
        attack.attacker = "test-agent"

        runner._add_attack_as_vulnerability(attack, result)

        vuln = result.vulnerabilities[0]
        assert vuln.severity == SeverityLevel.LOW

    def test_vulnerability_counter_increments(self):
        """Vulnerability counter increments with each addition."""
        runner = GauntletRunner()
        result = GauntletResult(
            gauntlet_id="test",
            input_hash="abc",
            input_summary="test",
            started_at=datetime.now().isoformat(),
        )

        for i in range(3):
            attack = MagicMock()
            attack.severity = 0.5
            attack.attack_type = MagicMock()
            attack.attack_type.value = "test"
            attack.attack_description = f"Issue {i}"
            attack.evidence = ""
            attack.mitigation = None
            attack.exploitability = 0.5
            attack.attacker = "test-agent"
            runner._add_attack_as_vulnerability(attack, result)

        assert len(result.vulnerabilities) == 3
        assert result.vulnerabilities[0].id == "vuln-0001"
        assert result.vulnerabilities[1].id == "vuln-0002"
        assert result.vulnerabilities[2].id == "vuln-0003"

    def test_add_probe_as_vulnerability(self):
        """_add_probe_as_vulnerability converts probe result correctly."""
        runner = GauntletRunner()
        result = GauntletResult(
            gauntlet_id="test",
            input_hash="abc",
            input_summary="test",
            started_at=datetime.now().isoformat(),
        )

        probe_result = MagicMock()
        probe_result.probe_type = MagicMock()
        probe_result.probe_type.value = "hallucination"
        probe_result.vulnerability_description = "Agent hallucinated facts"
        probe_result.severity = MagicMock()
        probe_result.severity.value = "high"
        probe_result.evidence = "Factual error detected"

        runner._add_probe_as_vulnerability(probe_result, "test-agent", result)

        assert len(result.vulnerabilities) == 1
        vuln = result.vulnerabilities[0]
        assert vuln.severity == SeverityLevel.HIGH
        assert vuln.source == "capability_probe"
        assert vuln.agent_name == "test-agent"


class TestDefaultRunAgent:
    """Tests for GauntletRunner._default_run_agent() method."""

    @pytest.mark.asyncio
    async def test_default_run_agent_calls_run_method(self):
        """_default_run_agent calls agent.run() if available."""
        runner = GauntletRunner()

        agent = MagicMock()
        agent.run = AsyncMock(return_value="Agent response")

        response = await runner._default_run_agent(agent, "Test prompt")

        assert response == "Agent response"
        agent.run.assert_called_once_with("Test prompt")

    @pytest.mark.asyncio
    async def test_default_run_agent_without_run_method(self):
        """_default_run_agent returns placeholder without run method."""
        runner = GauntletRunner()

        agent = "not-callable"

        response = await runner._default_run_agent(agent, "Test prompt")

        assert "No response" in response


class TestRunGauntletConvenience:
    """Tests for run_gauntlet() convenience function."""

    @pytest.mark.asyncio
    async def test_run_gauntlet_basic(self):
        """run_gauntlet creates runner and runs validation."""
        result = await run_gauntlet("Test input content")

        assert isinstance(result, GauntletResult)
        assert result.gauntlet_id.startswith("gauntlet-")

    @pytest.mark.asyncio
    async def test_run_gauntlet_with_config(self):
        """run_gauntlet accepts custom config."""
        config = GauntletConfig(name="Custom")
        result = await run_gauntlet("Test input", config=config)

        assert result.config_used["name"] == "Custom"

    @pytest.mark.asyncio
    async def test_run_gauntlet_with_context(self):
        """run_gauntlet accepts context parameter."""
        result = await run_gauntlet(
            "Test input",
            context="Additional context for validation",
        )

        assert isinstance(result, GauntletResult)


class TestIntegration:
    """Integration tests for complete gauntlet runs."""

    @pytest.mark.asyncio
    async def test_full_run_quick_config(self):
        """Full run with quick config completes successfully."""
        config = GauntletConfig.quick()
        runner = GauntletRunner(config=config)

        result = await runner.run(
            "System specification for review",
            context="This is a test system",
        )

        assert isinstance(result, GauntletResult)
        assert result.completed_at != ""
        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_full_run_security_focused(self):
        """Full run with security-focused config completes successfully."""
        config = GauntletConfig.security_focused()
        runner = GauntletRunner(config=config)

        result = await runner.run("Security-critical system spec")

        assert isinstance(result, GauntletResult)
        assert AttackCategory.SECURITY in [
            AttackCategory(c) for c in result.config_used["attack_categories"]
        ]

    @pytest.mark.asyncio
    async def test_full_run_compliance_focused(self):
        """Full run with compliance-focused config completes successfully."""
        config = GauntletConfig.compliance_focused()
        runner = GauntletRunner(config=config)

        result = await runner.run("Policy document for compliance review")

        assert isinstance(result, GauntletResult)
        # Compliance config disables scenarios
        assert result.config_used["run_scenario_matrix"] is False
