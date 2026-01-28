"""
Tests for Gauntlet Runner.

Tests the runner module including:
- GauntletRunner initialization
- Red team execution
- Probe execution
- Scenario execution
- Vulnerability tracking
- Code extraction for sandbox
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.gauntlet.config import AttackCategory, GauntletConfig, ProbeCategory
from aragora.gauntlet.result import GauntletResult, SeverityLevel, Verdict, Vulnerability
from aragora.gauntlet.runner import GauntletRunner, run_gauntlet


# =============================================================================
# GauntletRunner Initialization Tests
# =============================================================================


class TestGauntletRunnerInit:
    """Test GauntletRunner initialization."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        runner = GauntletRunner()

        assert runner.config is not None
        assert runner.agent_factory is None
        assert runner.run_agent_fn is None
        assert runner._vulnerability_counter == 0

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = GauntletConfig(
            name="Custom Gauntlet",
            attack_rounds=5,
            agents=["agent-1", "agent-2"],
        )
        runner = GauntletRunner(config=config)

        assert runner.config.name == "Custom Gauntlet"
        assert runner.config.attack_rounds == 5

    def test_init_with_agent_factory(self):
        """Test initialization with agent factory."""
        factory = MagicMock(return_value="mock_agent")
        runner = GauntletRunner(agent_factory=factory)

        assert runner.agent_factory == factory

    def test_init_with_run_agent_fn(self):
        """Test initialization with run_agent_fn."""
        run_fn = AsyncMock(return_value="response")
        runner = GauntletRunner(run_agent_fn=run_fn)

        assert runner.run_agent_fn == run_fn

    def test_init_sandbox_disabled_by_default(self):
        """Test sandbox is disabled by default."""
        runner = GauntletRunner()

        assert runner.enable_sandbox is False
        assert runner._sandbox is None


# =============================================================================
# GauntletRunner Run Tests
# =============================================================================


class TestGauntletRunnerRun:
    """Test GauntletRunner run method."""

    @pytest.fixture
    def minimal_config(self):
        """Create minimal config for fast tests."""
        return GauntletConfig(
            attack_categories=[],  # No attacks
            probe_categories=[],  # No probes
            run_scenario_matrix=False,
            agents=["test-agent"],
        )

    @pytest.fixture
    def runner_with_config(self, minimal_config):
        """Create runner with minimal config."""
        return GauntletRunner(config=minimal_config)

    @pytest.mark.asyncio
    async def test_run_returns_result(self, runner_with_config):
        """Test run returns GauntletResult."""
        result = await runner_with_config.run("Test input")

        assert isinstance(result, GauntletResult)

    @pytest.mark.asyncio
    async def test_run_sets_gauntlet_id(self, runner_with_config):
        """Test run sets unique gauntlet_id."""
        result = await runner_with_config.run("Test input")

        assert result.gauntlet_id.startswith("gauntlet-")
        assert len(result.gauntlet_id) > 10

    @pytest.mark.asyncio
    async def test_run_computes_input_hash(self, runner_with_config):
        """Test run computes input hash."""
        input_text = "Test input content"
        result = await runner_with_config.run(input_text)

        expected_hash = hashlib.sha256(input_text.encode()).hexdigest()
        assert result.input_hash == expected_hash

    @pytest.mark.asyncio
    async def test_run_stores_input_summary(self, runner_with_config):
        """Test run stores input summary."""
        input_text = "This is a test input for the gauntlet"
        result = await runner_with_config.run(input_text)

        assert input_text in result.input_summary

    @pytest.mark.asyncio
    async def test_run_records_timing(self, runner_with_config):
        """Test run records timing information."""
        result = await runner_with_config.run("Test")

        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_run_calculates_verdict(self, runner_with_config):
        """Test run calculates verdict."""
        result = await runner_with_config.run("Test")

        assert result.verdict in (Verdict.PASS, Verdict.CONDITIONAL, Verdict.FAIL)

    @pytest.mark.asyncio
    async def test_run_with_progress_callback(self, runner_with_config):
        """Test run calls progress callback."""
        progress_calls = []

        def on_progress(phase: str, percent: float):
            progress_calls.append((phase, percent))

        await runner_with_config.run("Test", on_progress=on_progress)

        # Should have some progress calls
        assert len(progress_calls) > 0

    @pytest.mark.asyncio
    async def test_run_with_context(self, runner_with_config):
        """Test run accepts context."""
        result = await runner_with_config.run(
            "Test input",
            context="Additional context for validation",
        )

        assert isinstance(result, GauntletResult)


# =============================================================================
# Vulnerability Tracking Tests
# =============================================================================


class TestVulnerabilityTracking:
    """Test vulnerability tracking."""

    def test_add_attack_as_vulnerability(self):
        """Test converting attack to vulnerability."""
        runner = GauntletRunner()
        result = GauntletResult(
            gauntlet_id="test-gauntlet",
            input_hash="abc123",
            input_summary="Test",
            started_at=datetime.now().isoformat(),
            config_used={},
            agents_used=[],
        )

        # Create mock attack
        attack = MagicMock()
        attack.severity = 0.85
        attack.attack_type = MagicMock(value="security")
        attack.attack_description = "Test attack description"
        attack.evidence = "Attack evidence"
        attack.mitigation = "Fix the issue"
        attack.exploitability = 0.7
        attack.attacker = "test-agent"

        runner._add_attack_as_vulnerability(attack, result)

        assert len(result.vulnerabilities) == 1
        assert result.vulnerabilities[0].severity == SeverityLevel.HIGH

    def test_vulnerability_counter_increments(self):
        """Test vulnerability counter increments."""
        runner = GauntletRunner()
        result = GauntletResult(
            gauntlet_id="test",
            input_hash="abc",
            input_summary="Test",
            started_at=datetime.now().isoformat(),
            config_used={},
            agents_used=[],
        )

        attack = MagicMock()
        attack.severity = 0.5
        attack.attack_type = MagicMock(value="test")
        attack.attack_description = "Test"
        attack.evidence = ""
        attack.mitigation = ""
        attack.exploitability = 0.5
        attack.attacker = "agent"

        runner._add_attack_as_vulnerability(attack, result)
        runner._add_attack_as_vulnerability(attack, result)

        assert runner._vulnerability_counter == 2
        assert result.vulnerabilities[0].id == "vuln-0001"
        assert result.vulnerabilities[1].id == "vuln-0002"

    def test_severity_mapping_critical(self):
        """Test critical severity mapping."""
        runner = GauntletRunner()
        result = GauntletResult(
            gauntlet_id="test",
            input_hash="abc",
            input_summary="Test",
            started_at=datetime.now().isoformat(),
            config_used={},
            agents_used=[],
        )

        attack = MagicMock()
        attack.severity = 0.95
        attack.attack_type = MagicMock(value="test")
        attack.attack_description = "Critical attack"
        attack.evidence = ""
        attack.mitigation = ""
        attack.exploitability = 0.9
        attack.attacker = "agent"

        runner._add_attack_as_vulnerability(attack, result)

        assert result.vulnerabilities[0].severity == SeverityLevel.CRITICAL

    def test_severity_mapping_low(self):
        """Test low severity mapping."""
        runner = GauntletRunner()
        result = GauntletResult(
            gauntlet_id="test",
            input_hash="abc",
            input_summary="Test",
            started_at=datetime.now().isoformat(),
            config_used={},
            agents_used=[],
        )

        attack = MagicMock()
        attack.severity = 0.2
        attack.attack_type = MagicMock(value="test")
        attack.attack_description = "Low severity attack"
        attack.evidence = ""
        attack.mitigation = ""
        attack.exploitability = 0.2
        attack.attacker = "agent"

        runner._add_attack_as_vulnerability(attack, result)

        assert result.vulnerabilities[0].severity == SeverityLevel.LOW


# =============================================================================
# Code Extraction Tests
# =============================================================================


class TestCodeExtraction:
    """Test code extraction from evidence."""

    def test_extract_markdown_code_block_python(self):
        """Test extracting Python code from markdown block."""
        runner = GauntletRunner()
        evidence = """
        Here is the exploit:
        ```python
        import os
        os.system('whoami')
        ```
        End of exploit.
        """

        code, language = runner._extract_code_from_evidence(evidence)

        assert code is not None
        assert "import os" in code
        assert language == "python"

    def test_extract_markdown_code_block_javascript(self):
        """Test extracting JavaScript code from markdown block."""
        runner = GauntletRunner()
        evidence = """
        ```javascript
        fetch('/api/secret').then(r => r.json())
        ```
        """

        code, language = runner._extract_code_from_evidence(evidence)

        assert code is not None
        assert "fetch" in code
        assert language == "javascript"

    def test_extract_markdown_code_block_bash(self):
        """Test extracting Bash code from markdown block."""
        runner = GauntletRunner()
        evidence = """
        ```bash
        curl -X POST http://target/exploit
        ```
        """

        code, language = runner._extract_code_from_evidence(evidence)

        assert code is not None
        assert "curl" in code
        assert language == "bash"

    def test_extract_no_code_returns_none(self):
        """Test extracting from text without code returns None."""
        runner = GauntletRunner()
        evidence = "This is just plain text without any code blocks."

        code, language = runner._extract_code_from_evidence(evidence)

        assert code is None
        assert language == ""

    def test_extract_empty_evidence(self):
        """Test extracting from empty evidence."""
        runner = GauntletRunner()

        code, language = runner._extract_code_from_evidence("")

        assert code is None
        assert language == ""


# =============================================================================
# Sandbox Execution Tests
# =============================================================================


class TestSandboxExecution:
    """Test sandboxed code execution."""

    @pytest.mark.asyncio
    async def test_execute_without_sandbox(self):
        """Test execution when sandbox is disabled."""
        runner = GauntletRunner(enable_sandbox=False)

        result = await runner.execute_code_sandboxed("print('test')", "python")

        assert result["status"] == "sandbox_disabled"
        assert result["executed"] is False

    @pytest.mark.asyncio
    async def test_execute_attack_evidence_no_code(self):
        """Test executing evidence without code."""
        runner = GauntletRunner(enable_sandbox=False)

        result = await runner._execute_attack_evidence("Plain text without code")

        assert result is None


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestRunGauntletConvenience:
    """Test run_gauntlet convenience function."""

    @pytest.mark.asyncio
    async def test_run_gauntlet_returns_result(self):
        """Test run_gauntlet returns GauntletResult."""
        result = await run_gauntlet("Test input")

        assert isinstance(result, GauntletResult)

    @pytest.mark.asyncio
    async def test_run_gauntlet_with_config(self):
        """Test run_gauntlet with custom config."""
        config = GauntletConfig(
            attack_categories=[],
            probe_categories=[],
            run_scenario_matrix=False,
        )

        result = await run_gauntlet("Test", config=config)

        assert isinstance(result, GauntletResult)

    @pytest.mark.asyncio
    async def test_run_gauntlet_with_context(self):
        """Test run_gauntlet with context."""
        result = await run_gauntlet(
            "Test input",
            context="Additional context",
        )

        assert isinstance(result, GauntletResult)


# =============================================================================
# Integration Tests
# =============================================================================


class TestGauntletRunnerIntegration:
    """Integration tests for GauntletRunner."""

    @pytest.mark.asyncio
    async def test_full_run_minimal(self):
        """Test full run with minimal configuration."""
        config = GauntletConfig(
            name="Integration Test",
            attack_categories=[],
            probe_categories=[],
            run_scenario_matrix=False,
            agents=["test-agent"],
        )

        runner = GauntletRunner(config=config)
        result = await runner.run("Test input for validation")

        assert result.gauntlet_id is not None
        assert result.input_hash is not None
        assert result.verdict is not None
        assert result.completed_at is not None
