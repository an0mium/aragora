"""
Tests for Workflow Gauntlet Node.

Tests cover:
- GauntletStep class attributes (categories, frameworks, severity levels)
- Initialization defaults
- Execute: input resolution (context.inputs, context.state, step_outputs)
- Execute: no input content error
- Execute: successful run with mocked GauntletRunner
- Execute: severity threshold pass/fail logic
- Execute: require_passing=False always succeeds
- Execute: max_findings limit
- Execute: ImportError handling for gauntlet modules
- Compliance checks: framework persona mapping
- Compliance checks: individual framework failure handling
- Compliance checks: ImportError for personas
- Checkpoint and restore round-trip
- validate_config: valid and invalid severity thresholds
- Severity determination edge cases
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.workflow.step import WorkflowContext


# ============================================================================
# Helpers
# ============================================================================


def _make_context(inputs=None, state=None, step_outputs=None, current_step_config=None):
    """Create a WorkflowContext for testing."""
    return WorkflowContext(
        workflow_id="wf_test",
        definition_id="def_test",
        inputs=inputs or {},
        state=state or {},
        step_outputs=step_outputs or {},
        current_step_config=current_step_config or {},
    )


def _make_mock_finding(
    severity="low", category="test", description="test finding", finding_id="f1"
):
    """Create a mock finding object."""
    finding = MagicMock()
    finding.severity = severity
    finding.category = category
    finding.description = description
    finding.id = finding_id
    return finding


def _make_mock_result(vulnerabilities=None, risk_score=0.5):
    """Create a mock GauntletRunner result."""
    result = MagicMock()
    result.vulnerabilities = vulnerabilities or []
    result.risk_score = risk_score
    result.attack_summary = MagicMock()
    result.attack_summary.total_attacks = 7
    result.attack_summary.successful_attacks = 2
    result.probe_summary = MagicMock()
    result.probe_summary.probes_run = 4
    result.probe_summary.vulnerabilities_found = 1
    return result


# ============================================================================
# Class Attributes Tests
# ============================================================================


class TestGauntletStepAttributes:
    """Tests for GauntletStep class-level attributes."""

    def test_attack_categories_count(self):
        """Test ATTACK_CATEGORIES has 7 items."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        assert len(GauntletStep.ATTACK_CATEGORIES) == 7

    def test_attack_categories_contents(self):
        """Test ATTACK_CATEGORIES contains expected items."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        expected = [
            "prompt_injection",
            "jailbreak",
            "data_extraction",
            "hallucination",
            "bias",
            "privacy",
            "safety",
        ]
        assert GauntletStep.ATTACK_CATEGORIES == expected

    def test_probe_categories_count(self):
        """Test PROBE_CATEGORIES has 4 items."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        assert len(GauntletStep.PROBE_CATEGORIES) == 4

    def test_probe_categories_contents(self):
        """Test PROBE_CATEGORIES contains expected items."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        expected = ["reasoning", "factuality", "consistency", "boundaries"]
        assert GauntletStep.PROBE_CATEGORIES == expected

    def test_compliance_frameworks_count(self):
        """Test COMPLIANCE_FRAMEWORKS has 7 items."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        assert len(GauntletStep.COMPLIANCE_FRAMEWORKS) == 7

    def test_compliance_frameworks_contents(self):
        """Test COMPLIANCE_FRAMEWORKS contains expected items."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        expected = ["gdpr", "hipaa", "soc2", "pci_dss", "nist_csf", "ai_act", "sox"]
        assert GauntletStep.COMPLIANCE_FRAMEWORKS == expected

    def test_severity_levels_count(self):
        """Test SEVERITY_LEVELS has 4 items."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        assert len(GauntletStep.SEVERITY_LEVELS) == 4

    def test_severity_levels_contents(self):
        """Test SEVERITY_LEVELS contains low, medium, high, critical."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        assert GauntletStep.SEVERITY_LEVELS == ["low", "medium", "high", "critical"]

    def test_severity_levels_order(self):
        """Test SEVERITY_LEVELS are in ascending order of severity."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        levels = GauntletStep.SEVERITY_LEVELS
        assert levels.index("low") < levels.index("medium")
        assert levels.index("medium") < levels.index("high")
        assert levels.index("high") < levels.index("critical")


# ============================================================================
# Initialization Tests
# ============================================================================


class TestGauntletStepInit:
    """Tests for GauntletStep initialization."""

    def test_basic_init(self):
        """Test basic GauntletStep initialization."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")
        assert step.name == "Test Gauntlet"

    def test_default_config(self):
        """Test GauntletStep with no config uses empty dict."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")
        assert step.config == {}

    def test_custom_config(self):
        """Test GauntletStep with custom config."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        config = {"severity_threshold": "high", "require_passing": False}
        step = GauntletStep(name="Test Gauntlet", config=config)
        assert step.config["severity_threshold"] == "high"
        assert step.config["require_passing"] is False

    def test_findings_count_initial_value(self):
        """Test _findings_count is initialized to 0."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")
        assert step._findings_count == 0

    def test_highest_severity_initial_value(self):
        """Test _highest_severity is initialized to None."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")
        assert step._highest_severity is None

    def test_inherits_base_step(self):
        """Test GauntletStep inherits from BaseStep."""
        from aragora.workflow.nodes.gauntlet import GauntletStep
        from aragora.workflow.step import BaseStep

        step = GauntletStep(name="Test Gauntlet")
        assert isinstance(step, BaseStep)


# ============================================================================
# Execute - Input Resolution Tests
# ============================================================================


class TestGauntletStepInputResolution:
    """Tests for input content resolution in execute."""

    @pytest.mark.asyncio
    async def test_no_input_content_returns_error(self):
        """Test that missing input content returns error dict."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")
        ctx = _make_context()
        result = await step.execute(ctx)

        assert result["success"] is False
        assert "No input content found" in result["error"]
        assert result["findings"] == []

    @pytest.mark.asyncio
    async def test_no_input_content_error_includes_key_name(self):
        """Test error message mentions the input key that was not found."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet", config={"input_key": "my_content"})
        ctx = _make_context()
        result = await step.execute(ctx)

        assert "my_content" in result["error"]

    @pytest.mark.asyncio
    async def test_input_found_in_context_inputs(self):
        """Test input resolved from context.inputs via get_input."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")
        ctx = _make_context(inputs={"content": "Test content for gauntlet"})

        mock_result = _make_mock_result()
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_input_found_in_context_state(self):
        """Test input resolved from context.state when not in inputs."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")
        ctx = _make_context(state={"content": "State content"})

        mock_result = _make_mock_result()
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_input_found_in_step_outputs(self):
        """Test input resolved from step_outputs when not in inputs or state."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")
        ctx = _make_context(step_outputs={"prev_step": {"content": "Previous step content"}})

        mock_result = _make_mock_result()
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_custom_input_key(self):
        """Test using a custom input_key to resolve content."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet", config={"input_key": "document"})
        ctx = _make_context(inputs={"document": "Document content"})

        mock_result = _make_mock_result()
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        assert result["success"] is True


# ============================================================================
# Execute - Successful Run Tests
# ============================================================================


class _FakeSeverityLevel:
    """A real class to use as a stand-in for SeverityLevel in isinstance() checks.

    Since our mock findings use plain strings for severity, isinstance(str, _FakeSeverityLevel)
    will correctly return False, avoiding TypeError from passing a MagicMock to isinstance().
    """

    def __init__(self, value):
        self.value = value


def _mock_gauntlet_import(mock_runner, attack_cats=None, probe_cats=None):
    """Create a mock_import function for gauntlet modules."""
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    attack_values = attack_cats or []
    probe_values = probe_cats or []

    def mock_import(name, *args, **kwargs):
        if name == "aragora.gauntlet.runner":
            mod = MagicMock()
            mod.GauntletRunner = MagicMock(return_value=mock_runner)
            return mod
        if name == "aragora.gauntlet.config":
            mod = MagicMock()
            mock_ac = MagicMock()
            mock_ac_values = [MagicMock(value=v) for v in attack_values]
            mock_ac.__iter__ = MagicMock(return_value=iter(mock_ac_values))
            mod.AttackCategory = mock_ac
            mock_pc = MagicMock()
            mock_pc_values = [MagicMock(value=v) for v in probe_values]
            mock_pc.__iter__ = MagicMock(return_value=iter(mock_pc_values))
            mod.ProbeCategory = mock_pc
            mod.GauntletConfig = MagicMock()
            return mod
        if name == "aragora.gauntlet.result":
            mod = MagicMock()
            mod.SeverityLevel = _FakeSeverityLevel
            return mod
        return original_import(name, *args, **kwargs)

    return mock_import


class TestGauntletStepExecution:
    """Tests for successful GauntletStep execution."""

    @pytest.mark.asyncio
    async def test_successful_run_no_findings(self):
        """Test successful gauntlet run with no findings."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")
        ctx = _make_context(inputs={"content": "Safe content"})

        mock_result = _make_mock_result(vulnerabilities=[])
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        assert result["success"] is True
        assert result["passed"] is True
        assert result["findings_count"] == 0
        assert result["highest_severity"] is None

    @pytest.mark.asyncio
    async def test_successful_run_with_findings(self):
        """Test successful gauntlet run with findings below threshold."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet", config={"severity_threshold": "high"})
        ctx = _make_context(inputs={"content": "Content with minor issues"})

        findings = [_make_mock_finding(severity="low"), _make_mock_finding(severity="medium")]
        mock_result = _make_mock_result(vulnerabilities=findings)
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        assert result["success"] is True
        assert result["passed"] is True
        assert result["findings_count"] == 2
        assert result["highest_severity"] == "medium"

    @pytest.mark.asyncio
    async def test_result_contains_attack_summary(self):
        """Test result includes attack summary fields."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")
        ctx = _make_context(inputs={"content": "Test content"})

        mock_result = _make_mock_result()
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        assert "attack_summary" in result
        assert result["attack_summary"]["total"] == 7
        assert result["attack_summary"]["successful"] == 2

    @pytest.mark.asyncio
    async def test_result_contains_probe_summary(self):
        """Test result includes probe summary fields."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")
        ctx = _make_context(inputs={"content": "Test content"})

        mock_result = _make_mock_result()
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        assert "probe_summary" in result
        assert result["probe_summary"]["total"] == 4
        assert result["probe_summary"]["passed"] == 3

    @pytest.mark.asyncio
    async def test_result_contains_risk_score(self):
        """Test result includes risk_score from runner."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")
        ctx = _make_context(inputs={"content": "Test content"})

        mock_result = _make_mock_result(risk_score=0.75)
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        assert result["risk_score"] == 0.75

    @pytest.mark.asyncio
    async def test_result_contains_counts(self):
        """Test result includes attacks_run, probes_run, compliance_checks_run."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(
            name="Test Gauntlet",
            config={
                "attack_categories": ["prompt_injection", "jailbreak"],
                "probe_categories": ["reasoning"],
                "compliance_frameworks": [],
            },
        )
        ctx = _make_context(inputs={"content": "Test content"})

        mock_result = _make_mock_result()
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        assert result["attacks_run"] == 2
        assert result["probes_run"] == 1
        assert result["compliance_checks_run"] == 0

    @pytest.mark.asyncio
    async def test_config_merged_with_current_step_config(self):
        """Test step config is merged with current_step_config from context."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(
            name="Test Gauntlet",
            config={"severity_threshold": "medium"},
        )
        ctx = _make_context(
            inputs={"content": "Test content"},
            current_step_config={"severity_threshold": "critical"},
        )

        mock_result = _make_mock_result(
            vulnerabilities=[
                _make_mock_finding(severity="high"),
            ]
        )
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        # With threshold=critical, high severity should pass
        assert result["passed"] is True
        assert result["severity_threshold"] == "critical"


# ============================================================================
# Execute - Severity Threshold Tests
# ============================================================================


class TestGauntletSeverityThreshold:
    """Tests for severity threshold pass/fail logic."""

    @pytest.mark.asyncio
    async def test_low_severity_below_medium_threshold_passes(self):
        """Test low severity findings pass with medium threshold."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet", config={"severity_threshold": "medium"})
        ctx = _make_context(inputs={"content": "Test content"})

        findings = [_make_mock_finding(severity="low")]
        mock_result = _make_mock_result(vulnerabilities=findings)
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        assert result["passed"] is True
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_medium_severity_at_medium_threshold_fails(self):
        """Test medium severity findings fail with medium threshold (>= means fail)."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet", config={"severity_threshold": "medium"})
        ctx = _make_context(inputs={"content": "Test content"})

        findings = [_make_mock_finding(severity="medium")]
        mock_result = _make_mock_result(vulnerabilities=findings)
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        # highest_idx (1 for medium) is NOT < threshold_idx (1 for medium), so passed=False
        assert result["passed"] is False
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_high_severity_above_medium_threshold_fails(self):
        """Test high severity findings fail with medium threshold."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet", config={"severity_threshold": "medium"})
        ctx = _make_context(inputs={"content": "Test content"})

        findings = [_make_mock_finding(severity="high")]
        mock_result = _make_mock_result(vulnerabilities=findings)
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        assert result["passed"] is False
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_critical_severity_always_fails(self):
        """Test critical severity findings fail regardless of threshold."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet", config={"severity_threshold": "critical"})
        ctx = _make_context(inputs={"content": "Test content"})

        findings = [_make_mock_finding(severity="critical")]
        mock_result = _make_mock_result(vulnerabilities=findings)
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        # highest_idx (3 for critical) is NOT < threshold_idx (3 for critical)
        assert result["passed"] is False

    @pytest.mark.asyncio
    async def test_no_findings_passes_with_medium_threshold(self):
        """Test no findings means passed with medium threshold.

        With no findings, _highest_severity stays None, so highest_idx=0.
        With medium threshold, threshold_idx=1, and 0 < 1 is True => passed.
        """
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet", config={"severity_threshold": "medium"})
        ctx = _make_context(inputs={"content": "Test content"})

        mock_result = _make_mock_result(vulnerabilities=[])
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        assert result["passed"] is True
        assert result["findings_count"] == 0

    @pytest.mark.asyncio
    async def test_no_findings_with_low_threshold_edge_case(self):
        """Test no findings with low threshold is an edge case.

        With no findings, _highest_severity stays None, so highest_idx=0.
        With low threshold, threshold_idx=0, and 0 < 0 is False => not passed.
        This is a known edge case in the source code.
        """
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet", config={"severity_threshold": "low"})
        ctx = _make_context(inputs={"content": "Test content"})

        mock_result = _make_mock_result(vulnerabilities=[])
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        assert result["passed"] is False
        assert result["findings_count"] == 0

    @pytest.mark.asyncio
    async def test_low_threshold_fails_on_low_severity(self):
        """Test low threshold fails even on low severity findings."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet", config={"severity_threshold": "low"})
        ctx = _make_context(inputs={"content": "Test content"})

        findings = [_make_mock_finding(severity="low")]
        mock_result = _make_mock_result(vulnerabilities=findings)
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        # low (0) is NOT < low (0), so passed=False
        assert result["passed"] is False

    @pytest.mark.asyncio
    async def test_highest_severity_tracked_across_findings(self):
        """Test that highest severity is determined across all findings."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet", config={"severity_threshold": "critical"})
        ctx = _make_context(inputs={"content": "Test content"})

        findings = [
            _make_mock_finding(severity="low"),
            _make_mock_finding(severity="high"),
            _make_mock_finding(severity="medium"),
        ]
        mock_result = _make_mock_result(vulnerabilities=findings)
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        assert result["highest_severity"] == "high"


# ============================================================================
# Execute - require_passing Tests
# ============================================================================


class TestGauntletRequirePassing:
    """Tests for require_passing flag behavior."""

    @pytest.mark.asyncio
    async def test_require_passing_false_always_succeeds(self):
        """Test require_passing=False makes success=True even when not passed."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(
            name="Test Gauntlet",
            config={"require_passing": False, "severity_threshold": "low"},
        )
        ctx = _make_context(inputs={"content": "Test content"})

        findings = [_make_mock_finding(severity="critical")]
        mock_result = _make_mock_result(vulnerabilities=findings)
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        assert result["passed"] is False
        assert result["success"] is True  # success=True because require_passing=False

    @pytest.mark.asyncio
    async def test_require_passing_true_fails_when_not_passed(self):
        """Test require_passing=True makes success=False when not passed."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(
            name="Test Gauntlet",
            config={"require_passing": True, "severity_threshold": "medium"},
        )
        ctx = _make_context(inputs={"content": "Test content"})

        findings = [_make_mock_finding(severity="high")]
        mock_result = _make_mock_result(vulnerabilities=findings)
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        assert result["passed"] is False
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_require_passing_default_is_true(self):
        """Test that require_passing defaults to True."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")
        ctx = _make_context(inputs={"content": "Test content"})

        findings = [_make_mock_finding(severity="high")]
        mock_result = _make_mock_result(vulnerabilities=findings)
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        # Default severity_threshold is "medium", high >= medium => passed=False
        # Default require_passing is True => success=False
        assert result["success"] is False


# ============================================================================
# Execute - max_findings Tests
# ============================================================================


class TestGauntletMaxFindings:
    """Tests for max_findings limiting behavior."""

    @pytest.mark.asyncio
    async def test_max_findings_limits_output(self):
        """Test that findings in result are limited by max_findings."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(
            name="Test Gauntlet",
            config={"max_findings": 2, "severity_threshold": "critical"},
        )
        ctx = _make_context(inputs={"content": "Test content"})

        findings = [_make_mock_finding(severity="low", finding_id=f"f{i}") for i in range(5)]
        mock_result = _make_mock_result(vulnerabilities=findings)
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        assert len(result["findings"]) == 2
        assert result["findings_count"] == 5  # Total count is still 5

    @pytest.mark.asyncio
    async def test_max_findings_default_is_100(self):
        """Test that default max_findings is 100 (not limiting small sets)."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet", config={"severity_threshold": "critical"})
        ctx = _make_context(inputs={"content": "Test content"})

        findings = [_make_mock_finding(severity="low", finding_id=f"f{i}") for i in range(10)]
        mock_result = _make_mock_result(vulnerabilities=findings)
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        assert len(result["findings"]) == 10  # All 10 fit within default 100


# ============================================================================
# Execute - ImportError Handling Tests
# ============================================================================


class TestGauntletImportError:
    """Tests for ImportError handling in execute."""

    @pytest.mark.asyncio
    async def test_import_error_returns_error_dict(self):
        """Test that ImportError when importing gauntlet modules returns error."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")
        ctx = _make_context(inputs={"content": "Test content"})

        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def mock_import(name, *args, **kwargs):
            if name == "aragora.gauntlet.runner":
                raise ImportError("No module named 'aragora.gauntlet.runner'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = await step.execute(ctx)

        assert result["success"] is False
        assert "not available" in result["error"]
        assert result["findings"] == []

    @pytest.mark.asyncio
    async def test_import_error_includes_module_name(self):
        """Test that ImportError message includes the module name."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")
        ctx = _make_context(inputs={"content": "Test content"})

        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def mock_import(name, *args, **kwargs):
            if name == "aragora.gauntlet.runner":
                raise ImportError("No module named 'aragora.gauntlet.runner'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = await step.execute(ctx)

        assert "aragora.gauntlet.runner" in result["error"]


# ============================================================================
# Compliance Checks Tests
# ============================================================================


class TestGauntletComplianceChecks:
    """Tests for _run_compliance_checks method."""

    @pytest.mark.asyncio
    async def test_compliance_framework_persona_mapping(self):
        """Test that each framework maps to its expected persona class."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")

        # Create mock persona classes
        mock_personas = {}
        for name in [
            "GDPRPersona",
            "HIPAAPersona",
            "SOC2Persona",
            "PCIDSSPersona",
            "NISTCSFPersona",
            "AIActPersona",
            "SOXPersona",
        ]:
            persona_cls = MagicMock()
            persona_instance = MagicMock()
            persona_instance.evaluate = AsyncMock(
                return_value={
                    "compliant": True,
                    "findings": [],
                    "score": 0.95,
                }
            )
            persona_cls.return_value = persona_instance
            mock_personas[name] = persona_cls

        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def mock_import(name, *args, **kwargs):
            if name == "aragora.gauntlet.personas":
                mod = MagicMock()
                for pname, pcls in mock_personas.items():
                    setattr(mod, pname, pcls)
                return mod
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            results = await step._run_compliance_checks(
                input_content="Test content",
                frameworks=["gdpr", "hipaa", "soc2", "pci_dss", "nist_csf", "ai_act", "sox"],
                agents=["claude"],
            )

        assert len(results) == 7
        for r in results:
            assert r["passed"] is True
            assert r["score"] == 0.95

    @pytest.mark.asyncio
    async def test_compliance_unknown_framework_skipped(self):
        """Test that unknown frameworks are skipped."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")

        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def mock_import(name, *args, **kwargs):
            if name == "aragora.gauntlet.personas":
                mod = MagicMock()
                # Only set up one persona
                persona_cls = MagicMock()
                persona_instance = MagicMock()
                persona_instance.evaluate = AsyncMock(
                    return_value={"compliant": True, "findings": [], "score": 1.0}
                )
                persona_cls.return_value = persona_instance
                mod.GDPRPersona = persona_cls
                mod.HIPAAPersona = MagicMock()
                mod.SOC2Persona = MagicMock()
                mod.PCIDSSPersona = MagicMock()
                mod.NISTCSFPersona = MagicMock()
                mod.AIActPersona = MagicMock()
                mod.SOXPersona = MagicMock()
                return mod
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            results = await step._run_compliance_checks(
                input_content="Test content",
                frameworks=["unknown_framework"],
                agents=["claude"],
            )

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_compliance_individual_framework_failure_handled(self):
        """Test that individual framework evaluation failures are caught."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")

        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def mock_import(name, *args, **kwargs):
            if name == "aragora.gauntlet.personas":
                mod = MagicMock()
                # GDPR persona that raises an exception
                gdpr_cls = MagicMock()
                gdpr_instance = MagicMock()
                gdpr_instance.evaluate = AsyncMock(
                    side_effect=RuntimeError("GDPR evaluation failed")
                )
                gdpr_cls.return_value = gdpr_instance
                mod.GDPRPersona = gdpr_cls
                # HIPAA persona that works
                hipaa_cls = MagicMock()
                hipaa_instance = MagicMock()
                hipaa_instance.evaluate = AsyncMock(
                    return_value={"compliant": True, "findings": [], "score": 0.9}
                )
                hipaa_cls.return_value = hipaa_instance
                mod.HIPAAPersona = hipaa_cls
                mod.SOC2Persona = MagicMock()
                mod.PCIDSSPersona = MagicMock()
                mod.NISTCSFPersona = MagicMock()
                mod.AIActPersona = MagicMock()
                mod.SOXPersona = MagicMock()
                return mod
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            results = await step._run_compliance_checks(
                input_content="Test content",
                frameworks=["gdpr", "hipaa"],
                agents=["claude"],
            )

        assert len(results) == 2
        # GDPR failed
        gdpr_result = next(r for r in results if r["framework"] == "gdpr")
        assert gdpr_result["passed"] is False
        assert "error" in gdpr_result
        # HIPAA succeeded
        hipaa_result = next(r for r in results if r["framework"] == "hipaa")
        assert hipaa_result["passed"] is True

    @pytest.mark.asyncio
    async def test_compliance_import_error_returns_empty(self):
        """Test that ImportError for personas returns empty results."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")

        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def mock_import(name, *args, **kwargs):
            if name == "aragora.gauntlet.personas":
                raise ImportError("No module named 'aragora.gauntlet.personas'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            results = await step._run_compliance_checks(
                input_content="Test content",
                frameworks=["gdpr", "hipaa"],
                agents=["claude"],
            )

        assert results == []

    @pytest.mark.asyncio
    async def test_compliance_results_included_in_execute(self):
        """Test compliance results are included in execute result."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(
            name="Test Gauntlet",
            config={"compliance_frameworks": ["gdpr"], "severity_threshold": "critical"},
        )
        ctx = _make_context(inputs={"content": "Test content"})

        mock_result = _make_mock_result(vulnerabilities=[])
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def mock_import(name, *args, **kwargs):
            if name == "aragora.gauntlet.runner":
                mod = MagicMock()
                mod.GauntletRunner = MagicMock(return_value=mock_runner)
                return mod
            if name == "aragora.gauntlet.config":
                mod = MagicMock()
                mock_ac = MagicMock()
                mock_ac.__iter__ = MagicMock(return_value=iter([]))
                mod.AttackCategory = mock_ac
                mock_pc = MagicMock()
                mock_pc.__iter__ = MagicMock(return_value=iter([]))
                mod.ProbeCategory = mock_pc
                mod.GauntletConfig = MagicMock()
                return mod
            if name == "aragora.gauntlet.result":
                mod = MagicMock()
                mod.SeverityLevel = _FakeSeverityLevel
                return mod
            if name == "aragora.gauntlet.personas":
                mod = MagicMock()
                gdpr_cls = MagicMock()
                gdpr_instance = MagicMock()
                gdpr_instance.evaluate = AsyncMock(
                    return_value={
                        "compliant": True,
                        "findings": [],
                        "score": 0.95,
                    }
                )
                gdpr_cls.return_value = gdpr_instance
                mod.GDPRPersona = gdpr_cls
                mod.HIPAAPersona = MagicMock()
                mod.SOC2Persona = MagicMock()
                mod.PCIDSSPersona = MagicMock()
                mod.NISTCSFPersona = MagicMock()
                mod.AIActPersona = MagicMock()
                mod.SOXPersona = MagicMock()
                return mod
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = await step.execute(ctx)

        assert "compliance_results" in result
        assert len(result["compliance_results"]) == 1
        assert result["compliance_results"][0]["framework"] == "gdpr"
        assert result["compliance_results"][0]["passed"] is True


# ============================================================================
# Checkpoint / Restore Tests
# ============================================================================


class TestGauntletCheckpointRestore:
    """Tests for checkpoint and restore methods."""

    @pytest.mark.asyncio
    async def test_checkpoint_returns_state(self):
        """Test checkpoint returns findings_count and highest_severity."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")
        step._findings_count = 5
        step._highest_severity = "high"

        state = await step.checkpoint()

        assert state["findings_count"] == 5
        assert state["highest_severity"] == "high"

    @pytest.mark.asyncio
    async def test_checkpoint_default_values(self):
        """Test checkpoint returns default values when not modified."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")
        state = await step.checkpoint()

        assert state["findings_count"] == 0
        assert state["highest_severity"] is None

    @pytest.mark.asyncio
    async def test_restore_sets_state(self):
        """Test restore sets _findings_count and _highest_severity."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")
        await step.restore({"findings_count": 10, "highest_severity": "critical"})

        assert step._findings_count == 10
        assert step._highest_severity == "critical"

    @pytest.mark.asyncio
    async def test_restore_with_defaults(self):
        """Test restore with empty state uses defaults."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")
        step._findings_count = 5
        step._highest_severity = "high"

        await step.restore({})

        assert step._findings_count == 0
        assert step._highest_severity is None

    @pytest.mark.asyncio
    async def test_checkpoint_restore_round_trip(self):
        """Test checkpoint/restore round-trip preserves values."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step1 = GauntletStep(name="Step1")
        step1._findings_count = 42
        step1._highest_severity = "medium"

        state = await step1.checkpoint()

        step2 = GauntletStep(name="Step2")
        await step2.restore(state)

        assert step2._findings_count == 42
        assert step2._highest_severity == "medium"

    @pytest.mark.asyncio
    async def test_restore_partial_state(self):
        """Test restore with partial state (only findings_count)."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet")
        await step.restore({"findings_count": 7})

        assert step._findings_count == 7
        assert step._highest_severity is None


# ============================================================================
# validate_config Tests
# ============================================================================


class TestGauntletValidateConfig:
    """Tests for validate_config method."""

    def test_valid_severity_low(self):
        """Test validate_config passes for severity_threshold=low."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test", config={"severity_threshold": "low"})
        assert step.validate_config() is True

    def test_valid_severity_medium(self):
        """Test validate_config passes for severity_threshold=medium."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test", config={"severity_threshold": "medium"})
        assert step.validate_config() is True

    def test_valid_severity_high(self):
        """Test validate_config passes for severity_threshold=high."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test", config={"severity_threshold": "high"})
        assert step.validate_config() is True

    def test_valid_severity_critical(self):
        """Test validate_config passes for severity_threshold=critical."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test", config={"severity_threshold": "critical"})
        assert step.validate_config() is True

    def test_invalid_severity_threshold(self):
        """Test validate_config fails for invalid severity_threshold."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test", config={"severity_threshold": "extreme"})
        assert step.validate_config() is False

    def test_default_severity_threshold_valid(self):
        """Test validate_config passes when no severity_threshold configured."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test")
        assert step.validate_config() is True

    def test_invalid_severity_empty_string(self):
        """Test validate_config fails for empty string severity_threshold."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test", config={"severity_threshold": ""})
        assert step.validate_config() is False

    def test_invalid_severity_case_sensitive(self):
        """Test validate_config is case sensitive (Medium != medium)."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test", config={"severity_threshold": "Medium"})
        assert step.validate_config() is False


# ============================================================================
# Findings Detail Tests
# ============================================================================


class TestGauntletFindingsDetail:
    """Tests for finding detail output in results."""

    @pytest.mark.asyncio
    async def test_findings_include_id_severity_category_description(self):
        """Test each finding in results has id, severity, category, description."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet", config={"severity_threshold": "critical"})
        ctx = _make_context(inputs={"content": "Test content"})

        finding = _make_mock_finding(
            severity="medium",
            category="prompt_injection",
            description="Found injection vulnerability",
            finding_id="vuln-001",
        )
        mock_result = _make_mock_result(vulnerabilities=[finding])
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        assert len(result["findings"]) == 1
        f = result["findings"][0]
        assert f["id"] == "vuln-001"
        assert f["severity"] == "medium"
        assert f["category"] == "prompt_injection"
        assert f["description"] == "Found injection vulnerability"

    @pytest.mark.asyncio
    async def test_findings_missing_attrs_use_defaults(self):
        """Test findings with missing attributes use fallback values."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet", config={"severity_threshold": "critical"})
        ctx = _make_context(inputs={"content": "Test content"})

        # Create a finding that is a simple string (no attrs)
        finding = "bare string finding"
        mock_result = _make_mock_result(vulnerabilities=[finding])
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        with patch("builtins.__import__", side_effect=_mock_gauntlet_import(mock_runner)):
            result = await step.execute(ctx)

        assert len(result["findings"]) == 1
        f = result["findings"][0]
        # Fallback values for getattr
        assert f["id"] == "0"  # str(i) where i=0
        assert f["severity"] == "unknown"
        assert f["category"] == "unknown"
        assert f["description"] == "bare string finding"

    @pytest.mark.asyncio
    async def test_severity_level_enum_handled(self):
        """Test that SeverityLevel enum values are converted to strings."""
        from aragora.workflow.nodes.gauntlet import GauntletStep

        step = GauntletStep(name="Test Gauntlet", config={"severity_threshold": "critical"})
        ctx = _make_context(inputs={"content": "Test content"})

        # Create a finding with SeverityLevel enum-like severity
        mock_severity = MagicMock()
        mock_severity.value = "high"
        finding = MagicMock()
        finding.severity = mock_severity
        finding.category = "test"
        finding.description = "enum severity finding"
        finding.id = "e1"

        mock_result = _make_mock_result(vulnerabilities=[finding])
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_result)

        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def mock_import(name, *args, **kwargs):
            if name == "aragora.gauntlet.runner":
                mod = MagicMock()
                mod.GauntletRunner = MagicMock(return_value=mock_runner)
                return mod
            if name == "aragora.gauntlet.config":
                mod = MagicMock()
                mock_ac = MagicMock()
                mock_ac.__iter__ = MagicMock(return_value=iter([]))
                mod.AttackCategory = mock_ac
                mock_pc = MagicMock()
                mock_pc.__iter__ = MagicMock(return_value=iter([]))
                mod.ProbeCategory = mock_pc
                mod.GauntletConfig = MagicMock()
                return mod
            if name == "aragora.gauntlet.result":
                mod = MagicMock()
                # Make isinstance check return True for the mock_severity
                severity_level_cls = type(mock_severity)
                mod.SeverityLevel = severity_level_cls
                return mod
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = await step.execute(ctx)

        assert result["highest_severity"] == "high"
