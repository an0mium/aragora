"""Tests for Gauntlet - Adversarial Validation Engine.

Tests cover:
- Configuration types (AttackCategory, ProbeCategory, GauntletConfig)
- Result types (GauntletFinding, GauntletResult from config module)
- Decision receipts
- Risk heatmap
- Runner instantiation
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch


class TestAttackCategory:
    """Tests for AttackCategory enum."""

    def test_attack_categories_exist(self):
        """Should have all expected attack categories."""
        from aragora.gauntlet.config import AttackCategory

        assert hasattr(AttackCategory, "SECURITY")
        assert hasattr(AttackCategory, "LOGIC")
        assert hasattr(AttackCategory, "ARCHITECTURE")
        assert hasattr(AttackCategory, "COMPLIANCE")

    def test_attack_category_values(self):
        """AttackCategory values should be strings."""
        from aragora.gauntlet.config import AttackCategory

        assert AttackCategory.SECURITY.value == "security"
        assert AttackCategory.LOGIC.value == "logic"


class TestProbeCategory:
    """Tests for ProbeCategory enum."""

    def test_probe_categories_exist(self):
        """Should have all expected probe categories."""
        from aragora.gauntlet.config import ProbeCategory

        assert hasattr(ProbeCategory, "HALLUCINATION")
        assert hasattr(ProbeCategory, "SYCOPHANCY")
        assert hasattr(ProbeCategory, "CONTRADICTION")

    def test_probe_category_values(self):
        """ProbeCategory values should be strings."""
        from aragora.gauntlet.config import ProbeCategory

        assert ProbeCategory.HALLUCINATION.value == "hallucination"
        assert ProbeCategory.SYCOPHANCY.value == "sycophancy"


class TestGauntletConfig:
    """Tests for GauntletConfig dataclass."""

    def test_config_with_defaults(self):
        """Should create config with default values."""
        from aragora.gauntlet.config import GauntletConfig

        config = GauntletConfig()

        assert config is not None
        assert config.attack_rounds >= 1
        assert config.name == "Gauntlet Validation"

    def test_config_with_custom_values(self):
        """Should create config with custom values."""
        from aragora.gauntlet.config import GauntletConfig, AttackCategory

        config = GauntletConfig(
            name="Custom Gauntlet",
            attack_rounds=5,
            attack_categories=[AttackCategory.SECURITY],
            agents=["claude", "gpt-4"],
        )

        assert config.attack_rounds == 5
        assert config.name == "Custom Gauntlet"
        assert AttackCategory.SECURITY in config.attack_categories
        assert "claude" in config.agents

    def test_config_presets(self):
        """Should have preset configurations."""
        from aragora.gauntlet.config import GauntletConfig

        quick = GauntletConfig.quick()
        security = GauntletConfig.security_focused()
        compliance = GauntletConfig.compliance_focused()

        assert quick.attack_rounds == 1
        assert security is not None
        assert compliance is not None

    def test_config_to_dict(self):
        """Should convert config to dictionary."""
        from aragora.gauntlet.config import GauntletConfig

        config = GauntletConfig(name="Test Config")
        data = config.to_dict()

        assert isinstance(data, dict)
        assert data["name"] == "Test Config"

    def test_config_from_dict(self):
        """Should create config from dictionary."""
        from aragora.gauntlet.config import GauntletConfig

        data = {"name": "From Dict", "attack_rounds": 3}
        config = GauntletConfig.from_dict(data)

        assert config.name == "From Dict"
        assert config.attack_rounds == 3


class TestPassFailCriteria:
    """Tests for PassFailCriteria dataclass."""

    def test_criteria_defaults(self):
        """Should create criteria with defaults."""
        from aragora.gauntlet.config import PassFailCriteria

        criteria = PassFailCriteria()

        assert criteria.max_critical_findings == 0
        assert criteria.max_high_findings == 2

    def test_criteria_strict(self):
        """Should have strict preset."""
        from aragora.gauntlet.config import PassFailCriteria

        strict = PassFailCriteria.strict()

        assert strict.max_critical_findings == 0
        assert strict.max_high_findings == 0
        assert strict.require_formal_verification is True

    def test_criteria_lenient(self):
        """Should have lenient preset."""
        from aragora.gauntlet.config import PassFailCriteria

        lenient = PassFailCriteria.lenient()

        assert lenient.max_critical_findings == 1
        assert lenient.max_high_findings == 5


class TestGauntletTypes:
    """Tests for shared Gauntlet types."""

    def test_severity_level_exists(self):
        """GauntletSeverity should have levels."""
        from aragora.gauntlet.types import GauntletSeverity

        assert hasattr(GauntletSeverity, "LOW")
        assert hasattr(GauntletSeverity, "MEDIUM")
        assert hasattr(GauntletSeverity, "HIGH")
        assert hasattr(GauntletSeverity, "CRITICAL")

    def test_gauntlet_phase_types(self):
        """GauntletPhase should have all phases."""
        from aragora.gauntlet.types import GauntletPhase

        assert hasattr(GauntletPhase, "NOT_STARTED")
        assert hasattr(GauntletPhase, "RED_TEAM")
        assert hasattr(GauntletPhase, "ADVERSARIAL_PROBING")

    def test_verdict_types(self):
        """Verdict should have all types."""
        from aragora.gauntlet.types import Verdict

        assert hasattr(Verdict, "PASS")
        assert hasattr(Verdict, "FAIL")


class TestGauntletFinding:
    """Tests for GauntletFinding dataclass."""

    def test_finding_creation(self):
        """Should create GauntletFinding."""
        from aragora.gauntlet.config import GauntletFinding, GauntletPhase
        from aragora.gauntlet.types import GauntletSeverity

        finding = GauntletFinding(
            title="SQL Injection Risk",
            description="Input not sanitized",
            severity=GauntletSeverity.HIGH,
            source_phase=GauntletPhase.RED_TEAM,
            category="security",
        )

        assert finding.title == "SQL Injection Risk"
        assert finding.severity == GauntletSeverity.HIGH

    def test_finding_defaults(self):
        """Finding should have sensible defaults."""
        from aragora.gauntlet.config import GauntletFinding

        finding = GauntletFinding()

        assert finding.id is not None
        assert finding.is_verified is False


class TestGauntletResult:
    """Tests for GauntletResult dataclass (from config module)."""

    def test_result_creation(self):
        """Should create GauntletResult."""
        from aragora.gauntlet.config import GauntletResult

        result = GauntletResult()

        assert result is not None
        assert result.passed is False
        assert result.risk_score == 0.0

    def test_result_severity_counts(self):
        """Should count findings by severity."""
        from aragora.gauntlet.config import GauntletResult, GauntletFinding
        from aragora.gauntlet.types import GauntletSeverity

        result = GauntletResult()
        result.findings = [
            GauntletFinding(severity=GauntletSeverity.HIGH),
            GauntletFinding(severity=GauntletSeverity.HIGH),
            GauntletFinding(severity=GauntletSeverity.LOW),
        ]

        counts = result.severity_counts
        assert counts["high"] == 2
        assert counts["low"] == 1

    def test_result_evaluate_pass_fail(self):
        """Should evaluate pass/fail verdict."""
        from aragora.gauntlet.config import GauntletResult, GauntletConfig

        result = GauntletResult()
        result.config = GauntletConfig()
        result.robustness_score = 0.8

        result.evaluate_pass_fail()

        assert result.passed is True

    def test_result_to_receipt(self):
        """Should convert result to decision receipt."""
        from aragora.gauntlet.config import GauntletResult

        result = GauntletResult()
        receipt = result.to_receipt()

        assert receipt is not None


class TestDecisionReceipt:
    """Tests for DecisionReceipt dataclass."""

    def test_receipt_from_result(self):
        """Should create receipt from gauntlet result."""
        from aragora.gauntlet.config import GauntletResult
        from aragora.gauntlet.receipt import DecisionReceipt

        result = GauntletResult()
        result.passed = True
        result.risk_score = 0.2
        result.robustness_score = 0.8

        receipt = DecisionReceipt.from_gauntlet_result(result)

        assert receipt is not None

    def test_receipt_to_markdown(self):
        """Should convert receipt to markdown."""
        from aragora.gauntlet.config import GauntletResult
        from aragora.gauntlet.receipt import DecisionReceipt

        result = GauntletResult()
        result.passed = True
        receipt = DecisionReceipt.from_gauntlet_result(result)

        md = receipt.to_markdown()
        assert isinstance(md, str)


class TestRiskHeatmap:
    """Tests for RiskHeatmap."""

    def test_heatmap_creation(self):
        """Should create RiskHeatmap."""
        from aragora.gauntlet.heatmap import RiskHeatmap

        heatmap = RiskHeatmap()
        assert heatmap is not None

    def test_heatmap_add_cell(self):
        """Should add cells to heatmap."""
        from aragora.gauntlet.heatmap import RiskHeatmap, HeatmapCell

        heatmap = RiskHeatmap()
        cell = HeatmapCell(
            category="Security",
            severity="medium",
            count=3,
        )
        heatmap.cells.append(cell)

        assert len(heatmap.cells) == 1

    def test_heatmap_cell_creation(self):
        """Should create HeatmapCell."""
        from aragora.gauntlet.heatmap import HeatmapCell

        cell = HeatmapCell(
            category="Compliance",
            severity="high",
            count=5,
        )

        assert cell.category == "Compliance"
        assert cell.count == 5


class TestGauntletRunner:
    """Tests for GauntletRunner."""

    def test_runner_instantiation(self):
        """Should instantiate GauntletRunner."""
        from aragora.gauntlet.runner import GauntletRunner
        from aragora.gauntlet.config import GauntletConfig

        config = GauntletConfig(attack_rounds=2)
        runner = GauntletRunner(config)

        assert runner is not None
        assert runner.config.attack_rounds == 2

    def test_runner_with_agents(self):
        """Should instantiate runner with agents."""
        from aragora.gauntlet.runner import GauntletRunner
        from aragora.gauntlet.config import GauntletConfig

        config = GauntletConfig(
            attack_rounds=3,
            agents=["claude", "gpt-4"],
        )
        runner = GauntletRunner(config)

        assert "claude" in runner.config.agents
        assert "gpt-4" in runner.config.agents


class TestGauntletIntegration:
    """Integration tests for Gauntlet module."""

    def test_imports_work(self):
        """Should successfully import from gauntlet modules."""
        from aragora.gauntlet.config import (
            GauntletConfig,
            AttackCategory,
            ProbeCategory,
            GauntletFinding,
            GauntletResult,
            PassFailCriteria,
        )
        from aragora.gauntlet.runner import GauntletRunner
        from aragora.gauntlet.receipt import DecisionReceipt
        from aragora.gauntlet.heatmap import RiskHeatmap, HeatmapCell
        from aragora.gauntlet.types import (
            Verdict,
            GauntletSeverity,
            GauntletPhase,
        )

        # All imports should work
        assert GauntletConfig is not None
        assert AttackCategory is not None
        assert ProbeCategory is not None
        assert GauntletFinding is not None
        assert GauntletResult is not None
        assert GauntletRunner is not None
        assert DecisionReceipt is not None
        assert RiskHeatmap is not None
        assert HeatmapCell is not None

    def test_config_to_runner_to_result_flow(self):
        """Should create config, runner, and mock result."""
        from aragora.gauntlet.config import (
            GauntletConfig,
            GauntletResult,
            AttackCategory,
        )
        from aragora.gauntlet.runner import GauntletRunner

        # Create config
        config = GauntletConfig(
            attack_rounds=2,
            attack_categories=[AttackCategory.SECURITY],
        )

        # Create runner
        runner = GauntletRunner(config)

        # Create mock result (not running actual gauntlet)
        result = GauntletResult()
        result.config = config
        result.robustness_score = 0.8
        result.evaluate_pass_fail()

        # Convert to receipt
        receipt = result.to_receipt()

        assert receipt is not None
        assert runner is not None
        assert result.passed is True
