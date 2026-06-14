"""
Tests for aragora.gauntlet.config module.

Tests GauntletConfig, AttackCategory, ProbeCategory, and PassFailCriteria.
"""

import pytest

from aragora.gauntlet.config import (
    GauntletConfig,
    AttackCategory,
    ProbeCategory,
    PassFailCriteria,
)


class TestAttackCategory:
    """Tests for AttackCategory enum."""

    def test_security_categories(self):
        """Security-related attack categories exist."""
        assert AttackCategory.SECURITY.value == "security"
        assert AttackCategory.INJECTION.value == "injection"
        assert AttackCategory.PRIVILEGE_ESCALATION.value == "privilege_escalation"
        assert AttackCategory.ADVERSARIAL_INPUT.value == "adversarial_input"

    def test_compliance_categories(self):
        """Compliance-related attack categories exist."""
        assert AttackCategory.COMPLIANCE.value == "compliance"
        assert AttackCategory.GDPR.value == "gdpr"
        assert AttackCategory.HIPAA.value == "hipaa"
        assert AttackCategory.AI_ACT.value == "ai_act"
        assert AttackCategory.REGULATORY_VIOLATION.value == "regulatory_violation"

    def test_architecture_categories(self):
        """Architecture-related attack categories exist."""
        assert AttackCategory.ARCHITECTURE.value == "architecture"
        assert AttackCategory.SCALABILITY.value == "scalability"
        assert AttackCategory.PERFORMANCE.value == "performance"
        assert AttackCategory.RESOURCE_EXHAUSTION.value == "resource_exhaustion"

    def test_logic_categories(self):
        """Logic-related attack categories exist."""
        assert AttackCategory.LOGIC.value == "logic"
        assert AttackCategory.LOGICAL_FALLACY.value == "logical_fallacy"
        assert AttackCategory.EDGE_CASES.value == "edge_cases"
        assert AttackCategory.ASSUMPTIONS.value == "assumptions"
        assert AttackCategory.STAKEHOLDER_CONFLICT.value == "stakeholder_conflict"

    def test_operational_categories(self):
        """Operational attack categories exist."""
        assert AttackCategory.OPERATIONAL.value == "operational"
        assert AttackCategory.DEPENDENCY_FAILURE.value == "dependency_failure"
        assert AttackCategory.RACE_CONDITION.value == "race_condition"

    def test_category_from_string(self):
        """AttackCategory can be created from string value."""
        assert AttackCategory("security") == AttackCategory.SECURITY
        assert AttackCategory("gdpr") == AttackCategory.GDPR


class TestProbeCategory:
    """Tests for ProbeCategory enum."""

    def test_all_probe_categories_exist(self):
        """All expected probe categories exist."""
        categories = [
            ("CONTRADICTION", "contradiction"),
            ("HALLUCINATION", "hallucination"),
            ("SYCOPHANCY", "sycophancy"),
            ("PERSISTENCE", "persistence"),
            ("CALIBRATION", "calibration"),
            ("REASONING_DEPTH", "reasoning_depth"),
            ("EDGE_CASE", "edge_case"),
            ("INSTRUCTION_INJECTION", "instruction_injection"),
            ("CAPABILITY_EXAGGERATION", "capability_exaggeration"),
        ]

        for name, value in categories:
            assert getattr(ProbeCategory, name).value == value

    def test_category_from_string(self):
        """ProbeCategory can be created from string value."""
        assert ProbeCategory("contradiction") == ProbeCategory.CONTRADICTION
        assert ProbeCategory("hallucination") == ProbeCategory.HALLUCINATION


class TestPassFailCriteria:
    """Tests for PassFailCriteria dataclass."""

    def test_default_values(self):
        """PassFailCriteria has expected default values."""
        criteria = PassFailCriteria()

        assert criteria.max_critical_findings == 0
        assert criteria.max_high_findings == 2
        assert criteria.min_robustness_score == 0.7
        assert criteria.min_verification_coverage == 0.0
        assert criteria.require_formal_verification is False
        assert criteria.require_consensus is False
        assert criteria.min_confidence == 0.5

    def test_strict_factory(self):
        """PassFailCriteria.strict() returns strict criteria."""
        criteria = PassFailCriteria.strict()

        assert criteria.max_critical_findings == 0
        assert criteria.max_high_findings == 0
        assert criteria.min_robustness_score == 0.85
        assert criteria.min_verification_coverage == 0.5
        assert criteria.require_formal_verification is True
        assert criteria.require_consensus is True
        assert criteria.min_confidence == 0.7

    def test_lenient_factory(self):
        """PassFailCriteria.lenient() returns lenient criteria."""
        criteria = PassFailCriteria.lenient()

        assert criteria.max_critical_findings == 1
        assert criteria.max_high_findings == 5
        assert criteria.min_robustness_score == 0.5
        assert criteria.min_verification_coverage == 0.0
        assert criteria.require_consensus is False
        assert criteria.min_confidence == 0.5


class TestGauntletConfigDefaults:
    """Tests for GauntletConfig default values."""

    def test_default_name(self):
        """Config has default name."""
        config = GauntletConfig()
        assert config.name == "Gauntlet Validation"

    def test_default_agents(self):
        """Config has default agents."""
        config = GauntletConfig()
        assert config.agents == ["anthropic-api", "openai-api"]

    def test_default_attack_categories(self):
        """Config has default attack categories."""
        config = GauntletConfig()
        assert AttackCategory.SECURITY in config.attack_categories
        assert AttackCategory.LOGIC in config.attack_categories
        assert AttackCategory.ARCHITECTURE in config.attack_categories

    def test_default_probe_categories(self):
        """Config has default probe categories."""
        config = GauntletConfig()
        assert ProbeCategory.CONTRADICTION in config.probe_categories
        assert ProbeCategory.HALLUCINATION in config.probe_categories
        assert ProbeCategory.SYCOPHANCY in config.probe_categories

    def test_default_thresholds(self):
        """Config has expected default thresholds."""
        config = GauntletConfig()
        assert config.critical_threshold == 0
        assert config.high_threshold == 2
        assert config.vulnerability_rate_threshold == 0.2
        assert config.consensus_threshold == 0.7
        assert config.robustness_threshold == 0.6

    def test_default_timeouts(self):
        """Config has expected default timeouts."""
        config = GauntletConfig()
        assert config.timeout_seconds == 300
        assert config.attack_timeout == 60
        assert config.probe_timeout == 30
        assert config.scenario_timeout == 120

    def test_default_output_formats(self):
        """Config has expected default output formats."""
        config = GauntletConfig()
        assert config.output_formats == ["json", "md"]


class TestGauntletConfigValidation:
    """Tests for GauntletConfig __post_init__ validation."""

    def test_empty_agents_raises(self):
        """Config raises ValueError for empty agents list."""
        with pytest.raises(ValueError, match="At least one agent required"):
            GauntletConfig(agents=[])

    def test_invalid_max_agents_raises(self):
        """Config raises ValueError for max_agents < 1."""
        with pytest.raises(ValueError, match="max_agents must be >= 1"):
            GauntletConfig(max_agents=0)

    def test_invalid_attack_rounds_raises(self):
        """Config raises ValueError for attack_rounds < 1."""
        with pytest.raises(ValueError, match="attack_rounds must be >= 1"):
            GauntletConfig(attack_rounds=0)

    def test_invalid_timeout_raises(self):
        """Config raises ValueError for timeout_seconds < 1."""
        with pytest.raises(ValueError, match="timeout_seconds must be >= 1"):
            GauntletConfig(timeout_seconds=0)

    def test_invalid_vulnerability_rate_raises(self):
        """Config raises ValueError for invalid vulnerability_rate_threshold."""
        with pytest.raises(ValueError, match="vulnerability_rate_threshold must be 0-1"):
            GauntletConfig(vulnerability_rate_threshold=1.5)

        with pytest.raises(ValueError, match="vulnerability_rate_threshold must be 0-1"):
            GauntletConfig(vulnerability_rate_threshold=-0.1)

    def test_invalid_consensus_threshold_raises(self):
        """Config raises ValueError for invalid consensus_threshold."""
        with pytest.raises(ValueError, match="consensus_threshold must be 0-1"):
            GauntletConfig(consensus_threshold=1.5)

    def test_invalid_robustness_threshold_raises(self):
        """Config raises ValueError for invalid robustness_threshold."""
        with pytest.raises(ValueError, match="robustness_threshold must be 0-1"):
            GauntletConfig(robustness_threshold=-0.1)

    def test_scenario_analysis_sync(self):
        """Config syncs enable_scenario_analysis with run_scenario_matrix."""
        config = GauntletConfig(enable_scenario_analysis=False)
        assert config.enable_scenario_analysis is False
        assert config.run_scenario_matrix is False

    def test_scenario_presets_populated(self):
        """Config populates scenario_presets from scenario_preset."""
        config = GauntletConfig(scenario_preset="scale")
        assert "scale" in config.scenario_presets


class TestGauntletConfigSerialization:
    """Tests for GauntletConfig serialization methods."""

    def test_to_dict_includes_all_fields(self):
        """to_dict() includes all configuration fields."""
        config = GauntletConfig(name="Test Config")
        data = config.to_dict()

        assert data["name"] == "Test Config"
        assert "attack_categories" in data
        assert "probe_categories" in data
        assert "agents" in data
        assert "criteria" in data
        assert "timeout_seconds" in data

    def test_to_dict_converts_enums(self):
        """to_dict() converts enum values to strings."""
        config = GauntletConfig(
            attack_categories=[AttackCategory.SECURITY],
            probe_categories=[ProbeCategory.HALLUCINATION],
        )
        data = config.to_dict()

        assert data["attack_categories"] == ["security"]
        assert data["probe_categories"] == ["hallucination"]

    def test_to_dict_includes_criteria(self):
        """to_dict() includes PassFailCriteria as dict."""
        config = GauntletConfig(criteria=PassFailCriteria.strict())
        data = config.to_dict()

        assert data["criteria"]["max_critical_findings"] == 0
        assert data["criteria"]["max_high_findings"] == 0
        assert data["criteria"]["require_formal_verification"] is True

    def test_from_dict_creates_config(self):
        """from_dict() creates valid GauntletConfig."""
        data = {
            "name": "Restored Config",
            "agents": ["test-agent"],
            "attack_categories": ["security", "logic"],
            "probe_categories": ["hallucination"],
            "attack_rounds": 3,
        }
        config = GauntletConfig.from_dict(data)

        assert config.name == "Restored Config"
        assert config.agents == ["test-agent"]
        assert AttackCategory.SECURITY in config.attack_categories
        assert AttackCategory.LOGIC in config.attack_categories
        assert ProbeCategory.HALLUCINATION in config.probe_categories
        assert config.attack_rounds == 3

    def test_from_dict_converts_criteria(self):
        """from_dict() converts criteria dict to PassFailCriteria."""
        data = {
            "agents": ["test"],
            "criteria": {
                "max_critical_findings": 1,
                "max_high_findings": 3,
                "min_robustness_score": 0.6,
                "min_verification_coverage": 0.0,
                "require_formal_verification": False,
                "require_consensus": True,
                "min_confidence": 0.6,
            },
        }
        config = GauntletConfig.from_dict(data)

        assert config.criteria.max_critical_findings == 1
        assert config.criteria.max_high_findings == 3
        assert config.criteria.require_consensus is True

    def test_roundtrip_serialization(self):
        """to_dict() and from_dict() roundtrip preserves config."""
        original = GauntletConfig(
            name="Roundtrip Test",
            attack_categories=[AttackCategory.GDPR, AttackCategory.HIPAA],
            probe_categories=[ProbeCategory.CALIBRATION],
            agents=["test-agent-1", "test-agent-2"],
            attack_rounds=5,
            critical_threshold=1,
            criteria=PassFailCriteria(max_critical_findings=2),
        )

        data = original.to_dict()
        restored = GauntletConfig.from_dict(data)

        assert restored.name == original.name
        assert restored.agents == original.agents
        assert restored.attack_rounds == original.attack_rounds
        assert restored.critical_threshold == original.critical_threshold
        assert restored.criteria.max_critical_findings == original.criteria.max_critical_findings


class TestGauntletConfigFactoryMethods:
    """Tests for GauntletConfig factory methods."""

    def test_security_focused(self):
        """security_focused() creates security-oriented config."""
        config = GauntletConfig.security_focused()

        assert AttackCategory.SECURITY in config.attack_categories
        assert AttackCategory.INJECTION in config.attack_categories
        assert AttackCategory.PRIVILEGE_ESCALATION in config.attack_categories
        assert ProbeCategory.INSTRUCTION_INJECTION in config.probe_categories
        assert config.attack_rounds == 3
        assert config.critical_threshold == 0
        assert config.high_threshold == 0

    def test_compliance_focused(self):
        """compliance_focused() creates compliance-oriented config."""
        config = GauntletConfig.compliance_focused()

        assert AttackCategory.COMPLIANCE in config.attack_categories
        assert AttackCategory.GDPR in config.attack_categories
        assert AttackCategory.HIPAA in config.attack_categories
        assert AttackCategory.AI_ACT in config.attack_categories
        assert ProbeCategory.HALLUCINATION in config.probe_categories
        assert ProbeCategory.CALIBRATION in config.probe_categories
        assert config.run_scenario_matrix is False

    def test_quick(self):
        """quick() creates minimal config for fast validation."""
        config = GauntletConfig.quick()

        assert len(config.attack_categories) == 2
        assert AttackCategory.SECURITY in config.attack_categories
        assert AttackCategory.LOGIC in config.attack_categories
        assert len(config.probe_categories) == 1
        assert ProbeCategory.CONTRADICTION in config.probe_categories
        assert config.attack_rounds == 1
        assert config.attacks_per_category == 2
        assert config.probes_per_category == 1
        assert config.run_scenario_matrix is False


class TestGauntletConfigCustomization:
    """Tests for customizing GauntletConfig."""

    def test_custom_attack_categories(self):
        """Config accepts custom attack categories."""
        config = GauntletConfig(
            attack_categories=[
                AttackCategory.GDPR,
                AttackCategory.HIPAA,
                AttackCategory.SOX if hasattr(AttackCategory, "SOX") else AttackCategory.COMPLIANCE,
            ]
        )
        assert AttackCategory.GDPR in config.attack_categories
        assert AttackCategory.HIPAA in config.attack_categories

    def test_custom_probe_categories(self):
        """Config accepts custom probe categories."""
        config = GauntletConfig(
            probe_categories=[
                ProbeCategory.REASONING_DEPTH,
                ProbeCategory.CAPABILITY_EXAGGERATION,
            ]
        )
        assert len(config.probe_categories) == 2
        assert ProbeCategory.REASONING_DEPTH in config.probe_categories

    def test_custom_agents(self):
        """Config accepts custom agent list."""
        config = GauntletConfig(agents=["custom-agent-1", "custom-agent-2", "custom-agent-3"])
        assert len(config.agents) == 3
        assert "custom-agent-1" in config.agents

    def test_custom_thresholds(self):
        """Config accepts custom thresholds."""
        config = GauntletConfig(
            critical_threshold=2,
            high_threshold=5,
            vulnerability_rate_threshold=0.3,
            robustness_threshold=0.5,
        )
        assert config.critical_threshold == 2
        assert config.high_threshold == 5
        assert config.vulnerability_rate_threshold == 0.3
        assert config.robustness_threshold == 0.5

    def test_custom_timeouts(self):
        """Config accepts custom timeout values."""
        config = GauntletConfig(
            timeout_seconds=600,
            attack_timeout=120,
            probe_timeout=60,
            scenario_timeout=180,
        )
        assert config.timeout_seconds == 600
        assert config.attack_timeout == 120
        assert config.probe_timeout == 60
        assert config.scenario_timeout == 180

    def test_output_configuration(self):
        """Config accepts output configuration."""
        config = GauntletConfig(
            output_dir="/tmp/gauntlet-results",
            output_formats=["json", "md", "html"],
            save_artifacts=True,
            generate_receipt=False,
        )
        assert config.output_dir == "/tmp/gauntlet-results"
        assert "html" in config.output_formats
        assert config.save_artifacts is True
        assert config.generate_receipt is False


class TestGauntletConfigMetadata:
    """Tests for GauntletConfig metadata fields."""

    def test_template_metadata(self):
        """Config stores template metadata."""
        config = GauntletConfig(
            name="Security Audit Template",
            description="Comprehensive security validation",
            template_id="sec-audit-v1",
            input_type="code",
            domain="security",
            tags=["security", "audit", "code-review"],
        )

        assert config.name == "Security Audit Template"
        assert config.description == "Comprehensive security validation"
        assert config.template_id == "sec-audit-v1"
        assert config.input_type == "code"
        assert config.domain == "security"
        assert "security" in config.tags
        assert "audit" in config.tags

    def test_pipeline_toggles(self):
        """Config respects pipeline toggles."""
        config = GauntletConfig(
            enable_scenario_analysis=True,
            enable_adversarial_probing=False,
            enable_formal_verification=True,
            enable_deep_audit=True,
        )

        assert config.enable_scenario_analysis is True
        assert config.enable_adversarial_probing is False
        assert config.enable_formal_verification is True
        assert config.enable_deep_audit is True
