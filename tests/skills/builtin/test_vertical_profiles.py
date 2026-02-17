"""Tests for vertical profile skills."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.skills.builtin.vertical_profiles import (
    ComplianceSOXSkill,
    FinancialAuditSkill,
    FinancialRiskSkill,
    HealthcareClinicalSkill,
    HealthcareHIPAASkill,
    LegalContractSkill,
    LegalDueDiligenceSkill,
    VerticalConfig,
    VerticalProfileSkill,
    VERTICAL_SKILLS,
    get_vertical_skill,
    list_vertical_profiles,
)
from aragora.skills.base import SkillResult, SkillStatus


class TestVerticalConfig:
    """Tests for VerticalConfig dataclass."""

    def test_basic_creation(self):
        config = VerticalConfig(
            name="test",
            display_name="Test Profile",
            weight_profile="test_profile",
            description="A test profile",
        )
        assert config.name == "test"
        assert config.display_name == "Test Profile"
        assert config.compliance_frameworks == []
        assert config.required_dimensions == []
        assert config.recommended_agents == []
        assert config.arena_overrides == {}
        assert config.km_adapters == []
        assert config.preset_base == "enterprise"

    def test_full_creation(self):
        config = VerticalConfig(
            name="full",
            display_name="Full Profile",
            weight_profile="full_weights",
            description="Full test",
            compliance_frameworks=["sox", "hipaa"],
            required_dimensions=["accuracy", "safety"],
            recommended_agents=["claude", "gpt4"],
            arena_overrides={"enable_provenance": True},
            km_adapters=["compliance"],
            preset_base="startup",
        )
        assert config.compliance_frameworks == ["sox", "hipaa"]
        assert config.required_dimensions == ["accuracy", "safety"]
        assert config.preset_base == "startup"

    @patch("aragora.debate.presets.get_preset")
    def test_to_arena_config(self, mock_get_preset):
        mock_get_preset.return_value = {"rounds": 3, "consensus": "majority"}
        config = VerticalConfig(
            name="test",
            display_name="Test",
            weight_profile="test_wp",
            description="Test",
            arena_overrides={"enable_provenance": True},
        )
        result = config.to_arena_config()
        mock_get_preset.assert_called_once_with("enterprise")
        assert result["rounds"] == 3
        assert result["enable_provenance"] is True
        assert result["_vertical_profile"] == "test"
        assert result["_weight_profile"] == "test_wp"

    @patch("aragora.debate.presets.get_preset", side_effect=ImportError)
    def test_to_arena_config_missing_presets(self, mock_get_preset):
        config = VerticalConfig(
            name="test",
            display_name="Test",
            weight_profile="test_wp",
            description="Test",
        )
        with pytest.raises(ImportError):
            config.to_arena_config()


class TestVerticalProfileSkill:
    """Tests for the base VerticalProfileSkill."""

    def test_manifest(self):
        config = VerticalConfig(
            name="test_vertical",
            display_name="Test",
            weight_profile="tw",
            description="Test description",
        )
        skill = VerticalProfileSkill(config)
        manifest = skill.manifest
        assert manifest.name == "vertical_test_vertical"
        assert manifest.version == "1.0.0"
        assert manifest.description == "Test description"
        assert "vertical" in manifest.tags
        assert "test_vertical" in manifest.tags

    @pytest.mark.asyncio
    async def test_info_action(self):
        config = VerticalConfig(
            name="test",
            display_name="Test Display",
            weight_profile="tw",
            description="Test",
            compliance_frameworks=["sox"],
            required_dimensions=["accuracy"],
            recommended_agents=["claude"],
        )
        skill = VerticalProfileSkill(config)
        result = await skill.execute({"action": "info"})
        assert result.data["name"] == "test"
        assert result.data["display_name"] == "Test Display"
        assert result.data["compliance_frameworks"] == ["sox"]

    @pytest.mark.asyncio
    async def test_default_action_is_info(self):
        config = VerticalConfig(
            name="test", display_name="T", weight_profile="tw", description="T"
        )
        skill = VerticalProfileSkill(config)
        result = await skill.execute({})
        assert "name" in result.data

    @pytest.mark.asyncio
    @patch("aragora.debate.presets.get_preset")
    async def test_configure_action(self, mock_get_preset):
        mock_get_preset.return_value = {"rounds": 3}
        config = VerticalConfig(
            name="test",
            display_name="Test",
            weight_profile="tw",
            description="T",
            compliance_frameworks=["hipaa"],
        )
        skill = VerticalProfileSkill(config)
        result = await skill.execute({"action": "configure"})
        assert result.data["applied"] is True
        assert result.data["profile"] == "test"
        assert result.data["compliance"] == ["hipaa"]

    @pytest.mark.asyncio
    @patch("aragora.debate.presets.get_preset", side_effect=ImportError("no presets"))
    async def test_configure_handles_import_error(self, mock_get_preset):
        config = VerticalConfig(
            name="test", display_name="T", weight_profile="tw", description="T"
        )
        skill = VerticalProfileSkill(config)
        result = await skill.execute({"action": "configure"})
        assert result.status == SkillStatus.FAILURE

    @pytest.mark.asyncio
    async def test_validate_action_weight_profile_found(self):
        config = VerticalConfig(
            name="test",
            display_name="T",
            weight_profile="healthcare_hipaa",
            description="T",
        )
        skill = VerticalProfileSkill(config)
        result = await skill.execute({"action": "validate"})
        # The result depends on whether WEIGHT_PROFILES exists
        assert "checks" in result.data
        assert "valid" in result.data

    @pytest.mark.asyncio
    async def test_validate_action_with_compliance(self):
        config = VerticalConfig(
            name="test",
            display_name="T",
            weight_profile="tw",
            description="T",
            compliance_frameworks=["sox"],
        )
        skill = VerticalProfileSkill(config)
        result = await skill.execute({"action": "validate"})
        checks = result.data["checks"]
        check_names = [c["check"] for c in checks]
        assert "weight_profile" in check_names
        assert "compliance_generator" in check_names


class TestConcreteSkills:
    """Tests for all 7 concrete vertical profile skills."""

    @pytest.mark.parametrize(
        "cls,expected_name,expected_frameworks",
        [
            (HealthcareHIPAASkill, "healthcare_hipaa", ["hipaa", "hitech"]),
            (HealthcareClinicalSkill, "healthcare_clinical", ["hipaa"]),
            (FinancialAuditSkill, "financial_audit", ["sox", "gaap"]),
            (FinancialRiskSkill, "financial_risk", ["sox"]),
            (LegalContractSkill, "legal_contract", []),
            (LegalDueDiligenceSkill, "legal_due_diligence", []),
            (ComplianceSOXSkill, "compliance_sox", ["sox", "coso"]),
        ],
    )
    def test_skill_config(self, cls, expected_name, expected_frameworks):
        skill = cls()
        assert skill._config.name == expected_name
        assert skill._config.compliance_frameworks == expected_frameworks
        assert skill._config.weight_profile == expected_name
        assert len(skill._config.description) > 10
        assert len(skill._config.recommended_agents) >= 2

    @pytest.mark.parametrize(
        "cls",
        [
            HealthcareHIPAASkill,
            HealthcareClinicalSkill,
            FinancialAuditSkill,
            FinancialRiskSkill,
            LegalContractSkill,
            LegalDueDiligenceSkill,
            ComplianceSOXSkill,
        ],
    )
    def test_skill_manifest(self, cls):
        skill = cls()
        manifest = skill.manifest
        assert manifest.name.startswith("vertical_")
        assert "vertical" in manifest.tags

    def test_healthcare_hipaa_overrides(self):
        skill = HealthcareHIPAASkill()
        overrides = skill._config.arena_overrides
        assert overrides.get("enable_compliance_artifacts") is True
        assert overrides.get("enable_provenance") is True
        assert overrides.get("enable_receipt_generation") is True

    def test_financial_risk_overrides(self):
        skill = FinancialRiskSkill()
        overrides = skill._config.arena_overrides
        assert overrides.get("enable_trickster") is True

    def test_legal_due_diligence_overrides(self):
        skill = LegalDueDiligenceSkill()
        overrides = skill._config.arena_overrides
        assert overrides.get("enable_trickster") is True
        assert overrides.get("enable_knowledge_extraction") is True

    def test_compliance_sox_overrides(self):
        skill = ComplianceSOXSkill()
        overrides = skill._config.arena_overrides
        assert overrides.get("enable_bead_tracking") is True
        assert overrides.get("enable_compliance_artifacts") is True


class TestRegistry:
    """Tests for the VERTICAL_SKILLS registry."""

    def test_registry_count(self):
        assert len(VERTICAL_SKILLS) == 7

    def test_registry_keys(self):
        expected = {
            "healthcare_hipaa",
            "healthcare_clinical",
            "financial_audit",
            "financial_risk",
            "legal_contract",
            "legal_due_diligence",
            "compliance_sox",
        }
        assert set(VERTICAL_SKILLS.keys()) == expected

    def test_get_vertical_skill(self):
        skill = get_vertical_skill("healthcare_hipaa")
        assert isinstance(skill, HealthcareHIPAASkill)

    def test_get_vertical_skill_unknown(self):
        with pytest.raises(ValueError, match="Unknown vertical profile"):
            get_vertical_skill("nonexistent")

    def test_list_vertical_profiles(self):
        profiles = list_vertical_profiles()
        assert len(profiles) == 7
        names = {p["name"] for p in profiles}
        assert "healthcare_hipaa" in names
        assert "financial_audit" in names
        for p in profiles:
            assert "display_name" in p
            assert "description" in p
            assert "compliance" in p


class TestRegistration:
    """Tests for registration in builtin __init__."""

    def test_builtin_register_includes_verticals(self):
        from aragora.skills.builtin import register_skills

        skills = register_skills()
        vertical_skills = [s for s in skills if hasattr(s, "_config")]
        assert len(vertical_skills) >= 7
        names = {s._config.name for s in vertical_skills}
        assert "healthcare_hipaa" in names
        assert "compliance_sox" in names
