"""Tests for User Type Presets."""

from __future__ import annotations

import pytest
from aragora.pipeline.user_presets import (
    CTO_PRESET,
    FOUNDER_PRESET,
    NON_TECHNICAL_PRESET,
    PRESETS,
    TEAM_PRESET,
    UserPreset,
    create_custom_preset,
    get_preset,
    list_presets,
)


class TestPresets:
    def test_four_presets_exist(self):
        assert len(PRESETS) == 4
        assert set(PRESETS.keys()) == {"founder", "cto", "team", "non_technical"}

    def test_founder_fast(self):
        assert FOUNDER_PRESET.interrogation_depth == "minimal"
        assert FOUNDER_PRESET.max_questions == 3
        assert FOUNDER_PRESET.autonomy_level == "metrics_driven"
        assert FOUNDER_PRESET.agent_count == 3
        assert FOUNDER_PRESET.debate_rounds == 2

    def test_cto_balanced(self):
        assert CTO_PRESET.interrogation_depth == "standard"
        assert CTO_PRESET.include_alternatives
        assert CTO_PRESET.include_risk_analysis
        assert CTO_PRESET.agent_count == 5

    def test_team_thorough(self):
        assert TEAM_PRESET.interrogation_depth == "deep"
        assert TEAM_PRESET.max_questions == 8
        assert TEAM_PRESET.autonomy_level == "human_guided"
        assert TEAM_PRESET.consensus_threshold == 0.7

    def test_non_technical_accessible(self):
        assert NON_TECHNICAL_PRESET.explain_questions
        assert NON_TECHNICAL_PRESET.include_rationale
        assert not NON_TECHNICAL_PRESET.include_risk_analysis
        assert "business" in NON_TECHNICAL_PRESET.default_domains


class TestGetPreset:
    def test_valid(self):
        preset = get_preset("founder")
        assert preset.name == "founder"

    def test_invalid(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("nonexistent")


class TestListPresets:
    def test_list_all(self):
        presets = list_presets()
        assert len(presets) == 4
        assert all(isinstance(p, UserPreset) for p in presets)


class TestCustomPreset:
    def test_create_from_base(self):
        custom = create_custom_preset("my_config", base="cto", max_questions=10)
        assert custom.name == "my_config"
        assert custom.max_questions == 10
        assert custom.include_alternatives  # Inherited from CTO

    def test_override_autonomy(self):
        custom = create_custom_preset(
            "auto_builder", base="founder", autonomy_level="fully_autonomous"
        )
        assert custom.autonomy_level == "fully_autonomous"


class TestToConfig:
    def test_pipeline_config(self):
        config = CTO_PRESET.to_pipeline_config()
        assert config["interrogation"]["depth"] == "standard"
        assert config["autonomy"]["level"] == "propose_and_approve"
        assert config["output"]["rationale"] is True
        assert config["debate"]["agent_count"] == 5
        assert "technical" in config["domains"]

    def test_founder_minimal_config(self):
        config = FOUNDER_PRESET.to_pipeline_config()
        assert config["interrogation"]["max_questions"] == 3
        assert config["output"]["format"] == "minimal"
        assert config["debate"]["rounds"] == 2
