"""Tests for extended debate presets and config flag wiring."""

from __future__ import annotations

import pytest

from aragora.debate.presets import (
    _PRESETS,
    _PRESET_DESCRIPTIONS,
    apply_preset,
    get_preset,
    get_preset_info,
    list_presets,
)


class TestNewPresets:
    """Test the new preset configurations."""

    def test_visual_preset_exists(self):
        preset = get_preset("visual")
        assert preset["enable_cartographer"] is True
        assert preset["enable_spectator"] is True
        assert preset["enable_position_ledger"] is True
        assert preset["enable_introspection"] is True

    def test_compliance_preset_exists(self):
        preset = get_preset("compliance")
        assert preset["enable_receipt_generation"] is True
        assert preset["enable_provenance"] is True
        assert preset["enable_compliance_artifacts"] is True
        assert preset["enable_privacy_anonymization"] is True

    def test_research_preset_exists(self):
        preset = get_preset("research")
        assert preset["enable_knowledge_extraction"] is True
        assert preset["enable_supermemory"] is True
        assert preset["enable_power_sampling"] is True
        assert preset["enable_debate_forking"] is True
        assert preset["enable_cartographer"] is True
        assert preset["enable_introspection"] is True

    def test_financial_preset_exists(self):
        preset = get_preset("financial")
        assert preset["enable_receipt_generation"] is True
        assert preset["enable_compliance_artifacts"] is True
        assert preset["vertical"] == "financial_audit"

    def test_healthcare_preset_exists(self):
        preset = get_preset("healthcare")
        assert preset["enable_privacy_anonymization"] is True
        assert preset["vertical"] == "healthcare_hipaa"

    def test_all_presets_have_descriptions(self):
        for name in _PRESETS:
            assert name in _PRESET_DESCRIPTIONS, f"Missing description for preset '{name}'"
            assert len(_PRESET_DESCRIPTIONS[name]) > 10

    def test_list_presets_includes_new(self):
        names = list_presets()
        assert "visual" in names
        assert "compliance" in names
        assert "research" in names
        assert "financial" in names

    def test_get_preset_info_new_presets(self):
        for name in ("visual", "compliance", "research", "financial"):
            info = get_preset_info(name)
            assert info["name"] == name
            assert info["description"]
            assert info["flags"]

    def test_apply_preset_with_overrides(self):
        merged = apply_preset("visual", overrides={"enable_cartographer": False})
        assert merged["enable_cartographer"] is False
        assert merged["enable_spectator"] is True

    def test_apply_preset_kwargs_override(self):
        merged = apply_preset("research", enable_power_sampling=False)
        assert merged["enable_power_sampling"] is False
        assert merged["enable_supermemory"] is True

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError, match="Unknown preset"):
            get_preset("nonexistent")

    def test_preset_returns_copy(self):
        p1 = get_preset("visual")
        p2 = get_preset("visual")
        p1["extra"] = True
        assert "extra" not in p2


class TestCartographerConfig:
    """Test cartographer config flag wiring."""

    def test_arena_config_accepts_cartographer_flag(self):
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig(enable_cartographer=False)
        assert config.enable_cartographer is False

    def test_arena_config_cartographer_default_true(self):
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig()
        assert config.enable_cartographer is True

    def test_arena_config_introspection_default_true(self):
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig()
        assert config.enable_introspection is True

    def test_arena_config_accepts_introspection_flag(self):
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig(enable_introspection=False)
        assert config.enable_introspection is False


class TestCLIPresetChoices:
    """Test CLI parser exposes new presets."""

    def test_preset_flag_accepts_new_presets(self):
        from aragora.cli.parser import build_parser

        parser = build_parser()
        for preset_name in ("visual", "compliance", "research", "healthcare", "financial"):
            args = parser.parse_args(["ask", "test question", "--preset", preset_name])
            assert args.preset == preset_name

    def test_cartographer_flag(self):
        from aragora.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["ask", "test question", "--no-cartographer"])
        assert args.enable_cartographer is False

    def test_introspection_flag(self):
        from aragora.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["ask", "test question", "--no-introspection"])
        assert args.enable_introspection is False

    def test_auto_execute_flag(self):
        from aragora.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["ask", "test question", "--auto-execute"])
        assert args.auto_execute is True

    def test_defaults_cartographer_on(self):
        from aragora.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["ask", "test question"])
        assert args.enable_cartographer is True

    def test_defaults_introspection_on(self):
        from aragora.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["ask", "test question"])
        assert args.enable_introspection is True
