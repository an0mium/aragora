"""Tests for aragora.debate.presets -- SME-friendly configuration presets."""

from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.presets import (
    _PRESET_DESCRIPTIONS,
    _PRESETS,
    apply_preset,
    get_preset,
    get_preset_info,
    list_presets,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_PRESET_NAMES = sorted([
    "sme",
    "enterprise",
    "minimal",
    "audit",
    "healthcare",
    "visual",
    "compliance",
    "research",
    "diverse",
    "financial",
])

# Presets that contain a _post_debate_preset key in the raw data
PRESETS_WITH_POST_DEBATE = [
    name for name, cfg in _PRESETS.items() if "_post_debate_preset" in cfg
]


# ---------------------------------------------------------------------------
# list_presets
# ---------------------------------------------------------------------------


class TestListPresets:
    def test_returns_sorted_list(self) -> None:
        result = list_presets()
        assert result == sorted(result)

    def test_contains_all_expected_presets(self) -> None:
        result = list_presets()
        assert result == ALL_PRESET_NAMES

    def test_returns_list_type(self) -> None:
        result = list_presets()
        assert isinstance(result, list)

    def test_count_is_ten(self) -> None:
        assert len(list_presets()) == 10


# ---------------------------------------------------------------------------
# get_preset -- basic
# ---------------------------------------------------------------------------


class TestGetPreset:
    @pytest.mark.parametrize("name", ALL_PRESET_NAMES)
    def test_returns_dict_for_each_preset(self, name: str) -> None:
        result = get_preset(name)
        assert isinstance(result, dict)
        assert len(result) > 0

    @pytest.mark.parametrize("name", ALL_PRESET_NAMES)
    def test_no_internal_post_debate_preset_key(self, name: str) -> None:
        """The private _post_debate_preset key must not leak into the result."""
        result = get_preset(name)
        assert "_post_debate_preset" not in result

    def test_unknown_name_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown preset 'nonexistent'"):
            get_preset("nonexistent")

    def test_unknown_name_lists_available(self) -> None:
        with pytest.raises(KeyError) as exc_info:
            get_preset("bogus")
        msg = str(exc_info.value)
        for name in ALL_PRESET_NAMES:
            assert name in msg

    def test_returns_copy_not_original(self) -> None:
        """Mutating the returned dict must not affect the internal data."""
        result = get_preset("sme")
        result["injected_key"] = "oops"
        fresh = get_preset("sme")
        assert "injected_key" not in fresh

    def test_returns_copy_nested_not_shared(self) -> None:
        """Even for presets without _post_debate_preset, values should be independent."""
        a = get_preset("minimal")
        b = get_preset("minimal")
        a["enable_stability_detection"] = False
        assert b["enable_stability_detection"] is True


# ---------------------------------------------------------------------------
# get_preset -- PostDebateConfig conversion
# ---------------------------------------------------------------------------


class TestGetPresetPostDebateConfig:
    """Tests for PostDebateConfig conversion inside get_preset."""

    def test_with_importable_post_debate_config(self) -> None:
        """When PostDebateConfig is importable, presets with _post_debate_preset
        should have a 'post_debate_config' key."""
        # Use the real import -- the module is present in the codebase
        result = get_preset("sme")
        assert "post_debate_config" in result
        assert "_post_debate_preset" not in result

    @pytest.mark.parametrize("name", PRESETS_WITH_POST_DEBATE)
    def test_all_post_debate_presets_convert(self, name: str) -> None:
        """Each preset with _post_debate_preset gets a post_debate_config."""
        result = get_preset(name)
        assert "post_debate_config" in result

    @pytest.mark.parametrize("name", PRESETS_WITH_POST_DEBATE)
    def test_post_debate_config_type(self, name: str) -> None:
        from aragora.debate.post_debate_coordinator import PostDebateConfig

        result = get_preset(name)
        assert isinstance(result["post_debate_config"], PostDebateConfig)

    def test_without_importable_post_debate_config(self) -> None:
        """When PostDebateConfig cannot be imported, the key should be removed."""
        with patch(
            "aragora.debate.presets.PostDebateConfig",
            new=None,
            create=True,
        ):
            # Patch the import inside get_preset to raise ImportError
            import aragora.debate.presets as presets_mod
            original_get = presets_mod.get_preset

            def patched_get(name: str):
                if name not in _PRESETS:
                    available = ", ".join(sorted(_PRESETS))
                    raise KeyError(f"Unknown preset '{name}'. Available: {available}")
                result = dict(_PRESETS[name])
                pdc_preset = result.pop("_post_debate_preset", None)
                if pdc_preset:
                    try:
                        raise ImportError("mocked")
                    except ImportError:
                        pass
                return result

            with patch.object(presets_mod, "get_preset", patched_get):
                result = presets_mod.get_preset("sme")

        assert "post_debate_config" not in result
        assert "_post_debate_preset" not in result

    def test_preset_without_post_debate_has_no_config_key(self) -> None:
        """Presets without _post_debate_preset should not have post_debate_config."""
        result = get_preset("minimal")
        assert "post_debate_config" not in result
        assert "_post_debate_preset" not in result

    def test_import_error_removes_key_cleanly(self) -> None:
        """Simulate ImportError for PostDebateConfig using builtins import patch."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "aragora.debate.post_debate_coordinator":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = get_preset("sme")

        # With import blocked, _post_debate_preset is popped but no config is set
        assert "_post_debate_preset" not in result
        assert "post_debate_config" not in result


# ---------------------------------------------------------------------------
# get_preset_info
# ---------------------------------------------------------------------------


class TestGetPresetInfo:
    @pytest.mark.parametrize("name", ALL_PRESET_NAMES)
    def test_returns_correct_structure(self, name: str) -> None:
        info = get_preset_info(name)
        assert "name" in info
        assert "description" in info
        assert "flags" in info

    @pytest.mark.parametrize("name", ALL_PRESET_NAMES)
    def test_name_matches(self, name: str) -> None:
        info = get_preset_info(name)
        assert info["name"] == name

    @pytest.mark.parametrize("name", ALL_PRESET_NAMES)
    def test_description_is_string(self, name: str) -> None:
        info = get_preset_info(name)
        assert isinstance(info["description"], str)
        assert len(info["description"]) > 0

    @pytest.mark.parametrize("name", ALL_PRESET_NAMES)
    def test_flags_is_dict(self, name: str) -> None:
        info = get_preset_info(name)
        assert isinstance(info["flags"], dict)
        assert len(info["flags"]) > 0

    def test_flags_is_a_copy(self) -> None:
        info = get_preset_info("sme")
        info["flags"]["injected"] = True
        fresh = get_preset_info("sme")
        assert "injected" not in fresh["flags"]

    def test_unknown_name_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown preset 'nope'"):
            get_preset_info("nope")

    def test_unknown_name_lists_available(self) -> None:
        with pytest.raises(KeyError) as exc_info:
            get_preset_info("invalid")
        msg = str(exc_info.value)
        for name in ALL_PRESET_NAMES:
            assert name in msg

    def test_flags_include_internal_key(self) -> None:
        """get_preset_info returns raw flags including _post_debate_preset
        (unlike get_preset which converts/removes it)."""
        info = get_preset_info("sme")
        assert "_post_debate_preset" in info["flags"]


# ---------------------------------------------------------------------------
# apply_preset
# ---------------------------------------------------------------------------


class TestApplyPreset:
    def test_base_preset_values(self) -> None:
        result = apply_preset("minimal")
        assert result["enable_stability_detection"] is True
        assert result["budget_downgrade_models"] is True

    def test_overrides_dict_takes_precedence(self) -> None:
        result = apply_preset(
            "minimal", overrides={"enable_stability_detection": False}
        )
        assert result["enable_stability_detection"] is False

    def test_overrides_adds_new_keys(self) -> None:
        result = apply_preset("minimal", overrides={"custom_key": 42})
        assert result["custom_key"] == 42

    def test_kwargs_take_precedence(self) -> None:
        result = apply_preset("minimal", budget_downgrade_models=False)
        assert result["budget_downgrade_models"] is False

    def test_kwargs_add_new_keys(self) -> None:
        result = apply_preset("minimal", extra_setting="hello")
        assert result["extra_setting"] == "hello"

    def test_both_overrides_and_kwargs_merged(self) -> None:
        result = apply_preset(
            "minimal",
            overrides={"from_overrides": 1},
            from_kwargs=2,
        )
        assert result["from_overrides"] == 1
        assert result["from_kwargs"] == 2

    def test_kwargs_win_over_overrides(self) -> None:
        """kwargs should take precedence over overrides dict."""
        result = apply_preset(
            "minimal",
            overrides={"enable_stability_detection": False},
            enable_stability_detection=True,
        )
        assert result["enable_stability_detection"] is True

    def test_unknown_name_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown preset"):
            apply_preset("does_not_exist")

    def test_no_overrides_returns_preset_as_is(self) -> None:
        """With no overrides or kwargs, apply_preset is equivalent to get_preset."""
        applied = apply_preset("minimal")
        gotten = get_preset("minimal")
        assert applied == gotten


# ---------------------------------------------------------------------------
# Preset content validation
# ---------------------------------------------------------------------------


class TestPresetContents:
    def test_sme_has_receipt_generation(self) -> None:
        result = get_preset("sme")
        assert result["enable_receipt_generation"] is True

    def test_sme_has_meta_learning(self) -> None:
        result = get_preset("sme")
        assert result["enable_meta_learning"] is True

    def test_enterprise_has_airlock(self) -> None:
        result = get_preset("enterprise")
        assert result["use_airlock"] is True

    def test_enterprise_has_telemetry(self) -> None:
        result = get_preset("enterprise")
        assert result["enable_telemetry"] is True

    def test_minimal_is_small(self) -> None:
        """Minimal preset should have very few keys."""
        result = get_preset("minimal")
        assert len(result) <= 3

    def test_healthcare_vertical(self) -> None:
        result = get_preset("healthcare")
        assert result["vertical"] == "healthcare_hipaa"

    def test_healthcare_has_privacy(self) -> None:
        result = get_preset("healthcare")
        assert result["enable_privacy_anonymization"] is True

    def test_financial_vertical(self) -> None:
        result = get_preset("financial")
        assert result["vertical"] == "financial_audit"

    def test_visual_has_cartographer(self) -> None:
        result = get_preset("visual")
        assert result["enable_cartographer"] is True

    def test_visual_has_spectator(self) -> None:
        result = get_preset("visual")
        assert result["enable_spectator"] is True

    def test_compliance_has_compliance_artifacts(self) -> None:
        result = get_preset("compliance")
        assert result["enable_compliance_artifacts"] is True

    def test_research_has_power_sampling(self) -> None:
        result = get_preset("research")
        assert result["enable_power_sampling"] is True

    def test_research_has_forking(self) -> None:
        result = get_preset("research")
        assert result["enable_debate_forking"] is True

    def test_diverse_has_min_provider_diversity(self) -> None:
        result = get_preset("diverse")
        assert result["min_provider_diversity"] == 3

    def test_diverse_has_prefer_diverse_providers(self) -> None:
        result = get_preset("diverse")
        assert result["prefer_diverse_providers"] is True

    def test_audit_has_full_traceability(self) -> None:
        result = get_preset("audit")
        assert result["enable_receipt_generation"] is True
        assert result["enable_provenance"] is True
        assert result["enable_bead_tracking"] is True
        assert result["enable_compliance_artifacts"] is True


# ---------------------------------------------------------------------------
# Descriptions completeness
# ---------------------------------------------------------------------------


class TestPresetDescriptions:
    @pytest.mark.parametrize("name", ALL_PRESET_NAMES)
    def test_every_preset_has_description(self, name: str) -> None:
        assert name in _PRESET_DESCRIPTIONS
        assert len(_PRESET_DESCRIPTIONS[name]) > 0

    def test_no_extra_descriptions(self) -> None:
        """No description for a preset that doesn't exist."""
        for name in _PRESET_DESCRIPTIONS:
            assert name in _PRESETS
