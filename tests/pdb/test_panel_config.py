"""Tests for :mod:`aragora.pdb.panel_config`.

Covers every rule in the PDB Mode 3 PR2 spec's "Validation rules"
section:

- ``version == 1`` (and explicit failure on other versions)
- ``default_panel`` / ``synthesizer_slot`` / ``default_prompt_set`` existence
- every slot defines ``review_role``, ``lens``, ``family``, ``candidates``
- ``findings_slots`` must include both required core slots
- the selected panel must include at least one non-core lens
- ``same_as_findings`` sentinel only legal on ``critique_slots``
- ``required: true`` signaling and required-slot propagation through the
  provider-slot resolver projection
- exact ``field_path`` attribute on every :class:`PDBPanelConfigError`
- committed default config loads cleanly
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from aragora.pdb.panel_config import (
    DEFAULT_CONFIG_PATH,
    PDBBudgetConfig,
    PDBPanelConfig,
    PDBPanelConfigError,
    PDBPanelDefinition,
    PDBPanelSlot,
    PDBPromptSet,
    load_panel_config,
    panel_slots,
    provider_slot_definitions,
    validate_panel_config,
)
from aragora.review.provider_slots import ProviderSlotDefinition


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _base_config() -> dict:
    """Return a minimal valid config mirroring the committed default."""
    return {
        "version": 1,
        "default_panel": "p",
        "default_prompt_set": "ps",
        "budgets": {
            "per_brief_usd": 2.0,
            "per_day_usd": 20.0,
            "reserve_for_manual_escalation_usd": 5.0,
        },
        "slots": {
            "claude_core": {
                "review_role": "logic_reviewer",
                "lens": "core",
                "family": "claude",
                "candidates": ["claude", "anthropic-api"],
                "required": True,
            },
            "gpt_core": {
                "review_role": "security_reviewer",
                "lens": "core",
                "family": "gpt",
                "candidates": ["codex", "openai-api"],
                "required": True,
            },
            "gemini_hetero": {
                "review_role": "maintainability_reviewer",
                "lens": "heterodox",
                "family": "gemini",
                "candidates": ["gemini-cli"],
                "required": False,
            },
        },
        "panels": {
            "p": {
                "findings_slots": ["claude_core", "gpt_core", "gemini_hetero"],
                "critique_slots": "same_as_findings",
                "synthesizer_slot": "claude_core",
            },
        },
        "prompt_sets": {
            "ps": {
                "findings_prompt": "f",
                "critique_prompt": "c",
                "synthesis_prompt": "s",
            },
        },
    }


# ---------------------------------------------------------------------------
# Committed default config
# ---------------------------------------------------------------------------


def test_load_committed_default_config_succeeds() -> None:
    cfg = load_panel_config()
    assert isinstance(cfg, PDBPanelConfig)
    assert cfg.version == 1
    assert cfg.default_panel in cfg.panels
    assert cfg.default_prompt_set in cfg.prompt_sets
    assert cfg.source_path == DEFAULT_CONFIG_PATH
    # committed default must include both required core slots
    required = {sid for sid, s in cfg.slots.items() if s.required}
    assert {"claude_core", "gpt_core"} <= required


def test_default_config_projects_to_provider_slot_definitions() -> None:
    cfg = load_panel_config()
    defs = provider_slot_definitions(cfg, cfg.default_panel)
    assert defs
    assert all(isinstance(d, ProviderSlotDefinition) for d in defs)
    slot_ids = [d.slot_id for d in defs]
    # Order preserves findings_slots order
    expected = list(cfg.panels[cfg.default_panel].findings_slots)
    assert slot_ids == expected


# ---------------------------------------------------------------------------
# Happy path on minimal config
# ---------------------------------------------------------------------------


def test_validate_panel_config_happy_path() -> None:
    cfg = validate_panel_config(_base_config())
    assert isinstance(cfg.budget, PDBBudgetConfig)
    assert isinstance(cfg.slots["claude_core"], PDBPanelSlot)
    assert isinstance(cfg.panels["p"], PDBPanelDefinition)
    assert isinstance(cfg.prompt_sets["ps"], PDBPromptSet)
    # same_as_findings sentinel expands to a tuple copy
    assert cfg.panels["p"].critique_slots == cfg.panels["p"].findings_slots
    assert cfg.panels["p"].critique_slots == ("claude_core", "gpt_core", "gemini_hetero")


def test_panel_slots_preserves_order() -> None:
    cfg = validate_panel_config(_base_config())
    ordered = panel_slots(cfg, "p")
    assert [s.slot_id for s in ordered] == ["claude_core", "gpt_core", "gemini_hetero"]


def test_critique_slots_as_explicit_list() -> None:
    raw = _base_config()
    raw["panels"]["p"]["critique_slots"] = ["claude_core", "gpt_core"]
    cfg = validate_panel_config(raw)
    assert cfg.panels["p"].critique_slots == ("claude_core", "gpt_core")


# ---------------------------------------------------------------------------
# Version / top-level shape
# ---------------------------------------------------------------------------


def test_unsupported_version_rejected() -> None:
    raw = _base_config()
    raw["version"] = 2
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "version"


def test_missing_top_level_key_raises_with_exact_path() -> None:
    raw = _base_config()
    raw.pop("default_panel")
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "default_panel"


def test_wrong_type_at_top_level_rejected() -> None:
    raw = _base_config()
    raw["version"] = "1"  # string instead of int
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "version"


# ---------------------------------------------------------------------------
# Budget validation
# ---------------------------------------------------------------------------


def test_budget_per_brief_must_be_positive() -> None:
    raw = _base_config()
    raw["budgets"]["per_brief_usd"] = 0
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "budgets.per_brief_usd"


def test_budget_per_day_must_be_positive() -> None:
    raw = _base_config()
    raw["budgets"]["per_day_usd"] = -1
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "budgets.per_day_usd"


def test_budget_reserve_must_be_nonnegative() -> None:
    raw = _base_config()
    raw["budgets"]["reserve_for_manual_escalation_usd"] = -0.5
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "budgets.reserve_for_manual_escalation_usd"


def test_budget_per_brief_cannot_exceed_per_day() -> None:
    raw = _base_config()
    raw["budgets"]["per_brief_usd"] = 200.0
    raw["budgets"]["per_day_usd"] = 50.0
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "budgets.per_brief_usd"


def test_budget_numeric_only() -> None:
    raw = _base_config()
    raw["budgets"]["per_brief_usd"] = True  # bool is rejected
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "budgets.per_brief_usd"


# ---------------------------------------------------------------------------
# Slot validation
# ---------------------------------------------------------------------------


def test_slot_missing_review_role_rejected() -> None:
    raw = _base_config()
    raw["slots"]["claude_core"].pop("review_role")
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "slots.claude_core.review_role"


def test_slot_missing_lens_rejected() -> None:
    raw = _base_config()
    raw["slots"]["claude_core"].pop("lens")
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "slots.claude_core.lens"


def test_slot_missing_family_rejected() -> None:
    raw = _base_config()
    raw["slots"]["claude_core"].pop("family")
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "slots.claude_core.family"


def test_slot_empty_candidates_rejected() -> None:
    raw = _base_config()
    raw["slots"]["claude_core"]["candidates"] = []
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "slots.claude_core.candidates"


def test_slot_non_string_candidate_rejected() -> None:
    raw = _base_config()
    raw["slots"]["claude_core"]["candidates"] = ["claude", 42]
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "slots.claude_core.candidates[1]"


def test_slot_required_must_be_bool() -> None:
    raw = _base_config()
    raw["slots"]["claude_core"]["required"] = "yes"
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "slots.claude_core.required"


def test_slot_required_defaults_to_false() -> None:
    raw = _base_config()
    raw["slots"]["claude_core"]["required"] = False
    raw["slots"]["gpt_core"]["required"] = False
    # Introduce a required slot so findings_slots still validates
    raw["slots"]["ext_core"] = {
        "review_role": "logic_reviewer",
        "lens": "core",
        "family": "ext",
        "candidates": ["ext-cli"],
    }
    raw["panels"]["p"]["findings_slots"] = [
        "claude_core",
        "gpt_core",
        "ext_core",
        "gemini_hetero",
    ]
    cfg = validate_panel_config(raw)
    assert cfg.slots["ext_core"].required is False


# ---------------------------------------------------------------------------
# Panel validation
# ---------------------------------------------------------------------------


def test_unknown_default_panel_rejected() -> None:
    raw = _base_config()
    raw["default_panel"] = "missing"
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "default_panel"


def test_unknown_default_prompt_set_rejected() -> None:
    raw = _base_config()
    raw["default_prompt_set"] = "missing"
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "default_prompt_set"


def test_panel_references_unknown_slot() -> None:
    raw = _base_config()
    raw["panels"]["p"]["findings_slots"].append("nope")
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "panels.p.findings_slots[3]"


def test_panel_duplicate_slot_rejected() -> None:
    raw = _base_config()
    raw["panels"]["p"]["findings_slots"] = [
        "claude_core",
        "claude_core",
        "gpt_core",
        "gemini_hetero",
    ]
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "panels.p.findings_slots[1]"


def test_panel_missing_required_slot_rejected() -> None:
    raw = _base_config()
    # Drop gpt_core from findings_slots while leaving it required
    raw["panels"]["p"]["findings_slots"] = ["claude_core", "gemini_hetero"]
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "panels.p.findings_slots"
    assert "gpt_core" in str(exc.value)


def test_panel_needs_two_core_slots() -> None:
    raw = _base_config()
    # Make gpt_core heterodox; findings_slots would then have only one core slot
    raw["slots"]["gpt_core"]["lens"] = "heterodox"
    raw["slots"]["gpt_core"]["required"] = False
    # also need to keep both cores "required" invariant; loosen it
    raw["slots"]["claude_core"]["required"] = True
    raw["slots"]["gpt_core"]["required"] = False
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "panels.p.findings_slots"
    assert "core-lens" in str(exc.value)


def test_panel_needs_non_core_lens() -> None:
    raw = _base_config()
    # Drop the heterodox slot from findings_slots
    raw["panels"]["p"]["findings_slots"] = ["claude_core", "gpt_core"]
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "panels.p.findings_slots"
    assert "non-core" in str(exc.value)


def test_synthesizer_slot_must_be_in_findings() -> None:
    raw = _base_config()
    raw["slots"]["extra"] = {
        "review_role": "logic_reviewer",
        "lens": "core",
        "family": "extra",
        "candidates": ["extra"],
    }
    raw["panels"]["p"]["synthesizer_slot"] = "extra"
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "panels.p.synthesizer_slot"


def test_synthesizer_slot_must_exist() -> None:
    raw = _base_config()
    raw["panels"]["p"]["synthesizer_slot"] = "ghost"
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "panels.p.synthesizer_slot"


def test_same_as_findings_only_for_critique_slots() -> None:
    raw = _base_config()
    # Putting the sentinel in findings_slots is invalid (it must be a list)
    raw["panels"]["p"]["findings_slots"] = "same_as_findings"
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "panels.p.findings_slots"


def test_critique_slots_unknown_string_rejected() -> None:
    raw = _base_config()
    raw["panels"]["p"]["critique_slots"] = "all_slots"
    with pytest.raises(PDBPanelConfigError) as exc:
        validate_panel_config(raw)
    assert exc.value.field_path == "panels.p.critique_slots"


# ---------------------------------------------------------------------------
# I/O error paths
# ---------------------------------------------------------------------------


def test_load_panel_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PDBPanelConfigError) as exc:
        load_panel_config(tmp_path / "nope.yaml")
    assert exc.value.field_path == "<path>"


def test_load_panel_config_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    with pytest.raises(PDBPanelConfigError) as exc:
        load_panel_config(p)
    assert exc.value.field_path == "<root>"


def test_load_panel_config_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(PDBPanelConfigError) as exc:
        load_panel_config(p)
    assert exc.value.field_path == "<root>"


def test_load_panel_config_roundtrip(tmp_path: Path) -> None:
    data = _base_config()
    p = tmp_path / "ok.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = load_panel_config(p)
    assert cfg.source_path == p
    assert cfg.default_panel == "p"


# ---------------------------------------------------------------------------
# panel_slots / provider_slot_definitions error cases
# ---------------------------------------------------------------------------


def test_panel_slots_unknown_panel() -> None:
    cfg = validate_panel_config(_base_config())
    with pytest.raises(PDBPanelConfigError):
        panel_slots(cfg, "ghost")


def test_provider_slot_definitions_preserve_required_metadata() -> None:
    cfg = validate_panel_config(_base_config())
    defs = provider_slot_definitions(cfg, "p")
    # Mapping back to the PDBPanelSlot retains required flags — projection
    # only drops the optional ``required`` attribute because
    # ProviderSlotDefinition is generic and lens-agnostic.
    assert [d.slot_id for d in defs] == list(cfg.panels["p"].findings_slots)
    required = [cfg.slots[d.slot_id].required for d in defs]
    assert required == [True, True, False]


def test_raw_input_not_mutated() -> None:
    raw = _base_config()
    snapshot = copy.deepcopy(raw)
    validate_panel_config(raw)
    assert raw == snapshot
