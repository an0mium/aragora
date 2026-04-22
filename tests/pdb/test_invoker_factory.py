"""Tests for :mod:`aragora.pdb.invoker_factory`.

Focus: env-var gating, fail-closed when both core keys are missing,
correct slot-level unavailability mapping, test-friendly factory
injection.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.pdb.invoker_factory import (
    CLAUDE_MODEL_ENV,
    InvokerFactoryError,
    OPENAI_MODEL_ENV,
    build_default_invoker,
    unavailable_slots_for,
)
from aragora.pdb.panel_config import (
    PDBBudgetConfig,
    PDBPanelConfig,
    PDBPanelDefinition,
    PDBPanelSlot,
    PDBPromptSet,
)
from aragora.pdb.real_invoker import FAMILY_CLAUDE, FAMILY_GPT, RealProviderInvoker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slot(slot_id: str, *, family: str, lens: str = "core") -> PDBPanelSlot:
    return PDBPanelSlot(
        slot_id=slot_id,
        review_role="logic_reviewer",
        lens=lens,
        family=family,
        candidates=(f"{family}-cli",),
        required=False,
    )


def _config() -> PDBPanelConfig:
    slots = {
        "claude_core": _slot("claude_core", family=FAMILY_CLAUDE, lens="core"),
        "gpt_core": _slot("gpt_core", family=FAMILY_GPT, lens="core"),
        "gemini_heterodox": _slot("gemini_heterodox", family="gemini", lens="heterodox"),
        "grok_heterodox": _slot("grok_heterodox", family="grok", lens="heterodox"),
        "deepseek_heterodox": _slot("deepseek_heterodox", family="deepseek", lens="heterodox"),
        "kimi_heterodox": _slot("kimi_heterodox", family="kimi", lens="heterodox"),
        "qwen_heterodox": _slot("qwen_heterodox", family="qwen", lens="heterodox"),
        "mistral_regulatory": _slot("mistral_regulatory", family="mistral", lens="regulatory"),
    }
    return PDBPanelConfig(
        version=1,
        default_panel="p",
        default_prompt_set="ps",
        budget=PDBBudgetConfig(
            per_brief_usd=8.0,
            per_day_usd=200.0,
            reserve_for_manual_escalation_usd=10.0,
        ),
        slots=slots,
        panels={
            "p": PDBPanelDefinition(
                panel_id="p",
                findings_slots=("claude_core", "gpt_core", "gemini_heterodox"),
                critique_slots=("claude_core", "gpt_core", "gemini_heterodox"),
                synthesizer_slot="claude_core",
            ),
        },
        prompt_sets={
            "ps": PDBPromptSet(
                prompt_set_id="ps",
                findings_prompt="f",
                critique_prompt="c",
                synthesis_prompt="s",
            ),
        },
    )


def _fake_claude(model: str, api_key: str | None) -> Any:
    mock = MagicMock(spec_set=["model", "last_tokens_in", "last_tokens_out", "generate"])
    mock.model = model
    mock.last_tokens_in = 0
    mock.last_tokens_out = 0
    return mock


def _fake_gpt(model: str, api_key: str | None) -> Any:
    mock = MagicMock(spec_set=["model", "last_tokens_in", "last_tokens_out", "generate"])
    mock.model = model
    mock.last_tokens_in = 0
    mock.last_tokens_out = 0
    return mock


# ---------------------------------------------------------------------------
# unavailable_slots_for
# ---------------------------------------------------------------------------


class TestUnavailableSlotsFor:
    def test_heterodox_always_unavailable(self) -> None:
        unavailable = unavailable_slots_for(
            config=_config(),
            have_claude=True,
            have_gpt=True,
        )
        assert "gemini_heterodox" in unavailable
        assert "grok_heterodox" in unavailable
        assert "deepseek_heterodox" in unavailable
        assert "kimi_heterodox" in unavailable
        assert "qwen_heterodox" in unavailable
        assert "mistral_regulatory" in unavailable
        # Core slots remain available
        assert "claude_core" not in unavailable
        assert "gpt_core" not in unavailable

    def test_missing_claude_marks_claude_slots(self) -> None:
        unavailable = unavailable_slots_for(
            config=_config(),
            have_claude=False,
            have_gpt=True,
        )
        assert "claude_core" in unavailable
        assert "gpt_core" not in unavailable

    def test_missing_gpt_marks_gpt_slots(self) -> None:
        unavailable = unavailable_slots_for(
            config=_config(),
            have_claude=True,
            have_gpt=False,
        )
        assert "gpt_core" in unavailable
        assert "claude_core" not in unavailable


# ---------------------------------------------------------------------------
# build_default_invoker
# ---------------------------------------------------------------------------


class TestBuildDefaultInvoker:
    def test_both_keys_set_wires_both_cores(self) -> None:
        invoker = build_default_invoker(
            config=_config(),
            env={
                "ANTHROPIC_API_KEY": "sk-ant-test",
                "OPENAI_API_KEY": "sk-oa-test",
            },
            anthropic_agent_factory=_fake_claude,
            openai_agent_factory=_fake_gpt,
        )
        assert isinstance(invoker, RealProviderInvoker)
        # Core slots wired
        assert invoker._agents[FAMILY_CLAUDE] is not None
        assert invoker._agents[FAMILY_GPT] is not None
        # Heterodox slots in unavailable set
        assert "grok_heterodox" in invoker._unavailable_slots
        assert "gemini_heterodox" in invoker._unavailable_slots
        # Core slots NOT in unavailable set
        assert "claude_core" not in invoker._unavailable_slots
        assert "gpt_core" not in invoker._unavailable_slots

    def test_only_anthropic_key_sets_gpt_slots_unavailable(self) -> None:
        invoker = build_default_invoker(
            config=_config(),
            env={"ANTHROPIC_API_KEY": "sk-ant-test"},
            anthropic_agent_factory=_fake_claude,
            openai_agent_factory=_fake_gpt,
        )
        assert invoker._agents[FAMILY_CLAUDE] is not None
        assert invoker._agents[FAMILY_GPT] is None
        assert "gpt_core" in invoker._unavailable_slots
        assert "claude_core" not in invoker._unavailable_slots

    def test_only_openai_key_sets_claude_slots_unavailable(self) -> None:
        invoker = build_default_invoker(
            config=_config(),
            env={"OPENAI_API_KEY": "sk-oa-test"},
            anthropic_agent_factory=_fake_claude,
            openai_agent_factory=_fake_gpt,
        )
        assert invoker._agents[FAMILY_CLAUDE] is None
        assert invoker._agents[FAMILY_GPT] is not None
        assert "claude_core" in invoker._unavailable_slots
        assert "gpt_core" not in invoker._unavailable_slots

    def test_both_keys_missing_raises_fail_closed(self) -> None:
        with pytest.raises(InvokerFactoryError) as exc:
            build_default_invoker(
                config=_config(),
                env={},
                anthropic_agent_factory=_fake_claude,
                openai_agent_factory=_fake_gpt,
            )
        msg = str(exc.value)
        assert "ANTHROPIC_API_KEY" in msg
        assert "OPENAI_API_KEY" in msg

    def test_empty_string_keys_treated_as_missing(self) -> None:
        with pytest.raises(InvokerFactoryError):
            build_default_invoker(
                config=_config(),
                env={"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "   "},
                anthropic_agent_factory=_fake_claude,
                openai_agent_factory=_fake_gpt,
            )

    def test_model_override_env_vars(self) -> None:
        calls: list[tuple[str, str, str | None]] = []

        def _recording_claude(model: str, api_key: str | None) -> Any:
            calls.append(("claude", model, api_key))
            return _fake_claude(model, api_key)

        def _recording_gpt(model: str, api_key: str | None) -> Any:
            calls.append(("gpt", model, api_key))
            return _fake_gpt(model, api_key)

        build_default_invoker(
            config=_config(),
            env={
                "ANTHROPIC_API_KEY": "sk-ant-test",
                "OPENAI_API_KEY": "sk-oa-test",
                CLAUDE_MODEL_ENV: "claude-opus-4-7",
                OPENAI_MODEL_ENV: "gpt-5.4-pro",
            },
            anthropic_agent_factory=_recording_claude,
            openai_agent_factory=_recording_gpt,
        )
        by_family = {family: (model, key) for family, model, key in calls}
        assert by_family["claude"][0] == "claude-opus-4-7"
        assert by_family["gpt"][0] == "gpt-5.4-pro"
        # API keys get passed through verbatim.
        assert by_family["claude"][1] == "sk-ant-test"
        assert by_family["gpt"][1] == "sk-oa-test"

    def test_uses_process_env_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        invoker = build_default_invoker(
            config=_config(),
            anthropic_agent_factory=_fake_claude,
            openai_agent_factory=_fake_gpt,
        )
        assert invoker._agents[FAMILY_CLAUDE] is not None
        assert invoker._agents[FAMILY_GPT] is None
