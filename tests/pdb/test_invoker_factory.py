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
    DEEPSEEK_MODEL_DEFAULT,
    DEEPSEEK_MODEL_ENV,
    GEMINI_MODEL_DEFAULT,
    GEMINI_MODEL_ENV,
    GROK_MODEL_DEFAULT,
    GROK_MODEL_ENV,
    InvokerFactoryError,
    KIMI_MODEL_DEFAULT,
    KIMI_MODEL_ENV,
    MISTRAL_MODEL_DEFAULT,
    MISTRAL_MODEL_ENV,
    OPENAI_MODEL_ENV,
    QWEN_MODEL_DEFAULT,
    QWEN_MODEL_ENV,
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
from aragora.pdb.real_invoker import (
    FAMILY_CLAUDE,
    FAMILY_DEEPSEEK,
    FAMILY_GEMINI,
    FAMILY_GPT,
    FAMILY_GROK,
    FAMILY_KIMI,
    FAMILY_MISTRAL,
    FAMILY_QWEN,
    RealProviderInvoker,
)


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


def _fake_agent(model: str, api_key: str | None) -> Any:
    """Generic stand-in for any heterodox agent in the factory."""
    mock = MagicMock(spec_set=["model", "last_tokens_in", "last_tokens_out", "generate"])
    mock.model = model
    mock.last_tokens_in = 0
    mock.last_tokens_out = 0
    return mock


def _all_fake_factories() -> dict[str, Any]:
    """Inject fakes for every agent factory; keeps build calls terse."""
    return dict(
        anthropic_agent_factory=_fake_claude,
        openai_agent_factory=_fake_gpt,
        gemini_agent_factory=_fake_agent,
        grok_agent_factory=_fake_agent,
        openrouter_agent_factory=_fake_agent,
        mistral_agent_factory=_fake_agent,
    )


# ---------------------------------------------------------------------------
# unavailable_slots_for
# ---------------------------------------------------------------------------


class TestUnavailableSlotsFor:
    def test_heterodox_unavailable_when_no_keys(self) -> None:
        # Default: only core keys present → every heterodox family is
        # unavailable.
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

    def test_gemini_key_unlocks_gemini_slot(self) -> None:
        unavailable = unavailable_slots_for(
            config=_config(),
            have_claude=True,
            have_gpt=True,
            have_gemini=True,
        )
        assert "gemini_heterodox" not in unavailable
        # Other heterodox still blocked.
        assert "grok_heterodox" in unavailable
        assert "deepseek_heterodox" in unavailable
        assert "mistral_regulatory" in unavailable

    def test_grok_key_unlocks_grok_slot(self) -> None:
        unavailable = unavailable_slots_for(
            config=_config(),
            have_claude=True,
            have_gpt=True,
            have_grok=True,
        )
        assert "grok_heterodox" not in unavailable
        assert "gemini_heterodox" in unavailable

    def test_openrouter_key_unlocks_deepseek_kimi_qwen(self) -> None:
        unavailable = unavailable_slots_for(
            config=_config(),
            have_claude=True,
            have_gpt=True,
            have_openrouter=True,
        )
        assert "deepseek_heterodox" not in unavailable
        assert "kimi_heterodox" not in unavailable
        assert "qwen_heterodox" not in unavailable
        # Gemini / Grok / Mistral stay blocked without their own keys.
        assert "gemini_heterodox" in unavailable
        assert "grok_heterodox" in unavailable
        assert "mistral_regulatory" in unavailable

    def test_mistral_key_unlocks_mistral_slot(self) -> None:
        unavailable = unavailable_slots_for(
            config=_config(),
            have_claude=True,
            have_gpt=True,
            have_mistral=True,
        )
        assert "mistral_regulatory" not in unavailable
        assert "gemini_heterodox" in unavailable

    def test_full_key_set_unlocks_entire_panel(self) -> None:
        unavailable = unavailable_slots_for(
            config=_config(),
            have_claude=True,
            have_gpt=True,
            have_gemini=True,
            have_grok=True,
            have_openrouter=True,
            have_mistral=True,
        )
        assert unavailable == frozenset()


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
            **_all_fake_factories(),
        )
        assert isinstance(invoker, RealProviderInvoker)
        # Core slots wired
        assert invoker._agents[FAMILY_CLAUDE] is not None
        assert invoker._agents[FAMILY_GPT] is not None
        # Heterodox slots in unavailable set (no heterodox keys supplied)
        assert "grok_heterodox" in invoker._unavailable_slots
        assert "gemini_heterodox" in invoker._unavailable_slots
        assert "deepseek_heterodox" in invoker._unavailable_slots
        assert "kimi_heterodox" in invoker._unavailable_slots
        assert "qwen_heterodox" in invoker._unavailable_slots
        assert "mistral_regulatory" in invoker._unavailable_slots
        # Core slots NOT in unavailable set
        assert "claude_core" not in invoker._unavailable_slots
        assert "gpt_core" not in invoker._unavailable_slots
        # Heterodox agents default to None when no key is set.
        assert invoker._agents[FAMILY_GEMINI] is None
        assert invoker._agents[FAMILY_GROK] is None
        assert invoker._agents[FAMILY_DEEPSEEK] is None
        assert invoker._agents[FAMILY_KIMI] is None
        assert invoker._agents[FAMILY_QWEN] is None
        assert invoker._agents[FAMILY_MISTRAL] is None

    def test_only_anthropic_key_sets_gpt_slots_unavailable(self) -> None:
        invoker = build_default_invoker(
            config=_config(),
            env={"ANTHROPIC_API_KEY": "sk-ant-test"},
            **_all_fake_factories(),
        )
        assert invoker._agents[FAMILY_CLAUDE] is not None
        assert invoker._agents[FAMILY_GPT] is None
        assert "gpt_core" in invoker._unavailable_slots
        assert "claude_core" not in invoker._unavailable_slots

    def test_only_openai_key_sets_claude_slots_unavailable(self) -> None:
        invoker = build_default_invoker(
            config=_config(),
            env={"OPENAI_API_KEY": "sk-oa-test"},
            **_all_fake_factories(),
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
                **_all_fake_factories(),
            )
        msg = str(exc.value)
        assert "ANTHROPIC_API_KEY" in msg
        assert "OPENAI_API_KEY" in msg

    def test_empty_string_keys_treated_as_missing(self) -> None:
        with pytest.raises(InvokerFactoryError):
            build_default_invoker(
                config=_config(),
                env={"ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "   "},
                **_all_fake_factories(),
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
            gemini_agent_factory=_fake_agent,
            grok_agent_factory=_fake_agent,
            openrouter_agent_factory=_fake_agent,
            mistral_agent_factory=_fake_agent,
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
        # Clear heterodox keys so the test is deterministic regardless
        # of the user's shell environment.
        for key in (
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "GROK_API_KEY",
            "XAI_API_KEY",
            "OPENROUTER_API_KEY",
            "MISTRAL_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        invoker = build_default_invoker(
            config=_config(),
            **_all_fake_factories(),
        )
        assert invoker._agents[FAMILY_CLAUDE] is not None
        assert invoker._agents[FAMILY_GPT] is None


# ---------------------------------------------------------------------------
# Heterodox / regulatory key wiring (Phase B)
# ---------------------------------------------------------------------------


class TestHeterodoxKeyWiring:
    def test_gemini_api_key_wires_gemini_agent(self) -> None:
        calls: list[tuple[str, str | None]] = []

        def _recording(model: str, api_key: str | None) -> Any:
            calls.append((model, api_key))
            return _fake_agent(model, api_key)

        invoker = build_default_invoker(
            config=_config(),
            env={
                "ANTHROPIC_API_KEY": "sk-ant",
                "OPENAI_API_KEY": "sk-oa",
                "GEMINI_API_KEY": "goog-key",
            },
            anthropic_agent_factory=_fake_claude,
            openai_agent_factory=_fake_gpt,
            gemini_agent_factory=_recording,
            grok_agent_factory=_fake_agent,
            openrouter_agent_factory=_fake_agent,
            mistral_agent_factory=_fake_agent,
        )
        assert invoker._agents[FAMILY_GEMINI] is not None
        assert "gemini_heterodox" not in invoker._unavailable_slots
        # Default model used when no override env.
        assert calls == [(GEMINI_MODEL_DEFAULT, "goog-key")]

    def test_google_api_key_also_wires_gemini(self) -> None:
        invoker = build_default_invoker(
            config=_config(),
            env={
                "ANTHROPIC_API_KEY": "sk-ant",
                "OPENAI_API_KEY": "sk-oa",
                "GOOGLE_API_KEY": "goog-key",
            },
            **_all_fake_factories(),
        )
        assert invoker._agents[FAMILY_GEMINI] is not None
        assert "gemini_heterodox" not in invoker._unavailable_slots

    def test_xai_and_grok_keys_both_wire_grok(self) -> None:
        invoker_xai = build_default_invoker(
            config=_config(),
            env={
                "ANTHROPIC_API_KEY": "sk-ant",
                "OPENAI_API_KEY": "sk-oa",
                "XAI_API_KEY": "xai-key",
            },
            **_all_fake_factories(),
        )
        assert invoker_xai._agents[FAMILY_GROK] is not None
        invoker_grok = build_default_invoker(
            config=_config(),
            env={
                "ANTHROPIC_API_KEY": "sk-ant",
                "OPENAI_API_KEY": "sk-oa",
                "GROK_API_KEY": "grok-key",
            },
            **_all_fake_factories(),
        )
        assert invoker_grok._agents[FAMILY_GROK] is not None

    def test_openrouter_key_wires_deepseek_kimi_qwen(self) -> None:
        calls: list[tuple[str, str | None]] = []

        def _recording(model: str, api_key: str | None) -> Any:
            calls.append((model, api_key))
            return _fake_agent(model, api_key)

        invoker = build_default_invoker(
            config=_config(),
            env={
                "ANTHROPIC_API_KEY": "sk-ant",
                "OPENAI_API_KEY": "sk-oa",
                "OPENROUTER_API_KEY": "or-key",
            },
            anthropic_agent_factory=_fake_claude,
            openai_agent_factory=_fake_gpt,
            gemini_agent_factory=_fake_agent,
            grok_agent_factory=_fake_agent,
            openrouter_agent_factory=_recording,
            mistral_agent_factory=_fake_agent,
        )
        assert invoker._agents[FAMILY_DEEPSEEK] is not None
        assert invoker._agents[FAMILY_KIMI] is not None
        assert invoker._agents[FAMILY_QWEN] is not None
        for slot_id in ("deepseek_heterodox", "kimi_heterodox", "qwen_heterodox"):
            assert slot_id not in invoker._unavailable_slots, slot_id
        # Openrouter factory called once per family with the correct
        # default model and the same credential.
        models = {model for model, _ in calls}
        assert DEEPSEEK_MODEL_DEFAULT in models
        assert KIMI_MODEL_DEFAULT in models
        assert QWEN_MODEL_DEFAULT in models
        assert {key for _, key in calls} == {"or-key"}

    def test_mistral_api_key_wires_mistral_agent(self) -> None:
        invoker = build_default_invoker(
            config=_config(),
            env={
                "ANTHROPIC_API_KEY": "sk-ant",
                "OPENAI_API_KEY": "sk-oa",
                "MISTRAL_API_KEY": "mi-key",
            },
            **_all_fake_factories(),
        )
        assert invoker._agents[FAMILY_MISTRAL] is not None
        assert "mistral_regulatory" not in invoker._unavailable_slots

    def test_only_core_keys_marks_all_heterodox_unavailable(self) -> None:
        invoker = build_default_invoker(
            config=_config(),
            env={
                "ANTHROPIC_API_KEY": "sk-ant",
                "OPENAI_API_KEY": "sk-oa",
            },
            **_all_fake_factories(),
        )
        for slot_id in (
            "gemini_heterodox",
            "grok_heterodox",
            "deepseek_heterodox",
            "kimi_heterodox",
            "qwen_heterodox",
            "mistral_regulatory",
        ):
            assert slot_id in invoker._unavailable_slots, slot_id

    def test_all_keys_set_full_roster_available(self) -> None:
        invoker = build_default_invoker(
            config=_config(),
            env={
                "ANTHROPIC_API_KEY": "sk-ant",
                "OPENAI_API_KEY": "sk-oa",
                "GEMINI_API_KEY": "goog",
                "XAI_API_KEY": "xai",
                "OPENROUTER_API_KEY": "or",
                "MISTRAL_API_KEY": "mi",
            },
            **_all_fake_factories(),
        )
        assert invoker._unavailable_slots == frozenset()
        # All 8 agents wired.
        assert invoker._agents[FAMILY_CLAUDE] is not None
        assert invoker._agents[FAMILY_GPT] is not None
        assert invoker._agents[FAMILY_GEMINI] is not None
        assert invoker._agents[FAMILY_GROK] is not None
        assert invoker._agents[FAMILY_DEEPSEEK] is not None
        assert invoker._agents[FAMILY_KIMI] is not None
        assert invoker._agents[FAMILY_QWEN] is not None
        assert invoker._agents[FAMILY_MISTRAL] is not None

    def test_partial_keys_produces_mixed_roster(self) -> None:
        # Mission criterion (c): partial keys → only matching slots
        # available.
        invoker = build_default_invoker(
            config=_config(),
            env={
                "ANTHROPIC_API_KEY": "sk-ant",
                "OPENAI_API_KEY": "sk-oa",
                "OPENROUTER_API_KEY": "or",  # wires deepseek/kimi/qwen
                "MISTRAL_API_KEY": "mi",  # wires mistral
                # Intentionally omit GEMINI and GROK/XAI keys.
            },
            **_all_fake_factories(),
        )
        assert "claude_core" not in invoker._unavailable_slots
        assert "gpt_core" not in invoker._unavailable_slots
        assert "deepseek_heterodox" not in invoker._unavailable_slots
        assert "kimi_heterodox" not in invoker._unavailable_slots
        assert "qwen_heterodox" not in invoker._unavailable_slots
        assert "mistral_regulatory" not in invoker._unavailable_slots
        # Missing-key families still blocked.
        assert "gemini_heterodox" in invoker._unavailable_slots
        assert "grok_heterodox" in invoker._unavailable_slots
        # And their agents are None.
        assert invoker._agents[FAMILY_GEMINI] is None
        assert invoker._agents[FAMILY_GROK] is None

    def test_heterodox_model_override_envs(self) -> None:
        calls: dict[str, list[tuple[str, str | None]]] = {
            "gemini": [],
            "grok": [],
            "openrouter": [],
            "mistral": [],
        }

        def _make(kind: str):
            def _fn(model: str, api_key: str | None) -> Any:
                calls[kind].append((model, api_key))
                return _fake_agent(model, api_key)

            return _fn

        build_default_invoker(
            config=_config(),
            env={
                "ANTHROPIC_API_KEY": "sk-ant",
                "OPENAI_API_KEY": "sk-oa",
                "GEMINI_API_KEY": "goog",
                "GROK_API_KEY": "grok",
                "OPENROUTER_API_KEY": "or",
                "MISTRAL_API_KEY": "mi",
                GEMINI_MODEL_ENV: "gemini-3-flash-preview",
                GROK_MODEL_ENV: "grok-4-fast",
                DEEPSEEK_MODEL_ENV: "deepseek/deepseek-r1",
                KIMI_MODEL_ENV: "moonshotai/kimi-k2-thinking",
                QWEN_MODEL_ENV: "qwen/qwen3-max",
                MISTRAL_MODEL_ENV: "mistral-medium-latest",
            },
            anthropic_agent_factory=_fake_claude,
            openai_agent_factory=_fake_gpt,
            gemini_agent_factory=_make("gemini"),
            grok_agent_factory=_make("grok"),
            openrouter_agent_factory=_make("openrouter"),
            mistral_agent_factory=_make("mistral"),
        )
        assert calls["gemini"] == [("gemini-3-flash-preview", "goog")]
        assert calls["grok"] == [("grok-4-fast", "grok")]
        # Openrouter factory is called once per family.
        openrouter_models = {m for m, _ in calls["openrouter"]}
        assert openrouter_models == {
            "deepseek/deepseek-r1",
            "moonshotai/kimi-k2-thinking",
            "qwen/qwen3-max",
        }
        assert calls["mistral"] == [("mistral-medium-latest", "mi")]
