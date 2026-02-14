"""
Tests for RLM Tier 1 deep integration features.

Covers:
- RLMResult trajectory fields (trajectory_log_path, rlm_iterations, code_blocks_executed)
- RLMBackendConfig new fields (sub_backend, sub_backend_model, trajectory_log_dir, custom_system_prompt)
- DEBATE_RLM_SYSTEM_PROMPT constant content
- _init_official_rlm() behaviour (trajectory logging, multi-backend routing, custom system prompt, env overrides)
- Lifecycle methods (close, __enter__/__exit__, get_trajectory_log_path)
- create_aragora_rlm() new params (sub_backend, sub_model, trajectory_log_dir, persistent, debate_mode)
- Prompt instructions use llm_query() not RLM_M()
- RLMResult trajectory data extraction from official RLM logger
"""

from __future__ import annotations

import os
import sys
import types as builtin_types
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Helpers – fake 'rlm' package module so imports inside bridge.py succeed
# when we flip HAS_OFFICIAL_RLM to True for certain tests.
# ---------------------------------------------------------------------------

def _make_fake_rlm_module():
    """Create a minimal fake ``rlm`` package with ``rlm.RLM`` and ``rlm.logger.RLMLogger``."""
    fake_rlm = builtin_types.ModuleType("rlm")
    fake_rlm.RLM = MagicMock(name="FakeOfficialRLM_class")
    # sub-module rlm.logger
    fake_logger_mod = builtin_types.ModuleType("rlm.logger")
    fake_logger_mod.RLMLogger = MagicMock(name="FakeRLMLogger_class")
    fake_rlm.logger = fake_logger_mod
    return fake_rlm, fake_logger_mod


# ===================================================================
# 1. RLMResult trajectory fields
# ===================================================================

class TestRLMResultTrajectoryFields:
    """Verify the three new trajectory fields on RLMResult."""

    def test_trajectory_log_path_default_none(self):
        from aragora.rlm.types import RLMResult
        r = RLMResult(answer="ok")
        assert r.trajectory_log_path is None

    def test_trajectory_log_path_set(self):
        from aragora.rlm.types import RLMResult
        r = RLMResult(answer="ok", trajectory_log_path="/tmp/traj.jsonl")
        assert r.trajectory_log_path == "/tmp/traj.jsonl"

    def test_rlm_iterations_default_zero(self):
        from aragora.rlm.types import RLMResult
        r = RLMResult(answer="ok")
        assert r.rlm_iterations == 0

    def test_rlm_iterations_set(self):
        from aragora.rlm.types import RLMResult
        r = RLMResult(answer="ok", rlm_iterations=7)
        assert r.rlm_iterations == 7

    def test_code_blocks_executed_default_zero(self):
        from aragora.rlm.types import RLMResult
        r = RLMResult(answer="ok")
        assert r.code_blocks_executed == 0

    def test_code_blocks_executed_set(self):
        from aragora.rlm.types import RLMResult
        r = RLMResult(answer="ok", code_blocks_executed=42)
        assert r.code_blocks_executed == 42

    def test_all_trajectory_fields_together(self):
        from aragora.rlm.types import RLMResult
        r = RLMResult(
            answer="done",
            trajectory_log_path="/logs/run.jsonl",
            rlm_iterations=5,
            code_blocks_executed=12,
        )
        assert r.trajectory_log_path == "/logs/run.jsonl"
        assert r.rlm_iterations == 5
        assert r.code_blocks_executed == 12


# ===================================================================
# 2. RLMBackendConfig new fields
# ===================================================================

class TestRLMBackendConfigNewFields:
    """Verify the four new configuration fields on RLMBackendConfig."""

    def test_sub_backend_default_none(self):
        from aragora.rlm.bridge import RLMBackendConfig
        cfg = RLMBackendConfig()
        assert cfg.sub_backend is None

    def test_sub_backend_model_default_none(self):
        from aragora.rlm.bridge import RLMBackendConfig
        cfg = RLMBackendConfig()
        assert cfg.sub_backend_model is None

    def test_trajectory_log_dir_default_none(self):
        from aragora.rlm.bridge import RLMBackendConfig
        cfg = RLMBackendConfig()
        assert cfg.trajectory_log_dir is None

    def test_custom_system_prompt_default_none(self):
        from aragora.rlm.bridge import RLMBackendConfig
        cfg = RLMBackendConfig()
        assert cfg.custom_system_prompt is None

    def test_sub_backend_set(self):
        from aragora.rlm.bridge import RLMBackendConfig
        cfg = RLMBackendConfig(sub_backend="anthropic")
        assert cfg.sub_backend == "anthropic"

    def test_sub_backend_model_set(self):
        from aragora.rlm.bridge import RLMBackendConfig
        cfg = RLMBackendConfig(sub_backend_model="claude-3-haiku")
        assert cfg.sub_backend_model == "claude-3-haiku"

    def test_trajectory_log_dir_set(self):
        from aragora.rlm.bridge import RLMBackendConfig
        cfg = RLMBackendConfig(trajectory_log_dir="/tmp/traj")
        assert cfg.trajectory_log_dir == "/tmp/traj"

    def test_custom_system_prompt_set(self):
        from aragora.rlm.bridge import RLMBackendConfig
        cfg = RLMBackendConfig(custom_system_prompt="You are a helpful bot.")
        assert cfg.custom_system_prompt == "You are a helpful bot."

    def test_all_new_fields_together(self):
        from aragora.rlm.bridge import RLMBackendConfig
        cfg = RLMBackendConfig(
            sub_backend="openrouter",
            sub_backend_model="deepseek/deepseek-chat",
            trajectory_log_dir="/var/log/rlm",
            custom_system_prompt="Analyse carefully.",
        )
        assert cfg.sub_backend == "openrouter"
        assert cfg.sub_backend_model == "deepseek/deepseek-chat"
        assert cfg.trajectory_log_dir == "/var/log/rlm"
        assert cfg.custom_system_prompt == "Analyse carefully."


# ===================================================================
# 3. DEBATE_RLM_SYSTEM_PROMPT constant
# ===================================================================

class TestDebateRLMSystemPrompt:
    """Verify the debate-optimised system prompt contains required elements."""

    def _prompt(self):
        from aragora.rlm.bridge import DEBATE_RLM_SYSTEM_PROMPT
        return DEBATE_RLM_SYSTEM_PROMPT

    def test_contains_llm_query(self):
        assert "llm_query" in self._prompt()

    def test_contains_llm_query_batched(self):
        assert "llm_query_batched" in self._prompt()

    def test_contains_FINAL(self):
        assert "FINAL(" in self._prompt()

    def test_contains_FINAL_VAR(self):
        assert "FINAL_VAR" in self._prompt()

    def test_contains_debate_navigation_load(self):
        assert "load_debate_context" in self._prompt()

    def test_contains_debate_navigation_get_round(self):
        assert "get_round" in self._prompt()

    def test_contains_debate_navigation_get_proposals(self):
        assert "get_proposals_by_agent" in self._prompt()

    def test_contains_debate_navigation_search(self):
        assert "search_debate" in self._prompt()

    def test_contains_debate_navigation_partition(self):
        assert "partition_debate" in self._prompt()

    def test_contains_knowledge_navigation(self):
        prompt = self._prompt()
        assert "load_knowledge_context" in prompt
        assert "get_facts" in prompt
        assert "search_knowledge" in prompt

    def test_contains_memory_navigation(self):
        prompt = self._prompt()
        assert "load_memory_context" in prompt
        assert "search_memory" in prompt
        assert "filter_by_importance" in prompt

    def test_contains_strategy_guidance(self):
        prompt = self._prompt()
        assert "Strategy Guidance" in prompt

    def test_strategy_uses_batched_queries(self):
        prompt = self._prompt()
        assert "llm_query_batched" in prompt

    def test_prompt_is_string(self):
        assert isinstance(self._prompt(), str)


# ===================================================================
# 4. _init_official_rlm() behaviour (mocked)
# ===================================================================

class TestInitOfficialRLM:
    """Verify _init_official_rlm() wiring when the official rlm package is mocked."""

    def _build_rlm_with_mock(
        self,
        *,
        sub_backend: str | None = None,
        sub_backend_model: str | None = None,
        trajectory_log_dir: str | None = None,
        custom_system_prompt: str | None = None,
        env_overrides: dict[str, str] | None = None,
    ):
        """Construct AragoraRLM with a mocked official RLM package."""
        fake_rlm_mod, fake_logger_mod = _make_fake_rlm_module()
        mock_rlm_cls = MagicMock(name="OfficialRLM")
        fake_rlm_mod.RLM = mock_rlm_cls

        from aragora.rlm.bridge import RLMBackendConfig

        cfg = RLMBackendConfig(
            sub_backend=sub_backend,
            sub_backend_model=sub_backend_model,
            trajectory_log_dir=trajectory_log_dir,
            custom_system_prompt=custom_system_prompt,
        )

        env = env_overrides or {}
        with patch.dict(os.environ, env, clear=False):
            with patch.dict(sys.modules, {"rlm": fake_rlm_mod, "rlm.logger": fake_logger_mod}):
                with patch("aragora.rlm.bridge.HAS_OFFICIAL_RLM", True):
                    with patch("aragora.rlm.bridge.OfficialRLM", mock_rlm_cls):
                        from aragora.rlm.bridge import AragoraRLM
                        instance = AragoraRLM(backend_config=cfg)
        return instance, mock_rlm_cls, fake_logger_mod

    # -- Trajectory logging --

    def test_trajectory_logging_creates_logger(self, tmp_path):
        """When trajectory_log_dir is set and RLMLogger is importable, a logger should be created."""
        log_dir = str(tmp_path / "traj")
        inst, mock_cls, fake_logger_mod = self._build_rlm_with_mock(
            trajectory_log_dir=log_dir,
        )
        # The RLMLogger constructor should have been called with log_dir
        fake_logger_mod.RLMLogger.assert_called_once_with(log_dir=log_dir)

    def test_trajectory_logging_passed_to_rlm(self, tmp_path):
        """The logger instance should appear in the OfficialRLM init kwargs."""
        log_dir = str(tmp_path / "traj2")
        inst, mock_cls, fake_logger_mod = self._build_rlm_with_mock(
            trajectory_log_dir=log_dir,
        )
        # OfficialRLM should have been called; check 'logger' kwarg
        init_call = mock_cls.call_args
        assert "logger" in init_call.kwargs or any(
            "logger" in str(c) for c in init_call
        ), "Expected 'logger' kwarg to be passed to OfficialRLM"

    def test_trajectory_log_dir_creates_directory(self, tmp_path):
        """The log directory should be created if it doesn't exist."""
        log_dir = str(tmp_path / "new_traj_dir")
        self._build_rlm_with_mock(trajectory_log_dir=log_dir)
        assert (tmp_path / "new_traj_dir").exists()

    def test_no_trajectory_logging_when_dir_not_set(self):
        """When trajectory_log_dir is None, no RLMLogger should be created."""
        inst, mock_cls, fake_logger_mod = self._build_rlm_with_mock(
            trajectory_log_dir=None,
        )
        fake_logger_mod.RLMLogger.assert_not_called()

    # -- Multi-backend routing --

    def test_sub_backend_passes_other_backends(self):
        """When sub_backend is set, other_backends should be passed to OfficialRLM."""
        inst, mock_cls, _ = self._build_rlm_with_mock(
            sub_backend="anthropic",
            sub_backend_model="claude-3-haiku",
        )
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs.get("other_backends") == ["anthropic"]
        assert call_kwargs.get("other_backend_kwargs") == [{"model_name": "claude-3-haiku"}]

    def test_sub_backend_without_model_uses_sub_model_name(self):
        """When sub_backend_model is None, should fall back to sub_model_name."""
        from aragora.rlm.bridge import RLMBackendConfig
        # sub_model_name defaults to "gpt-4o-mini"
        inst, mock_cls, _ = self._build_rlm_with_mock(
            sub_backend="openrouter",
            sub_backend_model=None,
        )
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs.get("other_backends") == ["openrouter"]
        # should fall back to sub_model_name (gpt-4o-mini) since no explicit sub_backend_model
        backend_kwargs = call_kwargs.get("other_backend_kwargs", [{}])
        assert backend_kwargs[0]["model_name"] == "gpt-4o-mini"

    def test_no_sub_backend_means_no_other_backends(self):
        """When sub_backend is None, other_backends should NOT be in kwargs."""
        inst, mock_cls, _ = self._build_rlm_with_mock(sub_backend=None)
        call_kwargs = mock_cls.call_args.kwargs
        assert "other_backends" not in call_kwargs

    # -- Custom system prompt --

    def test_custom_system_prompt_passed(self):
        """When custom_system_prompt is set, system_prompt kwarg should be passed."""
        inst, mock_cls, _ = self._build_rlm_with_mock(
            custom_system_prompt="Be concise.",
        )
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs.get("system_prompt") == "Be concise."

    def test_no_custom_system_prompt_means_no_kwarg(self):
        """When custom_system_prompt is None and no env var, system_prompt should NOT be in kwargs."""
        inst, mock_cls, _ = self._build_rlm_with_mock(
            custom_system_prompt=None,
        )
        call_kwargs = mock_cls.call_args.kwargs
        assert "system_prompt" not in call_kwargs

    def test_system_prompt_env_var_override(self):
        """ARAGORA_RLM_SYSTEM_PROMPT env var should serve as fallback for system_prompt."""
        inst, mock_cls, _ = self._build_rlm_with_mock(
            custom_system_prompt=None,
            env_overrides={"ARAGORA_RLM_SYSTEM_PROMPT": "env prompt"},
        )
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs.get("system_prompt") == "env prompt"

    def test_custom_system_prompt_overrides_env_var(self):
        """Explicit custom_system_prompt should take priority over env var."""
        inst, mock_cls, _ = self._build_rlm_with_mock(
            custom_system_prompt="explicit prompt",
            env_overrides={"ARAGORA_RLM_SYSTEM_PROMPT": "env prompt"},
        )
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs.get("system_prompt") == "explicit prompt"

    # -- Env var override for sub_model --

    def test_env_var_sub_model_fallback(self):
        """ARAGORA_RLM_SUB_MODEL env var should be used when sub_backend_model is None."""
        inst, mock_cls, _ = self._build_rlm_with_mock(
            sub_backend="openai",
            sub_backend_model=None,
            env_overrides={"ARAGORA_RLM_SUB_MODEL": "gpt-4o-mini-from-env"},
        )
        call_kwargs = mock_cls.call_args.kwargs
        backend_kwargs = call_kwargs.get("other_backend_kwargs", [{}])
        assert backend_kwargs[0]["model_name"] == "gpt-4o-mini-from-env"


# ===================================================================
# 5. Lifecycle methods
# ===================================================================

class TestLifecycleMethods:
    """Verify close(), __enter__/__exit__, and get_trajectory_log_path()."""

    def test_close_calls_official_close(self):
        from aragora.rlm.bridge import AragoraRLM
        rlm = AragoraRLM()
        mock_official = MagicMock()
        mock_official.close = MagicMock()
        rlm._official_rlm = mock_official
        rlm.close()
        mock_official.close.assert_called_once()

    def test_close_calls_fallback_close(self):
        from aragora.rlm.bridge import AragoraRLM
        rlm = AragoraRLM()
        mock_fallback = MagicMock()
        mock_fallback.close = MagicMock()
        rlm._fallback_rlm = mock_fallback
        rlm.close()
        mock_fallback.close.assert_called_once()

    def test_close_calls_both_official_and_fallback(self):
        from aragora.rlm.bridge import AragoraRLM
        rlm = AragoraRLM()
        mock_official = MagicMock()
        mock_fallback = MagicMock()
        rlm._official_rlm = mock_official
        rlm._fallback_rlm = mock_fallback
        rlm.close()
        mock_official.close.assert_called_once()
        mock_fallback.close.assert_called_once()

    def test_close_handles_missing_close_attr_on_official(self):
        from aragora.rlm.bridge import AragoraRLM
        rlm = AragoraRLM()
        # Object without close attribute
        rlm._official_rlm = object()
        # Should not raise
        rlm.close()

    def test_close_handles_exception_in_official(self):
        from aragora.rlm.bridge import AragoraRLM
        rlm = AragoraRLM()
        mock_official = MagicMock()
        mock_official.close.side_effect = RuntimeError("boom")
        rlm._official_rlm = mock_official
        # Should not raise
        rlm.close()

    def test_close_handles_exception_in_fallback(self):
        from aragora.rlm.bridge import AragoraRLM
        rlm = AragoraRLM()
        mock_fallback = MagicMock()
        mock_fallback.close.side_effect = RuntimeError("boom")
        rlm._fallback_rlm = mock_fallback
        # Should not raise
        rlm.close()

    def test_context_manager_enter_returns_self(self):
        from aragora.rlm.bridge import AragoraRLM
        rlm = AragoraRLM()
        result = rlm.__enter__()
        assert result is rlm

    def test_context_manager_exit_calls_close(self):
        from aragora.rlm.bridge import AragoraRLM
        rlm = AragoraRLM()
        rlm.close = MagicMock()
        rlm.__exit__(None, None, None)
        rlm.close.assert_called_once()

    def test_context_manager_exit_returns_false(self):
        from aragora.rlm.bridge import AragoraRLM
        rlm = AragoraRLM()
        result = rlm.__exit__(None, None, None)
        assert result is False

    def test_context_manager_with_statement(self):
        from aragora.rlm.bridge import AragoraRLM
        with AragoraRLM() as rlm:
            assert isinstance(rlm, AragoraRLM)
        # close is called implicitly via __exit__

    def test_get_trajectory_log_path_default_none(self):
        from aragora.rlm.bridge import AragoraRLM
        rlm = AragoraRLM()
        assert rlm.get_trajectory_log_path() is None

    def test_get_trajectory_log_path_returns_dir(self):
        from aragora.rlm.bridge import AragoraRLM, RLMBackendConfig
        cfg = RLMBackendConfig(trajectory_log_dir="/var/log/rlm_traj")
        rlm = AragoraRLM(backend_config=cfg)
        assert rlm.get_trajectory_log_path() == "/var/log/rlm_traj"

    def test_trajectory_log_dir_stored_on_instance(self):
        from aragora.rlm.bridge import AragoraRLM, RLMBackendConfig
        cfg = RLMBackendConfig(trajectory_log_dir="/tmp/my_traj")
        rlm = AragoraRLM(backend_config=cfg)
        assert rlm._trajectory_log_dir == "/tmp/my_traj"


# ===================================================================
# 6. create_aragora_rlm() new params
# ===================================================================

class TestCreateAragoraRLMNewParams:
    """Verify create_aragora_rlm() passes through new parameters."""

    def test_sub_backend_passed(self):
        from aragora.rlm.bridge import create_aragora_rlm
        rlm = create_aragora_rlm(sub_backend="anthropic")
        assert rlm.backend_config.sub_backend == "anthropic"

    def test_sub_model_passed(self):
        from aragora.rlm.bridge import create_aragora_rlm
        rlm = create_aragora_rlm(sub_model="claude-3-haiku")
        assert rlm.backend_config.sub_backend_model == "claude-3-haiku"

    def test_trajectory_log_dir_passed(self):
        from aragora.rlm.bridge import create_aragora_rlm
        rlm = create_aragora_rlm(trajectory_log_dir="/tmp/logs")
        assert rlm.backend_config.trajectory_log_dir == "/tmp/logs"
        assert rlm.get_trajectory_log_path() == "/tmp/logs"

    def test_persistent_passed(self):
        from aragora.rlm.bridge import create_aragora_rlm
        rlm = create_aragora_rlm(persistent=True)
        assert rlm.backend_config.persistent is True

    def test_persistent_default_false(self):
        from aragora.rlm.bridge import create_aragora_rlm
        rlm = create_aragora_rlm()
        assert rlm.backend_config.persistent is False

    def test_debate_mode_sets_system_prompt(self):
        from aragora.rlm.bridge import create_aragora_rlm, DEBATE_RLM_SYSTEM_PROMPT
        rlm = create_aragora_rlm(debate_mode=True)
        assert rlm.backend_config.custom_system_prompt == DEBATE_RLM_SYSTEM_PROMPT

    def test_debate_mode_false_no_system_prompt(self):
        from aragora.rlm.bridge import create_aragora_rlm
        rlm = create_aragora_rlm(debate_mode=False)
        assert rlm.backend_config.custom_system_prompt is None

    def test_all_new_params_together(self):
        from aragora.rlm.bridge import create_aragora_rlm, DEBATE_RLM_SYSTEM_PROMPT
        rlm = create_aragora_rlm(
            sub_backend="openrouter",
            sub_model="deepseek/deepseek-chat",
            trajectory_log_dir="/opt/traj",
            persistent=True,
            debate_mode=True,
        )
        assert rlm.backend_config.sub_backend == "openrouter"
        assert rlm.backend_config.sub_backend_model == "deepseek/deepseek-chat"
        assert rlm.backend_config.trajectory_log_dir == "/opt/traj"
        assert rlm.backend_config.persistent is True
        assert rlm.backend_config.custom_system_prompt == DEBATE_RLM_SYSTEM_PROMPT

    def test_backend_and_model_passed(self):
        from aragora.rlm.bridge import create_aragora_rlm
        rlm = create_aragora_rlm(backend="anthropic", model="claude-3-opus")
        assert rlm.backend_config.backend == "anthropic"
        assert rlm.backend_config.model_name == "claude-3-opus"

    def test_verbose_passed(self):
        from aragora.rlm.bridge import create_aragora_rlm
        rlm = create_aragora_rlm(verbose=True)
        assert rlm.backend_config.verbose is True


# ===================================================================
# 7. Prompt instructions use llm_query() not RLM_M()
# ===================================================================

class TestPromptUsesLlmQuery:
    """Verify prompts reference llm_query() rather than the old RLM_M() API."""

    def test_true_rlm_query_prompt_contains_llm_query(self):
        """The _true_rlm_query prompt should reference llm_query(prompt)."""
        from aragora.rlm.bridge import AragoraRLM
        import inspect
        source = inspect.getsource(AragoraRLM._true_rlm_query)
        assert "llm_query(" in source or "llm_query(prompt)" in source

    def test_true_rlm_query_prompt_not_contains_rlm_m(self):
        """The _true_rlm_query prompt should NOT reference RLM_M(prompt)."""
        from aragora.rlm.bridge import AragoraRLM
        import inspect
        source = inspect.getsource(AragoraRLM._true_rlm_query)
        # It should not contain RLM_M( in the prompt strings
        # (filter out any comments)
        lines = [l for l in source.split("\n") if not l.strip().startswith("#")]
        prompt_section = "\n".join(lines)
        assert "RLM_M(" not in prompt_section

    def test_default_feedback_mentions_llm_query(self):
        """The _default_feedback method should mention llm_query() not RLM_M()."""
        from aragora.rlm.bridge import AragoraRLM
        import inspect
        source = inspect.getsource(AragoraRLM._default_feedback)
        assert "llm_query()" in source

    def test_default_feedback_not_mentions_rlm_m(self):
        """The _default_feedback method should NOT mention RLM_M()."""
        from aragora.rlm.bridge import AragoraRLM
        import inspect
        source = inspect.getsource(AragoraRLM._default_feedback)
        assert "RLM_M()" not in source

    def test_system_prompt_uses_llm_query(self):
        """DEBATE_RLM_SYSTEM_PROMPT should reference llm_query not RLM_M."""
        from aragora.rlm.bridge import DEBATE_RLM_SYSTEM_PROMPT
        assert "llm_query(" in DEBATE_RLM_SYSTEM_PROMPT
        assert "RLM_M(" not in DEBATE_RLM_SYSTEM_PROMPT


# ===================================================================
# 8. RLMResult trajectory data extraction
# ===================================================================

class TestRLMResultTrajectoryExtraction:
    """Verify that trajectory data from the official RLM logger propagates to RLMResult."""

    def _build_rlm_with_logger(self, *, log_path="/tmp/log.jsonl", iterations=5, code_blocks=10):
        """Build an AragoraRLM whose _official_rlm has a logger with stats."""
        from aragora.rlm.bridge import AragoraRLM

        rlm = AragoraRLM()

        # Mock the official RLM
        mock_official = MagicMock()
        mock_completion = MagicMock()
        mock_completion.response = "The answer is 42"
        mock_completion.execution_time = 1.5
        mock_official.completion.return_value = mock_completion

        # Mock the logger on the official RLM
        mock_logger = MagicMock()
        mock_logger.log_path = log_path
        mock_logger.get_stats.return_value = {
            "iterations": iterations,
            "code_blocks": code_blocks,
        }
        mock_official.logger = mock_logger

        rlm._official_rlm = mock_official
        return rlm

    @pytest.mark.asyncio
    async def test_trajectory_log_path_propagated(self):
        rlm = self._build_rlm_with_logger(log_path="/tmp/traj_out.jsonl")
        from aragora.rlm.types import RLMContext
        ctx = RLMContext(original_content="test", original_tokens=10)
        result = await rlm._true_rlm_query("question", ctx, "auto")
        assert result.trajectory_log_path == "/tmp/traj_out.jsonl"

    @pytest.mark.asyncio
    async def test_rlm_iterations_propagated(self):
        rlm = self._build_rlm_with_logger(iterations=8)
        from aragora.rlm.types import RLMContext
        ctx = RLMContext(original_content="test", original_tokens=10)
        result = await rlm._true_rlm_query("question", ctx, "auto")
        assert result.rlm_iterations == 8

    @pytest.mark.asyncio
    async def test_code_blocks_executed_propagated(self):
        rlm = self._build_rlm_with_logger(code_blocks=15)
        from aragora.rlm.types import RLMContext
        ctx = RLMContext(original_content="test", original_tokens=10)
        result = await rlm._true_rlm_query("question", ctx, "auto")
        assert result.code_blocks_executed == 15

    @pytest.mark.asyncio
    async def test_no_logger_means_default_values(self):
        """When official RLM has no logger, trajectory fields should be None/0."""
        from aragora.rlm.bridge import AragoraRLM
        from aragora.rlm.types import RLMContext

        rlm = AragoraRLM()
        mock_official = MagicMock()
        mock_completion = MagicMock()
        mock_completion.response = "answer"
        mock_completion.execution_time = 0.5
        mock_official.completion.return_value = mock_completion
        mock_official.logger = None
        rlm._official_rlm = mock_official

        ctx = RLMContext(original_content="test", original_tokens=10)
        result = await rlm._true_rlm_query("question", ctx, "auto")
        assert result.trajectory_log_path is None
        assert result.rlm_iterations == 0
        assert result.code_blocks_executed == 0

    @pytest.mark.asyncio
    async def test_logger_without_get_stats(self):
        """When logger exists but has no get_stats(), iterations/blocks should be 0."""
        from aragora.rlm.bridge import AragoraRLM
        from aragora.rlm.types import RLMContext

        rlm = AragoraRLM()
        mock_official = MagicMock()
        mock_completion = MagicMock()
        mock_completion.response = "answer"
        mock_completion.execution_time = 0.5
        mock_official.completion.return_value = mock_completion

        # Logger without get_stats
        mock_logger = MagicMock(spec=[])
        mock_logger.log_path = "/tmp/log.jsonl"
        # Make hasattr return False for get_stats
        del mock_logger.get_stats
        mock_official.logger = mock_logger
        rlm._official_rlm = mock_official

        ctx = RLMContext(original_content="test", original_tokens=10)
        result = await rlm._true_rlm_query("question", ctx, "auto")
        assert result.trajectory_log_path == "/tmp/log.jsonl"
        assert result.rlm_iterations == 0
        assert result.code_blocks_executed == 0

    @pytest.mark.asyncio
    async def test_logger_without_log_path(self):
        """When logger exists but has no log_path, trajectory_log_path should be None."""
        from aragora.rlm.bridge import AragoraRLM
        from aragora.rlm.types import RLMContext

        rlm = AragoraRLM()
        mock_official = MagicMock()
        mock_completion = MagicMock()
        mock_completion.response = "answer"
        mock_completion.execution_time = 0.5
        mock_official.completion.return_value = mock_completion

        # Logger without log_path attr
        mock_logger = MagicMock()
        del mock_logger.log_path
        mock_logger.get_stats.return_value = {"iterations": 3, "code_blocks": 7}
        mock_official.logger = mock_logger
        rlm._official_rlm = mock_official

        ctx = RLMContext(original_content="test", original_tokens=10)
        result = await rlm._true_rlm_query("question", ctx, "auto")
        assert result.trajectory_log_path is None
        assert result.rlm_iterations == 3
        assert result.code_blocks_executed == 7

    @pytest.mark.asyncio
    async def test_result_used_true_rlm_flag(self):
        """After successful _true_rlm_query, used_true_rlm should be True."""
        rlm = self._build_rlm_with_logger()
        from aragora.rlm.types import RLMContext
        ctx = RLMContext(original_content="test", original_tokens=10)
        result = await rlm._true_rlm_query("question", ctx, "auto")
        assert rlm._last_query_used_true_rlm is True

    @pytest.mark.asyncio
    async def test_result_answer_from_completion(self):
        """The answer in RLMResult should come from the completion response."""
        rlm = self._build_rlm_with_logger()
        from aragora.rlm.types import RLMContext
        ctx = RLMContext(original_content="test", original_tokens=10)
        result = await rlm._true_rlm_query("question", ctx, "auto")
        assert result.answer == "The answer is 42"

    @pytest.mark.asyncio
    async def test_result_confidence_is_default(self):
        """TRUE RLM result should have default confidence of 0.8."""
        rlm = self._build_rlm_with_logger()
        from aragora.rlm.types import RLMContext
        ctx = RLMContext(original_content="test", original_tokens=10)
        result = await rlm._true_rlm_query("question", ctx, "auto")
        assert result.confidence == 0.8
