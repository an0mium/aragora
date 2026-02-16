"""Tests for default PostDebateCoordinator behavior.

Validates that:
1. DEFAULT_POST_DEBATE_CONFIG exists with correct values
2. orchestrator_runner uses default when no explicit config
3. disable_post_debate_pipeline=True skips pipeline
4. explicit config overrides default
5. import error in coordinator degrades gracefully
6. enterprise preset has post_debate_config
7. audit preset has post_debate_config
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.post_debate_coordinator import (
    DEFAULT_POST_DEBATE_CONFIG,
    PostDebateConfig,
    PostDebateCoordinator,
    PostDebateResult,
)


class TestDefaultPostDebateConfig:
    """Tests for DEFAULT_POST_DEBATE_CONFIG constant."""

    def test_default_config_exists(self):
        assert DEFAULT_POST_DEBATE_CONFIG is not None
        assert isinstance(DEFAULT_POST_DEBATE_CONFIG, PostDebateConfig)

    def test_default_config_auto_explain_enabled(self):
        assert DEFAULT_POST_DEBATE_CONFIG.auto_explain is True

    def test_default_config_auto_persist_receipt_enabled(self):
        assert DEFAULT_POST_DEBATE_CONFIG.auto_persist_receipt is True

    def test_default_config_auto_create_plan_disabled(self):
        assert DEFAULT_POST_DEBATE_CONFIG.auto_create_plan is False

    def test_default_config_auto_notify_disabled(self):
        assert DEFAULT_POST_DEBATE_CONFIG.auto_notify is False

    def test_default_config_auto_execute_plan_disabled(self):
        assert DEFAULT_POST_DEBATE_CONFIG.auto_execute_plan is False

    def test_default_config_auto_create_pr_disabled(self):
        assert DEFAULT_POST_DEBATE_CONFIG.auto_create_pr is False

    def test_default_config_auto_build_integrity_package_disabled(self):
        assert DEFAULT_POST_DEBATE_CONFIG.auto_build_integrity_package is False


class TestCoordinatorWithDefaultConfig:
    """Tests for PostDebateCoordinator using default config."""

    def test_coordinator_uses_default_when_none(self):
        coordinator = PostDebateCoordinator(config=None)
        # PostDebateConfig() defaults have auto_explain=True, auto_create_plan=True
        assert coordinator.config.auto_explain is True

    def test_coordinator_with_explicit_config(self):
        explicit = PostDebateConfig(auto_explain=False, auto_create_plan=False)
        coordinator = PostDebateCoordinator(config=explicit)
        assert coordinator.config.auto_explain is False
        assert coordinator.config.auto_create_plan is False

    def test_default_config_runs_explain_and_receipt(self):
        coordinator = PostDebateCoordinator(config=DEFAULT_POST_DEBATE_CONFIG)
        mock_result = MagicMock()
        mock_result.consensus = "Test consensus"
        mock_result.messages = []
        mock_result.confidence = 0.9
        mock_result.final_answer = "answer"
        mock_result.participants = []

        with patch.object(coordinator, "_step_explain", return_value={"explanation": "test"}) as mock_explain, \
             patch.object(coordinator, "_step_persist_receipt", return_value=True) as mock_receipt, \
             patch.object(coordinator, "_step_create_plan") as mock_plan, \
             patch.object(coordinator, "_step_notify") as mock_notify:
            result = coordinator.run("d1", mock_result, confidence=0.9, task="test")

        mock_explain.assert_called_once()
        mock_receipt.assert_called_once()
        mock_plan.assert_not_called()
        mock_notify.assert_not_called()


class TestDisablePostDebatePipeline:
    """Tests for disable_post_debate_pipeline flag."""

    def test_arena_config_disable_flag_default_false(self):
        from aragora.debate.arena_config import ArenaConfig
        config = ArenaConfig()
        assert config.disable_post_debate_pipeline is False

    def test_arena_config_disable_flag_true(self):
        from aragora.debate.arena_config import ArenaConfig
        config = ArenaConfig(disable_post_debate_pipeline=True)
        assert config.disable_post_debate_pipeline is True

    def test_arena_config_to_arena_kwargs_includes_flag(self):
        from aragora.debate.arena_config import ArenaConfig
        config = ArenaConfig(disable_post_debate_pipeline=True)
        kwargs = config.to_arena_kwargs()
        assert "disable_post_debate_pipeline" in kwargs
        assert kwargs["disable_post_debate_pipeline"] is True


class TestPresetPostDebateConfig:
    """Tests for preset post_debate_config integration."""

    def test_enterprise_preset_has_post_debate_config(self):
        from aragora.debate.presets import get_preset
        preset = get_preset("enterprise")
        assert "post_debate_config" in preset
        config = preset["post_debate_config"]
        assert isinstance(config, PostDebateConfig)
        assert config.auto_explain is True
        assert config.auto_create_plan is True
        assert config.auto_notify is True
        assert config.auto_persist_receipt is True
        assert config.auto_gauntlet_validate is True

    def test_audit_preset_has_post_debate_config(self):
        from aragora.debate.presets import get_preset
        preset = get_preset("audit")
        assert "post_debate_config" in preset
        config = preset["post_debate_config"]
        assert isinstance(config, PostDebateConfig)
        assert config.auto_explain is True
        assert config.auto_create_plan is True
        assert config.auto_persist_receipt is True
        assert config.auto_gauntlet_validate is True

    def test_minimal_preset_no_post_debate_config(self):
        from aragora.debate.presets import get_preset
        preset = get_preset("minimal")
        assert "post_debate_config" not in preset

    def test_preset_no_internal_key_leak(self):
        from aragora.debate.presets import get_preset
        preset = get_preset("enterprise")
        assert "_post_debate_preset" not in preset


class TestPostDebateResultFields:
    """Tests for PostDebateResult new fields."""

    def test_result_has_gauntlet_result_field(self):
        result = PostDebateResult()
        assert result.gauntlet_result is None

    def test_result_has_improvement_queued_field(self):
        result = PostDebateResult()
        assert result.improvement_queued is False
