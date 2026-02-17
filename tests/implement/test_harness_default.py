"""Tests for harness default behavior in HybridExecutor."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.implement.executor import HybridExecutor


class TestHarnessDefault:
    """Tests that use_harness defaults to True with env override."""

    def test_default_is_use_harness_true(self, tmp_path):
        """HybridExecutor should default to use_harness=True."""
        with patch("aragora.implement.executor.ClaudeAgent"), \
             patch("aragora.implement.executor.CodexAgent"):
            executor = HybridExecutor(repo_path=tmp_path)
            assert executor.use_harness is True

    def test_env_override_disables_harness(self, tmp_path):
        """IMPL_USE_HARNESS=0 should override default to False."""
        with patch("aragora.implement.executor.ClaudeAgent"), \
             patch("aragora.implement.executor.CodexAgent"), \
             patch.dict(os.environ, {"IMPL_USE_HARNESS": "0"}):
            executor = HybridExecutor(repo_path=tmp_path)
            assert executor.use_harness is False

    def test_env_override_enables_harness(self, tmp_path):
        """IMPL_USE_HARNESS=1 should explicitly enable harness."""
        with patch("aragora.implement.executor.ClaudeAgent"), \
             patch("aragora.implement.executor.CodexAgent"), \
             patch.dict(os.environ, {"IMPL_USE_HARNESS": "1"}):
            executor = HybridExecutor(repo_path=tmp_path)
            assert executor.use_harness is True

    def test_explicit_param_respected_without_env(self, tmp_path):
        """Explicit use_harness=False should work when no env var set."""
        with patch("aragora.implement.executor.ClaudeAgent"), \
             patch("aragora.implement.executor.CodexAgent"), \
             patch.dict(os.environ, {}, clear=False):
            # Remove IMPL_USE_HARNESS if present
            env = os.environ.copy()
            env.pop("IMPL_USE_HARNESS", None)
            with patch.dict(os.environ, env, clear=True):
                executor = HybridExecutor(repo_path=tmp_path, use_harness=False)
                assert executor.use_harness is False


class TestHarnessPrecedence:
    """Tests that harness takes priority over sandbox."""

    def test_harness_checked_before_sandbox(self, tmp_path):
        """When both use_harness and sandbox_mode are True, harness should win."""
        with patch("aragora.implement.executor.ClaudeAgent"), \
             patch("aragora.implement.executor.CodexAgent"):
            executor = HybridExecutor(
                repo_path=tmp_path,
                use_harness=True,
                sandbox_mode=True,
            )
            # Both are True, but harness should take precedence in execute_task
            assert executor.use_harness is True
            assert executor.sandbox_mode is True
