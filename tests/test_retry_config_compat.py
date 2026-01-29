"""
Tests for RetryConfig backward compatibility.

Verifies the `jitter` bool alias maps correctly to `jitter_mode` enum,
enabling modules that use `jitter=True/False` to work with the canonical
RetryConfig from aragora.resilience_patterns.retry.
"""

from __future__ import annotations

import pytest

from aragora.resilience_patterns.retry import JitterMode, RetryConfig


class TestRetryConfigJitterCompat:
    """Tests for jitter backward-compatibility alias."""

    def test_jitter_true_sets_multiplicative_mode(self):
        """Test jitter=True maps to MULTIPLICATIVE jitter mode."""
        config = RetryConfig(jitter=True)
        assert config.jitter_mode == JitterMode.MULTIPLICATIVE

    def test_jitter_false_sets_none_mode(self):
        """Test jitter=False maps to NONE jitter mode."""
        config = RetryConfig(jitter=False)
        assert config.jitter_mode == JitterMode.NONE

    def test_jitter_none_preserves_default(self):
        """Test jitter=None (default) preserves the default jitter_mode."""
        config = RetryConfig()
        assert config.jitter is None
        assert config.jitter_mode == JitterMode.MULTIPLICATIVE

    def test_explicit_jitter_mode_overridden_by_jitter(self):
        """Test that jitter bool takes precedence when both are set."""
        # When both are provided, __post_init__ applies the bool alias
        config = RetryConfig(jitter_mode=JitterMode.FULL, jitter=False)
        assert config.jitter_mode == JitterMode.NONE

    def test_jitter_mode_alone_works(self):
        """Test that setting only jitter_mode still works."""
        config = RetryConfig(jitter_mode=JitterMode.ADDITIVE)
        assert config.jitter_mode == JitterMode.ADDITIVE
        assert config.jitter is None

    def test_calculate_delay_with_jitter_compat(self):
        """Test that delay calculation works with jitter compat alias."""
        config = RetryConfig(jitter=True, base_delay=1.0)
        delay = config.calculate_delay(0)
        # With MULTIPLICATIVE jitter (±25%), delay should be near 1.0
        assert 0.5 <= delay <= 2.0

    def test_calculate_delay_without_jitter(self):
        """Test that jitter=False produces deterministic delays."""
        config = RetryConfig(jitter=False, base_delay=1.0)
        delay = config.calculate_delay(0)
        # With no jitter, delay should be exactly base_delay
        assert delay == 1.0
