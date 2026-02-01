"""
Tests for PostgreSQL pool configuration validation.

Tests cover:
- Valid min/max pool size values accepted without clamping
- POOL_MIN_SIZE < 1 is clamped to 1
- POOL_MAX_SIZE < POOL_MIN_SIZE is clamped to POOL_MIN_SIZE
- POOL_MAX_SIZE > 500 is capped to 500
- Default values pass validation
"""

from __future__ import annotations

import importlib
import logging
from unittest import mock

import pytest


def _reload_pool_module(env_overrides: dict[str, str] | None = None):
    """Reload the postgres_pool module with optional env var overrides.

    Module-level globals are set at import time, so we must reload the module
    to re-evaluate them with different environment variables.
    """
    env = {
        "ARAGORA_POSTGRES_PRIMARY": "",
        "ARAGORA_POSTGRES_REPLICAS": "",
    }
    if env_overrides:
        env.update(env_overrides)

    with mock.patch.dict("os.environ", env, clear=False):
        import aragora.persistence.postgres_pool as pool_mod

        importlib.reload(pool_mod)
        return pool_mod


class TestPoolSizeDefaults:
    """Test that default pool size values are valid."""

    def test_default_values_are_valid(self):
        """Default POOL_MIN_SIZE and POOL_MAX_SIZE pass validation."""
        mod = _reload_pool_module()
        assert mod.POOL_MIN_SIZE >= 1
        assert mod.POOL_MAX_SIZE >= mod.POOL_MIN_SIZE
        assert mod.POOL_MAX_SIZE <= 500

    def test_default_min_size_is_2(self):
        """Default POOL_MIN_SIZE is 2 when env var is not set."""
        mod = _reload_pool_module()
        assert mod.POOL_MIN_SIZE == 2

    def test_default_max_size_is_10(self):
        """Default POOL_MAX_SIZE is 10 when env var is not set."""
        mod = _reload_pool_module()
        assert mod.POOL_MAX_SIZE == 10


class TestPoolMinSizeValidation:
    """Test POOL_MIN_SIZE validation and clamping."""

    def test_valid_min_size_accepted(self):
        """A valid POOL_MIN_SIZE (e.g. 5) is accepted without clamping."""
        mod = _reload_pool_module({"ARAGORA_POSTGRES_POOL_MIN": "5"})
        assert mod.POOL_MIN_SIZE == 5

    def test_min_size_of_1_accepted(self):
        """POOL_MIN_SIZE of exactly 1 is accepted (boundary)."""
        mod = _reload_pool_module({"ARAGORA_POSTGRES_POOL_MIN": "1"})
        assert mod.POOL_MIN_SIZE == 1

    def test_min_size_zero_clamped_to_1(self):
        """POOL_MIN_SIZE of 0 is clamped to 1."""
        mod = _reload_pool_module({"ARAGORA_POSTGRES_POOL_MIN": "0"})
        assert mod.POOL_MIN_SIZE == 1

    def test_min_size_negative_clamped_to_1(self):
        """POOL_MIN_SIZE of -5 is clamped to 1."""
        mod = _reload_pool_module({"ARAGORA_POSTGRES_POOL_MIN": "-5"})
        assert mod.POOL_MIN_SIZE == 1

    def test_min_size_clamping_logs_warning(self, caplog):
        """Clamping POOL_MIN_SIZE logs a warning."""
        with caplog.at_level(logging.WARNING, logger="aragora.persistence.postgres_pool"):
            _reload_pool_module({"ARAGORA_POSTGRES_POOL_MIN": "0"})
        assert any("clamping to 1" in record.message for record in caplog.records)


class TestPoolMaxSizeValidation:
    """Test POOL_MAX_SIZE validation and clamping."""

    def test_valid_max_size_accepted(self):
        """A valid POOL_MAX_SIZE (e.g. 20) is accepted without clamping."""
        mod = _reload_pool_module(
            {
                "ARAGORA_POSTGRES_POOL_MIN": "2",
                "ARAGORA_POSTGRES_POOL_MAX": "20",
            }
        )
        assert mod.POOL_MAX_SIZE == 20

    def test_max_size_equal_to_min_accepted(self):
        """POOL_MAX_SIZE equal to POOL_MIN_SIZE is accepted."""
        mod = _reload_pool_module(
            {
                "ARAGORA_POSTGRES_POOL_MIN": "5",
                "ARAGORA_POSTGRES_POOL_MAX": "5",
            }
        )
        assert mod.POOL_MAX_SIZE == 5
        assert mod.POOL_MIN_SIZE == 5

    def test_max_size_exceeding_500_capped(self):
        """POOL_MAX_SIZE > 500 is capped to 500."""
        mod = _reload_pool_module(
            {
                "ARAGORA_POSTGRES_POOL_MIN": "2",
                "ARAGORA_POSTGRES_POOL_MAX": "1000",
            }
        )
        assert mod.POOL_MAX_SIZE == 500

    def test_max_size_exactly_500_accepted(self):
        """POOL_MAX_SIZE of exactly 500 is accepted (boundary)."""
        mod = _reload_pool_module(
            {
                "ARAGORA_POSTGRES_POOL_MIN": "2",
                "ARAGORA_POSTGRES_POOL_MAX": "500",
            }
        )
        assert mod.POOL_MAX_SIZE == 500

    def test_max_size_less_than_min_clamped(self):
        """POOL_MAX_SIZE < POOL_MIN_SIZE is clamped to POOL_MIN_SIZE."""
        mod = _reload_pool_module(
            {
                "ARAGORA_POSTGRES_POOL_MIN": "10",
                "ARAGORA_POSTGRES_POOL_MAX": "5",
            }
        )
        assert mod.POOL_MAX_SIZE == mod.POOL_MIN_SIZE
        assert mod.POOL_MAX_SIZE == 10

    def test_max_size_cap_logs_warning(self, caplog):
        """Capping POOL_MAX_SIZE to 500 logs a warning."""
        with caplog.at_level(logging.WARNING, logger="aragora.persistence.postgres_pool"):
            _reload_pool_module(
                {
                    "ARAGORA_POSTGRES_POOL_MIN": "2",
                    "ARAGORA_POSTGRES_POOL_MAX": "999",
                }
            )
        assert any("capping at 500" in record.message for record in caplog.records)

    def test_max_size_less_than_min_logs_warning(self, caplog):
        """Clamping POOL_MAX_SIZE to POOL_MIN_SIZE logs a warning."""
        with caplog.at_level(logging.WARNING, logger="aragora.persistence.postgres_pool"):
            _reload_pool_module(
                {
                    "ARAGORA_POSTGRES_POOL_MIN": "10",
                    "ARAGORA_POSTGRES_POOL_MAX": "3",
                }
            )
        assert any("less than POOL_MIN_SIZE" in record.message for record in caplog.records)


class TestPoolSizeCombinedEdgeCases:
    """Test edge cases involving both min and max together."""

    def test_both_at_minimum(self):
        """Both min and max at their minimum valid value of 1."""
        mod = _reload_pool_module(
            {
                "ARAGORA_POSTGRES_POOL_MIN": "1",
                "ARAGORA_POSTGRES_POOL_MAX": "1",
            }
        )
        assert mod.POOL_MIN_SIZE == 1
        assert mod.POOL_MAX_SIZE == 1

    def test_both_negative_clamps_correctly(self):
        """Both negative: min clamped to 1, max clamped to min (1)."""
        mod = _reload_pool_module(
            {
                "ARAGORA_POSTGRES_POOL_MIN": "-2",
                "ARAGORA_POSTGRES_POOL_MAX": "-5",
            }
        )
        assert mod.POOL_MIN_SIZE == 1
        assert mod.POOL_MAX_SIZE == 1

    def test_max_over_500_with_min_under_1(self):
        """Min under 1 and max over 500: both are clamped."""
        mod = _reload_pool_module(
            {
                "ARAGORA_POSTGRES_POOL_MIN": "0",
                "ARAGORA_POSTGRES_POOL_MAX": "600",
            }
        )
        assert mod.POOL_MIN_SIZE == 1
        assert mod.POOL_MAX_SIZE == 500
