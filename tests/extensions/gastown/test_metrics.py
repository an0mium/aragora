"""
Tests for Gastown metrics adapter module.

Tests the public metric functions for observability.
"""

from __future__ import annotations

import pytest


class TestMetricExports:
    """Tests for metrics module exports."""

    def test_get_beads_completed_count_exported(self):
        """get_beads_completed_count function is exported."""
        from aragora.extensions.gastown.metrics import get_beads_completed_count

        assert callable(get_beads_completed_count)

    def test_get_convoy_completion_rate_exported(self):
        """get_convoy_completion_rate function is exported."""
        from aragora.extensions.gastown.metrics import get_convoy_completion_rate

        assert callable(get_convoy_completion_rate)

    def test_get_gupp_recovery_count_exported(self):
        """get_gupp_recovery_count function is exported."""
        from aragora.extensions.gastown.metrics import get_gupp_recovery_count

        assert callable(get_gupp_recovery_count)

    def test_all_exports_in_module_all(self):
        """All documented exports are in __all__."""
        from aragora.extensions.gastown import metrics

        expected = {
            "get_beads_completed_count",
            "get_convoy_completion_rate",
            "get_gupp_recovery_count",
        }
        assert set(metrics.__all__) == expected


class TestMetricFunctionality:
    """Tests for metric function behavior."""

    def test_get_beads_completed_count_callable(self):
        """get_beads_completed_count returns a numeric result."""
        from aragora.extensions.gastown.metrics import get_beads_completed_count

        result = get_beads_completed_count()
        assert isinstance(result, (int, float))
        assert result >= 0

    def test_get_convoy_completion_rate_callable(self):
        """get_convoy_completion_rate returns a percentage value."""
        from aragora.extensions.gastown.metrics import get_convoy_completion_rate

        result = get_convoy_completion_rate()
        assert isinstance(result, (int, float))
        assert 0 <= result <= 100.0

    def test_get_gupp_recovery_count_callable(self):
        """get_gupp_recovery_count returns a count value."""
        from aragora.extensions.gastown.metrics import get_gupp_recovery_count

        result = get_gupp_recovery_count()
        assert isinstance(result, (int, float))
        assert result >= 0
