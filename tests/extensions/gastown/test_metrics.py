"""
Tests for Gastown metrics adapter module.

Tests metric stubs and dynamic loading from nomic.metrics.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock


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


class TestStubMetrics:
    """Tests for metric stub functions."""

    def test_beads_completed_count_stub_returns_int(self):
        """Stub returns integer 0."""
        from aragora.extensions.gastown.metrics import _stub_beads_completed_count

        result = _stub_beads_completed_count()
        assert isinstance(result, int)
        assert result == 0

    def test_convoy_completion_rate_stub_returns_float(self):
        """Stub returns float 0.0."""
        from aragora.extensions.gastown.metrics import _stub_convoy_completion_rate

        result = _stub_convoy_completion_rate()
        assert isinstance(result, float)
        assert result == 0.0

    def test_gupp_recovery_count_stub_returns_int(self):
        """Stub returns integer 0."""
        from aragora.extensions.gastown.metrics import _stub_gupp_recovery_count

        result = _stub_gupp_recovery_count()
        assert isinstance(result, int)
        assert result == 0


class TestMetricLoading:
    """Tests for dynamic metric loading."""

    def test_load_metric_function_loads_from_nomic(self):
        """_load_metric attempts to load from aragora.nomic.metrics."""
        from aragora.extensions.gastown.metrics import _load_metric

        def stub():
            return 42

        # Test with a stub
        loaded = _load_metric("nonexistent_metric", stub)
        # Should return stub if metric doesn't exist
        assert loaded() == 42

    def test_load_metric_falls_back_to_stub(self):
        """_load_metric falls back to stub when import fails."""
        from aragora.extensions.gastown.metrics import _load_metric

        def fallback():
            return "fallback"

        with patch("importlib.import_module", side_effect=ImportError):
            loaded = _load_metric("any_metric", fallback)
            assert loaded() == "fallback"


class TestMetricFunctionality:
    """Tests for metric function behavior."""

    def test_get_beads_completed_count_callable(self):
        """get_beads_completed_count returns a numeric result."""
        from aragora.extensions.gastown.metrics import get_beads_completed_count

        result = get_beads_completed_count()
        assert isinstance(result, (int, float))
        assert result >= 0

    def test_get_convoy_completion_rate_callable(self):
        """get_convoy_completion_rate returns a rate value."""
        from aragora.extensions.gastown.metrics import get_convoy_completion_rate

        result = get_convoy_completion_rate()
        assert isinstance(result, (int, float))
        assert 0 <= result <= 1.0

    def test_get_gupp_recovery_count_callable(self):
        """get_gupp_recovery_count returns a count value."""
        from aragora.extensions.gastown.metrics import get_gupp_recovery_count

        result = get_gupp_recovery_count()
        assert isinstance(result, (int, float))
        assert result >= 0
