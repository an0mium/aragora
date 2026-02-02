"""Tests for elo_adapter backward-compatibility shim."""

import warnings

import pytest


class TestEloAdapterDeprecation:
    """Verify the deprecated elo_adapter module works and warns."""

    def test_import_issues_deprecation_warning(self):
        """Importing elo_adapter should issue a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Force a fresh import by removing from cache
            import sys

            sys.modules.pop("aragora.knowledge.mound.adapters.elo_adapter", None)

            import aragora.knowledge.mound.adapters.elo_adapter  # noqa: F401

            # Check that a DeprecationWarning was issued
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "deprecated" in str(dep_warnings[0].message).lower()

    def test_exports_elo_adapter_class(self):
        """Should re-export EloAdapter (alias for PerformanceAdapter)."""
        from aragora.knowledge.mound.adapters.elo_adapter import EloAdapter
        from aragora.knowledge.mound.adapters.performance_adapter import (
            EloAdapter as OriginalEloAdapter,
        )

        assert EloAdapter is OriginalEloAdapter

    def test_exports_performance_adapter_class(self):
        """Should re-export PerformanceAdapter."""
        from aragora.knowledge.mound.adapters.elo_adapter import PerformanceAdapter
        from aragora.knowledge.mound.adapters.performance_adapter import (
            PerformanceAdapter as OriginalPerformanceAdapter,
        )

        assert PerformanceAdapter is OriginalPerformanceAdapter

    def test_exports_all_types(self):
        """Should re-export all support types."""
        from aragora.knowledge.mound.adapters.elo_adapter import (
            RatingSearchResult,
            KMEloPattern,
            EloAdjustmentRecommendation,
            EloSyncResult,
        )

        # Verify they are accessible
        assert RatingSearchResult is not None
        assert KMEloPattern is not None
        assert EloAdjustmentRecommendation is not None
        assert EloSyncResult is not None

    def test_all_list(self):
        """__all__ should list all re-exported names."""
        from aragora.knowledge.mound.adapters import elo_adapter

        expected = {
            "EloAdapter",
            "PerformanceAdapter",
            "RatingSearchResult",
            "KMEloPattern",
            "EloAdjustmentRecommendation",
            "EloSyncResult",
        }
        assert set(elo_adapter.__all__) == expected
