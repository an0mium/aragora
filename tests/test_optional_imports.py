"""
Tests for optional import utilities.

Tests the try_import, try_import_class, and LazyImport utilities
that provide consistent handling of optional dependencies.
"""

import logging
import sys
import pytest
from unittest.mock import patch, MagicMock

from aragora.utils.optional_imports import try_import, try_import_class, LazyImport


# ============================================================================
# Test try_import Function
# ============================================================================


class TestTryImport:
    """Tests for the try_import function."""

    def test_import_single_class_success(self):
        """Import a single class from an available module."""
        imported, available = try_import("json", "JSONEncoder")

        assert available is True
        assert "JSONEncoder" in imported
        assert imported["JSONEncoder"] is not None

    def test_import_multiple_classes_success(self):
        """Import multiple classes from an available module."""
        imported, available = try_import("json", "JSONEncoder", "JSONDecoder")

        assert available is True
        assert len(imported) == 2
        assert imported["JSONEncoder"] is not None
        assert imported["JSONDecoder"] is not None

    def test_import_unavailable_module(self):
        """Import from a module that doesn't exist."""
        imported, available = try_import("nonexistent.module.that.doesnt.exist", "SomeClass")

        assert available is False
        assert imported["SomeClass"] is None

    def test_import_unavailable_attribute(self):
        """Import an attribute that doesn't exist in the module."""
        imported, available = try_import("json", "NonexistentClass")

        assert available is False
        assert imported["NonexistentClass"] is None

    def test_import_partial_failure(self):
        """When one of multiple imports fails, all should fail."""
        imported, available = try_import(
            "json", "JSONEncoder", "NonexistentClass"  # exists  # doesn't exist
        )

        assert available is False
        # First one was found, second wasn't
        assert imported["JSONEncoder"] is not None
        assert imported["NonexistentClass"] is None

    def test_import_with_logging_enabled(self, caplog):
        """Verify logging works when enabled."""
        with caplog.at_level(logging.DEBUG):
            imported, available = try_import(
                "nonexistent.module", "SomeClass", log_on_failure=True, log_level="debug"
            )

        assert available is False
        assert "Optional module not available" in caplog.text

    def test_import_without_logging(self, caplog):
        """Verify no logging when disabled."""
        with caplog.at_level(logging.DEBUG):
            imported, available = try_import(
                "nonexistent.module", "SomeClass", log_on_failure=False
            )

        assert available is False
        assert caplog.text == ""

    def test_import_real_aragora_module(self):
        """Test with a real aragora module that should exist."""
        imported, available = try_import("aragora.core", "Environment")

        # This should succeed if aragora.core exists
        if available:
            assert imported["Environment"] is not None
        # If not available, that's also valid - module may not be installed

    def test_import_returns_correct_types(self):
        """Verify return types are correct."""
        imported, available = try_import("json", "JSONEncoder")

        assert isinstance(imported, dict)
        assert isinstance(available, bool)

    def test_import_empty_names(self):
        """Import with no names should succeed with empty dict."""
        imported, available = try_import("json")

        assert available is True
        assert imported == {}


# ============================================================================
# Test try_import_class Function
# ============================================================================


class TestTryImportClass:
    """Tests for the try_import_class convenience function."""

    def test_import_class_success(self):
        """Import a single class successfully."""
        cls, available = try_import_class("json", "JSONEncoder")

        assert available is True
        assert cls is not None

    def test_import_class_failure(self):
        """Import a class that doesn't exist."""
        cls, available = try_import_class("json", "NonexistentClass")

        assert available is False
        assert cls is None

    def test_import_class_from_missing_module(self):
        """Import from a module that doesn't exist."""
        cls, available = try_import_class("nonexistent.module", "SomeClass")

        assert available is False
        assert cls is None

    def test_import_class_with_logging(self, caplog):
        """Verify logging works for class import."""
        with caplog.at_level(logging.DEBUG):
            cls, available = try_import_class(
                "nonexistent.module", "SomeClass", log_on_failure=True
            )

        assert available is False
        assert "Optional module not available" in caplog.text


# ============================================================================
# Test LazyImport Class
# ============================================================================


class TestLazyImport:
    """Tests for the LazyImport class."""

    def test_lazy_import_deferred(self):
        """Import should be deferred until first access."""
        lazy = LazyImport("json", "JSONEncoder")

        # _imported should be None before access
        assert lazy._imported is None

        # Access triggers import
        cls = lazy.get("JSONEncoder")

        # Now it should be loaded
        assert lazy._imported is not None
        assert cls is not None

    def test_lazy_import_caching(self):
        """Multiple accesses should reuse cached import."""
        lazy = LazyImport("json", "JSONEncoder")

        # First access
        cls1 = lazy.get("JSONEncoder")
        imported_ref = lazy._imported

        # Second access should use cache
        cls2 = lazy.get("JSONEncoder")

        assert cls1 is cls2
        assert lazy._imported is imported_ref

    def test_lazy_import_available_property(self):
        """Test the available property."""
        lazy_success = LazyImport("json", "JSONEncoder")
        lazy_fail = LazyImport("nonexistent.module", "SomeClass")

        assert lazy_success.available is True
        assert lazy_fail.available is False

    def test_lazy_import_all_method(self):
        """Test the all() method returns both dict and flag."""
        lazy = LazyImport("json", "JSONEncoder", "JSONDecoder")

        imported, available = lazy.all()

        assert available is True
        assert "JSONEncoder" in imported
        assert "JSONDecoder" in imported

    def test_lazy_import_missing_module(self):
        """LazyImport with missing module."""
        lazy = LazyImport("nonexistent.module", "SomeClass")

        cls = lazy.get("SomeClass")

        assert cls is None
        assert lazy.available is False

    def test_lazy_import_missing_attribute(self):
        """LazyImport with missing attribute."""
        lazy = LazyImport("json", "NonexistentClass")

        cls = lazy.get("NonexistentClass")

        assert cls is None
        assert lazy.available is False

    def test_lazy_import_get_unknown_name(self):
        """Getting a name not in the import list."""
        lazy = LazyImport("json", "JSONEncoder")

        # Request something not in the list
        cls = lazy.get("SomethingElse")

        assert cls is None

    def test_lazy_import_with_logging(self, caplog):
        """LazyImport with logging enabled."""
        with caplog.at_level(logging.DEBUG):
            lazy = LazyImport("nonexistent.module", "SomeClass", log_on_failure=True)
            lazy.get("SomeClass")

        assert "Optional module not available" in caplog.text


# ============================================================================
# Integration Tests
# ============================================================================


class TestOptionalImportsIntegration:
    """Integration tests simulating real usage patterns."""

    def test_typical_unified_server_pattern(self):
        """Test the pattern used in unified_server.py."""
        # Old pattern:
        # try:
        #     from aragora.ranking.elo import EloSystem
        #     RANKING_AVAILABLE = True
        # except ImportError:
        #     RANKING_AVAILABLE = False
        #     EloSystem = None

        # New pattern:
        imported, RANKING_AVAILABLE = try_import("aragora.ranking.elo", "EloSystem")
        EloSystem = imported["EloSystem"]

        # Works regardless of whether module exists
        assert isinstance(RANKING_AVAILABLE, bool)
        # EloSystem is either a class or None
        assert EloSystem is None or callable(EloSystem)

    def test_typical_multi_import_pattern(self):
        """Test pattern for importing multiple classes."""
        # Old pattern:
        # try:
        #     from aragora.memory.consensus import ConsensusMemory, DissentRetriever
        #     CONSENSUS_AVAILABLE = True
        # except ImportError:
        #     CONSENSUS_AVAILABLE = False
        #     ConsensusMemory = None
        #     DissentRetriever = None

        # New pattern:
        imported, CONSENSUS_AVAILABLE = try_import(
            "aragora.memory.consensus", "ConsensusMemory", "DissentRetriever"
        )
        ConsensusMemory = imported["ConsensusMemory"]
        DissentRetriever = imported["DissentRetriever"]

        assert isinstance(CONSENSUS_AVAILABLE, bool)

    def test_typical_lazy_load_pattern(self):
        """Test pattern for lazy loading to avoid circular imports."""
        # Old pattern in orchestrator.py:
        # BeliefNetwork = None
        # def _get_belief_analyzer():
        #     global BeliefNetwork
        #     if BeliefNetwork is None:
        #         try:
        #             from aragora.reasoning.belief import BeliefNetwork as _BN
        #             BeliefNetwork = _BN
        #         except ImportError:
        #             pass
        #     return BeliefNetwork

        # New pattern:
        _belief_imports = LazyImport(
            "aragora.reasoning.belief", "BeliefNetwork", "BeliefPropagationAnalyzer"
        )

        def get_belief_network():
            return _belief_imports.get("BeliefNetwork")

        # First call triggers import
        bn1 = get_belief_network()
        # Second call uses cache
        bn2 = get_belief_network()

        assert bn1 is bn2

    def test_handler_pattern_with_availability_check(self):
        """Test pattern used in handlers."""
        imported, FEATURE_AVAILABLE = try_import("aragora.some.feature", "FeatureClass")
        FeatureClass = imported["FeatureClass"]

        # Handler method pattern
        def handle_request():
            if not FEATURE_AVAILABLE:
                return {"error": "Feature not available", "status": 503}

            # Would use FeatureClass here
            return {"status": 200}

        result = handle_request()
        assert result["status"] in (200, 503)


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_import_builtin_module(self):
        """Import from a builtin module."""
        imported, available = try_import("os", "path")

        assert available is True
        assert imported["path"] is not None

    def test_import_with_different_log_levels(self, caplog):
        """Test different log levels."""
        for level in ["debug", "info", "warning"]:
            caplog.clear()
            with caplog.at_level(logging.DEBUG):
                try_import("nonexistent.module", "SomeClass", log_on_failure=True, log_level=level)
            assert "Optional module not available" in caplog.text

    def test_lazy_import_thread_safety(self):
        """Basic thread safety test for LazyImport."""
        import threading

        lazy = LazyImport("json", "JSONEncoder")
        results = []

        def access_lazy():
            cls = lazy.get("JSONEncoder")
            results.append(cls)

        threads = [threading.Thread(target=access_lazy) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be the same class
        assert len(results) == 10
        assert all(r is results[0] for r in results)

    def test_import_with_special_characters_in_name(self):
        """Module paths with dots work correctly."""
        # This should work - os.path is valid
        imported, available = try_import("os.path", "join")

        assert available is True
        assert imported["join"] is not None
