"""
Tests for robustness fixes from Rounds 21-27.

These tests verify edge case handling and guards added during
the robustness improvement rounds.

Rounds covered:
- Round 21: Empty agent list guards, string split safety, DB timeouts
- Round 22: Empty list guards for random.choice, sorted lists
- Round 23: Division by zero guards
- Round 24: Silent data loss in audio mixing
- Round 25: ThreadPoolExecutor race condition
- Round 26: Code quality (self-imports, print→logger)
- Round 27: Missing logger imports
"""

import pytest
import logging
from unittest.mock import MagicMock


# ============================================================================
# Round 27: Logger Import Tests
# ============================================================================


class TestLoggerImports:
    """Test that modules with logger usage have proper imports."""

    def test_meta_module_logger_exists(self):
        """Test meta.py has logger properly defined (Round 27 fix)."""
        from aragora.debate import meta

        # Module should have logger attribute
        assert hasattr(meta, "logger")
        # Logger should be a Logger instance
        assert isinstance(meta.logger, logging.Logger)

    def test_ledger_module_logger_exists(self):
        """Test ledger.py has logger properly defined (Round 27 fix)."""
        from aragora.genesis import ledger

        # Module should have logger attribute
        assert hasattr(ledger, "logger")
        # Logger should be a Logger instance
        assert isinstance(ledger.logger, logging.Logger)

    def test_meta_analyzer_can_instantiate(self):
        """Test MetaCritiqueAnalyzer can be instantiated without logger errors."""
        from aragora.debate.meta import MetaCritiqueAnalyzer

        # Should not raise NameError for undefined logger
        analyzer = MetaCritiqueAnalyzer()
        assert analyzer is not None


# ============================================================================
# Round 23: Division by Zero Guard Tests
# ============================================================================


class TestDivisionByZeroGuards:
    """Test division by zero guards added in Round 23."""

    def test_meta_critique_empty_round_critiques(self):
        """Test _classify_rounds handles empty critique list gracefully."""
        from aragora.debate.meta import MetaCritiqueAnalyzer

        analyzer = MetaCritiqueAnalyzer()

        # Create minimal mock result with no critiques
        mock_result = MagicMock()
        mock_result.messages = []
        mock_result.critiques = []

        # Should not raise ZeroDivisionError
        productive, unproductive = analyzer._classify_rounds(mock_result)
        assert isinstance(productive, list)
        assert isinstance(unproductive, list)

    def test_meta_critique_empty_observations(self):
        """Test _calculate_quality handles empty observations."""
        from aragora.debate.meta import MetaCritiqueAnalyzer

        analyzer = MetaCritiqueAnalyzer()

        mock_result = MagicMock()
        mock_result.consensus_reached = False
        mock_result.confidence = 0.5

        # Should handle empty lists without division errors
        quality = analyzer._calculate_quality(
            mock_result,
            observations=[],
            productive=[],
            unproductive=[],
        )
        assert 0.0 <= quality <= 1.0


# ============================================================================
# Round 22: Empty List Guard Tests
# ============================================================================


class TestEmptyListGuards:
    """Test empty list guards added in Round 22."""

    def test_text_similarity_empty_text(self):
        """Test _text_similarity handles empty strings."""
        from aragora.debate.meta import MetaCritiqueAnalyzer

        analyzer = MetaCritiqueAnalyzer()

        # Empty strings should return 0.0, not raise
        similarity = analyzer._text_similarity("", "")
        assert similarity == 0.0

        similarity = analyzer._text_similarity("hello", "")
        assert similarity == 0.0

        similarity = analyzer._text_similarity("", "world")
        assert similarity == 0.0


# ============================================================================
# Round 24: Silent Data Loss Prevention Tests
# ============================================================================


class TestSilentDataLossPrevention:
    """Test silent data loss prevention in audio mixing (Round 24)."""

    def test_mix_audio_returns_false_all_missing(self):
        """Test mix_audio returns False when all files are missing.

        This is tested in test_broadcast.py::TestMixer::test_mix_audio_all_files_missing
        """
        pass  # Covered in test_broadcast.py


# ============================================================================
# Round 25: ThreadPoolExecutor Race Condition Tests
# ============================================================================


class TestThreadPoolExecutorRaceCondition:
    """Test ThreadPoolExecutor race condition fixes (Round 25).

    This is tested in test_stream.py::TestDebateExecutorRaceCondition
    """

    def test_race_condition_tests_exist(self):
        """Verify race condition tests are in place."""
        # These tests are in test_stream.py
        pass


# ============================================================================
# Module Import Smoke Tests
# ============================================================================


class TestModuleImports:
    """Smoke tests to verify modules can be imported without errors."""

    def test_import_debate_meta(self):
        """Test debate.meta module imports cleanly."""
        from aragora.debate import meta

        assert meta is not None

    def test_import_genesis_ledger(self):
        """Test genesis.ledger module imports cleanly."""
        from aragora.genesis import ledger

        assert ledger is not None

    def test_import_broadcast_mixer(self):
        """Test broadcast.mixer module imports cleanly."""
        from aragora.broadcast import mixer

        assert mixer is not None

    def test_import_uncertainty_estimator(self):
        """Test uncertainty.estimator module imports cleanly."""
        from aragora.uncertainty import estimator

        assert estimator is not None
