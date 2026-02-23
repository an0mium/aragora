"""Tests for cross-cycle calibration drift detection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.nomic.calibration_monitor import CalibrationDriftDetector, DriftWarning


class TestCalibrationDriftDetector:
    """Tests for CalibrationDriftDetector."""

    def test_no_drift_with_normal_scores(self) -> None:
        """Normal varying scores should produce no warnings."""
        detector = CalibrationDriftDetector(window_size=5)
        detector.record_cycle("c1", {"claude": 0.5, "gemini": 0.4})
        detector.record_cycle("c2", {"claude": 0.8, "gemini": 0.7})
        detector.record_cycle("c3", {"claude": 0.6, "gemini": 0.5})

        warnings = detector.detect_drift()
        assert len(warnings) == 0

    def test_stagnation_detected(self) -> None:
        """Scores with very low variance trigger stagnation warning."""
        detector = CalibrationDriftDetector(window_size=10, stagnation_threshold=0.01)
        # Agent with nearly identical scores
        for i in range(5):
            detector.record_cycle(f"c{i}", {"claude": 0.5 + (i % 2) * 0.001})

        warnings = detector.detect_drift()
        stagnation = [w for w in warnings if w.type == "stagnation"]
        assert len(stagnation) >= 1
        assert stagnation[0].agent_name == "claude"
        assert "stagnant" in stagnation[0].message

    def test_regression_detected(self) -> None:
        """Monotonically decreasing scores trigger regression warning."""
        detector = CalibrationDriftDetector(window_size=10, regression_threshold=0.05)
        detector.record_cycle("c1", {"claude": 0.8})
        detector.record_cycle("c2", {"claude": 0.75})
        detector.record_cycle("c3", {"claude": 0.7})
        detector.record_cycle("c4", {"claude": 0.65})

        warnings = detector.detect_drift()
        regression = [w for w in warnings if w.type == "regression"]
        assert len(regression) >= 1
        assert regression[0].agent_name == "claude"
        assert "regressing" in regression[0].message

    def test_inflation_detected(self) -> None:
        """Consistently high scores (> 0.95) trigger inflation warning."""
        detector = CalibrationDriftDetector(window_size=10)
        for i in range(5):
            detector.record_cycle(f"c{i}", {"claude": 0.97})

        warnings = detector.detect_drift()
        inflation = [w for w in warnings if w.type == "inflation"]
        assert len(inflation) >= 1
        assert inflation[0].agent_name == "claude"
        assert "inflation" in inflation[0].message

    def test_window_size_respected(self) -> None:
        """Only cycles within the window are considered."""
        detector = CalibrationDriftDetector(window_size=3)

        # Record 5 cycles, but window is 3
        detector.record_cycle("c1", {"claude": 0.5})
        detector.record_cycle("c2", {"claude": 0.5})
        detector.record_cycle("c3", {"claude": 0.5})
        detector.record_cycle("c4", {"claude": 0.7})
        detector.record_cycle("c5", {"claude": 0.8})

        # c1 and c2 should be trimmed, leaving [0.5, 0.7, 0.8] — not stagnant
        assert len(detector._cycles) == 3
        warnings = detector.detect_drift()
        stagnation = [w for w in warnings if w.type == "stagnation"]
        assert len(stagnation) == 0

    def test_insufficient_data_no_warnings(self) -> None:
        """Fewer than 2 cycles should produce no warnings."""
        detector = CalibrationDriftDetector()
        detector.record_cycle("c1", {"claude": 0.5})

        warnings = detector.detect_drift()
        assert len(warnings) == 0

    def test_empty_cycles_no_warnings(self) -> None:
        """No recorded cycles should produce no warnings."""
        detector = CalibrationDriftDetector()
        warnings = detector.detect_drift()
        assert len(warnings) == 0

    def test_multiple_agents_independent(self) -> None:
        """Each agent's drift is detected independently."""
        detector = CalibrationDriftDetector(window_size=10, stagnation_threshold=0.01)
        for i in range(5):
            detector.record_cycle(
                f"c{i}",
                {
                    "claude": 0.5,  # Stagnant
                    "gemini": 0.5 + i * 0.1,  # Improving
                },
            )

        warnings = detector.detect_drift()
        claude_warnings = [w for w in warnings if w.agent_name == "claude"]
        gemini_warnings = [w for w in warnings if w.agent_name == "gemini"]

        assert len(claude_warnings) >= 1  # Stagnation detected
        assert len(gemini_warnings) == 0  # No issues

    def test_regression_needs_three_cycles(self) -> None:
        """Regression needs at least 3 consecutive decreasing points."""
        detector = CalibrationDriftDetector(regression_threshold=0.05)
        detector.record_cycle("c1", {"claude": 0.8})
        detector.record_cycle("c2", {"claude": 0.7})
        # Only 2 cycles with decrease — insufficient for regression

        warnings = detector.detect_drift()
        regression = [w for w in warnings if w.type == "regression"]
        assert len(regression) == 0

    def test_high_severity_stagnation(self) -> None:
        """Very low variance triggers high severity stagnation."""
        detector = CalibrationDriftDetector(stagnation_threshold=0.01)
        for i in range(5):
            detector.record_cycle(f"c{i}", {"claude": 0.5})

        warnings = detector.detect_drift()
        stagnation = [w for w in warnings if w.type == "stagnation"]
        assert len(stagnation) >= 1
        assert stagnation[0].severity == "high"

    def test_high_severity_inflation(self) -> None:
        """All scores > 0.95 triggers high severity inflation."""
        detector = CalibrationDriftDetector()
        for i in range(5):
            detector.record_cycle(f"c{i}", {"claude": 0.99})

        warnings = detector.detect_drift()
        inflation = [w for w in warnings if w.type == "inflation"]
        assert len(inflation) >= 1
        assert inflation[0].severity == "high"

    def test_km_persistence_attempted(self) -> None:
        """KM persistence is attempted when drift is detected."""
        detector = CalibrationDriftDetector(stagnation_threshold=0.01)
        for i in range(5):
            detector.record_cycle(f"c{i}", {"claude": 0.5})

        with patch.object(detector, "_persist_to_km") as mock_persist:
            warnings = detector.detect_drift()
            assert len(warnings) > 0
            mock_persist.assert_called_once_with(warnings)

    def test_km_persistence_import_error(self) -> None:
        """KM persistence handles ImportError gracefully."""
        detector = CalibrationDriftDetector()
        warnings = [
            DriftWarning(
                type="stagnation",
                agent_name="claude",
                score_history=[0.5, 0.5, 0.5],
                severity="medium",
                message="test",
            )
        ]

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound.core": None,
                "aragora.knowledge.mound.adapters.receipt_adapter": None,
            },
        ):
            # Should not raise
            detector._persist_to_km(warnings)

    def test_emit_warnings_import_error(self) -> None:
        """Streaming emit handles ImportError gracefully."""
        detector = CalibrationDriftDetector()
        warnings = [
            DriftWarning(
                type="regression",
                agent_name="gemini",
                score_history=[0.8, 0.7, 0.6],
                severity="high",
                message="test",
            )
        ]

        with patch.dict("sys.modules", {"aragora.spectate.stream": None}):
            detector._emit_warnings(warnings)

    def test_drift_warning_dataclass(self) -> None:
        """DriftWarning dataclass fields are correct."""
        warning = DriftWarning(
            type="stagnation",
            agent_name="claude",
            score_history=[0.5, 0.5, 0.5],
            severity="medium",
            message="Test message",
        )
        assert warning.type == "stagnation"
        assert warning.agent_name == "claude"
        assert warning.score_history == [0.5, 0.5, 0.5]
        assert warning.severity == "medium"
        assert warning.message == "Test message"
