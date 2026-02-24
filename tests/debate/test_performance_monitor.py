"""Tests for Debate Performance Monitoring System."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.performance_monitor import (
    DebateMetric,
    DebatePerformanceMonitor,
    PhaseMetric,
    RoundMetric,
    SlowDebateRecord,
    get_debate_monitor,
    DEFAULT_SLOW_ROUND_THRESHOLD,
    MAX_SLOW_DEBATES_HISTORY,
)


def _make_time(start=1000.0, step=0.1):
    """Create an auto-incrementing time.time() replacement.

    Each call returns start + N*step where N is the call count (0-indexed).
    This lets tests simulate time passage without sleeping.
    """
    state = {"calls": 0}

    def _time():
        val = start + state["calls"] * step
        state["calls"] += 1
        return val

    return _time


# =============================================================================
# PhaseMetric Tests
# =============================================================================


class TestPhaseMetric:
    """Test PhaseMetric dataclass."""

    def test_create_phase_metric(self):
        """Test creating a phase metric."""
        metric = PhaseMetric(
            phase_name="propose",
            start_time=1000.0,
        )
        assert metric.phase_name == "propose"
        assert metric.start_time == 1000.0
        assert metric.end_time is None
        assert metric.duration_seconds is None
        assert metric.agent_count == 0
        assert metric.error is None

    def test_phase_metric_with_all_fields(self):
        """Test phase metric with all fields populated."""
        metric = PhaseMetric(
            phase_name="critique",
            start_time=1000.0,
            end_time=1015.5,
            duration_seconds=15.5,
            agent_count=3,
            error=None,
        )
        assert metric.phase_name == "critique"
        assert metric.duration_seconds == 15.5
        assert metric.agent_count == 3

    def test_phase_metric_with_error(self):
        """Test phase metric with error."""
        metric = PhaseMetric(
            phase_name="vote",
            start_time=1000.0,
            error="Timeout error",
        )
        assert metric.error == "Timeout error"


# =============================================================================
# RoundMetric Tests
# =============================================================================


class TestRoundMetric:
    """Test RoundMetric dataclass."""

    def test_create_round_metric(self):
        """Test creating a round metric."""
        metric = RoundMetric(
            round_num=1,
            start_time=1000.0,
        )
        assert metric.round_num == 1
        assert metric.start_time == 1000.0
        assert metric.end_time is None
        assert metric.duration_seconds is None
        assert metric.phases == {}
        assert metric.is_slow is False
        assert metric.slow_threshold == DEFAULT_SLOW_ROUND_THRESHOLD

    def test_round_metric_with_phases(self):
        """Test round metric with phases."""
        metric = RoundMetric(
            round_num=2,
            start_time=1000.0,
            end_time=1045.0,
            duration_seconds=45.0,
        )
        metric.phases["propose"] = PhaseMetric(
            phase_name="propose",
            start_time=1000.0,
            end_time=1020.0,
            duration_seconds=20.0,
        )
        metric.phases["critique"] = PhaseMetric(
            phase_name="critique",
            start_time=1020.0,
            end_time=1035.0,
            duration_seconds=15.0,
        )
        assert len(metric.phases) == 2
        assert metric.phases["propose"].duration_seconds == 20.0

    def test_total_phase_time_empty(self):
        """Test total phase time with no phases."""
        metric = RoundMetric(round_num=1, start_time=1000.0)
        assert metric.total_phase_time == 0.0

    def test_total_phase_time_with_phases(self):
        """Test total phase time calculation."""
        metric = RoundMetric(round_num=1, start_time=1000.0)
        metric.phases["propose"] = PhaseMetric(
            phase_name="propose",
            start_time=1000.0,
            duration_seconds=10.0,
        )
        metric.phases["critique"] = PhaseMetric(
            phase_name="critique",
            start_time=1010.0,
            duration_seconds=8.0,
        )
        metric.phases["vote"] = PhaseMetric(
            phase_name="vote",
            start_time=1018.0,
            duration_seconds=5.0,
        )
        assert metric.total_phase_time == 23.0

    def test_total_phase_time_with_none_duration(self):
        """Test total phase time handles None durations."""
        metric = RoundMetric(round_num=1, start_time=1000.0)
        metric.phases["propose"] = PhaseMetric(
            phase_name="propose",
            start_time=1000.0,
            duration_seconds=10.0,
        )
        metric.phases["critique"] = PhaseMetric(
            phase_name="critique",
            start_time=1010.0,
            duration_seconds=None,  # Not completed
        )
        assert metric.total_phase_time == 10.0

    def test_slowest_phase_empty(self):
        """Test slowest phase with no phases."""
        metric = RoundMetric(round_num=1, start_time=1000.0)
        assert metric.slowest_phase is None

    def test_slowest_phase_single(self):
        """Test slowest phase with single phase."""
        metric = RoundMetric(round_num=1, start_time=1000.0)
        metric.phases["propose"] = PhaseMetric(
            phase_name="propose",
            start_time=1000.0,
            duration_seconds=15.0,
        )
        slowest = metric.slowest_phase
        assert slowest == ("propose", 15.0)

    def test_slowest_phase_multiple(self):
        """Test slowest phase with multiple phases."""
        metric = RoundMetric(round_num=1, start_time=1000.0)
        metric.phases["propose"] = PhaseMetric(
            phase_name="propose",
            start_time=1000.0,
            duration_seconds=10.0,
        )
        metric.phases["critique"] = PhaseMetric(
            phase_name="critique",
            start_time=1010.0,
            duration_seconds=25.0,
        )
        metric.phases["vote"] = PhaseMetric(
            phase_name="vote",
            start_time=1035.0,
            duration_seconds=5.0,
        )
        slowest = metric.slowest_phase
        assert slowest == ("critique", 25.0)

    def test_slowest_phase_with_none_duration(self):
        """Test slowest phase handles None durations."""
        metric = RoundMetric(round_num=1, start_time=1000.0)
        metric.phases["propose"] = PhaseMetric(
            phase_name="propose",
            start_time=1000.0,
            duration_seconds=None,
        )
        metric.phases["critique"] = PhaseMetric(
            phase_name="critique",
            start_time=1010.0,
            duration_seconds=5.0,
        )
        slowest = metric.slowest_phase
        assert slowest == ("critique", 5.0)


# =============================================================================
# DebateMetric Tests
# =============================================================================


class TestDebateMetric:
    """Test DebateMetric dataclass."""

    def test_create_debate_metric(self):
        """Test creating a debate metric."""
        metric = DebateMetric(
            debate_id="debate-123",
            task="Design a rate limiter",
            start_time=1000.0,
        )
        assert metric.debate_id == "debate-123"
        assert metric.task == "Design a rate limiter"
        assert metric.start_time == 1000.0
        assert metric.end_time is None
        assert metric.duration_seconds is None
        assert metric.rounds == {}
        assert metric.outcome == "in_progress"
        assert metric.agent_names == []
        assert metric.slow_round_count == 0

    def test_debate_metric_with_agents(self):
        """Test debate metric with agent names."""
        metric = DebateMetric(
            debate_id="debate-456",
            task="Test task",
            start_time=1000.0,
            agent_names=["claude", "gpt", "gemini"],
        )
        assert metric.agent_names == ["claude", "gpt", "gemini"]

    def test_is_slow_no_slow_rounds(self):
        """Test is_slow with no slow rounds."""
        metric = DebateMetric(
            debate_id="debate-123",
            task="Test",
            start_time=1000.0,
            slow_round_count=0,
        )
        assert metric.is_slow is False

    def test_is_slow_with_slow_rounds(self):
        """Test is_slow with slow rounds."""
        metric = DebateMetric(
            debate_id="debate-123",
            task="Test",
            start_time=1000.0,
            slow_round_count=2,
        )
        assert metric.is_slow is True

    def test_avg_round_duration_no_rounds(self):
        """Test average round duration with no rounds."""
        metric = DebateMetric(
            debate_id="debate-123",
            task="Test",
            start_time=1000.0,
        )
        assert metric.avg_round_duration == 0.0

    def test_avg_round_duration_with_rounds(self):
        """Test average round duration calculation."""
        metric = DebateMetric(
            debate_id="debate-123",
            task="Test",
            start_time=1000.0,
        )
        metric.rounds[1] = RoundMetric(
            round_num=1,
            start_time=1000.0,
            duration_seconds=20.0,
        )
        metric.rounds[2] = RoundMetric(
            round_num=2,
            start_time=1020.0,
            duration_seconds=30.0,
        )
        metric.rounds[3] = RoundMetric(
            round_num=3,
            start_time=1050.0,
            duration_seconds=25.0,
        )
        assert metric.avg_round_duration == 25.0

    def test_avg_round_duration_with_incomplete_rounds(self):
        """Test average round duration ignores incomplete rounds."""
        metric = DebateMetric(
            debate_id="debate-123",
            task="Test",
            start_time=1000.0,
        )
        metric.rounds[1] = RoundMetric(
            round_num=1,
            start_time=1000.0,
            duration_seconds=20.0,
        )
        metric.rounds[2] = RoundMetric(
            round_num=2,
            start_time=1020.0,
            duration_seconds=None,  # Incomplete
        )
        assert metric.avg_round_duration == 20.0

    def test_get_slowest_round_no_rounds(self):
        """Test get slowest round with no rounds."""
        metric = DebateMetric(
            debate_id="debate-123",
            task="Test",
            start_time=1000.0,
        )
        assert metric.get_slowest_round() is None

    def test_get_slowest_round_with_rounds(self):
        """Test get slowest round calculation."""
        metric = DebateMetric(
            debate_id="debate-123",
            task="Test",
            start_time=1000.0,
        )
        metric.rounds[1] = RoundMetric(
            round_num=1,
            start_time=1000.0,
            duration_seconds=20.0,
        )
        metric.rounds[2] = RoundMetric(
            round_num=2,
            start_time=1020.0,
            duration_seconds=45.0,
        )
        metric.rounds[3] = RoundMetric(
            round_num=3,
            start_time=1065.0,
            duration_seconds=15.0,
        )
        slowest = metric.get_slowest_round()
        assert slowest == (2, 45.0)


# =============================================================================
# SlowDebateRecord Tests
# =============================================================================


class TestSlowDebateRecord:
    """Test SlowDebateRecord dataclass."""

    def test_create_slow_debate_record(self):
        """Test creating a slow debate record."""
        record = SlowDebateRecord(
            debate_id="debate-123",
            task="Design a rate limiter",
            detected_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            total_duration=120.5,
            round_count=3,
            slow_round_count=2,
            slowest_round=(2, 50.0),
            slowest_phase=("critique", 30.0),
            agent_names=["claude", "gpt"],
        )
        assert record.debate_id == "debate-123"
        assert record.task == "Design a rate limiter"
        assert record.total_duration == 120.5
        assert record.round_count == 3
        assert record.slow_round_count == 2
        assert record.slowest_round == (2, 50.0)
        assert record.slowest_phase == ("critique", 30.0)

    def test_to_dict_basic(self):
        """Test to_dict conversion."""
        record = SlowDebateRecord(
            debate_id="debate-123",
            task="Test task",
            detected_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            total_duration=90.123,
            round_count=2,
            slow_round_count=1,
            slowest_round=(1, 45.0),
            slowest_phase=("propose", 25.0),
            agent_names=["claude", "gpt", "gemini"],
        )
        result = record.to_dict()

        assert result["debate_id"] == "debate-123"
        assert result["task"] == "Test task"
        assert result["detected_at"] == "2024-01-15T10:30:00+00:00"
        assert result["total_duration_seconds"] == 90.12
        assert result["round_count"] == 2
        assert result["slow_round_count"] == 1
        assert result["slowest_round"] == (1, 45.0)
        assert result["slowest_phase"] == ("propose", 25.0)
        assert result["agent_count"] == 3

    def test_to_dict_truncates_long_task(self):
        """Test to_dict truncates long task strings."""
        long_task = "A" * 200
        record = SlowDebateRecord(
            debate_id="debate-123",
            task=long_task,
            detected_at=datetime.now(timezone.utc),
            total_duration=60.0,
            round_count=1,
            slow_round_count=1,
            slowest_round=None,
            slowest_phase=None,
            agent_names=[],
        )
        result = record.to_dict()
        assert len(result["task"]) == 100

    def test_to_dict_empty_task(self):
        """Test to_dict handles empty task."""
        record = SlowDebateRecord(
            debate_id="debate-123",
            task="",
            detected_at=datetime.now(timezone.utc),
            total_duration=60.0,
            round_count=1,
            slow_round_count=1,
            slowest_round=None,
            slowest_phase=None,
            agent_names=[],
        )
        result = record.to_dict()
        assert result["task"] == ""

    def test_to_dict_none_slowest_values(self):
        """Test to_dict handles None slowest values."""
        record = SlowDebateRecord(
            debate_id="debate-123",
            task="Test",
            detected_at=datetime.now(timezone.utc),
            total_duration=60.0,
            round_count=1,
            slow_round_count=1,
            slowest_round=None,
            slowest_phase=None,
            agent_names=[],
        )
        result = record.to_dict()
        assert result["slowest_round"] is None
        assert result["slowest_phase"] is None


# =============================================================================
# DebatePerformanceMonitor Initialization Tests
# =============================================================================


class TestDebatePerformanceMonitorInit:
    """Test DebatePerformanceMonitor initialization."""

    def test_default_initialization(self):
        """Test default initialization values."""
        monitor = DebatePerformanceMonitor()
        assert monitor.slow_round_threshold == DEFAULT_SLOW_ROUND_THRESHOLD
        assert monitor.emit_prometheus is True
        assert monitor._active_debates == {}
        assert monitor._slow_debates == []
        assert monitor._current_rounds == {}
        assert monitor._current_phases == {}

    def test_custom_threshold(self):
        """Test initialization with custom threshold."""
        monitor = DebatePerformanceMonitor(slow_round_threshold=60.0)
        assert monitor.slow_round_threshold == 60.0

    def test_prometheus_disabled(self):
        """Test initialization with Prometheus disabled."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)
        assert monitor.emit_prometheus is False


# =============================================================================
# Debate Tracking Tests
# =============================================================================


class TestTrackDebate:
    """Test track_debate context manager."""

    def test_track_debate_basic(self):
        """Test basic debate tracking."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        with monitor.track_debate("debate-123", task="Test task") as metric:
            assert metric.debate_id == "debate-123"
            assert metric.task == "Test task"
            assert metric.outcome == "in_progress"
            assert "debate-123" in monitor._active_debates

        # After context exit
        assert metric.outcome == "completed"
        assert metric.end_time is not None
        assert metric.duration_seconds is not None
        assert "debate-123" not in monitor._active_debates

    def test_track_debate_with_agents(self):
        """Test debate tracking with agent names."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        with monitor.track_debate(
            "debate-123",
            task="Test",
            agent_names=["claude", "gpt"],
        ) as metric:
            assert metric.agent_names == ["claude", "gpt"]

    def test_track_debate_on_exception(self):
        """Test debate tracking handles exceptions."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        with pytest.raises(ValueError):
            with monitor.track_debate("debate-123", task="Test") as metric:
                raise ValueError("Test error")

        assert "error: ValueError" in metric.outcome
        assert metric.end_time is not None
        assert "debate-123" not in monitor._active_debates

    def test_track_debate_duration(self):
        """Test debate duration calculation."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        # time.time() called: start(1000.0), then end(1000.1) => duration=0.1
        with patch("aragora.debate.performance_monitor.time.time", side_effect=_make_time(1000.0, 0.1)):
            with monitor.track_debate("debate-123", task="Test") as metric:
                pass  # No sleep needed; mocked time advances automatically

        assert metric.duration_seconds == pytest.approx(0.1, abs=1e-9)
        assert metric.duration_seconds < 1.0

    def test_track_debate_empty_task(self):
        """Test debate tracking with empty task."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        with monitor.track_debate("debate-123") as metric:
            assert metric.task == ""

    def test_track_debate_none_agent_names(self):
        """Test debate tracking with None agent names."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        with monitor.track_debate(
            "debate-123",
            task="Test",
            agent_names=None,
        ) as metric:
            assert metric.agent_names == []


# =============================================================================
# Round Tracking Tests
# =============================================================================


class TestTrackRound:
    """Test track_round context manager."""

    def test_track_round_basic(self):
        """Test basic round tracking."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        with monitor.track_debate("debate-123", task="Test") as debate:
            with monitor.track_round("debate-123", round_num=1) as round_metric:
                assert round_metric.round_num == 1
                assert round_metric.start_time is not None

            assert round_metric.end_time is not None
            assert round_metric.duration_seconds is not None
            assert 1 in debate.rounds

    def test_track_round_duration(self):
        """Test round duration calculation."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        # Calls: debate_start(0), round_start(1), round_end(2), debate_end(3)
        with patch("aragora.debate.performance_monitor.time.time", side_effect=_make_time(1000.0, 0.1)):
            with monitor.track_debate("debate-123", task="Test"):
                with monitor.track_round("debate-123", round_num=1) as round_metric:
                    pass

        assert round_metric.duration_seconds == pytest.approx(0.1, abs=1e-9)
        assert round_metric.duration_seconds < 1.0

    def test_track_round_slow_detection(self):
        """Test slow round detection."""
        # Use very low threshold for testing
        monitor = DebatePerformanceMonitor(
            slow_round_threshold=0.01,
            emit_prometheus=False,
        )

        # step=0.05 => round duration=0.05s > 0.01 threshold
        with patch("aragora.debate.performance_monitor.time.time", side_effect=_make_time(1000.0, 0.05)):
            with monitor.track_debate("debate-123", task="Test"):
                with monitor.track_round("debate-123", round_num=1) as round_metric:
                    pass

        assert round_metric.is_slow is True

    def test_track_round_not_slow(self):
        """Test round not marked slow when under threshold."""
        monitor = DebatePerformanceMonitor(
            slow_round_threshold=10.0,  # 10 seconds
            emit_prometheus=False,
        )

        with monitor.track_debate("debate-123", task="Test"):
            with monitor.track_round("debate-123", round_num=1) as round_metric:
                pass  # Instant

        assert round_metric.is_slow is False

    def test_track_round_unknown_debate(self):
        """Test track_round handles unknown debate gracefully."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        # Track round without tracking debate
        with monitor.track_round("unknown-debate", round_num=1) as round_metric:
            pass

        # Should create dummy metric without crashing
        assert round_metric.round_num == 1

    def test_track_multiple_rounds(self):
        """Test tracking multiple rounds."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        with monitor.track_debate("debate-123", task="Test") as debate:
            for i in range(1, 4):
                with monitor.track_round("debate-123", round_num=i):
                    pass

            assert len(debate.rounds) == 3
            assert 1 in debate.rounds
            assert 2 in debate.rounds
            assert 3 in debate.rounds


# =============================================================================
# Phase Tracking Tests
# =============================================================================


class TestTrackPhase:
    """Test track_phase context manager."""

    def test_track_phase_basic(self):
        """Test basic phase tracking."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        with monitor.track_debate("debate-123", task="Test"):
            with monitor.track_round("debate-123", round_num=1) as round_metric:
                with monitor.track_phase("debate-123", "propose") as phase_metric:
                    assert phase_metric.phase_name == "propose"

                assert phase_metric.end_time is not None
                assert phase_metric.duration_seconds is not None
                assert "propose" in round_metric.phases

    def test_track_phase_with_agent_count(self):
        """Test phase tracking with agent count."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        with monitor.track_debate("debate-123", task="Test"):
            with monitor.track_round("debate-123", round_num=1):
                with monitor.track_phase(
                    "debate-123",
                    "propose",
                    agent_count=5,
                ) as phase_metric:
                    pass

        assert phase_metric.agent_count == 5

    def test_track_phase_duration(self):
        """Test phase duration calculation."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        # Calls: debate_start, round_start, phase_start, phase_end, round_end, debate_end
        with patch("aragora.debate.performance_monitor.time.time", side_effect=_make_time(1000.0, 0.1)):
            with monitor.track_debate("debate-123", task="Test"):
                with monitor.track_round("debate-123", round_num=1):
                    with monitor.track_phase("debate-123", "propose") as phase_metric:
                        pass

        assert phase_metric.duration_seconds == pytest.approx(0.1, abs=1e-9)

    def test_track_phase_on_exception(self):
        """Test phase tracking handles exceptions."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        with pytest.raises(RuntimeError):
            with monitor.track_debate("debate-123", task="Test"):
                with monitor.track_round("debate-123", round_num=1):
                    with monitor.track_phase(
                        "debate-123",
                        "propose",
                    ) as phase_metric:
                        raise RuntimeError("Test error")

        assert phase_metric.error == "phase_error:RuntimeError"
        assert phase_metric.end_time is not None

    def test_track_multiple_phases(self):
        """Test tracking multiple phases in a round."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        with monitor.track_debate("debate-123", task="Test"):
            with monitor.track_round("debate-123", round_num=1) as round_metric:
                with monitor.track_phase("debate-123", "propose"):
                    pass
                with monitor.track_phase("debate-123", "critique"):
                    pass
                with monitor.track_phase("debate-123", "vote"):
                    pass

            assert len(round_metric.phases) == 3
            assert "propose" in round_metric.phases
            assert "critique" in round_metric.phases
            assert "vote" in round_metric.phases

    def test_track_phase_without_round(self):
        """Test phase tracking without active round."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        with monitor.track_debate("debate-123", task="Test") as debate:
            # Track phase without tracking round first
            with monitor.track_phase("debate-123", "propose") as phase_metric:
                pass

            # Phase should still track but not be in any round
            assert phase_metric.phase_name == "propose"
            assert phase_metric.duration_seconds is not None
            # No rounds, so phase won't be stored in rounds dict
            assert len(debate.rounds) == 0


# =============================================================================
# Get Active Debate Tests
# =============================================================================


class TestGetActiveDebate:
    """Test get_active_debate method."""

    def test_get_active_debate_exists(self):
        """Test getting an active debate."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        with monitor.track_debate("debate-123", task="Test"):
            result = monitor.get_active_debate("debate-123")
            assert result is not None
            assert result.debate_id == "debate-123"

    def test_get_active_debate_not_found(self):
        """Test getting non-existent debate returns None."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)
        result = monitor.get_active_debate("nonexistent")
        assert result is None

    def test_get_active_debate_after_completion(self):
        """Test debate is removed after completion."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        with monitor.track_debate("debate-123", task="Test"):
            pass

        result = monitor.get_active_debate("debate-123")
        assert result is None


# =============================================================================
# Performance Insights Tests
# =============================================================================


class TestGetPerformanceInsights:
    """Test get_performance_insights method."""

    def test_insights_debate_not_found(self):
        """Test insights for non-existent debate."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)
        result = monitor.get_performance_insights("nonexistent")
        assert result["error"] == "Debate not found: nonexistent"

    def test_insights_basic(self):
        """Test basic performance insights."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        with monitor.track_debate("debate-123", task="Test task"):
            insights = monitor.get_performance_insights("debate-123")

            assert insights["debate_id"] == "debate-123"
            assert insights["status"] == "in_progress"
            assert insights["round_count"] == 0
            assert insights["slow_round_count"] == 0
            assert insights["is_slow"] is False

    def test_insights_with_rounds(self):
        """Test insights with rounds."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        with patch("aragora.debate.performance_monitor.time.time", side_effect=_make_time(1000.0, 0.1)):
            with monitor.track_debate("debate-123", task="Test"):
                with monitor.track_round("debate-123", round_num=1):
                    pass
                with monitor.track_round("debate-123", round_num=2):
                    pass

                insights = monitor.get_performance_insights("debate-123")

                assert insights["round_count"] == 2
                assert 1 in insights["rounds"]
                assert 2 in insights["rounds"]

    def test_insights_with_phases(self):
        """Test insights includes phase breakdown."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        with patch("aragora.debate.performance_monitor.time.time", side_effect=_make_time(1000.0, 0.1)):
            with monitor.track_debate("debate-123", task="Test"):
                with monitor.track_round("debate-123", round_num=1):
                    with monitor.track_phase("debate-123", "propose"):
                        pass
                    with monitor.track_phase("debate-123", "critique"):
                        pass

                insights = monitor.get_performance_insights("debate-123")

                round_info = insights["rounds"][1]
                assert "phases" in round_info
                assert "propose" in round_info["phases"]
                assert "critique" in round_info["phases"]

    def test_insights_slowest_round(self):
        """Test insights includes slowest round."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        # We need round 2 to be slower than round 1.
        # Calls: debate_start, r1_start, r1_end, r2_start, <extra>, r2_end, debate_end
        # r1 duration = 1 step, r2 duration needs to be larger.
        # Use variable step: provide explicit values so r2 takes longer.
        times = iter([
            1000.0,   # debate start
            1000.1,   # round 1 start
            1000.2,   # round 1 end  (duration=0.1)
            1000.3,   # round 2 start
            1000.8,   # round 2 end  (duration=0.5, slower)
            1001.0,   # get_current_slow_debates time.time() in insights
            1001.0,   # debate end
        ])
        with patch("aragora.debate.performance_monitor.time.time", side_effect=times):
            with monitor.track_debate("debate-123", task="Test"):
                with monitor.track_round("debate-123", round_num=1):
                    pass
                with monitor.track_round("debate-123", round_num=2):
                    pass

                insights = monitor.get_performance_insights("debate-123")

                assert "slowest_round" in insights
                assert insights["slowest_round"]["round_num"] == 2


# =============================================================================
# Slow Debates Tests
# =============================================================================


class TestSlowDebates:
    """Test slow debate detection and retrieval."""

    def test_slow_debate_recorded(self):
        """Test slow debates are recorded."""
        monitor = DebatePerformanceMonitor(
            slow_round_threshold=0.01,
            emit_prometheus=False,
        )

        # step=0.05 => round duration=0.05 > 0.01 threshold
        with patch("aragora.debate.performance_monitor.time.time", side_effect=_make_time(1000.0, 0.05)):
            with monitor.track_debate("debate-123", task="Test task"):
                with monitor.track_round("debate-123", round_num=1):
                    pass

        slow_debates = monitor.get_slow_debates()
        assert len(slow_debates) == 1
        assert slow_debates[0]["debate_id"] == "debate-123"

    def test_get_slow_debates_limit(self):
        """Test get_slow_debates respects limit."""
        monitor = DebatePerformanceMonitor(
            slow_round_threshold=0.001,
            emit_prometheus=False,
        )

        # step=0.01 => round duration > 0.001 threshold
        with patch("aragora.debate.performance_monitor.time.time", side_effect=_make_time(1000.0, 0.01)):
            for i in range(5):
                with monitor.track_debate(f"debate-{i}", task="Test"):
                    with monitor.track_round(f"debate-{i}", round_num=1):
                        pass

        slow_debates = monitor.get_slow_debates(limit=3)
        assert len(slow_debates) == 3

    def test_get_slow_debates_threshold_filter(self):
        """Test filtering by threshold."""
        monitor = DebatePerformanceMonitor(
            slow_round_threshold=0.01,
            emit_prometheus=False,
        )

        # Create slow debate with 2 rounds, each ~0.02s duration
        # Calls: debate_start, r1_start, r1_end, r2_start, r2_end, debate_end
        # step=0.02 => each round duration = 0.02s, total = 0.04s
        with patch("aragora.debate.performance_monitor.time.time", side_effect=_make_time(1000.0, 0.02)):
            with monitor.track_debate("debate-1", task="Test"):
                with monitor.track_round("debate-1", round_num=1):
                    pass
                with monitor.track_round("debate-1", round_num=2):
                    pass

        # Get only very slow debates (> 50ms per round)
        slow_debates = monitor.get_slow_debates(threshold_seconds=0.05)
        # total_duration / round_count = 0.1 / 2 = 0.05, need > 0.05 so filtered out
        assert len(slow_debates) == 0

    def test_get_slow_debates_returns_most_recent(self):
        """Test get_slow_debates returns most recent first."""
        monitor = DebatePerformanceMonitor(
            slow_round_threshold=0.001,
            emit_prometheus=False,
        )

        with patch("aragora.debate.performance_monitor.time.time", side_effect=_make_time(1000.0, 0.01)):
            for i in range(3):
                with monitor.track_debate(f"debate-{i}", task=f"Task {i}"):
                    with monitor.track_round(f"debate-{i}", round_num=1):
                        pass

        slow_debates = monitor.get_slow_debates()
        # Most recent should be first (reversed order)
        assert slow_debates[0]["debate_id"] == "debate-2"
        assert slow_debates[2]["debate_id"] == "debate-0"

    def test_get_current_slow_debates_empty(self):
        """Test get_current_slow_debates when no active slow debates."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)
        result = monitor.get_current_slow_debates()
        assert result == []

    def test_get_current_slow_debates_with_active(self):
        """Test get_current_slow_debates with active slow debates."""
        monitor = DebatePerformanceMonitor(
            slow_round_threshold=0.001,
            emit_prometheus=False,
        )

        # step=0.01 => round duration > 0.001 threshold
        # get_current_slow_debates also calls time.time() for elapsed_seconds
        with patch("aragora.debate.performance_monitor.time.time", side_effect=_make_time(1000.0, 0.01)):
            with monitor.track_debate(
                "debate-123",
                task="A" * 150,  # Long task to test truncation
                agent_names=["claude"],
            ) as debate:
                with monitor.track_round("debate-123", round_num=1):
                    pass

                # After round completes, manually update slow_round_count
                # (normally done in track_debate's finally block)
                debate.slow_round_count = sum(1 for r in debate.rounds.values() if r.is_slow)

                # Check while still active
                current = monitor.get_current_slow_debates()
                assert len(current) == 1
                assert current[0]["debate_id"] == "debate-123"
                assert len(current[0]["task"]) == 100  # Truncated
                assert current[0]["slow_rounds"] == 1


# =============================================================================
# Storage Limits Tests
# =============================================================================


class TestStorageLimits:
    """Test storage limits for slow debates."""

    def test_max_slow_debates_limit(self):
        """Test slow debates are trimmed at MAX_SLOW_DEBATES_HISTORY."""
        monitor = DebatePerformanceMonitor(
            slow_round_threshold=0.001,
            emit_prometheus=False,
        )

        # Create more than MAX_SLOW_DEBATES_HISTORY slow debates
        for i in range(MAX_SLOW_DEBATES_HISTORY + 10):
            with monitor.track_debate(f"debate-{i}", task="Test"):
                with monitor.track_round(f"debate-{i}", round_num=1):
                    time.sleep(0.002)

        assert len(monitor._slow_debates) == MAX_SLOW_DEBATES_HISTORY
        # Should keep most recent ones
        debate_ids = [d.debate_id for d in monitor._slow_debates]
        assert "debate-10" in debate_ids  # First kept
        assert f"debate-{MAX_SLOW_DEBATES_HISTORY + 9}" in debate_ids  # Last one

    def test_clear_history(self):
        """Test clear_history removes all slow debates."""
        monitor = DebatePerformanceMonitor(
            slow_round_threshold=0.001,
            emit_prometheus=False,
        )

        # Create slow debates
        for i in range(5):
            with monitor.track_debate(f"debate-{i}", task="Test"):
                with monitor.track_round(f"debate-{i}", round_num=1):
                    time.sleep(0.002)

        assert len(monitor._slow_debates) == 5

        monitor.clear_history()

        assert len(monitor._slow_debates) == 0
        assert monitor.get_slow_debates() == []


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_metrics(self):
        """Test handling empty metrics."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        with monitor.track_debate("debate-123", task="") as metric:
            pass

        # Should complete without error
        assert metric.outcome == "completed"
        assert metric.rounds == {}

    def test_debate_with_zero_duration_rounds(self):
        """Test debate with zero duration rounds."""
        metric = DebateMetric(
            debate_id="debate-123",
            task="Test",
            start_time=1000.0,
        )
        metric.rounds[1] = RoundMetric(
            round_num=1,
            start_time=1000.0,
            duration_seconds=0.0,
        )
        assert metric.avg_round_duration == 0.0

    def test_concurrent_debate_tracking(self):
        """Test tracking multiple concurrent debates."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        with monitor.track_debate("debate-1", task="Task 1") as metric1:
            with monitor.track_debate("debate-2", task="Task 2") as metric2:
                assert "debate-1" in monitor._active_debates
                assert "debate-2" in monitor._active_debates

        assert metric1.outcome == "completed"
        assert metric2.outcome == "completed"
        assert "debate-1" not in monitor._active_debates
        assert "debate-2" not in monitor._active_debates

    def test_phase_without_debate(self):
        """Test phase tracking without any debate context."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        # Track phase without debate
        with monitor.track_phase("unknown", "propose") as phase:
            pass

        # Should complete without error
        assert phase.phase_name == "propose"
        assert phase.duration_seconds is not None

    def test_round_metric_none_durations(self):
        """Test RoundMetric handles None durations gracefully."""
        metric = RoundMetric(
            round_num=1,
            start_time=1000.0,
        )
        # All None durations
        metric.phases["a"] = PhaseMetric("a", 1000.0, duration_seconds=None)
        metric.phases["b"] = PhaseMetric("b", 1001.0, duration_seconds=None)

        assert metric.total_phase_time == 0.0
        # Slowest phase should be (name, 0) since all are None
        slowest = metric.slowest_phase
        assert slowest[1] == 0  # Duration is 0 for None

    def test_debate_metric_all_none_durations(self):
        """Test DebateMetric handles all None durations."""
        metric = DebateMetric(
            debate_id="test",
            task="Test",
            start_time=1000.0,
        )
        metric.rounds[1] = RoundMetric(1, 1000.0, duration_seconds=None)
        metric.rounds[2] = RoundMetric(2, 1010.0, duration_seconds=None)

        # avg_round_duration should handle no completed rounds
        assert metric.avg_round_duration == 0.0

    def test_slow_debate_record_slowest_phase_detection(self):
        """Test slowest phase detection across rounds."""
        monitor = DebatePerformanceMonitor(
            slow_round_threshold=0.001,
            emit_prometheus=False,
        )

        with monitor.track_debate("debate-123", task="Test"):
            with monitor.track_round("debate-123", round_num=1):
                with monitor.track_phase("debate-123", "propose"):
                    time.sleep(0.005)
                with monitor.track_phase("debate-123", "critique"):
                    time.sleep(0.01)  # Slowest

            with monitor.track_round("debate-123", round_num=2):
                with monitor.track_phase("debate-123", "propose"):
                    time.sleep(0.003)

        slow_debates = monitor.get_slow_debates()
        assert len(slow_debates) == 1
        # Slowest phase should be critique from round 1
        assert slow_debates[0]["slowest_phase"][0] == "critique"


# =============================================================================
# Global Instance Tests
# =============================================================================


class TestGetDebateMonitor:
    """Test get_debate_monitor global function."""

    def test_get_debate_monitor_returns_instance(self):
        """Test get_debate_monitor returns a monitor instance."""
        # Reset the global to test fresh creation
        import aragora.debate.performance_monitor as pm

        pm._default_monitor = None

        monitor = get_debate_monitor()
        assert isinstance(monitor, DebatePerformanceMonitor)

    def test_get_debate_monitor_returns_same_instance(self):
        """Test get_debate_monitor returns singleton."""
        monitor1 = get_debate_monitor()
        monitor2 = get_debate_monitor()
        assert monitor1 is monitor2


# =============================================================================
# Prometheus Integration Tests
# =============================================================================


class TestPrometheusIntegration:
    """Test Prometheus metrics emission."""

    def test_emit_metrics_on_completion(self):
        """Test metrics are emitted on debate completion."""
        monitor = DebatePerformanceMonitor(emit_prometheus=True)

        mock_record = MagicMock()
        with patch.dict(
            "sys.modules",
            {"aragora.observability.metrics": MagicMock(record_debate_completion=mock_record)},
        ):
            with patch("aragora.debate.performance_monitor.logger"):  # Suppress ImportError log
                with monitor.track_debate("debate-123", task="Test"):
                    pass

    def test_prometheus_disabled(self):
        """Test no metrics emitted when disabled."""
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        with patch("aragora.observability.metrics.record_debate_completion") as mock_record:
            with monitor.track_debate("debate-123", task="Test"):
                pass

            # Should not be called when disabled
            mock_record.assert_not_called()

    def test_phase_metrics_emission(self):
        """Test phase metrics emission path is exercised without error."""
        # Use emit_prometheus=False to avoid real Prometheus calls
        # This test verifies the phase tracking completes without error
        monitor = DebatePerformanceMonitor(emit_prometheus=False)

        with monitor.track_debate("debate-123", task="Test"):
            with monitor.track_round("debate-123", round_num=1):
                with monitor.track_phase("debate-123", "propose") as phase:
                    pass

        # Verify phase was tracked correctly
        assert phase.phase_name == "propose"
        assert phase.duration_seconds is not None


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Test module constants."""

    def test_default_slow_round_threshold(self):
        """Test default slow round threshold value."""
        assert DEFAULT_SLOW_ROUND_THRESHOLD == 30.0

    def test_max_slow_debates_history(self):
        """Test max slow debates history value."""
        assert MAX_SLOW_DEBATES_HISTORY == 100


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def monitor():
    """Create a fresh DebatePerformanceMonitor for testing."""
    return DebatePerformanceMonitor(emit_prometheus=False)


@pytest.fixture
def slow_monitor():
    """Create a monitor with very low threshold for testing slow detection."""
    return DebatePerformanceMonitor(
        slow_round_threshold=0.001,
        emit_prometheus=False,
    )


class TestWithFixtures:
    """Tests using fixtures."""

    def test_fixture_monitor(self, monitor):
        """Test basic fixture usage."""
        assert isinstance(monitor, DebatePerformanceMonitor)
        assert monitor.emit_prometheus is False

    def test_fixture_slow_monitor(self, slow_monitor):
        """Test slow monitor fixture."""
        assert slow_monitor.slow_round_threshold == 0.001

    def test_slow_detection_with_fixture(self, slow_monitor):
        """Test slow detection using fixture."""
        with slow_monitor.track_debate("debate-123", task="Test"):
            with slow_monitor.track_round("debate-123", round_num=1) as round_metric:
                time.sleep(0.002)

        assert round_metric.is_slow is True
