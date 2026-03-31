"""Tests for cross-loop outcome signal bus."""

import json
import tempfile
from pathlib import Path

from aragora.swarm.outcome_signals import (
    CalibrationSnapshot,
    GeneratedGoal,
    OutcomeSignal,
    OutcomeSignalBus,
    apply_calibration_to_estimator,
    compute_calibration,
    generate_goals_from_outcomes,
)


def _make_signal(
    source_loop: str = "boss",
    signal_type: str = "completed",
    entity_id: str = "1",
    **kwargs,
) -> OutcomeSignal:
    return OutcomeSignal(
        source_loop=source_loop,
        signal_type=signal_type,
        entity_id=entity_id,
        **kwargs,
    )


class TestOutcomeSignal:
    def test_success_and_failure_properties(self):
        s = _make_signal(signal_type="completed")
        assert s.is_success
        assert not s.is_failure

        f = _make_signal(signal_type="failed")
        assert f.is_failure
        assert not f.is_success

    def test_timestamp_auto_populated(self):
        s = _make_signal()
        assert s.timestamp  # Not empty
        assert "T" in s.timestamp  # ISO format

    def test_to_dict_roundtrip(self):
        s = _make_signal(entity_title="Fix bug", debate_id="debate-123", tokens_used=50000)
        d = s.to_dict()
        assert d["entity_title"] == "Fix bug"
        assert d["debate_id"] == "debate-123"
        assert d["tokens_used"] == 50000
        # Should be JSON-serializable
        json.dumps(d)

    def test_to_dict_includes_elapsed_seconds(self):
        s = _make_signal(elapsed_seconds=12.5)
        d = s.to_dict()
        assert "elapsed_seconds" in d
        assert d["elapsed_seconds"] == 12.5

    def test_debate_id_defaults_to_empty_string(self):
        s = _make_signal()
        assert s.debate_id == ""


class TestOutcomeSignalBus:
    def test_emit_persists_to_jsonl(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        bus = OutcomeSignalBus(log_path=path)
        bus.emit(_make_signal(entity_id="42"))

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["entity_id"] == "42"
        path.unlink()

    def test_subscribers_notified(self):
        received = []
        bus = OutcomeSignalBus(log_path=Path(tempfile.mktemp(suffix=".jsonl")))
        bus.subscribe_all(lambda s: received.append(s))

        bus.emit(_make_signal())
        assert len(received) == 1

    def test_handler_failure_does_not_break_bus(self):
        def bad_handler(s):
            raise RuntimeError("boom")

        bus = OutcomeSignalBus(log_path=Path(tempfile.mktemp(suffix=".jsonl")))
        bus.subscribe_all(bad_handler)
        # Should not raise
        bus.emit(_make_signal())
        assert bus.total_emitted == 1

    def test_recent_returns_signals(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        bus = OutcomeSignalBus(log_path=path)
        for i in range(5):
            bus.emit(_make_signal(entity_id=str(i)))

        recent = bus.recent(minutes=5)
        assert len(recent) == 5
        path.unlink()


class TestGoalGeneration:
    def test_generates_goal_from_repeated_failures(self):
        signals = [
            _make_signal(signal_type="failed", failure_reason="worker_exited_without_receipt")
            for _ in range(5)
        ]
        goals = generate_goals_from_outcomes(signals, min_failures=3)
        assert len(goals) >= 1
        assert "worker_exited_without_receipt" in goals[0].goal_text

    def test_generates_goal_from_low_merge_rate(self):
        signals = [_make_signal(signal_type="failed", did_merge=False) for _ in range(8)]
        signals.append(_make_signal(signal_type="completed", did_merge=True))
        goals = generate_goals_from_outcomes(signals, min_failures=2)
        merge_goals = [g for g in goals if "merge rate" in g.goal_text.lower()]
        assert len(merge_goals) >= 1

    def test_generates_goal_from_ralph_blockers(self):
        signals = [
            _make_signal(
                source_loop="ralph",
                signal_type="blocked",
                blocker_kind="worker_context_overflow",
            )
            for _ in range(3)
        ]
        goals = generate_goals_from_outcomes(signals, min_failures=2)
        assert any("worker_context_overflow" in g.goal_text for g in goals)

    def test_no_goals_from_empty_signals(self):
        assert generate_goals_from_outcomes([]) == []

    def test_goals_sorted_by_score(self):
        signals = [_make_signal(signal_type="failed", failure_reason="a") for _ in range(5)] + [
            _make_signal(signal_type="failed", failure_reason="b", tokens_used=100_000)
            for _ in range(5)
        ]
        goals = generate_goals_from_outcomes(signals, min_failures=3)
        if len(goals) >= 2:
            assert goals[0].score >= goals[1].score


class TestCalibration:
    def _make_signals(self, n: int = 10) -> list[OutcomeSignal]:
        signals = []
        for i in range(n):
            signals.append(
                _make_signal(
                    signal_type="completed" if i % 3 == 0 else "failed",
                    did_merge=i % 3 == 0,
                    needed_human_rescue=i % 4 == 0,
                    tokens_used=50000 + i * 1000,
                    elapsed_seconds=300 + i * 30,
                    agent_type="codex" if i % 2 == 0 else "claude",
                )
            )
        return signals

    def test_compute_calibration_returns_snapshot(self):
        signals = self._make_signals(10)
        snap = compute_calibration(signals)
        assert snap is not None
        assert snap.total_outcomes == 10
        assert 0 <= snap.merge_rate <= 1.0
        assert snap.avg_tokens_per_task > 0

    def test_compute_calibration_needs_minimum_signals(self):
        assert compute_calibration([_make_signal()]) is None

    def test_calibration_generates_recommendations(self):
        # All failures → should recommend investigating
        signals = [_make_signal(signal_type="failed") for _ in range(10)]
        snap = compute_calibration(signals)
        assert snap is not None
        assert len(snap.recommendations) > 0

    def test_apply_calibration_to_estimator(self):
        snap = CalibrationSnapshot(
            timestamp="",
            total_outcomes=20,
            merge_rate=0.2,
            avg_tokens_per_task=50000,
            avg_minutes_per_task=10.0,
            rescue_rate=0.4,
            loop_merge_rates={"boss": 0.2},
            blocker_frequency={"worker_context_overflow": 6},
            agent_success_rates={"codex": 0.1, "claude": 0.8},
            recommendations=[],
        )
        adjustments = apply_calibration_to_estimator(snap)
        assert "agent_penalty_codex" in adjustments
        assert "blocker_penalty_worker_context_overflow" in adjustments
        assert "global_p_success_damper" in adjustments

    def test_per_loop_merge_rates(self):
        signals = [
            _make_signal(source_loop="boss", signal_type="completed", did_merge=True),
            _make_signal(source_loop="boss", signal_type="failed"),
            _make_signal(source_loop="nomic", signal_type="completed", did_merge=True),
            _make_signal(source_loop="nomic", signal_type="completed", did_merge=True),
            _make_signal(source_loop="ralph", signal_type="failed"),
        ]
        snap = compute_calibration(signals)
        assert snap is not None
        assert snap.loop_merge_rates["boss"] == 0.5
        assert snap.loop_merge_rates["nomic"] == 1.0
        assert snap.loop_merge_rates["ralph"] == 0.0
