"""Tests for OutcomeLearner rolling summaries."""

from aragora.swarm.outcome_learner import OutcomeLearner
from aragora.swarm.outcome_signals import OutcomeSignal


def _signal(**overrides) -> OutcomeSignal:
    data = {
        "source_loop": "boss",
        "signal_type": "completed",
        "entity_id": "1",
        "agent_type": "codex",
    }
    data.update(overrides)
    return OutcomeSignal(**data)


def test_snapshot_rolls_up_by_loop_and_agent():
    learner = OutcomeLearner(window_size=10, min_samples=1, merge_rate_threshold=0.6)
    learner.ingest(_signal(did_merge=True))
    learner.ingest(_signal(signal_type="failed", agent_type="claude", failure_reason="boom"))
    learner.ingest(_signal(source_loop="nomic", signal_type="repaired", agent_type="codex"))

    snap = learner.snapshot()
    assert snap.total_signals == 3
    assert snap.by_loop["boss"].total == 2
    assert snap.by_loop["nomic"].total == 1
    assert snap.by_agent["codex"].total == 2
    assert snap.by_agent["claude"].total == 1
    assert snap.failure_taxonomy["boom"] == 1


def test_recommendations_and_routing_hints_are_deterministic():
    learner = OutcomeLearner(window_size=5, min_samples=1, merge_rate_threshold=0.5)
    learner.ingest(
        _signal(
            signal_type="failed",
            agent_type="claude",
            failure_reason="worker_exited_without_receipt",
            blocker_kind="missing_dependency",
        )
    )
    snap = learner.snapshot()
    assert any("worker_exited_without_receipt" in rec for rec in snap.recommendations)
    assert any("missing_dependency" in rec for rec in snap.recommendations)
    assert "deprioritize_loops" in snap.routing_hints
    assert "boss" in snap.routing_hints["deprioritize_loops"]
    assert "claude" in snap.routing_hints["deprioritize_agents"]
