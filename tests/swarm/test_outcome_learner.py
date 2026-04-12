"""Tests for OutcomeLearner rolling summaries."""

from pathlib import Path

from aragora.swarm.outcome_learner import OutcomeLearner, load_category_success_rates
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


def test_snapshot_exposes_category_success_rates_from_issue_titles():
    learner = OutcomeLearner(window_size=10, min_samples=1, merge_rate_threshold=0.5)
    learner.ingest(
        _signal(
            entity_title="Narrow broad except Exception in foo.py",
            did_merge=True,
        )
    )
    learner.ingest(
        _signal(
            signal_type="failed",
            entity_title="Add return type annotations to bar.py",
            failure_reason="worker_crash",
        )
    )

    snap = learner.snapshot()

    assert snap.routing_hints["category_success_rates"]["broad_exception"] == 1.0
    assert snap.routing_hints["category_success_rates"]["type_annotation"] == 0.0
    assert "type_annotation" in snap.routing_hints["deprioritize_categories"]
    assert "broad_exception" not in snap.routing_hints["deprioritize_categories"]


def test_snapshot_ignores_unclassified_issue_titles():
    learner = OutcomeLearner(window_size=10, min_samples=1, merge_rate_threshold=0.5)
    learner.ingest(_signal(entity_title="Unclassified task title", did_merge=True))

    snap = learner.snapshot()

    assert snap.routing_hints["category_success_rates"] == {}
    assert snap.routing_hints["deprioritize_categories"] == []


def test_load_category_success_rates_reads_recent_signal_log(tmp_path: Path):
    log_path = tmp_path / "outcome_signals.jsonl"
    rows = [
        _signal(
            entity_id="1",
            entity_title="Replace silent exception swallowing in foo.py",
            signal_type="completed",
        ).to_dict(),
        _signal(
            entity_id="2",
            entity_title="Replace silent exception swallowing in foo.py",
            signal_type="failed",
        ).to_dict(),
        _signal(
            entity_id="3",
            entity_title="Add request body validation to bar.py handlers",
            signal_type="completed",
        ).to_dict(),
    ]
    log_path.write_text("\n".join(__import__("json").dumps(row) for row in rows) + "\n")

    rates = load_category_success_rates(log_path=log_path, window_size=10, min_samples=1)

    assert rates == {
        "handler_validation": 1.0,
        "silent_exception": 0.5,
    }


def test_load_category_success_rates_respects_min_samples(tmp_path: Path):
    log_path = tmp_path / "outcome_signals.jsonl"
    row = _signal(
        entity_id="1",
        entity_title="Add unit tests for foo.py",
        signal_type="completed",
    ).to_dict()
    log_path.write_text(__import__("json").dumps(row) + "\n")

    rates = load_category_success_rates(log_path=log_path, window_size=10, min_samples=2)

    assert rates == {}
