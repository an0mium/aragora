from __future__ import annotations

import json
from pathlib import Path

from aragora.swarm.boss_loop_outcome import append_iteration_metrics


def test_append_iteration_metrics_persists_dispatch_gate_as_blocker_evidence(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "boss_metrics.jsonl"
    dispatch_gate = {
        "gate_type": "dispatch_ready",
        "verdict": "blocked",
        "failure_classes": ["contract_missing"],
        "repair_eligible": True,
        "required_evidence": ["validation_contract"],
        "notes": "validation contract required before dispatch",
    }

    append_iteration_metrics(
        metrics_jsonl_path=str(metrics_path),
        outcome_learner_window=5,
        deferred_queue_depth=0,
        iteration=1,
        issue_number=42,
        worker_result={
            "status": "needs_human",
            "outcome": "blocked",
            "reasons": ["Issue #42 is missing a validation contract."],
            "dispatch_gate": dispatch_gate,
            "receipt_metadata": {"issue_title": "Require validation contract"},
        },
        elapsed_seconds=1.25,
        files_changed=0,
        tests_run=0,
        tests_passed=0,
    )

    row = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[-1])

    assert row["failure_reason"] == "Issue #42 is missing a validation contract."
    assert json.loads(row["blocker_evidence"]) == dispatch_gate
    assert row["terminal_class"] == "blocked_not_dispatch_bounded"
