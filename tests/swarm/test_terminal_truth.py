from __future__ import annotations

import json
from pathlib import Path

from aragora.swarm.terminal_truth import (
    classify_from_metrics,
    qualify_run_terminal_state,
    qualify_work_order_terminal_state,
)


def test_qualify_work_order_terminal_state_preserves_mission_lineage() -> None:
    qualification = qualify_work_order_terminal_state(
        {
            "status": "completed",
            "branch": "feature/mission",
            "commit_shas": ["deadbeef"],
            "mission_id": "mission-rs-credential-envelope",
            "stage_id": "stage-contract-aware-preflight",
            "assertion_ids": ["RS-04-ASSERT-1"],
            "evidence_expectations": ["validation_command", "worker_contract", "receipt"],
            "gate_expectations": {
                "dispatch_ready": {
                    "gate_type": "dispatch_ready",
                    "verdict": "pass",
                    "failure_classes": [],
                }
            },
        }
    )

    assert qualification.mission_id == "mission-rs-credential-envelope"
    assert qualification.stage_id == "stage-contract-aware-preflight"
    assert qualification.assertion_ids == ["RS-04-ASSERT-1"]
    assert qualification.evidence_expectations == [
        "validation_command",
        "worker_contract",
        "receipt",
    ]
    assert qualification.gate_evaluations[0]["gate_type"] == "dispatch_ready"


def test_qualify_run_terminal_state_derives_failure_classes_from_lineage_and_outcome() -> None:
    qualification = qualify_run_terminal_state(
        {
            "status": "needs_human",
            "work_orders": [
                {
                    "status": "needs_human",
                    "worker_outcome": "scope_violation",
                    "blockers": ["unsafe scope expansion"],
                    "mission_id": "mission-bc-admission",
                    "stage_id": "stage-dispatch-ready",
                    "assertion_ids": ["BC-04-ASSERT-1"],
                    "gate_expectations": {
                        "dispatch_ready": {
                            "gate_type": "dispatch_ready",
                            "verdict": "blocked",
                            "failure_classes": ["unsafe_scope"],
                        }
                    },
                }
            ],
        }
    )

    assert qualification.mission_id == "mission-bc-admission"
    assert qualification.stage_id == "stage-dispatch-ready"
    assert qualification.failure_classes == ["unsafe_scope", "scope_violation"]
    assert qualification.to_dict()["receipt_outcome"] == "blocked"


def test_terminal_truth_fixtures_match_expected_classes() -> None:
    fixture_dir = (
        Path(__file__).resolve().parents[2] / "benchmarks" / "fixtures" / "swarm" / "terminal_truth"
    )
    fixture_files = sorted(fixture_dir.glob("*.json"))

    assert fixture_files, "expected terminal-truth fixtures to exist"

    for fixture_file in fixture_files:
        rows = json.loads(fixture_file.read_text(encoding="utf-8"))
        assert isinstance(rows, list), f"{fixture_file.name} must contain a list of examples"

        for row in rows:
            expected = row["expected_class"]
            observed = classify_from_metrics(row).value
            assert observed == expected, (
                f"{fixture_file.name}: expected {expected!r}, observed {observed!r} for row {row!r}"
            )
