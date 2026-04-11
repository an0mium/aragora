from __future__ import annotations

from aragora.pipeline.execution_mode import ExecutionMode
from aragora.swarm.worker_contract import build_worker_contract
from aragora.swarm.worker_process import LaunchConfig


def test_build_worker_contract_includes_mission_lineage_and_context_policy(tmp_path) -> None:
    worktree = tmp_path / "repo"
    worktree.mkdir()
    config = LaunchConfig(
        allow_codex_full_auto=True,
        execution_mode=ExecutionMode.AUTONOMOUS,
    )

    contract = build_worker_contract(
        agent="codex",
        config=config,
        worktree_path=str(worktree),
        env={},
        work_order={
            "mission_id": "mission-rs-credential-envelope",
            "stage_id": "stage-contract-aware-preflight",
            "assertion_ids": ["RS-04-ASSERT-1"],
            "file_scope": ["aragora/swarm/preflight.py"],
            "evidence_expectations": ["validation_command", "worker_contract", "receipt"],
        },
    )

    payload = contract.to_dict()

    assert payload["mission_id"] == "mission-rs-credential-envelope"
    assert payload["stage_id"] == "stage-contract-aware-preflight"
    assert payload["assertion_ids"] == ["RS-04-ASSERT-1"]
    assert payload["mission_context_policy"]["role"] == "worker"
    assert payload["mission_context_policy"]["transcript_allowance"] == "none"
    contract.validate()
