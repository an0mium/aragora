from __future__ import annotations

from pathlib import Path

from aragora.swarm.session_state import SessionStateStore
from aragora.swarm.supervisor_workers import _record_session_state


def test_record_session_state_persists_boss_repo(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("aragora.swarm.session_state.Path.home", lambda: tmp_path)
    work_order = {
        "work_order_id": "wo-123",
        "issue_number": 4242,
        "target_agent": "codex",
        "metadata": {"boss_repo": "synaptent/aragora"},
    }

    _record_session_state(
        work_order,
        status="needs_human",
        phase="repair",
        exit_code=1,
        worker_outcome="needs_human",
        changed_files=["aragora/swarm/boss_loop.py"],
        test_output="AssertionError: boom",
    )

    payload = work_order["metadata"]["session_state"]
    store = SessionStateStore()
    state = store.load(payload["session_id"])

    assert payload["metadata"]["boss_repo"] == "synaptent/aragora"
    assert state is not None
    assert state.metadata["boss_repo"] == "synaptent/aragora"
