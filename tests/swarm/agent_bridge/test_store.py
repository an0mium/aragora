from __future__ import annotations

from pathlib import Path

import pytest

from aragora.swarm.agent_bridge.store import BridgeStore
from aragora.swarm.agent_bridge.types import BridgeRun
from aragora.swarm.agent_bridge.types import BridgeSession
from aragora.swarm.agent_bridge.types import SessionRegistry
from aragora.swarm.agent_bridge.types import TurnRecord


def test_save_load_save_is_byte_identical(tmp_path: Path) -> None:
    store = BridgeStore(tmp_path)
    run = BridgeRun(
        run_id="bridge_store",
        created_at="2026-04-21T20:00:00Z",
        updated_at="2026-04-21T20:00:00Z",
        status="running",
        active_role="reviewer",
        footer_mode="prompt_injected",
        participants=["reviewer", "implementer"],
        worktree_path=str(tmp_path),
        worktree_agent_slug="codex",
    )
    registry = SessionRegistry(
        roles={
            "reviewer": BridgeSession(
                harness="codex",
                session_id=None,
                created_at="2026-04-21T20:00:00Z",
                last_turn_at=None,
                harness_options={"model": "gpt-5.4"},
            )
        }
    )

    store.save_run(run)
    store.save_sessions(run.run_id, registry)
    first_run_bytes = store.run_path(run.run_id).read_text(encoding="utf-8")
    first_sessions_bytes = store.sessions_path(run.run_id).read_text(encoding="utf-8")

    store.save_run(store.load_run(run.run_id))
    store.save_sessions(run.run_id, store.load_sessions(run.run_id))

    assert store.run_path(run.run_id).read_text(encoding="utf-8") == first_run_bytes
    assert store.sessions_path(run.run_id).read_text(encoding="utf-8") == first_sessions_bytes


def test_atomic_write_leaves_prior_run_file_intact_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = BridgeStore(tmp_path)
    run = BridgeRun(
        run_id="bridge_atomic",
        created_at="2026-04-21T20:00:00Z",
        updated_at="2026-04-21T20:00:00Z",
        status="running",
        active_role="reviewer",
        footer_mode="prompt_injected",
        participants=["reviewer"],
        worktree_path=str(tmp_path),
        worktree_agent_slug="codex",
    )
    store.save_run(run)
    original = store.run_path(run.run_id).read_text(encoding="utf-8")

    def explode_replace(source: Path, target: Path) -> None:
        del source, target
        raise OSError("simulated crash")

    monkeypatch.setattr(store, "_replace_file", explode_replace)
    run.updated_at = "2026-04-21T20:05:00Z"

    with pytest.raises(OSError, match="simulated crash"):
        store.save_run(run)

    assert store.run_path(run.run_id).read_text(encoding="utf-8") == original


def test_append_event_is_idempotent(tmp_path: Path) -> None:
    store = BridgeStore(tmp_path)
    event = TurnRecord(
        event_id="bridge_store:turn:001:turn_started:0",
        turn_index=1,
        type="turn_started",
        role="reviewer",
        parse_status=None,
        at="2026-04-21T20:00:00Z",
        payload={"prompt": "Review this"},
    )

    assert store.append_event("bridge_store", event) is True
    assert store.append_event("bridge_store", event) is False
    assert len(store.events_path("bridge_store").read_text(encoding="utf-8").splitlines()) == 1
    assert store.load_events("bridge_store") == [event]
