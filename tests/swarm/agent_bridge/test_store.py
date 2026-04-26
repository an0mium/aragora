from __future__ import annotations

from pathlib import Path

import pytest

from aragora.swarm.agent_bridge.footer import extract_footer
from aragora.swarm.agent_bridge.store import BridgeStore
from aragora.swarm.agent_bridge.types import BridgeRun
from aragora.swarm.agent_bridge.types import BridgeSession
from aragora.swarm.agent_bridge.types import Participant
from aragora.swarm.agent_bridge.types import SessionRegistry
from aragora.swarm.agent_bridge.types import TurnRecord


def test_save_load_save_is_byte_identical(tmp_path: Path) -> None:
    store = BridgeStore(tmp_path)
    run = BridgeRun(
        run_id="bridge_store",
        task="Review the plan",
        created_at="2026-04-21T20:00:00Z",
        updated_at="2026-04-21T20:00:00Z",
        status="running",
        completed_at=None,
        last_turn_index=0,
        next_actor="reviewer",
        repair_budget_per_turn=1,
        footer_mode="prompt_injected",
        worktree_cleanup_mode="operator_triggered",
        participants=[Participant(role="reviewer", harness="codex", model="gpt-5.4")],
        worktree_path=str(tmp_path),
        worktree_agent_slug="codex",
    )
    registry = SessionRegistry(
        run_id=run.run_id,
        updated_at="2026-04-21T20:00:00Z",
        sessions={
            "reviewer": BridgeSession(
                role="reviewer",
                harness="codex",
                model="gpt-5.4",
                session_id=None,
                worktree_agent_slug=None,
                worktree_path=None,
                branch=None,
                session_status="not_started",
                started_at=None,
                last_turn_index=0,
                last_completed_at=None,
            )
        },
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
        task="Review the plan",
        created_at="2026-04-21T20:00:00Z",
        updated_at="2026-04-21T20:00:00Z",
        status="running",
        completed_at=None,
        last_turn_index=0,
        next_actor="reviewer",
        repair_budget_per_turn=1,
        footer_mode="prompt_injected",
        worktree_cleanup_mode="operator_triggered",
        participants=[Participant(role="reviewer", harness="codex", model="gpt-5.4")],
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
        event_id="bridge_store:turn:001:turn.result:0",
        run_id="bridge_store",
        turn_index=1,
        event_type="turn.result",
        role="reviewer",
        harness="codex",
        session_id="thread-1",
        parse_status="ok",
        ts="2026-04-21T20:00:00Z",
        payload={"prompt": "Review this"},
    )

    assert store.append_event("bridge_store", event) is True
    assert store.append_event("bridge_store", event) is False
    assert len(store.events_path("bridge_store").read_text(encoding="utf-8").splitlines()) == 1
    assert store.load_events("bridge_store") == [event]


def test_write_turn_transcript_uses_scoped_front_matter_and_repair_section(tmp_path: Path) -> None:
    store = BridgeStore(tmp_path)
    initial = extract_footer(
        "Body\n\n---BRIDGE-FOOTER---\nsummary: Done\nnext_actor: reviewer\nneeds_human: false\ndone: false\nartifacts: []\ntests_run: []\n---BRIDGE-FOOTER-END---",
        allowed_roles={"reviewer"},
    )
    repair = extract_footer(
        "---BRIDGE-FOOTER---\nsummary: Fixed footer\nnext_actor: reviewer\nneeds_human: false\ndone: false\nartifacts: []\ntests_run: []\n---BRIDGE-FOOTER-END---",
        allowed_roles={"reviewer"},
    )

    transcript_path = store.write_turn_transcript(
        "bridge_store",
        turn_index=1,
        role="reviewer",
        harness="codex",
        model="gpt-5.4",
        session_id="thread-1",
        started_at="2026-04-21T20:00:00Z",
        completed_at="2026-04-21T20:01:00Z",
        exit_code=0,
        prompt="Review this",
        raw_stdout="stdout",
        raw_stderr="stderr",
        parsed_turn=initial,
        repair_attempts=[
            {
                "prompt": "Repair footer",
                "raw_stdout": "repair stdout",
                "raw_stderr": "",
                "parsed_turn": repair,
            }
        ],
    )

    text = transcript_path.read_text(encoding="utf-8")

    assert transcript_path.name == "001-codex-reviewer.md"
    assert "schema_version: 1" in text
    assert "run_id: bridge_store" in text
    assert "## Prompt" in text
    assert "## Raw Stdout" in text
    assert "## Raw Stderr" in text
    assert "## Parsed Message" in text
    assert "## Footer" in text
    assert "## Repair Attempt 1" in text
