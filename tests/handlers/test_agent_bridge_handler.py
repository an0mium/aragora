from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aragora.rbac.models import AuthorizationContext
from aragora.server.handlers.agent_bridge import AgentBridgeHandler
from aragora.swarm.agent_bridge.store import BridgeStore
from aragora.swarm.agent_bridge.types import BridgeFooter
from aragora.swarm.agent_bridge.types import BridgeRun
from aragora.swarm.agent_bridge.types import BridgeSession
from aragora.swarm.agent_bridge.types import Participant
from aragora.swarm.agent_bridge.types import SessionRegistry
from aragora.swarm.agent_bridge.types import TurnRecord


def _parse_json_body(result: object) -> dict[str, object]:
    body = getattr(result, "body", b"")
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    return json.loads(body) if body else {}


def _mock_http_handler(*, headers: dict[str, str] | None = None) -> MagicMock:
    mock = MagicMock()
    mock.headers = headers or {}
    mock.client_address = ("127.0.0.1", 12345)
    return mock


def _footer_text(
    body: str,
    *,
    summary: str,
    next_actor: str | None,
    needs_human: bool = False,
    done: bool = False,
) -> str:
    next_actor_text = "null" if next_actor is None else next_actor
    return (
        f"{body}\n\n---BRIDGE-FOOTER---\n"
        f"summary: {summary}\n"
        f"next_actor: {next_actor_text}\n"
        f"needs_human: {'true' if needs_human else 'false'}\n"
        f"done: {'true' if done else 'false'}\n"
        "artifacts: []\n"
        "tests_run: []\n"
        "---BRIDGE-FOOTER-END---"
    )


def _turn_event(
    *,
    run_id: str,
    turn_index: int,
    seq: int,
    event_type: str,
    role: str,
    ts: str,
    payload: dict[str, object] | None = None,
    parse_status: str | None = None,
    harness: str = "codex",
    session_id: str | None = "sess-impl",
) -> TurnRecord:
    return TurnRecord(
        event_id=f"{run_id}:turn:{turn_index:03d}:{event_type}:{seq}",
        run_id=run_id,
        turn_index=turn_index,
        event_type=event_type,  # type: ignore[arg-type]
        role=role,
        harness=harness,
        session_id=session_id,
        ts=ts,
        parse_status=parse_status,  # type: ignore[arg-type]
        payload=payload or {},
    )


def _write_run(
    store: BridgeStore,
    *,
    run_id: str,
    created_at: str,
    updated_at: str,
    status: str = "running",
    next_actor: str | None = "reviewer",
    last_turn_index: int = 0,
    last_event_id: str | None = None,
) -> BridgeRun:
    run = BridgeRun(
        run_id=run_id,
        task=f"Task for {run_id}",
        created_at=created_at,
        updated_at=updated_at,
        status=status,  # type: ignore[arg-type]
        completed_at=None,
        last_turn_index=last_turn_index,
        next_actor=next_actor,
        repair_budget_per_turn=1,
        footer_mode="prompt_injected",
        worktree_cleanup_mode="operator_triggered",
        participants=[
            Participant(role="implementer", harness="codex", model="gpt-5.4"),
            Participant(role="reviewer", harness="claude", model="claude-opus-4-7"),
        ],
        worktree_path=str(store.root),
        worktree_agent_slug="codex-bridge",
        last_event_id=last_event_id,
    )
    registry = SessionRegistry(
        run_id=run_id,
        updated_at=updated_at,
        sessions={
            "implementer": BridgeSession(
                role="implementer",
                harness="codex",
                model="gpt-5.4",
                session_id="sess-impl",
                worktree_agent_slug="codex-bridge",
                worktree_path=str(store.root / "worktrees" / "implementer"),
                branch="codex/implementer",
                session_status="active",
                started_at=created_at,
                last_turn_index=last_turn_index,
                last_completed_at=updated_at if last_turn_index else None,
                harness_options={"model": "gpt-5.4"},
            ),
            "reviewer": BridgeSession(
                role="reviewer",
                harness="claude",
                model="claude-opus-4-7",
                session_id=None,
                worktree_agent_slug=None,
                worktree_path=None,
                branch=None,
                session_status="not_started",
                started_at=None,
                last_turn_index=0,
                last_completed_at=None,
                harness_options={"resume": True},
            ),
        },
    )
    store.save_run(run)
    store.save_sessions(run_id, registry)
    return run


@pytest.fixture
def bridge_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / ".git").mkdir()
    monkeypatch.setenv("ARAGORA_AGENT_BRIDGE_REPO", str(tmp_path))
    return tmp_path


@pytest.fixture
def store(bridge_repo: Path) -> BridgeStore:
    return BridgeStore(bridge_repo)


@pytest.fixture
def handler(bridge_repo: Path) -> AgentBridgeHandler:
    return AgentBridgeHandler(ctx={})


def test_list_runs_paginates_newest_first_and_roundtrips_cursor(
    handler: AgentBridgeHandler,
    store: BridgeStore,
) -> None:
    _write_run(
        store,
        run_id="bridge-oldest",
        created_at="2026-04-21T20:00:00Z",
        updated_at="2026-04-21T20:01:00Z",
    )
    _write_run(
        store,
        run_id="bridge-middle",
        created_at="2026-04-21T20:02:00Z",
        updated_at="2026-04-21T20:03:00Z",
    )
    _write_run(
        store,
        run_id="bridge-newest",
        created_at="2026-04-21T20:04:00Z",
        updated_at="2026-04-21T20:05:00Z",
    )

    first = handler.handle(
        "/api/v1/agent-bridge/runs",
        {"limit": "2"},
        _mock_http_handler(),
    )

    assert first is not None
    assert first.status_code == 200
    first_payload = _parse_json_body(first)
    first_runs = first_payload["runs"]
    assert isinstance(first_runs, list)
    assert [item["run_id"] for item in first_runs] == ["bridge-newest", "bridge-middle"]
    assert first_runs[0]["next_actor"] == "reviewer"
    assert first_runs[0]["last_turn_index"] == 0
    assert first_runs[0]["participants"] == [
        {"role": "implementer", "harness": "codex", "model": "gpt-5.4"},
        {"role": "reviewer", "harness": "claude", "model": "claude-opus-4-7"},
    ]
    assert "active_role" not in first_runs[0]
    assert "turn_count" not in first_runs[0]
    assert isinstance(first_payload["next_cursor"], str)

    second = handler.handle(
        "/api/v1/agent-bridge/runs",
        {"limit": "2", "cursor": first_payload["next_cursor"]},
        _mock_http_handler(),
    )

    assert second is not None
    assert second.status_code == 200
    second_payload = _parse_json_body(second)
    second_runs = second_payload["runs"]
    assert isinstance(second_runs, list)
    assert [item["run_id"] for item in second_runs] == ["bridge-oldest"]
    assert "next_cursor" not in second_payload


def test_list_runs_enforces_max_page_size(
    handler: AgentBridgeHandler,
    store: BridgeStore,
) -> None:
    for index in range(505):
        minute = index // 60
        second = index % 60
        updated_at = f"2026-04-21T23:{minute:02d}:{second:02d}Z"
        _write_run(
            store,
            run_id=f"bridge-{index:03d}",
            created_at="2026-04-21T20:00:00Z",
            updated_at=updated_at,
        )

    result = handler.handle(
        "/api/v1/agent-bridge/runs",
        {"limit": "999"},
        _mock_http_handler(),
    )

    assert result is not None
    payload = _parse_json_body(result)
    runs = payload["runs"]
    assert isinstance(runs, list)
    assert len(runs) == 500
    assert isinstance(payload["next_cursor"], str)


def test_get_run_returns_role_keyed_detail_and_etag(
    handler: AgentBridgeHandler,
    store: BridgeStore,
) -> None:
    _write_run(
        store,
        run_id="bridge-detail",
        created_at="2026-04-21T21:00:00Z",
        updated_at="2026-04-21T21:05:00Z",
        last_turn_index=2,
        last_event_id="bridge-detail:turn:002:footer_ok:2",
    )

    result = handler.handle(
        "/api/v1/agent-bridge/runs/bridge-detail",
        {},
        _mock_http_handler(),
    )

    assert result is not None
    assert result.status_code == 200
    assert result.headers == {"ETag": 'W/"2026-04-21T21:05:00Z"'}

    payload = _parse_json_body(result)
    assert payload["run_id"] == "bridge-detail"
    assert payload["task"] == "Task for bridge-detail"
    assert payload["last_turn_index"] == 2
    assert payload["next_actor"] == "reviewer"
    assert payload["repair_budget_per_turn"] == 1
    assert payload["worktree_cleanup_mode"] == "operator_triggered"
    assert payload["participants"] == [
        {"role": "implementer", "harness": "codex", "model": "gpt-5.4"},
        {"role": "reviewer", "harness": "claude", "model": "claude-opus-4-7"},
    ]
    assert "sessions" not in payload

    roles = payload["roles"]
    assert isinstance(roles, dict)
    assert set(roles) == {"implementer", "reviewer"}
    assert roles["implementer"]["role"] == "implementer"
    assert roles["implementer"]["model"] == "gpt-5.4"
    assert roles["implementer"]["last_turn_index"] == 2
    assert roles["implementer"]["session_id"] == "sess-impl"
    assert roles["implementer"]["started_at"] == "2026-04-21T21:00:00Z"
    assert roles["reviewer"]["session_id"] is None
    assert roles["reviewer"]["started_at"] is None
    assert roles["reviewer"]["session_status"] == "not_started"
    assert roles["reviewer"]["harness_options"] == {"resume": True}


def test_get_events_supports_cursor_stability_and_if_none_match(
    handler: AgentBridgeHandler,
    store: BridgeStore,
) -> None:
    run = _write_run(
        store,
        run_id="bridge-events",
        created_at="2026-04-21T22:00:00Z",
        updated_at="2026-04-21T22:10:00Z",
        last_turn_index=1,
    )
    events = [
        _turn_event(
            run_id=run.run_id,
            turn_index=0,
            seq=0,
            event_type="run_started",
            role="implementer",
            ts="2026-04-21T22:00:00Z",
            payload={"task": run.task},
            harness="broker",
            session_id=None,
        ),
        _turn_event(
            run_id=run.run_id,
            turn_index=1,
            seq=0,
            event_type="turn.started",
            role="implementer",
            ts="2026-04-21T22:01:00Z",
            payload={"prompt": "Review the patch"},
        ),
        _turn_event(
            run_id=run.run_id,
            turn_index=1,
            seq=1,
            event_type="footer_ok",
            role="implementer",
            ts="2026-04-21T22:02:00Z",
            parse_status="ok",
            payload={"footer": BridgeFooter("Done", "reviewer", False, False, [], []).to_dict()},
        ),
    ]
    for event in events:
        store.append_event(run.run_id, event)
    run.last_event_id = events[-1].event_id
    run.updated_at = "2026-04-21T22:02:00Z"
    store.save_run(run)

    first = handler.handle(
        "/api/v1/agent-bridge/runs/bridge-events/events",
        {"limit": "2"},
        _mock_http_handler(),
    )

    assert first is not None
    assert first.status_code == 200
    assert first.headers == {"ETag": f'W/"{events[1].event_id}"'}
    first_payload = _parse_json_body(first)
    first_events = first_payload["events"]
    assert isinstance(first_events, list)
    assert [item["event_id"] for item in first_events] == [events[0].event_id, events[1].event_id]
    assert first_events[0]["schema_version"] == 1
    assert first_events[0]["run_id"] == run.run_id
    assert first_events[0]["event_type"] == "run_started"
    assert first_events[0]["ts"] == "2026-04-21T22:00:00Z"
    assert first_events[0]["harness"] == "broker"
    assert first_events[0]["session_id"] is None
    assert first_events[1]["event_type"] == "turn.started"
    assert isinstance(first_payload["next_cursor"], str)

    second = handler.handle(
        "/api/v1/agent-bridge/runs/bridge-events/events",
        {"limit": "2", "cursor": first_payload["next_cursor"]},
        _mock_http_handler(),
    )

    assert second is not None
    assert second.status_code == 200
    second_payload = _parse_json_body(second)
    second_events = second_payload["events"]
    assert isinstance(second_events, list)
    assert [item["event_id"] for item in second_events] == [events[2].event_id]
    assert second_events[0]["event_type"] == "footer_ok"

    not_modified = handler.handle(
        "/api/v1/agent-bridge/runs/bridge-events/events",
        {"limit": "2"},
        _mock_http_handler(headers={"If-None-Match": first.headers["ETag"]}),
    )

    assert not_modified is not None
    assert not_modified.status_code == 304
    assert not_modified.headers == {"ETag": first.headers["ETag"]}
    assert not not_modified.body


def test_get_transcript_reconstructs_turns_from_events(
    handler: AgentBridgeHandler,
    store: BridgeStore,
) -> None:
    run = _write_run(
        store,
        run_id="bridge-transcript",
        created_at="2026-04-21T23:00:00Z",
        updated_at="2026-04-21T23:06:00Z",
        last_turn_index=2,
    )
    repaired_footer = BridgeFooter(
        summary="Footer repaired",
        next_actor="reviewer",
        needs_human=False,
        done=False,
        artifacts=[],
        tests_run=[],
    )
    turn_two_footer = BridgeFooter(
        summary="Review complete",
        next_actor=None,
        needs_human=False,
        done=True,
        artifacts=[],
        tests_run=[],
    )
    events = [
        _turn_event(
            run_id=run.run_id,
            turn_index=0,
            seq=0,
            event_type="run_started",
            role="implementer",
            ts="2026-04-21T23:00:00Z",
            payload={"task": run.task},
            harness="broker",
            session_id=None,
        ),
        _turn_event(
            run_id=run.run_id,
            turn_index=1,
            seq=0,
            event_type="turn.started",
            role="implementer",
            ts="2026-04-21T23:01:00Z",
            payload={"prompt": "Implement the fix"},
        ),
        _turn_event(
            run_id=run.run_id,
            turn_index=1,
            seq=1,
            event_type="turn.result",
            role="implementer",
            ts="2026-04-21T23:02:00Z",
            parse_status="malformed",
            payload={
                "message_text": _footer_text(
                    "Implemented the handler.",
                    summary="bad",
                    next_actor="not-a-real-role",
                ),
            },
        ),
        _turn_event(
            run_id=run.run_id,
            turn_index=1,
            seq=2,
            event_type="footer_malformed",
            role="implementer",
            ts="2026-04-21T23:02:30Z",
            parse_status="malformed",
            payload={"errors": ["footer_not_final_block"]},
        ),
        _turn_event(
            run_id=run.run_id,
            turn_index=1,
            seq=3,
            event_type="turn.repair_requested",
            role="implementer",
            ts="2026-04-21T23:03:00Z",
            parse_status="malformed",
            payload={"errors": ["footer_not_final_block"]},
        ),
        _turn_event(
            run_id=run.run_id,
            turn_index=1,
            seq=4,
            event_type="turn.completed",
            role="implementer",
            ts="2026-04-21T23:04:00Z",
            parse_status="ok",
            payload={
                "message_text": _footer_text(
                    "",
                    summary="Footer repaired",
                    next_actor="reviewer",
                ),
                "footer": repaired_footer.to_dict(),
            },
        ),
        _turn_event(
            run_id=run.run_id,
            turn_index=1,
            seq=5,
            event_type="footer_ok",
            role="implementer",
            ts="2026-04-21T23:04:10Z",
            parse_status="ok",
            payload={"footer": repaired_footer.to_dict()},
        ),
        _turn_event(
            run_id=run.run_id,
            turn_index=2,
            seq=0,
            event_type="turn.started",
            role="reviewer",
            ts="2026-04-21T23:05:00Z",
            payload={"prompt": "Review the fix"},
            harness="claude",
            session_id="sess-review",
        ),
        _turn_event(
            run_id=run.run_id,
            turn_index=2,
            seq=1,
            event_type="turn.result",
            role="reviewer",
            ts="2026-04-21T23:06:00Z",
            parse_status="ok",
            payload={
                "message_text": _footer_text(
                    "Looks good.",
                    summary="Review complete",
                    next_actor=None,
                    done=True,
                ),
                "footer": turn_two_footer.to_dict(),
            },
            harness="claude",
            session_id="sess-review",
        ),
        _turn_event(
            run_id=run.run_id,
            turn_index=2,
            seq=2,
            event_type="footer_ok",
            role="reviewer",
            ts="2026-04-21T23:06:10Z",
            parse_status="ok",
            payload={"footer": turn_two_footer.to_dict()},
            harness="claude",
            session_id="sess-review",
        ),
    ]
    for event in events:
        store.append_event(run.run_id, event)
    run.last_event_id = events[-1].event_id
    run.updated_at = "2026-04-21T23:06:10Z"
    run.last_turn_index = 2
    store.save_run(run)

    result = handler.handle(
        "/api/v1/agent-bridge/runs/bridge-transcript/transcript",
        {},
        _mock_http_handler(),
    )

    assert result is not None
    assert result.status_code == 200
    payload = _parse_json_body(result)
    turns = payload["turns"]
    assert isinstance(turns, list)
    assert [turn["turn_index"] for turn in turns] == [1, 2]
    assert turns[0]["author_role"] == "implementer"
    assert turns[0]["body_markdown"] == "Implemented the handler."
    assert turns[0]["parse_status"] == "ok"
    assert turns[0]["footer"]["summary"] == "Footer repaired"
    assert turns[1]["author_role"] == "reviewer"
    assert turns[1]["body_markdown"] == "Looks good."
    assert turns[1]["footer"]["done"] is True


def test_unknown_run_returns_404(handler: AgentBridgeHandler) -> None:
    result = handler.handle(
        "/api/v1/agent-bridge/runs/missing-run",
        {},
        _mock_http_handler(),
    )

    assert result is not None
    assert result.status_code == 404
    assert _parse_json_body(result) == {"error": "Bridge run not found"}


def test_get_run_returns_404_when_sessions_are_missing(
    handler: AgentBridgeHandler,
    store: BridgeStore,
) -> None:
    run = _write_run(
        store,
        run_id="bridge-missing-sessions",
        created_at="2026-04-21T20:00:00Z",
        updated_at="2026-04-21T20:01:00Z",
    )
    store.sessions_path(run.run_id).unlink()

    result = handler.handle(
        "/api/v1/agent-bridge/runs/bridge-missing-sessions",
        {},
        _mock_http_handler(),
    )

    assert result is not None
    assert result.status_code == 404
    assert _parse_json_body(result) == {"error": "Bridge run not found"}


def test_malformed_cursor_returns_400(
    handler: AgentBridgeHandler,
    store: BridgeStore,
) -> None:
    _write_run(
        store,
        run_id="bridge-cursor",
        created_at="2026-04-21T20:00:00Z",
        updated_at="2026-04-21T20:01:00Z",
    )

    result = handler.handle(
        "/api/v1/agent-bridge/runs/bridge-cursor/events",
        {"cursor": "%%%not-a-valid-cursor%%%"},
        _mock_http_handler(),
    )

    assert result is not None
    assert result.status_code == 400
    assert _parse_json_body(result) == {"error": "Invalid bridge cursor"}


@pytest.mark.no_auto_auth
def test_rbac_rejects_requests_without_agent_bridge_permission(
    handler: AgentBridgeHandler,
    store: BridgeStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aragora.server.auth import auth_config

    monkeypatch.setattr(auth_config, "enabled", True)
    _write_run(
        store,
        run_id="bridge-rbac",
        created_at="2026-04-21T20:00:00Z",
        updated_at="2026-04-21T20:01:00Z",
    )

    http = _mock_http_handler()
    http._auth_context = AuthorizationContext(
        user_id="viewer-1",
        user_email="viewer@example.com",
        org_id="org-1",
        roles={"viewer"},
        permissions=set(),
    )

    result = handler.handle("/api/v1/agent-bridge/runs", {}, http)

    assert result is not None
    assert result.status_code == 403
