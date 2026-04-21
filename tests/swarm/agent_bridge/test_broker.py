from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from aragora.swarm.agent_bridge.broker import AgentBridgeBroker
from aragora.swarm.agent_bridge.footer import extract_footer
from aragora.swarm.agent_bridge.store import BridgeStore
from aragora.swarm.agent_bridge.types import BridgeSession
from aragora.swarm.agent_bridge.types import SessionRegistry
from aragora.swarm.agent_bridge.harnesses.base import TransportResult


def _make_transport_result(
    *,
    session_id: str,
    message_text: str,
    allowed_roles: set[str],
    command: list[str] | None = None,
) -> TransportResult:
    return TransportResult(
        session_id=session_id,
        command=command or ["fake"],
        exit_code=0,
        raw_stdout=message_text,
        raw_stderr="",
        message_text=message_text,
        parsed_turn=extract_footer(message_text, allowed_roles=allowed_roles),
        usage={},
    )


class FakeTransport:
    def __init__(self, role: str, queues: dict[str, list[TransportResult]]) -> None:
        self.role = role
        self.queues = queues

    def launch(self, prompt: str, *, allowed_roles: set[str]) -> TransportResult:
        del prompt, allowed_roles
        return self.queues[self.role].pop(0)

    def resume(self, session_id: str, prompt: str, *, allowed_roles: set[str]) -> TransportResult:
        del session_id, prompt, allowed_roles
        return self.queues[self.role].pop(0)


def _registry(tmp_path: Path) -> SessionRegistry:
    return SessionRegistry(
        roles={
            "reviewer": BridgeSession(
                harness="codex",
                session_id=None,
                created_at="2026-04-21T20:00:00Z",
                last_turn_at=None,
                harness_options={"role": "reviewer", "worktree_path": str(tmp_path)},
            ),
            "implementer": BridgeSession(
                harness="claude",
                session_id=None,
                created_at="2026-04-21T20:00:00Z",
                last_turn_at=None,
                harness_options={"role": "implementer", "worktree_path": str(tmp_path)},
            ),
        }
    )


def _transport_factory(queues: dict[str, list[TransportResult]]):
    def factory(
        harness_name: str,
        *,
        cwd: Path,
        model: str | None,
        harness_options: dict[str, Any] | None,
    ) -> FakeTransport:
        del harness_name, cwd, model
        assert harness_options is not None
        return FakeTransport(str(harness_options["role"]), queues)

    return factory


def test_broker_dispatches_by_role_and_advances_baton(tmp_path: Path) -> None:
    queues = defaultdict(
        list,
        {
            "reviewer": [
                _make_transport_result(
                    session_id="review-session",
                    message_text=(
                        "Reviewed.\n\n"
                        "---BRIDGE-FOOTER---\n"
                        "summary: Reviewed\n"
                        "next_actor: implementer\n"
                        "needs_human: false\n"
                        "done: false\n"
                        "artifacts: []\n"
                        "tests_run: []\n"
                        "---BRIDGE-FOOTER-END---"
                    ),
                    allowed_roles={"reviewer", "implementer"},
                )
            ]
        },
    )
    broker = AgentBridgeBroker(
        tmp_path,
        store=BridgeStore(tmp_path),
        transport_factory=_transport_factory(queues),
    )
    run = broker.start_run(
        roles=_registry(tmp_path).roles,
        active_role="reviewer",
        run_id="bridge_broker_role",
        worktree_path=str(tmp_path),
        worktree_agent_slug="codex",
    )

    broker.dispatch_turn(run_id=run.run_id, role="reviewer", prompt="Review it")

    persisted_run = broker.load_run(run.run_id)
    persisted_sessions = broker.load_sessions(run.run_id)
    event_types = [event.type for event in broker.load_events(run.run_id)]

    assert persisted_run.active_role == "implementer"
    assert persisted_run.turn_count == 1
    assert persisted_run.status == "running"
    assert persisted_sessions.roles["reviewer"].session_id == "review-session"
    assert event_types == ["run_started", "turn_started", "turn_completed", "footer_ok"]


def test_broker_uses_data_driven_repair_routing(tmp_path: Path) -> None:
    queues = defaultdict(
        list,
        {
            "reviewer": [
                _make_transport_result(
                    session_id="review-session",
                    message_text="Missing footer response",
                    allowed_roles={"reviewer", "implementer"},
                ),
                _make_transport_result(
                    session_id="review-session",
                    message_text=(
                        "---BRIDGE-FOOTER---\n"
                        "summary: Repaired footer\n"
                        "next_actor: implementer\n"
                        "needs_human: false\n"
                        "done: false\n"
                        "artifacts: []\n"
                        "tests_run: []\n"
                        "---BRIDGE-FOOTER-END---"
                    ),
                    allowed_roles={"reviewer", "implementer"},
                ),
            ]
        },
    )
    broker = AgentBridgeBroker(
        tmp_path,
        store=BridgeStore(tmp_path),
        transport_factory=_transport_factory(queues),
    )
    run = broker.start_run(
        roles=_registry(tmp_path).roles,
        active_role="reviewer",
        run_id="bridge_broker_repair",
        worktree_path=str(tmp_path),
        worktree_agent_slug="codex",
    )

    broker.dispatch_turn(run_id=run.run_id, role="reviewer", prompt="Review it")

    persisted_run = broker.load_run(run.run_id)
    event_types = [event.type for event in broker.load_events(run.run_id)]

    assert persisted_run.active_role == "implementer"
    assert persisted_run.status == "running"
    assert event_types == [
        "run_started",
        "turn_started",
        "turn_completed",
        "footer_missing",
        "footer_repair_requested",
        "turn_completed",
        "footer_ok",
    ]


def test_broker_surfaces_for_human_after_repair_exhaustion(tmp_path: Path) -> None:
    queues = defaultdict(
        list,
        {
            "reviewer": [
                _make_transport_result(
                    session_id="review-session",
                    message_text="Missing footer response",
                    allowed_roles={"reviewer", "implementer"},
                ),
                _make_transport_result(
                    session_id="review-session",
                    message_text=(
                        "---BRIDGE-FOOTER---\n"
                        "summary: Broken\n"
                        "next_actor: qa\n"
                        "needs_human: false\n"
                        "done: false\n"
                        "artifacts: []\n"
                        "tests_run: []\n"
                        "---BRIDGE-FOOTER-END---"
                    ),
                    allowed_roles={"reviewer", "implementer"},
                ),
            ]
        },
    )
    broker = AgentBridgeBroker(
        tmp_path,
        store=BridgeStore(tmp_path),
        transport_factory=_transport_factory(queues),
    )
    run = broker.start_run(
        roles=_registry(tmp_path).roles,
        active_role="reviewer",
        run_id="bridge_broker_surface",
        worktree_path=str(tmp_path),
        worktree_agent_slug="codex",
    )

    result = broker.dispatch_turn(run_id=run.run_id, role="reviewer", prompt="Review it")

    persisted_run = broker.load_run(run.run_id)
    event_types = [event.type for event in broker.load_events(run.run_id)]

    assert result.type == "footer_malformed"
    assert persisted_run.status == "awaiting_human"
    assert persisted_run.active_role == "reviewer"
    assert "run_failed" not in event_types
    assert event_types == [
        "run_started",
        "turn_started",
        "turn_completed",
        "footer_missing",
        "footer_repair_requested",
        "turn_completed",
        "footer_malformed",
    ]
