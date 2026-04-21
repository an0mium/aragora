from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from aragora.swarm.agent_bridge.broker import AgentBridgeBroker
from aragora.swarm.agent_bridge.footer import extract_footer
from aragora.swarm.agent_bridge.harnesses.base import TransportResult
from aragora.swarm.agent_bridge.store import BridgeStore
from aragora.swarm.agent_bridge.types import BridgeSession


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


def _sessions(tmp_path: Path) -> dict[str, BridgeSession]:
    return {
        "reviewer": BridgeSession(
            role="reviewer",
            harness="codex",
            model="gpt-5.4",
            session_id=None,
            worktree_agent_slug="bridge-reviewer",
            worktree_path=str(tmp_path),
            branch="codex/reviewer",
            session_status="not_started",
            started_at=None,
            last_turn_index=0,
            last_completed_at=None,
        ),
        "implementer": BridgeSession(
            role="implementer",
            harness="claude",
            model="claude-opus-4-7",
            session_id=None,
            worktree_agent_slug="bridge-implementer",
            worktree_path=str(tmp_path),
            branch="codex/implementer",
            session_status="not_started",
            started_at=None,
            last_turn_index=0,
            last_completed_at=None,
        ),
    }


def _transport_factory(queues: dict[str, list[TransportResult]]):
    def factory(
        harness_name: str,
        *,
        cwd: Path,
        model: str | None,
        harness_options: dict[str, Any] | None,
    ) -> FakeTransport:
        del harness_name, cwd, model
        role = (
            "reviewer" if harness_options is None else str(harness_options.get("role", "reviewer"))
        )
        return FakeTransport(role, queues)

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
    sessions = _sessions(tmp_path)
    sessions["reviewer"].harness_options["role"] = "reviewer"
    sessions["implementer"].harness_options["role"] = "implementer"
    broker = AgentBridgeBroker(
        tmp_path,
        store=BridgeStore(tmp_path),
        transport_factory=_transport_factory(queues),
    )
    run = broker.start_run(
        task="Review it",
        sessions=sessions,
        next_actor="reviewer",
        run_id="bridge_broker_role",
        worktree_path=str(tmp_path),
        worktree_agent_slug="codex",
    )

    broker.dispatch_turn(run_id=run.run_id, role="reviewer", prompt="Review it")

    persisted_run = broker.load_run(run.run_id)
    persisted_sessions = broker.load_sessions(run.run_id)
    event_types = [event.event_type for event in broker.load_events(run.run_id)]

    assert persisted_run.next_actor == "implementer"
    assert persisted_run.last_turn_index == 1
    assert persisted_run.status == "running"
    assert persisted_sessions.sessions["reviewer"].session_id == "review-session"
    assert event_types == ["run_started", "turn.started", "turn.result", "footer_ok"]


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
    sessions = _sessions(tmp_path)
    sessions["reviewer"].harness_options["role"] = "reviewer"
    sessions["implementer"].harness_options["role"] = "implementer"
    broker = AgentBridgeBroker(
        tmp_path,
        store=BridgeStore(tmp_path),
        transport_factory=_transport_factory(queues),
    )
    run = broker.start_run(
        task="Review it",
        sessions=sessions,
        next_actor="reviewer",
        run_id="bridge_broker_repair",
        worktree_path=str(tmp_path),
        worktree_agent_slug="codex",
    )

    broker.dispatch_turn(run_id=run.run_id, role="reviewer", prompt="Review it")

    persisted_run = broker.load_run(run.run_id)
    event_types = [event.event_type for event in broker.load_events(run.run_id)]
    transcript = (
        tmp_path
        / ".aragora"
        / "agent_bridge"
        / "runs"
        / run.run_id
        / "turns"
        / "001-codex-reviewer.md"
    ).read_text(encoding="utf-8")

    assert persisted_run.next_actor == "implementer"
    assert persisted_run.status == "running"
    assert "## Repair Attempt 1" in transcript
    assert event_types == [
        "run_started",
        "turn.started",
        "turn.result",
        "footer_missing",
        "turn.repair_requested",
        "turn.completed",
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
    sessions = _sessions(tmp_path)
    sessions["reviewer"].harness_options["role"] = "reviewer"
    sessions["implementer"].harness_options["role"] = "implementer"
    broker = AgentBridgeBroker(
        tmp_path,
        store=BridgeStore(tmp_path),
        transport_factory=_transport_factory(queues),
    )
    run = broker.start_run(
        task="Review it",
        sessions=sessions,
        next_actor="reviewer",
        run_id="bridge_broker_surface",
        worktree_path=str(tmp_path),
        worktree_agent_slug="codex",
    )

    result = broker.dispatch_turn(run_id=run.run_id, role="reviewer", prompt="Review it")

    persisted_run = broker.load_run(run.run_id)
    event_types = [event.event_type for event in broker.load_events(run.run_id)]

    assert result.event_type == "footer_malformed"
    assert persisted_run.status == "awaiting_human"
    assert persisted_run.next_actor == "reviewer"
    assert "run_failed" not in event_types
    assert event_types == [
        "run_started",
        "turn.started",
        "turn.result",
        "footer_missing",
        "turn.repair_requested",
        "turn.completed",
        "footer_malformed",
    ]
