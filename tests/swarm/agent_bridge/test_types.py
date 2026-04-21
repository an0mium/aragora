from __future__ import annotations

from aragora.swarm.agent_bridge.types import BridgeFooter
from aragora.swarm.agent_bridge.types import BridgeRun
from aragora.swarm.agent_bridge.types import BridgeSession
from aragora.swarm.agent_bridge.types import ParsedTurn
from aragora.swarm.agent_bridge.types import Participant
from aragora.swarm.agent_bridge.types import SCHEMA_VERSION
from aragora.swarm.agent_bridge.types import SessionRegistry
from aragora.swarm.agent_bridge.types import TurnRecord


def test_dataclass_roundtrips_include_schema_version() -> None:
    run = BridgeRun(
        run_id="bridge_123",
        task="Review the plan",
        created_at="2026-04-21T20:00:00Z",
        updated_at="2026-04-21T20:00:00Z",
        status="running",
        completed_at=None,
        last_turn_index=2,
        next_actor="reviewer",
        repair_budget_per_turn=1,
        footer_mode="prompt_injected",
        worktree_cleanup_mode="operator_triggered",
        participants=[
            Participant(role="reviewer", harness="claude", model="claude-opus-4-7"),
            Participant(role="implementer", harness="codex", model="gpt-5.4"),
        ],
        worktree_path="/tmp/run",
        worktree_agent_slug="codex",
        last_event_id="bridge_123:turn:002:footer_ok:1",
    )
    registry = SessionRegistry(
        run_id="bridge_123",
        updated_at="2026-04-21T20:05:00Z",
        sessions={
            "reviewer": BridgeSession(
                role="reviewer",
                harness="claude",
                model="claude-opus-4-7",
                session_id="review-session",
                worktree_agent_slug="bridge-reviewer",
                worktree_path="/tmp/reviewer",
                branch="codex/reviewer",
                session_status="active",
                started_at="2026-04-21T20:01:00Z",
                last_turn_index=1,
                last_completed_at="2026-04-21T20:05:00Z",
                harness_options={"--verbose": True},
            )
        },
    )
    turn = TurnRecord(
        event_id="bridge_123:turn:002:footer_ok:1",
        run_id="bridge_123",
        turn_index=2,
        event_type="footer_ok",
        role="reviewer",
        harness="claude",
        session_id="review-session",
        parse_status="ok",
        ts="2026-04-21T20:05:00Z",
        payload={"footer": {"summary": "done"}},
    )
    parsed = ParsedTurn(
        footer=BridgeFooter(
            summary="done",
            next_actor="implementer",
            needs_human=False,
            done=False,
            artifacts=[],
            tests_run=[],
        ),
        body_without_footer="done",
        parse_status="ok",
    )

    assert BridgeRun.from_dict(run.to_dict()) == run
    assert SessionRegistry.from_dict(registry.to_dict()) == registry
    assert TurnRecord.from_dict(turn.to_dict()) == turn
    assert run.to_dict()["schema_version"] == SCHEMA_VERSION
    assert registry.to_dict()["schema_version"] == SCHEMA_VERSION
    assert registry.to_dict()["sessions"]["reviewer"]["model"] == "claude-opus-4-7"
    assert turn.to_dict()["schema_version"] == SCHEMA_VERSION
    assert parsed.to_dict()["parse_status"] == "ok"
