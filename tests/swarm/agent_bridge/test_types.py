from __future__ import annotations

from aragora.swarm.agent_bridge.types import BridgeFooter
from aragora.swarm.agent_bridge.types import BridgeRun
from aragora.swarm.agent_bridge.types import BridgeSession
from aragora.swarm.agent_bridge.types import ParsedTurn
from aragora.swarm.agent_bridge.types import SCHEMA_VERSION
from aragora.swarm.agent_bridge.types import SessionRegistry
from aragora.swarm.agent_bridge.types import TurnRecord


def test_dataclass_roundtrips_include_schema_version() -> None:
    run = BridgeRun(
        run_id="bridge_123",
        created_at="2026-04-21T20:00:00Z",
        updated_at="2026-04-21T20:00:00Z",
        status="running",
        active_role="reviewer",
        footer_mode="prompt_injected",
        participants=["reviewer", "implementer"],
        worktree_path="/tmp/run",
        worktree_agent_slug="codex",
        turn_count=2,
        last_event_id="bridge_123:turn:002:footer_ok:1",
    )
    registry = SessionRegistry(
        roles={
            "reviewer": BridgeSession(
                harness="claude",
                session_id="review-session",
                created_at="2026-04-21T20:00:00Z",
                last_turn_at="2026-04-21T20:05:00Z",
                harness_options={"--verbose": True},
            )
        }
    )
    turn = TurnRecord(
        event_id="bridge_123:turn:002:footer_ok:1",
        turn_index=2,
        type="footer_ok",
        role="reviewer",
        parse_status="ok",
        at="2026-04-21T20:05:00Z",
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
    assert turn.to_dict()["schema_version"] == SCHEMA_VERSION
    assert parsed.to_dict()["parse_status"] == "ok"
