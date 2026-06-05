from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from aragora.swarm.agent_bridge.exceptions import TransportLaunchError
from aragora.swarm.agent_bridge.types import BridgeRun
from aragora.swarm.agent_bridge.types import SessionRegistry
from aragora.swarm.agent_bridge.types import TurnRecord


def _load_script_module():
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "agent_bridge_broker.py"
    spec = importlib.util.spec_from_file_location("agent_bridge_broker_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBroker:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def start_run(self, **_: Any) -> BridgeRun:
        return BridgeRun(
            run_id="bridge_cli",
            task="Review",
            created_at="2026-04-21T20:00:00Z",
            updated_at="2026-04-21T20:00:00Z",
            status="running",
            completed_at=None,
            last_turn_index=0,
            next_actor="reviewer",
            repair_budget_per_turn=1,
            footer_mode="prompt_injected",
            worktree_cleanup_mode="operator_triggered",
            participants=[],
            worktree_path=str(self.repo_root),
            worktree_agent_slug="codex",
        )

    def dispatch_turn(self, **_: Any) -> TurnRecord:
        return TurnRecord(
            event_id="bridge_cli:turn:001:footer_ok:0",
            run_id="bridge_cli",
            turn_index=1,
            event_type="footer_ok",
            role="reviewer",
            harness="codex",
            session_id="thread-1",
            ts="2026-04-21T20:01:00Z",
            parse_status="ok",
            payload={"footer": {"summary": "done"}},
        )

    def load_run(self, run_id: str) -> BridgeRun:
        del run_id
        return self.start_run()

    def load_sessions(self, run_id: str) -> SessionRegistry:
        del run_id
        return SessionRegistry(run_id="bridge_cli", updated_at="2026-04-21T20:00:00Z", sessions={})

    def load_events(self, run_id: str) -> list[TurnRecord]:
        del run_id
        return []

    def list_runs(self, *, status: str | None = None) -> list[BridgeRun]:
        del status
        return [self.start_run()]


def test_cli_bad_usage_returns_1() -> None:
    module = _load_script_module()

    assert module.main(["dispatch-turn"]) == 1


def test_cli_run_not_found_returns_2() -> None:
    module = _load_script_module()

    class MissingRunBroker(FakeBroker):
        def dispatch_turn(self, **_: Any) -> TurnRecord:
            raise KeyError("missing run")

    assert (
        module.main(
            ["dispatch-turn", "--run-id", "bridge_cli", "--role", "reviewer", "--prompt", "Review"],
            broker_factory=MissingRunBroker,
        )
        == 2
    )


def test_cli_transport_failure_returns_3() -> None:
    module = _load_script_module()

    class BrokenBroker(FakeBroker):
        def dispatch_turn(self, **_: Any) -> TurnRecord:
            raise TransportLaunchError("transport failed")

    assert (
        module.main(
            ["dispatch-turn", "--run-id", "bridge_cli", "--role", "reviewer", "--prompt", "Review"],
            broker_factory=BrokenBroker,
        )
        == 3
    )


def test_cli_footer_repair_exhaustion_returns_4() -> None:
    module = _load_script_module()

    class ExhaustedBroker(FakeBroker):
        def dispatch_turn(self, **_: Any) -> TurnRecord:
            return TurnRecord(
                event_id="bridge_cli:turn:001:footer_malformed:0",
                run_id="bridge_cli",
                turn_index=1,
                event_type="footer_malformed",
                role="reviewer",
                harness="codex",
                session_id="thread-1",
                ts="2026-04-21T20:01:00Z",
                parse_status="malformed",
                payload={"repair_exhausted": True},
            )

    assert (
        module.main(
            ["dispatch-turn", "--run-id", "bridge_cli", "--role", "reviewer", "--prompt", "Review"],
            broker_factory=ExhaustedBroker,
        )
        == 4
    )


def test_cli_persistence_failure_returns_5() -> None:
    module = _load_script_module()

    class PersistenceBroker(FakeBroker):
        def show_run(self, **_: Any) -> None:
            raise OSError("disk error")

        def load_run(self, run_id: str) -> BridgeRun:
            del run_id
            raise OSError("disk error")

    assert (
        module.main(
            ["show-run", "--run-id", "bridge_cli"],
            broker_factory=PersistenceBroker,
        )
        == 5
    )


def test_cli_healthcheck_failure_returns_6() -> None:
    module = _load_script_module()

    class BrokenTransport:
        def healthcheck(self) -> bool:
            return False

    assert (
        module.main(
            ["start-run", "--task", "Review", "--actor", "reviewer:codex:gpt-5.4"],
            broker_factory=FakeBroker,
            transport_factory=lambda *args, **kwargs: BrokenTransport(),
        )
        == 6
    )


def test_parse_actors_sets_droid_auto_high() -> None:
    module = _load_script_module()

    registry = module._parse_actors(["reviewer:droid:claude-opus-4-7"])
    session = registry.sessions["reviewer"]

    assert session.harness == "droid"
    assert session.model == "claude-opus-4-7"
    assert session.harness_options == {"auto": "high"}


def test_cli_success_returns_0_and_emits_json(capsys) -> None:
    module = _load_script_module()

    class HealthyTransport:
        def healthcheck(self) -> bool:
            return True

    code = module.main(
        [
            "start-run",
            "--task",
            "Review",
            "--actor",
            "reviewer:codex:gpt-5.4",
            "--json",
        ],
        broker_factory=FakeBroker,
        transport_factory=lambda *args, **kwargs: HealthyTransport(),
    )

    captured = capsys.readouterr()
    assert code == 0
    assert '"run_id": "bridge_cli"' in captured.out


def test_cli_list_runs_json_emits_full_runs(capsys) -> None:
    module = _load_script_module()

    code = module.main(["list-runs", "--json"], broker_factory=FakeBroker)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["runs"][0]["run_id"] == "bridge_cli"
    assert payload["runs"][0]["task"] == "Review"
    assert "top_runs" not in payload


def test_cli_list_runs_summary_only_omits_full_runs(capsys) -> None:
    module = _load_script_module()

    class MultiRunBroker(FakeBroker):
        def list_runs(self, *, status: str | None = None) -> list[BridgeRun]:
            del status
            running = self.start_run()
            completed = BridgeRun(
                run_id="bridge_done",
                task="Large completed transcript",
                created_at="2026-04-21T19:00:00Z",
                updated_at="2026-04-21T19:30:00Z",
                status="completed",
                completed_at="2026-04-21T19:30:00Z",
                last_turn_index=3,
                next_actor=None,
                repair_budget_per_turn=1,
                footer_mode="prompt_injected",
                worktree_cleanup_mode="operator_triggered",
                participants=[],
                worktree_path=str(self.repo_root),
                worktree_agent_slug="codex",
            )
            return [running, completed]

    code = module.main(["list-runs", "--json", "--summary-only"], broker_factory=MultiRunBroker)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["details_omitted"] is True
    assert payload["run_count"] == 2
    assert payload["emitted_count"] == 2
    assert payload["runs_omitted"] == 0
    assert payload["status_counts"] == {"completed": 1, "running": 1}
    assert payload["top_runs"][0] == {
        "run_id": "bridge_cli",
        "status": "running",
        "updated_at": "2026-04-21T20:00:00Z",
        "next_actor": "reviewer",
        "last_turn_index": 0,
        "participant_count": 0,
        "worktree_agent_slug": "codex",
    }
    assert "runs" not in payload
