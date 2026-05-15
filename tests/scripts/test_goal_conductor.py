"""Tests for scripts/goal_conductor.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


@pytest.fixture(autouse=True)
def _setup_path():
    sys.path.insert(0, str(SCRIPTS_DIR))
    yield
    sys.path.remove(str(SCRIPTS_DIR))


class FakeRunner:
    def __init__(
        self,
        mod,
        *,
        open_prs: list[dict] | None = None,
        dirty: bool = False,
        merge_packet: dict | None = None,
        pr_query_returncode: int = 0,
        merge_packet_returncode: int = 0,
    ):
        self.mod = mod
        self.open_prs = open_prs or []
        self.dirty = dirty
        self.merge_packet = merge_packet or {"packets": []}
        self.pr_query_returncode = pr_query_returncode
        self.merge_packet_returncode = merge_packet_returncode
        self.calls: list[list[str]] = []
        self.executed: list[list[str]] = []

    def run(self, args: list[str], *, timeout: int = 60):
        self.calls.append(args)
        command = " ".join(args)
        if args[:3] == ["git", "status", "--short"]:
            status = "## main...origin/main\n"
            if self.dirty:
                status += " M scripts/example.py\n"
            return self.mod.CommandResult(args=args, returncode=0, stdout=status)
        if args[:3] == ["git", "rev-parse", "--short"]:
            return self.mod.CommandResult(args=args, returncode=0, stdout="abcdef1\n")
        if args[:3] == ["gh", "pr", "list"]:
            return self.mod.CommandResult(
                args=args,
                returncode=self.pr_query_returncode,
                stdout=json.dumps(self.open_prs),
            )
        if "review-queue merge-packet --json" in command:
            return self.mod.CommandResult(
                args=args,
                returncode=self.merge_packet_returncode,
                stdout=json.dumps(self.merge_packet),
            )
        if "publisher_freshness_check.py" in command:
            return self.mod.CommandResult(
                args=args,
                returncode=0,
                stdout=json.dumps({"verdict": "ready", "summary": "ready"}),
            )
        if "agent_bridge.py --json operator-snapshot" in command:
            return self.mod.CommandResult(
                args=args,
                returncode=0,
                stdout=json.dumps({"sessions": 0, "lanes": 0}),
            )
        if "agent_bridge.py --json sessions" in command:
            return self.mod.CommandResult(
                args=args,
                returncode=0,
                stdout=json.dumps([{"name": "existing-lane"}]),
            )
        if "review-queue health --json" in command:
            return self.mod.CommandResult(
                args=args,
                returncode=0,
                stdout=json.dumps({"overall_status": "fresh"}),
            )
        self.executed.append(args)
        return self.mod.CommandResult(args=args, returncode=0, stdout="")


def _mission_dict(tmp_path: Path) -> dict:
    return {
        "name": "proof-loop-goal",
        "objective": "Advance the proof-loop operating baseline.",
        "stop_condition": "Stop at queue cap or Tier 4 settlement.",
        "checkpoints": ["snapshot", "assign bounded lanes", "write handoff"],
        "external_references": [
            "https://developers.openai.com/codex/use-cases/follow-goals",
            "https://github.com/Dicklesworthstone/mcp_agent_mail",
        ],
        "output_dir": str(tmp_path / "goal-output"),
        "limits": {
            "queue_cap": 2,
            "max_implementation_lanes": 1,
            "max_review_lanes": 1,
        },
        "collect_merge_packets": True,
        "max_merge_packets": 5,
        "lanes": [
            {
                "id": "impl",
                "agent": "codex",
                "mode": "implementation",
                "goal": "Make one bounded code change.",
                "prompt": "Implement only the assigned file.",
            },
            {
                "id": "panel",
                "mode": "panel",
                "goal": "Review the current gate.",
                "prompt": "Adversarially review the merge gate.",
                "agents_spec": "heterogeneous",
            },
        ],
    }


def test_load_mission_preserves_follow_goal_fields(tmp_path: Path) -> None:
    import goal_conductor as mod

    mission_path = tmp_path / "mission.yaml"
    mission_path.write_text(
        """
name: proof-loop-goal
objective: Advance proof-loop reliability.
stop_condition: Stop at hard gates.
checkpoints:
  - Snapshot live truth
  - Assign lanes
external_references:
  - https://developers.openai.com/codex/use-cases/follow-goals
limits:
  queue_cap: 3
lanes:
  - id: impl
    agent: codex
    goal: Patch one bounded file.
    prompt: Patch it.
""",
        encoding="utf-8",
    )

    mission = mod.load_mission(mission_path)

    assert mission.name == "proof-loop-goal"
    assert mission.objective == "Advance proof-loop reliability."
    assert mission.stop_condition == "Stop at hard gates."
    assert mission.checkpoints == ["Snapshot live truth", "Assign lanes"]
    assert mission.external_references == [
        "https://developers.openai.com/codex/use-cases/follow-goals"
    ]
    assert mission.limits.queue_cap == 3
    assert mission.lanes[0].lane_id == "impl"


def test_run_once_blocks_mutating_lane_at_queue_cap_but_allows_panel(tmp_path: Path) -> None:
    import goal_conductor as mod

    mission = mod.Mission.from_dict(_mission_dict(tmp_path))
    open_prs = [
        {"number": 1, "title": "ready", "isDraft": False, "mergeStateStatus": "CLEAN"},
        {"number": 2, "title": "draft", "isDraft": True, "mergeStateStatus": "BLOCKED"},
    ]
    runner = FakeRunner(mod, open_prs=open_prs)
    conductor = mod.GoalConductor(
        mission=mission,
        repo_root=tmp_path,
        execute=False,
        runner=runner,
    )

    result = conductor.run_once()

    assert result.hard_gates == ["open PR queue at/above cap (2/2)"]
    assert result.snapshot["merge_packets"] == {"packets": []}
    assert any("merge-packet" in " ".join(call) for call in runner.calls)
    assert [decision.action for decision in result.decisions] == ["blocked", "dry_run"]
    assert "queue cap reached" in result.decisions[0].reason
    assert result.decisions[1].commands[0][:2] == ["python3", "scripts/multi_agent_dialog.py"]
    assert result.jsonl_path.exists()
    assert result.markdown_path.exists()
    assert "Initial" not in result.markdown_path.read_text(encoding="utf-8")
    assert (
        "Objective: Advance the proof-loop operating baseline."
        in result.markdown_path.read_text(encoding="utf-8")
    )
    assert "https://developers.openai.com/codex/use-cases/follow-goals" in (
        result.markdown_path.read_text(encoding="utf-8")
    )


def test_execute_reuses_existing_agent_lane_and_sends_prompt(tmp_path: Path) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    payload["lanes"] = [
        {
            "id": "existing-lane",
            "agent": "claude",
            "mode": "implementation",
            "goal": "Continue an existing lane.",
            "prompt": "Continue safely.",
            "source": "#123",
            "next_action": "open draft PR",
        }
    ]
    mission = mod.Mission.from_dict(payload)
    runner = FakeRunner(mod, open_prs=[])
    conductor = mod.GoalConductor(
        mission=mission,
        repo_root=tmp_path,
        execute=True,
        runner=runner,
    )

    result = conductor.run_once()

    assert result.decisions[0].action == "execute"
    assert not any("launch" in call for command in runner.executed for call in command)
    send_commands = [command for command in runner.executed if "send" in command]
    assert len(send_commands) == 1
    assert send_commands[0][:3] == ["python3", "scripts/agent_bridge.py", "send"]
    assert "--lane" in send_commands[0]
    assert "existing-lane" in send_commands[0]


def test_loop_stops_after_first_hard_gate(tmp_path: Path) -> None:
    import goal_conductor as mod

    mission = mod.Mission.from_dict(_mission_dict(tmp_path))
    open_prs = [
        {"number": 1, "title": "one", "isDraft": False},
        {"number": 2, "title": "two", "isDraft": True},
    ]
    runner = FakeRunner(mod, open_prs=open_prs)
    conductor = mod.GoalConductor(
        mission=mission,
        repo_root=tmp_path,
        execute=False,
        runner=runner,
    )

    results = conductor.run_loop(max_cycles=3, interval_seconds=0)

    assert len(results) == 1
    assert results[0].hard_gates == ["open PR queue at/above cap (2/2)"]


def test_execute_blocks_all_lanes_when_root_is_dirty(tmp_path: Path) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    mission = mod.Mission.from_dict(payload)
    runner = FakeRunner(mod, open_prs=[], dirty=True)
    conductor = mod.GoalConductor(
        mission=mission,
        repo_root=tmp_path,
        execute=True,
        runner=runner,
    )

    result = conductor.run_once()

    assert result.hard_gates == ["root checkout is dirty"]
    assert [decision.action for decision in result.decisions] == ["blocked", "blocked"]
    assert all("fatal hard gate" in decision.reason for decision in result.decisions)
    assert runner.executed == []


def test_execute_blocks_all_lanes_when_human_settlement_gate_present(tmp_path: Path) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    mission = mod.Mission.from_dict(payload)
    open_prs = [{"number": 7156, "title": "tier 4 gate", "isDraft": False}]
    merge_packet = {
        "entries": [
            {
                "pr_number": 7156,
                "tier": 4,
                "tier_name": "tier_4_preapproval_required",
                "requires_human_risk_settlement": True,
            }
        ]
    }
    runner = FakeRunner(mod, open_prs=open_prs, merge_packet=merge_packet)
    conductor = mod.GoalConductor(
        mission=mission,
        repo_root=tmp_path,
        execute=True,
        runner=runner,
    )

    result = conductor.run_once()

    assert result.hard_gates == [
        "human/non-author settlement gate present: #7156 tier_4_preapproval_required"
    ]
    assert [decision.action for decision in result.decisions] == ["blocked", "blocked"]
    assert all("fatal hard gate" in decision.reason for decision in result.decisions)
    assert runner.executed == []


def test_execute_blocks_all_lanes_when_pr_query_fails(tmp_path: Path) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    mission = mod.Mission.from_dict(payload)
    runner = FakeRunner(mod, pr_query_returncode=1)
    conductor = mod.GoalConductor(
        mission=mission,
        repo_root=tmp_path,
        execute=True,
        runner=runner,
    )

    result = conductor.run_once()

    assert "open PR query failed: rc=1" in result.hard_gates
    assert [decision.action for decision in result.decisions] == ["blocked", "blocked"]
    assert runner.executed == []


def test_execute_blocks_all_lanes_when_merge_packet_fails(tmp_path: Path) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    mission = mod.Mission.from_dict(payload)
    open_prs = [{"number": 7156, "title": "needs packet", "isDraft": False}]
    runner = FakeRunner(mod, open_prs=open_prs, merge_packet_returncode=1)
    conductor = mod.GoalConductor(
        mission=mission,
        repo_root=tmp_path,
        execute=True,
        runner=runner,
    )

    result = conductor.run_once()

    assert "merge-packet query failed for 7156" in result.hard_gates
    assert [decision.action for decision in result.decisions] == ["blocked", "blocked"]
    assert runner.executed == []


def test_discover_loop_surfaces_reports_existing_tools(tmp_path: Path) -> None:
    import goal_conductor as mod

    tool = tmp_path / "scripts/agent_bridge.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("# bridge\n", encoding="utf-8")

    surfaces = mod.discover_loop_surfaces(tmp_path)

    assert surfaces["agent_bridge"]["exists"] is True
    assert surfaces["boss_loop"]["exists"] is False
