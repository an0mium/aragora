"""Tests for scripts/goal_conductor.py."""

from __future__ import annotations

import json
import subprocess
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
        settle_payload: dict | None = None,
        settle_returncode: int = 0,
        merge_returncode: int = 0,
        bridge_snapshot: dict | None = None,
    ):
        self.mod = mod
        self.open_prs = open_prs or []
        self.dirty = dirty
        self.merge_packet = merge_packet or {"packets": []}
        self.pr_query_returncode = pr_query_returncode
        self.merge_packet_returncode = merge_packet_returncode
        self.settle_payload = settle_payload or {"blockers": [], "head_sha": ""}
        self.settle_returncode = settle_returncode
        self.merge_returncode = merge_returncode
        self.bridge_snapshot = bridge_snapshot or {
            "health": {"ok": True, "issues": []},
            "recent_blockers": [],
            "summary": {"active_lanes": 0, "fresh_agent_heartbeats": 0},
        }
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
                stdout=json.dumps(self.bridge_snapshot),
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
        if args[:3] == ["python3", "scripts/settle_one_pr.py", "--pr"]:
            return self.mod.CommandResult(
                args=args,
                returncode=self.settle_returncode,
                stdout=json.dumps(self.settle_payload),
            )
        if args[:3] == ["gh", "pr", "merge"]:
            self.executed.append(args)
            return self.mod.CommandResult(args=args, returncode=self.merge_returncode, stdout="")
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
                "task_id": "mission-impl",
                "claimed_paths": ["scripts/example.py"],
                "tests": ["python3 -m pytest tests/scripts/test_example.py -q"],
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


def test_main_validate_json_suppresses_flush_time_broken_pipe_without_closing_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import goal_conductor as mod

    mission_path = tmp_path / "mission.yaml"
    mission_path.write_text(
        """
name: pipe-smoke
objective: Verify sampled JSON output.
lanes:
  - id: panel
    mode: panel
    goal: Review sampled output.
    prompt: Review only.
""",
        encoding="utf-8",
    )

    class FlushBrokenStdout:
        def __init__(self) -> None:
            self.writes: list[str] = []
            self.closed = False

        def write(self, text: str) -> int:
            self.writes.append(text)
            return len(text)

        def flush(self) -> None:
            raise BrokenPipeError("downstream closed")

        def close(self) -> None:
            self.closed = True

    stream = FlushBrokenStdout()
    monkeypatch.setattr(mod.sys, "stdout", stream)

    assert mod.main(["validate", "--mission", str(mission_path), "--json"]) == 0
    assert stream.writes
    assert stream.closed is False
    assert mod.sys.stdout is stream


def test_emit_output_suppresses_write_time_broken_pipe_without_closing_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import goal_conductor as mod

    class WriteBrokenStdout:
        def __init__(self) -> None:
            self.closed = False

        def write(self, text: str) -> int:
            raise BrokenPipeError("downstream closed")

        def flush(self) -> None:
            raise AssertionError("flush should not run after write failure")

        def close(self) -> None:
            self.closed = True

    stream = WriteBrokenStdout()
    monkeypatch.setattr(mod.sys, "stdout", stream)

    mod._emit_output("payload")

    assert stream.closed is False
    assert mod.sys.stdout is stream


def test_mute_stdout_redirects_fd_without_replacing_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import goal_conductor as mod

    class FileStdout:
        def __init__(self) -> None:
            self.closed = False

        def fileno(self) -> int:
            return 123

        def close(self) -> None:
            self.closed = True

    dup2_calls: list[tuple[int, int]] = []
    close_calls: list[int] = []

    stream = FileStdout()
    monkeypatch.setattr(mod.sys, "stdout", stream)
    monkeypatch.setattr(mod.os, "open", lambda *_args: 456)
    monkeypatch.setattr(mod.os, "dup2", lambda src, dst: dup2_calls.append((src, dst)))
    monkeypatch.setattr(mod.os, "close", lambda fd: close_calls.append(fd))

    assert mod._mute_stdout_after_broken_pipe() is True

    assert dup2_calls == [(456, 123)]
    assert close_calls == [456]
    assert stream.closed is False
    assert mod.sys.stdout is stream


def test_mute_stdout_leaves_unfiled_stream_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    import goal_conductor as mod

    class UnfiledStdout:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    stream = UnfiledStdout()
    monkeypatch.setattr(mod.sys, "stdout", stream)

    assert mod._mute_stdout_after_broken_pipe() is False

    assert stream.closed is False
    assert mod.sys.stdout is stream


def test_emit_output_ignores_missing_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    import goal_conductor as mod

    monkeypatch.setattr(mod.sys, "stdout", None)

    mod._emit_output("payload")


def test_emit_output_accepts_stream_without_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    import goal_conductor as mod

    class WriteOnlyStdout:
        def __init__(self) -> None:
            self.writes: list[str] = []

        def write(self, text: str) -> int:
            self.writes.append(text)
            return len(text)

    stream = WriteOnlyStdout()
    monkeypatch.setattr(mod.sys, "stdout", stream)

    mod._emit_output("payload")

    assert stream.writes == ["payload", "\n"]


def test_emit_output_propagates_closed_stream_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import goal_conductor as mod

    class ClosedStdout:
        def __init__(self) -> None:
            self.closed = False

        def write(self, text: str) -> int:
            raise ValueError("I/O operation on closed file")

        def close(self) -> None:
            self.closed = True

    stream = ClosedStdout()
    monkeypatch.setattr(mod.sys, "stdout", stream)

    with pytest.raises(ValueError, match="closed file"):
        mod._emit_output("payload")

    assert stream.closed is False
    assert mod.sys.stdout is stream


def test_emit_output_propagates_bad_file_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import errno

    import goal_conductor as mod

    class BadFdStdout:
        def write(self, text: str) -> int:
            raise OSError(errno.EBADF, "bad file descriptor")

    stream = BadFdStdout()
    monkeypatch.setattr(mod.sys, "stdout", stream)

    with pytest.raises(OSError) as exc_info:
        mod._emit_output("payload")

    assert exc_info.value.errno == errno.EBADF
    assert mod.sys.stdout is stream


def test_validate_json_real_pipe_close_exits_zero(tmp_path: Path) -> None:
    mission_path = tmp_path / "mission.yaml"
    mission_path.write_text(
        """
name: pipe-smoke
objective: Verify real early-close pipe behavior.
lanes:
  - id: panel
    mode: panel
    goal: Review sampled output.
    prompt: Review only.
""",
        encoding="utf-8",
    )

    proc = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPTS_DIR / "goal_conductor.py"),
            "validate",
            "--mission",
            str(mission_path),
            "--json",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    proc.stdout.close()
    _, stderr = proc.communicate(timeout=10)

    assert proc.returncode == 0
    assert "BrokenPipeError" not in stderr


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


def test_execute_blocks_existing_autonomous_codex_lane_reuse(
    tmp_path: Path,
) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    payload["lanes"] = [
        {
            "id": "existing-lane",
            "agent": "codex",
            "mode": "implementation",
            "goal": "Continue an existing Codex lane.",
            "task_id": "Q-mission-conductor",
            "claimed_paths": ["docs/guides/CONDUCTOR_WORKFLOW.md"],
            "prompt": "Continue safely.",
        }
    ]
    runner = FakeRunner(mod, open_prs=[])
    conductor = mod.GoalConductor(
        mission=mod.Mission.from_dict(payload),
        repo_root=tmp_path,
        execute=True,
        runner=runner,
    )

    result = conductor.run_once()

    assert result.decisions[0].action == "blocked"
    assert "existing autonomous Codex lanes are not reused" in result.decisions[0].reason
    assert runner.executed == []


def test_execute_blocks_codex_lane_when_bridge_reports_missing_liveness(
    tmp_path: Path,
) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    payload["lanes"] = [
        {
            "id": "codex-lane",
            "agent": "codex",
            "mode": "implementation",
            "goal": "Patch conductor docs.",
            "prompt": "Patch only the scoped docs.",
            "task_id": "Q-mission-conductor",
            "claimed_paths": ["docs/guides/CONDUCTOR_WORKFLOW.md"],
        }
    ]
    runner = FakeRunner(
        mod,
        open_prs=[],
        bridge_snapshot={
            "health": {
                "ok": False,
                "issues": [
                    {
                        "type": "lane_missing_heartbeat",
                        "lane_id": "codex-lane",
                        "owner_state": "active_lane_missing_liveness",
                        "detail": "active lane 'codex-lane' has no heartbeat timestamp",
                        "recommended_operator_action": (
                            "start or refresh agent_heartbeat.py before treating owner as live"
                        ),
                    }
                ],
            },
            "recent_blockers": [],
            "summary": {"active_lanes": 1, "fresh_agent_heartbeats": 0},
        },
    )
    conductor = mod.GoalConductor(
        mission=mod.Mission.from_dict(payload),
        repo_root=tmp_path,
        execute=True,
        runner=runner,
    )

    result = conductor.run_once()

    assert result.decisions[0].action == "blocked"
    assert "operator snapshot reports lane_missing_heartbeat" in result.decisions[0].reason
    assert "agent_heartbeat.py" in result.decisions[0].reason
    assert runner.executed == []


def test_execute_blocks_codex_lane_when_bridge_reports_reconciler_conflict(
    tmp_path: Path,
) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    payload["lanes"] = [
        {
            "id": "codex-lane",
            "agent": "codex",
            "mode": "implementation",
            "goal": "Patch conductor docs.",
            "prompt": "Patch only the scoped docs.",
            "task_id": "Q-mission-conductor",
            "claimed_paths": ["docs/guides/CONDUCTOR_WORKFLOW.md"],
        }
    ]
    runner = FakeRunner(
        mod,
        open_prs=[],
        bridge_snapshot={
            "health": {
                "ok": False,
                "issues": [
                    {
                        "type": "lane_conflict",
                        "lane_id": "codex-lane",
                        "owner_state": "lane_conflict",
                        "detail": "lane 'codex-lane' in conflict with codex-other",
                        "recommended_operator_action": (
                            "resolve_lane_conflicts.py dry-run before mutation or cleanup"
                        ),
                    }
                ],
            },
            "recent_blockers": [],
            "summary": {"active_lanes": 1, "conflict_lanes": 1},
        },
    )
    conductor = mod.GoalConductor(
        mission=mod.Mission.from_dict(payload),
        repo_root=tmp_path,
        execute=True,
        runner=runner,
    )

    result = conductor.run_once()

    assert result.decisions[0].action == "blocked"
    assert "operator snapshot reports lane_conflict" in result.decisions[0].reason
    assert "resolve_lane_conflicts.py" in result.decisions[0].reason
    assert runner.executed == []


def test_execute_launches_new_codex_lane_with_required_lease_flags(tmp_path: Path) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    payload["lanes"] = [
        {
            "id": "codex-lane",
            "agent": "codex",
            "mode": "implementation",
            "goal": "Patch conductor docs.",
            "prompt": "Patch only the scoped docs.",
            "task_id": "Q-mission-conductor",
            "claimed_paths": ["docs/guides/CONDUCTOR_WORKFLOW.md"],
            "write_scopes": ["docs/guides/"],
            "forbidden_paths": ["docs/guides/DO_NOT_TOUCH.md"],
            "tests": ["pre-commit run --files docs/guides/CONDUCTOR_WORKFLOW.md"],
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
    assert len(runner.executed) == 1
    (launch,) = runner.executed
    assert launch[:3] == ["python3", "scripts/agent_bridge.py", "launch"]
    assert launch.count("--goal") == 1
    assert launch[launch.index("--goal") + 1] == "Patch conductor docs."
    assert launch[launch.index("--task-id") + 1] == "Q-mission-conductor"
    assert launch[launch.index("--claimed-path") + 1] == "docs/guides/CONDUCTOR_WORKFLOW.md"
    assert launch[launch.index("--write-scope") + 1] == "docs/guides/"
    assert launch[launch.index("--forbidden-path") + 1] == "docs/guides/DO_NOT_TOUCH.md"
    assert (
        launch[launch.index("--test") + 1]
        == "pre-commit run --files docs/guides/CONDUCTOR_WORKFLOW.md"
    )
    assert launch[launch.index("--lane") + 1] == "codex-lane"
    assert launch[launch.index("--source") + 1] == ""
    assert launch[launch.index("--status") + 1] == "active"
    assert launch[launch.index("--next-action") + 1] == ""
    assert "--file" in launch
    assert "--strict-verify" not in launch
    prompt_path = Path(launch[launch.index("--file") + 1])
    prompt = prompt_path.read_text(encoding="utf-8")
    assert "Mission lane contract:" in prompt
    assert "task_id: Q-mission-conductor" in prompt
    assert "Tier 3/4 settlement" in prompt


def test_execute_launches_new_codex_lane_with_opt_in_strict_verify(
    tmp_path: Path,
) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    payload["lanes"] = [
        {
            "id": "codex-lane",
            "agent": "codex",
            "mode": "implementation",
            "goal": "Patch conductor docs.",
            "prompt": "Patch only the scoped docs.",
            "task_id": "Q-mission-conductor",
            "claimed_paths": ["docs/guides/CONDUCTOR_WORKFLOW.md"],
            "strict_launch_verify": True,
        }
    ]
    runner = FakeRunner(mod, open_prs=[])
    conductor = mod.GoalConductor(
        mission=mod.Mission.from_dict(payload),
        repo_root=tmp_path,
        execute=True,
        runner=runner,
    )

    result = conductor.run_once()

    assert result.decisions[0].action == "execute"
    assert "--strict-verify" in runner.executed[0]


def test_execute_launches_new_codex_lane_can_opt_out_of_strict_verify(
    tmp_path: Path,
) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    payload["lanes"] = [
        {
            "id": "codex-lane",
            "agent": "codex",
            "mode": "implementation",
            "goal": "Patch conductor docs.",
            "prompt": "Patch only the scoped docs.",
            "task_id": "Q-mission-conductor",
            "claimed_paths": ["docs/guides/CONDUCTOR_WORKFLOW.md"],
            "strict_launch_verify": False,
        }
    ]
    runner = FakeRunner(mod, open_prs=[])
    conductor = mod.GoalConductor(
        mission=mod.Mission.from_dict(payload),
        repo_root=tmp_path,
        execute=True,
        runner=runner,
    )

    result = conductor.run_once()

    assert result.decisions[0].action == "execute"
    assert "--strict-verify" not in runner.executed[0]


def test_autonomous_codex_lane_without_lease_scope_is_blocked(tmp_path: Path) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    payload["lanes"] = [
        {
            "id": "unleased",
            "agent": "codex",
            "mode": "implementation",
            "goal": "Patch something.",
            "prompt": "Patch broadly.",
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

    assert result.decisions[0].action == "blocked"
    assert "requires task_id plus at least one" in result.decisions[0].reason
    assert runner.executed == []


def test_autonomous_codex_lane_with_only_tests_is_blocked(tmp_path: Path) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    payload["lanes"] = [
        {
            "id": "tests-only",
            "agent": "codex",
            "mode": "implementation",
            "goal": "Patch something.",
            "prompt": "Patch broadly.",
            "task_id": "Q-tests-only",
            "tests": ["python3 -m pytest tests/scripts/test_goal_conductor.py -q"],
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

    assert result.decisions[0].action == "blocked"
    assert "claimed_path or write_scope" in result.decisions[0].reason
    assert runner.executed == []


def test_loop_continues_on_queue_cap_gate_only(tmp_path: Path) -> None:
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

    assert len(results) == 3
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


def test_execute_does_not_treat_string_false_as_human_settlement_gate(
    tmp_path: Path,
) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    open_prs = [{"number": 7156, "title": "tier 2 gate", "isDraft": False}]
    merge_packet = {
        "entries": [
            {
                "pr_number": 7156,
                "tier": 2,
                "tier_name": "tier_2_live_automation",
                "requires_human_risk_settlement": "false",
            }
        ]
    }
    runner = FakeRunner(mod, open_prs=open_prs, merge_packet=merge_packet)
    conductor = mod.GoalConductor(
        mission=mod.Mission.from_dict(payload),
        repo_root=tmp_path,
        execute=True,
        runner=runner,
    )

    result = conductor.run_once()

    assert not any("human/non-author settlement gate" in gate for gate in result.hard_gates)
    assert [decision.action for decision in result.decisions] == ["execute", "execute"]


def test_execute_blocks_all_lanes_when_unresolved_dissent_present(tmp_path: Path) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    open_prs = [{"number": 7156, "title": "dissent", "isDraft": False}]
    merge_packet = {
        "entries": [
            {
                "pr_number": 7156,
                "tier": 2,
                "tier_name": "tier_2_live_automation",
                "unresolved_dissent": True,
                "requires_human_risk_settlement": False,
            }
        ]
    }
    runner = FakeRunner(mod, open_prs=open_prs, merge_packet=merge_packet)
    conductor = mod.GoalConductor(
        mission=mod.Mission.from_dict(payload),
        repo_root=tmp_path,
        execute=True,
        runner=runner,
    )

    result = conductor.run_once()

    assert result.hard_gates == ["unresolved model dissent present: #7156"]
    assert [decision.action for decision in result.decisions] == ["blocked", "blocked"]
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


def test_opt_in_exact_gated_merge_runs_settle_then_normal_protected_squash(
    tmp_path: Path,
) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    payload["merge_policy"] = "exact_gated_tier_0_2"
    head = "abc123def456"
    open_prs = [{"number": 77, "title": "ready", "isDraft": False}]
    merge_packet = {
        "admin_squash_order": [77],
        "not_ready": [],
        "entries": [
            {
                "pr_number": 77,
                "head_sha": head,
                "tier": 2,
                "status": "satisfied",
                "verdict": "admin_squash_allowed",
                "admin_squash_allowed": True,
                "unresolved_dissent": False,
                "requires_human_risk_settlement": False,
            }
        ],
    }
    runner = FakeRunner(
        mod,
        open_prs=open_prs,
        merge_packet=merge_packet,
        settle_payload={
            "status": "packet_authorized_dry_run",
            "blockers": [],
            "head_sha": head,
            "selected_pr": 77,
        },
    )
    conductor = mod.GoalConductor(
        mission=mod.Mission.from_dict(payload),
        repo_root=tmp_path,
        execute=True,
        runner=runner,
    )

    result = conductor.run_once()

    assert [decision.action for decision in result.decisions] == ["merged", "execute", "execute"]
    assert runner.executed[0] == [
        "gh",
        "pr",
        "merge",
        "77",
        "--squash",
        "--match-head-commit",
        head,
    ]
    assert "--admin" not in runner.executed[0]


def test_opt_in_exact_gated_merge_dry_run_does_not_probe_settle_one(
    tmp_path: Path,
) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    payload["merge_policy"] = "exact_gated_tier_0_2"
    head = "abc123def456"
    open_prs = [{"number": 77, "title": "ready", "isDraft": False}]
    merge_packet = {
        "admin_squash_order": [77],
        "not_ready": [],
        "entries": [
            {
                "pr_number": 77,
                "head_sha": head,
                "tier": 2,
                "status": "satisfied",
                "verdict": "admin_squash_allowed",
                "admin_squash_allowed": True,
                "unresolved_dissent": False,
                "requires_human_risk_settlement": False,
            }
        ],
    }
    runner = FakeRunner(
        mod,
        open_prs=open_prs,
        merge_packet=merge_packet,
        settle_payload={
            "status": "needs_packet_rerun",
            "blockers": [],
            "head_sha": head,
            "selected_pr": 77,
        },
    )
    conductor = mod.GoalConductor(
        mission=mod.Mission.from_dict(payload),
        repo_root=tmp_path,
        execute=False,
        runner=runner,
    )

    result = conductor.run_once()

    assert result.decisions[0].action == "dry_run"
    assert not any(
        call[:3] == ["python3", "scripts/settle_one_pr.py", "--pr"] for call in runner.calls
    )
    assert result.decisions[0].commands == [
        ["python3", "scripts/settle_one_pr.py", "--pr", "77", "--json"],
        ["gh", "pr", "merge", "77", "--squash", "--match-head-commit", head],
    ]


def test_opt_in_exact_gated_merge_blocks_on_settle_one_blockers(tmp_path: Path) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    payload["merge_policy"] = "exact_gated_tier_0_2"
    head = "abc123def456"
    open_prs = [{"number": 78, "title": "ready", "isDraft": False}]
    merge_packet = {
        "admin_squash_order": [78],
        "not_ready": [],
        "entries": [
            {
                "pr_number": 78,
                "head_sha": head,
                "tier": 1,
                "status": "satisfied",
                "verdict": "admin_squash_allowed",
                "admin_squash_allowed": True,
            }
        ],
    }
    runner = FakeRunner(
        mod,
        open_prs=open_prs,
        merge_packet=merge_packet,
        settle_payload={"blockers": ["active owner"], "head_sha": head},
    )
    conductor = mod.GoalConductor(
        mission=mod.Mission.from_dict(payload),
        repo_root=tmp_path,
        execute=True,
        runner=runner,
    )

    result = conductor.run_once()

    assert [decision.action for decision in result.decisions] == ["blocked", "execute", "execute"]
    assert "active owner" in result.decisions[0].reason
    assert not any(command[:3] == ["gh", "pr", "merge"] for command in runner.executed)


def test_opt_in_exact_gated_merge_ignores_stale_packet_pr_not_open(
    tmp_path: Path,
) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    payload["merge_policy"] = "exact_gated_tier_0_2"
    head = "abc123def456"
    merge_packet = {
        "admin_squash_order": [79],
        "not_ready": [],
        "entries": [
            {
                "pr_number": 79,
                "head_sha": head,
                "tier": 2,
                "status": "satisfied",
                "verdict": "admin_squash_allowed",
                "admin_squash_allowed": True,
                "unresolved_dissent": False,
                "requires_human_risk_settlement": False,
            }
        ],
    }
    runner = FakeRunner(
        mod,
        open_prs=[],
        merge_packet=merge_packet,
        settle_payload={
            "status": "packet_authorized_dry_run",
            "blockers": [],
            "head_sha": head,
            "selected_pr": 79,
        },
    )
    conductor = mod.GoalConductor(
        mission=mod.Mission.from_dict(payload),
        repo_root=tmp_path,
        execute=True,
        runner=runner,
    )

    result = conductor.run_once()

    assert [decision.action for decision in result.decisions] == ["execute", "execute"]
    assert not any(
        command[:3] == ["python3", "scripts/settle_one_pr.py", "--pr"]
        for command in runner.executed
    )
    assert not any(command[:3] == ["gh", "pr", "merge"] for command in runner.executed)


def test_opt_in_exact_gated_merge_respects_admin_order_and_not_ready(
    tmp_path: Path,
) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    payload["merge_policy"] = "exact_gated_tier_0_2"
    open_prs = [
        {"number": 81, "title": "ready-second", "isDraft": False},
        {"number": 82, "title": "blocked-first", "isDraft": False},
    ]
    merge_packet = {
        "admin_squash_order": [82, 81],
        "not_ready": [82],
        "entries": [
            {
                "pr_number": 81,
                "head_sha": "head81",
                "tier": 2,
                "status": "satisfied",
                "verdict": "admin_squash_allowed",
                "admin_squash_allowed": True,
                "unresolved_dissent": False,
                "requires_human_risk_settlement": False,
            },
            {
                "pr_number": 82,
                "head_sha": "head82",
                "tier": 2,
                "status": "satisfied",
                "verdict": "admin_squash_allowed",
                "admin_squash_allowed": True,
                "unresolved_dissent": False,
                "requires_human_risk_settlement": False,
            },
        ],
    }
    runner = FakeRunner(
        mod,
        open_prs=open_prs,
        merge_packet=merge_packet,
        settle_payload={
            "status": "packet_authorized_dry_run",
            "blockers": [],
            "head_sha": "head81",
            "selected_pr": 81,
        },
    )
    conductor = mod.GoalConductor(
        mission=mod.Mission.from_dict(payload),
        repo_root=tmp_path,
        execute=True,
        runner=runner,
    )

    result = conductor.run_once()

    assert [decision.action for decision in result.decisions] == ["merged", "execute", "execute"]
    assert runner.executed[0] == [
        "gh",
        "pr",
        "merge",
        "81",
        "--squash",
        "--match-head-commit",
        "head81",
    ]


def test_opt_in_exact_gated_merge_skips_unparseable_tier(tmp_path: Path) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    payload["merge_policy"] = "exact_gated_tier_0_2"
    open_prs = [{"number": 83, "title": "ready", "isDraft": False}]
    merge_packet = {
        "admin_squash_order": [83],
        "not_ready": [],
        "entries": [
            {
                "pr_number": 83,
                "head_sha": "head83",
                "tier": "not-a-tier",
                "status": "satisfied",
                "verdict": "admin_squash_allowed",
                "admin_squash_allowed": True,
                "unresolved_dissent": False,
                "requires_human_risk_settlement": False,
            }
        ],
    }
    runner = FakeRunner(mod, open_prs=open_prs, merge_packet=merge_packet)
    conductor = mod.GoalConductor(
        mission=mod.Mission.from_dict(payload),
        repo_root=tmp_path,
        execute=True,
        runner=runner,
    )

    result = conductor.run_once()

    assert "merge-packet entry has unparseable tier: #83" not in result.hard_gates
    assert not any(command[:3] == ["gh", "pr", "merge"] for command in runner.executed)


def test_opt_in_exact_gated_merge_runs_under_queue_cap_gate(
    tmp_path: Path,
) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 2
    payload["merge_policy"] = "exact_gated_tier_0_2"
    head = "abc123def456"
    open_prs = [
        {"number": 83, "title": "ready", "isDraft": False},
        {"number": 84, "title": "cap filler", "isDraft": False},
    ]
    merge_packet = {
        "admin_squash_order": [83],
        "not_ready": [],
        "entries": [
            {
                "pr_number": 83,
                "head_sha": head,
                "tier": 2,
                "status": "satisfied",
                "verdict": "admin_squash_allowed",
                "admin_squash_allowed": True,
                "unresolved_dissent": False,
                "requires_human_risk_settlement": False,
            }
        ],
    }
    runner = FakeRunner(
        mod,
        open_prs=open_prs,
        merge_packet=merge_packet,
        settle_payload={
            "status": "packet_authorized_dry_run",
            "blockers": [],
            "head_sha": head,
            "selected_pr": 83,
        },
    )
    conductor = mod.GoalConductor(
        mission=mod.Mission.from_dict(payload),
        repo_root=tmp_path,
        execute=True,
        runner=runner,
    )

    result = conductor.run_once()

    assert result.hard_gates == ["open PR queue at/above cap (2/2)"]
    assert [decision.action for decision in result.decisions] == ["merged", "blocked", "execute"]
    assert any(
        command[:3] == ["python3", "scripts/settle_one_pr.py", "--pr"] for command in runner.calls
    )
    assert any(command[:3] == ["gh", "pr", "merge"] for command in runner.calls)


def test_opt_in_exact_gated_merge_requires_admin_squash_order(tmp_path: Path) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    payload["merge_policy"] = "exact_gated_tier_0_2"
    head = "abc123def456"
    open_prs = [{"number": 79, "title": "ready", "isDraft": False}]
    merge_packet = {
        "not_ready": [],
        "entries": [
            {
                "pr_number": 79,
                "head_sha": head,
                "tier": 2,
                "status": "satisfied",
                "verdict": "admin_squash_allowed",
                "admin_squash_allowed": True,
                "unresolved_dissent": False,
                "requires_human_risk_settlement": False,
            }
        ],
    }
    runner = FakeRunner(
        mod,
        open_prs=open_prs,
        merge_packet=merge_packet,
        settle_payload={
            "status": "packet_authorized_dry_run",
            "blockers": [],
            "head_sha": head,
            "selected_pr": 79,
        },
    )
    conductor = mod.GoalConductor(
        mission=mod.Mission.from_dict(payload),
        repo_root=tmp_path,
        execute=True,
        runner=runner,
    )

    result = conductor.run_once()

    assert [decision.action for decision in result.decisions] == ["execute", "execute"]
    assert not any(
        command[:3] == ["python3", "scripts/settle_one_pr.py", "--pr"]
        for command in runner.executed
    )
    assert not any(command[:3] == ["gh", "pr", "merge"] for command in runner.executed)


def test_opt_in_exact_gated_merge_requires_settle_authorized_status(tmp_path: Path) -> None:
    import goal_conductor as mod

    payload = _mission_dict(tmp_path)
    payload["limits"]["queue_cap"] = 5
    payload["merge_policy"] = "exact_gated_tier_0_2"
    head = "abc123def456"
    open_prs = [{"number": 80, "title": "ready", "isDraft": False}]
    merge_packet = {
        "admin_squash_order": [80],
        "not_ready": [],
        "entries": [
            {
                "pr_number": 80,
                "head_sha": head,
                "tier": 2,
                "status": "satisfied",
                "verdict": "admin_squash_allowed",
                "admin_squash_allowed": True,
                "unresolved_dissent": False,
                "requires_human_risk_settlement": False,
            }
        ],
    }
    runner = FakeRunner(
        mod,
        open_prs=open_prs,
        merge_packet=merge_packet,
        settle_payload={
            "status": "needs_packet_rerun",
            "blockers": [],
            "head_sha": head,
            "selected_pr": 80,
        },
    )
    conductor = mod.GoalConductor(
        mission=mod.Mission.from_dict(payload),
        repo_root=tmp_path,
        execute=True,
        runner=runner,
    )

    result = conductor.run_once()

    assert result.decisions[0].action == "blocked"
    assert result.decisions[0].reason == "settle_one_pr.py status=needs_packet_rerun"
    assert [decision.action for decision in result.decisions] == ["blocked", "execute", "execute"]
    assert not any(command[:3] == ["gh", "pr", "merge"] for command in runner.executed)


def test_discover_loop_surfaces_reports_existing_tools(tmp_path: Path) -> None:
    import goal_conductor as mod

    tool = tmp_path / "scripts/agent_bridge.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("# bridge\n", encoding="utf-8")

    surfaces = mod.discover_loop_surfaces(tmp_path)

    assert surfaces["agent_bridge"]["exists"] is True
    assert surfaces["boss_loop"]["exists"] is False
