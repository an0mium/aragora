#!/usr/bin/env python3
"""Goal-mode conductor for Aragora agent lanes.

This is a thin orchestration wrapper around existing repo surfaces:

* ``scripts/agent_bridge.py`` for long-running tmux agent lanes.
* ``scripts/multi_agent_dialog.py`` for bounded heterogeneous review panels.
* ``aragora review-queue merge-packet`` and GitHub state for gates.

The conductor is intentionally conservative. It is read-only by default; pass
``--execute`` before it will launch/send agents or run panel prompts.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path(".aragora/goal-conductor")
DEFAULT_QUEUE_CAP = 6
DEFAULT_MAX_IMPLEMENTATION_LANES = 2
DEFAULT_MAX_REVIEW_LANES = 1
ALLOWED_AGENTS = {"codex", "claude", "droid", "factory"}
PANEL_MODE = "panel"
IMPLEMENTATION_MODES = {"implementation", "implement", "write"}
REVIEW_MODES = {"review", "watch", "validator", PANEL_MODE}
MUTATING_LANE_MODES = IMPLEMENTATION_MODES


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug[:80] or "goal"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_str_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value)]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mission file.

    PyYAML is already a repo dependency. Import it lazily so ``--help`` remains
    usable even in partially bootstrapped environments.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment failure
        raise SystemExit("PyYAML is required to load mission YAML files") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"mission file must contain a mapping: {path}")
    return data


@dataclass(frozen=True)
class LaneSpec:
    lane_id: str
    agent: str
    goal: str
    prompt: str = ""
    prompt_file: str = ""
    source: str = ""
    mode: str = "implementation"
    cwd: str = "."
    autonomous: bool = True
    status: str = "active"
    next_action: str = ""
    agents_spec: str = "heterogeneous"
    context_file: str = ""
    round_id: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, index: int) -> "LaneSpec":
        lane_id = str(payload.get("id") or payload.get("lane_id") or f"lane-{index}").strip()
        agent = str(payload.get("agent") or "codex").strip().lower()
        mode = str(payload.get("mode") or "implementation").strip().lower()
        goal = str(payload.get("goal") or payload.get("title") or "").strip()
        if not goal:
            raise ValueError(f"lane {lane_id!r} must define goal")
        if mode == PANEL_MODE:
            agent = PANEL_MODE
        elif agent not in ALLOWED_AGENTS:
            raise ValueError(f"lane {lane_id!r} has unsupported agent {agent!r}")
        prompt = str(payload.get("prompt") or "").strip()
        prompt_file = str(payload.get("prompt_file") or payload.get("file") or "").strip()
        if not prompt and not prompt_file:
            raise ValueError(f"lane {lane_id!r} must define prompt or prompt_file")
        return cls(
            lane_id=lane_id,
            agent=agent,
            goal=goal,
            prompt=prompt,
            prompt_file=prompt_file,
            source=str(payload.get("source") or "").strip(),
            mode=mode,
            cwd=str(payload.get("cwd") or ".").strip(),
            autonomous=bool(payload.get("autonomous", True)),
            status=str(payload.get("status") or "active").strip(),
            next_action=str(payload.get("next_action") or "").strip(),
            agents_spec=str(payload.get("agents_spec") or "heterogeneous").strip(),
            context_file=str(payload.get("context_file") or "").strip(),
            round_id=str(payload.get("round_id") or "").strip(),
        )

    @property
    def mutates(self) -> bool:
        return self.mode in MUTATING_LANE_MODES

    @property
    def is_review(self) -> bool:
        return self.mode in REVIEW_MODES


@dataclass(frozen=True)
class MissionLimits:
    queue_cap: int = DEFAULT_QUEUE_CAP
    max_implementation_lanes: int = DEFAULT_MAX_IMPLEMENTATION_LANES
    max_review_lanes: int = DEFAULT_MAX_REVIEW_LANES

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MissionLimits":
        return cls(
            queue_cap=int(payload.get("queue_cap", DEFAULT_QUEUE_CAP)),
            max_implementation_lanes=int(
                payload.get("max_implementation_lanes", DEFAULT_MAX_IMPLEMENTATION_LANES)
            ),
            max_review_lanes=int(payload.get("max_review_lanes", DEFAULT_MAX_REVIEW_LANES)),
        )


@dataclass(frozen=True)
class Mission:
    name: str
    lanes: list[LaneSpec]
    objective: str = ""
    stop_condition: str = ""
    base_branch: str = "main"
    output_dir: Path = DEFAULT_OUTPUT_DIR
    limits: MissionLimits = field(default_factory=MissionLimits)
    checkpoints: list[str] = field(default_factory=list)
    external_references: list[str] = field(default_factory=list)
    stop_conditions: list[str] = field(default_factory=list)
    allowed_mutations: list[str] = field(default_factory=list)
    collect_merge_packets: bool = True
    max_merge_packets: int = 5

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Mission":
        lanes_payload = payload.get("lanes")
        if not isinstance(lanes_payload, list) or not lanes_payload:
            raise ValueError("mission must define at least one lane")
        raw_limits = payload.get("limits") or {}
        if not isinstance(raw_limits, dict):
            raise ValueError("mission limits must be a mapping")
        name = str(payload.get("name") or "goal-mode").strip()
        return cls(
            name=name,
            objective=str(payload.get("objective") or "").strip(),
            stop_condition=str(payload.get("stop_condition") or "").strip(),
            lanes=[
                LaneSpec.from_dict(lane, index=i + 1)
                for i, lane in enumerate(lanes_payload)
                if isinstance(lane, dict)
            ],
            base_branch=str(payload.get("base_branch") or "main").strip(),
            output_dir=Path(str(payload.get("output_dir") or DEFAULT_OUTPUT_DIR)),
            limits=MissionLimits.from_dict(raw_limits),
            checkpoints=_as_str_list(payload.get("checkpoints")),
            external_references=_as_str_list(payload.get("external_references")),
            stop_conditions=_as_str_list(payload.get("stop_conditions")),
            allowed_mutations=_as_str_list(payload.get("allowed_mutations")),
            collect_merge_packets=bool(payload.get("collect_merge_packets", True)),
            max_merge_packets=int(payload.get("max_merge_packets", 5)),
        )


@dataclass
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    def json(self) -> Any:
        return json.loads(self.stdout or "null")


class CommandRunner:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def run(self, args: list[str], *, timeout: int = 60) -> CommandResult:
        try:
            proc = subprocess.run(
                args,
                cwd=self.repo_root,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            return CommandResult(
                args=args,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                args=args,
                returncode=124,
                stdout=exc.stdout if isinstance(exc.stdout, str) else "",
                stderr=exc.stderr if isinstance(exc.stderr, str) else "",
                timed_out=True,
            )


def _path_summary(path: Path) -> dict[str, Any]:
    """Return stable local state for a file/directory without opening it."""
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "is_dir": path.is_dir(),
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
    }


def discover_loop_surfaces(repo_root: Path) -> dict[str, Any]:
    """Detect repo-native long-running loop surfaces used by goal mode.

    This deliberately avoids starting, stopping, or querying privileged launchd
    state. The conductor only needs enough local truth to route work: boss loop
    for queued implementation, Ralph for campaign-style incident repair, nomic
    loop for experimental self-improvement, and bridge/dialog scripts for
    explicit full-agent coordination.
    """
    paths = {
        "agent_bridge": repo_root / "scripts/agent_bridge.py",
        "tmux_launcher": repo_root / "scripts/tmux_session_launcher.sh",
        "multi_agent_dialog": repo_root / "scripts/multi_agent_dialog.py",
        "boss_loop": repo_root / "aragora/swarm/boss_loop.py",
        "boss_metrics": repo_root / ".aragora/overnight/boss_metrics.jsonl",
        "boss_launchd_log": repo_root / ".aragora/overnight/boss-loop-launchd.log",
        "ralph_supervisor": repo_root / "aragora/ralph/supervisor.py",
        "ralph_cli": repo_root / "aragora/cli/commands/ralph.py",
        "nomic_loop": repo_root / "scripts/nomic_loop.py",
        "nomic_orchestrator": repo_root / "aragora/nomic/autonomous_orchestrator.py",
        "review_merge_packet": repo_root / "aragora/cli/commands/review_queue.py",
    }
    return {name: _path_summary(path) for name, path in paths.items()}


@dataclass
class ConductorEvent:
    timestamp: str
    phase: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass
class LaneDecision:
    lane_id: str
    action: str
    reason: str
    commands: list[list[str]] = field(default_factory=list)


@dataclass
class ConductorResult:
    mission_name: str
    execute: bool
    snapshot: dict[str, Any]
    decisions: list[LaneDecision]
    hard_gates: list[str]
    jsonl_path: Path
    markdown_path: Path


def _merge_packet_entries(packet: Any) -> list[dict[str, Any]]:
    if isinstance(packet, dict):
        entries = packet.get("entries")
        if isinstance(entries, list):
            return [entry for entry in entries if isinstance(entry, dict)]
        packets = packet.get("packets")
        if isinstance(packets, list):
            return [entry for entry in packets if isinstance(entry, dict)]
    if isinstance(packet, list):
        return [entry for entry in packet if isinstance(entry, dict)]
    return []


class GoalConductor:
    def __init__(
        self,
        *,
        mission: Mission,
        repo_root: Path,
        execute: bool = False,
        runner: CommandRunner | None = None,
    ):
        self.mission = mission
        self.repo_root = repo_root.resolve()
        self.execute = execute
        self.runner = runner or CommandRunner(self.repo_root)
        self.events: list[ConductorEvent] = []

    def emit(self, phase: str, message: str, **data: Any) -> None:
        self.events.append(
            ConductorEvent(timestamp=_utc_now(), phase=phase, message=message, data=data)
        )

    def _run_json(self, args: list[str], *, timeout: int = 60) -> tuple[Any, CommandResult]:
        result = self.runner.run(args, timeout=timeout)
        if result.returncode != 0:
            return None, result
        try:
            return result.json(), result
        except json.JSONDecodeError:
            return None, result

    def snapshot(self) -> dict[str, Any]:
        root_status = self.runner.run(["git", "status", "--short", "--branch"]).stdout.strip()
        head = self.runner.run(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()
        origin_main = self.runner.run(
            ["git", "rev-parse", "--short", f"refs/remotes/origin/{self.mission.base_branch}"]
        ).stdout.strip()
        prs, pr_result = self._run_json(
            [
                "gh",
                "pr",
                "list",
                "--state",
                "open",
                "--limit",
                "100",
                "--json",
                "number,title,isDraft,headRefName,mergeStateStatus,reviewDecision,url",
            ],
            timeout=60,
        )
        if not isinstance(prs, list):
            prs = []
        merge_packets: Any = []
        merge_packet_status: dict[str, Any] = {
            "targets": [],
            "returncode": None,
            "parse_ok": True,
        }
        if self.mission.collect_merge_packets:
            packet_targets = [
                int(pr["number"])
                for pr in prs
                if isinstance(pr, dict) and pr.get("number") and not bool(pr.get("isDraft"))
            ][: self.mission.max_merge_packets]
            merge_packet_status["targets"] = packet_targets
            if packet_targets:
                packet_args = [
                    "python3",
                    "-m",
                    "aragora.cli.main",
                    "review-queue",
                    "merge-packet",
                    "--json",
                ]
                for number in packet_targets:
                    packet_args.extend(["--pr", str(number)])
                merge_packets, packet_result = self._run_json(packet_args, timeout=120)
                merge_packet_status["returncode"] = packet_result.returncode
                merge_packet_status["parse_ok"] = isinstance(merge_packets, (dict, list))
                if merge_packets is None:
                    merge_packets = []
        publisher, _ = self._run_json(["python3", "scripts/publisher_freshness_check.py", "--json"])
        bridge, _ = self._run_json(
            [
                "python3",
                "scripts/agent_bridge.py",
                "--json",
                "operator-snapshot",
                "--summary-only",
            ],
            timeout=30,
        )
        proof_health, _ = self._run_json(
            [
                "python3",
                "-m",
                "aragora.cli.main",
                "review-queue",
                "health",
                "--json",
            ],
            timeout=60,
        )
        dirty_lines = [line for line in root_status.splitlines()[1:] if line.strip()]
        snapshot = {
            "generated_at": _utc_now(),
            "repo_root": str(self.repo_root),
            "root": {
                "status": root_status,
                "head": head,
                "origin_base": origin_main,
                "dirty_file_count": len(dirty_lines),
            },
            "open_prs": prs,
            "open_pr_count": len(prs),
            "open_non_draft_count": sum(1 for pr in prs if not bool(pr.get("isDraft"))),
            "merge_packets": merge_packets,
            "merge_packet_status": merge_packet_status,
            "publisher": publisher,
            "agent_bridge": bridge,
            "loop_surfaces": discover_loop_surfaces(self.repo_root),
            "proof_loop_health": proof_health,
            "pr_query_returncode": pr_result.returncode,
        }
        self.emit("snapshot", "captured live state", open_pr_count=len(prs), dirty=len(dirty_lines))
        return snapshot

    def hard_gates(self, snapshot: dict[str, Any]) -> list[str]:
        gates: list[str] = []
        if int(snapshot.get("pr_query_returncode") or 0) != 0:
            gates.append(f"open PR query failed: rc={snapshot.get('pr_query_returncode')}")
        packet_status = snapshot.get("merge_packet_status")
        if isinstance(packet_status, dict) and packet_status.get("targets"):
            if int(packet_status.get("returncode") or 0) != 0 or not bool(
                packet_status.get("parse_ok", True)
            ):
                gates.append(
                    "merge-packet query failed for "
                    f"{','.join(str(item) for item in packet_status.get('targets') or [])}"
                )
        if snapshot["root"]["dirty_file_count"]:
            gates.append("root checkout is dirty")
        if snapshot["open_pr_count"] >= self.mission.limits.queue_cap:
            gates.append(
                f"open PR queue at/above cap ({snapshot['open_pr_count']}/{self.mission.limits.queue_cap})"
            )
        if snapshot.get("publisher") and snapshot["publisher"].get("verdict") not in {
            None,
            "ready",
        }:
            gates.append(f"publisher not ready: {snapshot['publisher'].get('summary', 'unknown')}")
        for entry in _merge_packet_entries(snapshot.get("merge_packets")):
            tier = int(entry.get("tier") or 0)
            if tier >= 4 or bool(entry.get("requires_human_risk_settlement")):
                pr_number = entry.get("pr_number", "?")
                tier_name = entry.get("tier_name") or f"tier_{tier}"
                gates.append(f"human/non-author settlement gate present: #{pr_number} {tier_name}")
        return gates

    def _prompt_file_for(self, lane: LaneSpec, run_dir: Path) -> Path:
        if lane.prompt_file:
            path = Path(lane.prompt_file)
            return path if path.is_absolute() else self.repo_root / path
        prompt_dir = run_dir / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        path = prompt_dir / f"{_slug(lane.lane_id)}.md"
        path.write_text(lane.prompt + "\n", encoding="utf-8")
        return path

    def _known_sessions(self) -> set[str]:
        data, _ = self._run_json(["python3", "scripts/agent_bridge.py", "--json", "sessions"])
        if not isinstance(data, list):
            return set()
        return {str(item.get("name", "")) for item in data if isinstance(item, dict)}

    def _agent_commands(self, lane: LaneSpec, run_dir: Path, sessions: set[str]) -> list[list[str]]:
        prompt_file = self._prompt_file_for(lane, run_dir)
        cwd_path = Path(lane.cwd)
        cwd = str(cwd_path if cwd_path.is_absolute() else self.repo_root / cwd_path)
        commands: list[list[str]] = []
        if lane.lane_id not in sessions:
            launch = [
                "python3",
                "scripts/agent_bridge.py",
                "launch",
                "--name",
                lane.lane_id,
                "--agent",
                lane.agent,
                "--cwd",
                cwd,
            ]
            if lane.autonomous:
                launch.append("--autonomous")
            commands.append(launch)
        commands.append(
            [
                "python3",
                "scripts/agent_bridge.py",
                "send",
                lane.lane_id,
                "--file",
                str(prompt_file),
                "--lane",
                lane.lane_id,
                "--goal",
                lane.goal,
                "--source",
                lane.source,
                "--status",
                lane.status,
                "--next-action",
                lane.next_action,
            ]
        )
        return commands

    def _panel_commands(self, lane: LaneSpec, run_dir: Path) -> list[list[str]]:
        prompt_file = self._prompt_file_for(lane, run_dir)
        round_id = lane.round_id or _slug(lane.lane_id)
        output_dir = run_dir / "panels" / _slug(lane.lane_id)
        command = [
            "python3",
            "scripts/multi_agent_dialog.py",
            "--round-id",
            round_id,
            "--prompt-file",
            str(prompt_file),
            "--agents-spec",
            lane.agents_spec,
            "--output-dir",
            str(output_dir),
        ]
        if lane.context_file:
            command.extend(["--context-file", lane.context_file])
        return [command]

    def plan_lanes(self, snapshot: dict[str, Any], run_dir: Path) -> list[LaneDecision]:
        sessions = self._known_sessions()
        implementation_used = 0
        review_used = 0
        at_cap = snapshot["open_pr_count"] >= self.mission.limits.queue_cap
        decisions: list[LaneDecision] = []
        for lane in self.mission.lanes:
            if lane.mutates and at_cap:
                decisions.append(
                    LaneDecision(
                        lane_id=lane.lane_id,
                        action="blocked",
                        reason="queue cap reached; mutating implementation lanes are disabled",
                    )
                )
                continue
            if lane.mutates:
                if implementation_used >= self.mission.limits.max_implementation_lanes:
                    decisions.append(
                        LaneDecision(
                            lane_id=lane.lane_id,
                            action="blocked",
                            reason="max implementation lanes already assigned",
                        )
                    )
                    continue
                implementation_used += 1
            elif lane.is_review:
                if review_used >= self.mission.limits.max_review_lanes:
                    decisions.append(
                        LaneDecision(
                            lane_id=lane.lane_id,
                            action="blocked",
                            reason="max review lanes already assigned",
                        )
                    )
                    continue
                review_used += 1
            commands = (
                self._panel_commands(lane, run_dir)
                if lane.mode == PANEL_MODE
                else self._agent_commands(lane, run_dir, sessions)
            )
            decisions.append(
                LaneDecision(
                    lane_id=lane.lane_id,
                    action="execute" if self.execute else "dry_run",
                    reason="lane accepted by queue and concurrency gates",
                    commands=commands,
                )
            )
        return decisions

    def apply_decisions(self, decisions: list[LaneDecision]) -> None:
        for decision in decisions:
            self.emit(
                "decision",
                f"{decision.lane_id}: {decision.action}",
                reason=decision.reason,
                commands=decision.commands,
            )
            if decision.action != "execute":
                continue
            for command in decision.commands:
                result = self.runner.run(command, timeout=180)
                self.emit(
                    "command",
                    "executed command",
                    args=command,
                    returncode=result.returncode,
                    timed_out=result.timed_out,
                    stdout_tail=result.stdout[-2000:],
                    stderr_tail=result.stderr[-2000:],
                )
                if result.returncode != 0:
                    break

    def run_once(self) -> ConductorResult:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = self.repo_root / self.mission.output_dir / _slug(self.mission.name) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self.emit(
            "start", "goal conductor run started", mission=self.mission.name, execute=self.execute
        )
        snapshot = self.snapshot()
        gates = self.hard_gates(snapshot)
        for gate in gates:
            self.emit("hard_gate", gate)
        if self.execute and gates:
            decisions = [
                LaneDecision(
                    lane_id=lane.lane_id,
                    action="blocked",
                    reason=f"fatal hard gate: {'; '.join(gates)}",
                )
                for lane in self.mission.lanes
            ]
        else:
            decisions = self.plan_lanes(snapshot, run_dir)
        self.apply_decisions(decisions)
        jsonl_path = run_dir / "conductor.jsonl"
        markdown_path = run_dir / "handoff.md"
        result = ConductorResult(
            mission_name=self.mission.name,
            execute=self.execute,
            snapshot=snapshot,
            decisions=decisions,
            hard_gates=gates,
            jsonl_path=jsonl_path,
            markdown_path=markdown_path,
        )
        self._write_outputs(result)
        return result

    def run_loop(
        self,
        *,
        max_cycles: int,
        interval_seconds: float,
        stop_on_hard_gate: bool = True,
    ) -> list[ConductorResult]:
        """Run repeated goal cycles with explicit finite bounds."""
        results: list[ConductorResult] = []
        for cycle in range(1, max_cycles + 1):
            self.emit("loop", "starting cycle", cycle=cycle, max_cycles=max_cycles)
            result = self.run_once()
            results.append(result)
            if result.hard_gates and stop_on_hard_gate:
                self.emit(
                    "loop", "stopping on hard gate", cycle=cycle, hard_gates=result.hard_gates
                )
                break
            if cycle < max_cycles and interval_seconds > 0:
                time.sleep(interval_seconds)
        return results

    def _write_outputs(self, result: ConductorResult) -> None:
        result.jsonl_path.write_text(
            "\n".join(event.to_json() for event in self.events) + "\n",
            encoding="utf-8",
        )
        lines = [
            f"# Goal conductor handoff — {result.mission_name}",
            "",
            f"- Generated: {_utc_now()}",
            f"- Mode: {'execute' if result.execute else 'dry-run'}",
            f"- Open PRs: {result.snapshot['open_pr_count']}/{self.mission.limits.queue_cap}",
            f"- Root dirty files: {result.snapshot['root']['dirty_file_count']}",
        ]
        if self.mission.objective:
            lines.append(f"- Objective: {self.mission.objective}")
        if self.mission.stop_condition:
            lines.append(f"- Stop condition: {self.mission.stop_condition}")
        if self.mission.checkpoints:
            lines.extend(["", "## Checkpoints", ""])
            lines.extend(f"- {checkpoint}" for checkpoint in self.mission.checkpoints)
        if self.mission.external_references:
            lines.extend(["", "## External References", ""])
            lines.extend(f"- {reference}" for reference in self.mission.external_references)
        lines.extend(["", "## Hard gates", ""])
        if result.hard_gates:
            lines.extend(f"- {gate}" for gate in result.hard_gates)
        else:
            lines.append("- None")
        lines.extend(["", "## Lane decisions", ""])
        for decision in result.decisions:
            lines.append(f"- `{decision.lane_id}`: {decision.action} — {decision.reason}")
            for command in decision.commands:
                lines.append(f"  - `{' '.join(command)}`")
        lines.extend(["", "## Open PRs", ""])
        for pr in result.snapshot.get("open_prs", []):
            title = str(pr.get("title", ""))
            lines.append(
                f"- #{pr.get('number')} draft={pr.get('isDraft')} "
                f"state={pr.get('mergeStateStatus')} — {title}"
            )
        result.markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_mission(path: Path) -> Mission:
    return Mission.from_dict(load_yaml(path))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "snapshot", "run-once", "loop"))
    parser.add_argument("--mission", type=Path, required=True, help="Mission YAML file")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch/send agents. Default is dry-run.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=3,
        help="Maximum cycles for loop mode.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=300.0,
        help="Sleep between loop cycles.",
    )
    parser.add_argument(
        "--continue-on-hard-gate",
        action="store_true",
        help="Do not stop loop mode when a hard gate is detected.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    mission = load_mission(args.mission)
    conductor = GoalConductor(mission=mission, repo_root=args.repo_root, execute=args.execute)
    if args.command == "validate":
        payload = {
            "ok": True,
            "mission": mission.name,
            "objective": mission.objective,
            "stop_condition": mission.stop_condition,
            "checkpoints": mission.checkpoints,
            "external_references": mission.external_references,
            "lanes": [asdict(lane) for lane in mission.lanes],
            "limits": asdict(mission.limits),
            "collect_merge_packets": mission.collect_merge_packets,
            "max_merge_packets": mission.max_merge_packets,
        }
        print(json.dumps(payload, indent=2) if args.json else f"mission ok: {mission.name}")
        return 0
    if args.command == "snapshot":
        snapshot = conductor.snapshot()
        print(json.dumps(snapshot, indent=2) if args.json else json.dumps(snapshot, indent=2))
        return 0
    if args.command == "loop":
        results = conductor.run_loop(
            max_cycles=args.max_cycles,
            interval_seconds=args.interval_seconds,
            stop_on_hard_gate=not args.continue_on_hard_gate,
        )
        payload = {
            "mission": mission.name,
            "execute": args.execute,
            "cycles": len(results),
            "results": [
                {
                    "open_pr_count": result.snapshot["open_pr_count"],
                    "hard_gates": result.hard_gates,
                    "decisions": [asdict(decision) for decision in result.decisions],
                    "jsonl_path": str(result.jsonl_path),
                    "markdown_path": str(result.markdown_path),
                }
                for result in results
            ],
        }
        print(json.dumps(payload, indent=2) if args.json else f"cycles: {len(results)}")
        return 0
    result = conductor.run_once()
    payload = {
        "mission": result.mission_name,
        "execute": result.execute,
        "open_pr_count": result.snapshot["open_pr_count"],
        "hard_gates": result.hard_gates,
        "decisions": [asdict(decision) for decision in result.decisions],
        "jsonl_path": str(result.jsonl_path),
        "markdown_path": str(result.markdown_path),
    }
    print(json.dumps(payload, indent=2) if args.json else f"handoff: {result.markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
