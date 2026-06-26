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
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path(".aragora/goal-conductor")
DEFAULT_QUEUE_CAP = 6
DEFAULT_MAX_IMPLEMENTATION_LANES = 2
DEFAULT_MAX_REVIEW_LANES = 1
DEFAULT_MAX_NEW_LANES_PER_CYCLE = 2
MERGE_POLICY_REPORT_ONLY = "report_only"
MERGE_POLICY_EXACT_GATED_TIER_0_2 = "exact_gated_tier_0_2"
MERGE_POLICIES = {MERGE_POLICY_REPORT_ONLY, MERGE_POLICY_EXACT_GATED_TIER_0_2}
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


def _mute_stdout_after_broken_pipe() -> bool:
    """Redirect stdout's fd after a real broken pipe without mutating wrappers."""

    current = sys.stdout
    fileno = getattr(current, "fileno", None)
    if not callable(fileno):
        return False
    try:
        stdout_fd = fileno()
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull_fd, stdout_fd)
        finally:
            os.close(devnull_fd)
    except (AttributeError, OSError, ValueError):
        return False
    return True


def _emit_output(output: str) -> None:
    stream = sys.stdout
    if stream is None:
        return
    write = getattr(stream, "write", None)
    if not callable(write):
        return
    try:
        write(output)
        write("\n")
        flush = getattr(stream, "flush", None)
        if callable(flush):
            flush()
    except BrokenPipeError:
        _mute_stdout_after_broken_pipe()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_str_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value)]


def _as_nonempty_str_list(*values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in _as_str_list(value):
            text = item.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
    return out


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
    task_id: str = ""
    claimed_paths: list[str] = field(default_factory=list)
    write_scopes: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    forbidden_paths: list[str] = field(default_factory=list)
    stop_rule: str = ""
    agents_spec: str = "heterogeneous"
    context_file: str = ""
    round_id: str = ""
    strict_launch_verify: bool = False

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
        strict_launch_verify = payload.get(
            "strict_launch_verify",
            payload.get("strict_verify", False),
        )
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
            task_id=str(payload.get("task_id") or payload.get("task") or lane_id).strip(),
            claimed_paths=_as_nonempty_str_list(
                payload.get("claimed_path"),
                payload.get("claimed_paths"),
                payload.get("file_scope"),
                payload.get("allowed_paths"),
            ),
            write_scopes=_as_nonempty_str_list(
                payload.get("write_scope"),
                payload.get("write_scopes"),
            ),
            tests=_as_nonempty_str_list(
                payload.get("test"),
                payload.get("tests"),
                payload.get("validation_command"),
                payload.get("validation_commands"),
            ),
            forbidden_paths=_as_nonempty_str_list(
                payload.get("forbidden_path"),
                payload.get("forbidden_paths"),
            ),
            stop_rule=str(payload.get("stop_rule") or payload.get("stop_condition") or "").strip(),
            agents_spec=str(payload.get("agents_spec") or "heterogeneous").strip(),
            context_file=str(payload.get("context_file") or "").strip(),
            round_id=str(payload.get("round_id") or "").strip(),
            strict_launch_verify=_truthy(strict_launch_verify),
        )

    @property
    def mutates(self) -> bool:
        return self.mode in MUTATING_LANE_MODES

    @property
    def is_review(self) -> bool:
        return self.mode in REVIEW_MODES

    @property
    def has_lease_scope(self) -> bool:
        return bool(self.claimed_paths or self.write_scopes)

    def lease_blocker(self) -> str | None:
        if self.agent == "codex" and self.autonomous and not self.has_lease_scope:
            return (
                "autonomous Codex lane requires task_id plus at least one "
                "claimed_path or write_scope"
            )
        return None


@dataclass(frozen=True)
class MissionLimits:
    queue_cap: int = DEFAULT_QUEUE_CAP
    max_implementation_lanes: int = DEFAULT_MAX_IMPLEMENTATION_LANES
    max_review_lanes: int = DEFAULT_MAX_REVIEW_LANES
    max_new_lanes_per_cycle: int = DEFAULT_MAX_NEW_LANES_PER_CYCLE

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MissionLimits":
        return cls(
            queue_cap=int(payload.get("queue_cap", DEFAULT_QUEUE_CAP)),
            max_implementation_lanes=int(
                payload.get("max_implementation_lanes", DEFAULT_MAX_IMPLEMENTATION_LANES)
            ),
            max_review_lanes=int(payload.get("max_review_lanes", DEFAULT_MAX_REVIEW_LANES)),
            max_new_lanes_per_cycle=int(
                payload.get("max_new_lanes_per_cycle", DEFAULT_MAX_NEW_LANES_PER_CYCLE)
            ),
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
    merge_policy: str = MERGE_POLICY_REPORT_ONLY
    merge_on_green_max_tier: int = 2
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
        merge_policy = str(payload.get("merge_policy") or MERGE_POLICY_REPORT_ONLY).strip()
        if merge_policy not in MERGE_POLICIES:
            raise ValueError(f"mission merge_policy must be one of {sorted(MERGE_POLICIES)}")
        merge_on_green_max_tier = int(payload.get("merge_on_green_max_tier", 2))
        if not 0 <= merge_on_green_max_tier <= 2:
            raise ValueError("merge_on_green_max_tier must be in [0, 2]")
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
            merge_policy=merge_policy,
            merge_on_green_max_tier=merge_on_green_max_tier,
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


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}
    return bool(value)


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _merge_packet_pr_number(entry: dict[str, Any]) -> int | None:
    for key in ("pr_number", "number", "pr"):
        value = entry.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _merge_packet_head(entry: dict[str, Any]) -> str:
    for key in ("head_sha", "headRefOid", "head"):
        value = str(entry.get(key) or "").strip()
        if value:
            return value
    return ""


def _merge_packet_satisfied(entry: dict[str, Any]) -> bool:
    status = str(entry.get("status") or "").strip().lower()
    verdict = str(entry.get("verdict") or "").strip().lower()
    return (
        status == "satisfied"
        and verdict == "admin_squash_allowed"
        and _truthy(entry.get("admin_squash_allowed"))
    )


def _is_queue_cap_gate(gate: str) -> bool:
    return gate.startswith("open PR queue at/above cap")


def _fatal_hard_gates(gates: list[str]) -> list[str]:
    return [gate for gate in gates if not _is_queue_cap_gate(gate)]


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


def _merge_packet_admin_squash_order(packet: Any) -> list[int]:
    if not isinstance(packet, dict):
        return []
    order = packet.get("admin_squash_order")
    if not isinstance(order, list):
        return []
    ordered: list[int] = []
    for item in order:
        try:
            ordered.append(int(item))
        except (TypeError, ValueError):
            continue
    return ordered


def _merge_packet_not_ready(packet: Any) -> set[int]:
    if not isinstance(packet, dict):
        return set()
    not_ready = packet.get("not_ready")
    if not isinstance(not_ready, list):
        return set()
    blocked: set[int] = set()
    for item in not_ready:
        try:
            blocked.add(int(item))
        except (TypeError, ValueError):
            continue
    return blocked


def _snapshot_open_prs(snapshot: dict[str, Any]) -> dict[int, dict[str, Any]]:
    open_prs: dict[int, dict[str, Any]] = {}
    for item in snapshot.get("open_prs") or []:
        if not isinstance(item, dict):
            continue
        number = _coerce_int(item.get("number"))
        if number is None:
            continue
        state = str(item.get("state") or "OPEN").strip().upper()
        if state not in {"", "OPEN"}:
            continue
        if _truthy(item.get("isDraft")) or _truthy(item.get("draft")):
            continue
        open_prs[number] = item
    return open_prs


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

    def exact_gated_merge_candidates(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        """Return Tier 0-2 merge candidates that still require helper confirmation.

        This is only a fast local filter over merge-packet JSON. Execute mode
        still calls ``scripts/settle_one_pr.py`` immediately before merging.
        """
        merge_packet = snapshot.get("merge_packets")
        admin_order = _merge_packet_admin_squash_order(merge_packet)
        not_ready = _merge_packet_not_ready(merge_packet)
        if not admin_order:
            return []
        open_prs = _snapshot_open_prs(snapshot)
        candidates: list[dict[str, Any]] = []
        entries_by_pr: dict[int, dict[str, Any]] = {}
        for entry in _merge_packet_entries(merge_packet):
            pr_number = _merge_packet_pr_number(entry)
            head_sha = _merge_packet_head(entry)
            if pr_number is None or not head_sha:
                continue
            entries_by_pr[pr_number] = entry
        for pr_number in admin_order:
            if pr_number in not_ready:
                continue
            entry = entries_by_pr.get(pr_number)
            if entry is None:
                continue
            if pr_number not in open_prs:
                continue
            head_sha = _merge_packet_head(entry)
            tier = _coerce_int(entry.get("tier"))
            if not head_sha:
                continue
            if tier is None:
                continue
            if tier > self.mission.merge_on_green_max_tier:
                continue
            if _truthy(entry.get("requires_human_risk_settlement")):
                continue
            if _truthy(entry.get("unresolved_dissent")):
                continue
            if not _merge_packet_satisfied(entry):
                continue
            candidates.append({"pr_number": pr_number, "head_sha": head_sha, "tier": tier})
        return candidates

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
            tier = _coerce_int(entry.get("tier"))
            if tier is None:
                continue
            if tier >= 4 or _truthy(entry.get("requires_human_risk_settlement")):
                pr_number = entry.get("pr_number", "?")
                tier_name = entry.get("tier_name") or f"tier_{tier}"
                gates.append(f"human/non-author settlement gate present: #{pr_number} {tier_name}")
            if _truthy(entry.get("unresolved_dissent")):
                pr_number = entry.get("pr_number", "?")
                gates.append(f"unresolved model dissent present: #{pr_number}")
        return gates

    def _prompt_file_for(self, lane: LaneSpec, run_dir: Path) -> Path:
        if lane.prompt_file:
            path = Path(lane.prompt_file)
            return path if path.is_absolute() else self.repo_root / path
        prompt_dir = run_dir / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        path = prompt_dir / f"{_slug(lane.lane_id)}.md"
        path.write_text(self._render_lane_prompt(lane) + "\n", encoding="utf-8")
        return path

    def _render_lane_prompt(self, lane: LaneSpec) -> str:
        lines = [
            "Start from live repo truth. Do not rely on transcript state.",
            "Operating contract: re-read docs/AGENT_OPERATING_CONTRACT.md §Conductor this cycle.",
            "",
            "Mission lane contract:",
            f"- lane_id: {lane.lane_id}",
            f"- task_id: {lane.task_id or lane.lane_id}",
            f"- goal: {lane.goal}",
            f"- mode: {lane.mode}",
            "- allowed autonomy: Tier 0-2 only; Tier 3/4 settlement and approval-required surfaces are hard stops.",
            "- stop condition: open one draft PR, post one exact blocker, or park the lane with a durable handoff.",
            "- never merge, never use --admin, never touch branch protection/workflows/secrets.",
        ]
        if lane.claimed_paths:
            lines.extend(["- claimed paths:"] + [f"  - {path}" for path in lane.claimed_paths])
        if lane.write_scopes:
            lines.extend(["- write scopes:"] + [f"  - {scope}" for scope in lane.write_scopes])
        if lane.forbidden_paths:
            lines.extend(["- forbidden paths:"] + [f"  - {path}" for path in lane.forbidden_paths])
        if lane.tests:
            lines.extend(["- validation commands:"] + [f"  - {command}" for command in lane.tests])
        if lane.stop_rule:
            lines.append(f"- lane-specific stop rule: {lane.stop_rule}")
        if lane.next_action:
            lines.append(f"- next action: {lane.next_action}")
        lines.extend(["", lane.prompt.strip()])
        return "\n".join(line for line in lines if line is not None).rstrip()

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
            launch.extend(
                [
                    "--file",
                    str(prompt_file),
                    "--timeout-seconds",
                    "180",
                    "--submit-verify-timeout",
                    "30",
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
            if lane.strict_launch_verify:
                launch.append("--strict-verify")
            if lane.agent == "codex":
                launch.extend(["--task-id", lane.task_id or lane.lane_id])
                for path in lane.claimed_paths:
                    launch.extend(["--claimed-path", path])
                for scope in lane.write_scopes:
                    launch.extend(["--write-scope", scope])
                for test_cmd in lane.tests:
                    launch.extend(["--test", test_cmd])
                for path in lane.forbidden_paths:
                    launch.extend(["--forbidden-path", path])
            commands.append(launch)
            return commands
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
        new_lanes_used = 0
        at_cap = snapshot["open_pr_count"] >= self.mission.limits.queue_cap
        decisions: list[LaneDecision] = []
        for lane in self.mission.lanes:
            lease_blocker = lane.lease_blocker()
            if lease_blocker:
                decisions.append(
                    LaneDecision(
                        lane_id=lane.lane_id,
                        action="blocked",
                        reason=lease_blocker,
                    )
                )
                continue
            if lane.mutates and at_cap:
                decisions.append(
                    LaneDecision(
                        lane_id=lane.lane_id,
                        action="blocked",
                        reason="queue cap reached; mutating implementation lanes are disabled",
                    )
                )
                continue
            if lane.lane_id in sessions and lane.agent == "codex" and lane.autonomous:
                decisions.append(
                    LaneDecision(
                        lane_id=lane.lane_id,
                        action="blocked",
                        reason=(
                            "existing autonomous Codex lanes are not reused; relaunch "
                            "through agent_bridge.py launch so the dev lease is freshly "
                            "claimed for the current mission scope"
                        ),
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
            if lane.lane_id not in sessions and lane.mode != PANEL_MODE:
                if new_lanes_used >= self.mission.limits.max_new_lanes_per_cycle:
                    decisions.append(
                        LaneDecision(
                            lane_id=lane.lane_id,
                            action="blocked",
                            reason="max new lanes per cycle already assigned",
                        )
                    )
                    continue
                new_lanes_used += 1
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

    def _maybe_apply_one_exact_gated_merge(self, snapshot: dict[str, Any]) -> LaneDecision | None:
        if self.mission.merge_policy != MERGE_POLICY_EXACT_GATED_TIER_0_2:
            return None
        candidates = self.exact_gated_merge_candidates(snapshot)
        if not candidates:
            return None
        candidate = candidates[0]
        pr_number = int(candidate["pr_number"])
        head_sha = str(candidate["head_sha"])
        settle_cmd = ["python3", "scripts/settle_one_pr.py", "--pr", str(pr_number), "--json"]
        merge_cmd = [
            "gh",
            "pr",
            "merge",
            str(pr_number),
            "--squash",
            "--match-head-commit",
            head_sha,
        ]
        commands = [settle_cmd, merge_cmd]
        if not self.execute:
            return LaneDecision(
                lane_id=f"merge-pr-{pr_number}",
                action="dry_run",
                reason=(
                    "exact-gated Tier 0-2 merge candidate; execute mode would "
                    "confirm settle_one_pr.py then run protected squash"
                ),
                commands=commands,
            )
        settle_result = self.runner.run(settle_cmd, timeout=180)
        if settle_result.returncode != 0:
            return LaneDecision(
                lane_id=f"merge-pr-{pr_number}",
                action="blocked",
                reason=f"settle_one_pr.py failed rc={settle_result.returncode}",
                commands=commands,
            )
        try:
            settle_payload = json.loads(settle_result.stdout or "{}")
        except json.JSONDecodeError:
            return LaneDecision(
                lane_id=f"merge-pr-{pr_number}",
                action="blocked",
                reason="settle_one_pr.py returned non-JSON output",
                commands=commands,
            )
        blockers = settle_payload.get("blockers") if isinstance(settle_payload, dict) else None
        settle_status = (
            str(settle_payload.get("status") or "").strip()
            if isinstance(settle_payload, dict)
            else ""
        )
        selected_pr = (
            _coerce_int(settle_payload.get("selected_pr"))
            if isinstance(settle_payload, dict)
            else None
        )
        settle_head = (
            str(settle_payload.get("head_sha") or head_sha)
            if isinstance(settle_payload, dict)
            else head_sha
        )
        if blockers or settle_status != "packet_authorized_dry_run" or settle_head != head_sha:
            reason = f"settle_one_pr.py blockers={blockers or []}"
            if not blockers and settle_status != "packet_authorized_dry_run":
                reason = f"settle_one_pr.py status={settle_status or '<missing>'}"
            if settle_head != head_sha:
                reason = f"head drift: packet {head_sha} settle_one {settle_head}"
            return LaneDecision(
                lane_id=f"merge-pr-{pr_number}",
                action="blocked",
                reason=reason,
                commands=commands,
            )
        if selected_pr is not None and selected_pr != pr_number:
            return LaneDecision(
                lane_id=f"merge-pr-{pr_number}",
                action="blocked",
                reason=f"settle_one_pr.py selected_pr={selected_pr}",
                commands=commands,
            )
        merge_result = self.runner.run(merge_cmd, timeout=180)
        if merge_result.returncode != 0:
            return LaneDecision(
                lane_id=f"merge-pr-{pr_number}",
                action="blocked",
                reason=f"protected squash merge failed rc={merge_result.returncode}",
                commands=commands,
            )
        return LaneDecision(
            lane_id=f"merge-pr-{pr_number}",
            action="merged",
            reason="exact-gated Tier 0-2 normal protected squash completed",
            commands=commands,
        )

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
        fatal_gates = _fatal_hard_gates(gates)
        if self.execute and fatal_gates:
            decisions = [
                LaneDecision(
                    lane_id=lane.lane_id,
                    action="blocked",
                    reason=f"fatal hard gate: {'; '.join(fatal_gates)}",
                )
                for lane in self.mission.lanes
            ]
        else:
            merge_decision = None
            if not (self.execute and fatal_gates):
                merge_decision = self._maybe_apply_one_exact_gated_merge(snapshot)
            decisions = self.plan_lanes(snapshot, run_dir)
            if merge_decision is not None:
                decisions.insert(0, merge_decision)
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
            fatal_gates = _fatal_hard_gates(result.hard_gates)
            if fatal_gates and stop_on_hard_gate:
                self.emit("loop", "stopping on hard gate", cycle=cycle, hard_gates=fatal_gates)
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
            "merge_policy": mission.merge_policy,
            "merge_on_green_max_tier": mission.merge_on_green_max_tier,
            "collect_merge_packets": mission.collect_merge_packets,
            "max_merge_packets": mission.max_merge_packets,
        }
        _emit_output(json.dumps(payload, indent=2) if args.json else f"mission ok: {mission.name}")
        return 0
    if args.command == "snapshot":
        snapshot = conductor.snapshot()
        _emit_output(json.dumps(snapshot, indent=2))
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
        _emit_output(json.dumps(payload, indent=2) if args.json else f"cycles: {len(results)}")
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
    _emit_output(json.dumps(payload, indent=2) if args.json else f"handoff: {result.markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
