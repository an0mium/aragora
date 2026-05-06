"""Optional adapters for local Agent Flywheel-style tooling.

This module deliberately treats Flywheel repositories as external tools. It
does not import, vendor, install, or require any of them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


Runner = Callable[..., subprocess.CompletedProcess[str]]
Which = Callable[[str], str | None]


class FlywheelToolError(RuntimeError):
    """Raised when a guarded Flywheel tool invocation is unsafe or fails."""


@dataclass(frozen=True)
class FlywheelToolSpec:
    """Static metadata used to detect an optional external tool."""

    name: str
    category: str
    description: str
    commands: tuple[str, ...]
    repo_url: str
    license_url: str
    marker_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class FlywheelToolStatus:
    """Read-only local detection result for one optional external tool."""

    name: str
    category: str
    description: str
    available: bool
    executable: str | None
    matched_command: str | None
    candidate_commands: tuple[str, ...]
    marker_paths_found: tuple[str, ...]
    version: str | None
    help_excerpt: str | None
    repo_url: str
    license_url: str
    detection_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_TOOL_SPECS: tuple[FlywheelToolSpec, ...] = (
    FlywheelToolSpec(
        name="agentic_coding_flywheel_setup",
        category="bootstrap",
        description="ACFS bootstrap repository and installer patterns",
        commands=("acfs",),
        repo_url="https://github.com/Dicklesworthstone/agentic_coding_flywheel_setup",
        license_url=(
            "https://github.com/Dicklesworthstone/agentic_coding_flywheel_setup/blob/main/LICENSE"
        ),
        marker_paths=(
            "~/.aragora/flywheel-lab/agentic_coding_flywheel_setup",
            "~/agentic_coding_flywheel_setup",
        ),
    ),
    FlywheelToolSpec(
        name="ntm",
        category="session-orchestration",
        description="tmux swarm/session management patterns",
        commands=("ntm",),
        repo_url="https://github.com/Dicklesworthstone/ntm",
        license_url="https://github.com/Dicklesworthstone/ntm/blob/main/LICENSE",
    ),
    FlywheelToolSpec(
        name="mcp_agent_mail",
        category="coordination",
        description="agent inbox/outbox and advisory file reservation patterns",
        commands=("agent-mail", "agent_mail", "am"),
        repo_url="https://github.com/Dicklesworthstone/mcp_agent_mail",
        license_url="https://github.com/Dicklesworthstone/mcp_agent_mail/blob/main/LICENSE",
    ),
    FlywheelToolSpec(
        name="coding_agent_session_search",
        category="memory",
        description="cross-agent transcript and session search patterns",
        commands=("coding-agent-session-search", "coding_agent_session_search", "cass"),
        repo_url="https://github.com/Dicklesworthstone/coding_agent_session_search",
        license_url=(
            "https://github.com/Dicklesworthstone/coding_agent_session_search/blob/main/LICENSE"
        ),
    ),
    FlywheelToolSpec(
        name="cass_memory_system",
        category="memory",
        description="procedural memory and command memory patterns",
        commands=("cass-memory", "cass_memory", "cm"),
        repo_url="https://github.com/Dicklesworthstone/cass_memory_system",
        license_url="https://github.com/Dicklesworthstone/cass_memory_system/blob/main/LICENSE",
    ),
    FlywheelToolSpec(
        name="beads",
        category="task-graph",
        description="task graph prioritization and dependency tracking patterns",
        commands=("beads", "br", "bv"),
        repo_url="https://github.com/Dicklesworthstone/beads_rust",
        license_url="https://github.com/Dicklesworthstone/beads_rust/blob/main/LICENSE",
    ),
    FlywheelToolSpec(
        name="destructive_command_guard",
        category="safety",
        description="guardrails for destructive shell commands",
        commands=("destructive-command-guard", "dcg"),
        repo_url="https://github.com/Dicklesworthstone/destructive_command_guard",
        license_url="https://github.com/Dicklesworthstone/destructive_command_guard/blob/main/LICENSE",
    ),
    FlywheelToolSpec(
        name="slb",
        category="safety",
        description="safety lockbox and two-person-rule command patterns",
        commands=("slb",),
        repo_url="https://github.com/Dicklesworthstone/slb",
        license_url="https://github.com/Dicklesworthstone/slb/blob/main/LICENSE",
    ),
    FlywheelToolSpec(
        name="ultimate_bug_scanner",
        category="analysis",
        description="multi-tool bug scanning workflow patterns",
        commands=("ultimate-bug-scanner", "ubs"),
        repo_url="https://github.com/Dicklesworthstone/ultimate_bug_scanner",
        license_url="https://github.com/Dicklesworthstone/ultimate_bug_scanner/blob/main/LICENSE",
    ),
)

DEFAULT_ALLOWED_BINARIES: frozenset[str] = frozenset(
    command for spec in DEFAULT_TOOL_SPECS for command in spec.commands
)


def _excerpt(value: str, *, limit: int = 1200) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _expand_marker_paths(paths: Sequence[str]) -> tuple[str, ...]:
    found: list[str] = []
    for raw_path in paths:
        path = Path(os.path.expandvars(os.path.expanduser(raw_path)))
        if path.exists():
            found.append(str(path))
    return tuple(found)


def _run_probe_command(
    executable: str,
    args: Sequence[str],
    *,
    runner: Runner,
    timeout_seconds: float,
) -> str | None:
    try:
        completed = runner(
            [executable, *args],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    output = (completed.stdout or completed.stderr or "").strip()
    if not output:
        return None
    return _excerpt(output)


def probe_flywheel_tools(
    specs: Sequence[FlywheelToolSpec] = DEFAULT_TOOL_SPECS,
    *,
    include_help: bool = True,
    timeout_seconds: float = 2.0,
    which: Which = shutil.which,
    runner: Runner = subprocess.run,
) -> list[FlywheelToolStatus]:
    """Inspect optional local Flywheel-style tools without installing anything."""

    statuses: list[FlywheelToolStatus] = []
    for spec in specs:
        executable: str | None = None
        matched_command: str | None = None
        detection_error: str | None = None

        for command in spec.commands:
            try:
                executable = which(command)
            except OSError as exc:
                detection_error = f"which_failed:{type(exc).__name__}"
                executable = None
            if executable:
                matched_command = command
                break

        marker_paths_found = _expand_marker_paths(spec.marker_paths)
        version = None
        help_excerpt = None
        if executable:
            version = _run_probe_command(
                executable, ("--version",), runner=runner, timeout_seconds=timeout_seconds
            )
            if include_help:
                help_excerpt = _run_probe_command(
                    executable, ("--help",), runner=runner, timeout_seconds=timeout_seconds
                )

        statuses.append(
            FlywheelToolStatus(
                name=spec.name,
                category=spec.category,
                description=spec.description,
                available=bool(executable or marker_paths_found),
                executable=executable,
                matched_command=matched_command,
                candidate_commands=spec.commands,
                marker_paths_found=marker_paths_found,
                version=version,
                help_excerpt=help_excerpt,
                repo_url=spec.repo_url,
                license_url=spec.license_url,
                detection_error=detection_error,
            )
        )
    return statuses


def summarize_probe(statuses: Sequence[FlywheelToolStatus]) -> dict[str, Any]:
    """Return a compact summary suitable for CLI JSON output."""

    available = [status.name for status in statuses if status.available]
    return {
        "tool_count": len(statuses),
        "available_count": len(available),
        "available_tools": available,
        "missing_tools": [status.name for status in statuses if not status.available],
        "categories": sorted({status.category for status in statuses}),
    }


def run_json_tool(
    command: Sequence[str],
    *,
    allowed_binaries: Collection[str] | None = None,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float = 10.0,
    which: Which = shutil.which,
    runner: Runner = subprocess.run,
) -> Any:
    """Run an allowlisted external tool command that is expected to emit JSON.

    The guard intentionally rejects shell strings and path-qualified binaries.
    This adapter is for narrow local integrations, not arbitrary command
    execution.
    """

    if not command:
        raise FlywheelToolError("command is required")

    binary = command[0]
    if Path(binary).name != binary:
        raise FlywheelToolError("path-qualified binaries are not allowed")

    allowed = set(allowed_binaries or DEFAULT_ALLOWED_BINARIES)
    if binary not in allowed:
        raise FlywheelToolError(f"binary is not allowlisted: {binary}")

    executable = which(binary)
    if not executable:
        raise FlywheelToolError(f"binary not found on PATH: {binary}")

    try:
        completed = runner(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            cwd=str(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
        )
    except subprocess.TimeoutExpired as exc:
        raise FlywheelToolError(f"command timed out: {binary}") from exc
    except OSError as exc:
        raise FlywheelToolError(f"command failed to start: {binary}") from exc

    if completed.returncode != 0:
        stderr = _excerpt(completed.stderr or completed.stdout or "")
        raise FlywheelToolError(f"command exited {completed.returncode}: {stderr}")

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FlywheelToolError(f"command did not emit valid JSON: {binary}") from exc
