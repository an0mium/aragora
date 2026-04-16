#!/usr/bin/env python3
"""Bounded, observable probe for the boss-loop launchd LaunchAgent.

Addresses #5997: `launchctl kickstart` can block indefinitely while launchd
holds a job in `state = spawn scheduled` during its `ThrottleInterval`
cooldown. This helper wraps kickstart with hard timeouts, then polls
`launchctl print` to surface a truthful failure reason (state, runs,
last_exit_code, throttle bounds) instead of leaving operators staring at a
hung shell.

Output is a JSON document on stdout; exit code 0 on confirmed restart,
non-zero otherwise. Inherited-environment values from `launchctl print`
(API keys, etc.) are never echoed.

Usage:
    python scripts/probe_boss_loop_launchd.py
    python scripts/probe_boss_loop_launchd.py --label com.aragora.swarm-boss-loop
    python scripts/probe_boss_loop_launchd.py --wait-seconds 60 --no-kickstart
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_LABEL = "com.aragora.swarm-boss-loop"

_INT_FIELDS: dict[str, str] = {
    "active count": "active_count",
    "runs": "runs",
    "last exit code": "last_exit_code",
    "minimum runtime": "minimum_runtime_seconds",
    "exit timeout": "exit_timeout_seconds",
}


@dataclass(frozen=True)
class LaunchdState:
    label: str
    state: str
    active_count: int
    runs: int
    last_exit_code: int | None
    minimum_runtime_seconds: int | None
    exit_timeout_seconds: int | None
    spawn_type: str | None

    @property
    def is_running(self) -> bool:
        return self.active_count > 0

    @property
    def is_spawn_scheduled(self) -> bool:
        return self.state.strip() == "spawn scheduled"

    def to_safe_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RestartProbeResult:
    ok: bool
    reason: str
    state: LaunchdState | None
    elapsed_seconds: float

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "state": self.state.to_safe_dict() if self.state else None,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def parse_launchd_state(text: str, *, label: str) -> LaunchdState | None:
    """Parse `launchctl print` output for a single LaunchAgent.

    Returns ``None`` when the agent is not loaded. Inherited-environment
    blocks are skipped entirely so secrets are never returned.
    """

    if not text or "Could not find service" in text:
        return None

    fields: dict[str, Any] = {}
    skip_block_depth = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if skip_block_depth > 0:
            if line.endswith("{"):
                skip_block_depth += 1
            elif line == "}":
                skip_block_depth -= 1
            continue
        if line.startswith("inherited environment") or line.startswith("default environment"):
            if line.endswith("{"):
                skip_block_depth = 1
            continue
        match = re.match(r"^([^=]+?)\s*=\s*(.+?)\s*$", line)
        if not match:
            continue
        key = match.group(1).strip().lower()
        value = match.group(2).strip().rstrip(",")
        if key in _INT_FIELDS:
            try:
                fields[_INT_FIELDS[key]] = int(value)
            except ValueError:
                continue
        elif key == "state":
            fields["state"] = value
        elif key == "spawn type":
            spawn = value.split(" ", 1)[0]
            fields["spawn_type"] = spawn or None

    if "state" not in fields:
        return None

    return LaunchdState(
        label=label,
        state=fields["state"],
        active_count=int(fields.get("active_count", 0)),
        runs=int(fields.get("runs", 0)),
        last_exit_code=fields.get("last_exit_code"),
        minimum_runtime_seconds=fields.get("minimum_runtime_seconds"),
        exit_timeout_seconds=fields.get("exit_timeout_seconds"),
        spawn_type=fields.get("spawn_type"),
    )


def read_launchd_state(label: str, *, timeout_seconds: float = 10.0) -> LaunchdState | None:
    """Run `launchctl print` and return parsed state, or ``None`` on failure."""

    uid = os.getuid()
    try:
        proc = subprocess.run(
            ["launchctl", "print", f"gui/{uid}/{label}"],
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    return parse_launchd_state(proc.stdout, label=label)


def _kickstart(label: str, *, timeout_seconds: float) -> tuple[bool, str]:
    uid = os.getuid()
    try:
        proc = subprocess.run(
            ["launchctl", "kickstart", f"gui/{uid}/{label}"],
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"launchctl kickstart timed out after {timeout_seconds:.0f}s for {label}"
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "non-zero exit"
        return False, f"launchctl kickstart failed for {label}: {detail}"
    return True, ""


def _format_stuck_reason(label: str, state: LaunchdState) -> str:
    parts = [
        f"LaunchAgent {label} stuck in state={state.state!r}",
        f"runs={state.runs}",
    ]
    if state.last_exit_code is not None:
        parts.append(f"last_exit={state.last_exit_code}")
    if state.minimum_runtime_seconds is not None:
        parts.append(f"min_runtime={state.minimum_runtime_seconds}s")
    return "; ".join(parts)


def bounded_kickstart(
    label: str,
    *,
    kickstart_timeout_seconds: float = 10.0,
    wait_seconds: float = 30.0,
    poll_interval_seconds: float = 1.0,
    skip_kickstart: bool = False,
) -> RestartProbeResult:
    """Kickstart ``label`` with bounded waits; return truthful state on stuck."""

    started = time.monotonic()
    kicked = True
    detail = ""
    if not skip_kickstart:
        kicked, detail = _kickstart(label, timeout_seconds=kickstart_timeout_seconds)

    deadline = started + wait_seconds + (kickstart_timeout_seconds if not skip_kickstart else 0.0)
    last_state: LaunchdState | None = None
    while True:
        last_state = read_launchd_state(label)
        if last_state is not None and last_state.is_running:
            return RestartProbeResult(
                ok=True,
                reason=(
                    f"kickstart confirmed for {label} "
                    f"(state={last_state.state!r}, active={last_state.active_count})"
                ),
                state=last_state,
                elapsed_seconds=time.monotonic() - started,
            )
        if time.monotonic() >= deadline:
            break
        time.sleep(poll_interval_seconds)

    if last_state is None:
        reason = detail or f"LaunchAgent {label} is not loaded or unreadable"
        return RestartProbeResult(
            ok=False,
            reason=reason,
            state=None,
            elapsed_seconds=time.monotonic() - started,
        )

    base_reason = _format_stuck_reason(label, last_state)
    if not kicked and detail:
        base_reason = f"{base_reason}; kickstart_error={detail}"
    return RestartProbeResult(
        ok=False,
        reason=base_reason,
        state=last_state,
        elapsed_seconds=time.monotonic() - started,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=DEFAULT_LABEL, help="LaunchAgent label to probe")
    parser.add_argument(
        "--kickstart-timeout-seconds",
        type=float,
        default=10.0,
        help="Hard timeout for the launchctl kickstart submission",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=30.0,
        help="Maximum time to wait for the agent to transition into a running state",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=1.0,
        help="Polling interval while waiting for state transition",
    )
    parser.add_argument(
        "--no-kickstart",
        action="store_true",
        help="Skip kickstart; only read and report current launchd state",
    )
    args = parser.parse_args(argv)

    result = bounded_kickstart(
        args.label,
        kickstart_timeout_seconds=args.kickstart_timeout_seconds,
        wait_seconds=args.wait_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        skip_kickstart=args.no_kickstart,
    )
    print(json.dumps(result.to_safe_dict(), indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
