"""Tests for the boss-loop launchd probe smoke helper."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import probe_boss_loop_launchd as probe


SAMPLE_OUTPUT = """gui/501/com.aragora.swarm-boss-loop = {
\tactive count = 0
\tpath = /Users/armand/Library/LaunchAgents/com.aragora.swarm-boss-loop.plist
\ttype = LaunchAgent
\tstate = spawn scheduled

\tprogram = /bin/bash
\targuments = {
\t\t/bin/bash
\t\t-lc
\t\tcd /tmp && exec python3 -m something
\t}

\tworking directory = /Users/armand/Development/aragora

\tstdout path = /tmp/log
\tstderr path = /tmp/log
\tinherited environment = {
\t\tOPENAI_API_KEY => sk-PRETEND-SECRET-MUST-NOT-LEAK
\t\tANTHROPIC_API_KEY => sk-ant-PRETEND-SECRET
\t}

\tdefault environment = {
\t}

\t\tOSLogRateLimit => 64
\t}

\tdomain = gui/501 [100024]
\tasid = 100024
\tminimum runtime = 300
\texit timeout = 5
\truns = 224
\tlast exit code = 0

\tspawn type = daemon (3)
\tproperties = keepalive | runatload | inferred program
}
"""


def test_parse_launchd_state_extracts_safe_fields() -> None:
    state = probe.parse_launchd_state(SAMPLE_OUTPUT, label="com.aragora.swarm-boss-loop")
    assert state is not None
    assert state.label == "com.aragora.swarm-boss-loop"
    assert state.state == "spawn scheduled"
    assert state.active_count == 0
    assert state.runs == 224
    assert state.last_exit_code == 0
    assert state.minimum_runtime_seconds == 300
    assert state.is_spawn_scheduled is True
    assert state.is_running is False


def test_parse_launchd_state_redacts_inherited_environment() -> None:
    state = probe.parse_launchd_state(SAMPLE_OUTPUT, label="com.aragora.swarm-boss-loop")
    serialized = json.dumps(state.to_safe_dict())
    assert "PRETEND-SECRET" not in serialized
    assert "sk-" not in serialized
    assert "OPENAI_API_KEY" not in serialized


def test_parse_launchd_state_returns_none_when_unloaded() -> None:
    text = 'Could not find service "com.aragora.swarm-boss-loop" in domain for port'
    assert probe.parse_launchd_state(text, label="com.aragora.swarm-boss-loop") is None


def test_read_launchd_state_returns_none_on_print_failure() -> None:
    completed = subprocess.CompletedProcess(
        args=["launchctl", "print"], returncode=113, stdout="", stderr="not loaded"
    )
    with patch.object(probe.subprocess, "run", return_value=completed):
        assert probe.read_launchd_state("com.aragora.swarm-boss-loop") is None


def test_read_launchd_state_handles_print_timeout() -> None:
    with patch.object(
        probe.subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired(cmd=["launchctl"], timeout=5),
    ):
        assert probe.read_launchd_state("com.aragora.swarm-boss-loop", timeout_seconds=5) is None


def test_bounded_kickstart_succeeds_when_state_transitions_to_running() -> None:
    running_state = probe.LaunchdState(
        label="x",
        state="running",
        active_count=1,
        runs=10,
        last_exit_code=None,
        minimum_runtime_seconds=300,
        exit_timeout_seconds=5,
        spawn_type="daemon",
    )
    completed = subprocess.CompletedProcess(
        args=["launchctl", "kickstart"], returncode=0, stdout="", stderr=""
    )
    with (
        patch.object(probe.subprocess, "run", return_value=completed),
        patch.object(probe, "read_launchd_state", side_effect=[running_state]),
        patch.object(probe.time, "sleep", lambda _: None),
    ):
        result = probe.bounded_kickstart(
            "x", kickstart_timeout_seconds=5, wait_seconds=5, poll_interval_seconds=0.01
        )
    assert result.ok is True
    assert result.state is running_state
    assert "spawn scheduled" not in result.reason


def test_bounded_kickstart_fails_closed_when_stuck_spawn_scheduled() -> None:
    stuck_state = probe.LaunchdState(
        label="x",
        state="spawn scheduled",
        active_count=0,
        runs=224,
        last_exit_code=0,
        minimum_runtime_seconds=300,
        exit_timeout_seconds=5,
        spawn_type="daemon",
    )
    completed = subprocess.CompletedProcess(
        args=["launchctl", "kickstart"], returncode=0, stdout="", stderr=""
    )
    with (
        patch.object(probe.subprocess, "run", return_value=completed),
        patch.object(probe, "read_launchd_state", return_value=stuck_state),
        patch.object(probe.time, "sleep", lambda _: None),
    ):
        result = probe.bounded_kickstart(
            "x", kickstart_timeout_seconds=5, wait_seconds=0.05, poll_interval_seconds=0.01
        )
    assert result.ok is False
    assert "spawn scheduled" in result.reason
    assert "224" in result.reason
    assert result.state is stuck_state
    assert result.elapsed_seconds >= 0


def test_bounded_kickstart_treats_kickstart_timeout_as_failure() -> None:
    with (
        patch.object(
            probe.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd=["launchctl"], timeout=5),
        ),
        patch.object(probe, "read_launchd_state", return_value=None),
    ):
        result = probe.bounded_kickstart(
            "x", kickstart_timeout_seconds=5, wait_seconds=0.05, poll_interval_seconds=0.01
        )
    assert result.ok is False
    assert "timed out" in result.reason.lower() or "timeout" in result.reason.lower()


def test_bounded_kickstart_fails_when_label_not_loaded() -> None:
    completed = subprocess.CompletedProcess(
        args=["launchctl", "kickstart"], returncode=113, stdout="", stderr="No such process"
    )
    with (
        patch.object(probe.subprocess, "run", return_value=completed),
        patch.object(probe, "read_launchd_state", return_value=None),
    ):
        result = probe.bounded_kickstart(
            "x", kickstart_timeout_seconds=5, wait_seconds=0.05, poll_interval_seconds=0.01
        )
    assert result.ok is False
    assert result.state is None


def test_main_exits_zero_on_success(capsys: pytest.CaptureFixture[str]) -> None:
    success = probe.RestartProbeResult(
        ok=True,
        reason="kickstart confirmed (state=running, active=1)",
        state=None,
        elapsed_seconds=1.2,
    )
    with patch.object(probe, "bounded_kickstart", return_value=success):
        rc = probe.main(["--label", "x"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True


def test_main_exits_nonzero_on_stuck(capsys: pytest.CaptureFixture[str]) -> None:
    stuck_state = probe.LaunchdState(
        label="x",
        state="spawn scheduled",
        active_count=0,
        runs=224,
        last_exit_code=0,
        minimum_runtime_seconds=300,
        exit_timeout_seconds=5,
        spawn_type="daemon",
    )
    failure = probe.RestartProbeResult(
        ok=False,
        reason="LaunchAgent stuck in state=spawn scheduled (runs=224, last_exit=0)",
        state=stuck_state,
        elapsed_seconds=2.5,
    )
    with patch.object(probe, "bounded_kickstart", return_value=failure):
        rc = probe.main(["--label", "x"])
    assert rc != 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["state"]["state"] == "spawn scheduled"
    assert payload["state"]["runs"] == 224
    serialized = json.dumps(payload)
    assert "OPENAI_API_KEY" not in serialized
    assert "ANTHROPIC_API_KEY" not in serialized
