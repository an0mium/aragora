from __future__ import annotations

import subprocess

import pytest

from aragora.integrations.flywheel import (
    FlywheelToolError,
    FlywheelToolSpec,
    probe_flywheel_tools,
    run_json_tool,
    summarize_probe,
)


def test_probe_reports_available_tool_with_version_and_help() -> None:
    spec = FlywheelToolSpec(
        name="ntm",
        category="session-orchestration",
        description="tmux manager",
        commands=("ntm",),
        repo_url="https://example.test/ntm",
        license_url="https://example.test/license",
    )

    def fake_which(command: str) -> str | None:
        return "/usr/local/bin/ntm" if command == "ntm" else None

    def fake_runner(args, **_kwargs):
        if args[-1] == "--version":
            return subprocess.CompletedProcess(args, 0, stdout="ntm 1.2.3\n", stderr="")
        if args[-1] == "--help":
            return subprocess.CompletedProcess(args, 0, stdout="usage: ntm\n", stderr="")
        raise AssertionError(args)

    statuses = probe_flywheel_tools((spec,), which=fake_which, runner=fake_runner)

    assert len(statuses) == 1
    status = statuses[0]
    assert status.available is True
    assert status.executable == "/usr/local/bin/ntm"
    assert status.matched_command == "ntm"
    assert status.version == "ntm 1.2.3"
    assert status.help_excerpt == "usage: ntm"
    assert summarize_probe(statuses)["available_tools"] == ["ntm"]


def test_probe_reports_missing_tool_without_running_commands() -> None:
    spec = FlywheelToolSpec(
        name="agent_mail",
        category="coordination",
        description="mail",
        commands=("agent-mail", "am"),
        repo_url="https://example.test/mail",
        license_url="https://example.test/license",
    )
    runner_calls = []

    def fake_runner(args, **_kwargs):
        runner_calls.append(args)
        raise AssertionError("runner should not be called when commands are missing")

    statuses = probe_flywheel_tools((spec,), which=lambda _command: None, runner=fake_runner)

    assert statuses[0].available is False
    assert statuses[0].executable is None
    assert statuses[0].candidate_commands == ("agent-mail", "am")
    assert runner_calls == []


def test_probe_can_detect_repository_marker_without_binary(tmp_path) -> None:
    marker = tmp_path / "agentic_coding_flywheel_setup"
    marker.mkdir()
    spec = FlywheelToolSpec(
        name="agentic_coding_flywheel_setup",
        category="bootstrap",
        description="bootstrap repo",
        commands=("acfs",),
        repo_url="https://example.test/acfs",
        license_url="https://example.test/license",
        marker_paths=(str(marker),),
    )

    statuses = probe_flywheel_tools((spec,), which=lambda _command: None)

    assert statuses[0].available is True
    assert statuses[0].matched_command is None
    assert statuses[0].marker_paths_found == (str(marker),)


def test_run_json_tool_executes_allowlisted_binary_and_parses_json() -> None:
    def fake_runner(args, **kwargs):
        assert args == ["/usr/local/bin/ntm", "sessions", "--json"]
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(args, 0, stdout='{"sessions": []}', stderr="")

    payload = run_json_tool(
        ("ntm", "sessions", "--json"),
        allowed_binaries={"ntm"},
        which=lambda command: "/usr/local/bin/ntm" if command == "ntm" else None,
        runner=fake_runner,
    )

    assert payload == {"sessions": []}


def test_run_json_tool_rejects_unallowlisted_or_path_qualified_binary() -> None:
    with pytest.raises(FlywheelToolError, match="not allowlisted"):
        run_json_tool(("rm", "-rf", "/"), allowed_binaries={"ntm"})

    with pytest.raises(FlywheelToolError, match="path-qualified"):
        run_json_tool(("/usr/local/bin/ntm", "--json"), allowed_binaries={"ntm"})


def test_run_json_tool_rejects_non_json_output() -> None:
    def fake_runner(args, **_kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="not json", stderr="")

    with pytest.raises(FlywheelToolError, match="valid JSON"):
        run_json_tool(
            ("ntm", "sessions"),
            allowed_binaries={"ntm"},
            which=lambda command: "/usr/local/bin/ntm" if command == "ntm" else None,
            runner=fake_runner,
        )
