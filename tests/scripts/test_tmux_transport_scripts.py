from __future__ import annotations

import json
import os
import shlex
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_fake_tmux(tmp_path: Path, *, window_name: str = "_control") -> Path:
    fake_tmux = tmp_path / "tmux"
    fake_tmux.write_text(
        f"""#!/usr/bin/env python3
import json
import os
import sys

log_path = os.environ["FAKE_TMUX_LOG"]
with open(log_path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\\n")

cmd = sys.argv[1:]
if cmd[:3] == ["has-session", "-t", "aragora"]:
    raise SystemExit(0)
if cmd[:3] == ["list-windows", "-t", "aragora"]:
    print("0 {window_name}")
    raise SystemExit(0)
if cmd[:2] == ["list-panes", "-t"]:
    print("0")
    raise SystemExit(0)
if cmd[:2] == ["new-window", "-P"]:
    print("@17")
    raise SystemExit(0)
if cmd[:2] in (["new-session", "-d"], ["pipe-pane", "-t"], ["load-buffer", "-"]):
    raise SystemExit(0)
if cmd[:2] == ["kill-window", "-t"]:
    raise SystemExit(0)
if cmd[:2] in (
    ["send-keys", "-t"],
    ["set-buffer", "-b"],
    ["paste-buffer", "-b"],
    ["paste-buffer", "-d"],
    ["delete-buffer", "-b"],
):
    raise SystemExit(0)

print(f"unexpected tmux command: {{cmd}}", file=sys.stderr)
raise SystemExit(1)
""",
        encoding="utf-8",
    )
    fake_tmux.chmod(fake_tmux.stat().st_mode | stat.S_IEXEC)
    return fake_tmux


def _fake_tmux_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["FAKE_TMUX_LOG"] = str(tmp_path / "tmux-calls.jsonl")
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env["HOME"] = str(tmp_path / "home")
    env["ARAGORA_TMUX_PASTE_SETTLE_SECONDS"] = "0"
    return env


def _load_tmux_calls(env: dict[str, str]) -> list[list[str]]:
    log_path = Path(env["FAKE_TMUX_LOG"])
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _heartbeat_launcher_from_calls(calls: list[list[str]], *, name: str) -> Path:
    for call in calls:
        if call[:2] != ["send-keys", "-t"] or len(call) < 4:
            continue
        command = call[3]
        if ".heartbeat-launch." not in command:
            continue
        argv = shlex.split(command)
        assert argv[0] == "bash"
        path = Path(argv[1])
        assert path.name.startswith(f"{name}.heartbeat-launch.")
        assert not path.name.endswith(".sh")
        return path
    raise AssertionError(f"no heartbeat launcher sent for {name}")


def _assignment_path(script_body: str, key: str) -> Path:
    prefix = f"{key}="
    for line in script_body.splitlines():
        if line.startswith(prefix):
            return Path(shlex.split(line.removeprefix(prefix))[0])
    raise AssertionError(f"missing {key} assignment")


def _write_fake_python(tmp_path: Path, *, exit_code: int = 0) -> Path:
    fake_bin = tmp_path / "fakebin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        f"""#!/usr/bin/env bash
printf '%s\\n' "$*" >> "${{FAKE_PYTHON_LOG}}"
exit {exit_code}
""",
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IEXEC)
    return fake_bin


def test_tmux_send_prompt_uses_load_buffer_for_multiline_prompt(tmp_path: Path) -> None:
    _write_fake_tmux(tmp_path, window_name="testpane")
    env = _fake_tmux_env(tmp_path)
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("line one\nline two\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "tmux_send_prompt.sh"),
            "--name",
            "testpane",
            "--prompt-file",
            str(prompt_file),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Prompt sent to 'testpane'" in result.stdout
    calls = _load_tmux_calls(env)
    assert any(call[:2] == ["load-buffer", "-"] for call in calls)
    assert any(call[:2] == ["paste-buffer", "-d"] for call in calls)
    assert not any(call[:2] == ["set-buffer", "-b"] for call in calls)


def test_tmux_session_launcher_waits_for_readiness_marker_before_prompt_send(
    tmp_path: Path,
) -> None:
    _write_fake_tmux(tmp_path)
    env = _fake_tmux_env(tmp_path)
    env["ARAGORA_TMUX_INIT_WAIT_SECONDS"] = "1"
    env["ARAGORA_TMUX_REGISTRY_REPO_ROOT"] = str(tmp_path)

    log_dir = Path(env["HOME"]) / ".aragora" / "tmux-sessions"
    log_dir.mkdir(parents=True)
    (log_dir / "testpane.log").write_text(
        "boot\nFind and fix a bug in @filename\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "tmux_session_launcher.sh"),
            "--name",
            "testpane",
            "--agent",
            "codex",
            "--prompt",
            "hello from launcher",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Readiness markers detected for testpane." in result.stdout
    calls = _load_tmux_calls(env)
    assert any(call[:2] == ["new-window", "-P"] for call in calls)
    assert any(call[:2] == ["pipe-pane", "-t"] and call[2] == "@17" for call in calls)
    assert any(
        call[:2] == ["send-keys", "-t"] and call[2] == "@17" and "hello from launcher" in call
        for call in calls
    )
    assert any(
        call[:2] == ["send-keys", "-t"] and call[2] == "@17" and ".heartbeat-launch." in call[3]
        for call in calls
    )
    heartbeat_launcher = _heartbeat_launcher_from_calls(calls, name="testpane")
    heartbeat_body = heartbeat_launcher.read_text(encoding="utf-8")
    assert "scripts/agent_heartbeat.py" in heartbeat_body
    assert "--finalize" in heartbeat_body
    assert "WRAPPER_RUN_ID=" in heartbeat_body
    assert '--thread-id "${WRAPPER_RUN_ID}"' in heartbeat_body
    assert "ARAGORA_TMUX_HEARTBEAT_INTERVAL_SECONDS" in heartbeat_body
    assert '_heartbeat_interval="60"' in heartbeat_body
    assert "^[1-9][0-9]*$" in heartbeat_body
    assert "set -euo pipefail" not in heartbeat_body
    assert 'if [[ "${_finalized}" == "1" ]]; then' in heartbeat_body
    assert "kill" in heartbeat_body
    assert "_heartbeat_sleep_pid" in heartbeat_body
    assert 'pkill -TERM -P "${_heartbeat_loop_pid}"' not in heartbeat_body
    assert '_launch_pid="$!"' not in heartbeat_body
    assert 'wait "${_launch_pid}"' not in heartbeat_body
    assert 'trap "_finalizer_signal=HUP; exit 129" HUP' in heartbeat_body
    assert "agent_heartbeat.py record failed" in heartbeat_body
    assert "agent_heartbeat.py finalize failed" in heartbeat_body
    assert 'rm -f "${LAUNCH_FILE}" "$0"' in heartbeat_body
    assert 'bash "${LAUNCH_FILE}" &' not in heartbeat_body
    assert 'bash "${LAUNCH_FILE}"' in heartbeat_body.splitlines()
    agent_launch_file = _assignment_path(heartbeat_body, "LAUNCH_FILE")
    agent_launch_body = agent_launch_file.read_text(encoding="utf-8")
    assert "./scripts/codex_session.sh" in agent_launch_body
    assert "--agent testpane" in agent_launch_body
    registry_payload = json.loads(
        (tmp_path / ".aragora" / "session_mux" / "registry.json").read_text()
    )
    assert "testpane" in registry_payload["sessions"]
    assert registry_payload["sessions"]["testpane"]["tmux_window"] == "@17"
    assert "heartbeat-launch.sh" not in registry_payload["sessions"]["testpane"]["launcher_command"]
    assert (
        "./scripts/codex_session.sh" in registry_payload["sessions"]["testpane"]["launcher_command"]
    )


def test_tmux_session_launcher_heartbeat_wrapper_fails_closed_when_failed_launch_cannot_finalize(
    tmp_path: Path,
) -> None:
    _write_fake_tmux(tmp_path)
    env = _fake_tmux_env(tmp_path)
    env["ARAGORA_TMUX_INIT_WAIT_SECONDS"] = "1"
    env["ARAGORA_TMUX_REGISTRY_REPO_ROOT"] = str(tmp_path)

    workdir = tmp_path / "worker"
    (workdir / "scripts").mkdir(parents=True)
    codex_session = workdir / "scripts" / "codex_session.sh"
    codex_session.write_text(
        """#!/usr/bin/env bash
exit "${FAKE_LAUNCH_EXIT:-0}"
""",
        encoding="utf-8",
    )
    codex_session.chmod(codex_session.stat().st_mode | stat.S_IEXEC)

    subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "tmux_session_launcher.sh"),
            "--name",
            "exec-wrapper",
            "--agent",
            "codex",
            "--cwd",
            str(workdir),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    calls = _load_tmux_calls(env)
    heartbeat_launcher = _heartbeat_launcher_from_calls(calls, name="exec-wrapper")
    fake_python_bin = _write_fake_python(tmp_path, exit_code=1)
    wrapper_env = env.copy()
    wrapper_env["PATH"] = f"{fake_python_bin}:{wrapper_env['PATH']}"
    wrapper_env["FAKE_PYTHON_LOG"] = str(tmp_path / "python-calls.log")
    wrapper_env["FAKE_LAUNCH_EXIT"] = "7"
    wrapper_env["ARAGORA_TMUX_HEARTBEAT_INTERVAL_SECONDS"] = "999"

    result = subprocess.run(
        ["bash", str(heartbeat_launcher)],
        cwd=REPO_ROOT,
        env=wrapper_env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    heartbeat_log = (heartbeat_launcher.parent / "exec-wrapper.heartbeat.log").read_text(
        encoding="utf-8"
    )
    assert "agent_heartbeat.py record failed for exec-wrapper" in heartbeat_log
    assert "agent_heartbeat.py finalize failed for exec-wrapper (failed)" in heartbeat_log
    python_calls = Path(wrapper_env["FAKE_PYTHON_LOG"]).read_text(encoding="utf-8")
    assert "scripts/agent_heartbeat.py" in python_calls
    assert "--finalize" in python_calls


def test_tmux_session_launcher_heartbeat_wrapper_fails_when_successful_launch_cannot_finalize(
    tmp_path: Path,
) -> None:
    _write_fake_tmux(tmp_path)
    env = _fake_tmux_env(tmp_path)
    env["ARAGORA_TMUX_INIT_WAIT_SECONDS"] = "1"
    env["ARAGORA_TMUX_REGISTRY_REPO_ROOT"] = str(tmp_path)

    workdir = tmp_path / "worker"
    (workdir / "scripts").mkdir(parents=True)
    codex_session = workdir / "scripts" / "codex_session.sh"
    codex_session.write_text(
        """#!/usr/bin/env bash
exit "${FAKE_LAUNCH_EXIT:-0}"
""",
        encoding="utf-8",
    )
    codex_session.chmod(codex_session.stat().st_mode | stat.S_IEXEC)

    subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "tmux_session_launcher.sh"),
            "--name",
            "exec-wrapper-finalize",
            "--agent",
            "codex",
            "--cwd",
            str(workdir),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    calls = _load_tmux_calls(env)
    heartbeat_launcher = _heartbeat_launcher_from_calls(calls, name="exec-wrapper-finalize")
    fake_python_bin = _write_fake_python(tmp_path, exit_code=1)
    wrapper_env = env.copy()
    wrapper_env["PATH"] = f"{fake_python_bin}:{wrapper_env['PATH']}"
    wrapper_env["FAKE_PYTHON_LOG"] = str(tmp_path / "python-calls.log")
    wrapper_env["FAKE_LAUNCH_EXIT"] = "0"
    wrapper_env["ARAGORA_TMUX_HEARTBEAT_INTERVAL_SECONDS"] = "999"

    result = subprocess.run(
        ["bash", str(heartbeat_launcher)],
        cwd=REPO_ROOT,
        env=wrapper_env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    heartbeat_log = (heartbeat_launcher.parent / "exec-wrapper-finalize.heartbeat.log").read_text(
        encoding="utf-8"
    )
    assert (
        "agent_heartbeat.py finalize failed for exec-wrapper-finalize (completed)" in heartbeat_log
    )


def test_tmux_session_launcher_accepts_dotted_session_names(tmp_path: Path) -> None:
    _write_fake_tmux(tmp_path)
    env = _fake_tmux_env(tmp_path)
    env["ARAGORA_TMUX_INIT_WAIT_SECONDS"] = "1"
    env["ARAGORA_TMUX_REGISTRY_REPO_ROOT"] = str(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "tmux_session_launcher.sh"),
            "--name",
            "task.v2",
            "--agent",
            "codex",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Launched 'task.v2'" in result.stdout
    calls = _load_tmux_calls(env)
    heartbeat_launcher = _heartbeat_launcher_from_calls(calls, name="task.v2")
    assert heartbeat_launcher.exists()


def test_tmux_session_launcher_rejects_unsafe_session_names(tmp_path: Path) -> None:
    _write_fake_tmux(tmp_path)
    env = _fake_tmux_env(tmp_path)
    env["ARAGORA_TMUX_INIT_WAIT_SECONDS"] = "1"
    env["ARAGORA_TMUX_REGISTRY_REPO_ROOT"] = str(tmp_path)

    for name in (".hidden", "../escape", "bad/name", "bad:name"):
        result = subprocess.run(
            [
                "bash",
                str(REPO_ROOT / "scripts" / "tmux_session_launcher.sh"),
                "--name",
                name,
                "--agent",
                "codex",
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 2
        assert "Invalid session name" in result.stderr


def test_tmux_session_launcher_rejects_duplicate_session_name(tmp_path: Path) -> None:
    _write_fake_tmux(tmp_path, window_name="dupe-session")
    env = _fake_tmux_env(tmp_path)
    env["ARAGORA_TMUX_INIT_WAIT_SECONDS"] = "1"

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "tmux_session_launcher.sh"),
            "--name",
            "dupe-session",
            "--agent",
            "codex",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Refusing duplicate tmux session name 'dupe-session'" in result.stderr


def test_tmux_session_launcher_allows_duplicate_name_with_terminal_heartbeat(
    tmp_path: Path,
) -> None:
    _write_fake_tmux(tmp_path, window_name="dupe-session")
    env = _fake_tmux_env(tmp_path)
    env["ARAGORA_TMUX_INIT_WAIT_SECONDS"] = "1"
    env["ARAGORA_TMUX_REGISTRY_REPO_ROOT"] = str(tmp_path)
    heartbeat_path = tmp_path / ".aragora" / "agent-bridge" / "heartbeats.json"
    heartbeat_path.parent.mkdir(parents=True)
    heartbeat_path.write_text(
        json.dumps(
            [
                {
                    "schema_version": "aragora-agent-heartbeat/1.0",
                    "lane_id": "dupe-session",
                    "owner_session": "dupe-session",
                    "terminal": True,
                    "terminal_outcome": "completed",
                    "terminal_finalized_at": "2026-06-23T10:10:00Z",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "tmux_session_launcher.sh"),
            "--name",
            "dupe-session",
            "--agent",
            "codex",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Existing tmux window 'dupe-session' has terminal heartbeat state" in result.stderr
    assert "Removed terminal tmux window before relaunch: dupe-session" in result.stderr
    calls = _load_tmux_calls(env)
    assert ["kill-window", "-t", "aragora:0"] in calls
    assert "Launched 'dupe-session'" in result.stdout


def test_tmux_session_launcher_launch_wrapper_quotes_workdir(tmp_path: Path) -> None:
    _write_fake_tmux(tmp_path)
    env = _fake_tmux_env(tmp_path)
    env["ARAGORA_TMUX_INIT_WAIT_SECONDS"] = "1"
    env["ARAGORA_TMUX_REGISTRY_REPO_ROOT"] = str(tmp_path)

    workdir = tmp_path / "worker dir's"
    (workdir / "scripts").mkdir(parents=True)
    codex_session = workdir / "scripts" / "codex_session.sh"
    codex_session.write_text(
        """#!/usr/bin/env bash
pwd > "${FAKE_LAUNCH_PWD}"
""",
        encoding="utf-8",
    )
    codex_session.chmod(codex_session.stat().st_mode | stat.S_IEXEC)

    subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "tmux_session_launcher.sh"),
            "--name",
            "quoted-workdir",
            "--agent",
            "codex",
            "--cwd",
            str(workdir),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    calls = _load_tmux_calls(env)
    heartbeat_launcher = _heartbeat_launcher_from_calls(calls, name="quoted-workdir")
    fake_python_bin = _write_fake_python(tmp_path)
    wrapper_env = env.copy()
    wrapper_env["PATH"] = f"{fake_python_bin}:{wrapper_env['PATH']}"
    wrapper_env["FAKE_PYTHON_LOG"] = str(tmp_path / "python-calls.log")
    wrapper_env["FAKE_LAUNCH_PWD"] = str(tmp_path / "launch-pwd.txt")

    result = subprocess.run(
        ["bash", str(heartbeat_launcher)],
        cwd=REPO_ROOT,
        env=wrapper_env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert Path(wrapper_env["FAKE_LAUNCH_PWD"]).read_text(encoding="utf-8").strip() == str(workdir)


def test_tmux_session_launcher_heartbeat_log_uses_full_safe_session_name(
    tmp_path: Path,
) -> None:
    _write_fake_tmux(tmp_path)
    env = _fake_tmux_env(tmp_path)
    env["ARAGORA_TMUX_INIT_WAIT_SECONDS"] = "1"
    env["ARAGORA_TMUX_REGISTRY_REPO_ROOT"] = str(tmp_path)
    name = "foo.heartbeat-launch.bar"

    subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "tmux_session_launcher.sh"),
            "--name",
            name,
            "--agent",
            "codex",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    calls = _load_tmux_calls(env)
    heartbeat_launcher = _heartbeat_launcher_from_calls(calls, name=name)
    heartbeat_body = heartbeat_launcher.read_text(encoding="utf-8")

    heartbeat_log = _assignment_path(heartbeat_body, "HEARTBEAT_LOG")
    assert heartbeat_log.name == f"{name}.heartbeat.log"


def test_tmux_session_launcher_autonomous_codex_prompt_uses_exec(
    tmp_path: Path,
) -> None:
    _write_fake_tmux(tmp_path)
    env = _fake_tmux_env(tmp_path)
    env["ARAGORA_TMUX_INIT_WAIT_SECONDS"] = "1"
    env["ARAGORA_TMUX_REGISTRY_REPO_ROOT"] = str(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "tmux_session_launcher.sh"),
            "--name",
            "codex-auto",
            "--agent",
            "codex",
            "--autonomous",
            "--task-id",
            "Q123",
            "--claimed-path",
            "scripts/tmux_session_launcher.sh",
            "--prompt",
            "report git status only",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Waiting up to" not in result.stdout
    calls = _load_tmux_calls(env)
    assert any(
        call[:2] == ["send-keys", "-t"]
        and call[2] == "@17"
        and call[3].startswith("bash ")
        and ".heartbeat-launch." in call[3]
        and "--full-auto" not in call[3]
        for call in calls
    )
    launch_script = Path(env["HOME"]) / ".aragora" / "tmux-sessions" / "codex-auto.launch.sh"
    launch_body = launch_script.read_text(encoding="utf-8")
    assert "--task-id Q123" in launch_body
    assert "--claimed-path scripts/tmux_session_launcher.sh" in launch_body
    assert "codex exec --dangerously-bypass-approvals-and-sandbox - <" in launch_body
    assert "--ask-for-approval" not in launch_body
    assert "--full-auto" not in launch_body
    assert not any("report git status only" in json.dumps(call) for call in calls)
    assert (Path(env["HOME"]) / ".aragora" / "tmux-sessions" / "codex-auto.prompt.md").read_text(
        encoding="utf-8"
    ) == "report git status only\n"
    meta = json.loads(
        (Path(env["HOME"]) / ".aragora" / "tmux-sessions" / "codex-auto.meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert meta["has_prompt"] is True
    assert meta["prompt_file"].endswith("codex-auto.prompt.md")
    heartbeat_launcher = _heartbeat_launcher_from_calls(calls, name="codex-auto")
    heartbeat_body = heartbeat_launcher.read_text(encoding="utf-8")
    assert "scripts/agent_heartbeat.py" in heartbeat_body
    assert "LANE_ID=Q123" in heartbeat_body
    assert "OWNER_SESSION=codex-auto" in heartbeat_body
    agent_launch_file = _assignment_path(heartbeat_body, "LAUNCH_FILE")
    agent_launch_body = agent_launch_file.read_text(encoding="utf-8")
    assert "codex-auto.launch.sh" in agent_launch_body
    assert "codex exec --dangerously-bypass-approvals-and-sandbox" not in heartbeat_body


def test_tmux_session_launcher_rejects_autonomous_codex_without_lease(
    tmp_path: Path,
) -> None:
    _write_fake_tmux(tmp_path)
    env = _fake_tmux_env(tmp_path)
    env["ARAGORA_TMUX_REGISTRY_REPO_ROOT"] = str(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "tmux_session_launcher.sh"),
            "--name",
            "codex-auto",
            "--agent",
            "codex",
            "--autonomous",
            "--prompt",
            "report git status only",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Refusing unleased autonomous Codex launch." in result.stderr


def test_tmux_session_launcher_rejects_autonomous_codex_with_title_only(
    tmp_path: Path,
) -> None:
    _write_fake_tmux(tmp_path)
    env = _fake_tmux_env(tmp_path)
    env["ARAGORA_TMUX_REGISTRY_REPO_ROOT"] = str(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "tmux_session_launcher.sh"),
            "--name",
            "codex-auto",
            "--agent",
            "codex",
            "--autonomous",
            "--title",
            "descriptive goal only",
            "--prompt",
            "report git status only",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "requires --task-id plus at least one concrete" in result.stderr


def test_tmux_session_launcher_rejects_autonomous_codex_with_task_id_only(
    tmp_path: Path,
) -> None:
    _write_fake_tmux(tmp_path)
    env = _fake_tmux_env(tmp_path)
    env["ARAGORA_TMUX_REGISTRY_REPO_ROOT"] = str(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "tmux_session_launcher.sh"),
            "--name",
            "codex-auto",
            "--agent",
            "codex",
            "--autonomous",
            "--task-id",
            "Q123",
            "--prompt",
            "report git status only",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "requires --task-id plus at least one concrete" in result.stderr


def test_tmux_session_launcher_accepts_autonomous_codex_with_task_id_and_scope(
    tmp_path: Path,
) -> None:
    _write_fake_tmux(tmp_path)
    env = _fake_tmux_env(tmp_path)
    env["ARAGORA_TMUX_INIT_WAIT_SECONDS"] = "1"
    env["ARAGORA_TMUX_REGISTRY_REPO_ROOT"] = str(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "tmux_session_launcher.sh"),
            "--name",
            "codex-auto",
            "--agent",
            "codex",
            "--autonomous",
            "--task-id",
            "Q123",
            "--claimed-path",
            "scripts/tmux_session_launcher.sh",
            "--prompt",
            "report git status only",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Refusing unleased autonomous Codex launch." not in result.stderr
    launch_script = Path(env["HOME"]) / ".aragora" / "tmux-sessions" / "codex-auto.launch.sh"
    launch_body = launch_script.read_text(encoding="utf-8")
    assert "--task-id Q123" in launch_body
    assert "--claimed-path scripts/tmux_session_launcher.sh" in launch_body


def test_tmux_session_launcher_accepts_new_codex_readiness_markers(tmp_path: Path) -> None:
    _write_fake_tmux(tmp_path)
    env = _fake_tmux_env(tmp_path)
    env["ARAGORA_TMUX_INIT_WAIT_SECONDS"] = "1"
    env["ARAGORA_TMUX_REGISTRY_REPO_ROOT"] = str(tmp_path)

    log_dir = Path(env["HOME"]) / ".aragora" / "tmux-sessions"
    log_dir.mkdir(parents=True)
    (log_dir / "testpane.log").write_text(
        "boot\nFind and fix a bug in @filename\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "tmux_session_launcher.sh"),
            "--name",
            "testpane",
            "--agent",
            "codex",
            "--prompt",
            "hello from launcher",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Readiness markers detected for testpane." in result.stdout


def test_tmux_session_launcher_does_not_treat_codex_banner_as_ready(tmp_path: Path) -> None:
    _write_fake_tmux(tmp_path)
    env = _fake_tmux_env(tmp_path)
    env["ARAGORA_TMUX_INIT_WAIT_SECONDS"] = "1"
    env["ARAGORA_TMUX_REGISTRY_REPO_ROOT"] = str(tmp_path)

    log_dir = Path(env["HOME"]) / ".aragora" / "tmux-sessions"
    log_dir.mkdir(parents=True)
    (log_dir / "testpane.log").write_text(
        "boot\nOpenAI Codex (v0.125.0)\nUse /rename to rename your threads\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "tmux_session_launcher.sh"),
            "--name",
            "testpane",
            "--agent",
            "codex",
            "--prompt",
            "hello from launcher",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Timed out waiting for readiness markers for testpane; prompt not sent." in result.stdout
    calls = _load_tmux_calls(env)
    assert ["load-buffer", "-"] not in calls
    assert not any(
        call[:2] == ["send-keys", "-t"] and "hello from launcher" in call for call in calls
    )


def test_tmux_session_launcher_supports_droid_agent(tmp_path: Path) -> None:
    _write_fake_tmux(tmp_path)
    env = _fake_tmux_env(tmp_path)
    env["ARAGORA_TMUX_INIT_WAIT_SECONDS"] = "1"
    env["ARAGORA_TMUX_REGISTRY_REPO_ROOT"] = str(tmp_path)

    log_dir = Path(env["HOME"]) / ".aragora" / "tmux-sessions"
    log_dir.mkdir(parents=True)
    (log_dir / "factory-review.log").write_text("boot\nDroid\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "tmux_session_launcher.sh"),
            "--name",
            "factory-review",
            "--agent",
            "droid",
            "--prompt",
            "review only",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    calls = _load_tmux_calls(env)
    assert any(
        call[:2] == ["send-keys", "-t"] and call[2] == "@17" and ".heartbeat-launch." in call[3]
        for call in calls
    )
    heartbeat_launcher = _heartbeat_launcher_from_calls(calls, name="factory-review")
    heartbeat_body = heartbeat_launcher.read_text(encoding="utf-8")
    assert "scripts/agent_heartbeat.py" in heartbeat_body
    assert "droid exec --auto 'high'" not in heartbeat_body
    agent_launch_file = _assignment_path(heartbeat_body, "LAUNCH_FILE")
    agent_launch_body = agent_launch_file.read_text(encoding="utf-8")
    assert "droid exec --auto high" in agent_launch_body
    assert "-f" in agent_launch_body
    assert not any("review only" in json.dumps(call) for call in calls)
    meta = json.loads(
        (Path(env["HOME"]) / ".aragora" / "tmux-sessions" / "factory-review.meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert meta["has_prompt"] is True
    assert meta["prompt_file"].endswith("factory-review.prompt.md")
    assert Path(meta["prompt_file"]).read_text(encoding="utf-8") == "review only\n"


def test_tmux_session_launcher_supports_droid_prompt_file(tmp_path: Path) -> None:
    _write_fake_tmux(tmp_path)
    env = _fake_tmux_env(tmp_path)
    env["ARAGORA_TMUX_INIT_WAIT_SECONDS"] = "1"
    env["ARAGORA_TMUX_REGISTRY_REPO_ROOT"] = str(tmp_path)
    prompt_file = tmp_path / "review.md"
    prompt_file.write_text("review from file", encoding="utf-8")

    log_dir = Path(env["HOME"]) / ".aragora" / "tmux-sessions"
    log_dir.mkdir(parents=True)
    (log_dir / "factory-review.log").write_text("boot\nDroid\n", encoding="utf-8")

    subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "tmux_session_launcher.sh"),
            "--name",
            "factory-review",
            "--agent",
            "factory",
            "--prompt-file",
            str(prompt_file),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    calls = _load_tmux_calls(env)
    assert any(
        call[:2] == ["send-keys", "-t"] and call[2] == "@17" and ".heartbeat-launch." in call[3]
        for call in calls
    )
    heartbeat_launcher = _heartbeat_launcher_from_calls(calls, name="factory-review")
    heartbeat_body = heartbeat_launcher.read_text(encoding="utf-8")
    agent_launch_file = _assignment_path(heartbeat_body, "LAUNCH_FILE")
    agent_launch_body = agent_launch_file.read_text(encoding="utf-8")
    assert "droid exec --auto high" in agent_launch_body
    assert f"-f {shlex.quote(str(prompt_file))}" in agent_launch_body
    assert not any("review from file" in json.dumps(call) for call in calls)


def test_tmux_session_launcher_rejects_unprompted_droid_auto_off_by_default(
    tmp_path: Path,
) -> None:
    _write_fake_tmux(tmp_path)
    env = _fake_tmux_env(tmp_path)
    env["ARAGORA_TMUX_REGISTRY_REPO_ROOT"] = str(tmp_path)

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "tmux_session_launcher.sh"),
            "--name",
            "factory-review",
            "--agent",
            "droid",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "interactive Droid starts Auto Off" in result.stderr


def test_tmux_session_launcher_does_not_send_prompt_before_readiness_by_default(
    tmp_path: Path,
) -> None:
    _write_fake_tmux(tmp_path)
    env = _fake_tmux_env(tmp_path)
    env["ARAGORA_TMUX_INIT_WAIT_SECONDS"] = "1"
    env["ARAGORA_TMUX_REGISTRY_REPO_ROOT"] = str(tmp_path)

    log_dir = Path(env["HOME"]) / ".aragora" / "tmux-sessions"
    log_dir.mkdir(parents=True)
    (log_dir / "testpane.log").write_text("boot only\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "tmux_session_launcher.sh"),
            "--name",
            "testpane",
            "--agent",
            "codex",
            "--prompt",
            "do not send yet",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "prompt not sent" in result.stdout
    calls = _load_tmux_calls(env)
    assert not any(call[:2] == ["send-keys", "-t"] and "do not send yet" in call for call in calls)


def test_tmux_session_launcher_can_send_prompt_on_timeout_when_explicitly_enabled(
    tmp_path: Path,
) -> None:
    _write_fake_tmux(tmp_path)
    env = _fake_tmux_env(tmp_path)
    env["ARAGORA_TMUX_INIT_WAIT_SECONDS"] = "1"
    env["ARAGORA_TMUX_SEND_ON_TIMEOUT"] = "1"
    env["ARAGORA_TMUX_REGISTRY_REPO_ROOT"] = str(tmp_path)

    log_dir = Path(env["HOME"]) / ".aragora" / "tmux-sessions"
    log_dir.mkdir(parents=True)
    (log_dir / "testpane.log").write_text("boot only\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "tmux_session_launcher.sh"),
            "--name",
            "testpane",
            "--agent",
            "codex",
            "--prompt",
            "send despite timeout",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "sending prompt anyway because ARAGORA_TMUX_SEND_ON_TIMEOUT=1" in result.stdout
    calls = _load_tmux_calls(env)
    assert any(
        call[:2] == ["send-keys", "-t"] and call[2] == "@17" and "send despite timeout" in call
        for call in calls
    )
