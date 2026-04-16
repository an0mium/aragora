from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_fake_tmux(tmp_path: Path, *, window_name: str = "testpane") -> Path:
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
if cmd[:2] in (["new-session", "-d"], ["pipe-pane", "-t"]):
    raise SystemExit(0)
if cmd[:2] in (["send-keys", "-t"], ["set-buffer", "-b"], ["paste-buffer", "-b"], ["delete-buffer", "-b"]):
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
    return env


def _load_tmux_calls(env: dict[str, str]) -> list[list[str]]:
    log_path = Path(env["FAKE_TMUX_LOG"])
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_tmux_send_prompt_uses_unique_buffer_for_multiline_prompt(tmp_path: Path) -> None:
    _write_fake_tmux(tmp_path)
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
    set_buffer_call = next(call for call in calls if call[:2] == ["set-buffer", "-b"])
    paste_buffer_call = next(call for call in calls if call[:2] == ["paste-buffer", "-b"])
    delete_buffer_call = next(call for call in calls if call[:2] == ["delete-buffer", "-b"])

    buffer_name = set_buffer_call[2]
    assert buffer_name == paste_buffer_call[2] == delete_buffer_call[2]
    assert buffer_name.startswith("aragora-prompt-testpane-")
    assert buffer_name != "aragora-prompt"


def test_tmux_session_launcher_waits_for_readiness_marker_before_prompt_send(
    tmp_path: Path,
) -> None:
    _write_fake_tmux(tmp_path)
    env = _fake_tmux_env(tmp_path)
    env["ARAGORA_TMUX_INIT_WAIT_SECONDS"] = "1"
    env["ARAGORA_TMUX_REGISTRY_REPO_ROOT"] = str(tmp_path)

    log_dir = Path(env["HOME"]) / ".aragora" / "tmux-sessions"
    log_dir.mkdir(parents=True)
    (log_dir / "testpane.log").write_text("boot\nOpenAI Codex\n", encoding="utf-8")

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
        call[:2] == ["send-keys", "-t"]
        and call[2] == "@17"
        and "./scripts/codex_session.sh --agent 'testpane'" in call[3]
        for call in calls
    )
    registry_payload = json.loads(
        (tmp_path / ".aragora" / "session_mux" / "registry.json").read_text()
    )
    assert "testpane" in registry_payload["sessions"]
    assert registry_payload["sessions"]["testpane"]["tmux_window"] == "@17"


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
