"""Focused tests for the bounded Claude consult helper."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "consult_claude.py"
SPEC = importlib.util.spec_from_file_location("consult_claude_under_test", SCRIPT)
assert SPEC and SPEC.loader
consult_claude = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(consult_claude)


def test_build_cli_command_disables_mcp() -> None:
    with consult_claude._claude_empty_mcp_config_file() as mcp_config_path:
        command, _used_profile = consult_claude._build_cli_command(
            "claude-fable-5", mcp_config_path
        )

        assert "--strict-mcp-config" in command
        assert "--mcp-config" in command
        assert command[command.index("--mcp-config") + 1] == str(mcp_config_path)
        assert json.loads(mcp_config_path.read_text(encoding="utf-8")) == {"mcpServers": {}}
        assert "--model" in command
        assert command[command.index("--model") + 1] == "claude-fable-5"


def test_run_cli_uses_stdin_prompt_timeout_and_redacts_stderr(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakePopen:
        returncode = 1
        pid = 12345

        def __init__(self, command, *, stdin, stdout, stderr, text, start_new_session):
            mcp_config_path = Path(command[command.index("--mcp-config") + 1])
            captured.update(
                {
                    "command": command,
                    "stdin": stdin,
                    "stdout": stdout,
                    "stderr": stderr,
                    "text": text,
                    "start_new_session": start_new_session,
                    "mcp_exists_during_run": mcp_config_path.exists(),
                    "mcp_json": json.loads(mcp_config_path.read_text(encoding="utf-8")),
                }
            )

        def communicate(self, input, timeout):
            captured["input"] = input
            captured["timeout"] = timeout
            return "", "Using profile home: /secret/profile\nCommand: claude --print\ntoken=secret"

    monkeypatch.setattr(consult_claude.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(consult_claude.subprocess, "Popen", FakePopen)

    result = consult_claude._run_cli("live prompt", "claude-fable-5", 12.5)

    assert result["ok"] is False
    assert result["error"] == "claude CLI failed, rc=1, empty=True"
    assert "secret" not in json.dumps(result)
    assert "profile" not in json.dumps(result).lower()
    assert captured["input"] == "live prompt"
    assert captured["timeout"] == 12.5
    assert captured["mcp_exists_during_run"] is True
    assert captured["mcp_json"] == {"mcpServers": {}}
    assert captured["start_new_session"] is True
    assert "-p" in captured["command"]
    assert captured["command"][captured["command"].index("-p") + 1] == "-"


def test_consult_api_fallback_tries_fallback_model(monkeypatch) -> None:
    cli_models: list[str] = []
    api_models: list[str] = []

    def fake_cli(_prompt: str, model: str, _timeout: float) -> dict:
        cli_models.append(model)
        return {"ok": False, "backend": "cli", "error": f"{model} unavailable"}

    def fake_api(_prompt: str, model: str, _timeout: float, *, system: str | None = None) -> dict:
        del system
        api_models.append(model)
        if model == consult_claude.FALLBACK_MODEL:
            return {"ok": True, "backend": "api", "text": "fallback answer", "elapsed_s": 0.1}
        return {"ok": False, "backend": "api", "error": f"{model} unavailable"}

    monkeypatch.setattr(consult_claude, "_run_cli", fake_cli)
    monkeypatch.setattr(consult_claude, "_run_api", fake_api)

    result = consult_claude.consult("question")

    assert result["ok"] is True
    assert result["model"] == consult_claude.FALLBACK_MODEL
    assert cli_models == [consult_claude.DEFAULT_MODEL, consult_claude.FALLBACK_MODEL]
    assert api_models == [consult_claude.DEFAULT_MODEL, consult_claude.FALLBACK_MODEL]


def test_run_api_redacts_http_error_body(monkeypatch) -> None:
    class RaisingUrlopen:
        def __call__(self, *_args, **_kwargs):
            raise consult_claude.urllib.error.HTTPError(
                url=consult_claude.ANTHROPIC_API_URL,
                code=429,
                msg="Too Many Requests",
                hdrs={},
                fp=io.BytesIO(b"profile=/secret/path token=secret prompt text"),
            )

    monkeypatch.setattr(consult_claude, "_resolve_api_key", lambda: "test-key")
    monkeypatch.setattr(consult_claude.urllib.request, "urlopen", RaisingUrlopen())

    result = consult_claude._run_api("secret prompt", "claude-fable-5", 1.0, None)

    assert result["ok"] is False
    assert result["error"] == "API HTTP 429: response body redacted"
    assert "secret" not in json.dumps(result)
    assert "profile" not in json.dumps(result).lower()


def test_run_api_redacts_invalid_response_body(monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return b"\xff\xfe not utf-8"

    monkeypatch.setattr(consult_claude, "_resolve_api_key", lambda: "test-key")
    monkeypatch.setattr(
        consult_claude.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(),
    )

    result = consult_claude._run_api("secret prompt", "claude-fable-5", 1.0, None)

    assert result["ok"] is False
    assert result["error"] == "API response parse failed: response body redacted"
    assert "secret" not in json.dumps(result)


def test_consult_enforces_overall_timeout_before_fallbacks(monkeypatch) -> None:
    cli_models: list[str] = []
    monotonic_values = iter([0.0, 0.0, 10.0, 10.0, 10.0])

    def fake_cli(_prompt: str, model: str, _timeout: float) -> dict:
        cli_models.append(model)
        return {"ok": False, "backend": "cli", "timed_out": True, "error": "timeout"}

    monkeypatch.setattr(consult_claude.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(consult_claude, "_run_cli", fake_cli)

    result = consult_claude.consult("question", timeout=10, overall_timeout=10)

    assert result["ok"] is False
    assert result["timed_out"] is True
    assert cli_models == [consult_claude.DEFAULT_MODEL]
    assert [attempt["backend"] for attempt in result["attempts"]] == [
        "cli",
        "cli",
        "api",
        "api",
    ]
    assert all(attempt.get("timed_out") for attempt in result["attempts"])
    assert "overall consult timeout exhausted before attempt" in result["error"]


def test_consult_default_budget_allows_fallback_after_primary_timeout(monkeypatch) -> None:
    cli_models: list[str] = []
    cli_timeouts: list[float] = []
    monotonic_values = iter([0.0, 0.0, 10.0])

    def fake_cli(_prompt: str, model: str, timeout: float) -> dict:
        cli_models.append(model)
        cli_timeouts.append(timeout)
        if model == consult_claude.DEFAULT_MODEL:
            return {"ok": False, "backend": "cli", "timed_out": True, "error": "timeout"}
        return {"ok": True, "backend": "cli", "text": "fallback cli answer", "elapsed_s": 0.1}

    monkeypatch.setattr(consult_claude.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(consult_claude, "_run_cli", fake_cli)

    result = consult_claude.consult("question", timeout=10)

    assert result["ok"] is True
    assert result["model"] == consult_claude.FALLBACK_MODEL
    assert cli_models == [consult_claude.DEFAULT_MODEL, consult_claude.FALLBACK_MODEL]
    assert cli_timeouts == [10, 10]


def test_consult_reports_timeout_only_when_all_attempts_timeout(monkeypatch) -> None:
    attempt_timeouts: list[float] = []
    monotonic_values = iter([0.0, 0.0, 10.0, 20.0, 30.0])

    def fake_cli(_prompt: str, model: str, timeout: float) -> dict:
        attempt_timeouts.append(timeout)
        return {
            "ok": False,
            "backend": "cli",
            "timed_out": True,
            "error": f"{model} timed out",
        }

    def fake_api(_prompt: str, model: str, timeout: float, *, system: str | None = None) -> dict:
        del system
        attempt_timeouts.append(timeout)
        return {
            "ok": False,
            "backend": "api",
            "timed_out": True,
            "error": f"{model} timed out",
        }

    monkeypatch.setattr(consult_claude, "_run_cli", fake_cli)
    monkeypatch.setattr(consult_claude, "_run_api", fake_api)
    monkeypatch.setattr(consult_claude.time, "monotonic", lambda: next(monotonic_values))

    result = consult_claude.consult("question")

    assert result["ok"] is False
    assert result["timed_out"] is True
    assert [attempt["model"] for attempt in result["attempts"]] == [
        consult_claude.DEFAULT_MODEL,
        consult_claude.FALLBACK_MODEL,
        consult_claude.DEFAULT_MODEL,
        consult_claude.FALLBACK_MODEL,
    ]
    assert attempt_timeouts == [600, 600, 600, 600]


def test_run_cli_timeout_kills_process_group(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakePopen:
        returncode = None
        pid = 6789

        def __init__(self, *args, **kwargs):
            calls.append(("popen", kwargs["start_new_session"]))

        def communicate(self, input, timeout):
            calls.append(("communicate", timeout))
            raise subprocess.TimeoutExpired(cmd=["claude"], timeout=timeout)

        def wait(self, timeout):
            calls.append(("wait", timeout))

    monkeypatch.setattr(consult_claude.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(consult_claude.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(consult_claude.os, "killpg", lambda pid, sig: calls.append(("killpg", pid)))

    result = consult_claude._run_cli("question", "claude-fable-5", 1.5)

    assert result["timed_out"] is True
    assert ("popen", True) in calls
    assert ("killpg", 6789) in calls
    assert ("wait", 5) in calls


def test_consult_tries_fallback_cli_after_primary_timeout(monkeypatch) -> None:
    cli_models: list[str] = []

    def fake_cli(_prompt: str, model: str, _timeout: float) -> dict:
        cli_models.append(model)
        if model == consult_claude.DEFAULT_MODEL:
            return {"ok": False, "backend": "cli", "timed_out": True, "error": "timeout"}
        return {"ok": True, "backend": "cli", "text": "fallback cli answer", "elapsed_s": 0.1}

    monkeypatch.setattr(consult_claude, "_run_cli", fake_cli)

    result = consult_claude.consult("question", api_fallback=False)

    assert result["ok"] is True
    assert result["model"] == consult_claude.FALLBACK_MODEL
    assert cli_models == [consult_claude.DEFAULT_MODEL, consult_claude.FALLBACK_MODEL]


def test_main_rejects_non_positive_timeout(capsys) -> None:
    rc = consult_claude.main(["--timeout", "0", "question"])

    assert rc == consult_claude.EXIT_USAGE
    assert "positive finite" in capsys.readouterr().err


def test_main_rejects_non_positive_overall_timeout(capsys) -> None:
    rc = consult_claude.main(["--overall-timeout", "0", "question"])

    assert rc == consult_claude.EXIT_USAGE
    assert "positive finite" in capsys.readouterr().err


def test_main_reports_missing_prompt_file(capsys, tmp_path) -> None:
    missing = tmp_path / "missing.md"

    rc = consult_claude.main(["--prompt-file", str(missing)])

    assert rc == consult_claude.EXIT_NO_PROMPT
    assert "cannot read --prompt-file" in capsys.readouterr().err
