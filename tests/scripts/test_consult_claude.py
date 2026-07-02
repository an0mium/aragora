"""Focused tests for the bounded Claude consult helper."""

from __future__ import annotations

import importlib.util
import json
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


def test_run_cli_uses_stdin_prompt_and_timeout(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, *, input, capture_output, text, timeout):
        mcp_config_path = Path(command[command.index("--mcp-config") + 1])
        captured.update(
            {
                "command": command,
                "input": input,
                "capture_output": capture_output,
                "text": text,
                "timeout": timeout,
                "mcp_exists_during_run": mcp_config_path.exists(),
                "mcp_json": json.loads(mcp_config_path.read_text(encoding="utf-8")),
            }
        )
        return SimpleNamespace(returncode=0, stdout="answer\n", stderr="")

    monkeypatch.setattr(consult_claude.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(consult_claude.subprocess, "run", fake_run)

    result = consult_claude._run_cli("live prompt", "claude-fable-5", 12.5)

    assert result["ok"] is True
    assert result["text"] == "answer"
    assert captured["input"] == "live prompt"
    assert captured["timeout"] == 12.5
    assert captured["mcp_exists_during_run"] is True
    assert captured["mcp_json"] == {"mcpServers": {}}
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


def test_consult_reports_timeout_only_when_all_attempts_timeout(monkeypatch) -> None:
    def fake_cli(_prompt: str, model: str, _timeout: float) -> dict:
        return {
            "ok": False,
            "backend": "cli",
            "timed_out": True,
            "error": f"{model} timed out",
        }

    def fake_api(_prompt: str, model: str, _timeout: float, *, system: str | None = None) -> dict:
        del system
        return {
            "ok": False,
            "backend": "api",
            "timed_out": True,
            "error": f"{model} timed out",
        }

    monkeypatch.setattr(consult_claude, "_run_cli", fake_cli)
    monkeypatch.setattr(consult_claude, "_run_api", fake_api)

    result = consult_claude.consult("question")

    assert result["ok"] is False
    assert result["timed_out"] is True
    assert [attempt["model"] for attempt in result["attempts"]] == [
        consult_claude.DEFAULT_MODEL,
        consult_claude.DEFAULT_MODEL,
        consult_claude.FALLBACK_MODEL,
    ]
