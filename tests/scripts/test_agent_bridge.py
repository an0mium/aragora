"""Tests for scripts/agent_bridge.py."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _reload_agent_bridge():
    for name in ("scripts.agent_bridge", "agent_bridge_sessions"):
        sys.modules.pop(name, None)
    return importlib.import_module("scripts.agent_bridge")


def test_package_import_loads_bridge_module() -> None:
    mod = _reload_agent_bridge()

    bridge = mod._load_bridge()

    assert bridge is not None
    assert hasattr(bridge, "collect_sessions")


def test_discover_falls_back_when_bridge_module_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _reload_agent_bridge()
    tmux_dir = tmp_path / "tmux"
    tmux_dir.mkdir()
    (tmux_dir / "codex-lane.meta.json").write_text(
        json.dumps({"name": "codex-lane", "agent": "codex"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "_bridge_mod", None)
    monkeypatch.setattr(mod, "_load_bridge", lambda: None)
    monkeypatch.setattr(mod, "TMUX_SESSIONS_DIR", tmux_dir)

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["tmux"],
            returncode=0,
            stdout="codex-lane\n",
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    sessions = mod.discover()

    assert len(sessions) == 1
    assert sessions[0].name == "codex-lane"
    assert sessions[0].agent == "codex"
    assert sessions[0].status == "alive"
    assert sessions[0].tmux_target == "aragora:codex-lane"
