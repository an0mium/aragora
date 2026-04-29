"""Tests for scripts/agent_bridge.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


@pytest.fixture(autouse=True)
def _setup_path():
    sys.path.insert(0, str(SCRIPTS_DIR))
    yield
    sys.path.remove(str(SCRIPTS_DIR))


def _patch_bridge_paths(mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bridge_dir = tmp_path / "bridge"
    monkeypatch.setattr(mod, "AGENT_BRIDGE_DIR", bridge_dir)
    monkeypatch.setattr(mod, "SESSION_SNAPSHOT_FILE", bridge_dir / "sessions.json")
    monkeypatch.setattr(mod, "LANE_REGISTRY_FILE", bridge_dir / "lanes.json")


def test_cmd_send_persists_lane_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    session = mod.Session(
        name="codex-strategic",
        agent="codex",
        status="alive",
        tmux_target="aragora:codex-strategic",
        branch="codex/issue-5320",
        worktree="/tmp/aragora-5320",
    )
    monkeypatch.setattr(mod, "discover", lambda: [session])
    monkeypatch.setattr(mod, "_resolve_tmux_target", lambda _session: "aragora:codex-strategic")
    monkeypatch.setattr(mod, "_send_tmux", lambda _target, _prompt: True)

    def _enrich_prs(sessions):
        sessions[0].pr_number = 5401

    monkeypatch.setattr(mod, "_enrich_prs", _enrich_prs)

    args = argparse.Namespace(
        name="codex-strategic",
        prompt=["Continue", "#5320"],
        file=None,
        lane="bridge-hardening",
        goal="Persist lane registry",
        source="#5320",
        status="active",
        next_action="open PR",
        allow_conflict=False,
    )
    rc = mod.cmd_send(args)

    assert rc == 0
    payload = json.loads(mod.LANE_REGISTRY_FILE.read_text(encoding="utf-8"))
    assert payload == [
        {
            "lane_id": "bridge-hardening",
            "owner_session": "codex-strategic",
            "goal": "Persist lane registry",
            "source": "#5320",
            "status": "active",
            "next_action": "open PR",
            "updated_at": payload[0]["updated_at"],
            "branch": "codex/issue-5320",
            "worktree": "/tmp/aragora-5320",
            "pr_number": 5401,
        }
    ]


def test_cmd_send_rejects_active_lane_owner_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    mod.AGENT_BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    mod.LANE_REGISTRY_FILE.write_text(
        json.dumps(
            [
                {
                    "lane_id": "bridge-hardening",
                    "owner_session": "other-session",
                    "status": "active",
                    "updated_at": "2026-04-13T21:20:00+00:00",
                }
            ]
        ),
        encoding="utf-8",
    )

    session = mod.Session(name="codex-strategic", agent="codex", status="alive")
    monkeypatch.setattr(mod, "discover", lambda: [session])
    monkeypatch.setattr(mod, "_resolve_tmux_target", lambda _session: "aragora:codex-strategic")
    monkeypatch.setattr(mod, "_enrich_prs", lambda _sessions: None)

    def _unexpected_send(_target, _prompt):
        raise AssertionError("send should not run when lane ownership conflicts")

    monkeypatch.setattr(mod, "_send_tmux", _unexpected_send)

    args = argparse.Namespace(
        name="codex-strategic",
        prompt=["Continue"],
        file=None,
        lane="bridge-hardening",
        goal="",
        source="",
        status="active",
        next_action="",
        allow_conflict=False,
    )
    rc = mod.cmd_send(args)

    assert rc == 1
    assert "already owned by active session 'other-session'" in capsys.readouterr().err
    payload = json.loads(mod.LANE_REGISTRY_FILE.read_text(encoding="utf-8"))
    assert payload[0]["owner_session"] == "other-session"
    assert payload[0]["status"] == "active"


def test_cmd_send_allow_conflict_marks_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    mod.AGENT_BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    mod.LANE_REGISTRY_FILE.write_text(
        json.dumps(
            [
                {
                    "lane_id": "bridge-hardening",
                    "owner_session": "other-session",
                    "goal": "Persist lane registry",
                    "source": "#5320",
                    "status": "active",
                    "updated_at": "2026-04-13T21:20:00+00:00",
                    "branch": "codex/old",
                    "worktree": "/tmp/old",
                    "pr_number": 5399,
                }
            ]
        ),
        encoding="utf-8",
    )

    session = mod.Session(
        name="codex-strategic",
        agent="codex",
        status="alive",
        tmux_target="aragora:codex-strategic",
        branch="codex/issue-5320",
        worktree="/tmp/aragora-5320",
    )
    monkeypatch.setattr(mod, "discover", lambda: [session])
    monkeypatch.setattr(mod, "_resolve_tmux_target", lambda _session: "aragora:codex-strategic")
    monkeypatch.setattr(mod, "_send_tmux", lambda _target, _prompt: True)
    monkeypatch.setattr(mod, "_enrich_prs", lambda _sessions: None)

    args = argparse.Namespace(
        name="codex-strategic",
        prompt=["Continue"],
        file=None,
        lane="bridge-hardening",
        goal="",
        source="",
        status="active",
        next_action="triage conflicting ownership",
        allow_conflict=True,
    )
    rc = mod.cmd_send(args)

    assert rc == 0
    payload = json.loads(mod.LANE_REGISTRY_FILE.read_text(encoding="utf-8"))
    assert payload[0]["owner_session"] == "other-session"
    assert payload[0]["status"] == "conflict"
    assert payload[0]["conflict_session"] == "codex-strategic"
    assert payload[0]["conflict_reason"] == "conflicting active owner claim from codex-strategic"
    assert payload[0]["next_action"] == "triage conflicting ownership"


def test_cmd_lanes_json_prefers_registry_and_syncs_live_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    mod.AGENT_BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    mod.LANE_REGISTRY_FILE.write_text(
        json.dumps(
            [
                {
                    "lane_id": "bridge-hardening",
                    "owner_session": "codex-strategic",
                    "goal": "Persist lane registry",
                    "source": "#5320",
                    "status": "active",
                    "updated_at": "2026-04-13T21:20:00+00:00",
                    "branch": "stale-branch",
                    "worktree": "/tmp/stale",
                    "pr_number": 5300,
                }
            ]
        ),
        encoding="utf-8",
    )

    session = mod.Session(
        name="codex-strategic",
        agent="codex",
        status="alive",
        branch="codex/issue-5320",
        worktree="/tmp/aragora-5320",
    )
    monkeypatch.setattr(mod, "discover", lambda: [session])

    def _enrich_prs(sessions):
        sessions[0].pr_number = 5402

    monkeypatch.setattr(mod, "_enrich_prs", _enrich_prs)

    rc = mod.cmd_lanes(argparse.Namespace(json=True))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "lane_id": "bridge-hardening",
            "owner_session": "codex-strategic",
            "goal": "Persist lane registry",
            "source": "#5320",
            "status": "active",
            "updated_at": payload[0]["updated_at"],
            "branch": "codex/issue-5320",
            "worktree": "/tmp/aragora-5320",
            "pr_number": 5402,
        }
    ]


def test_main_accepts_json_after_subcommand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "discover", lambda: [])
    monkeypatch.setattr(mod, "_write_session_snapshot", lambda _sessions: None)
    monkeypatch.setattr(sys, "argv", ["agent_bridge.py", "sessions", "--json"])

    assert mod.main() == 0
    assert json.loads(capsys.readouterr().out) == []


def test_health_ignores_dead_root_checkout_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "CANONICAL_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        mod,
        "discover",
        lambda: [
            mod.Session(
                name="codex-old-root",
                agent="codex",
                status="dead",
                worktree=str(tmp_path),
            )
        ],
    )
    monkeypatch.setattr(mod, "_enrich_prs", lambda _sessions: None)
    monkeypatch.setattr(mod, "_load_lane_registry", lambda: [])
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: argparse.Namespace(returncode=1, stdout="", stderr=""),
    )

    assert mod.cmd_health(argparse.Namespace(json=True)) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "issues": []}


def test_health_reports_dead_non_root_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    root = tmp_path / "repo"
    worktree = tmp_path / "old-worktree"
    root.mkdir()
    worktree.mkdir()
    monkeypatch.setattr(mod, "REPO_ROOT", root)
    monkeypatch.setattr(mod, "CANONICAL_REPO_ROOT", root)
    monkeypatch.setattr(
        mod,
        "discover",
        lambda: [
            mod.Session(
                name="codex-old-lane",
                agent="codex",
                status="dead",
                worktree=str(worktree),
            )
        ],
    )
    monkeypatch.setattr(mod, "_enrich_prs", lambda _sessions: None)
    monkeypatch.setattr(mod, "_load_lane_registry", lambda: [])
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: argparse.Namespace(returncode=1, stdout="", stderr=""),
    )

    assert mod.cmd_health(argparse.Namespace(json=True)) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["issues"] == [
        {
            "type": "stale_worktree",
            "session": "codex-old-lane",
            "detail": f"dead session with lingering worktree: {worktree}",
        }
    ]
