"""Tests for scripts/agent_bridge.py."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
    monkeypatch.setattr(mod, "CANONICAL_REPO_ROOT", tmp_path / "repo")


def test_send_tmux_multiline_uses_delete_on_paste_buffer_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_bridge as mod

    calls: list[tuple[list[str], str | None]] = []
    sleeps: list[float] = []

    def _fake_run(
        args: list[str],
        *,
        input: str | None = None,
        text: bool | None = None,
        check: bool | None = None,
        timeout: int | None = None,
        **_kwargs,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((args, input))
        assert check is True
        assert timeout == 5
        if args == ["tmux", "load-buffer", "-"]:
            assert text is True
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    monkeypatch.setenv("ARAGORA_TMUX_PASTE_SETTLE_SECONDS", "0.01")
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: sleeps.append(seconds))

    assert mod._send_tmux("aragora:codex-review", "line one\nline two") is True
    assert sleeps == [0.01]
    assert calls == [
        (["tmux", "load-buffer", "-"], "line one\nline two"),
        (["tmux", "paste-buffer", "-d", "-t", "aragora:codex-review"], None),
        (["tmux", "send-keys", "-t", "aragora:codex-review", "Enter"], None),
    ]


def test_lane_record_preserves_desktop_identity_metadata() -> None:
    import agent_bridge as mod

    record = mod.LaneRecord.from_dict(
        {
            "lane_id": "codex-b-review",
            "owner_session": "codex-B",
            "status": "active",
            "desktop_label": "Codex B",
            "codex_thread_id": "019e-test-thread",
            "codex_rollout_path": "/Users/armand/.codex/sessions/rollout.jsonl",
            "session_title": "Review #7286",
        }
    )

    payload = record.to_dict()
    assert payload["desktop_label"] == "Codex B"
    assert payload["codex_thread_id"] == "019e-test-thread"
    assert payload["codex_rollout_path"].endswith("rollout.jsonl")
    assert payload["session_title"] == "Review #7286"


def test_cmd_approve_droid_uses_enter_menu_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_bridge as mod

    session = mod.Session(
        name="factory-review",
        agent="droid",
        status="alive",
        tmux_target="aragora:factory-review",
    )
    calls: list[list[str]] = []

    def _fake_run(args: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(mod, "discover", lambda: [session])
    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    rc = mod.cmd_approve(argparse.Namespace(name="factory-review", json=False))

    assert rc == 0
    assert calls == [["tmux", "send-keys", "-t", "aragora:factory-review", "Enter"]]


def test_cmd_send_persists_lane_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    mod.LANE_REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    mod.LANE_REGISTRY_FILE.write_text("[]", encoding="utf-8")
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


def test_operator_snapshot_summary_only_json_omits_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    discover_include_summaries: list[bool] = []

    def fake_discover(
        *, include_summaries: bool = True, include_historical: bool = True, **_kwargs
    ):
        discover_include_summaries.append(include_summaries)
        assert include_historical is False
        return [
            mod.Session(
                name="codex-main",
                agent="codex",
                status="alive",
                lifecycle="live",
                branch="codex/example",
                worktree=str(tmp_path),
            )
        ]

    monkeypatch.setattr(mod, "discover", fake_discover)
    monkeypatch.setattr(
        mod,
        "_enrich_prs",
        lambda _sessions: (_ for _ in ()).throw(
            AssertionError("summary-only should not call GitHub PR enrichment")
        ),
    )
    monkeypatch.setattr(
        mod,
        "_write_session_snapshot",
        lambda _sessions: (_ for _ in ()).throw(
            AssertionError("summary-only should not overwrite detailed session snapshots")
        ),
    )
    monkeypatch.setattr(
        mod,
        "_collect_agent_process_census",
        lambda *, include_records=True, record_limit=None, ps_lines=None: {
            "ok": True,
            "total": 0,
            "by_role": {},
        },
    )

    rc = mod.cmd_operator_snapshot(argparse.Namespace(json=True, summary_only=True))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "sessions" not in payload
    assert "lanes" not in payload
    assert payload["records_omitted"] is True
    assert payload["summary"]["total_sessions"] == 1
    assert payload["summary"]["alive_sessions"] == 1
    assert payload["summary"]["live_sessions"] == 1
    assert payload["summary"]["historical_sessions"] == 0
    assert payload["summary"]["active_processes"] == 0
    assert payload["summary"]["active_process_roles"] == []
    assert payload["process_census"] == {"ok": True, "total": 0, "by_role": {}}
    assert payload["health"] == {"ok": True, "issues": []}
    assert discover_include_summaries == [False]


def test_operator_snapshot_summary_counts_repo_local_lane_when_user_registry_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    user_bridge_dir = tmp_path / "user-bridge"
    repo_root = tmp_path / "repo"
    repo_bridge_dir = repo_root / ".aragora" / "agent-bridge"
    user_bridge_dir.mkdir(parents=True)
    repo_bridge_dir.mkdir(parents=True)
    monkeypatch.delenv("ARAGORA_AUTOMATION_STATE_ROOT", raising=False)
    monkeypatch.setattr(mod, "AGENT_BRIDGE_DIR", user_bridge_dir)
    monkeypatch.setattr(mod, "SESSION_SNAPSHOT_FILE", user_bridge_dir / "sessions.json")
    monkeypatch.setattr(mod, "LANE_REGISTRY_FILE", user_bridge_dir / "lanes.json")
    monkeypatch.setattr(mod, "CANONICAL_REPO_ROOT", repo_root)
    mod.LANE_REGISTRY_FILE.write_text(
        json.dumps(
            [
                {
                    "lane_id": "stale-user-lane",
                    "owner_session": "codex-old",
                    "status": "completed",
                    "updated_at": "2026-05-18T12:00:00Z",
                }
            ]
        ),
        encoding="utf-8",
    )
    (repo_bridge_dir / "lanes.json").write_text(
        json.dumps(
            [
                {
                    "lane_id": "repo-local-active",
                    "owner_session": "codex-active",
                    "status": "active",
                    "updated_at": "2026-05-18T12:10:00Z",
                    "branch": "codex/repo-local-active",
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "discover", lambda **_kwargs: [])
    monkeypatch.setattr(
        mod,
        "_collect_agent_process_census",
        lambda *, include_records=True, record_limit=None, ps_lines=None: {
            "ok": True,
            "total": 0,
            "by_role": {},
        },
    )

    rc = mod.cmd_operator_snapshot(argparse.Namespace(json=True, summary_only=True))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "lanes" not in payload
    assert payload["summary"]["active_lanes"] == 1


def test_operator_snapshot_counts_active_duplicate_pr_lanes_as_conflicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    mod.AGENT_BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    mod.LANE_REGISTRY_FILE.write_text(
        json.dumps(
            [
                {
                    "lane_id": "lane-a",
                    "owner_session": "codex-A",
                    "status": "active",
                    "pr_number": 7245,
                    "branch": "worktree-codex-insights",
                },
                {
                    "lane_id": "lane-b",
                    "owner_session": "codex-B",
                    "status": "active",
                    "pr_number": 7245,
                    "branch": "worktree-codex-insights",
                },
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "discover", lambda **_kwargs: [])
    monkeypatch.setattr(
        mod,
        "_collect_agent_process_census",
        lambda *, include_records=True, record_limit=None, ps_lines=None: {
            "ok": True,
            "total": 0,
            "by_role": {},
        },
    )

    rc = mod.cmd_operator_snapshot(argparse.Namespace(json=True, summary_only=True))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["conflict_lanes"] == 2
    assert payload["lane_conflicts"][0]["key_kind"] == "branch"
    assert payload["health"]["ok"] is False
    assert payload["health"]["issues"][0]["type"] == "lane_identity_conflict"


def test_collect_agent_process_census_redacts_commands_and_counts_roles() -> None:
    import agent_bridge as mod

    payload = mod._collect_agent_process_census(
        ps_lines=[
            " 101 01:02:03 bash /repo/scripts/run_boss_cycle.sh --token sk-secret",
            " 102 00:03:04 python3 scripts/codex_worktree_value_inventory.py --write-ledger",
            " 103 00:00:05 node /opt/homebrew/bin/codex --yolo",
            " 104 00:00:01 python3 scripts/agent_bridge.py processes --json",
            "bad-line",
        ]
    )

    assert payload["ok"] is True
    assert payload["total"] == 3
    assert payload["by_role"] == {
        "boss_cycle": 1,
        "codex_cli": 1,
        "worktree_inventory": 1,
    }
    assert [record["role"] for record in payload["records"]] == [
        "boss_cycle",
        "codex_cli",
        "worktree_inventory",
    ]
    assert all("command" not in record for record in payload["records"])
    assert "sk-secret" not in json.dumps(payload)


def test_collect_agent_process_census_keeps_total_when_records_limited() -> None:
    import agent_bridge as mod

    payload = mod._collect_agent_process_census(
        record_limit=1,
        ps_lines=[
            " 101 01:02:03 bash /repo/scripts/run_boss_cycle.sh",
            " 102 00:03:04 python3 scripts/codex_worktree_value_inventory.py",
        ],
    )

    assert payload["total"] == 2
    assert len(payload["records"]) == 1
    assert payload["records_omitted"] == 1


def test_session_lifecycle_classifies_claude_transcripts_as_historical() -> None:
    import agent_bridge as mod

    lifecycle = mod._session_lifecycle(
        source="claude_jsonl",
        status="unknown",
        updated_at="2026-05-15T00:00:00Z",
        session_id="claude-session",
    )

    assert lifecycle == "historical"
    assert mod._session_status_for_lifecycle("unknown", lifecycle) == "historical"


def test_discover_excludes_historical_transcripts_by_default_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_bridge as mod

    record = SimpleNamespace(
        name="claude-deadbeef",
        agent="claude",
        status="unknown",
        source="claude_jsonl",
        tmux_target="",
        branch="main",
        cwd="/tmp/old",
        session_id="deadbeef",
        updated_at="2026-05-15T00:00:00Z",
        summary="old desktop chat",
        log_file=None,
        transcript_file="/tmp/claude.jsonl",
    )
    monkeypatch.setattr(
        mod.agent_bridge_sessions,
        "collect_sessions",
        lambda **_kwargs: [record],
    )

    assert mod.discover(include_historical=False) == []
    all_sessions = mod.discover(include_historical=True)
    assert all_sessions[0].status == "historical"
    assert all_sessions[0].lifecycle == "historical"


def test_discover_keeps_active_broker_session_current(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_bridge as mod

    record = SimpleNamespace(
        name="droid-broker",
        agent="droid",
        status="dead",
        source="tmux",
        tmux_target="",
        branch="codex/bridge",
        cwd="/tmp/bridge",
        session_id="broker-session",
        updated_at="2026-05-13T00:00:00Z",
        summary="broker-owned droid lane",
        log_file=None,
        transcript_file=None,
    )
    monkeypatch.setattr(
        mod.agent_bridge_sessions,
        "collect_sessions",
        lambda **_kwargs: [record],
    )

    sessions = mod.discover(
        include_historical=False,
        active_broker_session_ids={"broker-session"},
    )

    assert len(sessions) == 1
    assert sessions[0].status == "active_broker"
    assert sessions[0].lifecycle == "active_broker"


def test_operator_snapshot_include_historical_restores_transcript_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)

    def fake_discover(*, include_historical: bool, **_kwargs):
        if not include_historical:
            return []
        return [
            mod.Session(
                name="claude-history",
                agent="claude",
                status="historical",
                source="claude_jsonl",
                lifecycle="historical",
            )
        ]

    monkeypatch.setattr(mod, "discover", fake_discover)
    monkeypatch.setattr(mod, "_enrich_prs", lambda _sessions: None)
    monkeypatch.setattr(mod, "_load_lane_registry", lambda: [])
    monkeypatch.setattr(mod, "_load_broker_run_summaries", lambda: [])
    monkeypatch.setattr(
        mod,
        "_collect_agent_process_census",
        lambda *, include_records=True, record_limit=None, ps_lines=None: {
            "ok": True,
            "total": 0,
            "by_role": {},
            **({"records": []} if include_records else {}),
        },
    )

    assert (
        mod.cmd_operator_snapshot(
            argparse.Namespace(
                json=True,
                summary_only=False,
                include_historical=True,
                scope="current",
            )
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["historical_sessions"] == 1
    assert payload["sessions"][0]["name"] == "claude-history"


def test_operator_snapshot_current_output_preserves_full_canonical_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    bridge_dir = tmp_path / "bridge"
    _patch_bridge_paths(mod, tmp_path, monkeypatch)

    def fake_discover(*, include_historical: bool, **_kwargs):
        assert include_historical is True
        return [
            mod.Session(
                name="codex-live",
                agent="codex",
                status="alive",
                lifecycle="live",
            ),
            mod.Session(
                name="claude-history",
                agent="claude",
                status="historical",
                source="claude_jsonl",
                lifecycle="historical",
            ),
        ]

    monkeypatch.setattr(mod, "discover", fake_discover)
    monkeypatch.setattr(mod, "_enrich_prs", lambda _sessions: None)
    monkeypatch.setattr(mod, "_load_lane_registry", lambda: [])
    monkeypatch.setattr(mod, "_load_broker_run_summaries", lambda: [])
    monkeypatch.setattr(
        mod,
        "_collect_agent_process_census",
        lambda *, include_records=True, record_limit=None, ps_lines=None: {
            "ok": True,
            "total": 1,
            "by_role": {"boss_cycle": 1},
            **(
                {
                    "records": [
                        {
                            "pid": 101,
                            "elapsed": "00:01",
                            "role": "boss_cycle",
                            "summary": "boss-loop control process",
                        }
                    ]
                }
                if include_records
                else {}
            ),
        },
    )

    assert (
        mod.cmd_operator_snapshot(
            argparse.Namespace(
                json=True,
                summary_only=False,
                include_historical=False,
                scope="current",
            )
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert [session["name"] for session in payload["sessions"]] == ["codex-live"]
    assert payload["summary"]["active_processes"] == 1
    assert payload["summary"]["active_process_roles"] == ["boss_cycle"]
    assert payload["summary"]["historical_sessions"] == 0
    snapshot = json.loads((bridge_dir / "sessions.json").read_text(encoding="utf-8"))
    assert [session["name"] for session in snapshot] == ["codex-live", "claude-history"]


def test_operator_snapshot_includes_broker_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "discover", lambda **_kwargs: [])
    monkeypatch.setattr(mod, "_enrich_prs", lambda _sessions: None)
    monkeypatch.setattr(mod, "_load_lane_registry", lambda: [])
    monkeypatch.setattr(
        mod,
        "_load_broker_run_summaries",
        lambda: [
            {
                "run_id": "bridge-next-work",
                "status": "running",
                "updated_at": "2026-05-15T15:00:00Z",
                "next_actor": "critic",
                "last_turn_index": 1,
                "participants": [],
                "sessions": {},
            }
        ],
    )

    assert (
        mod.cmd_operator_snapshot(
            argparse.Namespace(
                json=True,
                summary_only=False,
                include_historical=False,
                scope="current",
            )
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["active_broker_runs"] == 1
    assert payload["broker_runs"][0]["run_id"] == "bridge-next-work"


def test_cmd_launch_invokes_tmux_launcher_for_droid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    repo_root = tmp_path / "repo"
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True)
    launcher = scripts_dir / "tmux_session_launcher.sh"
    launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    review_worktree = tmp_path / "review-worktree"
    review_worktree.mkdir()
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("review only\n", encoding="utf-8")
    monkeypatch.setattr(mod, "CANONICAL_REPO_ROOT", repo_root)

    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return argparse.Namespace(returncode=0, stdout="launched\n", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    rc = mod.cmd_launch(
        argparse.Namespace(
            name="factory-review",
            agent="droid",
            prompt=[],
            file=str(prompt_file),
            cwd=str(review_worktree),
            autonomous=False,
            timeout_seconds=10,
            json=False,
        )
    )

    assert rc == 0
    assert capsys.readouterr().out == "launched\n"
    assert calls == [
        (
            [
                "bash",
                str(launcher),
                "--name",
                "factory-review",
                "--agent",
                "droid",
                "--cwd",
                str(review_worktree),
                "--prompt-file",
                str(prompt_file),
            ],
            {
                "cwd": str(repo_root),
                "capture_output": False,
                "text": True,
                "timeout": 30,
                "check": False,
            },
        )
    ]


def test_write_session_snapshot_falls_back_to_state_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_bridge as mod

    blocked_dir = tmp_path / "home" / ".aragora" / "agent-bridge"
    canonical_root = tmp_path / "repo"
    canonical_root.mkdir()
    monkeypatch.setattr(mod, "AGENT_BRIDGE_DIR", blocked_dir)
    monkeypatch.setattr(mod, "SESSION_SNAPSHOT_FILE", blocked_dir / "sessions.json")
    monkeypatch.setattr(mod, "CANONICAL_REPO_ROOT", canonical_root)
    monkeypatch.delenv("ARAGORA_AGENT_BRIDGE_DIR", raising=False)
    monkeypatch.delenv("ARAGORA_AUTOMATION_STATE_ROOT", raising=False)

    def _fake_writable_dir(path: Path) -> None:
        if path == blocked_dir:
            raise PermissionError("sandbox denied home bridge state")
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mod, "_assert_writable_dir", _fake_writable_dir)

    mod._write_session_snapshot([mod.Session(name="codex-main", agent="codex")])

    fallback_file = canonical_root / ".aragora" / "agent-bridge" / "sessions.json"
    payload = json.loads(fallback_file.read_text(encoding="utf-8"))
    assert payload[0]["name"] == "codex-main"
    assert not (blocked_dir / "sessions.json").exists()


def test_write_session_snapshot_accepts_direct_dot_aragora_state_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_bridge as mod

    blocked_dir = tmp_path / "home" / ".aragora" / "agent-bridge"
    state_root = tmp_path / "shared" / ".aragora"
    monkeypatch.setattr(mod, "AGENT_BRIDGE_DIR", blocked_dir)
    monkeypatch.setattr(mod, "SESSION_SNAPSHOT_FILE", blocked_dir / "sessions.json")
    monkeypatch.setenv("ARAGORA_AUTOMATION_STATE_ROOT", str(state_root))
    monkeypatch.delenv("ARAGORA_AGENT_BRIDGE_DIR", raising=False)

    def _fake_writable_dir(path: Path) -> None:
        if path == blocked_dir:
            raise PermissionError("sandbox denied home bridge state")
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mod, "_assert_writable_dir", _fake_writable_dir)

    mod._write_session_snapshot([mod.Session(name="codex-shared", agent="codex")])

    fallback_file = state_root / "agent-bridge" / "sessions.json"
    payload = json.loads(fallback_file.read_text(encoding="utf-8"))
    assert payload[0]["name"] == "codex-shared"
    assert not (blocked_dir / "sessions.json").exists()


def test_write_session_snapshot_uses_per_write_tempfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    temp_paths: list[Path] = []
    original_mkstemp = mod.tempfile.mkstemp

    def _recording_mkstemp(*args, **kwargs):
        fd, name = original_mkstemp(*args, **kwargs)
        temp_paths.append(Path(name))
        return fd, name

    monkeypatch.setattr(mod.tempfile, "mkstemp", _recording_mkstemp)

    mod._write_session_snapshot([mod.Session(name="codex-main", agent="codex")])

    assert len(temp_paths) == 1
    assert temp_paths[0].parent == tmp_path / "bridge"
    assert temp_paths[0].name.startswith(".sessions.json.")
    assert temp_paths[0].name.endswith(".tmp")
    assert temp_paths[0].name != "sessions.json.tmp"
    assert not temp_paths[0].exists()
    payload = json.loads((tmp_path / "bridge" / "sessions.json").read_text())
    assert payload[0]["name"] == "codex-main"


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


def test_health_ignores_dead_tmux_session_kept_current_by_broker_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    root = tmp_path / "repo"
    worktree = tmp_path / "broker-worktree"
    root.mkdir()
    worktree.mkdir()
    monkeypatch.setattr(mod, "REPO_ROOT", root)
    monkeypatch.setattr(mod, "CANONICAL_REPO_ROOT", root)
    monkeypatch.setattr(
        mod,
        "_load_broker_run_summaries",
        lambda: [
            {
                "run_id": "broker-run",
                "status": "running",
                "sessions": {"critic": {"session_id": "broker-session"}},
            }
        ],
    )
    monkeypatch.setattr(
        mod.agent_bridge_sessions,
        "collect_sessions",
        lambda **_kwargs: [
            SimpleNamespace(
                name="droid-broker",
                agent="droid",
                status="dead",
                source="tmux",
                branch="codex/bridge",
                cwd=str(worktree),
                session_id="broker-session",
                updated_at="2026-05-13T00:00:00Z",
                summary="broker-owned droid lane",
                log_file=None,
                transcript_file=None,
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


def test_health_ignores_dead_session_with_removed_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    root = tmp_path / "repo"
    removed_worktree = tmp_path / "already-removed"
    root.mkdir()
    monkeypatch.setattr(mod, "REPO_ROOT", root)
    monkeypatch.setattr(mod, "CANONICAL_REPO_ROOT", root)
    monkeypatch.setattr(
        mod,
        "discover",
        lambda: [
            mod.Session(
                name="codex-finished-lane",
                agent="codex",
                status="dead",
                worktree=str(removed_worktree),
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


def test_health_ignores_orphan_claude_transcript_missing_worktree(tmp_path: Path) -> None:
    import agent_bridge as mod

    removed_worktree = tmp_path / "removed-review-worktree"

    issues = mod._collect_health_issues(
        [
            mod.Session(
                name="claude-review",
                agent="claude",
                status="unknown",
                source="claude_jsonl",
                worktree=str(removed_worktree),
            )
        ],
        [],
    )

    assert issues == []


def test_health_reports_claimed_claude_transcript_missing_worktree(tmp_path: Path) -> None:
    import agent_bridge as mod

    removed_worktree = tmp_path / "removed-review-worktree"

    issues = mod._collect_health_issues(
        [
            mod.Session(
                name="claude-review",
                agent="claude",
                status="unknown",
                source="claude_jsonl",
                worktree=str(removed_worktree),
            )
        ],
        [
            mod.LaneRecord(
                lane_id="review",
                owner_session="claude-review",
                status="active",
            )
        ],
    )

    assert issues == [
        {
            "type": "stale_worktree",
            "session": "claude-review",
            "detail": f"worktree path missing: {removed_worktree}",
        }
    ]


def test_gc_dry_run_archives_only_bridge_owned_tmux_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    tmux_dir = tmp_path / "tmux"
    tmux_dir.mkdir()
    meta = tmux_dir / "factory-old.meta.json"
    log = tmux_dir / "factory-old.log"
    meta.write_text("{}", encoding="utf-8")
    log.write_text("old log", encoding="utf-8")
    transcript = tmp_path / "claude.jsonl"
    transcript.write_text("external transcript", encoding="utf-8")
    monkeypatch.setattr(mod, "TMUX_SESSIONS_DIR", tmux_dir)
    monkeypatch.setattr(
        mod.agent_bridge_sessions,
        "load_tmux_sessions",
        lambda **_kwargs: [
            SimpleNamespace(
                name="factory-old",
                source="tmux",
                status="dead",
                updated_at="2026-05-13T00:00:00Z",
                session_id="factory-old",
                log_file=str(log),
            )
        ],
    )

    rc = mod.cmd_gc(argparse.Namespace(json=True, write=False, ttl_hours=24))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["external_transcripts_touched"] is False
    assert payload["actions"][0]["name"] == "factory-old"
    assert meta.exists()
    assert log.exists()
    assert transcript.exists()


def test_gc_dry_run_skips_stale_tmux_session_kept_current_by_broker_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    tmux_dir = tmp_path / "tmux"
    tmux_dir.mkdir()
    meta = tmux_dir / "factory-broker.meta.json"
    log = tmux_dir / "factory-broker.log"
    meta.write_text("{}", encoding="utf-8")
    log.write_text("broker log", encoding="utf-8")
    monkeypatch.setattr(mod, "TMUX_SESSIONS_DIR", tmux_dir)
    monkeypatch.setattr(
        mod,
        "_load_broker_run_summaries",
        lambda: [
            {
                "run_id": "broker-run",
                "status": "running",
                "sessions": {"critic": {"session_id": "factory-broker"}},
            }
        ],
    )
    monkeypatch.setattr(
        mod.agent_bridge_sessions,
        "load_tmux_sessions",
        lambda **_kwargs: [
            SimpleNamespace(
                name="factory-broker",
                source="tmux",
                status="dead",
                updated_at="2026-05-13T00:00:00Z",
                session_id="factory-broker",
                log_file=str(log),
            )
        ],
    )

    rc = mod.cmd_gc(argparse.Namespace(json=True, write=False, ttl_hours=24))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["actions"] == []
    assert meta.exists()
    assert log.exists()


def test_gc_write_moves_stale_tmux_files_and_rewrites_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    bridge_dir = tmp_path / "bridge"
    tmux_dir = tmp_path / "tmux"
    tmux_dir.mkdir()
    meta = tmux_dir / "factory-old.meta.json"
    log = tmux_dir / "factory-old.log"
    meta.write_text("{}", encoding="utf-8")
    log.write_text("old log", encoding="utf-8")
    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "TMUX_SESSIONS_DIR", tmux_dir)
    monkeypatch.setattr(
        mod.agent_bridge_sessions,
        "load_tmux_sessions",
        lambda **_kwargs: [
            SimpleNamespace(
                name="factory-old",
                source="tmux",
                status="dead",
                updated_at="2026-05-13T00:00:00Z",
                session_id="factory-old",
                log_file=str(log),
            )
        ],
    )
    monkeypatch.setattr(mod, "discover", lambda **_kwargs: [])

    rc = mod.cmd_gc(argparse.Namespace(json=True, write=True, ttl_hours=24))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is False
    assert not meta.exists()
    assert not log.exists()
    assert Path(payload["actions"][0]["archive_files"][0]).exists()
    assert json.loads((bridge_dir / "sessions.json").read_text(encoding="utf-8")) == []


def test_gc_write_preserves_historical_sessions_in_canonical_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    bridge_dir = tmp_path / "bridge"
    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "_gc_tmux_candidates", lambda *, ttl_hours: [])

    def fake_discover(*, include_historical: bool, include_summaries: bool = True, **_kwargs):
        assert include_historical is True
        assert include_summaries is True
        return [
            mod.Session(
                name="claude-history",
                agent="claude",
                status="historical",
                source="claude_jsonl",
                lifecycle="historical",
                summary="old desktop context",
            )
        ]

    monkeypatch.setattr(mod, "discover", fake_discover)

    rc = mod.cmd_gc(argparse.Namespace(json=True, write=True, ttl_hours=24))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is False
    snapshot = json.loads((bridge_dir / "sessions.json").read_text(encoding="utf-8"))
    assert [session["name"] for session in snapshot] == ["claude-history"]
    assert snapshot[0]["summary"] == "old desktop context"
