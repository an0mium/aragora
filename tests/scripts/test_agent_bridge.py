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
    monkeypatch.setattr(
        mod,
        "_collect_agent_heartbeats",
        lambda: {
            "count": 2,
            "fresh_count": 1,
            "stale_count": 1,
            "latest_by_owner": {
                "codex-owner": {
                    "owner_session": "codex-owner",
                    "cwd": "/tmp/large-detail",
                    "worktree": "/tmp/large-detail",
                    "last_seen_at": "2026-06-06T00:00:00Z",
                }
            },
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
    assert payload["agent_heartbeats"] == {"count": 2, "fresh_count": 1, "stale_count": 1}
    assert payload["health"] == {"ok": True, "issues": []}
    assert discover_include_summaries == [False]


def test_operator_snapshot_json_suppresses_broken_pipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    muted_stdout: list[bool] = []

    def fake_discover(
        *, include_summaries: bool = True, include_historical: bool = True, **_kwargs
    ):
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

    def broken_print(*_args, **_kwargs) -> None:
        raise BrokenPipeError("downstream closed")

    monkeypatch.setattr(mod, "discover", fake_discover)
    monkeypatch.setattr(mod, "_enrich_prs", lambda _sessions: None)
    monkeypatch.setattr(mod, "_write_session_snapshot", lambda _sessions: None)
    monkeypatch.setattr(
        mod,
        "_collect_agent_process_census",
        lambda *, include_records=True, record_limit=None, ps_lines=None: {
            "ok": True,
            "total": 0,
            "by_role": {},
        },
    )
    monkeypatch.setattr("builtins.print", broken_print)
    monkeypatch.setattr(
        mod,
        "_mute_stdout_after_broken_pipe",
        lambda: muted_stdout.append(True),
    )

    rc = mod.cmd_operator_snapshot(argparse.Namespace(json=True, summary_only=True))

    assert rc == 0
    assert muted_stdout == [True]


def test_operator_snapshot_text_suppresses_broken_pipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    muted_stdout: list[bool] = []

    def fake_discover(
        *, include_summaries: bool = True, include_historical: bool = True, **_kwargs
    ):
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

    def broken_print(*_args, **_kwargs) -> None:
        raise BrokenPipeError("downstream closed")

    monkeypatch.setattr(mod, "discover", fake_discover)
    monkeypatch.setattr(mod, "_enrich_prs", lambda _sessions: None)
    monkeypatch.setattr(mod, "_write_session_snapshot", lambda _sessions: None)
    monkeypatch.setattr(
        mod,
        "_collect_agent_process_census",
        lambda *, include_records=True, record_limit=None, ps_lines=None: {
            "ok": True,
            "total": 0,
            "by_role": {},
        },
    )
    monkeypatch.setattr("builtins.print", broken_print)
    monkeypatch.setattr(
        mod,
        "_mute_stdout_after_broken_pipe",
        lambda: muted_stdout.append(True),
    )

    rc = mod.cmd_operator_snapshot(argparse.Namespace(json=False, summary_only=True))

    assert rc == 0
    assert muted_stdout == [True]


def test_operator_snapshot_exposes_b0_issue_contract_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    repo_root = tmp_path / "repo"
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "docs" / "benchmarks").mkdir(parents=True)
    (repo_root / "scripts" / "measure_b0_scorecard.py").write_text("# fixture\n")
    (repo_root / "docs" / "benchmarks" / "corpus.json").write_text("{}\n")
    mod.AGENT_BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    mod.LANE_REGISTRY_FILE.write_text(
        json.dumps(
            [
                {
                    "lane_id": "b0-5426",
                    "owner_session": "codex-b0",
                    "status": "active",
                    "next_action": "finish operator snapshot contract",
                    "last_steering_outcome": "obeyed",
                    "last_heartbeat_at": "2026-05-28T12:00:00Z",
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        mod,
        "_discover_with_broker_state",
        lambda **_kwargs: (
            [
                mod.Session(
                    name="codex-b0",
                    agent="codex",
                    status="alive",
                    lifecycle="live",
                    session_id="session-b0",
                )
            ],
            [
                {
                    "run_id": "boss-loop",
                    "status": "running",
                    "updated_at": "2026-05-28T12:00:01Z",
                    "next_actor": "codex",
                    "last_turn_index": 3,
                    "participants": [],
                    "sessions": {},
                }
            ],
            set(),
        ),
    )
    monkeypatch.setattr(
        mod,
        "_collect_agent_process_census",
        lambda *, include_records=True, record_limit=None, ps_lines=None: {
            "ok": True,
            "total": 1,
            "by_role": {"boss_cycle": 1},
        },
    )
    monkeypatch.setattr(
        mod,
        "_collect_pending_steering_messages",
        lambda _recipient: {
            "count": 1,
            "latest_three": [
                {
                    "subject": "Need B0 action",
                    "sent_at_utc": "2026-05-28T12:01:00Z",
                    "priority": "high",
                    "lane_id_hint": "b0-5426",
                    "pr_hint": 5426,
                }
            ],
        },
    )
    monkeypatch.setattr(
        mod,
        "_collect_agent_heartbeats",
        lambda: {"count": 1, "fresh_count": 1, "stale_count": 0, "latest_by_owner": {}},
    )

    def fake_run(args: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        assert "measure_b0_scorecard.py" in args[1]
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=json.dumps({"no_rescue_success_rate": 0.625, "status": "active"}),
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    rc = mod.cmd_operator_snapshot(argparse.Namespace(json=True, summary_only=True))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["queue_depth"] == 3
    assert payload["success_rate"] == 0.625
    assert payload["boss_loop_alive"] is True
    assert payload["boss_loop_status"] == {
        "alive": True,
        "reason": "active_broker_runs",
        "active_broker_runs": 1,
        "fresh_agent_heartbeats": 1,
        "has_boss_cycle_process": True,
        "active_process_roles": ["boss_cycle"],
    }
    assert payload["recent_blockers"] == [
        {
            "type": "pending_steering",
            "source": "operator_steering",
            "detail": "Need B0 action",
            "priority": "high",
            "lane_id_hint": "b0-5426",
            "pr_hint": 5426,
        }
    ]


def test_operator_boss_loop_status_reports_idle_without_live_signal() -> None:
    import agent_bridge as mod

    status = mod._operator_boss_loop_status(
        {
            "active_broker_runs": 0,
            "fresh_agent_heartbeats": 0,
            "active_process_roles": ["publisher", "review_queue"],
        }
    )

    assert status == {
        "alive": False,
        "reason": "idle_no_live_boss_loop_signal",
        "active_broker_runs": 0,
        "fresh_agent_heartbeats": 0,
        "has_boss_cycle_process": False,
        "active_process_roles": ["publisher", "review_queue"],
    }


def test_operator_boss_loop_status_tolerates_malformed_roles() -> None:
    import agent_bridge as mod

    assert mod._operator_boss_loop_status({"active_process_roles": None}) == {
        "alive": False,
        "reason": "idle_no_live_boss_loop_signal",
        "active_broker_runs": 0,
        "fresh_agent_heartbeats": 0,
        "has_boss_cycle_process": False,
        "active_process_roles": [],
    }
    assert mod._operator_boss_loop_status({"active_process_roles": object()}) == {
        "alive": False,
        "reason": "idle_no_live_boss_loop_signal",
        "active_broker_runs": 0,
        "fresh_agent_heartbeats": 0,
        "has_boss_cycle_process": False,
        "active_process_roles": [],
    }


def test_operator_boss_loop_alive_preserves_legacy_boolean() -> None:
    import agent_bridge as mod

    assert (
        mod._operator_boss_loop_alive(
            {
                "active_broker_runs": 0,
                "fresh_agent_heartbeats": 1,
                "active_process_roles": [],
            }
        )
        is True
    )
    assert (
        mod._operator_boss_loop_alive(
            {
                "active_broker_runs": 0,
                "fresh_agent_heartbeats": 0,
                "active_process_roles": ["publisher", "review_queue"],
            }
        )
        is False
    )


def test_collect_pending_steering_messages_completed_receipts_are_not_actionable(
    tmp_path: Path,
) -> None:
    import agent_bridge as mod

    steering_root = tmp_path / "operator-steering"
    inbox = steering_root / "owner-session"
    receipts = inbox / "_read_receipts"
    receipts.mkdir(parents=True)
    message = {
        "subject": "Operator selected Option 1 for #7741",
        "sent_at_utc": "2026-06-04T15:49:57.982Z",
        "priority": "blocking",
        "lane_id_hint": "Q324-repair-build-next-prompt-merged-active-lane-handoff",
        "pr_hint": 7741,
        "message_sha256": "sha-completed",
    }
    (inbox / "2026-06-04T15-49-57-982Z-message.json").write_text(
        json.dumps(message),
        encoding="utf-8",
    )
    receipt = {
        "read_at_utc": "2026-06-04T21:32:54.925Z",
        "read_by_session": "codex-repair",
        "message_filename": "2026-06-04T15-49-57-982Z-message.json",
        "message_sha256": "sha-completed",
        "outcome": "completed",
        "subject": message["subject"],
    }
    (receipts / "2026-06-04T21-32-54-925Z-receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )

    payload = mod._collect_pending_steering_messages(None, steering_root)

    assert payload["count"] == 1
    assert payload["by_recipient"] == {"owner-session": 1}
    assert payload["unread_message_count"] == 0
    assert payload["unresolved_count"] == 0
    assert payload["unresolved_by_recipient"] == {}
    assert payload["latest_three"][0]["subject"] == message["subject"]
    assert payload["latest_unresolved_three"] == []
    assert mod._operator_recent_blockers([], payload) == []
    assert mod._operator_queue_depth({"active_lanes": 2, "active_broker_runs": 0}, payload) == 2


def test_collect_pending_steering_messages_normalizes_terminal_receipt_outcomes(
    tmp_path: Path,
) -> None:
    import agent_bridge as mod

    steering_root = tmp_path / "operator-steering"
    inbox = steering_root / "owner-session"
    receipts = inbox / "_read_receipts"
    receipts.mkdir(parents=True)
    message = {
        "subject": "Completed with padded outcome",
        "sent_at_utc": "2026-06-05T12:01:00Z",
        "priority": "blocking",
        "lane_id_hint": "completed-lane",
        "pr_hint": None,
        "message_sha256": "sha-completed-padded",
    }
    (inbox / "2026-06-05T12-01-00Z-completed.json").write_text(
        json.dumps(message),
        encoding="utf-8",
    )
    receipt = {
        "read_at_utc": "2026-06-05T12:02:00Z",
        "read_by_session": "codex-worker",
        "message_filename": "2026-06-05T12-01-00Z-completed.json",
        "message_sha256": "sha-completed-padded",
        "outcome": " Completed ",
        "subject": message["subject"],
    }
    (receipts / "2026-06-05T12-02-00Z-completed-receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )

    payload = mod._collect_pending_steering_messages("owner-session", steering_root)

    assert payload["count"] == 1
    assert payload["unresolved_count"] == 0
    assert payload["latest_unresolved_three"] == []


def test_collect_pending_steering_messages_read_receipts_remain_actionable(
    tmp_path: Path,
) -> None:
    import agent_bridge as mod

    steering_root = tmp_path / "operator-steering"
    inbox = steering_root / "owner-session"
    receipts = inbox / "_read_receipts"
    receipts.mkdir(parents=True)
    message = {
        "subject": "Need B0 action",
        "sent_at_utc": "2026-05-28T12:01:00Z",
        "priority": "high",
        "lane_id_hint": "b0-5426",
        "pr_hint": 5426,
        "message_sha256": "sha-read",
    }
    (inbox / "2026-05-28T12-01-00Z-message.json").write_text(
        json.dumps(message),
        encoding="utf-8",
    )
    receipt = {
        "read_at_utc": "2026-05-28T12:02:00Z",
        "read_by_session": "codex-b0",
        "message_filename": "2026-05-28T12-01-00Z-message.json",
        "message_sha256": "sha-read",
        "outcome": "read",
        "subject": message["subject"],
    }
    (receipts / "2026-05-28T12-02-00Z-receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )

    payload = mod._collect_pending_steering_messages("owner-session", steering_root)

    assert payload["count"] == 1
    assert payload["unread_message_count"] == 0
    assert payload["unresolved_count"] == 1
    assert payload["latest_unresolved_three"][0]["subject"] == "Need B0 action"
    assert mod._operator_pending_steering_count(payload) == 1
    assert mod._operator_recent_blockers([], payload) == [
        {
            "type": "pending_steering",
            "source": "operator_steering",
            "detail": "Need B0 action",
            "priority": "high",
            "lane_id_hint": "b0-5426",
            "pr_hint": 5426,
        }
    ]


@pytest.mark.parametrize("outcome", ["blocked", "held"])
def test_collect_pending_steering_messages_blocked_or_held_receipts_remain_actionable(
    tmp_path: Path,
    outcome: str,
) -> None:
    import agent_bridge as mod

    steering_root = tmp_path / "operator-steering"
    inbox = steering_root / "owner-session"
    receipts = inbox / "_read_receipts"
    receipts.mkdir(parents=True)
    message = {
        "subject": f"Steering {outcome}",
        "sent_at_utc": "2026-06-05T12:01:00Z",
        "priority": "blocking",
        "lane_id_hint": "blocked-lane",
        "pr_hint": None,
        "message_sha256": f"sha-{outcome}",
    }
    (inbox / f"2026-06-05T12-01-00Z-{outcome}.json").write_text(
        json.dumps(message),
        encoding="utf-8",
    )
    receipt = {
        "read_at_utc": "2026-06-05T12:02:00Z",
        "read_by_session": "codex-worker",
        "message_filename": f"2026-06-05T12-01-00Z-{outcome}.json",
        "message_sha256": f"sha-{outcome}",
        "outcome": outcome,
        "subject": message["subject"],
    }
    (receipts / f"2026-06-05T12-02-00Z-{outcome}-receipt.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )

    payload = mod._collect_pending_steering_messages(None, steering_root)

    assert payload["count"] == 1
    assert payload["unresolved_count"] == 1
    assert payload["latest_unresolved_three"][0]["subject"] == f"Steering {outcome}"
    assert mod._operator_pending_steering_count(payload) == 1
    assert mod._operator_recent_blockers([], payload)[0]["detail"] == f"Steering {outcome}"


def test_collect_b0_success_rate_times_out_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_bridge as mod

    repo_root = tmp_path / "repo"
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "docs" / "benchmarks").mkdir(parents=True)
    (repo_root / "scripts" / "measure_b0_scorecard.py").write_text("# fixture\n")
    (repo_root / "docs" / "benchmarks" / "corpus.json").write_text("{}\n")
    monkeypatch.setenv("AGENT_BRIDGE_B0_SCORECARD_TIMEOUT_SECONDS", "0.25")

    def fake_run(args: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        assert "measure_b0_scorecard.py" in args[1]
        assert kwargs["timeout"] == 0.25
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    assert mod._collect_b0_success_rate(repo_root) is None


@pytest.mark.parametrize(
    ("raw_rate", "expected"),
    [
        (0, 0.0),
        (1, 1.0),
        (0.625, 0.625),
        ("0.25", 0.25),
        (None, None),
        (True, None),
        ("not-a-number", None),
        (float("nan"), None),
        (float("inf"), None),
        ("-inf", None),
        (-0.01, None),
        (1.01, None),
        ("nan", None),
        ("1.01", None),
    ],
)
def test_coerce_success_rate_rejects_non_finite_and_out_of_range_values(
    raw_rate: object,
    expected: float | None,
) -> None:
    import agent_bridge as mod

    actual = mod._coerce_success_rate(raw_rate)

    if expected is None:
        assert actual is None
    else:
        assert actual == expected


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
                    "next_action": "settle PR 7245",
                    "last_steering_outcome": "obeyed",
                    "last_heartbeat_at": "2026-05-28T12:00:00Z",
                },
                {
                    "lane_id": "lane-b",
                    "owner_session": "codex-B",
                    "status": "active",
                    "pr_number": 7245,
                    "branch": "worktree-codex-insights",
                    "next_action": "settle PR 7245",
                    "last_steering_outcome": "obeyed",
                    "last_heartbeat_at": "2026-05-28T12:00:01Z",
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
    issue = payload["health"]["issues"][0]
    assert issue["type"] == "lane_identity_conflict"
    assert issue["owner_state"] == "duplicate_active_owner"
    assert issue["key_kind"] == "branch"
    assert issue["key_value"] == "worktree-codex-insights"
    assert issue["owner_sessions"] == ["codex-A", "codex-B"]
    assert issue["recommended_operator_action"] == (
        "resolve duplicate active owners before mutation or cleanup"
    )


def test_health_missing_heartbeat_has_liveness_diagnostic() -> None:
    import agent_bridge as mod

    issues = mod._collect_health_issues(
        [],
        [
            mod.LaneRecord(
                lane_id="lane-no-heartbeat",
                owner_session="codex-worker",
                status="active",
                next_action="finish one safe cleanup diagnostic",
                last_steering_outcome="obeyed",
                worktree="/tmp/owned-worktree",
            )
        ],
    )

    assert issues == [
        {
            "type": "lane_missing_heartbeat",
            "session": "codex-worker",
            "detail": "active lane 'lane-no-heartbeat' has no heartbeat timestamp",
            "owner_state": "active_lane_missing_liveness",
            "lane_id": "lane-no-heartbeat",
            "status": "active",
            "worktree": "/tmp/owned-worktree",
            "heartbeat_state": "missing",
            "recommended_operator_action": (
                "start or refresh agent_heartbeat.py before treating owner as live"
            ),
        }
    ]


def test_operator_snapshot_does_not_conflict_same_owner_refreshes(
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
                    "owner_session": "codex-owner",
                    "status": "active",
                    "pr_number": 7245,
                    "branch": "worktree-codex-insights",
                    "next_action": "settle PR 7245",
                    "last_steering_outcome": "obeyed",
                    "last_heartbeat_at": "2026-05-28T12:00:00Z",
                },
                {
                    "lane_id": "lane-b",
                    "owner_session": "codex-owner",
                    "status": "active",
                    "pr_number": 7245,
                    "branch": "worktree-codex-insights",
                    "next_action": "settle PR 7245",
                    "last_steering_outcome": "obeyed",
                    "last_heartbeat_at": "2026-05-28T12:00:01Z",
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
            "total": 400,
            "by_role": {"codex_app_server": 400},
        },
    )

    rc = mod.cmd_operator_snapshot(argparse.Namespace(json=True, summary_only=True))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["active_lanes"] == 2
    assert payload["summary"]["conflict_lanes"] == 0
    assert payload["lane_conflicts"] == []
    assert payload["health"]["ok"] is True


def test_operator_snapshot_current_scope_ignores_resolved_conflict_lane(
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
                    "lane_id": "p104-stands-down",
                    "owner_session": "codex-p104",
                    "status": "conflict",
                    "conflict_session": "codex-r03",
                    "conflict_reason": "r03 owns follow-through",
                    "updated_at": "2026-05-21T18:13:49Z",
                },
                {
                    "lane_id": "r03-follow-through",
                    "owner_session": "codex-r03",
                    "status": "completed",
                    "updated_at": "2026-05-21T18:37:57Z",
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
    assert payload["summary"]["conflict_lanes"] == 0
    assert payload["health"] == {"ok": True, "issues": []}
    assert payload["lane_conflicts"] == []


def test_cmd_owner_json_reports_active_pr_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    worktree = tmp_path / "owned-worktree"
    worktree.mkdir()
    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    mod.AGENT_BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    mod.LANE_REGISTRY_FILE.write_text(
        json.dumps(
            [
                {
                    "lane_id": "q01-settle-7292",
                    "owner_session": "codex-owner",
                    "status": "active",
                    "updated_at": "2026-05-18T17:00:00Z",
                    "branch": "droid/P16-stage2",
                    "worktree": str(worktree),
                    "pr_number": 7292,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "discover", lambda **_kwargs: [])
    monkeypatch.setattr(
        mod,
        "_head_for_worktree",
        lambda path: "a" * 40 if str(path) == str(worktree) else None,
    )

    rc = mod.cmd_owner(argparse.Namespace(json=True, pr=7292, branch=None, worktree=None))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "owner_status": "owned",
        "active_owner": True,
        "lane_id": "q01-settle-7292",
        "owner_session": "codex-owner",
        "pr_number": 7292,
        "branch": "droid/P16-stage2",
        "worktree": str(worktree),
        "head": "a" * 40,
        "status": "active",
        "updated_at": "2026-05-18T17:00:00Z",
        "recommended_operator_action": "route mutation/comment work to owner_session codex-owner; non-owners should stop or request release",
    }


def test_cmd_owner_preserves_registry_identity_when_live_session_is_sparse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    worktree = tmp_path / "owned-worktree"
    worktree.mkdir()
    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    mod.AGENT_BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    mod.LANE_REGISTRY_FILE.write_text(
        json.dumps(
            [
                {
                    "lane_id": "q01-settle-7292",
                    "owner_session": "codex-owner",
                    "status": "active",
                    "updated_at": "2026-05-18T17:00:00Z",
                    "branch": "droid/P16-stage2",
                    "worktree": str(worktree),
                    "pr_number": 7292,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mod,
        "discover",
        lambda **_kwargs: [
            mod.Session(
                name="codex-owner",
                agent="codex",
                status="alive",
                lifecycle="live",
            )
        ],
    )
    monkeypatch.setattr(mod, "_enrich_prs", lambda _sessions: None)
    monkeypatch.setattr(mod, "_head_for_worktree", lambda _path: "b" * 40)

    rc = mod.cmd_owner(argparse.Namespace(json=True, pr=7292, branch=None, worktree=None))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["owner_status"] == "owned"
    assert payload["lane_id"] == "q01-settle-7292"
    assert payload["owner_session"] == "codex-owner"
    assert payload["pr_number"] == 7292
    assert payload["branch"] == "droid/P16-stage2"
    assert payload["worktree"] == str(worktree)


def test_cmd_owner_preserves_identity_when_newer_registry_row_is_sparse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    worktree = tmp_path / "owned-worktree"
    worktree.mkdir()
    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    mod.AGENT_BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    repo_registry = mod.CANONICAL_REPO_ROOT / ".aragora" / "agent-bridge" / "lanes.json"
    repo_registry.parent.mkdir(parents=True, exist_ok=True)
    mod.LANE_REGISTRY_FILE.write_text(
        json.dumps(
            [
                {
                    "lane_id": "q01-settle-7292",
                    "owner_session": "codex-owner",
                    "status": "active",
                    "updated_at": "2026-05-18T17:01:00Z",
                    "worktree": str(worktree),
                }
            ]
        ),
        encoding="utf-8",
    )
    repo_registry.write_text(
        json.dumps(
            [
                {
                    "lane_id": "q01-settle-7292",
                    "owner_session": "codex-owner",
                    "status": "active",
                    "updated_at": "2026-05-18T17:00:00Z",
                    "branch": "droid/P16-stage2",
                    "worktree": str(worktree),
                    "pr_number": 7292,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "discover", lambda **_kwargs: [])
    monkeypatch.setattr(mod, "_head_for_worktree", lambda _path: "c" * 40)

    rc = mod.cmd_owner(argparse.Namespace(json=True, pr=7292, branch=None, worktree=None))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["owner_status"] == "owned"
    assert payload["lane_id"] == "q01-settle-7292"
    assert payload["pr_number"] == 7292
    assert payload["branch"] == "droid/P16-stage2"
    assert payload["worktree"] == str(worktree)
    assert payload["updated_at"] == "2026-05-18T17:01:00Z"


def test_cmd_owner_json_reports_unowned_pr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    mod.AGENT_BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    mod.LANE_REGISTRY_FILE.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(mod, "discover", lambda **_kwargs: [])

    rc = mod.cmd_owner(argparse.Namespace(json=True, pr=7292, branch=None, worktree=None))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "owner_status": "unowned",
        "active_owner": False,
        "lane_id": None,
        "owner_session": None,
        "pr_number": 7292,
        "branch": None,
        "worktree": None,
        "head": None,
        "status": None,
        "updated_at": None,
        "recommended_operator_action": "no active owner found; claim the lane before mutation",
    }


def test_cmd_owner_json_reports_duplicate_pr_conflict(
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
                    "updated_at": "2026-05-18T17:00:00Z",
                    "branch": "feature/a",
                    "pr_number": 7292,
                },
                {
                    "lane_id": "lane-b",
                    "owner_session": "codex-B",
                    "status": "active",
                    "updated_at": "2026-05-18T17:01:00Z",
                    "branch": "feature/b",
                    "pr_number": 7292,
                },
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "discover", lambda **_kwargs: [])

    rc = mod.cmd_owner(argparse.Namespace(json=True, pr=7292, branch=None, worktree=None))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["owner_status"] == "conflict"
    assert payload["active_owner"] is True
    assert payload["lane_id"] == "lane-a,lane-b"
    assert payload["owner_session"] == "codex-A,codex-B"
    assert payload["pr_number"] == 7292
    assert payload["recommended_operator_action"] == (
        "pause duplicate mutation; resolve active owner conflict before mutation"
    )


def test_cmd_owner_json_reports_conflict_status_lane_instead_of_unowned(
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
                    "lane_id": "newer-released",
                    "owner_session": "codex-released",
                    "status": "released",
                    "updated_at": "2026-05-18T17:10:00Z",
                    "branch": "droid/P16-stage2",
                    "pr_number": 7292,
                },
                {
                    "lane_id": "q26-conflict",
                    "owner_session": "codex-q26",
                    "status": "conflict",
                    "updated_at": "2026-05-18T17:00:00Z",
                    "branch": "droid/P16-stage2",
                    "pr_number": 7292,
                    "conflict_session": "gemini-review",
                    "conflict_reason": "fresh-head validation blocker",
                },
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "discover", lambda **_kwargs: [])

    rc = mod.cmd_owner(argparse.Namespace(json=True, pr=7292, branch=None, worktree=None))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["owner_status"] == "conflict"
    assert payload["active_owner"] is False
    assert payload["lane_id"] == "q26-conflict"
    assert payload["owner_session"] == "codex-q26"
    assert payload["pr_number"] == 7292
    assert payload["branch"] == "droid/P16-stage2"
    assert payload["status"] == "conflict"
    assert payload["conflict_session"] == "gemini-review"
    assert payload["conflict_reason"] == "fresh-head validation blocker"
    assert payload["recommended_operator_action"] == (
        "resolve lane conflict before mutation: fresh-head validation blocker"
    )


def test_cmd_owner_json_prefers_active_lane_over_conflict_status_lane(
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
                    "lane_id": "q26-conflict",
                    "owner_session": "codex-q26",
                    "status": "conflict",
                    "updated_at": "2026-05-18T17:10:00Z",
                    "branch": "droid/P16-stage2",
                    "pr_number": 7292,
                    "conflict_reason": "stale blocker",
                },
                {
                    "lane_id": "q27-active",
                    "owner_session": "codex-q27",
                    "status": "active",
                    "updated_at": "2026-05-18T17:00:00Z",
                    "branch": "droid/P16-stage2",
                    "pr_number": 7292,
                },
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "discover", lambda **_kwargs: [])

    rc = mod.cmd_owner(argparse.Namespace(json=True, pr=7292, branch=None, worktree=None))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["owner_status"] == "owned"
    assert payload["lane_id"] == "q27-active"
    assert payload["owner_session"] == "codex-q27"


def test_cmd_owner_json_reports_newest_historical_match_when_unowned(
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
                    "lane_id": "older-completed",
                    "owner_session": "codex-old",
                    "status": "completed",
                    "updated_at": "2026-05-18T17:00:00Z",
                    "branch": "droid/P16-stage2",
                    "pr_number": 7292,
                },
                {
                    "lane_id": "newer-released",
                    "owner_session": "codex-new",
                    "status": "released",
                    "updated_at": "2026-05-18T17:10:00Z",
                    "branch": "droid/P16-stage2",
                    "pr_number": 7292,
                },
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "discover", lambda **_kwargs: [])

    rc = mod.cmd_owner(argparse.Namespace(json=True, pr=7292, branch=None, worktree=None))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["owner_status"] == "unowned"
    assert payload["active_owner"] is False
    assert payload["lane_id"] == "newer-released"
    assert payload["owner_session"] == "codex-new"
    assert payload["status"] == "released"
    assert payload["recommended_operator_action"] == (
        "latest matching lane is released; claim the lane before mutation"
    )


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


def test_operator_snapshot_current_scope_filters_terminal_broker_runs(
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
            {"run_id": "completed-run", "status": "completed"},
            {"run_id": "failed-run", "status": "failed"},
            {"run_id": "running-run", "status": "running"},
            {"run_id": "human-run", "status": "awaiting_human"},
        ],
    )
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
    monkeypatch.setattr(mod, "_collect_pending_steering_messages", lambda _recipient: {"count": 0})
    monkeypatch.setattr(
        mod,
        "_collect_agent_heartbeats",
        lambda: {"count": 0, "fresh_count": 0, "stale_count": 0, "latest_by_owner": {}},
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
    assert [run["run_id"] for run in payload["broker_runs"]] == ["running-run", "human-run"]
    assert payload["summary"]["active_broker_runs"] == 2


def test_load_broker_run_summaries_reads_json_without_package_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_bridge as mod

    repo_root = tmp_path / "repo"
    run_dir = repo_root / ".aragora" / "agent_bridge" / "runs" / "bridge-next-work"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "bridge-next-work",
                "status": "running",
                "updated_at": "2026-05-31T12:00:00Z",
                "next_actor": "critic",
                "last_turn_index": 2,
                "participants": [{"role": "critic", "harness": "codex"}],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "sessions.json").write_text(
        json.dumps(
            {
                "sessions": {
                    "critic": {
                        "role": "critic",
                        "session_id": "session-critic",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "CANONICAL_REPO_ROOT", repo_root)

    assert mod._load_broker_run_summaries() == [
        {
            "run_id": "bridge-next-work",
            "status": "running",
            "updated_at": "2026-05-31T12:00:00Z",
            "next_actor": "critic",
            "last_turn_index": 2,
            "participants": [{"role": "critic", "harness": "codex"}],
            "sessions": {
                "critic": {
                    "role": "critic",
                    "session_id": "session-critic",
                }
            },
        }
    ]


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


def test_cmd_launch_rejects_autonomous_droid_tmux(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    review_worktree = tmp_path / "review-worktree"
    review_worktree.mkdir()
    monkeypatch.setattr(mod, "CANONICAL_REPO_ROOT", repo_root)

    def _unexpected_run(*_args, **_kwargs):
        raise AssertionError("interactive droid tmux launcher should not run")

    monkeypatch.setattr(mod.subprocess, "run", _unexpected_run)

    rc = mod.cmd_launch(
        argparse.Namespace(
            name="factory-review",
            agent="droid",
            prompt=["review", "#7292"],
            file=None,
            cwd=str(review_worktree),
            autonomous=True,
            timeout_seconds=10,
            json=True,
        )
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["agent"] == "droid"
    assert "cannot be made autonomous" in payload["error"]
    assert "agent_bridge.py exec --agent droid --auto high" in payload["error"]


def test_cmd_exec_droid_uses_transport_auto_high(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod
    from aragora.swarm.agent_bridge.harnesses.droid import DroidTransport

    fixture = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "fixtures"
        / "agent_bridge"
        / "droid_start.json"
    ).read_text(encoding="utf-8")
    captured: dict[str, object] = {}
    commands: list[list[str]] = []

    def _fake_runner(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=fixture, stderr="")

    def _fake_create_transport(agent, *, cwd, model, harness_options):
        captured.update(
            {
                "agent": agent,
                "cwd": cwd,
                "model": model,
                "harness_options": harness_options,
            }
        )
        return DroidTransport(
            cwd=cwd,
            model=model,
            harness_options=harness_options,
            runner=_fake_runner,
            binary_resolver=lambda _: "/usr/bin/droid",
        )

    monkeypatch.setattr(mod, "create_transport", _fake_create_transport)

    rc = mod.cmd_exec(
        argparse.Namespace(
            agent="droid",
            cwd=str(tmp_path),
            model=None,
            auto="high",
            allowed_role=["reviewer"],
            file=None,
            prompt=["Review", "#7292"],
            json=True,
        )
    )

    assert rc == 0
    assert captured == {
        "agent": "droid",
        "cwd": tmp_path.resolve(),
        "model": None,
        "harness_options": {"auto": "high"},
    }
    assert commands == [
        [
            "droid",
            "exec",
            "--auto",
            "high",
            "--output-format",
            "json",
            "--cwd",
            str(tmp_path.resolve()),
            "Review #7292",
        ]
    ]
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["command"] == commands[0]
    assert payload["message_text"].startswith("Synthesized the review findings.")
    assert payload["parse_status"] == "ok"


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
    assert len(payload["issues"]) == 1
    issue = payload["issues"][0]
    assert issue["type"] == "stale_worktree"
    assert issue["session"] == "codex-old-lane"
    assert issue["detail"] == f"dead session with lingering worktree: {worktree}"
    assert issue["owner_state"] == "stale_session"
    assert issue["worktree"] == str(worktree)
    assert issue["worktree_exists"] is True
    assert issue["cleanup_state"] == "stale_lingering_worktree"
    assert issue["recommended_operator_action"] == (
        "inspect with safe_worktree_cleanup.py before any removal"
    )


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


def test_cmd_health_summary_only_json_counts_issue_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    root = tmp_path / "repo"
    missing_worktree = tmp_path / "missing-worktree"
    root.mkdir()
    monkeypatch.setattr(mod, "CANONICAL_REPO_ROOT", root)
    monkeypatch.setattr(
        mod,
        "discover",
        lambda: [
            mod.Session(
                name="codex-active",
                agent="codex",
                status="alive",
                worktree=str(missing_worktree),
            )
        ],
    )
    monkeypatch.setattr(mod, "_enrich_prs", lambda _sessions: None)
    monkeypatch.setattr(
        mod,
        "_load_lane_registry",
        lambda: [
            mod.LaneRecord(
                lane_id="Q390-health",
                owner_session="codex-active",
                status="active",
            )
        ],
    )
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: argparse.Namespace(returncode=1, stdout="", stderr=""),
    )

    rc = mod.cmd_health(argparse.Namespace(json=True, summary_only=True))

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "ok": False,
        "issue_count": 4,
        "issue_type_counts": {
            "lane_missing_heartbeat": 1,
            "lane_missing_next_action": 1,
            "lane_missing_steering_outcome": 1,
            "stale_worktree": 1,
        },
        "issue_examples": payload["issue_examples"],
        "issues_omitted": 1,
        "details_omitted": True,
    }
    assert len(payload["issue_examples"]) == 3
    assert "issues" not in payload


def test_main_accepts_health_summary_only_after_subcommand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(mod, "CANONICAL_REPO_ROOT", root)
    monkeypatch.setattr(mod, "discover", lambda: [])
    monkeypatch.setattr(mod, "_enrich_prs", lambda _sessions: None)
    monkeypatch.setattr(mod, "_load_lane_registry", lambda: [])
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: argparse.Namespace(returncode=1, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["agent_bridge.py", "health", "--json", "--summary-only"],
    )

    assert mod.main() == 0
    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "issue_count": 0,
        "issue_type_counts": {},
        "issue_examples": [],
        "issues_omitted": 0,
        "details_omitted": True,
    }


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
                next_action="finish review",
                last_steering_outcome="obeyed",
                last_heartbeat_at="2026-05-28T12:00:00Z",
            )
        ],
    )

    assert len(issues) == 1
    issue = issues[0]
    assert issue["type"] == "stale_worktree"
    assert issue["session"] == "claude-review"
    assert issue["detail"] == f"worktree path missing: {removed_worktree}"
    assert issue["owner_state"] == "active_or_current_session"
    assert issue["worktree"] == str(removed_worktree)
    assert issue["worktree_exists"] is False
    assert issue["cleanup_state"] == "missing_path_metadata"
    assert issue["recommended_operator_action"] == "verify lane ownership before pruning metadata"


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


# ---------------------------------------------------------------------------
# Launch dispatch verification (issue #8317)
# ---------------------------------------------------------------------------


class _FakeTmuxRunner:
    """Injected tmux runner: scripted capture-pane panes + recorded calls.

    ``capture_returncode`` / ``capture_raises`` simulate a failed pane capture
    (non-zero exit or subprocess error).  ``send_returncode`` simulates tmux
    rejecting the ``send-keys`` Enter nudge.
    """

    def __init__(
        self,
        panes: list[str],
        *,
        panes_after_enter: list[str] | None = None,
        capture_returncode: int = 0,
        capture_raises: BaseException | None = None,
        send_returncode: int = 0,
    ) -> None:
        self.panes = list(panes)
        self.panes_after_enter = list(panes_after_enter or [])
        self.calls: list[list[str]] = []
        self.enter_sent = False
        self.capture_returncode = capture_returncode
        self.capture_raises = capture_raises
        self.send_returncode = send_returncode

    def __call__(self, cmd: list[str]) -> SimpleNamespace:
        self.calls.append(list(cmd))
        if cmd[:2] == ["tmux", "send-keys"]:
            self.enter_sent = True
            return SimpleNamespace(returncode=self.send_returncode, stdout="")
        if self.capture_raises is not None:
            raise self.capture_raises
        if self.capture_returncode != 0:
            return SimpleNamespace(returncode=self.capture_returncode, stdout="")
        panes = self.panes_after_enter if self.enter_sent and self.panes_after_enter else self.panes
        pane = panes.pop(0) if len(panes) > 1 else panes[0]
        return SimpleNamespace(returncode=0, stdout=pane)

    def enter_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c[:2] == ["tmux", "send-keys"]]

    def capture_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c[:2] == ["tmux", "capture-pane"]]


_PROMPT = "Fix the flaky lane test.\nReport results when done."
# Marker-in-tail but NO paste placeholder -> low-confidence staged (no nudge).
_STAGED_PANE = "Claude Code v2\n> Fix the flaky lane test.\nReport results when done.\n"
# Paste placeholder present -> high-confidence staged (safe to nudge Enter).
_PLACEHOLDER_PANE = "Claude Code v2\n> [Pasted Content 5120 chars]\n"
_SUBMITTED_PANE = (
    "Acknowledged.\nWorking on it\nThinking hard\nRunning tests\nReading files\nEditing now\n"
)


def test_verify_prompt_submission_delivered_first_try_sends_no_enter() -> None:
    import agent_bridge as mod

    runner = _FakeTmuxRunner([_SUBMITTED_PANE])
    outcome = mod._verify_prompt_submission(
        "aragora:claude-lane",
        _PROMPT,
        timeout_seconds=0.3,
        runner=runner,
        sleep=lambda _s: None,
    )

    assert outcome["delivered"] is True
    assert outcome["attempts"] == 1
    assert outcome["enter_nudges"] == 0
    assert outcome["pane_target"] == "aragora:claude-lane"
    assert runner.enter_calls() == []
    assert runner.capture_calls() == [
        ["tmux", "capture-pane", "-t", "aragora:claude-lane", "-p", "-S", "-40"],
    ]


def test_verify_prompt_submission_placeholder_then_enter_then_submitted() -> None:
    import agent_bridge as mod

    # Paste placeholder -> high-confidence staged, so Enter is nudged.
    runner = _FakeTmuxRunner([_PLACEHOLDER_PANE], panes_after_enter=[_SUBMITTED_PANE])
    sleeps: list[float] = []
    outcome = mod._verify_prompt_submission(
        "aragora:claude-lane",
        _PROMPT,
        timeout_seconds=0.3,
        runner=runner,
        sleep=sleeps.append,
    )

    assert outcome["delivered"] is True
    assert outcome["enter_nudges"] == 1
    assert runner.enter_calls() == [
        ["tmux", "send-keys", "-t", "aragora:claude-lane", "Enter"],
    ]
    # Placeholder is high-confidence, so the initial poll loop breaks at once
    # (1 capture); the single Enter nudge is followed by a bounded re-poll that
    # breaks as soon as the placeholder clears (1 more capture) -> 2 captures.
    assert outcome["attempts"] == 2
    assert len(runner.capture_calls()) == 2


def test_verify_prompt_submission_placeholder_still_staged_one_enter_only() -> None:
    import agent_bridge as mod

    runner = _FakeTmuxRunner([_PLACEHOLDER_PANE])
    outcome = mod._verify_prompt_submission(
        "aragora:claude-lane",
        _PROMPT,
        timeout_seconds=0.3,
        runner=runner,
        sleep=lambda _s: None,
    )

    assert outcome["delivered"] is False
    assert outcome["enter_nudges"] == 1
    # 1 initial poll (placeholder -> break) + 2 bounded re-polls after the nudge
    # (placeholder positively persists through both) = 3 captures.
    assert outcome["attempts"] == 3
    # Exactly ONE Enter, never a second one.
    assert len(runner.enter_calls()) == 1


def test_verify_prompt_submission_marker_only_is_unverifiable_without_enter() -> None:
    import agent_bridge as mod

    # Marker-in-tail but no paste placeholder: too false-positive-prone (a
    # harness echoes the submitted prompt back as a quoted line after submit).
    # Recorded as UNVERIFIABLE (delivered=None), never a confident False, and
    # NO corrective Enter is sent -- blindly accepting could submit an unrelated
    # harness confirmation prompt.
    runner = _FakeTmuxRunner([_STAGED_PANE])
    outcome = mod._verify_prompt_submission(
        "aragora:claude-lane",
        _PROMPT,
        timeout_seconds=0.3,
        runner=runner,
        sleep=lambda _s: None,
    )

    assert outcome["delivered"] is None
    assert outcome["error"] == "marker-only"
    assert outcome["enter_nudges"] == 0
    assert runner.enter_calls() == []
    # 3 polls, no nudge, no re-verify.
    assert outcome["attempts"] == 3


def test_verify_prompt_submission_capture_nonzero_is_unverifiable() -> None:
    import agent_bridge as mod

    # tmux capture-pane returns non-zero (e.g. dead/missing pane): capture
    # failed, so submission is UNVERIFIABLE -- not silently delivered=True.
    runner = _FakeTmuxRunner([""], capture_returncode=1)
    outcome = mod._verify_prompt_submission(
        "aragora:claude-lane",
        _PROMPT,
        timeout_seconds=0.3,
        runner=runner,
        sleep=lambda _s: None,
    )

    assert outcome["delivered"] is None
    assert outcome["error"] == "capture-failed"
    assert outcome["enter_nudges"] == 0
    # No Enter sent when we cannot even read the pane.
    assert runner.enter_calls() == []


def test_verify_prompt_submission_capture_oserror_is_unverifiable() -> None:
    import agent_bridge as mod

    runner = _FakeTmuxRunner([""], capture_raises=OSError("tmux not found"))
    outcome = mod._verify_prompt_submission(
        "aragora:claude-lane",
        _PROMPT,
        timeout_seconds=0.3,
        runner=runner,
        sleep=lambda _s: None,
    )

    assert outcome["delivered"] is None
    assert outcome["error"] == "capture-failed"
    assert outcome["enter_nudges"] == 0
    assert runner.enter_calls() == []


def test_verify_prompt_submission_failed_send_keys_not_counted_as_nudge() -> None:
    import agent_bridge as mod

    # Placeholder staged -> nudge attempted, but tmux rejects send-keys
    # (returncode != 0).  A rejected nudge means we never confirmed submission,
    # so the receipt must NOT falsely attest an Enter was sent and the outcome
    # is UNVERIFIABLE (delivered=None), never a confident False.
    runner = _FakeTmuxRunner([_PLACEHOLDER_PANE], send_returncode=1)
    outcome = mod._verify_prompt_submission(
        "aragora:claude-lane",
        _PROMPT,
        timeout_seconds=0.3,
        runner=runner,
        sleep=lambda _s: None,
    )

    assert outcome["delivered"] is None
    assert outcome["error"] == "nudge-failed"
    assert outcome["enter_nudges"] == 0
    assert outcome["nudge_failed"] is True
    # send-keys was attempted exactly once, but it failed; no re-verify poll.
    assert len(runner.enter_calls()) == 1
    # Placeholder breaks the initial loop on the first poll; the rejected nudge
    # adds no re-verify capture.
    assert outcome["attempts"] == 1


def test_pane_shows_staged_prompt_detects_paste_placeholder() -> None:
    import agent_bridge as mod

    marker = mod._prompt_tail_marker(_PROMPT)
    assert marker == "Report results when done."
    # (staged, placeholder) tuple: placeholder is the high-confidence positive.
    assert mod._pane_shows_staged_prompt("composer:\n[Pasted Content 5120 chars]\n", marker) == (
        True,
        True,
    )
    # Bare marker-in-tail: staged but low-confidence (no placeholder).
    assert mod._pane_shows_staged_prompt(_STAGED_PANE, marker) == (True, False)
    assert mod._pane_shows_staged_prompt(_SUBMITTED_PANE, marker) == (False, False)


def test_pane_shows_staged_prompt_scans_wider_window() -> None:
    import agent_bridge as mod

    marker = mod._prompt_tail_marker(_PROMPT)
    # Staged paste, then a handful of harness chrome lines that would scroll the
    # marker off a 5-line tail but stay inside the wider (~25-line) window.
    chrome = "\n".join(f"harness chrome line {i}" for i in range(8))
    pane = f"{_STAGED_PANE}\n{chrome}\n"
    assert mod._pane_shows_staged_prompt(pane, marker) == (True, False)


def test_record_dispatch_receipt_merges_into_existing_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent_bridge as mod

    sessions_dir = tmp_path / "tmux-sessions"
    sessions_dir.mkdir()
    meta_path = sessions_dir / "claude-lane.meta.json"
    meta_path.write_text(
        json.dumps({"name": "claude-lane", "agent": "claude", "tmux_window_target": "@7"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "TMUX_SESSIONS_DIR", sessions_dir)

    dispatch = {"delivered": False, "attempts": 4, "enter_nudges": 1, "pane_target": "@7"}
    written = mod._record_dispatch_receipt("claude-lane", dispatch)

    assert written == meta_path
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    # Existing launcher keys preserved; dispatch sub-object added additively.
    assert payload["agent"] == "claude"
    assert payload["tmux_window_target"] == "@7"
    assert payload["dispatch"] == dispatch


def test_launched_pane_target_distinguishes_absent_from_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent_bridge as mod

    sessions_dir = tmp_path / "tmux-sessions"
    sessions_dir.mkdir()
    monkeypatch.setattr(mod, "TMUX_SESSIONS_DIR", sessions_dir)

    # Absent meta -> documented fallback target, no failure reason.
    target, reason = mod._launched_pane_target("absent-lane")
    assert target == f"{mod.TMUX_SESSION}:absent-lane"
    assert reason is None

    # Readable meta with a target -> that target.
    (sessions_dir / "good-lane.meta.json").write_text(
        json.dumps({"tmux_window_target": "@9"}), encoding="utf-8"
    )
    target, reason = mod._launched_pane_target("good-lane")
    assert target == "@9"
    assert reason is None

    # Present but corrupt meta -> fail CLOSED to unverifiable (no fallback pane).
    (sessions_dir / "bad-lane.meta.json").write_text("{not json", encoding="utf-8")
    target, reason = mod._launched_pane_target("bad-lane")
    assert target is None
    assert reason == "meta-unreadable"

    # Present meta missing the target key -> also unverifiable.
    (sessions_dir / "blank-lane.meta.json").write_text(
        json.dumps({"name": "blank-lane"}), encoding="utf-8"
    )
    target, reason = mod._launched_pane_target("blank-lane")
    assert target is None
    assert reason == "meta-unreadable"


def _launch_namespace(tmp_path: Path, **overrides) -> argparse.Namespace:
    prompt_file = tmp_path / "prompt.md"
    if not prompt_file.exists():
        prompt_file.write_text(_PROMPT, encoding="utf-8")
    defaults = dict(
        name="claude-lane",
        agent="claude",
        prompt=[],
        file=str(prompt_file),
        cwd=str(tmp_path),
        autonomous=False,
        timeout_seconds=10,
        submit_verify_timeout=0.05,
        strict_verify=False,
        json=True,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _setup_launch_repo(mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo_root = tmp_path / "repo"
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "scripts" / "tmux_session_launcher.sh").write_text(
        "#!/usr/bin/env bash\n", encoding="utf-8"
    )
    sessions_dir = tmp_path / "tmux-sessions"
    sessions_dir.mkdir()
    monkeypatch.setattr(mod, "CANONICAL_REPO_ROOT", repo_root)
    monkeypatch.setattr(mod, "TMUX_SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    return sessions_dir


def _fake_launch_subprocess(mod, monkeypatch: pytest.MonkeyPatch, pane: str) -> list[list[str]]:
    calls: list[list[str]] = []

    def _fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        if cmd[0] == "bash":
            return SimpleNamespace(returncode=0, stdout="launched\n", stderr="")
        if cmd[:2] == ["tmux", "capture-pane"]:
            return SimpleNamespace(returncode=0, stdout=pane, stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    return calls


def test_cmd_launch_writes_dispatch_receipt_and_annotates_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    sessions_dir = _setup_launch_repo(mod, tmp_path, monkeypatch)
    meta_path = sessions_dir / "claude-lane.meta.json"
    meta_path.write_text(
        json.dumps({"name": "claude-lane", "agent": "claude", "tmux_window_target": "@7"}),
        encoding="utf-8",
    )
    calls = _fake_launch_subprocess(mod, monkeypatch, _SUBMITTED_PANE)

    rc = mod.cmd_launch(_launch_namespace(tmp_path))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["dispatch"]["delivered"] is True
    assert payload["dispatch"]["enter_nudges"] == 0
    assert payload["dispatch"]["pane_target"] == "@7"
    # Receipt persisted into the launcher meta entry, existing keys intact.
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["agent"] == "claude"
    assert meta["dispatch"]["delivered"] is True
    # Pane was verified against the meta-resolved window target.
    assert ["tmux", "capture-pane", "-t", "@7", "-p", "-S", "-40"] in calls


def test_cmd_launch_undelivered_is_observational_rc0_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    sessions_dir = _setup_launch_repo(mod, tmp_path, monkeypatch)
    # Placeholder pane -> high-confidence staged, persists -> delivered=False.
    calls = _fake_launch_subprocess(mod, monkeypatch, _PLACEHOLDER_PANE)

    rc = mod.cmd_launch(_launch_namespace(tmp_path))

    # Default: verification is OBSERVATIONAL. A successful launch keeps rc 0
    # regardless of the dispatch outcome -- exit-code enforcement is opt-in.
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["returncode"] == 0
    # The tri-state receipt is ALWAYS written (lane_liveness reads it) and a
    # single Enter is still nudged on the high-confidence placeholder.
    assert payload["dispatch"]["delivered"] is False
    enters = [c for c in calls if c[:2] == ["tmux", "send-keys"]]
    assert enters == [["tmux", "send-keys", "-t", "aragora:claude-lane", "Enter"]]
    meta = json.loads((sessions_dir / "claude-lane.meta.json").read_text(encoding="utf-8"))
    assert meta["dispatch"]["delivered"] is False
    assert meta["dispatch"]["enter_nudges"] == 1
    assert meta["dispatch"]["verified_at"]


def test_cmd_launch_strict_verify_returns_nonzero_on_undelivered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    sessions_dir = _setup_launch_repo(mod, tmp_path, monkeypatch)
    calls = _fake_launch_subprocess(mod, monkeypatch, _PLACEHOLDER_PANE)

    rc = mod.cmd_launch(_launch_namespace(tmp_path, strict_verify=True))

    # --strict-verify opts in to exit-code enforcement: delivered=False -> rc 1.
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["returncode"] == 0  # launcher itself succeeded
    assert payload["dispatch"]["delivered"] is False
    enters = [c for c in calls if c[:2] == ["tmux", "send-keys"]]
    assert enters == [["tmux", "send-keys", "-t", "aragora:claude-lane", "Enter"]]
    meta = json.loads((sessions_dir / "claude-lane.meta.json").read_text(encoding="utf-8"))
    assert meta["dispatch"]["delivered"] is False


def test_cmd_launch_unverifiable_capture_is_observational_rc0_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    sessions_dir = _setup_launch_repo(mod, tmp_path, monkeypatch)
    calls: list[list[str]] = []

    def _fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        if cmd[0] == "bash":
            return SimpleNamespace(returncode=0, stdout="launched\n", stderr="")
        if cmd[:2] == ["tmux", "capture-pane"]:
            # Capture fails: submission is unverifiable, not delivered.
            return SimpleNamespace(returncode=1, stdout="", stderr="no such pane")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    rc = mod.cmd_launch(_launch_namespace(tmp_path))

    # Default observational: unverifiable does not flip the exit code, but the
    # tri-state receipt records delivered=None for lane_liveness to read.
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["returncode"] == 0
    assert payload["dispatch"]["delivered"] is None
    assert payload["dispatch"]["error"] == "capture-failed"
    # No Enter sent when the pane could not be read.
    assert [c for c in calls if c[:2] == ["tmux", "send-keys"]] == []
    meta = json.loads((sessions_dir / "claude-lane.meta.json").read_text(encoding="utf-8"))
    assert meta["dispatch"]["delivered"] is None
    assert meta["dispatch"]["error"] == "capture-failed"


def test_cmd_launch_strict_verify_returns_nonzero_on_unverifiable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    _setup_launch_repo(mod, tmp_path, monkeypatch)

    def _fake_run(cmd, **_kwargs):
        if cmd[0] == "bash":
            return SimpleNamespace(returncode=0, stdout="launched\n", stderr="")
        if cmd[:2] == ["tmux", "capture-pane"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="no such pane")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    rc = mod.cmd_launch(_launch_namespace(tmp_path, strict_verify=True))

    # Under --strict-verify, an unverifiable (None) dispatch also fails closed.
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["dispatch"]["delivered"] is None
    assert payload["dispatch"]["error"] == "capture-failed"


def test_cmd_launch_marker_only_is_unverifiable_no_enter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    sessions_dir = _setup_launch_repo(mod, tmp_path, monkeypatch)
    # Bare marker echo, no placeholder -> unverifiable, never a confident False.
    calls = _fake_launch_subprocess(mod, monkeypatch, _STAGED_PANE)

    rc = mod.cmd_launch(_launch_namespace(tmp_path))

    assert rc == 0  # observational default
    payload = json.loads(capsys.readouterr().out)
    assert payload["dispatch"]["delivered"] is None
    assert payload["dispatch"]["error"] == "marker-only"
    # No Enter is ever sent on a bare marker match.
    assert [c for c in calls if c[:2] == ["tmux", "send-keys"]] == []
    meta = json.loads((sessions_dir / "claude-lane.meta.json").read_text(encoding="utf-8"))
    assert meta["dispatch"]["delivered"] is None
    assert meta["dispatch"]["error"] == "marker-only"


def test_cmd_launch_meta_unreadable_is_unverifiable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    sessions_dir = _setup_launch_repo(mod, tmp_path, monkeypatch)
    # Meta file EXISTS but is corrupt -> cannot trust a pane target. Fail closed
    # to unverifiable rather than capturing a possibly-unrelated fallback pane.
    meta_path = sessions_dir / "claude-lane.meta.json"
    meta_path.write_text("{not valid json", encoding="utf-8")
    calls = _fake_launch_subprocess(mod, monkeypatch, _SUBMITTED_PANE)

    rc = mod.cmd_launch(_launch_namespace(tmp_path))

    assert rc == 0  # observational default
    payload = json.loads(capsys.readouterr().out)
    assert payload["dispatch"]["delivered"] is None
    assert payload["dispatch"]["error"] == "meta-unreadable"
    assert payload["dispatch"]["pane_target"] is None
    # Never captured a fallback pane when the meta target was untrustworthy.
    assert [c for c in calls if c[:2] == ["tmux", "capture-pane"]] == []
    meta = json.loads((sessions_dir / "claude-lane.meta.json").read_text(encoding="utf-8"))
    assert meta["dispatch"]["delivered"] is None
    assert meta["dispatch"]["error"] == "meta-unreadable"


def test_cmd_launch_submit_verify_timeout_zero_disables_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    sessions_dir = _setup_launch_repo(mod, tmp_path, monkeypatch)
    calls = _fake_launch_subprocess(mod, monkeypatch, _STAGED_PANE)

    rc = mod.cmd_launch(_launch_namespace(tmp_path, submit_verify_timeout=0))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "dispatch" not in payload
    assert all(c[0] == "bash" for c in calls)
    assert not (sessions_dir / "claude-lane.meta.json").exists()


def test_cmd_launch_skips_submit_verification_for_droid_exec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    _setup_launch_repo(mod, tmp_path, monkeypatch)
    calls = _fake_launch_subprocess(mod, monkeypatch, _STAGED_PANE)

    rc = mod.cmd_launch(_launch_namespace(tmp_path, name="droid-lane", agent="droid"))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "dispatch" not in payload
    # droid prompted lanes deliver via `droid exec -f`; no pane verification.
    assert all(c[0] == "bash" for c in calls)


def test_cmd_launch_broken_pipe_on_report_preserves_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_bridge as mod

    _setup_launch_repo(mod, tmp_path, monkeypatch)
    _fake_launch_subprocess(mod, monkeypatch, _SUBMITTED_PANE)
    muted_stdout: list[bool] = []

    def broken_print(*_args, **_kwargs) -> None:
        raise BrokenPipeError("downstream closed")

    monkeypatch.setattr("builtins.print", broken_print)
    monkeypatch.setattr(mod, "_mute_stdout_after_broken_pipe", lambda: muted_stdout.append(True))

    rc = mod.cmd_launch(_launch_namespace(tmp_path))

    assert rc == 0
    assert muted_stdout == [True]


def test_cmd_launch_broken_pipe_does_not_mask_undelivered_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_bridge as mod

    _setup_launch_repo(mod, tmp_path, monkeypatch)
    # Placeholder pane persists -> delivered=False; under --strict-verify this
    # is an enforced rc 1 that must survive a broken pipe on the report write.
    _fake_launch_subprocess(mod, monkeypatch, _PLACEHOLDER_PANE)

    def broken_print(*_args, **_kwargs) -> None:
        raise BrokenPipeError("downstream closed")

    monkeypatch.setattr("builtins.print", broken_print)
    monkeypatch.setattr(mod, "_mute_stdout_after_broken_pipe", lambda: None)

    rc = mod.cmd_launch(_launch_namespace(tmp_path, strict_verify=True))

    assert rc == 1  # dispatch truth survives the broken pipe


def test_install_sigpipe_hygiene_ignores_sigpipe_on_posix() -> None:
    import signal as signal_module

    import agent_bridge as mod

    if not hasattr(signal_module, "SIGPIPE"):
        pytest.skip("SIGPIPE not available on this platform")
    previous = signal_module.getsignal(signal_module.SIGPIPE)
    try:
        mod._install_sigpipe_hygiene()
        assert signal_module.getsignal(signal_module.SIGPIPE) is signal_module.SIG_IGN
    finally:
        signal_module.signal(signal_module.SIGPIPE, previous)


def test_main_does_not_install_sigpipe_for_non_launch_subcommands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_bridge as mod

    # SIGPIPE hygiene is scoped to launch only; other subcommands must keep
    # their prior clean SIGPIPE-killed behavior under `| head`.
    installs: list[bool] = []
    monkeypatch.setattr(mod, "_install_sigpipe_hygiene", lambda: installs.append(True))
    monkeypatch.setattr(mod, "cmd_sessions", lambda _args: 0)
    monkeypatch.setattr(sys, "argv", ["agent_bridge.py", "sessions"])

    rc = mod.main()

    assert rc == 0
    assert installs == []


def test_cmd_launch_installs_sigpipe_hygiene(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    _setup_launch_repo(mod, tmp_path, monkeypatch)
    _fake_launch_subprocess(mod, monkeypatch, _SUBMITTED_PANE)
    installs: list[bool] = []
    monkeypatch.setattr(mod, "_install_sigpipe_hygiene", lambda: installs.append(True))

    rc = mod.cmd_launch(_launch_namespace(tmp_path))

    capsys.readouterr()
    assert rc == 0
    # SIGPIPE hygiene is installed exactly once, scoped to the launch path.
    assert installs == [True]
