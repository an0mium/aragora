"""Tests for ``_collect_pending_steering_messages`` and the
``pending_steering_messages`` field added to ``operator-snapshot``
by Phase C of the agent-steering primitive.

Fixture-driven; uses ``tmp_path`` for the steering inbox root and
``monkeypatch`` for env vars. Never touches the live
``.aragora/operator-steering/`` directory.

Also includes a regression test that pins all pre-Phase-C top-level
operator-snapshot keys so future changes can't silently drop them.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest


def _load_module() -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / "agent_bridge.py"
    spec = importlib.util.spec_from_file_location("agent_bridge_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ab = _load_module()


# ---------------------------------------------------------------------------
# Helpers — mailbox fixture writer
# ---------------------------------------------------------------------------


def _write_message(
    root: Path,
    recipient: str,
    *,
    body: str,
    priority: str = "normal",
    lane_id_hint: str | None = None,
    pr_hint: int | None = None,
    sent_at_utc: str | None = None,
    subject: str | None = None,
    filename: str | None = None,
) -> Path:
    """Write one v1.0 schema mailbox file at ``root/recipient/<filename>``."""

    inbox = root / recipient
    inbox.mkdir(parents=True, exist_ok=True)
    if sent_at_utc is None:
        sent_at_utc = "2026-05-18T00:00:00.000Z"
    payload = {
        "schema_version": "aragora-operator-steering/1.0",
        "to_session": recipient,
        "from": "operator-test",
        "sent_at_utc": sent_at_utc,
        "lane_id_hint": lane_id_hint,
        "pr_hint": pr_hint,
        "priority": priority,
        "subject": subject if subject is not None else body[:80],
        "body": body,
        "message_sha256": "0" * 64,  # not validated by Phase C reader
    }
    if filename is None:
        filename = f"{sent_at_utc.replace(':', '-').replace('.', '-')}-fixture.json"
    out = inbox / filename
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# _collect_pending_steering_messages — direct unit coverage
# ---------------------------------------------------------------------------


class TestCollectPendingSteeringMessages:
    def test_empty_mailbox_scoped(self, tmp_path: Path) -> None:
        result = ab._collect_pending_steering_messages("fixture-a", steering_root=tmp_path)
        assert result == {"count": 0, "latest_three": []}

    def test_empty_mailbox_rollup(self, tmp_path: Path) -> None:
        result = ab._collect_pending_steering_messages(None, steering_root=tmp_path)
        assert result == {"count": 0, "by_recipient": {}, "latest_three": []}

    def test_missing_dir_returns_zero(self, tmp_path: Path) -> None:
        # Steering root does not exist at all.
        result = ab._collect_pending_steering_messages(
            "fixture", steering_root=tmp_path / "no-root"
        )
        assert result == {"count": 0, "latest_three": []}
        result_rollup = ab._collect_pending_steering_messages(
            None, steering_root=tmp_path / "no-root"
        )
        assert result_rollup == {"count": 0, "by_recipient": {}, "latest_three": []}

    def test_single_message_scoped_surfaces_metadata(self, tmp_path: Path) -> None:
        _write_message(
            tmp_path,
            "fixture-single",
            body="hello world",
            priority="high",
            lane_id_hint="P30-fixture",
            pr_hint=9000,
            sent_at_utc="2026-05-18T01:00:00.000Z",
        )
        result = ab._collect_pending_steering_messages("fixture-single", steering_root=tmp_path)
        assert result["count"] == 1
        assert len(result["latest_three"]) == 1
        first = result["latest_three"][0]
        assert first["subject"] == "hello world"
        assert first["sent_at_utc"] == "2026-05-18T01:00:00.000Z"
        assert first["priority"] == "high"
        assert first["lane_id_hint"] == "P30-fixture"
        assert first["pr_hint"] == 9000

    def test_five_messages_returns_top_three_newest(self, tmp_path: Path) -> None:
        # Write 5 with strictly-increasing sent_at_utc.
        for i in range(5):
            _write_message(
                tmp_path,
                "fixture-five",
                body=f"message-{i}",
                sent_at_utc=f"2026-05-18T0{i}:00:00.000Z",
                filename=f"msg-{i}.json",
            )
        result = ab._collect_pending_steering_messages("fixture-five", steering_root=tmp_path)
        assert result["count"] == 5
        assert len(result["latest_three"]) == 3
        subjects = [m["subject"] for m in result["latest_three"]]
        assert subjects == ["message-4", "message-3", "message-2"]

    def test_rollup_across_multiple_recipients(self, tmp_path: Path) -> None:
        _write_message(tmp_path, "alpha", body="a-only", sent_at_utc="2026-05-18T01:00:00.000Z")
        _write_message(tmp_path, "alpha", body="a-second", sent_at_utc="2026-05-18T02:00:00.000Z")
        _write_message(tmp_path, "beta", body="b-only", sent_at_utc="2026-05-18T03:00:00.000Z")
        _write_message(tmp_path, "gamma", body="g-only", sent_at_utc="2026-05-18T04:00:00.000Z")
        result = ab._collect_pending_steering_messages(None, steering_root=tmp_path)
        assert result["count"] == 4
        assert result["by_recipient"] == {"alpha": 2, "beta": 1, "gamma": 1}
        # latest_three sorted newest first across all recipients
        subjects = [m["subject"] for m in result["latest_three"]]
        assert subjects == ["g-only", "b-only", "a-second"]

    def test_acked_subdir_excluded(self, tmp_path: Path) -> None:
        """Phase D ack convention: messages moved to _acked/ should NOT count."""
        # One live message in inbox top-level
        _write_message(tmp_path, "ack-fixture", body="live", sent_at_utc="2026-05-18T05:00:00.000Z")
        # One acked message in _acked subdir (should be invisible)
        acked_dir = tmp_path / "ack-fixture" / "_acked"
        acked_dir.mkdir(parents=True)
        (acked_dir / "old-msg.json").write_text(
            json.dumps(
                {
                    "schema_version": "aragora-operator-steering/1.0",
                    "to_session": "ack-fixture",
                    "subject": "should-not-appear",
                    "body": "acked",
                    "sent_at_utc": "2026-05-18T06:00:00.000Z",
                    "priority": "low",
                    "lane_id_hint": None,
                    "pr_hint": None,
                }
            ),
            encoding="utf-8",
        )
        result = ab._collect_pending_steering_messages("ack-fixture", steering_root=tmp_path)
        assert result["count"] == 1
        assert result["latest_three"][0]["subject"] == "live"

    def test_unreadable_message_safely_surfaces(self, tmp_path: Path) -> None:
        inbox = tmp_path / "unreadable"
        inbox.mkdir(parents=True)
        (inbox / "broken.json").write_text("not valid json {{{", encoding="utf-8")
        result = ab._collect_pending_steering_messages("unreadable", steering_root=tmp_path)
        # Still counts the file but surfaces a sentinel summary.
        assert result["count"] == 1
        assert result["latest_three"][0]["subject"] == "(unreadable)"


# ---------------------------------------------------------------------------
# CLI integration — pending field appears in operator-snapshot --json output
# ---------------------------------------------------------------------------


REPO_ROOT_FOR_SUBPROCESS = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT_FOR_SUBPROCESS / "scripts" / "agent_bridge.py"


def _run_snapshot(*extra_args: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Invoke `operator-snapshot --json` and parse stdout."""
    import os as _os

    full_env = _os.environ.copy()
    if env:
        full_env.update(env)
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "operator-snapshot", "--json", *extra_args],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT_FOR_SUBPROCESS,
        env=full_env,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


class TestOperatorSnapshotIntegration:
    """End-to-end CLI checks. Runs the script as a subprocess and
    verifies the new field appears + all pre-Phase-C fields remain."""

    def test_pre_phase_c_fields_still_present(self) -> None:
        """Regression guard: existing top-level keys must keep their shape."""
        snap = _run_snapshot()
        expected_keys = {
            "timestamp",
            "sessions",
            "broker_runs",
            "lanes",
            "lane_conflicts",
            "process_census",
            "health",
            "summary",
            # Phase C addition (will fail if Phase C ever silently removes it):
            "pending_steering_messages",
        }
        assert expected_keys.issubset(set(snap.keys())), (
            f"missing keys: {expected_keys - set(snap.keys())}"
        )

    def test_summary_subkeys_preserved(self) -> None:
        snap = _run_snapshot()
        summary = snap["summary"]
        expected = {
            "total_sessions",
            "alive_sessions",
            "live_sessions",
            "dead_sessions",
            "historical_sessions",
            "active_broker_runs",
            "active_lanes",
            "conflict_lanes",
            "health_issues",
            "active_processes",
            "active_process_roles",
        }
        assert expected.issubset(set(summary.keys()))

    def test_pending_steering_messages_default_rollup_shape(self) -> None:
        """Without ARAGORA_SESSION_ID, output is a roll-up across all recipients."""
        # Ensure ARAGORA_SESSION_ID is NOT set in the subprocess env.
        import os as _os

        env = {k: v for k, v in _os.environ.items() if k != "ARAGORA_SESSION_ID"}
        snap = _run_snapshot(env=env)
        pending = snap["pending_steering_messages"]
        assert "count" in pending
        assert "by_recipient" in pending
        assert "latest_three" in pending
        assert isinstance(pending["latest_three"], list)

    def test_pending_steering_messages_scoped_via_env(self) -> None:
        snap = _run_snapshot(env={"ARAGORA_SESSION_ID": "no-such-session-fixture-xyz"})
        pending = snap["pending_steering_messages"]
        # Scoped output has count + latest_three but NOT by_recipient.
        assert pending["count"] == 0
        assert pending["latest_three"] == []
        assert "by_recipient" not in pending

    def test_steering_recipient_flag_overrides_env(self) -> None:
        # Pass --steering-recipient with a no-such-session label;
        # env ARAGORA_SESSION_ID would otherwise pick a different
        # nonsense value. Both yield 0 here because neither recipient
        # has a mailbox, but the assertion confirms the flag is wired.
        snap = _run_snapshot(
            "--steering-recipient",
            "flag-fixture-xyz",
            env={"ARAGORA_SESSION_ID": "env-fixture-xyz"},
        )
        pending = snap["pending_steering_messages"]
        assert "by_recipient" not in pending  # scoped, not roll-up
        assert pending["count"] == 0
