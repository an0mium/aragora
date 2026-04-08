"""Tests for the session registry."""

from __future__ import annotations

import json
import os
import time

import pytest

from aragora.coordination.registry import SessionInfo, SessionRegistry


class TestSessionInfo:
    def test_roundtrip(self):
        info = SessionInfo(
            session_id="claude-abc1",
            agent="claude",
            worktree="/tmp/wt1",
            pid=os.getpid(),
            started_at=1000.0,
            last_heartbeat=1000.0,
            focus="SDK parity",
            track="developer",
        )
        data = info.to_dict()
        restored = SessionInfo.from_dict(data)
        assert restored.session_id == "claude-abc1"
        assert restored.agent == "claude"
        assert restored.focus == "SDK parity"

    def test_is_alive_current_pid(self):
        info = SessionInfo(
            session_id="test",
            agent="test",
            worktree=".",
            pid=os.getpid(),
            started_at=0,
            last_heartbeat=0,
        )
        assert info.is_alive is True

    def test_is_alive_dead_pid(self):
        info = SessionInfo(
            session_id="test",
            agent="test",
            worktree=".",
            pid=999999999,  # Almost certainly not running
            started_at=0,
            last_heartbeat=0,
        )
        assert info.is_alive is False

    def test_is_alive_zero_pid(self):
        info = SessionInfo(
            session_id="test",
            agent="test",
            worktree=".",
            pid=0,
            started_at=0,
            last_heartbeat=0,
        )
        assert info.is_alive is False

    def test_from_dict_defaults(self):
        info = SessionInfo.from_dict({})
        assert info.session_id == ""
        assert info.pid == 0


class TestSessionRegistry:
    @staticmethod
    def _rewrite_heartbeat(
        tmp_path,
        session_id: str,
        *,
        last_heartbeat: float,
    ) -> None:
        session_path = tmp_path / ".aragora_coordination" / "sessions" / f"{session_id}.json"
        payload = json.loads(session_path.read_text(encoding="utf-8"))
        payload["last_heartbeat"] = last_heartbeat
        session_path.write_text(json.dumps(payload), encoding="utf-8")

    def test_register_creates_file(self, tmp_path):
        reg = SessionRegistry(repo_path=tmp_path)
        session = reg.register(agent="claude", worktree="/tmp/wt1", focus="testing")

        sessions_dir = tmp_path / ".aragora_coordination" / "sessions"
        files = list(sessions_dir.glob("*.json"))
        assert len(files) == 1
        assert session.agent == "claude"
        assert session.focus == "testing"
        assert session.pid == os.getpid()

    def test_deregister_removes_file(self, tmp_path):
        reg = SessionRegistry(repo_path=tmp_path)
        session = reg.register(agent="claude", worktree="/tmp/wt1")
        assert reg.deregister(session.session_id) is True

        sessions_dir = tmp_path / ".aragora_coordination" / "sessions"
        assert list(sessions_dir.glob("*.json")) == []

    def test_deregister_missing_returns_false(self, tmp_path):
        reg = SessionRegistry(repo_path=tmp_path)
        assert reg.deregister("nonexistent-id") is False

    def test_discover_finds_live_sessions(self, tmp_path):
        reg = SessionRegistry(repo_path=tmp_path)
        reg.register(agent="claude", worktree="/tmp/wt1")
        reg.register(agent="codex", worktree="/tmp/wt2")

        sessions = reg.discover()
        assert len(sessions) == 2
        agents = {s.agent for s in sessions}
        assert agents == {"claude", "codex"}

    def test_discover_reaps_dead_sessions(self, tmp_path):
        reg = SessionRegistry(repo_path=tmp_path)
        # Register with a dead PID
        reg.register(agent="dead", worktree="/tmp/wt1", pid=999999999)
        reg.register(agent="alive", worktree="/tmp/wt2")

        sessions = reg.discover(reap_stale=True)
        assert len(sessions) == 1
        assert sessions[0].agent == "alive"

        # Dead session file should be removed
        sessions_dir = tmp_path / ".aragora_coordination" / "sessions"
        files = list(sessions_dir.glob("*.json"))
        assert len(files) == 1

    def test_discover_reaps_stale_sessions_even_if_pid_is_alive(self, tmp_path):
        reg = SessionRegistry(repo_path=tmp_path, stale_timeout_seconds=1)
        session = reg.register(agent="alive", worktree="/tmp/wt1", pid=os.getpid())
        self._rewrite_heartbeat(
            tmp_path,
            session.session_id,
            last_heartbeat=time.time() - 10,
        )

        sessions = reg.discover(reap_stale=True)

        assert sessions == []
        sessions_dir = tmp_path / ".aragora_coordination" / "sessions"
        assert list(sessions_dir.glob("*.json")) == []

    def test_discover_empty(self, tmp_path):
        reg = SessionRegistry(repo_path=tmp_path)
        assert reg.discover() == []

    def test_heartbeat_updates_timestamp(self, tmp_path):
        reg = SessionRegistry(repo_path=tmp_path)
        session = reg.register(agent="claude", worktree="/tmp/wt1")
        original_hb = session.last_heartbeat

        assert reg.heartbeat(session.session_id) is True

        # Re-read and check
        updated = reg.get(session.session_id)
        assert updated is not None
        assert updated.last_heartbeat >= original_hb

    def test_heartbeat_missing_returns_false(self, tmp_path):
        reg = SessionRegistry(repo_path=tmp_path)
        assert reg.heartbeat("nonexistent") is False

    def test_get_returns_session(self, tmp_path):
        reg = SessionRegistry(repo_path=tmp_path)
        session = reg.register(agent="claude", worktree="/tmp/wt1")
        found = reg.get(session.session_id)
        assert found is not None
        assert found.agent == "claude"

    def test_get_returns_none_for_dead_pid(self, tmp_path):
        reg = SessionRegistry(repo_path=tmp_path)
        reg.register(agent="dead", worktree="/tmp/wt1", pid=999999999)

        sessions_dir = tmp_path / ".aragora_coordination" / "sessions"
        files = list(sessions_dir.glob("*.json"))
        session_id = json.loads(files[0].read_text())["session_id"]

        assert reg.get(session_id) is None

    def test_get_include_dead_returns_dead_session(self, tmp_path):
        reg = SessionRegistry(repo_path=tmp_path)
        session = reg.register(agent="dead", worktree="/tmp/wt1", pid=999999999)

        found = reg.get(session.session_id, include_dead=True)

        assert found is not None
        assert found.session_id == session.session_id
        assert found.is_alive is False

    def test_get_returns_none_for_stale_live_session(self, tmp_path):
        reg = SessionRegistry(repo_path=tmp_path, stale_timeout_seconds=1)
        session = reg.register(agent="alive", worktree="/tmp/wt1", pid=os.getpid())
        self._rewrite_heartbeat(
            tmp_path,
            session.session_id,
            last_heartbeat=time.time() - 10,
        )

        assert reg.get(session.session_id) is None

        found = reg.get(session.session_id, include_dead=True)
        assert found is not None
        assert found.session_id == session.session_id
        assert found.pid_alive is True
        assert reg.session_state(found) == "stale"

    def test_describe_distinguishes_stale_heartbeat_from_dead_pid(self, tmp_path):
        reg = SessionRegistry(repo_path=tmp_path, stale_timeout_seconds=1)
        stale = reg.register(agent="alive", worktree="/tmp/wt1", pid=os.getpid())
        dead = reg.register(agent="dead", worktree="/tmp/wt2", pid=999999999)
        self._rewrite_heartbeat(
            tmp_path,
            stale.session_id,
            last_heartbeat=time.time() - 10,
        )

        stale_info = reg.get(stale.session_id, include_dead=True)
        dead_info = reg.get(dead.session_id, include_dead=True)

        assert stale_info is not None
        assert dead_info is not None

        stale_desc = reg.describe(stale_info)
        dead_desc = reg.describe(dead_info)

        assert stale_desc["status"] == "stale"
        assert stale_desc["pid_alive"] is True
        assert stale_desc["heartbeat_stale"] is True
        assert dead_desc["status"] == "dead"
        assert dead_desc["pid_alive"] is False
        assert dead_desc["heartbeat_stale"] is False

    def test_get_missing_returns_none(self, tmp_path):
        reg = SessionRegistry(repo_path=tmp_path)
        assert reg.get("nonexistent") is None

    def test_corrupt_file_skipped(self, tmp_path):
        reg = SessionRegistry(repo_path=tmp_path)
        sessions_dir = tmp_path / ".aragora_coordination" / "sessions"
        sessions_dir.mkdir(parents=True)
        (sessions_dir / "corrupt.json").write_text("{bad json")

        reg.register(agent="good", worktree="/tmp/wt1")
        sessions = reg.discover()
        assert len(sessions) == 1
