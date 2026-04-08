"""Session registry for multi-agent coordination.

Each agent session writes a JSON registration file on startup.  Other agents
discover peers by reading the registry directory.  Session health is determined
by both PID liveness and heartbeat freshness.  Sessions with stale heartbeats
are treated separately from dead PIDs so coordination surfaces can report the
difference truthfully while still reaping abandoned state.

Usage::

    from aragora.coordination.registry import SessionRegistry

    reg = SessionRegistry(repo_path=Path("."))
    session = reg.register(agent="claude", worktree=Path("/tmp/wt1"), focus="SDK parity")
    peers = reg.discover()
    reg.deregister(session.session_id)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)

_COORD_DIR = ".aragora_coordination"
_SESSIONS_DIR = "sessions"


@dataclass
class SessionInfo:
    """Registration data for a single agent session."""

    session_id: str
    agent: str
    worktree: str
    pid: int
    started_at: float
    last_heartbeat: float
    focus: str = ""
    track: str = ""
    intent: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SessionInfo:
        pid = data.get("pid", 0)
        started_at = data.get("started_at", 0)
        last_heartbeat = data.get("last_heartbeat", 0)
        return cls(
            session_id=str(data.get("session_id", "")),
            agent=str(data.get("agent", "")),
            worktree=str(data.get("worktree", "")),
            pid=int(pid) if isinstance(pid, (int, float, str)) else 0,
            started_at=float(started_at) if isinstance(started_at, (int, float, str)) else 0.0,
            last_heartbeat=float(last_heartbeat)
            if isinstance(last_heartbeat, (int, float, str))
            else 0.0,
            focus=str(data.get("focus", "")),
            track=str(data.get("track", "")),
            intent=str(data.get("intent", "")),
        )

    @property
    def pid_alive(self) -> bool:
        """Check if the session's PID is still running."""
        if self.pid <= 0:
            return False
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but we can't signal it — still alive
            return True
        return True

    @property
    def is_alive(self) -> bool:
        """Backward-compatible alias for process liveness."""
        return self.pid_alive


class SessionRegistry:
    """File-backed registry of active agent sessions.

    Sessions are stored as ``{agent}-{short_id}.json`` files in
    ``.aragora_coordination/sessions/``.  Discovery is a directory read.
    Stale sessions (dead PID or missed heartbeat) are auto-reaped on discover().
    """

    def __init__(
        self,
        repo_path: Path | None = None,
        *,
        stale_timeout_seconds: int = 300,
    ):
        self.repo_path = (repo_path or Path.cwd()).resolve()
        self._sessions_dir = self.repo_path / _COORD_DIR / _SESSIONS_DIR
        self._stale_timeout = stale_timeout_seconds

    def _ensure_dir(self) -> Path:
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        return self._sessions_dir

    def _session_path(self, session_id: str) -> Path:
        return self._sessions_dir / f"{session_id}.json"

    def _load_session(self, path: Path) -> SessionInfo | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return SessionInfo.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def _heartbeat_age_seconds(self, session: SessionInfo, *, now: float | None = None) -> float:
        current = time.time() if now is None else now
        return max(0.0, current - session.last_heartbeat)

    def _heartbeat_stale(self, session: SessionInfo, *, now: float | None = None) -> bool:
        if self._stale_timeout <= 0:
            return False
        return self._heartbeat_age_seconds(session, now=now) >= float(self._stale_timeout)

    def session_state(self, session: SessionInfo, *, now: float | None = None) -> str:
        """Return ``active``, ``stale``, or ``dead`` for the given session."""
        if not session.pid_alive:
            return "dead"
        if self._heartbeat_stale(session, now=now):
            return "stale"
        return "active"

    def describe(self, session: SessionInfo, *, now: float | None = None) -> dict[str, object]:
        """Return a truthfully annotated session payload for reporting."""
        current = time.time() if now is None else now
        status = self.session_state(session, now=current)
        data = session.to_dict()
        data.update(
            {
                "pid_alive": session.pid_alive,
                "heartbeat_stale": status == "stale",
                "heartbeat_age_seconds": round(
                    self._heartbeat_age_seconds(session, now=current), 2
                ),
                "status": status,
            }
        )
        return data

    def reap_abandoned(self) -> list[dict[str, object]]:
        """Remove and describe sessions that are stale or dead."""
        if not self._sessions_dir.exists():
            return []

        current = time.time()
        reaped: list[dict[str, object]] = []
        for path in sorted(self._sessions_dir.glob("*.json")):
            session = self._load_session(path)
            if session is None:
                logger.debug("skipping_corrupt_session path=%s", path)
                continue
            status = self.session_state(session, now=current)
            if status == "active":
                continue
            path.unlink(missing_ok=True)
            described = self.describe(session, now=current)
            reaped.append(described)
            logger.info(
                "session_reaped id=%s pid=%d status=%s",
                session.session_id,
                session.pid,
                status,
            )
        return reaped

    def register(
        self,
        agent: str,
        worktree: Path | str,
        *,
        focus: str = "",
        track: str = "",
        intent: str = "",
        pid: int | None = None,
    ) -> SessionInfo:
        """Register a new agent session. Returns the SessionInfo."""
        short_id = str(uuid4())[:8]
        now = time.time()
        session = SessionInfo(
            session_id=f"{agent}-{short_id}",
            agent=agent,
            worktree=str(worktree),
            pid=pid if pid is not None else os.getpid(),
            started_at=now,
            last_heartbeat=now,
            focus=focus,
            track=track,
            intent=intent,
        )

        d = self._ensure_dir()
        path = d / f"{session.session_id}.json"
        path.write_text(json.dumps(session.to_dict(), default=str), encoding="utf-8")

        logger.info(
            "session_registered id=%s agent=%s focus=%s",
            session.session_id,
            agent,
            focus,
        )
        return session

    def deregister(self, session_id: str) -> bool:
        """Remove a session registration. Returns True if file was removed."""
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            logger.info("session_deregistered id=%s", session_id)
            return True
        return False

    def heartbeat(self, session_id: str) -> bool:
        """Update the last_heartbeat timestamp. Returns True if session found."""
        path = self._session_path(session_id)
        if not path.exists():
            return False

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["last_heartbeat"] = time.time()
            path.write_text(json.dumps(data, default=str), encoding="utf-8")
        except (json.JSONDecodeError, OSError):
            return False
        return True

    def discover(self, *, reap_stale: bool = True) -> list[SessionInfo]:
        """Discover all active sessions.

        Args:
            reap_stale: If True, remove registrations for dead PIDs.

        Returns:
            List of active sessions with fresh heartbeats.
        """
        if not self._sessions_dir.exists():
            return []

        sessions: list[SessionInfo] = []
        current = time.time()
        for path in sorted(self._sessions_dir.glob("*.json")):
            session = self._load_session(path)
            if session is None:
                logger.debug("skipping_corrupt_session path=%s", path)
                continue

            status = self.session_state(session, now=current)
            if status != "active":
                if reap_stale:
                    path.unlink(missing_ok=True)
                    logger.info(
                        "session_reaped id=%s pid=%d status=%s",
                        session.session_id,
                        session.pid,
                        status,
                    )
                continue

            sessions.append(session)

        return sessions

    def get(self, session_id: str, *, include_dead: bool = False) -> SessionInfo | None:
        """Get a specific session by ID, or None if not found/inactive."""
        path = self._session_path(session_id)
        if not path.exists():
            return None
        session = self._load_session(path)
        if session is None:
            return None

        if not include_dead and self.session_state(session) != "active":
            return None
        return session


__all__ = [
    "SessionRegistry",
    "SessionInfo",
]
