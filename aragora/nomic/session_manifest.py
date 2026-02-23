"""Session Manifest for multi-agent coordination.

Tracks active Claude Code sessions, their goals, and track assignments.
Each session reads/writes a shared YAML manifest so the coordinator
can detect overlapping goals and route new work.

The manifest file (.aragora_sessions.yaml) is gitignored and local
to each developer's machine.

Usage:
    from aragora.nomic.session_manifest import SessionManifest, SessionEntry

    manifest = SessionManifest()
    manifest.register("core", goal="Implement semantic conflict resolution")
    active = manifest.list_active()
    manifest.deregister("core")
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SessionEntry:
    """A single active agent session."""

    track: str
    worktree: str = ""
    agent: str = "claude"
    current_goal: str = ""
    started_at: str = ""
    files_claimed: list[str] = field(default_factory=list)
    status: str = "active"  # active | paused | completed | error
    pid: int = 0


@dataclass
class SessionManifest:
    """Manages the shared session manifest file."""

    def __init__(
        self,
        manifest_path: Path | None = None,
        repo_root: Path | None = None,
    ):
        self.repo_root = repo_root or Path.cwd()
        self.manifest_path = manifest_path or (self.repo_root / ".aragora_sessions.yaml")

    def _load(self) -> dict[str, Any]:
        """Load manifest from YAML file."""
        if not self.manifest_path.exists():
            return {"sessions": []}

        try:
            import yaml
        except ImportError:
            # Fallback to JSON-like parsing if PyYAML not available
            return self._load_fallback()

        with open(self.manifest_path) as f:
            data = yaml.safe_load(f) or {}

        if "sessions" not in data:
            data["sessions"] = []
        return data

    def _load_fallback(self) -> dict[str, Any]:
        """Fallback loader when PyYAML is not available — uses json."""
        import json

        json_path = self.manifest_path.with_suffix(".json")
        if json_path.exists():
            with open(json_path) as f:
                return json.load(f)
        return {"sessions": []}

    def _save(self, data: dict[str, Any]) -> None:
        """Save manifest to YAML file."""
        try:
            import yaml

            with open(self.manifest_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        except ImportError:
            # Fallback to JSON
            import json

            json_path = self.manifest_path.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)

    def register(
        self,
        track: str,
        *,
        goal: str = "",
        agent: str = "claude",
        worktree: str = "",
        files_claimed: list[str] | None = None,
    ) -> SessionEntry:
        """Register a new agent session.

        Args:
            track: Track name (e.g., "core-track")
            goal: Current goal description
            agent: Agent type (e.g., "claude", "codex")
            worktree: Worktree path
            files_claimed: Files this session intends to modify

        Returns:
            The registered SessionEntry
        """
        data = self._load()

        entry = SessionEntry(
            track=track,
            worktree=worktree or str(self.repo_root / ".worktrees" / track),
            agent=agent,
            current_goal=goal,
            started_at=datetime.now(timezone.utc).isoformat(),
            files_claimed=files_claimed or [],
            status="active",
            pid=os.getpid(),
        )

        # Remove any existing entry for this track
        data["sessions"] = [s for s in data["sessions"] if s.get("track") != track]
        data["sessions"].append(asdict(entry))

        self._save(data)
        logger.info("session_registered track=%s goal=%s", track, goal[:60])
        return entry

    def deregister(self, track: str) -> bool:
        """Remove a session from the manifest.

        Args:
            track: Track name to deregister

        Returns:
            True if session was found and removed
        """
        data = self._load()
        original_count = len(data["sessions"])
        data["sessions"] = [s for s in data["sessions"] if s.get("track") != track]

        if len(data["sessions"]) < original_count:
            self._save(data)
            logger.info("session_deregistered track=%s", track)
            return True
        return False

    def update_goal(self, track: str, goal: str) -> bool:
        """Update the current goal for a session.

        Args:
            track: Track name
            goal: New goal description

        Returns:
            True if session was found and updated
        """
        data = self._load()
        for session in data["sessions"]:
            if session.get("track") == track:
                session["current_goal"] = goal
                self._save(data)
                return True
        return False

    def claim_files(self, track: str, files: list[str]) -> list[str]:
        """Claim files for a track, returning any conflicts.

        Args:
            track: Track claiming the files
            files: Files to claim

        Returns:
            List of files already claimed by other tracks
        """
        data = self._load()
        conflicts: list[str] = []

        # Check for conflicts
        for session in data["sessions"]:
            if session.get("track") == track:
                continue
            claimed = set(session.get("files_claimed", []))
            overlap = claimed & set(files)
            if overlap:
                conflicts.extend(f"{f} (claimed by {session['track']})" for f in overlap)

        # Update the track's claimed files
        for session in data["sessions"]:
            if session.get("track") == track:
                existing = set(session.get("files_claimed", []))
                session["files_claimed"] = list(existing | set(files))
                self._save(data)
                break

        return conflicts

    def list_active(self) -> list[SessionEntry]:
        """List all active sessions.

        Returns:
            List of active SessionEntry objects
        """
        data = self._load()
        entries: list[SessionEntry] = []

        for s in data["sessions"]:
            if s.get("status", "active") != "active":
                continue
            entries.append(
                SessionEntry(
                    track=s.get("track", ""),
                    worktree=s.get("worktree", ""),
                    agent=s.get("agent", "claude"),
                    current_goal=s.get("current_goal", ""),
                    started_at=s.get("started_at", ""),
                    files_claimed=s.get("files_claimed", []),
                    status=s.get("status", "active"),
                    pid=s.get("pid", 0),
                )
            )

        return entries

    def detect_overlapping_goals(self) -> list[tuple[str, str, str]]:
        """Detect sessions with potentially overlapping goals.

        Returns:
            List of (track1, track2, overlap_description) tuples
        """
        active = self.list_active()
        overlaps: list[tuple[str, str, str]] = []

        for i, a in enumerate(active):
            for b in active[i + 1 :]:
                # Check file claim overlaps
                a_files = set(a.files_claimed)
                b_files = set(b.files_claimed)
                shared = a_files & b_files
                if shared:
                    overlaps.append(
                        (
                            a.track,
                            b.track,
                            f"Shared files: {', '.join(list(shared)[:3])}",
                        )
                    )

        return overlaps

    def cleanup_stale(self, max_age_hours: float = 24.0) -> int:
        """Remove sessions older than max_age_hours.

        Returns:
            Number of sessions cleaned up
        """
        data = self._load()
        now = datetime.now(timezone.utc)
        cleaned = 0

        active_sessions = []
        for session in data["sessions"]:
            started = session.get("started_at", "")
            if started:
                try:
                    start_time = datetime.fromisoformat(started)
                    age_hours = (now - start_time).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        cleaned += 1
                        continue
                except (ValueError, TypeError):
                    pass
            active_sessions.append(session)

        if cleaned > 0:
            data["sessions"] = active_sessions
            self._save(data)
            logger.info("session_cleanup removed=%d", cleaned)

        return cleaned


__all__ = [
    "SessionEntry",
    "SessionManifest",
]
