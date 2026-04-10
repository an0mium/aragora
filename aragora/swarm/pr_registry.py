"""YAML-backed registry for canonical PR tracking across swarm workers.

When multiple agents (ralph repair, swarm workers, campaign projects) create
PRs for the same branch, the registry provides a single source of truth.
Superseded PRs are tracked in an audit trail on each entry.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PREntry:
    """A single canonical PR tracking entry keyed by branch name."""

    branch: str
    pr_url: str
    creator: str
    created_at: str = ""
    status: str = "active"  # "active", "merged", "closed", "superseded"
    superseded: list[dict[str, Any]] = field(default_factory=list)
    gate_snapshot: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class PullRequestRegistry:
    """YAML-backed registry for canonical PR tracking across swarm workers."""

    def __init__(self, state_dir: Path | None = None) -> None:
        self._state_dir = state_dir or Path.home() / ".aragora"
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._file = self._state_dir / "pr_registry.yaml"
        self._entries: dict[str, PREntry] = {}
        self._load()

    def _load(self) -> None:
        if self._file.exists():
            try:
                data = yaml.safe_load(self._file.read_text()) or {}
                for branch, raw in data.items():
                    if isinstance(raw, dict):
                        self._entries[branch] = PREntry(**raw)
            except (OSError, yaml.YAMLError, TypeError, KeyError):
                logger.warning("Failed to load PR registry from %s", self._file)

    def _save(self) -> None:
        data = {branch: asdict(entry) for branch, entry in self._entries.items()}
        self._file.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    def register(self, branch: str, pr_url: str, creator: str, **kwargs: Any) -> PREntry:
        """Register a PR for a branch, superseding any active PR on the same branch."""
        if branch in self._entries and self._entries[branch].status == "active":
            self.supersede(branch, pr_url, reason=f"replaced by {creator}")
            return self._entries[branch]
        entry = PREntry(branch=branch, pr_url=pr_url, creator=creator, **kwargs)
        self._entries[branch] = entry
        self._save()
        logger.info("Registered PR %s for branch %s by %s", pr_url, branch, creator)
        return entry

    def supersede(self, branch: str, new_pr_url: str, reason: str = "") -> PREntry | None:
        """Replace the active PR for a branch, recording the old one in the audit trail."""
        old = self._entries.get(branch)
        if old is None:
            return None
        old.superseded.append(
            {
                "pr_url": old.pr_url,
                "reason": reason,
                "superseded_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        old.pr_url = new_pr_url
        old.status = "active"
        self._save()
        return old

    def close(self, branch: str, outcome: str = "closed") -> PREntry | None:
        """Mark a PR as merged or closed."""
        entry = self._entries.get(branch)
        if entry:
            entry.status = outcome  # "merged" or "closed"
            self._save()
        return entry

    def get(self, branch: str) -> dict[str, Any] | None:
        """Return the entry for a branch as a dict, or None."""
        entry = self._entries.get(branch)
        return asdict(entry) if entry else None

    def list_active(self) -> list[dict[str, Any]]:
        """Return all entries with status == 'active'."""
        return [
            {"branch": b, **asdict(e)} for b, e in self._entries.items() if e.status == "active"
        ]

    def list_all(self) -> list[dict[str, Any]]:
        """Return all entries regardless of status."""
        return [{"branch": b, **asdict(e)} for b, e in self._entries.items()]
