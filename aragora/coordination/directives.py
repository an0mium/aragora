"""Durable directive board for cross-session coordination.

This stores the current assignment for each session or role in a single JSON
document under ``.aragora_coordination/directives.json``. The board is a thin,
durable layer above the existing coordination primitives: claims are still
handled by :mod:`aragora.coordination.claims`, findings by the event bus, and
session liveness by the registry.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_COORD_DIR = ".aragora_coordination"
_DIRECTIVES_FILE = "directives.json"


@dataclass
class SessionDirective:
    """Current assignment for one session or role."""

    target: str
    task: str
    scope: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    assigned_by: str = ""
    status: str = "active"
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SessionDirective:
        return cls(
            target=str(data.get("target", "")),
            task=str(data.get("task", "")),
            scope=[str(item) for item in list(data.get("scope", []))],
            constraints=[str(item) for item in list(data.get("constraints", []))],
            assigned_by=str(data.get("assigned_by", "")),
            status=str(data.get("status", "active") or "active"),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            updated_at=float(data.get("updated_at", 0.0) or 0.0),
        )


class DirectiveBoard:
    """Atomic JSON-backed board of current session directives."""

    def __init__(self, repo_path: Path | None = None):
        self.repo_path = (repo_path or Path.cwd()).resolve()
        self._coord_dir = self.repo_path / _COORD_DIR
        self._path = self._coord_dir / _DIRECTIVES_FILE

    def _ensure_dir(self) -> None:
        self._coord_dir.mkdir(parents=True, exist_ok=True)

    def _default_payload(self) -> dict[str, object]:
        return {
            "version": 1,
            "updated_at": 0.0,
            "directives": {},
        }

    def _read_unlocked(self, fh) -> dict[str, object]:
        fh.seek(0)
        raw = fh.read()
        if not raw.strip():
            return self._default_payload()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return self._default_payload()
        if not isinstance(data, dict):
            return self._default_payload()
        directives = data.get("directives")
        if not isinstance(directives, dict):
            directives = {}
        return {
            "version": int(data.get("version", 1) or 1),
            "updated_at": float(data.get("updated_at", 0.0) or 0.0),
            "directives": directives,
        }

    def _write_unlocked(self, fh, payload: dict[str, object]) -> None:
        fh.seek(0)
        fh.truncate()
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())

    def _mutate(self, fn):
        self._ensure_dir()
        with open(self._path, "a+", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                payload = self._read_unlocked(fh)
                result = fn(payload)
                payload["updated_at"] = time.time()
                self._write_unlocked(fh, payload)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
        return result

    def _read(self) -> dict[str, object]:
        self._ensure_dir()
        with open(self._path, "a+", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_SH)
            try:
                return self._read_unlocked(fh)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def assign(
        self,
        target: str,
        task: str,
        *,
        scope: list[str] | None = None,
        constraints: list[str] | None = None,
        assigned_by: str = "",
        status: str = "active",
    ) -> SessionDirective:
        def _update(payload: dict[str, object]) -> SessionDirective:
            directives = dict(payload.get("directives", {}))
            now = time.time()
            existing = directives.get(target)
            created_at = now
            if isinstance(existing, dict):
                created_at = float(existing.get("created_at", now) or now)
            directive = SessionDirective(
                target=target,
                task=task,
                scope=list(scope or []),
                constraints=list(constraints or []),
                assigned_by=assigned_by,
                status=status,
                created_at=created_at,
                updated_at=now,
            )
            directives[target] = directive.to_dict()
            payload["directives"] = directives
            return directive

        return self._mutate(_update)

    def clear(self, target: str) -> bool:
        def _clear(payload: dict[str, object]) -> bool:
            directives = dict(payload.get("directives", {}))
            if target not in directives:
                return False
            directives.pop(target, None)
            payload["directives"] = directives
            return True

        return self._mutate(_clear)

    def get(self, target: str) -> SessionDirective | None:
        payload = self._read()
        directives = payload.get("directives", {})
        if not isinstance(directives, dict):
            return None
        record = directives.get(target)
        if not isinstance(record, dict):
            return None
        return SessionDirective.from_dict(record)

    def list(self) -> list[SessionDirective]:
        payload = self._read()
        directives = payload.get("directives", {})
        if not isinstance(directives, dict):
            return []
        return [
            SessionDirective.from_dict(record)
            for _target, record in sorted(directives.items())
            if isinstance(record, dict)
        ]


__all__ = [
    "DirectiveBoard",
    "SessionDirective",
]
