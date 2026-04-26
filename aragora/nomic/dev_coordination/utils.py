"""Stateless helpers extracted from ``aragora.nomic.dev_coordination``.

These helpers have no dependencies on any other ``dev_coordination`` module
and can be imported safely from anywhere in the package.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc


def _safe_kill_probe(raw_pid: Any) -> Exception | None:
    """Probe whether *raw_pid* is alive.  Returns ``None`` if alive, the exception otherwise."""
    import os as _os

    try:
        _os.kill(int(raw_pid), 0)
        return None
    except (ProcessLookupError, PermissionError, TypeError, ValueError, OSError) as exc:
        return exc


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _json_dump(value: Any) -> str:
    return json.dumps(_json_compatible(value), sort_keys=True)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _artifact_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_json_compatible(payload), sort_keys=True).encode()
    ).hexdigest()


def _json_compatible(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): _json_compatible(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_compatible(item) for item in value]
    return value


def _normalize_claim(value: str) -> str:
    return value.strip().strip("/")


def _has_wildcard(pattern: str) -> bool:
    return any(token in pattern for token in ("*", "?", "["))


def _parse_worktree_entries(raw: str) -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    current_path: Path | None = None
    current_branch: str | None = None
    for line in raw.splitlines():
        text = line.strip()
        if text.startswith("worktree "):
            current_path = Path(text[len("worktree ") :]).resolve()
            current_branch = None
        elif text.startswith("branch refs/heads/"):
            current_branch = text[len("branch refs/heads/") :]
        elif text == "" and current_path is not None and current_branch is not None:
            entries.append((current_path, current_branch))
            current_path = None
            current_branch = None
    if current_path is not None and current_branch is not None:
        entries.append((current_path, current_branch))
    return entries


def _status_paths(lines: list[str]) -> list[str]:
    paths: list[str] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        path = text[3:] if len(text) > 3 else text
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(_normalize_claim(path))
    return paths


def _estimate_salvage_value(*, ahead: int, changed_paths: list[str], dirty: bool) -> float:
    value = 0.2
    if dirty:
        value += 0.2
    value += min(0.4, ahead * 0.1)
    value += min(0.2, len(set(changed_paths)) * 0.02)
    return max(0.0, min(1.0, value))


__all__ = [
    "UTC",
    "_safe_kill_probe",
    "_utcnow",
    "_parse_dt",
    "_json_dump",
    "_json_loads",
    "_artifact_hash",
    "_json_compatible",
    "_normalize_claim",
    "_has_wildcard",
    "_parse_worktree_entries",
    "_status_paths",
    "_estimate_salvage_value",
]
