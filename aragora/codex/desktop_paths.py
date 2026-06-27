"""Canonical paths for Codex Desktop local state.

All paths are derived from a single ``home`` directory (default ``~/.codex``)
that can be overridden via the ``ARAGORA_CODEX_HOME`` environment variable for
tests and alternate installs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CODEX_HOME = Path("~/.codex").expanduser()
HOME_ENV_VAR = "ARAGORA_CODEX_HOME"


@dataclass(frozen=True, slots=True)
class CodexDesktopPaths:
    """Frozen view of the Codex Desktop on-disk layout.

    Build via :func:`resolve` to honor ``ARAGORA_CODEX_HOME``.
    """

    home: Path

    @property
    def sqlite_path(self) -> Path:
        return self.home / "state_5.sqlite"

    @property
    def sessions_root(self) -> Path:
        return self.home / "sessions"

    @property
    def global_state_path(self) -> Path:
        return self.home / ".codex-global-state.json"

    @property
    def session_index_path(self) -> Path:
        return self.home / "session_index.jsonl"


def resolve(home: str | os.PathLike[str] | None = None) -> CodexDesktopPaths:
    """Return canonical paths, honoring ``ARAGORA_CODEX_HOME`` when ``home`` is None."""
    if home is not None:
        return CodexDesktopPaths(home=Path(home).expanduser())
    override = os.environ.get(HOME_ENV_VAR)
    if override:
        return CodexDesktopPaths(home=Path(override).expanduser())
    return CodexDesktopPaths(home=DEFAULT_CODEX_HOME)
