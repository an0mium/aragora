"""
Canonical path resolver for bead/convoy storage.

Order:
1) explicit override
2) ARAGORA_STORE_DIR / ARAGORA_BEAD_DIR env
3) workspace_root/.aragora_beads (if workspace_root provided)
4) legacy .gt if present and .aragora_beads absent
5) cwd/.aragora_beads
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


def _env_store_dir() -> Path | None:
    env_dir = os.getenv("ARAGORA_STORE_DIR") or os.getenv("ARAGORA_BEAD_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return None


def _should_persist_canonical_store() -> bool:
    env_value = os.getenv("ARAGORA_CANONICAL_STORE_PERSIST") or os.getenv(
        "NOMIC_CANONICAL_STORE_PERSIST"
    )
    if env_value is not None:
        return env_value.strip().lower() in {"1", "true", "yes", "on"}
    if _env_store_dir() is not None:
        return True
    return False


def resolve_store_dir(
    *,
    workspace_root: str | Path | None = None,
    override: str | Path | None = None,
    prefer_legacy_gt: bool = True,
) -> Path:
    """Resolve the canonical store directory for beads/convoys."""
    if override:
        return Path(override).expanduser()

    env_dir = _env_store_dir()
    if env_dir:
        return env_dir

    if workspace_root:
        return Path(workspace_root).expanduser() / ".aragora_beads"

    legacy_gt = Path(".gt")
    if prefer_legacy_gt and legacy_gt.exists() and not Path(".aragora_beads").exists():
        return legacy_gt / "beads"

    return Path(".aragora_beads")


def resolve_runtime_store_dir(
    *,
    workspace_root: str | Path | None = None,
    override: str | Path | None = None,
    prefer_legacy_gt: bool = True,
    prefer_ephemeral: bool = True,
) -> Path:
    """Resolve store dir, preferring ephemeral temp storage when unset."""
    if override:
        return Path(override).expanduser()

    env_dir = _env_store_dir()
    if env_dir:
        return env_dir

    if workspace_root:
        return Path(workspace_root).expanduser() / ".aragora_beads"

    legacy_gt = Path(".gt")
    if prefer_legacy_gt and legacy_gt.exists() and not Path(".aragora_beads").exists():
        return legacy_gt / "beads"

    if _should_persist_canonical_store():
        return resolve_store_dir(
            workspace_root=workspace_root,
            override=override,
            prefer_legacy_gt=prefer_legacy_gt,
        )

    if prefer_ephemeral:
        return Path(tempfile.mkdtemp(prefix="aragora-beads-"))

    return Path(".aragora_beads")


def resolve_bead_and_convoy_dirs(
    *,
    workspace_root: str | Path | None = None,
    override: str | Path | None = None,
    prefer_legacy_gt: bool = True,
) -> tuple[Path, Path]:
    """Return (bead_dir, convoy_dir) based on canonical store resolution."""
    store_dir = resolve_store_dir(
        workspace_root=workspace_root,
        override=override,
        prefer_legacy_gt=prefer_legacy_gt,
    )
    # Legacy .gt layout uses separate beads/convoys directories.
    if store_dir.name == "beads" and store_dir.parent.name == ".gt":
        return store_dir, store_dir.parent / "convoys"
    return store_dir, store_dir


def should_use_canonical_store(default: bool = False) -> bool:
    """Return True when env opts in to canonical bead/convoy storage."""
    env_value = os.getenv("ARAGORA_CONVOY_CANONICAL_STORE") or os.getenv(
        "NOMIC_CONVOY_CANONICAL_STORE"
    )
    if env_value is None:
        return default
    return env_value.strip().lower() in {"1", "true", "yes", "on"}
