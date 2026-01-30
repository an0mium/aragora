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


def resolve_store_dir(
    *,
    workspace_root: str | Path | None = None,
    override: str | Path | None = None,
    prefer_legacy_gt: bool = True,
) -> Path:
    """Resolve the canonical store directory for beads/convoys."""
    if override:
        return Path(override).expanduser()

    env_dir = os.getenv("ARAGORA_STORE_DIR") or os.getenv("ARAGORA_BEAD_DIR")
    if env_dir:
        return Path(env_dir).expanduser()

    if workspace_root:
        return Path(workspace_root).expanduser() / ".aragora_beads"

    legacy_gt = Path(".gt")
    if prefer_legacy_gt and legacy_gt.exists() and not Path(".aragora_beads").exists():
        return legacy_gt / "beads"

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
