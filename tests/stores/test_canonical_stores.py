"""Tests for canonical store helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from aragora.stores import get_canonical_workspace_stores


@pytest.mark.asyncio
async def test_canonical_workspace_stores_respect_convoy_dir(tmp_path: Path) -> None:
    """Convoy manager should honor custom convoy_dir when provided."""
    bead_dir = tmp_path / "beads"
    convoy_dir = tmp_path / "convoys"

    stores = get_canonical_workspace_stores(
        bead_dir=str(bead_dir),
        convoy_dir=str(convoy_dir),
        git_enabled=False,
        auto_commit=False,
    )

    bead_store = await stores.bead_store()
    convoy_manager = await stores.convoy_manager()

    assert bead_store.bead_dir == bead_dir
    assert convoy_manager.convoy_dir == convoy_dir
