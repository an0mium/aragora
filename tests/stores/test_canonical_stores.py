"""Tests for canonical store helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from aragora.stores import (
    CanonicalGatewayStores,
    CanonicalWorkspaceStores,
    get_canonical_gateway_stores,
    get_canonical_workspace_stores,
)


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


def test_canonical_workspace_stores_type() -> None:
    """Factory should return CanonicalWorkspaceStores instance."""
    stores = get_canonical_workspace_stores(git_enabled=False)
    assert isinstance(stores, CanonicalWorkspaceStores)


@pytest.mark.asyncio
async def test_canonical_workspace_stores_lazy_init(tmp_path: Path) -> None:
    """Stores should be lazily initialized on first access."""
    stores = get_canonical_workspace_stores(
        bead_dir=str(tmp_path / "beads"),
        convoy_dir=str(tmp_path / "convoys"),
        git_enabled=False,
    )
    # Before access, internal stores should be None
    assert stores._bead_store is None
    assert stores._convoy_manager is None

    # Access triggers initialization
    bead_store = await stores.bead_store()
    assert bead_store is not None
    assert stores._bead_store is not None

    convoy_mgr = await stores.convoy_manager()
    assert convoy_mgr is not None
    assert stores._convoy_manager is not None


@pytest.mark.asyncio
async def test_canonical_workspace_stores_reuse(tmp_path: Path) -> None:
    """Repeated access should return the same store instance."""
    stores = get_canonical_workspace_stores(
        bead_dir=str(tmp_path / "beads"),
        convoy_dir=str(tmp_path / "convoys"),
        git_enabled=False,
    )
    store1 = await stores.bead_store()
    store2 = await stores.bead_store()
    assert store1 is store2


def test_canonical_gateway_stores_type() -> None:
    """Factory should return CanonicalGatewayStores instance."""
    stores = get_canonical_gateway_stores()
    assert isinstance(stores, CanonicalGatewayStores)


def test_nomic_stores_package_exports() -> None:
    """The canonical stores package should re-export all expected symbols."""
    from aragora.nomic.stores import (
        Bead,
        BeadPriority,
        BeadRecord,
        BeadSpec,
        BeadStatus,
        BeadStore,
        BeadType,
        Convoy,
        ConvoyManager,
        ConvoyPriority,
        ConvoyProgress,
        ConvoyRecord,
        ConvoySpec,
        ConvoyStatus,
        ConvoyStore,
        create_bead_store,
        get_bead_store,
        get_convoy_manager,
        reset_bead_store,
        reset_convoy_manager,
    )

    # Verify types are importable and not None
    assert Bead is not None
    assert BeadStore is not None
    assert ConvoyManager is not None
    assert ConvoyStore is not None
    assert BeadRecord is not None
    assert ConvoyRecord is not None
    assert BeadSpec is not None
    assert ConvoySpec is not None


def test_stores_canonical_module_exports() -> None:
    """The aragora.stores module should export all canonical helpers."""
    from aragora.stores import (
        CanonicalGatewayStores,
        CanonicalWorkspaceStores,
        get_canonical_gateway_stores,
        get_canonical_workspace_stores,
    )

    assert CanonicalGatewayStores is not None
    assert CanonicalWorkspaceStores is not None
    assert callable(get_canonical_gateway_stores)
    assert callable(get_canonical_workspace_stores)
