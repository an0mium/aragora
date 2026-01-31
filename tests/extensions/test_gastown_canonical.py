"""Tests for Gastown canonical ownership and convoy interoperability.

Confirms that gastown and workspace ConvoyTracker implementations share
the Nomic backend and that status mappings are bidirectional.
"""

from __future__ import annotations

import pytest

from aragora.extensions.gastown.models import ConvoyStatus as GasConvoyStatus
from aragora.workspace.convoy import ConvoyStatus as WorkspaceConvoyStatus


class TestStatusMapping:
    """Verify bidirectional status mapping between gastown and workspace."""

    def test_gastown_to_workspace_roundtrip(self):
        """Every gastown status maps to a workspace status and back."""
        for gas_status in GasConvoyStatus:
            ws_value = gas_status.to_workspace_status()
            # Must be a valid workspace status value
            assert ws_value in {s.value for s in WorkspaceConvoyStatus}
            # Round-trip back
            back = GasConvoyStatus.from_workspace_status(ws_value)
            assert isinstance(back, GasConvoyStatus)

    def test_workspace_to_gastown_all_values(self):
        """Every workspace status value maps to a gastown status."""
        for ws_status in WorkspaceConvoyStatus:
            gas = GasConvoyStatus.from_workspace_status(ws_status.value)
            assert isinstance(gas, GasConvoyStatus)

    def test_completed_maps_correctly(self):
        assert GasConvoyStatus.COMPLETED.to_workspace_status() == "done"
        assert GasConvoyStatus.from_workspace_status("done") == GasConvoyStatus.COMPLETED

    def test_cancelled_maps_correctly(self):
        assert GasConvoyStatus.CANCELLED.to_workspace_status() == "cancelled"
        assert GasConvoyStatus.from_workspace_status("cancelled") == GasConvoyStatus.CANCELLED

    def test_in_progress_maps_to_executing(self):
        assert GasConvoyStatus.IN_PROGRESS.to_workspace_status() == "executing"
        assert GasConvoyStatus.from_workspace_status("executing") == GasConvoyStatus.IN_PROGRESS

    def test_pending_maps_to_created(self):
        assert GasConvoyStatus.PENDING.to_workspace_status() == "created"
        assert GasConvoyStatus.from_workspace_status("created") == GasConvoyStatus.PENDING

    def test_unknown_workspace_value_defaults_to_pending(self):
        assert GasConvoyStatus.from_workspace_status("unknown_value") == GasConvoyStatus.PENDING


class TestSharedNomicBackend:
    """Confirm both trackers import from the same Nomic backend."""

    def test_both_import_nomic_convoy_manager(self):
        """Both gastown and workspace ConvoyTracker use NomicConvoyManager."""
        from aragora.extensions.gastown.convoy import ConvoyTracker as GasTracker
        from aragora.workspace.convoy import ConvoyTracker as WsTracker

        # Both should be importable (confirms no circular imports)
        assert GasTracker is not None
        assert WsTracker is not None

    def test_nomic_convoy_manager_is_same_class(self):
        """Both trackers reference the same NomicConvoyManager class."""
        import aragora.extensions.gastown.convoy as gas_mod
        import aragora.workspace.convoy as ws_mod

        # Both import NomicConvoyManager from aragora.nomic.stores
        assert gas_mod.NomicConvoyManager is ws_mod.NomicConvoyManager

    def test_nomic_bead_store_is_same_class(self):
        """Both trackers reference the same NomicBeadStore class."""
        import aragora.extensions.gastown.convoy as gas_mod
        import aragora.workspace.convoy as ws_mod

        assert gas_mod.NomicBeadStore is ws_mod.NomicBeadStore

    @pytest.mark.asyncio
    async def test_gastown_tracker_creates_with_canonical_store(self, tmp_path):
        """Gastown tracker works with canonical store by default."""
        from aragora.extensions.gastown.convoy import ConvoyTracker

        tracker = ConvoyTracker(storage_path=tmp_path / "convoys")
        convoy = await tracker.create_convoy(
            rig_id="rig-1",
            title="Test convoy",
        )
        assert convoy.id
        assert convoy.status == GasConvoyStatus.PENDING

    @pytest.mark.asyncio
    async def test_workspace_tracker_creates_with_canonical_store(self, tmp_path):
        """Workspace tracker works with canonical store by default."""
        from aragora.workspace.convoy import ConvoyTracker
        from aragora.nomic.stores import BeadStore as NomicBeadStore

        bead_store = NomicBeadStore(tmp_path / "beads", git_enabled=False, auto_commit=False)
        tracker = ConvoyTracker(bead_store=bead_store, use_nomic_store=True)
        convoy = await tracker.create_convoy(
            workspace_id="ws-1",
            rig_id="rig-1",
            name="Test convoy",
        )
        assert convoy.convoy_id
        assert convoy.status == WorkspaceConvoyStatus.CREATED
