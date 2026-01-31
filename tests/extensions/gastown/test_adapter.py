"""
Tests for GastownConvoyAdapter.

Tests the adapter that bridges Gastown convoys to the canonical Nomic layer.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from aragora.extensions.gastown.adapter import GastownConvoyAdapter
from aragora.extensions.gastown.models import Convoy, ConvoyStatus


class TestGastownConvoyAdapter:
    """Tests for GastownConvoyAdapter."""

    @pytest.fixture
    def adapter(self, tmp_path: Path) -> GastownConvoyAdapter:
        """Create an adapter with temp storage."""
        return GastownConvoyAdapter(storage_path=tmp_path / "adapter")

    @pytest.mark.asyncio
    async def test_create_convoy(self, adapter: GastownConvoyAdapter):
        """Test creating a convoy via adapter."""
        convoy = await adapter.create_convoy(
            rig_id="rig-1",
            title="Adapter Test",
            description="Testing adapter",
        )

        assert convoy.id
        assert convoy.rig_id == "rig-1"
        assert convoy.title == "Adapter Test"
        assert convoy.description == "Testing adapter"
        assert convoy.status == ConvoyStatus.PENDING

    @pytest.mark.asyncio
    async def test_create_convoy_with_all_params(self, adapter: GastownConvoyAdapter):
        """Test creating convoy with all optional parameters."""
        convoy = await adapter.create_convoy(
            rig_id="rig-1",
            title="Full Test",
            description="Full params test",
            issue_ref="github:123",
            parent_convoy="parent-1",
            priority=10,
            tags=["bug", "urgent"],
            metadata={"key": "value"},
        )

        assert convoy.title == "Full Test"
        assert convoy.issue_ref == "github:123"
        assert convoy.parent_convoy == "parent-1"
        assert convoy.priority == 10
        assert "bug" in convoy.tags
        assert "urgent" in convoy.tags

    @pytest.mark.asyncio
    async def test_get_convoy(self, adapter: GastownConvoyAdapter):
        """Test getting a convoy by ID."""
        created = await adapter.create_convoy(
            rig_id="rig-1",
            title="Get Test",
        )

        found = await adapter.get_convoy(created.id)
        assert found is not None
        assert found.id == created.id
        assert found.title == "Get Test"

    @pytest.mark.asyncio
    async def test_get_convoy_not_found(self, adapter: GastownConvoyAdapter):
        """Test getting a non-existent convoy returns None."""
        found = await adapter.get_convoy("nonexistent-id")
        assert found is None

    @pytest.mark.asyncio
    async def test_list_convoys_empty(self, adapter: GastownConvoyAdapter):
        """Test listing convoys when none exist."""
        convoys = await adapter.list_convoys()
        assert convoys == []

    @pytest.mark.asyncio
    async def test_list_convoys(self, adapter: GastownConvoyAdapter):
        """Test listing all convoys."""
        await adapter.create_convoy(rig_id="rig-1", title="Convoy 1")
        await adapter.create_convoy(rig_id="rig-2", title="Convoy 2")

        convoys = await adapter.list_convoys()
        assert len(convoys) == 2
        titles = {c.title for c in convoys}
        assert titles == {"Convoy 1", "Convoy 2"}

    @pytest.mark.asyncio
    async def test_list_convoys_filter_by_rig(self, adapter: GastownConvoyAdapter):
        """Test filtering convoys by rig_id."""
        await adapter.create_convoy(rig_id="rig-1", title="Convoy 1")
        await adapter.create_convoy(rig_id="rig-1", title="Convoy 2")
        await adapter.create_convoy(rig_id="rig-2", title="Convoy 3")

        rig1_convoys = await adapter.list_convoys(rig_id="rig-1")
        assert len(rig1_convoys) == 2
        for c in rig1_convoys:
            assert c.rig_id == "rig-1"

    @pytest.mark.asyncio
    async def test_list_convoys_filter_by_status(self, adapter: GastownConvoyAdapter):
        """Test filtering convoys by status."""
        c1 = await adapter.create_convoy(rig_id="rig-1", title="Pending")
        await adapter.create_convoy(rig_id="rig-1", title="Also Pending")

        # Start one convoy
        await adapter._tracker.start_convoy(c1.id, "agent-1")

        pending = await adapter.list_convoys(status=ConvoyStatus.PENDING)
        assert len(pending) == 1
        assert pending[0].title == "Also Pending"

        in_progress = await adapter.list_convoys(status=ConvoyStatus.IN_PROGRESS)
        assert len(in_progress) == 1
        assert in_progress[0].title == "Pending"


class TestGastownConvoyAdapterWithCustomTracker:
    """Tests for adapter with custom tracker."""

    @pytest.mark.asyncio
    async def test_adapter_uses_provided_tracker(self, tmp_path: Path):
        """Test that adapter uses a provided tracker instance."""
        from aragora.extensions.gastown.convoy import ConvoyTracker

        tracker = ConvoyTracker(storage_path=tmp_path / "custom")
        adapter = GastownConvoyAdapter(tracker=tracker)

        # Create via adapter
        convoy = await adapter.create_convoy(
            rig_id="rig-1",
            title="Via Adapter",
        )

        # Should be visible via tracker directly
        found = await tracker.get_convoy(convoy.id)
        assert found is not None
        assert found.title == "Via Adapter"

    @pytest.mark.asyncio
    async def test_adapter_default_tracker(self):
        """Test adapter creates default tracker when none provided."""
        adapter = GastownConvoyAdapter()
        assert adapter._tracker is not None

        # Should work without storage
        convoy = await adapter.create_convoy(
            rig_id="rig-1",
            title="Default Tracker",
        )
        assert convoy.id
