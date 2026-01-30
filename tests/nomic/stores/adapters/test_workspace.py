"""Tests for workspace <-> nomic adapter layer.

Verifies status translations and metadata encoding between workspace
beads/convoys and the canonical nomic stores.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from aragora.nomic.beads import Bead as NomicBead, BeadStatus as NomicBeadStatus, BeadType
from aragora.nomic.convoys import Convoy as NomicConvoy, ConvoyStatus as NomicConvoyStatus
from aragora.nomic.stores.adapters.workspace import (
    workspace_bead_status_to_nomic,
    workspace_bead_to_nomic,
    nomic_bead_to_workspace,
    workspace_bead_metadata,
    resolve_workspace_bead_status,
    workspace_convoy_status_to_nomic,
    workspace_convoy_metadata,
    nomic_convoy_to_workspace,
    resolve_workspace_convoy_status,
)


class TestBeadStatusTranslation:
    """Tests for bead status translation between workspace and nomic."""

    @pytest.mark.parametrize(
        "workspace_status,expected_nomic",
        [
            ("pending", NomicBeadStatus.PENDING),
            ("assigned", NomicBeadStatus.CLAIMED),
            ("running", NomicBeadStatus.RUNNING),
            ("done", NomicBeadStatus.COMPLETED),
            ("failed", NomicBeadStatus.FAILED),
            ("skipped", NomicBeadStatus.CANCELLED),
        ],
    )
    def test_workspace_to_nomic_status(self, workspace_status, expected_nomic):
        """Workspace status maps to correct nomic status."""
        from aragora.workspace.bead import BeadStatus

        status = BeadStatus(workspace_status)
        result = workspace_bead_status_to_nomic(status)
        assert result == expected_nomic

    @pytest.mark.parametrize(
        "nomic_status,expected_workspace",
        [
            (NomicBeadStatus.PENDING, "pending"),
            (NomicBeadStatus.CLAIMED, "assigned"),
            (NomicBeadStatus.RUNNING, "running"),
            (NomicBeadStatus.COMPLETED, "done"),
            (NomicBeadStatus.FAILED, "failed"),
            (NomicBeadStatus.CANCELLED, "skipped"),
            (NomicBeadStatus.BLOCKED, "pending"),  # BLOCKED maps to pending
        ],
    )
    def test_nomic_to_workspace_status(self, nomic_status, expected_workspace):
        """Nomic status maps to correct workspace status."""
        from aragora.workspace.bead import BeadStatus

        result = resolve_workspace_bead_status(nomic_status, {}, BeadStatus)
        assert result.value == expected_workspace

    def test_workspace_status_roundtrip(self):
        """Status survives workspace -> nomic -> workspace roundtrip."""
        from aragora.workspace.bead import BeadStatus

        for status in BeadStatus:
            nomic = workspace_bead_status_to_nomic(status)
            back = resolve_workspace_bead_status(
                nomic, {"workspace_status": status.value}, BeadStatus
            )
            assert back == status, f"Roundtrip failed for {status}"

    def test_metadata_preserves_workspace_status(self):
        """Workspace status is preserved in metadata during translation."""
        from aragora.workspace.bead import BeadStatus

        metadata = {"workspace_status": "assigned"}
        result = resolve_workspace_bead_status(NomicBeadStatus.CLAIMED, metadata, BeadStatus)
        assert result == BeadStatus.ASSIGNED


class TestConvoyStatusTranslation:
    """Tests for convoy status translation between workspace and nomic."""

    @pytest.mark.parametrize(
        "workspace_status,expected_nomic",
        [
            ("created", NomicConvoyStatus.PENDING),
            ("assigning", NomicConvoyStatus.ACTIVE),
            ("executing", NomicConvoyStatus.ACTIVE),
            ("merging", NomicConvoyStatus.ACTIVE),
            ("done", NomicConvoyStatus.COMPLETED),
            ("failed", NomicConvoyStatus.FAILED),
            ("cancelled", NomicConvoyStatus.CANCELLED),
        ],
    )
    def test_workspace_to_nomic_status(self, workspace_status, expected_nomic):
        """Workspace status maps to correct nomic status."""
        from aragora.workspace.convoy import ConvoyStatus

        status = ConvoyStatus(workspace_status)
        result = workspace_convoy_status_to_nomic(status)
        assert result == expected_nomic

    @pytest.mark.parametrize(
        "nomic_status,expected_workspace",
        [
            (NomicConvoyStatus.PENDING, "created"),
            (NomicConvoyStatus.ACTIVE, "executing"),
            (NomicConvoyStatus.COMPLETED, "done"),
            (NomicConvoyStatus.FAILED, "failed"),
            (NomicConvoyStatus.CANCELLED, "cancelled"),
            (NomicConvoyStatus.PARTIAL, "executing"),
        ],
    )
    def test_nomic_to_workspace_status(self, nomic_status, expected_workspace):
        """Nomic status maps to correct workspace status."""
        from aragora.workspace.convoy import ConvoyStatus

        result = resolve_workspace_convoy_status(nomic_status, {}, ConvoyStatus)
        assert result.value == expected_workspace

    def test_workspace_status_roundtrip_with_metadata(self):
        """Sub-states (ASSIGNING, MERGING) survive roundtrip via metadata."""
        from aragora.workspace.convoy import ConvoyStatus

        # These sub-states all map to ACTIVE, but should roundtrip via metadata
        for status in [ConvoyStatus.ASSIGNING, ConvoyStatus.EXECUTING, ConvoyStatus.MERGING]:
            nomic = workspace_convoy_status_to_nomic(status)
            assert nomic == NomicConvoyStatus.ACTIVE
            back = resolve_workspace_convoy_status(
                nomic, {"workspace_status": status.value}, ConvoyStatus
            )
            assert back == status, f"Roundtrip failed for {status}"


class TestBeadConversion:
    """Tests for full bead object conversion."""

    def test_workspace_bead_to_nomic(self):
        """Workspace bead converts to nomic bead correctly."""
        from aragora.workspace.bead import Bead, BeadStatus
        import time

        workspace_bead = Bead(
            bead_id="bd-12345",
            convoy_id="cv-67890",
            workspace_id="ws-abcde",
            title="Test bead",
            description="A test bead",
            status=BeadStatus.RUNNING,
            assigned_agent="agent-001",
            payload={"key": "value"},
            depends_on=["bd-00001"],
            created_at=time.time(),
            updated_at=time.time(),
        )

        nomic_bead = workspace_bead_to_nomic(workspace_bead)

        assert nomic_bead.id == "bd-12345"
        assert nomic_bead.title == "Test bead"
        assert nomic_bead.status == NomicBeadStatus.RUNNING
        assert nomic_bead.claimed_by == "agent-001"
        assert nomic_bead.dependencies == ["bd-00001"]
        assert nomic_bead.metadata["convoy_id"] == "cv-67890"
        assert nomic_bead.metadata["workspace_id"] == "ws-abcde"
        assert nomic_bead.metadata["payload"] == {"key": "value"}

    def test_nomic_bead_to_workspace(self):
        """Nomic bead converts to workspace bead correctly."""
        from aragora.workspace.bead import Bead, BeadStatus

        now = datetime.now(timezone.utc)
        nomic_bead = NomicBead(
            id="bd-12345",
            bead_type=BeadType.TASK,
            status=NomicBeadStatus.COMPLETED,
            title="Test bead",
            description="A test bead",
            created_at=now,
            updated_at=now,
            claimed_by="agent-001",
            dependencies=["bd-00001"],
            metadata={
                "convoy_id": "cv-67890",
                "workspace_id": "ws-abcde",
                "payload": {"key": "value"},
                "workspace_status": "done",
            },
        )

        workspace_bead = nomic_bead_to_workspace(nomic_bead, bead_cls=Bead, status_cls=BeadStatus)

        assert workspace_bead.bead_id == "bd-12345"
        assert workspace_bead.convoy_id == "cv-67890"
        assert workspace_bead.workspace_id == "ws-abcde"
        assert workspace_bead.status == BeadStatus.DONE
        assert workspace_bead.assigned_agent == "agent-001"
        assert workspace_bead.depends_on == ["bd-00001"]


class TestConvoyConversion:
    """Tests for full convoy object conversion."""

    def test_nomic_convoy_to_workspace(self):
        """Nomic convoy converts to workspace convoy correctly."""
        from aragora.workspace.convoy import Convoy, ConvoyStatus

        now = datetime.now(timezone.utc)
        nomic_convoy = NomicConvoy(
            id="cv-12345",
            title="Test convoy",
            description="A test convoy",
            bead_ids=["bd-001", "bd-002"],
            status=NomicConvoyStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            assigned_to=["agent-001"],
            metadata={
                "workspace_id": "ws-abcde",
                "rig_id": "rig-001",
                "workspace_name": "Test",
                "workspace_description": "Test convoy",
                "workspace_status": "executing",
            },
        )

        workspace_convoy = nomic_convoy_to_workspace(
            nomic_convoy,
            convoy_cls=Convoy,
            status_cls=ConvoyStatus,
        )

        assert workspace_convoy.convoy_id == "cv-12345"
        assert workspace_convoy.workspace_id == "ws-abcde"
        assert workspace_convoy.rig_id == "rig-001"
        assert workspace_convoy.status == ConvoyStatus.EXECUTING
        assert workspace_convoy.assigned_agents == ["agent-001"]
        assert workspace_convoy.bead_ids == ["bd-001", "bd-002"]


class TestWorkspaceStatusEnumMethods:
    """Tests for the new to_nomic()/from_nomic() methods on workspace enums."""

    def test_bead_status_to_nomic(self):
        """BeadStatus.to_nomic() works correctly."""
        from aragora.workspace.bead import BeadStatus

        assert BeadStatus.PENDING.to_nomic() == NomicBeadStatus.PENDING
        assert BeadStatus.ASSIGNED.to_nomic() == NomicBeadStatus.CLAIMED
        assert BeadStatus.RUNNING.to_nomic() == NomicBeadStatus.RUNNING
        assert BeadStatus.DONE.to_nomic() == NomicBeadStatus.COMPLETED
        assert BeadStatus.FAILED.to_nomic() == NomicBeadStatus.FAILED
        assert BeadStatus.SKIPPED.to_nomic() == NomicBeadStatus.CANCELLED

    def test_bead_status_from_nomic(self):
        """BeadStatus.from_nomic() works correctly."""
        from aragora.workspace.bead import BeadStatus

        assert BeadStatus.from_nomic(NomicBeadStatus.PENDING) == BeadStatus.PENDING
        assert BeadStatus.from_nomic(NomicBeadStatus.CLAIMED) == BeadStatus.ASSIGNED
        assert BeadStatus.from_nomic(NomicBeadStatus.RUNNING) == BeadStatus.RUNNING
        assert BeadStatus.from_nomic(NomicBeadStatus.COMPLETED) == BeadStatus.DONE
        assert BeadStatus.from_nomic(NomicBeadStatus.FAILED) == BeadStatus.FAILED
        assert BeadStatus.from_nomic(NomicBeadStatus.CANCELLED) == BeadStatus.SKIPPED
        assert BeadStatus.from_nomic(NomicBeadStatus.BLOCKED) == BeadStatus.PENDING

    def test_convoy_status_to_nomic(self):
        """ConvoyStatus.to_nomic() works correctly."""
        from aragora.workspace.convoy import ConvoyStatus

        assert ConvoyStatus.CREATED.to_nomic() == NomicConvoyStatus.PENDING
        assert ConvoyStatus.ASSIGNING.to_nomic() == NomicConvoyStatus.ACTIVE
        assert ConvoyStatus.EXECUTING.to_nomic() == NomicConvoyStatus.ACTIVE
        assert ConvoyStatus.MERGING.to_nomic() == NomicConvoyStatus.ACTIVE
        assert ConvoyStatus.DONE.to_nomic() == NomicConvoyStatus.COMPLETED
        assert ConvoyStatus.FAILED.to_nomic() == NomicConvoyStatus.FAILED
        assert ConvoyStatus.CANCELLED.to_nomic() == NomicConvoyStatus.CANCELLED

    def test_convoy_status_from_nomic_with_metadata(self):
        """ConvoyStatus.from_nomic() preserves sub-states via metadata."""
        from aragora.workspace.convoy import ConvoyStatus

        # Without metadata, ACTIVE maps to EXECUTING
        assert ConvoyStatus.from_nomic(NomicConvoyStatus.ACTIVE) == ConvoyStatus.EXECUTING

        # With metadata, sub-states are preserved
        assert (
            ConvoyStatus.from_nomic(NomicConvoyStatus.ACTIVE, "assigning") == ConvoyStatus.ASSIGNING
        )
        assert ConvoyStatus.from_nomic(NomicConvoyStatus.ACTIVE, "merging") == ConvoyStatus.MERGING


class TestGastownStatusMethods:
    """Tests for gastown ConvoyStatus conversion methods."""

    def test_gastown_to_nomic(self):
        """GastownConvoyStatus.to_nomic() works correctly."""
        from aragora.extensions.gastown.models import ConvoyStatus as GastownConvoyStatus

        assert GastownConvoyStatus.PENDING.to_nomic() == NomicConvoyStatus.PENDING
        assert GastownConvoyStatus.IN_PROGRESS.to_nomic() == NomicConvoyStatus.ACTIVE
        assert GastownConvoyStatus.BLOCKED.to_nomic() == NomicConvoyStatus.FAILED
        assert GastownConvoyStatus.REVIEW.to_nomic() == NomicConvoyStatus.ACTIVE
        assert GastownConvoyStatus.COMPLETED.to_nomic() == NomicConvoyStatus.COMPLETED
        assert GastownConvoyStatus.CANCELLED.to_nomic() == NomicConvoyStatus.CANCELLED

    def test_gastown_from_nomic(self):
        """GastownConvoyStatus.from_nomic() works correctly."""
        from aragora.extensions.gastown.models import ConvoyStatus as GastownConvoyStatus

        assert (
            GastownConvoyStatus.from_nomic(NomicConvoyStatus.PENDING) == GastownConvoyStatus.PENDING
        )
        assert (
            GastownConvoyStatus.from_nomic(NomicConvoyStatus.ACTIVE)
            == GastownConvoyStatus.IN_PROGRESS
        )
        assert (
            GastownConvoyStatus.from_nomic(NomicConvoyStatus.FAILED) == GastownConvoyStatus.BLOCKED
        )
        assert (
            GastownConvoyStatus.from_nomic(NomicConvoyStatus.COMPLETED)
            == GastownConvoyStatus.COMPLETED
        )
        assert (
            GastownConvoyStatus.from_nomic(NomicConvoyStatus.CANCELLED)
            == GastownConvoyStatus.CANCELLED
        )

    def test_gastown_from_nomic_with_metadata(self):
        """GastownConvoyStatus.from_nomic() preserves sub-states via metadata."""
        from aragora.extensions.gastown.models import ConvoyStatus as GastownConvoyStatus

        # Without metadata
        assert (
            GastownConvoyStatus.from_nomic(NomicConvoyStatus.ACTIVE)
            == GastownConvoyStatus.IN_PROGRESS
        )

        # With metadata, REVIEW is preserved
        assert (
            GastownConvoyStatus.from_nomic(NomicConvoyStatus.ACTIVE, "review")
            == GastownConvoyStatus.REVIEW
        )

    def test_gastown_to_workspace_mapping(self):
        """GastownConvoyStatus maps correctly to workspace status."""
        from aragora.extensions.gastown.models import ConvoyStatus as GastownConvoyStatus

        assert GastownConvoyStatus.PENDING.to_workspace_status() == "created"
        assert GastownConvoyStatus.IN_PROGRESS.to_workspace_status() == "executing"
        assert GastownConvoyStatus.BLOCKED.to_workspace_status() == "failed"
        assert GastownConvoyStatus.REVIEW.to_workspace_status() == "merging"
        assert GastownConvoyStatus.COMPLETED.to_workspace_status() == "done"
        assert GastownConvoyStatus.CANCELLED.to_workspace_status() == "cancelled"
