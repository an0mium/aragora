"""
Tests for ConvoyRecord and BeadRecord protocols.

Verifies that all 3 Convoy implementations satisfy the ConvoyRecord protocol:
- Canonical (aragora.nomic.convoys.Convoy)
- Workspace (aragora.workspace.convoy.Convoy)
- Gastown (aragora.extensions.gastown.models.Convoy)
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aragora.nomic.stores.protocols import BeadRecord, ConvoyRecord


class TestConvoyRecordProtocol:
    """Tests for ConvoyRecord protocol compliance."""

    def test_canonical_convoy_satisfies_protocol(self) -> None:
        """Canonical Convoy should satisfy ConvoyRecord protocol."""
        from aragora.nomic.convoys import Convoy, ConvoyStatus

        now = datetime.now(timezone.utc)
        convoy = Convoy(
            id="cv-test-123",
            title="Test Convoy",
            description="A test convoy",
            bead_ids=["bead-1", "bead-2"],
            status=ConvoyStatus.ACTIVE,
            assigned_to=["agent-1", "agent-2"],
            created_at=now,
            updated_at=now,
            error_message="Test error",
            metadata={"key": "value"},
        )

        # Check isinstance (requires @runtime_checkable)
        assert isinstance(convoy, ConvoyRecord)

        # Verify protocol properties
        assert convoy.convoy_id == "cv-test-123"
        assert convoy.convoy_title == "Test Convoy"
        assert convoy.convoy_description == "A test convoy"
        assert convoy.convoy_bead_ids == ["bead-1", "bead-2"]
        assert convoy.convoy_status_value == "active"
        assert convoy.convoy_assigned_agents == ["agent-1", "agent-2"]
        assert convoy.convoy_error == "Test error"
        assert convoy.convoy_metadata == {"key": "value"}

        # Timestamps should be datetime
        assert isinstance(convoy.convoy_created_at, datetime)
        assert isinstance(convoy.convoy_updated_at, datetime)

    def test_workspace_convoy_satisfies_protocol(self) -> None:
        """Workspace Convoy should satisfy ConvoyRecord protocol."""
        from aragora.workspace.convoy import Convoy, ConvoyStatus
        import time

        now = time.time()
        convoy = Convoy(
            convoy_id="ws-cv-456",
            workspace_id="ws-1",
            rig_id="rig-1",
            name="Workspace Convoy",
            description="A workspace convoy",
            status=ConvoyStatus.EXECUTING,
            bead_ids=["bead-3"],
            assigned_agents=["agent-3"],
            created_at=now,
            updated_at=now,
            error="Workspace error",
            metadata={"ws_key": "ws_value"},
        )

        # Check isinstance
        assert isinstance(convoy, ConvoyRecord)

        # Verify protocol properties
        # Note: workspace uses convoy_id field, which satisfies the protocol
        assert convoy.convoy_id == "ws-cv-456"
        assert convoy.convoy_title == "Workspace Convoy"
        assert convoy.convoy_description == "A workspace convoy"
        assert convoy.convoy_bead_ids == ["bead-3"]
        assert convoy.convoy_status_value == "executing"
        assert convoy.convoy_assigned_agents == ["agent-3"]
        assert convoy.convoy_error == "Workspace error"
        assert convoy.convoy_metadata == {"ws_key": "ws_value"}

        # Timestamps should be converted from float to datetime
        assert isinstance(convoy.convoy_created_at, datetime)
        assert isinstance(convoy.convoy_updated_at, datetime)

    def test_gastown_convoy_satisfies_protocol(self) -> None:
        """Gastown Convoy should satisfy ConvoyRecord protocol."""
        from aragora.extensions.gastown.models import Convoy, ConvoyStatus

        now = datetime.now(timezone.utc)
        convoy = Convoy(
            id="gt-cv-789",
            rig_id="rig-2",
            title="Gastown Convoy",
            description="A gastown convoy",
            status=ConvoyStatus.IN_PROGRESS,
            created_at=now,
            updated_at=now,
            assigned_agents=["agent-4", "agent-5"],
            error="Gastown error",
            metadata={"gt_key": "gt_value"},
        )

        # Check isinstance
        assert isinstance(convoy, ConvoyRecord)

        # Verify protocol properties
        assert convoy.convoy_id == "gt-cv-789"
        assert convoy.convoy_title == "Gastown Convoy"
        assert convoy.convoy_description == "A gastown convoy"
        assert convoy.convoy_bead_ids == []  # Gastown doesn't track beads
        assert convoy.convoy_status_value == "in_progress"
        assert convoy.convoy_assigned_agents == ["agent-4", "agent-5"]
        assert convoy.convoy_error == "Gastown error"
        assert convoy.convoy_metadata == {"gt_key": "gt_value"}

        # Timestamps should be datetime
        assert isinstance(convoy.convoy_created_at, datetime)
        assert isinstance(convoy.convoy_updated_at, datetime)

    def test_protocol_import_from_stores(self) -> None:
        """ConvoyRecord should be importable from aragora.nomic.stores."""
        from aragora.nomic.stores import ConvoyRecord as ImportedConvoyRecord

        assert ImportedConvoyRecord is ConvoyRecord

    def test_all_convoys_interoperable(self) -> None:
        """All convoy types should work with protocol-typed functions."""
        from aragora.nomic.convoys import Convoy as NomicConvoy
        from aragora.nomic.convoys import ConvoyStatus as NomicStatus
        from aragora.workspace.convoy import Convoy as WorkspaceConvoy
        from aragora.workspace.convoy import ConvoyStatus as WorkspaceStatus
        from aragora.extensions.gastown.models import Convoy as GastownConvoy
        from aragora.extensions.gastown.models import ConvoyStatus as GastownStatus

        def get_convoy_summary(convoy: ConvoyRecord) -> dict:
            """Function that accepts any ConvoyRecord."""
            return {
                "id": convoy.convoy_id,
                "title": convoy.convoy_title,
                "status": convoy.convoy_status_value,
                "agents": convoy.convoy_assigned_agents,
            }

        # Create convoys of each type
        now = datetime.now(timezone.utc)
        nomic = NomicConvoy(
            id="nomic-1",
            title="Nomic",
            description="",
            bead_ids=[],
            status=NomicStatus.ACTIVE,
            assigned_to=["a1"],
            created_at=now,
            updated_at=now,
        )
        workspace = WorkspaceConvoy(
            convoy_id="ws-1",
            workspace_id="w",
            rig_id="r",
            name="Workspace",
            status=WorkspaceStatus.EXECUTING,
            assigned_agents=["a2"],
        )
        gastown = GastownConvoy(
            id="gt-1",
            rig_id="r",
            title="Gastown",
            status=GastownStatus.IN_PROGRESS,
            assigned_agents=["a3"],
        )

        # All should work with the protocol-typed function
        assert get_convoy_summary(nomic) == {
            "id": "nomic-1",
            "title": "Nomic",
            "status": "active",
            "agents": ["a1"],
        }
        assert get_convoy_summary(workspace) == {
            "id": "ws-1",
            "title": "Workspace",
            "status": "executing",
            "agents": ["a2"],
        }
        assert get_convoy_summary(gastown) == {
            "id": "gt-1",
            "title": "Gastown",
            "status": "in_progress",
            "agents": ["a3"],
        }


class TestBeadRecordProtocol:
    """Tests for BeadRecord protocol compliance."""

    def test_bead_record_import(self) -> None:
        """BeadRecord should be importable from aragora.nomic.stores."""
        from aragora.nomic.stores import BeadRecord as ImportedBeadRecord

        assert ImportedBeadRecord is BeadRecord

    def test_canonical_bead_satisfies_protocol(self) -> None:
        """Canonical Bead should satisfy BeadRecord protocol."""
        from aragora.nomic.beads import Bead, BeadStatus, BeadType

        now = datetime.now(timezone.utc)
        bead = Bead(
            id="bead-123",
            bead_type=BeadType.TASK,
            status=BeadStatus.PENDING,
            title="Test Bead",
            description="Test bead content",
            created_at=now,
            updated_at=now,
            metadata={"key": "value"},
        )

        # Check isinstance
        assert isinstance(bead, BeadRecord)

        # Verify protocol properties
        assert bead.bead_id == "bead-123"
        assert bead.bead_convoy_id is None  # Beads don't track convoy; convoys track beads
        assert bead.bead_status_value == "pending"
        assert bead.bead_content == "Test bead content"
        assert bead.bead_metadata == {"key": "value"}
        assert isinstance(bead.bead_created_at, datetime)
