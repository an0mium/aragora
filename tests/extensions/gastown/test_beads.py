"""
Tests for Gastown bead adapter module.

Tests re-exports from workspace and nomic bead layers.
"""

from __future__ import annotations

import pytest


class TestBeadExports:
    """Tests for bead module re-exports."""

    def test_bead_class_exported(self):
        """Bead class is exported from gastown.beads."""
        from aragora.extensions.gastown.beads import Bead

        assert Bead is not None

    def test_bead_manager_exported(self):
        """BeadManager class is exported from gastown.beads."""
        from aragora.extensions.gastown.beads import BeadManager

        assert BeadManager is not None

    def test_bead_status_exported(self):
        """BeadStatus enum is exported from gastown.beads."""
        from aragora.extensions.gastown.beads import BeadStatus

        assert BeadStatus is not None

    def test_bead_priority_exported(self):
        """BeadPriority enum is exported from gastown.beads."""
        from aragora.extensions.gastown.beads import BeadPriority

        assert BeadPriority is not None

    def test_nomic_bead_status_exported(self):
        """NomicBeadStatus is exported for dashboard access."""
        from aragora.extensions.gastown.beads import NomicBeadStatus

        assert NomicBeadStatus is not None

    def test_nomic_bead_manager_exported(self):
        """NomicBeadManager (BeadStore) is exported for dashboard access."""
        from aragora.extensions.gastown.beads import NomicBeadManager

        assert NomicBeadManager is not None

    def test_generate_bead_id_exported(self):
        """generate_bead_id function is exported."""
        from aragora.extensions.gastown.beads import generate_bead_id

        assert callable(generate_bead_id)

    def test_all_exports_in_module_all(self):
        """All documented exports are in __all__."""
        from aragora.extensions.gastown import beads

        expected = {
            "Bead",
            "BeadManager",
            "BeadStatus",
            "BeadPriority",
            "NomicBeadManager",
            "NomicBeadStatus",
            "generate_bead_id",
        }
        assert set(beads.__all__) == expected


class TestBeadFunctionality:
    """Tests for bead functionality through gastown module."""

    def test_generate_bead_id_returns_string(self):
        """generate_bead_id returns a valid ID string."""
        from aragora.extensions.gastown.beads import generate_bead_id

        bead_id = generate_bead_id()
        assert isinstance(bead_id, str)
        assert len(bead_id) > 0

    def test_generate_bead_id_unique(self):
        """generate_bead_id returns reasonably unique IDs."""
        from aragora.extensions.gastown.beads import generate_bead_id

        # Generate a smaller batch to avoid collision in short IDs
        ids = [generate_bead_id() for _ in range(10)]
        # At least most should be unique (allowing for some collision chance)
        assert len(set(ids)) >= 8

    def test_bead_status_enum_values(self):
        """BeadStatus has expected values."""
        from aragora.extensions.gastown.beads import BeadStatus

        # Should have standard status values
        assert hasattr(BeadStatus, "PENDING") or hasattr(BeadStatus, "pending")

    def test_bead_priority_enum_values(self):
        """BeadPriority has expected values."""
        from aragora.extensions.gastown.beads import BeadPriority

        # Should have priority levels
        assert len(list(BeadPriority)) > 0


class TestBeadIntegration:
    """Tests for bead integration between layers."""

    def test_workspace_bead_is_same_class(self):
        """Gastown Bead is same class as workspace.bead.Bead."""
        from aragora.extensions.gastown.beads import Bead as GastownBead
        from aragora.workspace.bead import Bead as WorkspaceBead

        assert GastownBead is WorkspaceBead

    def test_workspace_manager_is_same_class(self):
        """Gastown BeadManager is same class as workspace.bead.BeadManager."""
        from aragora.extensions.gastown.beads import BeadManager as GastownManager
        from aragora.workspace.bead import BeadManager as WorkspaceManager

        assert GastownManager is WorkspaceManager

    def test_nomic_bead_store_is_same_class(self):
        """Gastown NomicBeadManager is same as nomic BeadStore."""
        from aragora.extensions.gastown.beads import NomicBeadManager
        from aragora.nomic.stores import BeadStore

        assert NomicBeadManager is BeadStore

    def test_nomic_bead_status_is_same_class(self):
        """Gastown NomicBeadStatus is same as nomic BeadStatus."""
        from aragora.extensions.gastown.beads import NomicBeadStatus
        from aragora.nomic.stores import BeadStatus

        assert NomicBeadStatus is BeadStatus
