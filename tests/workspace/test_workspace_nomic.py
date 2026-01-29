"""
Nomic-backed workspace tests.

Validates BeadManager and ConvoyTracker when wired to Nomic stores.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aragora.nomic.beads import BeadStore as NomicBeadStore
from aragora.workspace.bead import BeadManager, BeadStatus
from aragora.workspace.convoy import ConvoyStatus, ConvoyTracker


@pytest.fixture
def nomic_store(tmp_path: Path) -> NomicBeadStore:
    return NomicBeadStore(tmp_path / "beads", git_enabled=False, auto_commit=False)


@pytest.mark.asyncio
async def test_bead_manager_nomic_roundtrip(tmp_path: Path, nomic_store: NomicBeadStore):
    mgr = BeadManager(
        storage_dir=tmp_path / "beads",
        use_nomic_store=True,
        nomic_store=nomic_store,
    )

    bead = await mgr.create_bead("cv-1", "ws-1", title="Build")
    assert bead.status == BeadStatus.PENDING

    assigned = await mgr.assign_bead(bead.bead_id, "agent-1")
    assert assigned is not None
    assert assigned.status == BeadStatus.ASSIGNED

    running = await mgr.start_bead(bead.bead_id)
    assert running is not None
    assert running.status == BeadStatus.RUNNING
    assert running.started_at is not None

    done = await mgr.complete_bead(bead.bead_id, {"ok": True})
    assert done is not None
    assert done.status == BeadStatus.DONE
    assert done.result == {"ok": True}

    fetched = await mgr.get_bead(bead.bead_id)
    assert fetched is not None
    assert fetched.status == BeadStatus.DONE


@pytest.mark.asyncio
async def test_convoy_tracker_nomic_state_machine(tmp_path: Path, nomic_store: NomicBeadStore):
    bead_mgr = BeadManager(
        storage_dir=tmp_path / "beads",
        use_nomic_store=True,
        nomic_store=nomic_store,
    )
    tracker = ConvoyTracker(bead_store=nomic_store, use_nomic_store=True)

    bead = await bead_mgr.create_bead("cv-1", "ws-1", title="Step 1")
    convoy = await tracker.create_convoy(
        workspace_id="ws-1",
        rig_id="rig-1",
        name="Sprint",
        bead_ids=[bead.bead_id],
        convoy_id="cv-1",
    )
    assert convoy.status == ConvoyStatus.CREATED
    assert convoy.bead_ids == [bead.bead_id]

    await tracker.start_assigning("cv-1")
    convoy = await tracker.get_convoy("cv-1")
    assert convoy is not None
    assert convoy.status == ConvoyStatus.ASSIGNING

    await tracker.start_executing("cv-1", ["agent-1"])
    convoy = await tracker.get_convoy("cv-1")
    assert convoy is not None
    assert convoy.status == ConvoyStatus.EXECUTING
    assert convoy.assigned_agents == ["agent-1"]

    await tracker.start_merging("cv-1")
    convoy = await tracker.get_convoy("cv-1")
    assert convoy is not None
    assert convoy.status == ConvoyStatus.MERGING

    await tracker.complete_convoy("cv-1", {"merged": True})
    convoy = await tracker.get_convoy("cv-1")
    assert convoy is not None
    assert convoy.status == ConvoyStatus.DONE
    assert convoy.merge_result == {"merged": True}
