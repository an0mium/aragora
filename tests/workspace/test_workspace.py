"""
Tests for Workspace Manager modules (Gastown parity).

Tests the workspace management layer including:
- Rig creation and lifecycle
- Bead creation and lifecycle
- Convoy tracking and state transitions
- WorkspaceManager integration
"""

from __future__ import annotations

import pytest

from aragora.workspace.rig import Rig, RigConfig, RigStatus
from aragora.nomic.stores import BeadStore as NomicBeadStore
from aragora.workspace.bead import Bead, BeadManager, BeadStatus, generate_bead_id
from aragora.workspace.convoy import Convoy, ConvoyStatus, ConvoyTracker
from aragora.workspace.manager import WorkspaceManager


# =============================================================================
# Rig Tests
# =============================================================================


class TestRig:
    """Test Rig dataclass."""

    def test_rig_creation(self):
        rig = Rig(rig_id="rig-abc", name="my-project", workspace_id="ws-1")
        assert rig.rig_id == "rig-abc"
        assert rig.name == "my-project"
        assert rig.status == RigStatus.INITIALIZING

    def test_rig_config_defaults(self):
        config = RigConfig()
        assert config.max_agents == 10
        assert config.max_concurrent_tasks == 5
        assert config.branch == "main"

    def test_rig_to_dict(self):
        rig = Rig(rig_id="rig-abc", name="test", workspace_id="ws-1")
        d = rig.to_dict()
        assert d["rig_id"] == "rig-abc"
        assert d["status"] == "initializing"

    def test_rig_from_dict(self):
        d = {
            "rig_id": "rig-abc",
            "name": "test",
            "workspace_id": "ws-1",
            "config": {
                "repo_url": "https://example.com",
                "repo_path": "",
                "branch": "main",
                "max_agents": 10,
                "max_concurrent_tasks": 5,
                "auto_assign_agents": True,
                "allowed_agent_types": [],
                "environment_vars": {},
                "labels": {},
            },
            "status": "ready",
            "assigned_agents": ["agent-1"],
            "active_convoys": [],
            "created_at": 1000.0,
            "updated_at": 1000.0,
            "metadata": {},
            "tasks_completed": 5,
            "tasks_failed": 1,
        }
        rig = Rig.from_dict(d)
        assert rig.status == RigStatus.READY
        assert rig.config.repo_url == "https://example.com"

    def test_rig_status_values(self):
        assert RigStatus.INITIALIZING.value == "initializing"
        assert RigStatus.READY.value == "ready"
        assert RigStatus.ACTIVE.value == "active"
        assert RigStatus.DRAINING.value == "draining"
        assert RigStatus.STOPPED.value == "stopped"
        assert RigStatus.ERROR.value == "error"


# =============================================================================
# Bead Tests
# =============================================================================


class TestBead:
    """Test Bead dataclass."""

    def test_bead_creation(self):
        bead = Bead(bead_id="bd-abc12", convoy_id="cv-1", workspace_id="ws-1")
        assert bead.bead_id == "bd-abc12"
        assert bead.status == BeadStatus.PENDING

    def test_bead_to_dict(self):
        bead = Bead(
            bead_id="bd-abc", convoy_id="cv-1", workspace_id="ws-1", status=BeadStatus.RUNNING
        )
        d = bead.to_dict()
        assert d["status"] == "running"

    def test_bead_from_dict(self):
        d = {
            "bead_id": "bd-abc",
            "convoy_id": "cv-1",
            "workspace_id": "ws-1",
            "title": "Build",
            "description": "Build the project",
            "status": "done",
            "assigned_agent": "agent-1",
            "payload": {},
            "result": {"success": True},
            "error": None,
            "created_at": 1000.0,
            "updated_at": 1000.0,
            "started_at": 1001.0,
            "completed_at": 1002.0,
            "git_ref": None,
            "depends_on": [],
            "metadata": {},
        }
        bead = Bead.from_dict(d)
        assert bead.status == BeadStatus.DONE
        assert bead.result == {"success": True}

    def test_generate_bead_id(self):
        bid = generate_bead_id("bd")
        assert bid.startswith("bd-")
        assert len(bid) == 8  # "bd-" + 5 chars


class TestBeadManager:
    """Test BeadManager."""

    @pytest.fixture
    def mgr(self, tmp_path):
        return BeadManager(storage_dir=tmp_path / "beads")

    @pytest.mark.asyncio
    async def test_create_bead(self, mgr):
        bead = await mgr.create_bead("cv-1", "ws-1", title="Build")
        assert bead.convoy_id == "cv-1"
        assert bead.title == "Build"
        assert bead.status == BeadStatus.PENDING

    @pytest.mark.asyncio
    async def test_assign_bead(self, mgr):
        bead = await mgr.create_bead("cv-1", "ws-1")
        result = await mgr.assign_bead(bead.bead_id, "agent-1")
        assert result.status == BeadStatus.ASSIGNED
        assert result.assigned_agent == "agent-1"

    @pytest.mark.asyncio
    async def test_start_bead(self, mgr):
        bead = await mgr.create_bead("cv-1", "ws-1")
        result = await mgr.start_bead(bead.bead_id)
        assert result.status == BeadStatus.RUNNING
        assert result.started_at is not None

    @pytest.mark.asyncio
    async def test_complete_bead(self, mgr):
        bead = await mgr.create_bead("cv-1", "ws-1")
        result = await mgr.complete_bead(bead.bead_id, {"output": "done"})
        assert result.status == BeadStatus.DONE
        assert result.result == {"output": "done"}

    @pytest.mark.asyncio
    async def test_fail_bead(self, mgr):
        bead = await mgr.create_bead("cv-1", "ws-1")
        result = await mgr.fail_bead(bead.bead_id, "compile error")
        assert result.status == BeadStatus.FAILED
        assert result.error == "compile error"

    @pytest.mark.asyncio
    async def test_list_beads_by_convoy(self, mgr):
        await mgr.create_bead("cv-1", "ws-1")
        await mgr.create_bead("cv-2", "ws-1")
        await mgr.create_bead("cv-1", "ws-1")
        beads = await mgr.list_beads(convoy_id="cv-1")
        assert len(beads) == 2

    @pytest.mark.asyncio
    async def test_list_beads_by_status(self, mgr):
        b1 = await mgr.create_bead("cv-1", "ws-1")
        b2 = await mgr.create_bead("cv-1", "ws-1")
        await mgr.complete_bead(b1.bead_id)
        beads = await mgr.list_beads(status=BeadStatus.DONE)
        assert len(beads) == 1

    @pytest.mark.asyncio
    async def test_get_ready_beads_no_deps(self, mgr):
        await mgr.create_bead("cv-1", "ws-1")
        await mgr.create_bead("cv-1", "ws-1")
        ready = await mgr.get_ready_beads("cv-1")
        assert len(ready) == 2

    @pytest.mark.asyncio
    async def test_get_ready_beads_with_deps(self, mgr):
        b1 = await mgr.create_bead("cv-1", "ws-1")
        b2 = await mgr.create_bead("cv-1", "ws-1", depends_on=[b1.bead_id])
        ready = await mgr.get_ready_beads("cv-1")
        assert len(ready) == 1
        assert ready[0].bead_id == b1.bead_id

    @pytest.mark.asyncio
    async def test_get_ready_beads_deps_met(self, mgr):
        b1 = await mgr.create_bead("cv-1", "ws-1")
        b2 = await mgr.create_bead("cv-1", "ws-1", depends_on=[b1.bead_id])
        await mgr.complete_bead(b1.bead_id)
        ready = await mgr.get_ready_beads("cv-1")
        assert len(ready) == 1
        assert ready[0].bead_id == b2.bead_id


# =============================================================================
# Convoy Tests
# =============================================================================


class TestConvoy:
    """Test Convoy dataclass."""

    def test_convoy_creation(self):
        convoy = Convoy(convoy_id="cv-abc", workspace_id="ws-1", rig_id="rig-1")
        assert convoy.status == ConvoyStatus.CREATED
        assert convoy.total_beads == 0

    def test_convoy_with_beads(self):
        convoy = Convoy(
            convoy_id="cv-abc",
            workspace_id="ws-1",
            rig_id="rig-1",
            bead_ids=["bd-1", "bd-2", "bd-3"],
        )
        assert convoy.total_beads == 3

    def test_convoy_status_values(self):
        assert ConvoyStatus.CREATED.value == "created"
        assert ConvoyStatus.ASSIGNING.value == "assigning"
        assert ConvoyStatus.EXECUTING.value == "executing"
        assert ConvoyStatus.MERGING.value == "merging"
        assert ConvoyStatus.DONE.value == "done"
        assert ConvoyStatus.FAILED.value == "failed"
        assert ConvoyStatus.CANCELLED.value == "cancelled"


class TestConvoyTracker:
    """Test ConvoyTracker."""

    @pytest.fixture
    def tracker(self, tmp_path):
        bead_store = NomicBeadStore(tmp_path / "beads", git_enabled=False, auto_commit=False)
        return ConvoyTracker(bead_store=bead_store, use_nomic_store=True)

    @pytest.mark.asyncio
    async def test_create_convoy(self, tracker):
        convoy = await tracker.create_convoy("ws-1", "rig-1", name="Build sprint")
        assert convoy.name == "Build sprint"
        assert convoy.status == ConvoyStatus.CREATED

    @pytest.mark.asyncio
    async def test_get_convoy(self, tracker):
        c = await tracker.create_convoy("ws-1", "rig-1", convoy_id="cv-get")
        result = await tracker.get_convoy("cv-get")
        assert result is not None
        assert result.convoy_id == "cv-get"

    @pytest.mark.asyncio
    async def test_convoy_state_machine(self, tracker):
        c = await tracker.create_convoy("ws-1", "rig-1", convoy_id="cv-sm")
        assert c.status == ConvoyStatus.CREATED

        await tracker.start_assigning("cv-sm")
        c = await tracker.get_convoy("cv-sm")
        assert c.status == ConvoyStatus.ASSIGNING

        await tracker.start_executing("cv-sm", ["agent-1"])
        c = await tracker.get_convoy("cv-sm")
        assert c.status == ConvoyStatus.EXECUTING
        assert c.assigned_agents == ["agent-1"]

        await tracker.start_merging("cv-sm")
        c = await tracker.get_convoy("cv-sm")
        assert c.status == ConvoyStatus.MERGING

        await tracker.complete_convoy("cv-sm", {"merged": True})
        c = await tracker.get_convoy("cv-sm")
        assert c.status == ConvoyStatus.DONE
        assert c.merge_result == {"merged": True}

    @pytest.mark.asyncio
    async def test_fail_convoy(self, tracker):
        await tracker.create_convoy("ws-1", "rig-1", convoy_id="cv-fail")
        await tracker.fail_convoy("cv-fail", "build error")
        c = await tracker.get_convoy("cv-fail")
        assert c.status == ConvoyStatus.FAILED
        assert c.error == "build error"

    @pytest.mark.asyncio
    async def test_cancel_convoy(self, tracker):
        await tracker.create_convoy("ws-1", "rig-1", convoy_id="cv-cancel")
        await tracker.cancel_convoy("cv-cancel")
        c = await tracker.get_convoy("cv-cancel")
        assert c.status == ConvoyStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_list_convoys_by_rig(self, tracker):
        await tracker.create_convoy("ws-1", "rig-1")
        await tracker.create_convoy("ws-1", "rig-2")
        await tracker.create_convoy("ws-1", "rig-1")
        convoys = await tracker.list_convoys(rig_id="rig-1")
        assert len(convoys) == 2

    @pytest.mark.asyncio
    async def test_convoy_stats(self, tracker):
        await tracker.create_convoy("ws-1", "rig-1")
        await tracker.create_convoy("ws-1", "rig-1", convoy_id="cv-s2")
        await tracker.complete_convoy("cv-s2")
        stats = await tracker.get_stats()
        assert stats["total_convoys"] == 2
        assert stats["by_status"]["created"] == 1
        assert stats["by_status"]["done"] == 1


# =============================================================================
# WorkspaceManager Integration Tests
# =============================================================================


class TestWorkspaceManager:
    """Test WorkspaceManager integration."""

    @pytest.fixture
    def ws(self, tmp_path):
        return WorkspaceManager(workspace_root=str(tmp_path), workspace_id="test-ws")

    @pytest.mark.asyncio
    async def test_create_rig(self, ws):
        rig = await ws.create_rig("my-project")
        assert rig.name == "my-project"
        assert rig.status == RigStatus.READY
        assert rig.rig_id.startswith("rig-")

    @pytest.mark.asyncio
    async def test_get_rig(self, ws):
        rig = await ws.create_rig("test", rig_id="rig-get")
        result = await ws.get_rig("rig-get")
        assert result is not None
        assert result.name == "test"

    @pytest.mark.asyncio
    async def test_list_rigs(self, ws):
        await ws.create_rig("r1")
        await ws.create_rig("r2")
        rigs = await ws.list_rigs()
        assert len(rigs) == 2

    @pytest.mark.asyncio
    async def test_assign_agent_to_rig(self, ws):
        rig = await ws.create_rig("test", rig_id="rig-assign")
        result = await ws.assign_agent_to_rig("rig-assign", "agent-1")
        assert "agent-1" in result.assigned_agents

    @pytest.mark.asyncio
    async def test_assign_agent_max_capacity(self, ws):
        config = RigConfig(max_agents=1)
        await ws.create_rig("test", config=config, rig_id="rig-max")
        await ws.assign_agent_to_rig("rig-max", "agent-1")
        with pytest.raises(ValueError, match="max capacity"):
            await ws.assign_agent_to_rig("rig-max", "agent-2")

    @pytest.mark.asyncio
    async def test_remove_agent_from_rig(self, ws):
        await ws.create_rig("test", rig_id="rig-rm")
        await ws.assign_agent_to_rig("rig-rm", "agent-1")
        result = await ws.remove_agent_from_rig("rig-rm", "agent-1")
        assert "agent-1" not in result.assigned_agents

    @pytest.mark.asyncio
    async def test_stop_rig(self, ws):
        await ws.create_rig("test", rig_id="rig-stop")
        result = await ws.stop_rig("rig-stop")
        assert result.status == RigStatus.DRAINING

    @pytest.mark.asyncio
    async def test_delete_rig(self, ws):
        await ws.create_rig("test", rig_id="rig-del")
        assert await ws.delete_rig("rig-del") is True
        assert await ws.get_rig("rig-del") is None

    @pytest.mark.asyncio
    async def test_create_convoy_with_beads(self, ws):
        await ws.create_rig("test", rig_id="rig-conv")
        convoy = await ws.create_convoy(
            rig_id="rig-conv",
            name="Build Sprint",
            bead_specs=[
                {"title": "Build frontend", "payload": {"cmd": "npm build"}},
                {"title": "Build backend", "payload": {"cmd": "make build"}},
            ],
        )
        assert convoy.name == "Build Sprint"
        assert len(convoy.bead_ids) == 2

    @pytest.mark.asyncio
    async def test_get_convoy_status(self, ws):
        await ws.create_rig("test", rig_id="rig-status")
        convoy = await ws.create_convoy(
            rig_id="rig-status",
            bead_specs=[{"title": "Task 1"}, {"title": "Task 2"}],
        )
        status = await ws.get_convoy_status(convoy.convoy_id)
        assert status is not None
        assert status["total_beads"] == 2
        assert status["pending"] == 2
        assert status["progress_percent"] == 0.0

    @pytest.mark.asyncio
    async def test_complete_bead_updates_progress(self, ws):
        await ws.create_rig("test", rig_id="rig-prog")
        convoy = await ws.create_convoy(
            rig_id="rig-prog",
            bead_specs=[{"title": "T1"}, {"title": "T2"}],
        )
        await ws.complete_bead(convoy.bead_ids[0], {"ok": True})
        status = await ws.get_convoy_status(convoy.convoy_id)
        assert status["done"] == 1
        assert status["progress_percent"] == 50.0

    @pytest.mark.asyncio
    async def test_workspace_stats(self, ws):
        await ws.create_rig("r1")
        await ws.create_rig("r2")
        stats = await ws.get_stats()
        assert stats["workspace_id"] == "test-ws"
        assert stats["total_rigs"] == 2
