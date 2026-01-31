"""
Tests for Gastown data models.

Tests ConvoyStatus mappings, dataclass structures, and LedgerEntry functionality.
"""

from __future__ import annotations

import pytest
from datetime import datetime

from aragora.extensions.gastown.models import (
    ConvoyStatus,
    ConvoyArtifact,
    Convoy,
    HookType,
    Hook,
    LedgerEntry,
    Rig,
    RigConfig,
    Workspace,
    WorkspaceConfig,
)


class TestConvoyStatus:
    """Tests for ConvoyStatus enum and mappings."""

    def test_all_statuses_exist(self):
        """All expected statuses are defined."""
        assert ConvoyStatus.PENDING
        assert ConvoyStatus.IN_PROGRESS
        assert ConvoyStatus.BLOCKED
        assert ConvoyStatus.REVIEW
        assert ConvoyStatus.COMPLETED
        assert ConvoyStatus.CANCELLED

    def test_status_values(self):
        """Status values are snake_case strings."""
        assert ConvoyStatus.PENDING.value == "pending"
        assert ConvoyStatus.IN_PROGRESS.value == "in_progress"
        assert ConvoyStatus.BLOCKED.value == "blocked"
        assert ConvoyStatus.REVIEW.value == "review"
        assert ConvoyStatus.COMPLETED.value == "completed"
        assert ConvoyStatus.CANCELLED.value == "cancelled"

    def test_to_nomic_mapping(self):
        """All statuses map to valid nomic statuses."""
        from aragora.nomic.convoys import ConvoyStatus as NomicConvoyStatus

        for status in ConvoyStatus:
            nomic_status = status.to_nomic()
            assert isinstance(nomic_status, NomicConvoyStatus)

    def test_from_nomic_mapping(self):
        """All nomic statuses map back to gastown statuses."""
        from aragora.nomic.convoys import ConvoyStatus as NomicConvoyStatus

        for nomic_status in NomicConvoyStatus:
            gas_status = ConvoyStatus.from_nomic(nomic_status)
            assert isinstance(gas_status, ConvoyStatus)

    def test_from_nomic_with_gastown_status_override(self):
        """Gastown status metadata overrides nomic status mapping."""
        from aragora.nomic.convoys import ConvoyStatus as NomicConvoyStatus

        # ACTIVE maps to IN_PROGRESS by default
        result = ConvoyStatus.from_nomic(NomicConvoyStatus.ACTIVE)
        assert result == ConvoyStatus.IN_PROGRESS

        # But with gastown_status override, uses that
        result = ConvoyStatus.from_nomic(NomicConvoyStatus.ACTIVE, "review")
        assert result == ConvoyStatus.REVIEW

        result = ConvoyStatus.from_nomic(NomicConvoyStatus.ACTIVE, "blocked")
        assert result == ConvoyStatus.BLOCKED

    def test_from_nomic_ignores_invalid_gastown_status(self):
        """Invalid gastown_status is ignored, falls back to mapping."""
        from aragora.nomic.convoys import ConvoyStatus as NomicConvoyStatus

        result = ConvoyStatus.from_nomic(NomicConvoyStatus.COMPLETED, "invalid_status")
        assert result == ConvoyStatus.COMPLETED

    def test_to_workspace_status_mapping(self):
        """All statuses map to valid workspace status values."""
        expected = {
            ConvoyStatus.PENDING: "created",
            ConvoyStatus.IN_PROGRESS: "executing",
            ConvoyStatus.BLOCKED: "failed",
            ConvoyStatus.REVIEW: "merging",
            ConvoyStatus.COMPLETED: "done",
            ConvoyStatus.CANCELLED: "cancelled",
        }
        for status, ws_value in expected.items():
            assert status.to_workspace_status() == ws_value

    def test_from_workspace_status_mapping(self):
        """Workspace status values map to gastown statuses."""
        test_cases = [
            ("created", ConvoyStatus.PENDING),
            ("assigning", ConvoyStatus.PENDING),
            ("executing", ConvoyStatus.IN_PROGRESS),
            ("merging", ConvoyStatus.REVIEW),
            ("done", ConvoyStatus.COMPLETED),
            ("failed", ConvoyStatus.BLOCKED),
            ("cancelled", ConvoyStatus.CANCELLED),
        ]
        for ws_value, expected in test_cases:
            assert ConvoyStatus.from_workspace_status(ws_value) == expected

    def test_from_workspace_status_unknown_defaults_pending(self):
        """Unknown workspace status values default to PENDING."""
        assert ConvoyStatus.from_workspace_status("unknown") == ConvoyStatus.PENDING
        assert ConvoyStatus.from_workspace_status("") == ConvoyStatus.PENDING


class TestHookType:
    """Tests for HookType enum."""

    def test_all_hook_types_exist(self):
        """All expected git hook types are defined."""
        assert HookType.PRE_COMMIT
        assert HookType.POST_COMMIT
        assert HookType.PRE_PUSH
        assert HookType.POST_CHECKOUT
        assert HookType.POST_MERGE
        assert HookType.PREPARE_COMMIT_MSG

    def test_hook_type_values(self):
        """Hook type values match git hook names."""
        assert HookType.PRE_COMMIT.value == "pre-commit"
        assert HookType.POST_COMMIT.value == "post-commit"
        assert HookType.PRE_PUSH.value == "pre-push"
        assert HookType.POST_CHECKOUT.value == "post-checkout"
        assert HookType.POST_MERGE.value == "post-merge"
        assert HookType.PREPARE_COMMIT_MSG.value == "prepare-commit-msg"


class TestWorkspaceConfig:
    """Tests for WorkspaceConfig dataclass."""

    def test_required_fields(self):
        """WorkspaceConfig requires name and root_path."""
        config = WorkspaceConfig(name="test", root_path="/path/to/workspace")
        assert config.name == "test"
        assert config.root_path == "/path/to/workspace"

    def test_default_values(self):
        """WorkspaceConfig has sensible defaults."""
        config = WorkspaceConfig(name="test", root_path="/path")
        assert config.description == ""
        assert config.max_rigs == 10
        assert config.max_agents_per_rig == 5
        assert config.default_model == "claude-3-opus"
        assert config.metadata == {}

    def test_custom_values(self):
        """WorkspaceConfig accepts custom values."""
        config = WorkspaceConfig(
            name="project",
            root_path="/root",
            description="A project",
            max_rigs=5,
            max_agents_per_rig=3,
            default_model="gpt-4",
            metadata={"key": "value"},
        )
        assert config.description == "A project"
        assert config.max_rigs == 5
        assert config.max_agents_per_rig == 3
        assert config.default_model == "gpt-4"
        assert config.metadata == {"key": "value"}


class TestWorkspace:
    """Tests for Workspace dataclass."""

    def test_required_fields(self):
        """Workspace requires id and config."""
        config = WorkspaceConfig(name="test", root_path="/path")
        workspace = Workspace(id="ws-1", config=config)
        assert workspace.id == "ws-1"
        assert workspace.config.name == "test"

    def test_default_values(self):
        """Workspace has sensible defaults."""
        config = WorkspaceConfig(name="test", root_path="/path")
        workspace = Workspace(id="ws-1", config=config)
        assert workspace.owner_id == ""
        assert workspace.tenant_id is None
        assert workspace.status == "active"
        assert workspace.rigs == []
        assert workspace.active_convoys == 0
        assert isinstance(workspace.created_at, datetime)
        assert isinstance(workspace.updated_at, datetime)

    def test_custom_values(self):
        """Workspace accepts custom values."""
        config = WorkspaceConfig(name="test", root_path="/path")
        workspace = Workspace(
            id="ws-1",
            config=config,
            owner_id="user-1",
            tenant_id="tenant-1",
            status="suspended",
            rigs=["rig-1", "rig-2"],
            active_convoys=5,
        )
        assert workspace.owner_id == "user-1"
        assert workspace.tenant_id == "tenant-1"
        assert workspace.status == "suspended"
        assert workspace.rigs == ["rig-1", "rig-2"]
        assert workspace.active_convoys == 5


class TestRigConfig:
    """Tests for RigConfig dataclass."""

    def test_required_fields(self):
        """RigConfig requires name and repo_path."""
        config = RigConfig(name="rig", repo_path="/repo")
        assert config.name == "rig"
        assert config.repo_path == "/repo"

    def test_default_values(self):
        """RigConfig has sensible defaults."""
        config = RigConfig(name="rig", repo_path="/repo")
        assert config.branch == "main"
        assert config.worktree_path is None
        assert config.description == ""
        assert config.max_agents == 5
        assert config.tools == []
        assert config.metadata == {}


class TestRig:
    """Tests for Rig dataclass."""

    def test_required_fields(self):
        """Rig requires id, workspace_id, and config."""
        config = RigConfig(name="rig", repo_path="/repo")
        rig = Rig(id="rig-1", workspace_id="ws-1", config=config)
        assert rig.id == "rig-1"
        assert rig.workspace_id == "ws-1"
        assert rig.config.name == "rig"

    def test_default_values(self):
        """Rig has sensible defaults."""
        config = RigConfig(name="rig", repo_path="/repo")
        rig = Rig(id="rig-1", workspace_id="ws-1", config=config)
        assert rig.status == "active"
        assert rig.agents == []
        assert rig.active_convoys == []
        assert rig.last_sync is None
        assert isinstance(rig.created_at, datetime)
        assert isinstance(rig.updated_at, datetime)


class TestConvoyArtifact:
    """Tests for ConvoyArtifact dataclass."""

    def test_required_fields(self):
        """ConvoyArtifact requires id, convoy_id, type, and path."""
        artifact = ConvoyArtifact(
            id="art-1",
            convoy_id="conv-1",
            type="diff",
            path="/path/to/artifact",
        )
        assert artifact.id == "art-1"
        assert artifact.convoy_id == "conv-1"
        assert artifact.type == "diff"
        assert artifact.path == "/path/to/artifact"

    def test_default_values(self):
        """ConvoyArtifact has sensible defaults."""
        artifact = ConvoyArtifact(
            id="art-1",
            convoy_id="conv-1",
            type="file",
            path="/path",
        )
        assert artifact.content_hash == ""
        assert artifact.size_bytes == 0
        assert artifact.metadata == {}
        assert isinstance(artifact.created_at, datetime)


class TestConvoy:
    """Tests for Convoy dataclass."""

    def test_required_fields(self):
        """Convoy requires id, rig_id, and title."""
        convoy = Convoy(id="conv-1", rig_id="rig-1", title="Fix bug")
        assert convoy.id == "conv-1"
        assert convoy.rig_id == "rig-1"
        assert convoy.title == "Fix bug"

    def test_default_values(self):
        """Convoy has sensible defaults."""
        convoy = Convoy(id="conv-1", rig_id="rig-1", title="Task")
        assert convoy.description == ""
        assert convoy.status == ConvoyStatus.PENDING
        assert convoy.issue_ref is None
        assert convoy.parent_convoy is None
        assert convoy.depends_on == []
        assert convoy.assigned_agents == []
        assert convoy.current_agent is None
        assert convoy.handoff_count == 0
        assert convoy.artifacts == []
        assert convoy.result == {}
        assert convoy.error is None
        assert convoy.priority == 0
        assert convoy.tags == []
        assert convoy.metadata == {}
        assert convoy.started_at is None
        assert convoy.completed_at is None

    def test_convoy_protocol_properties(self):
        """Convoy implements ConvoyRecord protocol properties."""
        convoy = Convoy(
            id="conv-1",
            rig_id="rig-1",
            title="Test Convoy",
            description="A test convoy",
            status=ConvoyStatus.IN_PROGRESS,
            assigned_agents=["agent-1", "agent-2"],
            error="Test error",
            metadata={"key": "value"},
        )

        assert convoy.convoy_id == "conv-1"
        assert convoy.convoy_title == "Test Convoy"
        assert convoy.convoy_description == "A test convoy"
        assert convoy.convoy_bead_ids == []  # Gastown doesn't track beads
        assert convoy.convoy_status_value == "in_progress"
        assert convoy.convoy_created_at == convoy.created_at
        assert convoy.convoy_updated_at == convoy.updated_at
        assert convoy.convoy_assigned_agents == ["agent-1", "agent-2"]
        assert convoy.convoy_error == "Test error"
        assert convoy.convoy_metadata == {"key": "value"}


class TestHook:
    """Tests for Hook dataclass."""

    def test_required_fields(self):
        """Hook requires id, rig_id, type, and path."""
        hook = Hook(
            id="hook-1",
            rig_id="rig-1",
            type=HookType.PRE_COMMIT,
            path="/path/to/hook",
        )
        assert hook.id == "hook-1"
        assert hook.rig_id == "rig-1"
        assert hook.type == HookType.PRE_COMMIT
        assert hook.path == "/path/to/hook"

    def test_default_values(self):
        """Hook has sensible defaults."""
        hook = Hook(
            id="hook-1",
            rig_id="rig-1",
            type=HookType.POST_COMMIT,
            path="/path",
        )
        assert hook.content == ""
        assert hook.enabled is True
        assert hook.last_triggered is None
        assert hook.trigger_count == 0
        assert hook.metadata == {}
        assert isinstance(hook.created_at, datetime)
        assert isinstance(hook.updated_at, datetime)


class TestLedgerEntry:
    """Tests for LedgerEntry dataclass."""

    def test_required_fields(self):
        """LedgerEntry requires id, workspace_id, type, and title."""
        entry = LedgerEntry(
            id="entry-1",
            workspace_id="ws-1",
            type="issue",
            title="Bug report",
        )
        assert entry.id == "entry-1"
        assert entry.workspace_id == "ws-1"
        assert entry.type == "issue"
        assert entry.title == "Bug report"

    def test_default_values(self):
        """LedgerEntry has sensible defaults."""
        entry = LedgerEntry(
            id="entry-1",
            workspace_id="ws-1",
            type="task",
            title="Task",
        )
        assert entry.body == ""
        assert entry.status == "open"
        assert entry.resolved_at is None
        assert entry.convoy_id is None
        assert entry.parent_id is None
        assert entry.related_entries == []
        assert entry.created_by == ""
        assert entry.assigned_to == []
        assert entry.labels == []
        assert entry.priority == 0
        assert entry.metadata == {}
        assert isinstance(entry.created_at, datetime)
        assert isinstance(entry.updated_at, datetime)

    def test_entry_types(self):
        """LedgerEntry supports all documented types."""
        for entry_type in ["issue", "task", "decision", "note"]:
            entry = LedgerEntry(
                id="entry-1",
                workspace_id="ws-1",
                type=entry_type,
                title="Test",
            )
            assert entry.type == entry_type

    def test_entry_statuses(self):
        """LedgerEntry supports all documented statuses."""
        for status in ["open", "in_progress", "resolved", "closed"]:
            entry = LedgerEntry(
                id="entry-1",
                workspace_id="ws-1",
                type="issue",
                title="Test",
                status=status,
            )
            assert entry.status == status
