"""
Tests for Control Plane Policy Store.

Tests cover:
- ControlPlanePolicyStore (SQLite backend):
  - CRUD operations for policies
  - Policy filtering and listing
  - Toggle enable/disable
  - Violation recording and querying
  - Violation status updates
  - Violation counting by type
- Factory function (get_control_plane_policy_store)
- Reset singleton (reset_control_plane_policy_store)
- Default DB path helper

Run with:
    pytest tests/control_plane/test_policy_store.py -v
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from aragora.control_plane.policy.types import (
    ControlPlanePolicy,
    EnforcementLevel,
    PolicyScope,
    PolicyViolation,
    RegionConstraint,
    SLARequirements,
)
from aragora.control_plane.policy_store import (
    ControlPlanePolicyStore,
    _get_default_db_path,
    get_control_plane_policy_store,
    reset_control_plane_policy_store,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the policy store singleton before each test."""
    reset_control_plane_policy_store()
    yield
    reset_control_plane_policy_store()


@pytest.fixture
def store(tmp_path: Path) -> ControlPlanePolicyStore:
    """Create a fresh policy store with a temp database."""
    db_path = tmp_path / "test_policies.db"
    return ControlPlanePolicyStore(db_path=db_path)


@pytest.fixture
def sample_policy() -> ControlPlanePolicy:
    """Create a sample control plane policy."""
    return ControlPlanePolicy(
        id="policy_test001",
        name="Test Policy",
        description="A test policy",
        scope=PolicyScope.GLOBAL,
        task_types=["debate", "analysis"],
        capabilities=["reasoning"],
        workspaces=["ws-1"],
        agent_allowlist=["claude-3-opus", "gpt-4"],
        agent_blocklist=["gpt-3.5-turbo"],
        enforcement_level=EnforcementLevel.HARD,
        enabled=True,
        priority=10,
        created_by="admin",
        metadata={"env": "test"},
    )


@pytest.fixture
def sample_policy_with_constraints() -> ControlPlanePolicy:
    """Create a policy with region constraints and SLA."""
    return ControlPlanePolicy(
        id="policy_test002",
        name="Constrained Policy",
        description="Policy with region and SLA constraints",
        scope=PolicyScope.REGION,
        region_constraint=RegionConstraint(
            allowed_regions=["us-east-1", "us-west-2"],
            blocked_regions=["cn-north-1"],
            require_data_residency=True,
        ),
        sla=SLARequirements(
            max_execution_seconds=120.0,
            max_queue_seconds=30.0,
            min_agents_available=2,
        ),
        enforcement_level=EnforcementLevel.SOFT,
        enabled=True,
        priority=20,
    )


@pytest.fixture
def sample_violation() -> PolicyViolation:
    """Create a sample policy violation."""
    return PolicyViolation(
        id=f"violation_{uuid.uuid4().hex[:12]}",
        policy_id="policy_test001",
        policy_name="Test Policy",
        violation_type="agent",
        description="Agent gpt-3.5-turbo not in allowlist",
        task_id="task-123",
        task_type="debate",
        agent_id="gpt-3.5-turbo",
        region="us-east-1",
        workspace_id="ws-1",
        enforcement_level=EnforcementLevel.HARD,
        metadata={"attempted_by": "user-1"},
    )


# =============================================================================
# Default DB Path Tests
# =============================================================================


class TestDefaultDbPath:
    """Tests for _get_default_db_path helper."""

    def test_returns_path_object(self):
        """Test that default path returns a Path object."""
        path = _get_default_db_path()
        assert isinstance(path, Path)

    def test_path_contains_policies_db(self):
        """Test that the path contains the expected filename."""
        path = _get_default_db_path()
        assert path.name == "policies.db"


# =============================================================================
# Store Initialization Tests
# =============================================================================


class TestStoreInitialization:
    """Tests for ControlPlanePolicyStore initialization."""

    def test_creates_database(self, tmp_path: Path):
        """Test that initializing the store creates the database file."""
        db_path = tmp_path / "init_test.db"
        store = ControlPlanePolicyStore(db_path=db_path)
        assert db_path.exists()

    def test_schema_name(self, store: ControlPlanePolicyStore):
        """Test the schema name constant."""
        assert ControlPlanePolicyStore.SCHEMA_NAME == "control_plane_policy_store"

    def test_schema_version(self, store: ControlPlanePolicyStore):
        """Test the schema version constant."""
        assert ControlPlanePolicyStore.SCHEMA_VERSION == 1


# =============================================================================
# Policy CRUD Tests
# =============================================================================


class TestCreatePolicy:
    """Tests for creating policies."""

    def test_create_basic_policy(
        self, store: ControlPlanePolicyStore, sample_policy: ControlPlanePolicy
    ):
        """Test creating a basic policy."""
        result = store.create_policy(sample_policy)
        assert result.id == sample_policy.id
        assert result.name == "Test Policy"

    def test_create_and_retrieve(
        self, store: ControlPlanePolicyStore, sample_policy: ControlPlanePolicy
    ):
        """Test creating then retrieving a policy."""
        store.create_policy(sample_policy)
        retrieved = store.get_policy(sample_policy.id)

        assert retrieved is not None
        assert retrieved.id == sample_policy.id
        assert retrieved.name == "Test Policy"
        assert retrieved.description == "A test policy"
        assert retrieved.scope == PolicyScope.GLOBAL
        assert retrieved.enforcement_level == EnforcementLevel.HARD
        assert retrieved.enabled is True
        assert retrieved.priority == 10

    def test_create_policy_preserves_lists(
        self, store: ControlPlanePolicyStore, sample_policy: ControlPlanePolicy
    ):
        """Test that list fields are preserved through store round-trip."""
        store.create_policy(sample_policy)
        retrieved = store.get_policy(sample_policy.id)

        assert retrieved.task_types == ["debate", "analysis"]
        assert retrieved.capabilities == ["reasoning"]
        assert retrieved.workspaces == ["ws-1"]
        assert retrieved.agent_allowlist == ["claude-3-opus", "gpt-4"]
        assert retrieved.agent_blocklist == ["gpt-3.5-turbo"]

    def test_create_policy_with_constraints(
        self,
        store: ControlPlanePolicyStore,
        sample_policy_with_constraints: ControlPlanePolicy,
    ):
        """Test creating a policy with region constraints and SLA."""
        store.create_policy(sample_policy_with_constraints)
        retrieved = store.get_policy(sample_policy_with_constraints.id)

        assert retrieved is not None
        assert retrieved.region_constraint is not None
        assert retrieved.region_constraint.allowed_regions == ["us-east-1", "us-west-2"]
        assert retrieved.region_constraint.blocked_regions == ["cn-north-1"]
        assert retrieved.region_constraint.require_data_residency is True

        assert retrieved.sla is not None
        assert retrieved.sla.max_execution_seconds == 120.0
        assert retrieved.sla.max_queue_seconds == 30.0
        assert retrieved.sla.min_agents_available == 2

    def test_create_policy_preserves_metadata(
        self, store: ControlPlanePolicyStore, sample_policy: ControlPlanePolicy
    ):
        """Test that metadata is preserved."""
        store.create_policy(sample_policy)
        retrieved = store.get_policy(sample_policy.id)
        assert retrieved.metadata == {"env": "test"}

    def test_create_policy_preserves_created_by(
        self, store: ControlPlanePolicyStore, sample_policy: ControlPlanePolicy
    ):
        """Test that created_by is preserved."""
        store.create_policy(sample_policy)
        retrieved = store.get_policy(sample_policy.id)
        assert retrieved.created_by == "admin"


class TestGetPolicy:
    """Tests for getting policies."""

    def test_get_nonexistent_policy(self, store: ControlPlanePolicyStore):
        """Test getting a policy that doesn't exist."""
        result = store.get_policy("nonexistent_id")
        assert result is None

    def test_get_existing_policy(
        self, store: ControlPlanePolicyStore, sample_policy: ControlPlanePolicy
    ):
        """Test getting an existing policy."""
        store.create_policy(sample_policy)
        result = store.get_policy(sample_policy.id)
        assert result is not None
        assert result.id == sample_policy.id


class TestListPolicies:
    """Tests for listing policies."""

    def test_list_empty_store(self, store: ControlPlanePolicyStore):
        """Test listing policies from empty store."""
        result = store.list_policies()
        assert result == []

    def test_list_all_policies(self, store: ControlPlanePolicyStore):
        """Test listing all policies."""
        for i in range(3):
            policy = ControlPlanePolicy(
                id=f"policy_list_{i}",
                name=f"Policy {i}",
                priority=i,
            )
            store.create_policy(policy)

        result = store.list_policies()
        assert len(result) == 3

    def test_list_enabled_only(self, store: ControlPlanePolicyStore):
        """Test filtering by enabled status."""
        store.create_policy(ControlPlanePolicy(id="p1", name="Enabled", enabled=True))
        store.create_policy(ControlPlanePolicy(id="p2", name="Disabled", enabled=False))

        all_policies = store.list_policies(enabled_only=False)
        enabled_policies = store.list_policies(enabled_only=True)

        assert len(all_policies) == 2
        assert len(enabled_policies) == 1
        assert enabled_policies[0].name == "Enabled"

    def test_list_with_workspace_filter(self, store: ControlPlanePolicyStore):
        """Test filtering by workspace."""
        store.create_policy(ControlPlanePolicy(id="p1", name="WS1", workspaces=["ws-1"]))
        store.create_policy(ControlPlanePolicy(id="p2", name="WS2", workspaces=["ws-2"]))
        store.create_policy(ControlPlanePolicy(id="p3", name="Global", workspaces=[]))

        ws1_policies = store.list_policies(workspace="ws-1")
        # Should include ws-1 specific AND global (empty workspaces)
        assert len(ws1_policies) == 2
        names = {p.name for p in ws1_policies}
        assert "WS1" in names
        assert "Global" in names

    def test_list_with_limit(self, store: ControlPlanePolicyStore):
        """Test pagination with limit."""
        for i in range(5):
            store.create_policy(ControlPlanePolicy(id=f"p_lim_{i}", name=f"Policy {i}"))

        result = store.list_policies(limit=2)
        assert len(result) == 2

    def test_list_with_offset(self, store: ControlPlanePolicyStore):
        """Test pagination with offset."""
        for i in range(5):
            store.create_policy(ControlPlanePolicy(id=f"p_off_{i}", name=f"Policy {i}", priority=i))

        result = store.list_policies(limit=2, offset=2)
        assert len(result) == 2

    def test_list_ordered_by_priority(self, store: ControlPlanePolicyStore):
        """Test that results are ordered by priority descending."""
        store.create_policy(ControlPlanePolicy(id="p_low", name="Low", priority=1))
        store.create_policy(ControlPlanePolicy(id="p_high", name="High", priority=100))
        store.create_policy(ControlPlanePolicy(id="p_mid", name="Mid", priority=50))

        result = store.list_policies()
        assert result[0].name == "High"
        assert result[1].name == "Mid"
        assert result[2].name == "Low"


class TestUpdatePolicy:
    """Tests for updating policies."""

    def test_update_name(self, store: ControlPlanePolicyStore, sample_policy: ControlPlanePolicy):
        """Test updating a policy's name."""
        store.create_policy(sample_policy)
        updated = store.update_policy(sample_policy.id, {"name": "Updated Name"})

        assert updated is not None
        assert updated.name == "Updated Name"

    def test_update_description(
        self, store: ControlPlanePolicyStore, sample_policy: ControlPlanePolicy
    ):
        """Test updating a policy's description."""
        store.create_policy(sample_policy)
        updated = store.update_policy(sample_policy.id, {"description": "New description"})

        assert updated is not None
        assert updated.description == "New description"

    def test_update_enabled(
        self, store: ControlPlanePolicyStore, sample_policy: ControlPlanePolicy
    ):
        """Test toggling enabled flag via update."""
        store.create_policy(sample_policy)
        updated = store.update_policy(sample_policy.id, {"enabled": False})

        assert updated is not None
        assert updated.enabled is False

    def test_update_priority(
        self, store: ControlPlanePolicyStore, sample_policy: ControlPlanePolicy
    ):
        """Test updating priority."""
        store.create_policy(sample_policy)
        updated = store.update_policy(sample_policy.id, {"priority": 99})

        assert updated is not None
        assert updated.priority == 99

    def test_update_enforcement_level_enum(
        self, store: ControlPlanePolicyStore, sample_policy: ControlPlanePolicy
    ):
        """Test updating enforcement level with enum value."""
        store.create_policy(sample_policy)
        updated = store.update_policy(
            sample_policy.id, {"enforcement_level": EnforcementLevel.WARN}
        )

        assert updated is not None
        assert updated.enforcement_level == EnforcementLevel.WARN

    def test_update_enforcement_level_string(
        self, store: ControlPlanePolicyStore, sample_policy: ControlPlanePolicy
    ):
        """Test updating enforcement level with string value."""
        store.create_policy(sample_policy)
        updated = store.update_policy(sample_policy.id, {"enforcement_level": "soft"})

        assert updated is not None
        assert updated.enforcement_level == EnforcementLevel.SOFT

    def test_update_json_fields(
        self, store: ControlPlanePolicyStore, sample_policy: ControlPlanePolicy
    ):
        """Test updating JSON list fields."""
        store.create_policy(sample_policy)
        updated = store.update_policy(
            sample_policy.id,
            {
                "task_types": ["new_type"],
                "agent_allowlist": ["claude-3-haiku"],
                "metadata": {"updated": True},
            },
        )

        assert updated is not None
        assert updated.task_types == ["new_type"]
        assert updated.agent_allowlist == ["claude-3-haiku"]
        assert updated.metadata == {"updated": True}

    def test_update_region_constraint(
        self, store: ControlPlanePolicyStore, sample_policy: ControlPlanePolicy
    ):
        """Test updating region constraint."""
        store.create_policy(sample_policy)
        new_rc = RegionConstraint(
            allowed_regions=["eu-west-1"],
            require_data_residency=True,
        )
        updated = store.update_policy(sample_policy.id, {"region_constraint": new_rc})

        assert updated is not None
        assert updated.region_constraint is not None
        assert updated.region_constraint.allowed_regions == ["eu-west-1"]

    def test_update_sla(self, store: ControlPlanePolicyStore, sample_policy: ControlPlanePolicy):
        """Test updating SLA requirements."""
        store.create_policy(sample_policy)
        new_sla = SLARequirements(max_execution_seconds=60.0)
        updated = store.update_policy(sample_policy.id, {"sla": new_sla})

        assert updated is not None
        assert updated.sla is not None
        assert updated.sla.max_execution_seconds == 60.0

    def test_update_nonexistent_policy(self, store: ControlPlanePolicyStore):
        """Test updating a nonexistent policy returns None."""
        result = store.update_policy("nonexistent", {"name": "Foo"})
        assert result is None

    def test_update_empty_updates(
        self, store: ControlPlanePolicyStore, sample_policy: ControlPlanePolicy
    ):
        """Test updating with empty dict returns original policy."""
        store.create_policy(sample_policy)
        result = store.update_policy(sample_policy.id, {})
        assert result is not None
        assert result.name == "Test Policy"

    def test_update_multiple_fields(
        self, store: ControlPlanePolicyStore, sample_policy: ControlPlanePolicy
    ):
        """Test updating multiple fields at once."""
        store.create_policy(sample_policy)
        updated = store.update_policy(
            sample_policy.id,
            {
                "name": "New Name",
                "priority": 50,
                "enabled": False,
            },
        )

        assert updated is not None
        assert updated.name == "New Name"
        assert updated.priority == 50
        assert updated.enabled is False


class TestDeletePolicy:
    """Tests for deleting policies."""

    def test_delete_existing_policy(
        self, store: ControlPlanePolicyStore, sample_policy: ControlPlanePolicy
    ):
        """Test deleting an existing policy."""
        store.create_policy(sample_policy)
        result = store.delete_policy(sample_policy.id)

        assert result is True
        assert store.get_policy(sample_policy.id) is None

    def test_delete_nonexistent_policy(self, store: ControlPlanePolicyStore):
        """Test deleting a nonexistent policy."""
        result = store.delete_policy("nonexistent")
        assert result is False


class TestTogglePolicy:
    """Tests for toggling policy enabled status."""

    def test_disable_policy(
        self, store: ControlPlanePolicyStore, sample_policy: ControlPlanePolicy
    ):
        """Test disabling a policy."""
        store.create_policy(sample_policy)
        result = store.toggle_policy(sample_policy.id, enabled=False)

        assert result is True
        retrieved = store.get_policy(sample_policy.id)
        assert retrieved.enabled is False

    def test_enable_policy(self, store: ControlPlanePolicyStore):
        """Test enabling a disabled policy."""
        policy = ControlPlanePolicy(id="p_toggle", name="Toggle Test", enabled=False)
        store.create_policy(policy)
        result = store.toggle_policy("p_toggle", enabled=True)

        assert result is True
        retrieved = store.get_policy("p_toggle")
        assert retrieved.enabled is True

    def test_toggle_nonexistent_policy(self, store: ControlPlanePolicyStore):
        """Test toggling a nonexistent policy."""
        result = store.toggle_policy("nonexistent", enabled=True)
        assert result is False


# =============================================================================
# Violation Tests
# =============================================================================


class TestCreateViolation:
    """Tests for recording policy violations."""

    def test_create_basic_violation(
        self, store: ControlPlanePolicyStore, sample_violation: PolicyViolation
    ):
        """Test recording a basic violation."""
        result = store.create_violation(sample_violation)
        assert result.id == sample_violation.id
        assert result.policy_id == "policy_test001"

    def test_create_and_retrieve_violation(
        self, store: ControlPlanePolicyStore, sample_violation: PolicyViolation
    ):
        """Test creating then retrieving a violation."""
        store.create_violation(sample_violation)
        violations = store.list_violations(policy_id="policy_test001")

        assert len(violations) == 1
        v = violations[0]
        assert v["policy_id"] == "policy_test001"
        assert v["policy_name"] == "Test Policy"
        assert v["violation_type"] == "agent"
        assert v["agent_id"] == "gpt-3.5-turbo"
        assert v["task_id"] == "task-123"
        assert v["status"] == "open"

    def test_violation_preserves_metadata(
        self, store: ControlPlanePolicyStore, sample_violation: PolicyViolation
    ):
        """Test that violation metadata is preserved."""
        store.create_violation(sample_violation)
        violations = store.list_violations(policy_id="policy_test001")

        assert violations[0]["metadata"] == {"attempted_by": "user-1"}


class TestListViolations:
    """Tests for listing violations."""

    def _create_violations(self, store: ControlPlanePolicyStore) -> list[str]:
        """Helper to create multiple violations."""
        ids = []
        types = ["agent", "region", "sla", "agent", "region"]
        workspaces = ["ws-1", "ws-1", "ws-2", "ws-2", "ws-1"]
        for i, (vtype, ws) in enumerate(zip(types, workspaces)):
            vid = f"violation_{i}"
            store.create_violation(
                PolicyViolation(
                    id=vid,
                    policy_id=f"policy_{i % 2}",
                    policy_name=f"Policy {i % 2}",
                    violation_type=vtype,
                    description=f"Violation {i}",
                    workspace_id=ws,
                    enforcement_level=EnforcementLevel.HARD,
                )
            )
            ids.append(vid)
        return ids

    def test_list_all_violations(self, store: ControlPlanePolicyStore):
        """Test listing all violations."""
        self._create_violations(store)
        result = store.list_violations()
        assert len(result) == 5

    def test_filter_by_policy_id(self, store: ControlPlanePolicyStore):
        """Test filtering violations by policy ID."""
        self._create_violations(store)
        result = store.list_violations(policy_id="policy_0")
        assert len(result) == 3  # indices 0, 2, 4

    def test_filter_by_violation_type(self, store: ControlPlanePolicyStore):
        """Test filtering violations by type."""
        self._create_violations(store)
        result = store.list_violations(violation_type="agent")
        assert len(result) == 2

    def test_filter_by_status(self, store: ControlPlanePolicyStore):
        """Test filtering violations by status."""
        self._create_violations(store)
        # All should be "open" by default
        result = store.list_violations(status="open")
        assert len(result) == 5

        result = store.list_violations(status="resolved")
        assert len(result) == 0

    def test_filter_by_workspace(self, store: ControlPlanePolicyStore):
        """Test filtering violations by workspace."""
        self._create_violations(store)
        result = store.list_violations(workspace_id="ws-1")
        assert len(result) == 3

    def test_list_with_limit(self, store: ControlPlanePolicyStore):
        """Test pagination with limit."""
        self._create_violations(store)
        result = store.list_violations(limit=2)
        assert len(result) == 2

    def test_list_with_offset(self, store: ControlPlanePolicyStore):
        """Test pagination with offset."""
        self._create_violations(store)
        result = store.list_violations(limit=2, offset=3)
        assert len(result) == 2

    def test_combined_filters(self, store: ControlPlanePolicyStore):
        """Test combining multiple filters."""
        self._create_violations(store)
        result = store.list_violations(policy_id="policy_0", violation_type="region")
        assert len(result) == 1  # Only index 4 matches both


class TestCountViolations:
    """Tests for counting violations."""

    def test_count_empty(self, store: ControlPlanePolicyStore):
        """Test counting with no violations."""
        result = store.count_violations()
        assert result == {}

    def test_count_by_type(self, store: ControlPlanePolicyStore):
        """Test counting violations grouped by type."""
        for vtype in ["agent", "agent", "region", "sla"]:
            store.create_violation(
                PolicyViolation(
                    id=f"v_{uuid.uuid4().hex[:8]}",
                    policy_id="p1",
                    policy_name="P1",
                    violation_type=vtype,
                    description=f"{vtype} violation",
                )
            )

        result = store.count_violations()
        assert result["agent"] == 2
        assert result["region"] == 1
        assert result["sla"] == 1

    def test_count_filtered_by_status(self, store: ControlPlanePolicyStore):
        """Test counting violations filtered by status."""
        v1_id = f"v_{uuid.uuid4().hex[:8]}"
        v2_id = f"v_{uuid.uuid4().hex[:8]}"

        store.create_violation(
            PolicyViolation(
                id=v1_id,
                policy_id="p1",
                policy_name="P1",
                violation_type="agent",
                description="Violation 1",
            )
        )
        store.create_violation(
            PolicyViolation(
                id=v2_id,
                policy_id="p1",
                policy_name="P1",
                violation_type="agent",
                description="Violation 2",
            )
        )

        # Resolve one
        store.update_violation_status(v1_id, "resolved", resolved_by="admin")

        open_count = store.count_violations(status="open")
        assert open_count.get("agent", 0) == 1

        resolved_count = store.count_violations(status="resolved")
        assert resolved_count.get("agent", 0) == 1

    def test_count_filtered_by_policy_id(self, store: ControlPlanePolicyStore):
        """Test counting violations filtered by policy ID."""
        for i, pid in enumerate(["p1", "p1", "p2"]):
            store.create_violation(
                PolicyViolation(
                    id=f"v_cnt_{i}",
                    policy_id=pid,
                    policy_name=f"Policy {pid}",
                    violation_type="agent",
                    description=f"Violation {i}",
                )
            )

        result = store.count_violations(policy_id="p1")
        assert result["agent"] == 2

        result = store.count_violations(policy_id="p2")
        assert result["agent"] == 1


class TestUpdateViolationStatus:
    """Tests for updating violation status."""

    def test_resolve_violation(
        self, store: ControlPlanePolicyStore, sample_violation: PolicyViolation
    ):
        """Test resolving a violation."""
        store.create_violation(sample_violation)
        result = store.update_violation_status(
            sample_violation.id,
            "resolved",
            resolved_by="admin",
            resolution_notes="False positive",
        )

        assert result is True

        violations = store.list_violations()
        v = violations[0]
        assert v["status"] == "resolved"
        assert v["resolved_by"] == "admin"
        assert v["resolution_notes"] == "False positive"
        assert v["resolved_at"] is not None

    def test_acknowledge_violation(
        self, store: ControlPlanePolicyStore, sample_violation: PolicyViolation
    ):
        """Test acknowledging a violation (not resolved, no resolved_at)."""
        store.create_violation(sample_violation)
        result = store.update_violation_status(sample_violation.id, "acknowledged")

        assert result is True

        violations = store.list_violations()
        v = violations[0]
        assert v["status"] == "acknowledged"
        assert v["resolved_at"] is None  # Not "resolved" status

    def test_update_nonexistent_violation(self, store: ControlPlanePolicyStore):
        """Test updating a nonexistent violation."""
        result = store.update_violation_status("nonexistent", "resolved")
        assert result is False


# =============================================================================
# Factory and Singleton Tests
# =============================================================================


class TestFactoryFunction:
    """Tests for get_control_plane_policy_store factory."""

    def test_returns_store_instance(self, tmp_path: Path):
        """Test that factory returns a store instance."""
        db_path = tmp_path / "factory_test.db"
        store = get_control_plane_policy_store(db_path=db_path)
        assert store is not None

    def test_returns_singleton(self, tmp_path: Path):
        """Test that factory returns the same instance on subsequent calls."""
        db_path = tmp_path / "singleton_test.db"
        store1 = get_control_plane_policy_store(db_path=db_path)
        store2 = get_control_plane_policy_store()
        assert store1 is store2

    def test_reset_clears_singleton(self, tmp_path: Path):
        """Test that reset clears the singleton."""
        db_path = tmp_path / "reset_test.db"
        store1 = get_control_plane_policy_store(db_path=db_path)
        reset_control_plane_policy_store()
        db_path2 = tmp_path / "reset_test2.db"
        store2 = get_control_plane_policy_store(db_path=db_path2)
        assert store1 is not store2


# =============================================================================
# Execute Method Tests
# =============================================================================


class TestExecuteMethod:
    """Tests for the execute method."""

    def test_raw_sql_query(self, store: ControlPlanePolicyStore):
        """Test raw SQL execution via execute method."""
        # Create a policy first
        policy = ControlPlanePolicy(id="p_exec", name="Execute Test")
        store.create_policy(policy)

        # Use execute to query directly
        cursor = store.execute(
            "SELECT name FROM control_plane_policies WHERE id = ?",
            ("p_exec",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row["name"] == "Execute Test"

    def test_execute_count(self, store: ControlPlanePolicyStore):
        """Test counting via raw execute."""
        for i in range(3):
            store.create_policy(ControlPlanePolicy(id=f"p_count_{i}", name=f"P{i}"))

        cursor = store.execute("SELECT COUNT(*) as cnt FROM control_plane_policies")
        row = cursor.fetchone()
        assert row["cnt"] == 3


# =============================================================================
# Integration Tests
# =============================================================================


class TestPolicyStoreIntegration:
    """Integration tests combining multiple operations."""

    def test_full_policy_lifecycle(self, store: ControlPlanePolicyStore):
        """Test complete policy lifecycle: create, read, update, disable, delete."""
        # Create
        policy = ControlPlanePolicy(
            id="lifecycle_1",
            name="Lifecycle Test",
            description="Will be updated",
            priority=5,
            enabled=True,
        )
        store.create_policy(policy)

        # Read
        retrieved = store.get_policy("lifecycle_1")
        assert retrieved.name == "Lifecycle Test"

        # Update
        updated = store.update_policy("lifecycle_1", {"name": "Updated Lifecycle"})
        assert updated.name == "Updated Lifecycle"

        # Toggle
        store.toggle_policy("lifecycle_1", enabled=False)
        toggled = store.get_policy("lifecycle_1")
        assert toggled.enabled is False

        # Delete
        deleted = store.delete_policy("lifecycle_1")
        assert deleted is True
        assert store.get_policy("lifecycle_1") is None

    def test_violation_lifecycle(self, store: ControlPlanePolicyStore):
        """Test violation lifecycle: create, query, resolve."""
        # Create violation
        violation = PolicyViolation(
            id="v_lifecycle",
            policy_id="p1",
            policy_name="P1",
            violation_type="agent",
            description="Unauthorized agent",
            agent_id="bad-agent",
        )
        store.create_violation(violation)

        # Query
        violations = store.list_violations(status="open")
        assert len(violations) == 1

        counts = store.count_violations(status="open")
        assert counts["agent"] == 1

        # Resolve
        store.update_violation_status(
            "v_lifecycle",
            "resolved",
            resolved_by="admin",
            resolution_notes="Agent has been re-certified",
        )

        # Verify resolution
        open_violations = store.list_violations(status="open")
        assert len(open_violations) == 0

        resolved_violations = store.list_violations(status="resolved")
        assert len(resolved_violations) == 1
        assert resolved_violations[0]["resolved_by"] == "admin"

    def test_multiple_policies_with_violations(self, store: ControlPlanePolicyStore):
        """Test multiple policies each with violations."""
        # Create policies
        for i in range(3):
            store.create_policy(
                ControlPlanePolicy(id=f"mp_{i}", name=f"Multi Policy {i}", priority=i)
            )

        # Create violations for each
        for i in range(3):
            for j in range(i + 1):  # 1, 2, 3 violations respectively
                store.create_violation(
                    PolicyViolation(
                        id=f"mv_{i}_{j}",
                        policy_id=f"mp_{i}",
                        policy_name=f"Multi Policy {i}",
                        violation_type="agent",
                        description=f"Violation {j} for policy {i}",
                    )
                )

        # Verify counts
        all_violations = store.list_violations()
        assert len(all_violations) == 6  # 1 + 2 + 3

        for i in range(3):
            policy_violations = store.list_violations(policy_id=f"mp_{i}")
            assert len(policy_violations) == i + 1
