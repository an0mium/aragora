"""
Tests for Control Plane Shared State.

Tests cover:
- AgentState dataclass creation and serialization
- TaskState dataclass creation and serialization
- SharedControlPlaneState singleton management
- get_shared_state / set_shared_state functions
"""

import pytest
from datetime import datetime, timezone

from aragora.control_plane.shared_state import (
    AgentState,
    TaskState,
    get_shared_state,
    get_shared_state_sync,
    set_shared_state,
    SharedControlPlaneState,
    _shared_state,
)


class TestAgentState:
    """Tests for AgentState dataclass."""

    def test_agent_state_creation(self):
        """Test creating an agent state."""
        agent = AgentState(
            id="agent_123",
            name="Claude Assistant",
            type="claude",
            model="claude-3-opus",
            status="active",
            role="reviewer",
            capabilities={"code_review", "security_audit"},
            tasks_completed=50,
            findings_generated=25,
            avg_response_time=2.5,
            error_rate=0.02,
            uptime_seconds=3600.0,
            created_at="2024-01-01T00:00:00Z",
        )

        assert agent.id == "agent_123"
        assert agent.name == "Claude Assistant"
        assert agent.type == "claude"
        assert agent.status == "active"
        assert "code_review" in agent.capabilities
        assert agent.tasks_completed == 50

    def test_agent_state_defaults(self):
        """Test agent state with default values."""
        agent = AgentState(
            id="agent_456",
            name="Test Agent",
            type="gpt",
            model="gpt-4",
            status="idle",
        )

        assert agent.role == ""
        assert agent.capabilities == set()
        assert agent.tasks_completed == 0
        assert agent.error_rate == 0.0
        assert agent.last_active is None
        assert agent.paused_at is None

    def test_agent_state_to_dict(self):
        """Test agent state serialization."""
        agent = AgentState(
            id="agent_test",
            name="Test",
            type="claude",
            model="claude-3",
            status="active",
            capabilities={"analysis"},
            tasks_completed=10,
        )

        data = agent.to_dict()
        assert data["id"] == "agent_test"
        assert data["status"] == "active"
        assert data["capabilities"] == ["analysis"]
        assert data["tasks_completed"] == 10

    def test_agent_state_from_dict(self):
        """Test agent state deserialization."""
        data = {
            "id": "agent_from_dict",
            "name": "From Dict",
            "type": "mistral",
            "model": "mistral-large",
            "status": "paused",
            "capabilities": ["translation", "summarization"],
            "tasks_completed": 100,
            "error_rate": 0.05,
        }

        agent = AgentState.from_dict(data)
        assert agent.id == "agent_from_dict"
        assert agent.name == "From Dict"
        assert agent.status == "paused"
        assert "translation" in agent.capabilities
        assert agent.tasks_completed == 100

    def test_agent_state_from_dict_defaults(self):
        """Test agent state from partial dict uses defaults."""
        data = {"id": "minimal", "name": "Minimal", "type": "gpt", "model": "gpt-4"}

        agent = AgentState.from_dict(data)
        assert agent.status == "idle"
        assert agent.role == ""
        assert agent.capabilities == set()

    def test_agent_state_roundtrip(self):
        """Test serialization/deserialization roundtrip."""
        original = AgentState(
            id="roundtrip",
            name="Roundtrip Test",
            type="claude",
            model="claude-3-sonnet",
            status="active",
            capabilities={"test1", "test2"},
            tasks_completed=42,
            metadata={"custom": "value"},
        )

        data = original.to_dict()
        restored = AgentState.from_dict(data)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.status == original.status
        assert restored.tasks_completed == original.tasks_completed


class TestTaskState:
    """Tests for TaskState dataclass."""

    def test_task_state_creation(self):
        """Test creating a task state."""
        task = TaskState(
            id="task_123",
            type="code_review",
            priority="high",
            status="pending",
            created_at="2024-01-01T00:00:00Z",
            assigned_agent="agent_456",
            document_id="doc_789",
            audit_type="security",
        )

        assert task.id == "task_123"
        assert task.type == "code_review"
        assert task.priority == "high"
        assert task.status == "pending"
        assert task.assigned_agent == "agent_456"

    def test_task_state_defaults(self):
        """Test task state with default values."""
        task = TaskState(
            id="task_456",
            type="analysis",
            priority="normal",
            status="processing",
        )

        assert task.created_at == ""
        assert task.assigned_agent is None
        assert task.document_id is None
        assert task.payload == {}
        assert task.metadata == {}

    def test_task_state_to_dict(self):
        """Test task state serialization."""
        task = TaskState(
            id="task_test",
            type="deliberation",
            priority="high",
            status="completed",
            payload={"topic": "Review this code"},
            metadata={"source": "api"},
        )

        data = task.to_dict()
        assert data["id"] == "task_test"
        assert data["type"] == "deliberation"
        assert data["priority"] == "high"
        assert data["payload"]["topic"] == "Review this code"


class TestSharedStateManagement:
    """Tests for shared state singleton management."""

    def test_get_shared_state_returns_none_initially(self):
        """Test that get_shared_state returns None when not set."""
        # Reset the module-level state
        import aragora.control_plane.shared_state as ss

        original = ss._shared_state
        ss._shared_state = None
        try:
            # Check sync getter behavior
            from aragora.control_plane.shared_state import get_shared_state_sync

            result = get_shared_state_sync()
            assert result is None
        finally:
            ss._shared_state = original

    def test_set_shared_state(self):
        """Test setting the shared state."""
        import aragora.control_plane.shared_state as ss

        # Create a real SharedControlPlaneState instance
        state = SharedControlPlaneState()
        original = ss._shared_state

        try:
            set_shared_state(state)
            assert ss._shared_state is state
        finally:
            ss._shared_state = original

    def test_shared_control_plane_state_creation(self):
        """Test creating SharedControlPlaneState."""

        state = SharedControlPlaneState(
            redis_url="redis://localhost:6379",
            key_prefix="test:cp:",
        )

        assert state is not None
        assert state._key_prefix == "test:cp:"
        # Not connected yet
        assert state.is_redis_connected is False


class TestAgentStatusValues:
    """Tests for valid agent status values."""

    @pytest.mark.parametrize(
        "status",
        ["active", "paused", "idle", "offline"],
    )
    def test_valid_agent_statuses(self, status):
        """Test that valid status values work."""
        agent = AgentState(
            id="test",
            name="Test",
            type="claude",
            model="claude-3",
            status=status,
        )
        assert agent.status == status


class TestTaskPriorityValues:
    """Tests for valid task priority values."""

    @pytest.mark.parametrize(
        "priority",
        ["high", "normal", "low"],
    )
    def test_valid_task_priorities(self, priority):
        """Test that valid priority values work."""
        task = TaskState(
            id="test",
            type="review",
            priority=priority,
            status="pending",
        )
        assert task.priority == priority


class TestTaskStatusValues:
    """Tests for valid task status values."""

    @pytest.mark.parametrize(
        "status",
        ["pending", "processing", "completed", "failed"],
    )
    def test_valid_task_statuses(self, status):
        """Test that valid status values work."""
        task = TaskState(
            id="test",
            type="review",
            priority="normal",
            status=status,
        )
        assert task.status == status
