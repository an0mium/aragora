"""
Tests for A2A Protocol Types.

Tests cover:
- Enum types: TaskStatus, TaskPriority, AgentCapability
- SecurityCard: validation, serialization
- AgentCard: supports_capability, to_dict/from_dict round-trip
- ContextItem: creation, serialization
- TaskRequest: to_dict/from_dict round-trip, defaults
- TaskResult: duration_ms, to_dict/from_dict round-trip
- ARAGORA_AGENT_CARDS pre-defined cards
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from aragora.protocols.a2a.types import (
    ARAGORA_AGENT_CARDS,
    AgentCapability,
    AgentCard,
    ContextItem,
    SecurityCard,
    TaskPriority,
    TaskRequest,
    TaskResult,
    TaskStatus,
)


# ============================================================================
# Enum Tests
# ============================================================================


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_all_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"
        assert TaskStatus.WAITING_INPUT.value == "waiting_input"

    def test_from_string(self):
        assert TaskStatus("pending") == TaskStatus.PENDING
        assert TaskStatus("completed") == TaskStatus.COMPLETED

    def test_is_str_enum(self):
        assert isinstance(TaskStatus.PENDING, str)


class TestTaskPriority:
    """Tests for TaskPriority enum."""

    def test_all_values(self):
        assert TaskPriority.LOW.value == "low"
        assert TaskPriority.NORMAL.value == "normal"
        assert TaskPriority.HIGH.value == "high"
        assert TaskPriority.URGENT.value == "urgent"

    def test_from_string(self):
        assert TaskPriority("normal") == TaskPriority.NORMAL
        assert TaskPriority("urgent") == TaskPriority.URGENT


class TestAgentCapability:
    """Tests for AgentCapability enum."""

    def test_all_values(self):
        expected = [
            "debate",
            "consensus",
            "critique",
            "synthesis",
            "audit",
            "verification",
            "code_review",
            "document_analysis",
            "research",
            "reasoning",
        ]
        for val in expected:
            assert AgentCapability(val) is not None

    def test_count(self):
        assert len(AgentCapability) == 10


# ============================================================================
# SecurityCard Tests
# ============================================================================


class TestSecurityCard:
    """Tests for SecurityCard dataclass."""

    def test_basic_creation(self):
        card = SecurityCard(issuer="aragora", subject="agent-1")
        assert card.issuer == "aragora"
        assert card.subject == "agent-1"
        assert card.public_key is None
        assert card.permissions == []

    def test_is_valid_no_expiry(self):
        """Card without expiry is always valid."""
        card = SecurityCard(issuer="aragora", subject="agent-1")
        assert card.is_valid() is True

    def test_is_valid_future_expiry(self):
        """Card with future expiry is valid."""
        card = SecurityCard(
            issuer="aragora",
            subject="agent-1",
            expires_at=datetime.now() + timedelta(hours=1),
        )
        assert card.is_valid() is True

    def test_is_valid_expired(self):
        """Card with past expiry is invalid."""
        card = SecurityCard(
            issuer="aragora",
            subject="agent-1",
            expires_at=datetime.now() - timedelta(hours=1),
        )
        assert card.is_valid() is False

    def test_to_dict(self):
        card = SecurityCard(
            issuer="aragora",
            subject="agent-1",
            permissions=["read", "write"],
        )
        d = card.to_dict()
        assert d["issuer"] == "aragora"
        assert d["subject"] == "agent-1"
        assert d["permissions"] == ["read", "write"]
        assert d["public_key"] is None
        assert "issued_at" in d

    def test_with_permissions(self):
        card = SecurityCard(
            issuer="aragora",
            subject="agent-1",
            permissions=["debate:read", "debate:write"],
        )
        assert len(card.permissions) == 2


# ============================================================================
# AgentCard Tests
# ============================================================================


class TestAgentCard:
    """Tests for AgentCard dataclass."""

    def test_basic_creation(self):
        card = AgentCard(name="test-agent", description="A test agent")
        assert card.name == "test-agent"
        assert card.description == "A test agent"
        assert card.version == "1.0.0"
        assert card.protocol == "a2a"

    def test_defaults(self):
        card = AgentCard(name="test", description="test")
        assert card.capabilities == []
        assert card.input_modes == ["text"]
        assert card.output_modes == ["text"]
        assert card.endpoint is None
        assert card.tags == []
        assert card.requires_auth is False
        assert card.max_concurrent_tasks == 10
        assert card.estimated_response_time_ms == 5000

    def test_supports_capability(self):
        card = AgentCard(
            name="test",
            description="test",
            capabilities=[AgentCapability.DEBATE, AgentCapability.CONSENSUS],
        )
        assert card.supports_capability(AgentCapability.DEBATE) is True
        assert card.supports_capability(AgentCapability.AUDIT) is False

    def test_to_dict(self):
        card = AgentCard(
            name="test-agent",
            description="A test agent",
            capabilities=[AgentCapability.DEBATE],
            tags=["test"],
        )
        d = card.to_dict()
        assert d["name"] == "test-agent"
        assert d["capabilities"] == ["debate"]
        assert d["tags"] == ["test"]
        assert d["protocol"] == "a2a"

    def test_from_dict(self):
        data = {
            "name": "round-trip",
            "description": "Test round trip",
            "version": "2.0.0",
            "capabilities": ["debate", "consensus"],
            "tags": ["tag1", "tag2"],
            "max_concurrent_tasks": 20,
        }
        card = AgentCard.from_dict(data)
        assert card.name == "round-trip"
        assert card.version == "2.0.0"
        assert AgentCapability.DEBATE in card.capabilities
        assert len(card.tags) == 2
        assert card.max_concurrent_tasks == 20

    def test_from_dict_defaults(self):
        """from_dict uses defaults for missing fields."""
        data = {"name": "minimal", "description": "Minimal agent"}
        card = AgentCard.from_dict(data)
        assert card.version == "1.0.0"
        assert card.capabilities == []
        assert card.protocol == "a2a"

    def test_round_trip(self):
        """to_dict -> from_dict produces equivalent card."""
        original = AgentCard(
            name="full-agent",
            description="Full featured agent",
            version="3.0.0",
            capabilities=[AgentCapability.RESEARCH, AgentCapability.REASONING],
            input_modes=["text", "file"],
            output_modes=["text", "structured"],
            endpoint="https://agent.example.com",
            tags=["research", "ai"],
            organization="aragora",
            requires_auth=True,
            max_concurrent_tasks=5,
            estimated_response_time_ms=10000,
        )
        d = original.to_dict()
        restored = AgentCard.from_dict(d)

        assert restored.name == original.name
        assert restored.version == original.version
        assert restored.capabilities == original.capabilities
        assert restored.tags == original.tags
        assert restored.max_concurrent_tasks == original.max_concurrent_tasks


# ============================================================================
# ContextItem Tests
# ============================================================================


class TestContextItem:
    """Tests for ContextItem dataclass."""

    def test_basic_creation(self):
        item = ContextItem(type="text", content="Hello world")
        assert item.type == "text"
        assert item.content == "Hello world"
        assert item.mime_type == "text/plain"
        assert item.metadata == {}

    def test_file_context(self):
        item = ContextItem(
            type="file",
            content="/path/to/file.py",
            mime_type="text/x-python",
            metadata={"filename": "file.py"},
        )
        assert item.type == "file"
        assert item.mime_type == "text/x-python"

    def test_to_dict(self):
        item = ContextItem(
            type="structured",
            content='{"key": "value"}',
            mime_type="application/json",
        )
        d = item.to_dict()
        assert d["type"] == "structured"
        assert d["content"] == '{"key": "value"}'
        assert d["mime_type"] == "application/json"


# ============================================================================
# TaskRequest Tests
# ============================================================================


class TestTaskRequest:
    """Tests for TaskRequest dataclass."""

    def test_basic_creation(self):
        req = TaskRequest(task_id="t_123", instruction="Do something")
        assert req.task_id == "t_123"
        assert req.instruction == "Do something"
        assert req.priority == TaskPriority.NORMAL
        assert req.timeout_ms == 300000
        assert req.stream_output is False

    def test_with_context(self):
        ctx = [ContextItem(type="text", content="Background info")]
        req = TaskRequest(
            task_id="t_456",
            instruction="Analyze",
            context=ctx,
        )
        assert len(req.context) == 1

    def test_to_dict(self):
        req = TaskRequest(
            task_id="t_789",
            instruction="Debate this",
            capability=AgentCapability.DEBATE,
            priority=TaskPriority.HIGH,
        )
        d = req.to_dict()
        assert d["task_id"] == "t_789"
        assert d["capability"] == "debate"
        assert d["priority"] == "high"
        assert d["instruction"] == "Debate this"

    def test_from_dict(self):
        data = {
            "task_id": "t_round",
            "instruction": "Test round trip",
            "capability": "audit",
            "priority": "urgent",
            "timeout_ms": 600000,
            "stream_output": True,
        }
        req = TaskRequest.from_dict(data)
        assert req.task_id == "t_round"
        assert req.capability == AgentCapability.AUDIT
        assert req.priority == TaskPriority.URGENT
        assert req.timeout_ms == 600000
        assert req.stream_output is True

    def test_from_dict_defaults(self):
        data = {"task_id": "t_min", "instruction": "Minimal request"}
        req = TaskRequest.from_dict(data)
        assert req.capability is None
        assert req.priority == TaskPriority.NORMAL
        assert req.context == []

    def test_from_dict_with_context(self):
        data = {
            "task_id": "t_ctx",
            "instruction": "With context",
            "context": [
                {"type": "text", "content": "Background"},
                {"type": "file", "content": "/path/to/file"},
            ],
        }
        req = TaskRequest.from_dict(data)
        assert len(req.context) == 2
        assert req.context[0].type == "text"

    def test_round_trip(self):
        original = TaskRequest(
            task_id="t_full",
            instruction="Full request",
            capability=AgentCapability.RESEARCH,
            priority=TaskPriority.HIGH,
            timeout_ms=120000,
            requester_agent="test-requester",
            stream_output=True,
            return_intermediate=True,
            metadata={"key": "value"},
        )
        d = original.to_dict()
        restored = TaskRequest.from_dict(d)

        assert restored.task_id == original.task_id
        assert restored.instruction == original.instruction
        assert restored.capability == original.capability
        assert restored.priority == original.priority
        assert restored.timeout_ms == original.timeout_ms
        assert restored.stream_output == original.stream_output


# ============================================================================
# TaskResult Tests
# ============================================================================


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_basic_creation(self):
        result = TaskResult(
            task_id="t_123",
            agent_name="test-agent",
            status=TaskStatus.COMPLETED,
            output="The answer is 42",
        )
        assert result.task_id == "t_123"
        assert result.status == TaskStatus.COMPLETED
        assert result.output == "The answer is 42"
        assert result.tokens_used == 0

    def test_failed_result(self):
        result = TaskResult(
            task_id="t_fail",
            agent_name="test-agent",
            status=TaskStatus.FAILED,
            error_message="Something went wrong",
        )
        assert result.status == TaskStatus.FAILED
        assert result.error_message == "Something went wrong"

    def test_duration_ms(self):
        """Test duration calculation from timestamps."""
        start = datetime(2026, 1, 1, 12, 0, 0)
        end = datetime(2026, 1, 1, 12, 0, 5)  # 5 seconds later
        result = TaskResult(
            task_id="t_dur",
            agent_name="test-agent",
            status=TaskStatus.COMPLETED,
            started_at=start,
            completed_at=end,
        )
        assert result.duration_ms == 5000

    def test_duration_ms_none(self):
        """Test duration is None when timestamps missing."""
        result = TaskResult(
            task_id="t_nodur",
            agent_name="test-agent",
            status=TaskStatus.RUNNING,
        )
        assert result.duration_ms is None

    def test_to_dict(self):
        result = TaskResult(
            task_id="t_dict",
            agent_name="test-agent",
            status=TaskStatus.COMPLETED,
            output="result text",
            tokens_used=150,
        )
        d = result.to_dict()
        assert d["task_id"] == "t_dict"
        assert d["status"] == "completed"
        assert d["output"] == "result text"
        assert d["tokens_used"] == 150

    def test_from_dict(self):
        data = {
            "task_id": "t_from",
            "agent_name": "restored-agent",
            "status": "failed",
            "error_message": "Timeout",
            "tokens_used": 50,
        }
        result = TaskResult.from_dict(data)
        assert result.task_id == "t_from"
        assert result.status == TaskStatus.FAILED
        assert result.error_message == "Timeout"
        assert result.tokens_used == 50

    def test_from_dict_with_timestamps(self):
        data = {
            "task_id": "t_ts",
            "agent_name": "test-agent",
            "status": "completed",
            "started_at": "2026-01-01T12:00:00",
            "completed_at": "2026-01-01T12:00:10",
        }
        result = TaskResult.from_dict(data)
        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.duration_ms == 10000

    def test_round_trip(self):
        original = TaskResult(
            task_id="t_rt",
            agent_name="full-agent",
            status=TaskStatus.COMPLETED,
            output="Full output",
            structured_output={"confidence": 0.95},
            tokens_used=200,
            subtasks=["sub-1", "sub-2"],
            metadata={"source": "test"},
        )
        d = original.to_dict()
        restored = TaskResult.from_dict(d)

        assert restored.task_id == original.task_id
        assert restored.status == original.status
        assert restored.output == original.output
        assert restored.structured_output == original.structured_output
        assert restored.tokens_used == original.tokens_used
        assert restored.subtasks == original.subtasks


# ============================================================================
# ARAGORA_AGENT_CARDS Tests
# ============================================================================


class TestPredefinedCards:
    """Tests for pre-defined Aragora agent cards."""

    def test_all_cards_exist(self):
        expected = ["debate-orchestrator", "audit-engine", "gauntlet", "research"]
        for name in expected:
            assert name in ARAGORA_AGENT_CARDS

    def test_debate_orchestrator_capabilities(self):
        card = ARAGORA_AGENT_CARDS["debate-orchestrator"]
        assert AgentCapability.DEBATE in card.capabilities
        assert AgentCapability.CONSENSUS in card.capabilities
        assert card.organization == "aragora"

    def test_audit_engine_capabilities(self):
        card = ARAGORA_AGENT_CARDS["audit-engine"]
        assert AgentCapability.AUDIT in card.capabilities
        assert AgentCapability.DOCUMENT_ANALYSIS in card.capabilities

    def test_gauntlet_capabilities(self):
        card = ARAGORA_AGENT_CARDS["gauntlet"]
        assert AgentCapability.CRITIQUE in card.capabilities
        assert AgentCapability.VERIFICATION in card.capabilities

    def test_all_cards_have_name_and_description(self):
        for key, card in ARAGORA_AGENT_CARDS.items():
            assert card.name, f"Card {key} missing name"
            assert card.description, f"Card {key} missing description"
            assert card.organization == "aragora"
