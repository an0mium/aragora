"""
Tests for handler TypedDict types.

Validates that TypedDict definitions are correctly structured and
work with type-compatible data.
"""

import pytest
from typing import get_type_hints, get_origin, get_args
from typing_extensions import NotRequired, Required

from aragora.server.handlers.types import (
    # Debate types
    CreateDebateRequest,
    DebateUpdateRequest,
    DebateSummaryResponse,
    DebateDetailResponse,
    DebateListResponse,
    # Fork types
    ForkRequest,
    ForkResponse,
    # Batch types
    BatchDebateItem,
    BatchSubmitRequest,
    BatchSubmitResponse,
    BatchStatusResponse,
    # Auth types
    UserRegisterRequest,
    UserLoginRequest,
    AuthResponse,
    UserResponse,
    # Org types
    OrgCreateRequest,
    OrgInviteRequest,
    OrgResponse,
    # Gauntlet types
    GauntletRunRequest,
    GauntletResponse,
    # Verification types
    VerificationRequest,
    VerificationResponse,
    # Memory types
    MemoryCleanupRequest,
    MemoryEntry,
    # Agent types
    AgentConfigRequest,
    AgentStatusResponse,
)


# =============================================================================
# Debate Type Tests
# =============================================================================


class TestCreateDebateRequest:
    """Tests for CreateDebateRequest TypedDict."""

    def test_empty_dict_allowed(self):
        """Empty dict is allowed since total=False."""
        request: CreateDebateRequest = {}
        assert request == {}

    def test_minimal_request(self):
        """Request with just task field."""
        request: CreateDebateRequest = {"task": "Test task"}
        assert request["task"] == "Test task"

    def test_full_request(self):
        """Request with all optional fields."""
        request: CreateDebateRequest = {
            "task": "Test task",
            "question": "Alternative question",
            "agents": ["claude", "gpt4"],
            "mode": "debate",
            "rounds": 3,
            "consensus": "majority",
        }
        assert len(request) == 6
        assert request["agents"] == ["claude", "gpt4"]

    def test_type_hints_available(self):
        """Type hints can be retrieved."""
        hints = get_type_hints(CreateDebateRequest)
        assert "task" in hints
        assert "agents" in hints


class TestDebateSummaryResponse:
    """Tests for DebateSummaryResponse TypedDict."""

    def test_required_fields(self):
        """All fields are required (total=True by default)."""
        response: DebateSummaryResponse = {
            "id": "debate-123",
            "task": "Test task",
            "status": "active",
            "created_at": "2024-01-01T00:00:00Z",
            "agents": ["claude"],
            "round_count": 1,
        }
        assert response["id"] == "debate-123"
        assert response["round_count"] == 1

    def test_type_hints(self):
        """Type hints are correctly defined."""
        hints = get_type_hints(DebateSummaryResponse)
        assert "id" in hints
        assert "status" in hints
        assert len(hints) == 6


class TestDebateListResponse:
    """Tests for DebateListResponse TypedDict."""

    def test_nested_structure(self):
        """Response contains list of summaries."""
        response: DebateListResponse = {
            "debates": [
                {
                    "id": "debate-1",
                    "task": "Task 1",
                    "status": "active",
                    "created_at": "2024-01-01T00:00:00Z",
                    "agents": ["claude"],
                    "round_count": 2,
                },
            ],
            "total": 1,
            "offset": 0,
            "limit": 20,
        }
        assert len(response["debates"]) == 1
        assert response["total"] == 1


# =============================================================================
# Fork Type Tests
# =============================================================================


class TestForkRequest:
    """Tests for ForkRequest TypedDict."""

    def test_required_branch_point(self):
        """branch_point is required."""
        request: ForkRequest = {"branch_point": 5}
        assert request["branch_point"] == 5

    def test_with_optional_context(self):
        """modified_context is optional."""
        request: ForkRequest = {
            "branch_point": 5,
            "modified_context": "New context",
        }
        assert "modified_context" in request


class TestForkResponse:
    """Tests for ForkResponse TypedDict."""

    def test_all_fields(self):
        """All fields are required."""
        response: ForkResponse = {
            "fork_id": "fork-123",
            "parent_id": "debate-456",
            "branch_point": 3,
            "created_at": "2024-01-01T00:00:00Z",
        }
        assert response["fork_id"] == "fork-123"


# =============================================================================
# Batch Type Tests
# =============================================================================


class TestBatchSubmitRequest:
    """Tests for BatchSubmitRequest TypedDict."""

    def test_minimal_request(self):
        """Request with just items."""
        request: BatchSubmitRequest = {"items": [{"task": "Task 1"}, {"task": "Task 2"}]}
        assert len(request["items"]) == 2

    def test_with_options(self):
        """Request with optional fields."""
        request: BatchSubmitRequest = {
            "items": [{"task": "Task 1"}],
            "webhook_url": "https://example.com/webhook",
            "max_parallel": 5,
        }
        assert request["max_parallel"] == 5


class TestBatchStatusResponse:
    """Tests for BatchStatusResponse TypedDict."""

    def test_minimal_response(self):
        """Response without results."""
        response: BatchStatusResponse = {
            "batch_id": "batch-123",
            "status": "running",
            "total": 10,
            "completed": 5,
            "failed": 0,
        }
        assert response["completed"] == 5

    def test_with_results(self):
        """Response with results."""
        response: BatchStatusResponse = {
            "batch_id": "batch-123",
            "status": "completed",
            "total": 2,
            "completed": 2,
            "failed": 0,
            "results": [{"id": "1"}, {"id": "2"}],
        }
        assert len(response["results"]) == 2


# =============================================================================
# Auth Type Tests
# =============================================================================


class TestUserRegisterRequest:
    """Tests for UserRegisterRequest TypedDict."""

    def test_minimal_registration(self):
        """Registration with just email and password."""
        request: UserRegisterRequest = {
            "email": "test@example.com",
            "password": "securepass123",
        }
        assert request["email"] == "test@example.com"

    def test_with_name(self):
        """Registration with optional name."""
        request: UserRegisterRequest = {
            "email": "test@example.com",
            "password": "securepass123",
            "name": "Test User",
        }
        assert request["name"] == "Test User"


class TestAuthResponse:
    """Tests for AuthResponse TypedDict."""

    def test_full_response(self):
        """Auth response with all fields."""
        response: AuthResponse = {
            "token": "jwt.token.here",
            "user": {"id": "user-123", "email": "test@example.com"},
            "expires_at": "2024-12-31T23:59:59Z",
        }
        assert response["token"].startswith("jwt")
        assert "id" in response["user"]


# =============================================================================
# Organization Type Tests
# =============================================================================


class TestOrgCreateRequest:
    """Tests for OrgCreateRequest TypedDict."""

    def test_minimal_create(self):
        """Create with just name."""
        request: OrgCreateRequest = {"name": "My Organization"}
        assert request["name"] == "My Organization"

    def test_with_slug(self):
        """Create with optional slug."""
        request: OrgCreateRequest = {
            "name": "My Organization",
            "slug": "my-org",
        }
        assert request["slug"] == "my-org"


class TestOrgInviteRequest:
    """Tests for OrgInviteRequest TypedDict."""

    def test_minimal_invite(self):
        """Invite with just email."""
        request: OrgInviteRequest = {"email": "user@example.com"}
        assert request["email"] == "user@example.com"

    def test_with_role(self):
        """Invite with specific role."""
        request: OrgInviteRequest = {
            "email": "admin@example.com",
            "role": "admin",
        }
        assert request["role"] == "admin"


# =============================================================================
# Gauntlet Type Tests
# =============================================================================


class TestGauntletRunRequest:
    """Tests for GauntletRunRequest TypedDict."""

    def test_minimal_request(self):
        """Request with just input content."""
        request: GauntletRunRequest = {"input_content": "Test content"}
        assert request["input_content"] == "Test content"

    def test_full_request(self):
        """Request with all options."""
        request: GauntletRunRequest = {
            "input_content": "Test content",
            "input_type": "code",
            "agents": ["claude", "gpt4"],
            "persona": "security-auditor",
            "profile": "comprehensive",
        }
        assert request["input_type"] == "code"
        assert len(request["agents"]) == 2


class TestGauntletResponse:
    """Tests for GauntletResponse TypedDict."""

    def test_minimal_response(self):
        """Response without result."""
        response: GauntletResponse = {
            "run_id": "gauntlet-123",
            "status": "running",
        }
        assert response["status"] == "running"

    def test_with_result(self):
        """Response with result."""
        response: GauntletResponse = {
            "run_id": "gauntlet-123",
            "status": "completed",
            "result": {"verdict": "pass", "score": 0.95},
        }
        assert "result" in response


# =============================================================================
# Memory Type Tests
# =============================================================================


class TestMemoryCleanupRequest:
    """Tests for MemoryCleanupRequest TypedDict."""

    def test_empty_request(self):
        """Empty request allowed since total=False."""
        request: MemoryCleanupRequest = {}
        assert request == {}

    def test_full_request(self):
        """Request with all options."""
        request: MemoryCleanupRequest = {
            "tier": "fast",
            "archive": "true",
            "max_age_hours": 24.0,
        }
        assert request["tier"] == "fast"


class TestMemoryEntry:
    """Tests for MemoryEntry TypedDict."""

    def test_minimal_entry(self):
        """Entry without expires_at."""
        entry: MemoryEntry = {
            "id": "mem-123",
            "tier": "medium",
            "content": "Test content",
            "metadata": {"key": "value"},
            "created_at": "2024-01-01T00:00:00Z",
        }
        assert entry["tier"] == "medium"

    def test_with_expiration(self):
        """Entry with expires_at."""
        entry: MemoryEntry = {
            "id": "mem-123",
            "tier": "fast",
            "content": "Ephemeral content",
            "metadata": {},
            "created_at": "2024-01-01T00:00:00Z",
            "expires_at": "2024-01-01T01:00:00Z",
        }
        assert "expires_at" in entry


# =============================================================================
# Agent Type Tests
# =============================================================================


class TestAgentConfigRequest:
    """Tests for AgentConfigRequest TypedDict."""

    def test_minimal_config(self):
        """Config with just name."""
        config: AgentConfigRequest = {"name": "claude"}
        assert config["name"] == "claude"

    def test_full_config(self):
        """Config with all options."""
        config: AgentConfigRequest = {
            "name": "custom-agent",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2000,
            "system_prompt": "You are a helpful assistant.",
        }
        assert config["temperature"] == 0.7


# =============================================================================
# Type Introspection Tests
# =============================================================================


class TestTypeIntrospection:
    """Tests for type introspection capabilities."""

    def test_all_types_importable(self):
        """All TypedDicts can be imported."""
        types = [
            CreateDebateRequest,
            DebateUpdateRequest,
            DebateSummaryResponse,
            DebateDetailResponse,
            DebateListResponse,
            ForkRequest,
            ForkResponse,
            BatchDebateItem,
            BatchSubmitRequest,
            BatchSubmitResponse,
            BatchStatusResponse,
            UserRegisterRequest,
            UserLoginRequest,
            AuthResponse,
            UserResponse,
            OrgCreateRequest,
            OrgInviteRequest,
            OrgResponse,
            GauntletRunRequest,
            GauntletResponse,
            VerificationRequest,
            VerificationResponse,
            MemoryCleanupRequest,
            MemoryEntry,
            AgentConfigRequest,
            AgentStatusResponse,
        ]
        for t in types:
            assert hasattr(t, "__annotations__")

    def test_typeddict_inheritance(self):
        """TypedDicts have proper base classes."""
        # TypedDicts should have __annotations__
        assert hasattr(CreateDebateRequest, "__annotations__")
        assert hasattr(DebateSummaryResponse, "__annotations__")

    def test_optional_vs_required(self):
        """Test that NotRequired fields are properly typed."""
        hints = get_type_hints(DebateDetailResponse, include_extras=True)
        # consensus should be NotRequired
        consensus_type = hints.get("consensus")
        assert consensus_type is not None


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_agents_list(self):
        """Request with empty agents list."""
        request: CreateDebateRequest = {
            "task": "Test",
            "agents": [],
        }
        assert request["agents"] == []

    def test_unicode_content(self):
        """Request with unicode content."""
        request: CreateDebateRequest = {
            "task": "Test unicode: \u2764 \u2728 \U0001f600",
        }
        assert "\u2764" in request["task"]

    def test_long_string_content(self):
        """Request with long string content."""
        long_content = "x" * 10000
        request: GauntletRunRequest = {"input_content": long_content}
        assert len(request["input_content"]) == 10000

    def test_nested_metadata(self):
        """Memory entry with deeply nested metadata."""
        entry: MemoryEntry = {
            "id": "mem-123",
            "tier": "slow",
            "content": "Test",
            "metadata": {"level1": {"level2": {"level3": {"value": 42}}}},
            "created_at": "2024-01-01T00:00:00Z",
        }
        assert entry["metadata"]["level1"]["level2"]["level3"]["value"] == 42
