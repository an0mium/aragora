"""Tests for aragora/server/handlers/types.py.

Comprehensive coverage of all type definitions in the handler types module:
1. HandlerProtocol - runtime_checkable protocol behavior
2. TypedDict definitions - instantiation, field access, totality semantics
3. Type aliases - HandlerFunction, AsyncHandlerFunction, middleware types
4. __all__ exports - completeness and importability
5. RequestContext - field enumeration
6. Parameter types - PaginationParams, FilterParams, SortParams, QueryParams
7. Request/response types - all domain-specific TypedDicts
8. Error types - ErrorDetail, ErrorResponse
9. Metrics types - ConvergenceMetrics, PhaseMetrics, DebateMetrics, etc.
10. Common response wrappers - PaginatedResponse, StatusResponse
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeVar, Union, get_type_hints
from unittest.mock import MagicMock

import pytest
from typing_extensions import NotRequired

from aragora.server.handlers.types import (
    # Protocol
    HandlerProtocol,
    # Request context
    RequestContext,
    # Response type alias
    ResponseType,
    # Handler function type aliases
    AsyncHandlerFunction,
    HandlerFunction,
    MaybeAsyncHandlerFunction,
    # Middleware function type aliases
    AsyncMiddlewareFunction,
    MaybeAsyncMiddlewareFunction,
    MiddlewareFactory,
    MiddlewareFunction,
    # Parameter types
    FilterParams,
    PaginationParams,
    QueryParams,
    SortParams,
    # Debate types
    CreateDebateRequest,
    DebateDetailResponse,
    DebateListResponse,
    DebateSummaryResponse,
    DebateUpdateRequest,
    # Fork types
    ForkRequest,
    ForkResponse,
    # Batch types
    BatchDebateItem,
    BatchStatusResponse,
    BatchSubmitRequest,
    BatchSubmitResponse,
    # Auth types
    AuthResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
    # Org types
    OrgCreateRequest,
    OrgInviteRequest,
    OrgResponse,
    # Gauntlet types
    GauntletResponse,
    GauntletRunRequest,
    # Verification types
    VerificationRequest,
    VerificationResponse,
    # Memory types
    MemoryCleanupRequest,
    MemoryEntry,
    # Agent types
    AgentConfigRequest,
    AgentStatusResponse,
    # Probe types
    ProbeResultResponse,
    ProbeRunRequest,
    # Social types
    ShareResponse,
    ShareUpdateRequest,
    SocialPublishRequest,
    # Billing types
    CheckoutSessionRequest,
    CheckoutSessionResponse,
    UsageResponse,
    # Plugin types
    PluginInstallRequest,
    PluginResponse,
    PluginRunRequest,
    # Metrics types
    AgentPerformanceMetrics,
    ConvergenceMetrics,
    DashboardResponse,
    DashboardSummary,
    DebateMetrics,
    PhaseMetrics,
    # Error types
    ErrorDetail,
    ErrorResponse,
    # Common types
    PaginatedResponse,
    StatusResponse,
)


# =============================================================================
# Helper classes for HandlerProtocol tests
# =============================================================================


class _CompleteHandler:
    """Implements all required HandlerProtocol methods."""

    def handle(self, path, query_params, handler):
        return None

    def handle_post(self, path, query_params, handler):
        return None

    def handle_delete(self, path, query_params, handler):
        return None

    def handle_patch(self, path, query_params, handler):
        return None

    def handle_put(self, path, query_params, handler):
        return None


class _AsyncCompleteHandler:
    """Implements all HandlerProtocol methods as async."""

    async def handle(self, path, query_params, handler):
        return None

    async def handle_post(self, path, query_params, handler):
        return None

    async def handle_delete(self, path, query_params, handler):
        return None

    async def handle_patch(self, path, query_params, handler):
        return None

    async def handle_put(self, path, query_params, handler):
        return None


class _HandleOnly:
    """Only has the handle method."""

    def handle(self, path, query_params, handler):
        return None


class _MissingPut:
    """Missing handle_put."""

    def handle(self, path, query_params, handler):
        return None

    def handle_post(self, path, query_params, handler):
        return None

    def handle_delete(self, path, query_params, handler):
        return None

    def handle_patch(self, path, query_params, handler):
        return None


class _EmptyClass:
    pass


class _WrongSignature:
    """Has methods but wrong names."""

    def process(self, path, query_params, handler):
        return None


# =============================================================================
# HandlerProtocol Tests
# =============================================================================


class TestHandlerProtocol:
    """Test HandlerProtocol runtime_checkable behavior."""

    def test_complete_handler_satisfies_protocol(self):
        assert isinstance(_CompleteHandler(), HandlerProtocol)

    def test_async_complete_handler_satisfies_protocol(self):
        assert isinstance(_AsyncCompleteHandler(), HandlerProtocol)

    def test_handle_only_does_not_satisfy(self):
        assert not isinstance(_HandleOnly(), HandlerProtocol)

    def test_missing_put_does_not_satisfy(self):
        assert not isinstance(_MissingPut(), HandlerProtocol)

    def test_empty_class_does_not_satisfy(self):
        assert not isinstance(_EmptyClass(), HandlerProtocol)

    def test_none_does_not_satisfy(self):
        assert not isinstance(None, HandlerProtocol)

    def test_string_does_not_satisfy(self):
        assert not isinstance("handler", HandlerProtocol)

    def test_int_does_not_satisfy(self):
        assert not isinstance(42, HandlerProtocol)

    def test_dict_does_not_satisfy(self):
        assert not isinstance({}, HandlerProtocol)

    def test_wrong_signature_class_does_not_satisfy(self):
        assert not isinstance(_WrongSignature(), HandlerProtocol)

    def test_protocol_is_runtime_checkable(self):
        """HandlerProtocol is decorated with @runtime_checkable."""
        assert hasattr(HandlerProtocol, "__protocol_attrs__") or hasattr(
            HandlerProtocol, "_is_runtime_protocol"
        )

    def test_protocol_has_handle_method(self):
        """Protocol defines handle method."""
        assert hasattr(HandlerProtocol, "handle")

    def test_protocol_has_handle_post_method(self):
        assert hasattr(HandlerProtocol, "handle_post")

    def test_protocol_has_handle_delete_method(self):
        assert hasattr(HandlerProtocol, "handle_delete")

    def test_protocol_has_handle_patch_method(self):
        assert hasattr(HandlerProtocol, "handle_patch")

    def test_protocol_has_handle_put_method(self):
        assert hasattr(HandlerProtocol, "handle_put")

    def test_protocol_is_a_protocol_class(self):
        assert issubclass(HandlerProtocol, Protocol)

    def test_mock_with_all_methods_satisfies(self):
        """A MagicMock with all required attrs satisfies the protocol."""
        m = MagicMock(spec=_CompleteHandler)
        assert isinstance(m, HandlerProtocol)


# =============================================================================
# RequestContext Tests
# =============================================================================


class TestRequestContext:
    """Test RequestContext TypedDict."""

    def test_create_empty(self):
        ctx: RequestContext = {}
        assert ctx == {}

    def test_create_with_user_id(self):
        ctx: RequestContext = {"user_id": "user-123"}
        assert ctx["user_id"] == "user-123"

    def test_create_with_all_fields(self):
        ctx: RequestContext = {
            "user_id": "u-1",
            "org_id": "org-1",
            "workspace_id": "ws-1",
            "trace_id": "trace-abc",
            "request_id": "req-xyz",
            "client_ip": "192.168.1.1",
            "user_agent": "Mozilla/5.0",
            "auth_method": "bearer",
            "permissions": ["read", "write"],
            "roles": ["admin"],
            "metadata": {"source": "api"},
        }
        assert len(ctx) == 11
        assert ctx["permissions"] == ["read", "write"]

    def test_annotations_contain_all_fields(self):
        hints = get_type_hints(RequestContext)
        expected = {
            "user_id",
            "org_id",
            "workspace_id",
            "trace_id",
            "request_id",
            "client_ip",
            "user_agent",
            "auth_method",
            "permissions",
            "roles",
            "metadata",
        }
        assert expected == set(hints.keys())


# =============================================================================
# PaginationParams Tests
# =============================================================================


class TestPaginationParams:
    """Test PaginationParams TypedDict."""

    def test_create_empty(self):
        params: PaginationParams = {}
        assert params == {}

    def test_create_with_limit_offset(self):
        params: PaginationParams = {"limit": 20, "offset": 0}
        assert params["limit"] == 20
        assert params["offset"] == 0

    def test_create_with_page(self):
        params: PaginationParams = {"page": 3}
        assert params["page"] == 3

    def test_create_with_cursor(self):
        params: PaginationParams = {"cursor": "eyJpZCI6MTAwfQ=="}
        assert params["cursor"].startswith("eyJ")

    def test_annotations(self):
        hints = get_type_hints(PaginationParams)
        assert "limit" in hints
        assert "offset" in hints
        assert "page" in hints
        assert "cursor" in hints
        assert len(hints) == 4


# =============================================================================
# FilterParams Tests
# =============================================================================


class TestFilterParams:
    """Test FilterParams TypedDict."""

    def test_create_empty(self):
        params: FilterParams = {}
        assert params == {}

    def test_create_with_status(self):
        params: FilterParams = {"status": "active"}
        assert params["status"] == "active"

    def test_create_with_date_range(self):
        params: FilterParams = {
            "created_after": "2026-01-01T00:00:00Z",
            "created_before": "2026-12-31T23:59:59Z",
        }
        assert "created_after" in params

    def test_create_with_tags_as_string(self):
        params: FilterParams = {"tags": "python,testing"}
        assert params["tags"] == "python,testing"

    def test_create_with_tags_as_list(self):
        params: FilterParams = {"tags": ["python", "testing"]}
        assert params["tags"] == ["python", "testing"]

    def test_create_with_ids_as_string(self):
        params: FilterParams = {"ids": "id1,id2,id3"}
        assert params["ids"] == "id1,id2,id3"

    def test_create_with_ids_as_list(self):
        params: FilterParams = {"ids": ["id1", "id2"]}
        assert len(params["ids"]) == 2

    def test_create_with_search(self):
        params: FilterParams = {"search": "rate limiter design"}
        assert params["search"] == "rate limiter design"

    def test_annotations_contain_11_fields(self):
        hints = get_type_hints(FilterParams)
        assert len(hints) == 11


# =============================================================================
# SortParams Tests
# =============================================================================


class TestSortParams:
    """Test SortParams TypedDict."""

    def test_create_empty(self):
        params: SortParams = {}
        assert params == {}

    def test_create_with_sort_by_and_order(self):
        params: SortParams = {"sort_by": "created_at", "sort_order": "desc"}
        assert params["sort_by"] == "created_at"
        assert params["sort_order"] == "desc"

    def test_create_with_order_by(self):
        params: SortParams = {"order_by": "name"}
        assert params["order_by"] == "name"

    def test_annotations(self):
        hints = get_type_hints(SortParams)
        assert "sort_by" in hints
        assert "sort_order" in hints
        assert "order_by" in hints
        assert len(hints) == 3


# =============================================================================
# QueryParams Tests
# =============================================================================


class TestQueryParams:
    """Test QueryParams combined TypedDict."""

    def test_create_empty(self):
        params: QueryParams = {}
        assert params == {}

    def test_create_with_pagination_fields(self):
        params: QueryParams = {"limit": 10, "offset": 20, "page": 3, "cursor": "abc"}
        assert params["limit"] == 10

    def test_create_with_filter_fields(self):
        params: QueryParams = {"status": "active", "agent": "claude", "search": "test"}
        assert params["agent"] == "claude"

    def test_create_with_sort_fields(self):
        params: QueryParams = {"sort_by": "name", "sort_order": "asc"}
        assert params["sort_order"] == "asc"

    def test_create_with_mixed_fields(self):
        params: QueryParams = {
            "limit": 5,
            "status": "completed",
            "sort_by": "created_at",
        }
        assert len(params) == 3

    def test_has_all_pagination_filter_sort_fields(self):
        hints = get_type_hints(QueryParams)
        # Pagination: limit, offset, page, cursor
        assert "limit" in hints
        assert "offset" in hints
        assert "page" in hints
        assert "cursor" in hints
        # Filter: status, agent, user_id, org_id, created_after, etc.
        assert "status" in hints
        assert "agent" in hints
        assert "search" in hints
        # Sort
        assert "sort_by" in hints
        assert "sort_order" in hints
        assert "order_by" in hints


# =============================================================================
# CreateDebateRequest Tests
# =============================================================================


class TestCreateDebateRequest:
    """Test CreateDebateRequest TypedDict."""

    def test_create_empty(self):
        req: CreateDebateRequest = {}
        assert req == {}

    def test_create_with_task(self):
        req: CreateDebateRequest = {"task": "Design a rate limiter"}
        assert req["task"] == "Design a rate limiter"

    def test_create_with_question(self):
        req: CreateDebateRequest = {"question": "Should we use Redis or Memcached?"}
        assert req["question"].startswith("Should")

    def test_create_full(self):
        req: CreateDebateRequest = {
            "task": "Design API gateway",
            "question": "How to handle auth?",
            "agents": ["claude", "gpt-4", "gemini"],
            "mode": "adversarial",
            "rounds": 5,
            "consensus": "unanimous",
        }
        assert len(req) == 6
        assert req["rounds"] == 5

    def test_agents_empty_list(self):
        req: CreateDebateRequest = {"agents": []}
        assert req["agents"] == []


# =============================================================================
# DebateUpdateRequest Tests
# =============================================================================


class TestDebateUpdateRequest:
    """Test DebateUpdateRequest TypedDict."""

    def test_create_empty(self):
        req: DebateUpdateRequest = {}
        assert req == {}

    def test_update_title(self):
        req: DebateUpdateRequest = {"title": "Updated Title"}
        assert req["title"] == "Updated Title"

    def test_update_status(self):
        req: DebateUpdateRequest = {"status": "archived"}
        assert req["status"] == "archived"

    def test_update_tags(self):
        req: DebateUpdateRequest = {"tags": ["important", "reviewed"]}
        assert len(req["tags"]) == 2


# =============================================================================
# DebateSummaryResponse Tests
# =============================================================================


class TestDebateSummaryResponse:
    """Test DebateSummaryResponse TypedDict."""

    def test_create_full(self):
        resp: DebateSummaryResponse = {
            "id": "debate-123",
            "task": "Test debate",
            "status": "active",
            "created_at": "2026-01-15T10:00:00Z",
            "agents": ["claude", "gpt-4"],
            "round_count": 3,
        }
        assert resp["id"] == "debate-123"
        assert len(resp["agents"]) == 2

    def test_has_6_required_fields(self):
        hints = get_type_hints(DebateSummaryResponse)
        assert len(hints) == 6


# =============================================================================
# DebateDetailResponse Tests
# =============================================================================


class TestDebateDetailResponse:
    """Test DebateDetailResponse TypedDict."""

    def test_create_without_consensus(self):
        resp: DebateDetailResponse = {
            "id": "d-1",
            "task": "Test",
            "status": "active",
            "created_at": "2026-01-01",
            "agents": ["claude"],
            "round_count": 2,
            "messages": [],
        }
        assert resp["messages"] == []

    def test_create_with_consensus(self):
        resp: DebateDetailResponse = {
            "id": "d-1",
            "task": "Test",
            "status": "completed",
            "created_at": "2026-01-01",
            "agents": ["claude"],
            "round_count": 3,
            "messages": [{"agent": "claude", "content": "My argument"}],
            "consensus": {"reached": True, "confidence": 0.95},
        }
        assert resp["consensus"]["reached"] is True

    def test_consensus_is_not_required(self):
        hints = get_type_hints(DebateDetailResponse, include_extras=True)
        assert "consensus" in hints


# =============================================================================
# DebateListResponse Tests
# =============================================================================


class TestDebateListResponse:
    """Test DebateListResponse TypedDict."""

    def test_create_empty_list(self):
        resp: DebateListResponse = {
            "debates": [],
            "total": 0,
            "offset": 0,
            "limit": 20,
        }
        assert resp["debates"] == []
        assert resp["total"] == 0

    def test_create_with_debates(self):
        resp: DebateListResponse = {
            "debates": [
                {
                    "id": "d-1",
                    "task": "Task 1",
                    "status": "active",
                    "created_at": "2026-01-01",
                    "agents": ["claude"],
                    "round_count": 1,
                },
            ],
            "total": 1,
            "offset": 0,
            "limit": 20,
        }
        assert len(resp["debates"]) == 1


# =============================================================================
# Fork Types Tests
# =============================================================================


class TestForkRequest:
    """Test ForkRequest TypedDict."""

    def test_create_minimal(self):
        req: ForkRequest = {"branch_point": 3}
        assert req["branch_point"] == 3

    def test_create_with_context(self):
        req: ForkRequest = {
            "branch_point": 5,
            "modified_context": "Different starting assumptions",
        }
        assert "modified_context" in req


class TestForkResponse:
    """Test ForkResponse TypedDict."""

    def test_create_full(self):
        resp: ForkResponse = {
            "fork_id": "fork-abc",
            "parent_id": "debate-123",
            "branch_point": 2,
            "created_at": "2026-02-01T12:00:00Z",
        }
        assert resp["fork_id"] == "fork-abc"
        assert resp["parent_id"] == "debate-123"


# =============================================================================
# Batch Types Tests
# =============================================================================


class TestBatchDebateItem:
    """Test BatchDebateItem TypedDict."""

    def test_create_minimal(self):
        item: BatchDebateItem = {"task": "Evaluate design"}
        assert item["task"] == "Evaluate design"

    def test_create_full(self):
        item: BatchDebateItem = {
            "task": "Design review",
            "question": "Is this architecture sound?",
            "agents": ["claude"],
            "rounds": 2,
            "mode": "collaborative",
        }
        assert len(item) == 5


class TestBatchSubmitRequest:
    """Test BatchSubmitRequest TypedDict."""

    def test_create_minimal(self):
        req: BatchSubmitRequest = {"items": []}
        assert req["items"] == []

    def test_create_with_webhook(self):
        req: BatchSubmitRequest = {
            "items": [{"task": "T1"}],
            "webhook_url": "https://hooks.example.com/batch",
            "max_parallel": 3,
        }
        assert req["max_parallel"] == 3


class TestBatchSubmitResponse:
    """Test BatchSubmitResponse TypedDict."""

    def test_create(self):
        resp: BatchSubmitResponse = {
            "batch_id": "batch-xyz",
            "total_items": 5,
            "status": "queued",
        }
        assert resp["batch_id"] == "batch-xyz"


class TestBatchStatusResponse:
    """Test BatchStatusResponse TypedDict."""

    def test_create_without_results(self):
        resp: BatchStatusResponse = {
            "batch_id": "b-1",
            "status": "running",
            "total": 10,
            "completed": 4,
            "failed": 1,
        }
        assert resp["completed"] == 4

    def test_create_with_results(self):
        resp: BatchStatusResponse = {
            "batch_id": "b-1",
            "status": "done",
            "total": 2,
            "completed": 2,
            "failed": 0,
            "results": [{"id": "r-1"}, {"id": "r-2"}],
        }
        assert len(resp["results"]) == 2


# =============================================================================
# Auth Types Tests
# =============================================================================


class TestUserRegisterRequest:
    """Test UserRegisterRequest TypedDict."""

    def test_create_minimal(self):
        req: UserRegisterRequest = {
            "email": "user@example.com",
            "password": "p@ssw0rd",
        }
        assert req["email"] == "user@example.com"

    def test_create_with_name(self):
        req: UserRegisterRequest = {
            "email": "user@example.com",
            "password": "secret",
            "name": "Alice Smith",
        }
        assert req["name"] == "Alice Smith"


class TestUserLoginRequest:
    """Test UserLoginRequest TypedDict."""

    def test_create(self):
        req: UserLoginRequest = {
            "email": "user@example.com",
            "password": "secret",
        }
        assert req["email"] == "user@example.com"
        assert req["password"] == "secret"


class TestAuthResponse:
    """Test AuthResponse TypedDict."""

    def test_create(self):
        resp: AuthResponse = {
            "token": "eyJhbGciOiJIUzI1NiJ9...",
            "user": {"id": "u-1", "email": "user@example.com"},
            "expires_at": "2026-12-31T23:59:59Z",
        }
        assert resp["token"].startswith("eyJ")


class TestUserResponse:
    """Test UserResponse TypedDict."""

    def test_create_minimal(self):
        resp: UserResponse = {
            "id": "u-1",
            "email": "user@example.com",
            "role": "member",
            "created_at": "2026-01-01T00:00:00Z",
        }
        assert resp["role"] == "member"

    def test_create_with_optionals(self):
        resp: UserResponse = {
            "id": "u-1",
            "email": "user@example.com",
            "name": "Alice",
            "role": "admin",
            "org_id": "org-1",
            "created_at": "2026-01-01T00:00:00Z",
        }
        assert resp["name"] == "Alice"
        assert resp["org_id"] == "org-1"


# =============================================================================
# Org Types Tests
# =============================================================================


class TestOrgCreateRequest:
    """Test OrgCreateRequest TypedDict."""

    def test_create_minimal(self):
        req: OrgCreateRequest = {"name": "Acme Corp"}
        assert req["name"] == "Acme Corp"

    def test_create_with_slug(self):
        req: OrgCreateRequest = {"name": "Acme Corp", "slug": "acme-corp"}
        assert req["slug"] == "acme-corp"


class TestOrgInviteRequest:
    """Test OrgInviteRequest TypedDict."""

    def test_create_minimal(self):
        req: OrgInviteRequest = {"email": "invite@example.com"}
        assert req["email"] == "invite@example.com"

    def test_create_with_role(self):
        req: OrgInviteRequest = {"email": "admin@example.com", "role": "admin"}
        assert req["role"] == "admin"


class TestOrgResponse:
    """Test OrgResponse TypedDict."""

    def test_create_minimal(self):
        resp: OrgResponse = {
            "id": "org-1",
            "name": "Acme",
            "slug": "acme",
            "owner_id": "u-1",
            "created_at": "2026-01-01",
        }
        assert resp["slug"] == "acme"

    def test_create_with_member_count(self):
        resp: OrgResponse = {
            "id": "org-1",
            "name": "Acme",
            "slug": "acme",
            "owner_id": "u-1",
            "created_at": "2026-01-01",
            "member_count": 42,
        }
        assert resp["member_count"] == 42


# =============================================================================
# Gauntlet Types Tests
# =============================================================================


class TestGauntletRunRequest:
    """Test GauntletRunRequest TypedDict."""

    def test_create_empty(self):
        req: GauntletRunRequest = {}
        assert req == {}

    def test_create_with_input(self):
        req: GauntletRunRequest = {"input_content": "def foo(): pass"}
        assert req["input_content"] == "def foo(): pass"

    def test_create_full(self):
        req: GauntletRunRequest = {
            "input_content": "code here",
            "input_type": "code",
            "agents": ["claude", "gpt-4"],
            "persona": "security-auditor",
            "profile": "comprehensive",
        }
        assert req["input_type"] == "code"


class TestGauntletResponse:
    """Test GauntletResponse TypedDict."""

    def test_create_without_result(self):
        resp: GauntletResponse = {"run_id": "g-1", "status": "running"}
        assert resp["status"] == "running"

    def test_create_with_result(self):
        resp: GauntletResponse = {
            "run_id": "g-1",
            "status": "completed",
            "result": {"verdict": "pass", "score": 0.98},
        }
        assert resp["result"]["verdict"] == "pass"


# =============================================================================
# Verification Types Tests
# =============================================================================


class TestVerificationRequest:
    """Test VerificationRequest TypedDict."""

    def test_create_minimal(self):
        req: VerificationRequest = {"claim": "The earth is round"}
        assert req["claim"] == "The earth is round"

    def test_create_with_context(self):
        req: VerificationRequest = {
            "claim": "This API handles 10k RPS",
            "context": "Based on load testing results",
        }
        assert "context" in req


class TestVerificationResponse:
    """Test VerificationResponse TypedDict."""

    def test_create(self):
        resp: VerificationResponse = {
            "verified": True,
            "confidence": 0.92,
            "evidence": [{"source": "test-results", "data": "..."}],
            "reasoning": "Evidence supports the claim",
        }
        assert resp["verified"] is True
        assert resp["confidence"] == 0.92


# =============================================================================
# Memory Types Tests
# =============================================================================


class TestMemoryCleanupRequest:
    """Test MemoryCleanupRequest TypedDict."""

    def test_create_empty(self):
        req: MemoryCleanupRequest = {}
        assert req == {}

    def test_create_full(self):
        req: MemoryCleanupRequest = {
            "tier": "fast",
            "archive": "true",
            "max_age_hours": 24.0,
        }
        assert req["tier"] == "fast"
        assert req["max_age_hours"] == 24.0


class TestMemoryEntry:
    """Test MemoryEntry TypedDict."""

    def test_create_without_expiration(self):
        entry: MemoryEntry = {
            "id": "mem-1",
            "tier": "medium",
            "content": "Debate context",
            "metadata": {"debate_id": "d-1"},
            "created_at": "2026-01-01T00:00:00Z",
        }
        assert entry["tier"] == "medium"

    def test_create_with_expiration(self):
        entry: MemoryEntry = {
            "id": "mem-2",
            "tier": "fast",
            "content": "Ephemeral",
            "metadata": {},
            "created_at": "2026-01-01T00:00:00Z",
            "expires_at": "2026-01-01T01:00:00Z",
        }
        assert "expires_at" in entry


# =============================================================================
# Agent Types Tests
# =============================================================================


class TestAgentConfigRequest:
    """Test AgentConfigRequest TypedDict."""

    def test_create_empty(self):
        req: AgentConfigRequest = {}
        assert req == {}

    def test_create_with_name(self):
        req: AgentConfigRequest = {"name": "claude"}
        assert req["name"] == "claude"

    def test_create_full(self):
        req: AgentConfigRequest = {
            "name": "custom-agent",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 4096,
            "system_prompt": "You are an expert.",
        }
        assert req["temperature"] == 0.7
        assert req["max_tokens"] == 4096


class TestAgentStatusResponse:
    """Test AgentStatusResponse TypedDict."""

    def test_create_minimal(self):
        resp: AgentStatusResponse = {
            "name": "claude",
            "status": "available",
            "model": "claude-3-opus",
        }
        assert resp["status"] == "available"

    def test_create_with_optionals(self):
        resp: AgentStatusResponse = {
            "name": "claude",
            "status": "busy",
            "model": "claude-3-opus",
            "elo_rating": 1850,
            "total_debates": 150,
        }
        assert resp["elo_rating"] == 1850


# =============================================================================
# Probe Types Tests
# =============================================================================


class TestProbeRunRequest:
    """Test ProbeRunRequest TypedDict."""

    def test_create_empty(self):
        req: ProbeRunRequest = {}
        assert req == {}

    def test_create_full(self):
        req: ProbeRunRequest = {
            "agent_name": "claude",
            "probe_types": ["consistency", "bias"],
            "probes_per_type": 5,
            "model_type": "chat",
        }
        assert req["agent_name"] == "claude"
        assert len(req["probe_types"]) == 2


class TestProbeResultResponse:
    """Test ProbeResultResponse TypedDict."""

    def test_create(self):
        resp: ProbeResultResponse = {
            "probe_id": "probe-1",
            "agent_name": "claude",
            "results": [{"type": "consistency", "score": 0.95}],
            "summary": {"overall_score": 0.95, "pass": True},
        }
        assert resp["probe_id"] == "probe-1"


# =============================================================================
# Social/Sharing Types Tests
# =============================================================================


class TestSocialPublishRequest:
    """Test SocialPublishRequest TypedDict."""

    def test_create_empty(self):
        req: SocialPublishRequest = {}
        assert req == {}

    def test_create_full(self):
        req: SocialPublishRequest = {
            "include_audio_link": "true",
            "thread_mode": "threaded",
            "title": "Rate Limiter Design Debate",
            "description": "A deep discussion",
            "tags": ["engineering", "design"],
        }
        assert req["title"] == "Rate Limiter Design Debate"


class TestShareUpdateRequest:
    """Test ShareUpdateRequest TypedDict."""

    def test_create_empty(self):
        req: ShareUpdateRequest = {}
        assert req == {}

    def test_create_full(self):
        req: ShareUpdateRequest = {
            "visibility": "public",
            "expires_in_hours": 48,
            "allow_comments": "true",
            "allow_forking": "false",
        }
        assert req["visibility"] == "public"
        assert req["expires_in_hours"] == 48


class TestShareResponse:
    """Test ShareResponse TypedDict."""

    def test_create_without_expiration(self):
        resp: ShareResponse = {
            "share_id": "share-abc",
            "debate_id": "d-1",
            "visibility": "team",
            "url": "https://app.aragora.com/share/share-abc",
        }
        assert resp["url"].endswith("share-abc")

    def test_create_with_expiration(self):
        resp: ShareResponse = {
            "share_id": "share-abc",
            "debate_id": "d-1",
            "visibility": "public",
            "url": "https://app.aragora.com/share/share-abc",
            "expires_at": "2026-03-01T00:00:00Z",
        }
        assert "expires_at" in resp


# =============================================================================
# Billing Types Tests
# =============================================================================


class TestCheckoutSessionRequest:
    """Test CheckoutSessionRequest TypedDict."""

    def test_create(self):
        req: CheckoutSessionRequest = {
            "tier": "professional",
            "success_url": "https://example.com/success",
            "cancel_url": "https://example.com/cancel",
        }
        assert req["tier"] == "professional"


class TestCheckoutSessionResponse:
    """Test CheckoutSessionResponse TypedDict."""

    def test_create(self):
        resp: CheckoutSessionResponse = {
            "session_id": "cs_abc123",
            "checkout_url": "https://checkout.stripe.com/abc",
        }
        assert resp["checkout_url"].startswith("https://")


class TestUsageResponse:
    """Test UsageResponse TypedDict."""

    def test_create(self):
        resp: UsageResponse = {
            "debates_used": 42,
            "debates_limit": 100,
            "tokens_used": 500000,
            "tokens_limit": 1000000,
            "period_start": "2026-02-01T00:00:00Z",
            "period_end": "2026-02-28T23:59:59Z",
        }
        assert resp["debates_used"] == 42
        assert resp["tokens_limit"] == 1000000


# =============================================================================
# Plugin Types Tests
# =============================================================================


class TestPluginRunRequest:
    """Test PluginRunRequest TypedDict."""

    def test_create_empty(self):
        req: PluginRunRequest = {}
        assert req == {}

    def test_create_full(self):
        req: PluginRunRequest = {
            "input": "analyze this code",
            "config": '{"timeout": 30}',
            "working_dir": "/tmp/plugin",
        }
        assert req["working_dir"] == "/tmp/plugin"


class TestPluginInstallRequest:
    """Test PluginInstallRequest TypedDict."""

    def test_create_empty(self):
        req: PluginInstallRequest = {}
        assert req == {}

    def test_create_full(self):
        req: PluginInstallRequest = {"config": '{"key": "val"}', "enabled": "true"}
        assert req["enabled"] == "true"


class TestPluginResponse:
    """Test PluginResponse TypedDict."""

    def test_create(self):
        resp: PluginResponse = {
            "name": "code-analyzer",
            "version": "1.2.0",
            "enabled": True,
            "status": "installed",
        }
        assert resp["version"] == "1.2.0"
        assert resp["enabled"] is True


# =============================================================================
# Metrics Types Tests
# =============================================================================


class TestConvergenceMetrics:
    """Test ConvergenceMetrics TypedDict."""

    def test_create_minimal(self):
        m: ConvergenceMetrics = {
            "status": "converged",
            "similarity": 0.95,
            "per_agent": {"claude": 0.98, "gpt-4": 0.92},
        }
        assert m["status"] == "converged"

    def test_create_with_rounds(self):
        m: ConvergenceMetrics = {
            "status": "converged",
            "similarity": 0.99,
            "per_agent": {},
            "rounds_to_converge": 4,
        }
        assert m["rounds_to_converge"] == 4


class TestPhaseMetrics:
    """Test PhaseMetrics TypedDict."""

    def test_create_minimal(self):
        m: PhaseMetrics = {
            "phase_name": "proposal",
            "duration_ms": 1200.5,
            "status": "completed",
        }
        assert m["duration_ms"] == 1200.5

    def test_create_with_error(self):
        m: PhaseMetrics = {
            "phase_name": "critique",
            "duration_ms": 500.0,
            "status": "failed",
            "error": "Agent timeout",
        }
        assert m["error"] == "Agent timeout"


class TestDebateMetrics:
    """Test DebateMetrics TypedDict."""

    def test_create_minimal(self):
        m: DebateMetrics = {
            "debate_id": "d-1",
            "total_duration_ms": 15000.0,
            "rounds_used": 3,
            "messages_count": 18,
            "consensus_reached": True,
            "confidence": 0.87,
        }
        assert m["consensus_reached"] is True

    def test_create_with_convergence_and_phases(self):
        m: DebateMetrics = {
            "debate_id": "d-2",
            "total_duration_ms": 20000.0,
            "rounds_used": 5,
            "messages_count": 30,
            "consensus_reached": False,
            "confidence": 0.45,
            "convergence": {
                "status": "diverging",
                "similarity": 0.3,
                "per_agent": {},
            },
            "phases": [
                {"phase_name": "proposal", "duration_ms": 3000.0, "status": "completed"},
            ],
        }
        assert m["convergence"]["status"] == "diverging"
        assert len(m["phases"]) == 1


class TestAgentPerformanceMetrics:
    """Test AgentPerformanceMetrics TypedDict."""

    def test_create_minimal(self):
        m: AgentPerformanceMetrics = {
            "agent_name": "claude",
            "elo_rating": 1850.0,
            "win_rate": 0.72,
            "debates_participated": 100,
            "avg_confidence": 0.88,
        }
        assert m["agent_name"] == "claude"

    def test_create_with_calibration(self):
        m: AgentPerformanceMetrics = {
            "agent_name": "gpt-4",
            "elo_rating": 1800.0,
            "win_rate": 0.68,
            "debates_participated": 80,
            "avg_confidence": 0.85,
            "calibration_score": 0.91,
        }
        assert m["calibration_score"] == 0.91


class TestDashboardSummary:
    """Test DashboardSummary TypedDict."""

    def test_create(self):
        s: DashboardSummary = {
            "total_debates": 1000,
            "active_debates": 12,
            "total_agents": 8,
            "avg_confidence": 0.82,
            "consensus_rate": 0.88,
        }
        assert s["total_debates"] == 1000
        assert s["consensus_rate"] == 0.88


class TestDashboardResponse:
    """Test DashboardResponse TypedDict."""

    def test_create(self):
        resp: DashboardResponse = {
            "summary": {
                "total_debates": 100,
                "active_debates": 3,
                "total_agents": 5,
                "avg_confidence": 0.85,
                "consensus_rate": 0.9,
            },
            "recent_activity": [{"type": "debate_completed", "id": "d-1"}],
            "agent_performance": [
                {
                    "agent_name": "claude",
                    "elo_rating": 1800.0,
                    "win_rate": 0.7,
                    "debates_participated": 50,
                    "avg_confidence": 0.85,
                },
            ],
            "top_domains": [{"domain": "engineering", "count": 42}],
            "generated_at": 1708000000.0,
        }
        assert resp["generated_at"] == 1708000000.0
        assert len(resp["agent_performance"]) == 1


# =============================================================================
# Error Types Tests
# =============================================================================


class TestErrorDetail:
    """Test ErrorDetail TypedDict."""

    def test_create_empty(self):
        d: ErrorDetail = {}
        assert d == {}

    def test_create_with_code_and_message(self):
        d: ErrorDetail = {
            "code": "NOT_FOUND",
            "message": "Resource not found",
        }
        assert d["code"] == "NOT_FOUND"

    def test_create_full(self):
        d: ErrorDetail = {
            "code": "VALIDATION_ERROR",
            "message": "Invalid input",
            "details": {"field": "email", "reason": "invalid format"},
            "trace_id": "trace-abc123",
            "suggestion": "Use a valid email address",
        }
        assert d["trace_id"] == "trace-abc123"


class TestErrorResponse:
    """Test ErrorResponse TypedDict."""

    def test_create_with_string_error(self):
        resp: ErrorResponse = {"error": "Something went wrong"}
        assert resp["error"] == "Something went wrong"

    def test_create_with_error_detail(self):
        resp: ErrorResponse = {"error": {"code": "AUTH_REQUIRED", "message": "Login required"}}
        assert resp["error"]["code"] == "AUTH_REQUIRED"

    def test_create_empty(self):
        resp: ErrorResponse = {}
        assert resp == {}


# =============================================================================
# Common Response Wrapper Tests
# =============================================================================


class TestPaginatedResponse:
    """Test PaginatedResponse TypedDict."""

    def test_create(self):
        resp: PaginatedResponse = {
            "items": [1, 2, 3],
            "total": 100,
            "offset": 0,
            "limit": 20,
            "has_more": True,
        }
        assert resp["has_more"] is True
        assert resp["total"] == 100

    def test_empty_items(self):
        resp: PaginatedResponse = {
            "items": [],
            "total": 0,
            "offset": 0,
            "limit": 20,
            "has_more": False,
        }
        assert resp["items"] == []
        assert resp["has_more"] is False


class TestStatusResponse:
    """Test StatusResponse TypedDict."""

    def test_create_status_only(self):
        resp: StatusResponse = {"status": "ok"}
        assert resp["status"] == "ok"

    def test_create_with_message(self):
        resp: StatusResponse = {"status": "error", "message": "Database unavailable"}
        assert resp["message"] == "Database unavailable"


# =============================================================================
# __all__ Export Tests
# =============================================================================


class TestAllExports:
    """Test that __all__ is comprehensive and all items are importable."""

    def test_all_exists(self):
        from aragora.server.handlers import types

        assert hasattr(types, "__all__")

    def test_all_is_list(self):
        from aragora.server.handlers import types

        assert isinstance(types.__all__, list)

    def test_all_items_are_strings(self):
        from aragora.server.handlers import types

        for name in types.__all__:
            assert isinstance(name, str)

    def test_all_items_importable(self):
        from aragora.server.handlers import types

        for name in types.__all__:
            assert hasattr(types, name), f"{name!r} in __all__ but not accessible"

    def test_handler_protocol_in_all(self):
        from aragora.server.handlers import types

        assert "HandlerProtocol" in types.__all__

    def test_request_context_in_all(self):
        from aragora.server.handlers import types

        assert "RequestContext" in types.__all__

    def test_response_type_in_all(self):
        from aragora.server.handlers import types

        assert "ResponseType" in types.__all__

    def test_handler_function_aliases_in_all(self):
        from aragora.server.handlers import types

        assert "HandlerFunction" in types.__all__
        assert "AsyncHandlerFunction" in types.__all__
        assert "MaybeAsyncHandlerFunction" in types.__all__

    def test_middleware_aliases_in_all(self):
        from aragora.server.handlers import types

        assert "MiddlewareFunction" in types.__all__
        assert "AsyncMiddlewareFunction" in types.__all__
        assert "MaybeAsyncMiddlewareFunction" in types.__all__
        assert "MiddlewareFactory" in types.__all__

    def test_parameter_types_in_all(self):
        from aragora.server.handlers import types

        assert "PaginationParams" in types.__all__
        assert "FilterParams" in types.__all__
        assert "SortParams" in types.__all__
        assert "QueryParams" in types.__all__

    def test_debate_types_in_all(self):
        from aragora.server.handlers import types

        for name in [
            "CreateDebateRequest",
            "DebateUpdateRequest",
            "DebateSummaryResponse",
            "DebateDetailResponse",
            "DebateListResponse",
        ]:
            assert name in types.__all__

    def test_all_has_no_duplicates(self):
        from aragora.server.handlers import types

        assert len(types.__all__) == len(set(types.__all__))

    def test_all_count(self):
        """Verify the expected number of exports."""
        from aragora.server.handlers import types

        # At least 40 exports based on module contents
        assert len(types.__all__) >= 40


# =============================================================================
# Type Alias Accessibility Tests
# =============================================================================


class TestTypeAliases:
    """Test that type aliases are accessible and well-defined."""

    def test_response_type_is_accessible(self):
        assert ResponseType is not None

    def test_handler_function_is_accessible(self):
        assert HandlerFunction is not None

    def test_async_handler_function_is_accessible(self):
        assert AsyncHandlerFunction is not None

    def test_maybe_async_handler_function_is_accessible(self):
        assert MaybeAsyncHandlerFunction is not None

    def test_middleware_function_is_accessible(self):
        assert MiddlewareFunction is not None

    def test_async_middleware_function_is_accessible(self):
        assert AsyncMiddlewareFunction is not None

    def test_maybe_async_middleware_function_is_accessible(self):
        assert MaybeAsyncMiddlewareFunction is not None

    def test_middleware_factory_is_accessible(self):
        assert MiddlewareFactory is not None


# =============================================================================
# Type Introspection Tests
# =============================================================================


class TestTypeIntrospection:
    """Test type hints and annotations on all TypedDicts."""

    def test_all_typeddicts_have_annotations(self):
        typeddicts = [
            RequestContext,
            PaginationParams,
            FilterParams,
            SortParams,
            QueryParams,
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
            ProbeRunRequest,
            ProbeResultResponse,
            SocialPublishRequest,
            ShareUpdateRequest,
            ShareResponse,
            CheckoutSessionRequest,
            CheckoutSessionResponse,
            UsageResponse,
            PluginRunRequest,
            PluginInstallRequest,
            PluginResponse,
            ConvergenceMetrics,
            PhaseMetrics,
            DebateMetrics,
            AgentPerformanceMetrics,
            DashboardSummary,
            DashboardResponse,
            ErrorDetail,
            ErrorResponse,
            PaginatedResponse,
            StatusResponse,
        ]
        for td in typeddicts:
            assert hasattr(td, "__annotations__"), f"{td.__name__} missing __annotations__"

    def test_debate_summary_has_6_fields(self):
        hints = get_type_hints(DebateSummaryResponse)
        assert len(hints) == 6

    def test_paginated_response_has_5_fields(self):
        hints = get_type_hints(PaginatedResponse)
        assert len(hints) == 5

    def test_usage_response_has_6_fields(self):
        hints = get_type_hints(UsageResponse)
        assert len(hints) == 6


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_strings_in_debate_request(self):
        req: CreateDebateRequest = {"task": "", "question": ""}
        assert req["task"] == ""

    def test_unicode_content(self):
        req: CreateDebateRequest = {"task": "Evaluate design \u2192 \u2714"}
        assert "\u2192" in req["task"]

    def test_very_long_string(self):
        long_str = "x" * 100000
        req: GauntletRunRequest = {"input_content": long_str}
        assert len(req["input_content"]) == 100000

    def test_nested_metadata_in_memory_entry(self):
        entry: MemoryEntry = {
            "id": "m-1",
            "tier": "slow",
            "content": "deep",
            "metadata": {"a": {"b": {"c": {"d": 42}}}},
            "created_at": "2026-01-01",
        }
        assert entry["metadata"]["a"]["b"]["c"]["d"] == 42

    def test_large_agents_list(self):
        agents = [f"agent-{i}" for i in range(100)]
        req: CreateDebateRequest = {"agents": agents}
        assert len(req["agents"]) == 100

    def test_negative_numbers(self):
        resp: UsageResponse = {
            "debates_used": -1,
            "debates_limit": 0,
            "tokens_used": -100,
            "tokens_limit": 0,
            "period_start": "",
            "period_end": "",
        }
        # TypedDicts don't enforce value constraints, only structural
        assert resp["debates_used"] == -1

    def test_float_precision_in_metrics(self):
        m: ConvergenceMetrics = {
            "status": "converged",
            "similarity": 0.9999999999999999,
            "per_agent": {"a": 1e-10},
        }
        assert m["similarity"] > 0.99

    def test_special_chars_in_string_fields(self):
        req: CreateDebateRequest = {
            "task": "Test <script>alert('xss')</script> & \"quotes\"",
        }
        assert "<script>" in req["task"]
