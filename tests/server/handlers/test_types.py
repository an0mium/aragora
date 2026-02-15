"""
Tests for aragora.server.handlers.types.

Tests cover:
1. HandlerProtocol runtime checkable behavior
2. TypedDict instantiation (ensuring type definitions are valid)
3. __all__ exports list completeness
4. Type alias accessibility
"""

from __future__ import annotations

from typing import Any
from collections.abc import Awaitable

import pytest

from aragora.server.handlers.types import (
    # Protocol
    HandlerProtocol,
    # Request context
    RequestContext,
    # Response type aliases
    ResponseType,
    # Handler function type aliases
    HandlerFunction,
    AsyncHandlerFunction,
    MaybeAsyncHandlerFunction,
    # Middleware function type aliases
    MiddlewareFunction,
    AsyncMiddlewareFunction,
    MaybeAsyncMiddlewareFunction,
    MiddlewareFactory,
    # Parameter types
    PaginationParams,
    FilterParams,
    SortParams,
    QueryParams,
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
    # Probe types
    ProbeRunRequest,
    ProbeResultResponse,
    # Social types
    SocialPublishRequest,
    ShareUpdateRequest,
    ShareResponse,
    # Billing types
    CheckoutSessionRequest,
    CheckoutSessionResponse,
    UsageResponse,
    # Plugin types
    PluginRunRequest,
    PluginInstallRequest,
    PluginResponse,
    # Metrics types
    ConvergenceMetrics,
    PhaseMetrics,
    DebateMetrics,
    AgentPerformanceMetrics,
    DashboardSummary,
    DashboardResponse,
    # Error types
    ErrorDetail,
    ErrorResponse,
    # Common types
    PaginatedResponse,
    StatusResponse,
)


# =============================================================================
# Helper Classes for Protocol Testing
# =============================================================================


class _FullHandler:
    """A class implementing all HandlerProtocol methods."""

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


class _IncompleteHandler:
    """A class missing required methods."""

    def handle(self, path, query_params, handler):
        return None


class _Empty:
    """A class with no handler methods."""

    pass


# =============================================================================
# HandlerProtocol Tests
# =============================================================================


class TestHandlerProtocol:
    """Test HandlerProtocol runtime_checkable behavior."""

    def test_full_handler_satisfies_protocol(self):
        handler = _FullHandler()
        assert isinstance(handler, HandlerProtocol)

    def test_incomplete_handler_does_not_satisfy(self):
        handler = _IncompleteHandler()
        assert not isinstance(handler, HandlerProtocol)

    def test_empty_class_does_not_satisfy(self):
        obj = _Empty()
        assert not isinstance(obj, HandlerProtocol)

    def test_none_does_not_satisfy(self):
        assert not isinstance(None, HandlerProtocol)

    def test_string_does_not_satisfy(self):
        assert not isinstance("not a handler", HandlerProtocol)


# =============================================================================
# TypedDict Instantiation Tests
# =============================================================================


class TestRequestContextTypedDict:
    """Test RequestContext TypedDict."""

    def test_create_minimal(self):
        ctx: RequestContext = {}
        assert isinstance(ctx, dict)

    def test_create_with_fields(self):
        ctx: RequestContext = {
            "user_id": "u-1",
            "org_id": "org-1",
            "trace_id": "trace-abc",
        }
        assert ctx["user_id"] == "u-1"
        assert ctx["org_id"] == "org-1"


class TestPaginationParams:
    """Test PaginationParams TypedDict."""

    def test_create_with_defaults(self):
        params: PaginationParams = {"limit": 20, "offset": 0}
        assert params["limit"] == 20

    def test_create_with_cursor(self):
        params: PaginationParams = {"cursor": "abc123"}
        assert params["cursor"] == "abc123"


class TestFilterParams:
    """Test FilterParams TypedDict."""

    def test_create_with_status(self):
        params: FilterParams = {"status": "active"}
        assert params["status"] == "active"

    def test_create_with_tags_as_list(self):
        params: FilterParams = {"tags": ["tag1", "tag2"]}
        assert params["tags"] == ["tag1", "tag2"]

    def test_create_with_search(self):
        params: FilterParams = {"search": "rate limiter"}
        assert params["search"] == "rate limiter"


class TestSortParams:
    """Test SortParams TypedDict."""

    def test_create(self):
        params: SortParams = {"sort_by": "created_at", "sort_order": "desc"}
        assert params["sort_by"] == "created_at"


class TestDebateRequestTypes:
    """Test debate request TypedDicts."""

    def test_create_debate_request(self):
        req: CreateDebateRequest = {
            "task": "Design a rate limiter",
            "agents": ["claude", "gpt-4"],
            "rounds": 3,
        }
        assert req["task"] == "Design a rate limiter"
        assert len(req["agents"]) == 2

    def test_debate_update_request(self):
        req: DebateUpdateRequest = {"status": "archived", "tags": ["important"]}
        assert req["status"] == "archived"


class TestDebateResponseTypes:
    """Test debate response TypedDicts."""

    def test_debate_summary(self):
        resp: DebateSummaryResponse = {
            "id": "d-1",
            "task": "Test",
            "status": "active",
            "created_at": "2026-01-01",
            "agents": ["claude"],
            "round_count": 3,
        }
        assert resp["id"] == "d-1"

    def test_debate_detail(self):
        resp: DebateDetailResponse = {
            "id": "d-1",
            "task": "Test",
            "status": "completed",
            "created_at": "2026-01-01",
            "agents": ["claude"],
            "round_count": 3,
            "messages": [{"agent": "claude", "content": "Hello"}],
        }
        assert len(resp["messages"]) == 1

    def test_debate_list(self):
        resp: DebateListResponse = {
            "debates": [],
            "total": 0,
            "offset": 0,
            "limit": 20,
        }
        assert resp["total"] == 0


class TestAuthTypes:
    """Test authentication TypedDicts."""

    def test_register_request(self):
        req: UserRegisterRequest = {"email": "test@example.com", "password": "secret"}
        assert req["email"] == "test@example.com"

    def test_login_request(self):
        req: UserLoginRequest = {"email": "test@example.com", "password": "secret"}
        assert req["email"] == "test@example.com"

    def test_auth_response(self):
        resp: AuthResponse = {
            "token": "jwt-token",
            "user": {"id": "u-1"},
            "expires_at": "2026-12-31",
        }
        assert resp["token"] == "jwt-token"


class TestOrgTypes:
    """Test organization TypedDicts."""

    def test_org_create(self):
        req: OrgCreateRequest = {"name": "Test Org"}
        assert req["name"] == "Test Org"

    def test_org_invite(self):
        req: OrgInviteRequest = {"email": "invite@example.com", "role": "member"}
        assert req["role"] == "member"

    def test_org_response(self):
        resp: OrgResponse = {
            "id": "org-1",
            "name": "Test Org",
            "slug": "test-org",
            "owner_id": "u-1",
            "created_at": "2026-01-01",
        }
        assert resp["slug"] == "test-org"


class TestMetricsTypes:
    """Test internal metrics TypedDicts."""

    def test_convergence_metrics(self):
        m: ConvergenceMetrics = {
            "status": "converged",
            "similarity": 0.95,
            "per_agent": {"claude": 0.98, "gpt-4": 0.92},
        }
        assert m["similarity"] == 0.95

    def test_phase_metrics(self):
        m: PhaseMetrics = {
            "phase_name": "proposal",
            "duration_ms": 1500.0,
            "status": "completed",
        }
        assert m["phase_name"] == "proposal"

    def test_debate_metrics(self):
        m: DebateMetrics = {
            "debate_id": "d-1",
            "total_duration_ms": 5000.0,
            "rounds_used": 3,
            "messages_count": 12,
            "consensus_reached": True,
            "confidence": 0.87,
        }
        assert m["consensus_reached"] is True

    def test_dashboard_summary(self):
        s: DashboardSummary = {
            "total_debates": 100,
            "active_debates": 5,
            "total_agents": 10,
            "avg_confidence": 0.85,
            "consensus_rate": 0.92,
        }
        assert s["total_debates"] == 100


class TestErrorTypes:
    """Test error response TypedDicts."""

    def test_error_detail(self):
        d: ErrorDetail = {
            "code": "NOT_FOUND",
            "message": "Debate not found",
        }
        assert d["code"] == "NOT_FOUND"

    def test_error_response(self):
        r: ErrorResponse = {"error": "Something went wrong"}
        assert r["error"] == "Something went wrong"


class TestCommonTypes:
    """Test common response TypedDicts."""

    def test_paginated_response(self):
        r: PaginatedResponse = {
            "items": [1, 2, 3],
            "total": 100,
            "offset": 0,
            "limit": 20,
            "has_more": True,
        }
        assert r["has_more"] is True
        assert r["total"] == 100

    def test_status_response(self):
        r: StatusResponse = {"status": "ok"}
        assert r["status"] == "ok"

    def test_status_response_with_message(self):
        r: StatusResponse = {"status": "error", "message": "Something failed"}
        assert r["message"] == "Something failed"


# =============================================================================
# __all__ Export Tests
# =============================================================================


class TestExports:
    """Test that __all__ contains expected exports."""

    def test_all_list_exists(self):
        from aragora.server.handlers import types

        assert hasattr(types, "__all__")

    def test_handler_protocol_exported(self):
        from aragora.server.handlers import types

        assert "HandlerProtocol" in types.__all__

    def test_request_context_exported(self):
        from aragora.server.handlers import types

        assert "RequestContext" in types.__all__

    def test_create_debate_request_exported(self):
        from aragora.server.handlers import types

        assert "CreateDebateRequest" in types.__all__

    def test_error_response_exported(self):
        from aragora.server.handlers import types

        assert "ErrorResponse" in types.__all__

    def test_paginated_response_exported(self):
        from aragora.server.handlers import types

        assert "PaginatedResponse" in types.__all__

    def test_all_items_importable(self):
        """Every name in __all__ should be importable from the module."""
        from aragora.server.handlers import types

        for name in types.__all__:
            assert hasattr(types, name), f"{name!r} in __all__ but not importable"
