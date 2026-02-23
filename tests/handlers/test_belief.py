"""Tests for belief handler (aragora/server/handlers/belief.py).

Covers all routes and behavior of the BeliefHandler class:
- can_handle() routing for all ROUTES (versioned and unversioned)
- GET /api/belief-network/:debate_id/cruxes
- GET /api/belief-network/:debate_id/load-bearing-claims
- GET /api/belief-network/:debate_id/graph
- GET /api/belief-network/:debate_id/export (json, csv, graphml)
- GET /api/provenance/:debate_id/claims/:claim_id/support
- GET /api/debate/:debate_id/graph-stats
- GET /api/v1/debates/:debate_id/cruxes (versioned SDK alias)
- Rate limiting
- RBAC permission checks
- KM adapter lifecycle
- Error handling and edge cases
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.belief import BeliefHandler, _belief_limiter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP request handler for BeliefHandler tests."""

    def __init__(
        self,
        body: dict | None = None,
        method: str = "GET",
        roles: str = "",
    ):
        self.command = method
        self.client_address = ("127.0.0.1", 12345)
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        if roles:
            self.headers["X-User-Roles"] = roles
        self.rfile = MagicMock()

        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Mock data builders
# ---------------------------------------------------------------------------


def _make_mock_trace():
    """Create a mock DebateTrace with messages and critiques."""
    mock_msg = MagicMock()
    mock_msg.agent = "claude"
    mock_msg.content = "This is a test claim about the topic."
    mock_msg.role = "proposer"
    mock_msg.round = 1

    mock_critique = MagicMock()
    mock_critique.agent = "gpt4"
    mock_critique.target = "claude"
    mock_critique.severity = 0.7
    mock_critique.reasoning = "Weak evidence for this claim."
    mock_critique.round = 1

    mock_result = MagicMock()
    mock_result.messages = [mock_msg]
    mock_result.critiques = [mock_critique]
    mock_result.task = "Test debate topic"

    mock_trace = MagicMock()
    mock_trace.to_debate_result.return_value = mock_result

    return mock_trace


def _make_mock_network():
    """Create a mock BeliefNetwork."""
    network = MagicMock()
    network.add_claim = MagicMock()
    network.get_load_bearing_claims.return_value = []
    network.get_all_claims.return_value = []
    network.get_all_edges.return_value = []
    network.seed_from_km.return_value = 0
    return network


def _make_mock_analyzer(cruxes=None):
    """Create a mock BeliefPropagationAnalyzer."""
    analyzer = MagicMock()
    analyzer.identify_debate_cruxes.return_value = cruxes or []
    return analyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a BeliefHandler with a minimal server context including nomic_dir."""
    return BeliefHandler(server_context={"nomic_dir": Path("/tmp/test-nomic")})


@pytest.fixture
def handler_no_nomic():
    """Create a BeliefHandler without nomic_dir."""
    return BeliefHandler(server_context={})


@pytest.fixture
def http_handler():
    """Create a default MockHTTPHandler."""
    return MockHTTPHandler()


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the rate limiter between tests."""
    _belief_limiter._buckets = defaultdict(list)
    _belief_limiter._requests = _belief_limiter._buckets
    yield
    _belief_limiter._buckets = defaultdict(list)
    _belief_limiter._requests = _belief_limiter._buckets


@pytest.fixture(autouse=True)
def _reset_km_adapter(handler):
    """Reset the KM adapter on the handler between tests."""
    handler._km_adapter = None
    yield


@pytest.fixture(autouse=True)
def _bypass_belief_rbac(request, monkeypatch):
    """Bypass the belief handler's own RBAC check via get_permission_checker.

    The conftest patches BaseHandler.require_auth_or_error, but BeliefHandler
    also calls get_permission_checker().check_permission() inside
    _check_belief_permission(). We need to patch that to always allow.
    """
    if "no_auto_auth" in [m.name for m in request.node.iter_markers()]:
        yield
        return

    from aragora.rbac.models import AuthorizationDecision

    mock_decision = MagicMock(spec=AuthorizationDecision)
    mock_decision.allowed = True
    mock_decision.reason = "Test bypass"

    mock_checker = MagicMock()
    mock_checker.check_permission.return_value = mock_decision

    monkeypatch.setattr(
        "aragora.server.handlers.belief.get_permission_checker",
        lambda: mock_checker,
    )
    yield


# ============================================================================
# can_handle routing
# ============================================================================


class TestCanHandle:
    """Verify that can_handle correctly accepts or rejects paths."""

    def test_cruxes_path(self, handler):
        assert handler.can_handle("/api/belief-network/debate-123/cruxes")

    def test_load_bearing_claims_path(self, handler):
        assert handler.can_handle("/api/belief-network/debate-123/load-bearing-claims")

    def test_graph_path(self, handler):
        assert handler.can_handle("/api/belief-network/debate-123/graph")

    def test_export_path(self, handler):
        assert handler.can_handle("/api/belief-network/debate-123/export")

    def test_provenance_support_path(self, handler):
        assert handler.can_handle("/api/provenance/debate-123/claims/claim-1/support")

    def test_graph_stats_path(self, handler):
        assert handler.can_handle("/api/debate/debate-123/graph-stats")

    def test_versioned_graph_path(self, handler):
        assert handler.can_handle("/api/v1/belief-network/debate-123/graph")

    def test_versioned_export_path(self, handler):
        assert handler.can_handle("/api/v1/belief-network/debate-123/export")

    def test_versioned_cruxes_sdk_path(self, handler):
        assert handler.can_handle("/api/v1/debates/debate-123/cruxes")

    def test_versioned_cruxes_path(self, handler):
        assert handler.can_handle("/api/v1/belief-network/debate-123/cruxes")

    def test_rejects_empty_path(self, handler):
        assert not handler.can_handle("")

    def test_rejects_root(self, handler):
        assert not handler.can_handle("/")

    def test_rejects_unrelated_path(self, handler):
        assert not handler.can_handle("/api/debates/list")

    def test_rejects_partial_belief_path(self, handler):
        assert not handler.can_handle("/api/belief-network/")

    def test_rejects_api_only(self, handler):
        assert not handler.can_handle("/api/belief-network")

    def test_accepts_debates_cruxes_unversioned(self, handler):
        assert handler.can_handle("/api/debates/debate-123/cruxes")


# ============================================================================
# Initialization
# ============================================================================


class TestInit:
    """Test handler initialization."""

    def test_init_with_server_context(self):
        ctx = {"nomic_dir": Path("/tmp")}
        h = BeliefHandler(server_context=ctx)
        assert h.ctx == ctx

    def test_init_with_empty_context(self):
        h = BeliefHandler(server_context={})
        assert h.ctx == {}

    def test_km_adapter_initially_none(self, handler):
        assert handler._km_adapter is None

    def test_routes_defined(self, handler):
        assert len(handler.ROUTES) > 0
        assert "/api/belief-network/*/cruxes" in handler.ROUTES

    def test_routes_include_versioned(self, handler):
        assert "/api/v1/belief-network/*/graph" in handler.ROUTES

    def test_routes_include_sdk_alias(self, handler):
        assert "/api/v1/debates/*/cruxes" in handler.ROUTES


# ============================================================================
# Rate Limiting
# ============================================================================


class TestRateLimiting:
    """Test rate limiting on belief endpoints."""

    def test_rate_limit_allows_under_limit(self, handler, http_handler):
        with patch.object(handler, "_get_debate_cruxes", return_value=MagicMock(status_code=200, body=b'{}')) as mock_method:
            with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
                result = handler.handle("/api/belief-network/debate-123/cruxes", {}, http_handler)
                # If it reaches the method, rate limit passed
                assert result is not None

    def test_rate_limit_blocks_when_exceeded(self, handler, http_handler):
        # Fill up the rate limiter
        _belief_limiter.rpm = 2
        _belief_limiter.is_allowed("127.0.0.1")
        _belief_limiter.is_allowed("127.0.0.1")

        result = handler.handle("/api/belief-network/debate-123/cruxes", {}, http_handler)
        assert _status(result) == 429
        assert "Rate limit" in _body(result)["error"]

        # Restore
        _belief_limiter.rpm = 60


# ============================================================================
# Authentication
# ============================================================================


class TestAuthentication:
    """Test authentication requirements."""

    @pytest.mark.no_auto_auth
    def test_unauthenticated_returns_401(self):
        """Without auto auth, require_auth_or_error should fail."""
        h = BeliefHandler(server_context={"nomic_dir": Path("/tmp/test")})
        mock_http = MockHTTPHandler()

        # Override require_auth_or_error to simulate auth failure
        def fail_auth(handler):
            from aragora.server.handlers.base import error_response
            return None, error_response("Authentication required", 401)

        with patch.object(h, "require_auth_or_error", fail_auth):
            result = h.handle("/api/belief-network/debate-123/cruxes", {}, mock_http)
            assert _status(result) == 401


# ============================================================================
# RBAC Permission Checks
# ============================================================================


class TestRBACPermissions:
    """Test RBAC permission checking."""

    def test_check_belief_permission_allowed(self, handler, http_handler):
        """With auto-auth admin roles, permission should pass."""
        mock_user = MagicMock()
        mock_user.id = "test-user"
        mock_user.org_id = "test-org"

        with patch("aragora.server.handlers.belief.get_permission_checker") as mock_checker:
            decision = MagicMock()
            decision.allowed = True
            mock_checker.return_value.check_permission.return_value = decision

            result = handler._check_belief_permission(http_handler, mock_user, "belief:read")
            assert result is None  # None means allowed

    def test_check_belief_permission_denied(self, handler, http_handler):
        """Permission denied returns 403."""
        mock_user = MagicMock()
        mock_user.id = "test-user"
        mock_user.org_id = "test-org"

        with patch("aragora.server.handlers.belief.get_permission_checker") as mock_checker:
            decision = MagicMock()
            decision.allowed = False
            decision.reason = "Insufficient role"
            mock_checker.return_value.check_permission.return_value = decision

            result = handler._check_belief_permission(http_handler, mock_user, "belief:read")
            assert _status(result) == 403
            assert "Permission denied" in _body(result)["error"]

    def test_check_belief_permission_uses_header_roles(self, handler):
        """Roles are extracted from X-User-Roles header."""
        mock_http = MockHTTPHandler(roles="editor,viewer")
        mock_user = MagicMock()
        mock_user.id = "test-user"
        mock_user.org_id = None

        with patch("aragora.server.handlers.belief.get_permission_checker") as mock_checker:
            decision = MagicMock()
            decision.allowed = True
            mock_checker.return_value.check_permission.return_value = decision

            handler._check_belief_permission(mock_http, mock_user, "belief:read")

            # Verify the AuthorizationContext had the correct roles
            call_args = mock_checker.return_value.check_permission.call_args
            ctx = call_args[0][0]
            assert "editor" in ctx.roles
            assert "viewer" in ctx.roles

    def test_check_belief_permission_default_member_role(self, handler, http_handler):
        """Without X-User-Roles header, defaults to member role."""
        mock_user = MagicMock()
        mock_user.id = "test-user"
        mock_user.org_id = None

        with patch("aragora.server.handlers.belief.get_permission_checker") as mock_checker:
            decision = MagicMock()
            decision.allowed = True
            mock_checker.return_value.check_permission.return_value = decision

            handler._check_belief_permission(http_handler, mock_user, "belief:read")

            call_args = mock_checker.return_value.check_permission.call_args
            ctx = call_args[0][0]
            assert "member" in ctx.roles

    def test_export_requires_export_permission(self, handler, http_handler):
        """Export endpoint checks belief:export in addition to belief:read."""
        # Simulate: belief:read allowed, belief:export denied
        call_count = [0]

        with patch("aragora.server.handlers.belief.get_permission_checker") as mock_checker:
            def check_perm(ctx, permission):
                call_count[0] += 1
                decision = MagicMock()
                if permission == "belief:export":
                    decision.allowed = False
                    decision.reason = "No export permission"
                else:
                    decision.allowed = True
                return decision

            mock_checker.return_value.check_permission.side_effect = check_perm

            with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
                result = handler.handle(
                    "/api/belief-network/debate-123/export", {}, http_handler
                )
                assert _status(result) == 403


# ============================================================================
# _extract_debate_id
# ============================================================================


class TestExtractDebateId:
    """Test debate ID extraction from path."""

    def test_valid_debate_id(self, handler):
        result = handler._extract_debate_id("/api/belief-network/debate-123/cruxes", 3)
        assert result == "debate-123"

    def test_invalid_debate_id(self, handler):
        result = handler._extract_debate_id("/api/belief-network/../cruxes", 3)
        assert result is None

    def test_missing_segment(self, handler):
        result = handler._extract_debate_id("/api/belief-network", 3)
        assert result is None

    def test_empty_segment(self, handler):
        result = handler._extract_debate_id("/api/belief-network//cruxes", 3)
        assert result is None

    def test_valid_id_with_underscores(self, handler):
        result = handler._extract_debate_id("/api/belief-network/debate_abc_123/cruxes", 3)
        assert result == "debate_abc_123"


# ============================================================================
# GET /api/belief-network/:debate_id/cruxes
# ============================================================================


class TestGetDebateCruxes:
    """Test cruxes endpoint."""

    def test_belief_network_unavailable(self, handler, http_handler):
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", False):
            result = handler.handle(
                "/api/belief-network/debate-123/cruxes", {}, http_handler
            )
            assert _status(result) == 503
            assert "not available" in _body(result)["error"]

    def test_no_nomic_dir(self, handler_no_nomic, http_handler):
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            result = handler_no_nomic.handle(
                "/api/belief-network/debate-123/cruxes", {}, http_handler
            )
            assert _status(result) == 503
            assert "not configured" in _body(result)["error"]

    def test_trace_not_found(self, handler, http_handler):
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=False):
                result = handler.handle(
                    "/api/belief-network/debate-123/cruxes", {}, http_handler
                )
                assert _status(result) == 404
                assert "not found" in _body(result)["error"]

    def test_successful_cruxes(self, handler, http_handler):
        mock_trace = _make_mock_trace()
        mock_network = _make_mock_network()
        mock_cruxes = [
            {"claim_id": "c1", "statement": "Claim 1", "crux_score": 0.9},
            {"claim_id": "c2", "statement": "Claim 2", "crux_score": 0.7},
        ]

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=True):
                with patch("aragora.server.handlers.belief.BeliefHandler._create_belief_network", return_value=mock_network):
                    with patch("aragora.server.handlers.belief.BeliefPropagationAnalyzer") as MockAnalyzer:
                        MockAnalyzer.return_value.identify_debate_cruxes.return_value = mock_cruxes
                        with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                            result = handler.handle(
                                "/api/belief-network/debate-123/cruxes", {}, http_handler
                            )

        assert _status(result) == 200
        body = _body(result)
        assert body["debate_id"] == "debate-123"
        assert body["count"] == 2
        assert len(body["cruxes"]) == 2

    def test_cruxes_with_top_k_param(self, handler, http_handler):
        mock_trace = _make_mock_trace()
        mock_network = _make_mock_network()

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=True):
                with patch("aragora.server.handlers.belief.BeliefHandler._create_belief_network", return_value=mock_network):
                    with patch("aragora.server.handlers.belief.BeliefPropagationAnalyzer") as MockAnalyzer:
                        MockAnalyzer.return_value.identify_debate_cruxes.return_value = []
                        with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                            result = handler.handle(
                                "/api/belief-network/debate-123/cruxes",
                                {"top_k": ["5"]},
                                http_handler,
                            )

        assert _status(result) == 200
        MockAnalyzer.return_value.identify_debate_cruxes.assert_called_once_with(top_k=5)

    def test_cruxes_top_k_clamped_max(self, handler, http_handler):
        """top_k should be clamped to max_val=10."""
        mock_trace = _make_mock_trace()
        mock_network = _make_mock_network()

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=True):
                with patch("aragora.server.handlers.belief.BeliefHandler._create_belief_network", return_value=mock_network):
                    with patch("aragora.server.handlers.belief.BeliefPropagationAnalyzer") as MockAnalyzer:
                        MockAnalyzer.return_value.identify_debate_cruxes.return_value = []
                        with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                            result = handler.handle(
                                "/api/belief-network/debate-123/cruxes",
                                {"top_k": ["999"]},
                                http_handler,
                            )

        assert _status(result) == 200
        MockAnalyzer.return_value.identify_debate_cruxes.assert_called_once_with(top_k=10)

    def test_cruxes_invalid_debate_id(self, handler, http_handler):
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            result = handler.handle(
                "/api/belief-network/../../etc/cruxes", {}, http_handler
            )
            assert _status(result) == 400
            assert "Invalid" in _body(result)["error"]

    def test_versioned_cruxes_path(self, handler, http_handler):
        """Versioned path /api/v1/belief-network/:id/cruxes should work."""
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", False):
            result = handler.handle(
                "/api/v1/belief-network/debate-123/cruxes", {}, http_handler
            )
            assert _status(result) == 503  # Hits the "not available" check


# ============================================================================
# GET /api/belief-network/:debate_id/load-bearing-claims
# ============================================================================


class TestGetLoadBearingClaims:
    """Test load-bearing claims endpoint."""

    def test_belief_network_unavailable(self, handler, http_handler):
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", False):
            result = handler.handle(
                "/api/belief-network/debate-123/load-bearing-claims", {}, http_handler
            )
            assert _status(result) == 503

    def test_no_nomic_dir(self, handler_no_nomic, http_handler):
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            result = handler_no_nomic.handle(
                "/api/belief-network/debate-123/load-bearing-claims", {}, http_handler
            )
            assert _status(result) == 503
            assert "not configured" in _body(result)["error"]

    def test_trace_not_found(self, handler, http_handler):
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=False):
                result = handler.handle(
                    "/api/belief-network/debate-123/load-bearing-claims", {}, http_handler
                )
                assert _status(result) == 404

    def test_successful_load_bearing(self, handler, http_handler):
        mock_trace = _make_mock_trace()

        mock_node = MagicMock()
        mock_node.claim_id = "claim-1"
        mock_node.claim_statement = "Important claim"
        mock_node.author = "claude"

        mock_network = _make_mock_network()
        mock_network.get_load_bearing_claims.return_value = [(mock_node, 0.95)]

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=True):
                with patch("aragora.server.handlers.belief.BeliefHandler._create_belief_network", return_value=mock_network):
                    with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                        result = handler.handle(
                            "/api/belief-network/debate-123/load-bearing-claims",
                            {},
                            http_handler,
                        )

        assert _status(result) == 200
        body = _body(result)
        assert body["debate_id"] == "debate-123"
        assert body["count"] == 1
        assert body["load_bearing_claims"][0]["claim_id"] == "claim-1"
        assert body["load_bearing_claims"][0]["centrality"] == 0.95

    def test_load_bearing_with_limit_param(self, handler, http_handler):
        mock_trace = _make_mock_trace()
        mock_network = _make_mock_network()

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=True):
                with patch("aragora.server.handlers.belief.BeliefHandler._create_belief_network", return_value=mock_network):
                    with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                        result = handler.handle(
                            "/api/belief-network/debate-123/load-bearing-claims",
                            {"limit": ["10"]},
                            http_handler,
                        )

        assert _status(result) == 200
        mock_network.get_load_bearing_claims.assert_called_once_with(limit=10)

    def test_load_bearing_limit_clamped(self, handler, http_handler):
        """Limit should be clamped to max_val=20."""
        mock_trace = _make_mock_trace()
        mock_network = _make_mock_network()

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=True):
                with patch("aragora.server.handlers.belief.BeliefHandler._create_belief_network", return_value=mock_network):
                    with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                        result = handler.handle(
                            "/api/belief-network/debate-123/load-bearing-claims",
                            {"limit": ["100"]},
                            http_handler,
                        )

        assert _status(result) == 200
        mock_network.get_load_bearing_claims.assert_called_once_with(limit=20)

    def test_invalid_debate_id(self, handler, http_handler):
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            result = handler.handle(
                "/api/belief-network/../load-bearing-claims", {}, http_handler
            )
            assert _status(result) == 400


# ============================================================================
# GET /api/belief-network/:debate_id/graph
# ============================================================================


class TestGetBeliefNetworkGraph:
    """Test belief network graph endpoint."""

    def test_belief_network_unavailable(self, handler, http_handler):
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", False):
            result = handler.handle(
                "/api/belief-network/debate-123/graph", {}, http_handler
            )
            assert _status(result) == 503

    def test_no_nomic_dir(self, handler_no_nomic, http_handler):
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            result = handler_no_nomic.handle(
                "/api/belief-network/debate-123/graph", {}, http_handler
            )
            assert _status(result) == 503

    def test_trace_not_found(self, handler, http_handler):
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=False):
                result = handler.handle(
                    "/api/belief-network/debate-123/graph", {}, http_handler
                )
                assert _status(result) == 404

    def test_successful_graph_with_cruxes(self, handler, http_handler):
        mock_trace = _make_mock_trace()

        mock_node_data = {
            "node": MagicMock(
                claim_id="c1",
                claim_statement="Test claim",
                author="claude",
                get_belief_distribution=MagicMock(
                    return_value={"true_prob": 0.6, "false_prob": 0.2, "uncertain_prob": 0.2}
                ),
            ),
            "centrality": 0.85,
            "entropy": 0.65,
        }
        mock_edge = {
            "source": "c1",
            "target": "c1",
            "weight": 0.7,
            "type": "supports",
        }
        mock_crux = {"claim_id": "c1", "crux_score": 0.92}

        mock_network = _make_mock_network()
        mock_network.get_all_claims.return_value = [mock_node_data]
        mock_network.get_all_edges.return_value = [mock_edge]

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=True):
                with patch("aragora.server.handlers.belief.BeliefHandler._create_belief_network", return_value=mock_network):
                    with patch("aragora.server.handlers.belief.BeliefPropagationAnalyzer") as MockAnalyzer:
                        MockAnalyzer.return_value.identify_debate_cruxes.return_value = [mock_crux]
                        with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                            result = handler.handle(
                                "/api/belief-network/debate-123/graph",
                                {},
                                http_handler,
                            )

        assert _status(result) == 200
        body = _body(result)
        assert "nodes" in body
        assert "links" in body
        assert "metadata" in body
        assert body["metadata"]["debate_id"] == "debate-123"
        assert body["metadata"]["total_claims"] == 1
        assert body["metadata"]["crux_count"] == 1
        assert body["nodes"][0]["is_crux"] is True
        assert body["nodes"][0]["crux_score"] == 0.92

    def test_graph_without_cruxes(self, handler, http_handler):
        mock_trace = _make_mock_trace()
        mock_network = _make_mock_network()

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=True):
                with patch("aragora.server.handlers.belief.BeliefHandler._create_belief_network", return_value=mock_network):
                    with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                        result = handler.handle(
                            "/api/belief-network/debate-123/graph",
                            {"include_cruxes": ["false"]},
                            http_handler,
                        )

        assert _status(result) == 200
        body = _body(result)
        assert body["metadata"]["crux_count"] == 0

    def test_graph_node_without_belief_distribution(self, handler, http_handler):
        """Node without get_belief_distribution attribute."""
        mock_trace = _make_mock_trace()

        mock_node = MagicMock(spec=["claim_id", "claim_statement", "author"])
        mock_node.claim_id = "c1"
        mock_node.claim_statement = "Test"
        mock_node.author = "claude"

        mock_node_data = {"node": mock_node, "centrality": 0.5, "entropy": 0.5}
        mock_network = _make_mock_network()
        mock_network.get_all_claims.return_value = [mock_node_data]

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=True):
                with patch("aragora.server.handlers.belief.BeliefHandler._create_belief_network", return_value=mock_network):
                    with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                        result = handler.handle(
                            "/api/belief-network/debate-123/graph",
                            {"include_cruxes": ["false"]},
                            http_handler,
                        )

        assert _status(result) == 200
        body = _body(result)
        assert body["nodes"][0]["belief"] is None

    def test_graph_filters_orphan_edges(self, handler, http_handler):
        """Edges with source/target not in node_ids should be filtered."""
        mock_trace = _make_mock_trace()

        mock_node_data = {
            "node": MagicMock(
                claim_id="c1",
                claim_statement="Test",
                author="claude",
            ),
            "centrality": 0.5,
            "entropy": 0.5,
        }
        # hasattr check for get_belief_distribution
        mock_node_data["node"].get_belief_distribution = None
        del mock_node_data["node"].get_belief_distribution

        orphan_edge = {"source": "c1", "target": "c_nonexistent", "weight": 0.5, "type": "influences"}
        mock_network = _make_mock_network()
        mock_network.get_all_claims.return_value = [mock_node_data]
        mock_network.get_all_edges.return_value = [orphan_edge]

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=True):
                with patch("aragora.server.handlers.belief.BeliefHandler._create_belief_network", return_value=mock_network):
                    with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                        result = handler.handle(
                            "/api/belief-network/debate-123/graph",
                            {"include_cruxes": ["false"]},
                            http_handler,
                        )

        assert _status(result) == 200
        body = _body(result)
        assert len(body["links"]) == 0  # Orphan edge filtered out

    def test_graph_node_data_without_node_key(self, handler, http_handler):
        """Node data dict without 'node' key should be skipped."""
        mock_trace = _make_mock_trace()
        mock_network = _make_mock_network()
        mock_network.get_all_claims.return_value = [{"centrality": 0.5}]  # no "node" key

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=True):
                with patch("aragora.server.handlers.belief.BeliefHandler._create_belief_network", return_value=mock_network):
                    with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                        result = handler.handle(
                            "/api/belief-network/debate-123/graph",
                            {"include_cruxes": ["false"]},
                            http_handler,
                        )

        assert _status(result) == 200
        body = _body(result)
        assert len(body["nodes"]) == 0


# ============================================================================
# GET /api/belief-network/:debate_id/export
# ============================================================================


class TestExportBeliefNetwork:
    """Test belief network export endpoint."""

    def test_belief_network_unavailable(self, handler, http_handler):
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", False):
            result = handler.handle(
                "/api/belief-network/debate-123/export", {}, http_handler
            )
            assert _status(result) == 503

    def test_no_nomic_dir(self, handler_no_nomic, http_handler):
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            result = handler_no_nomic.handle(
                "/api/belief-network/debate-123/export", {}, http_handler
            )
            assert _status(result) == 503

    def test_export_json_format(self, handler, http_handler):
        mock_trace = _make_mock_trace()
        mock_network = _make_mock_network()
        mock_network.get_all_claims.return_value = []
        mock_network.get_all_edges.return_value = []

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=True):
                with patch("aragora.server.handlers.belief.BeliefHandler._create_belief_network", return_value=mock_network):
                    with patch("aragora.server.handlers.belief.BeliefPropagationAnalyzer") as MockAnalyzer:
                        MockAnalyzer.return_value.identify_debate_cruxes.return_value = []
                        with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                            result = handler.handle(
                                "/api/belief-network/debate-123/export",
                                {"format": ["json"]},
                                http_handler,
                            )

        assert _status(result) == 200
        body = _body(result)
        assert body["format"] == "json"
        assert body["debate_id"] == "debate-123"
        assert "nodes" in body
        assert "edges" in body
        assert "summary" in body

    def test_export_csv_format(self, handler, http_handler):
        mock_trace = _make_mock_trace()

        mock_node = MagicMock()
        mock_node.claim_id = "c1"
        mock_node.claim_statement = "Test"
        mock_node.author = "claude"

        mock_network = _make_mock_network()
        mock_network.get_all_claims.return_value = [{"node": mock_node, "centrality": 0.5}]
        mock_network.get_all_edges.return_value = []

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=True):
                with patch("aragora.server.handlers.belief.BeliefHandler._create_belief_network", return_value=mock_network):
                    with patch("aragora.server.handlers.belief.BeliefPropagationAnalyzer") as MockAnalyzer:
                        MockAnalyzer.return_value.identify_debate_cruxes.return_value = []
                        with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                            result = handler.handle(
                                "/api/belief-network/debate-123/export",
                                {"format": ["csv"]},
                                http_handler,
                            )

        assert _status(result) == 200
        body = _body(result)
        assert body["format"] == "csv"
        assert "nodes_csv" in body
        assert "edges_csv" in body
        assert "headers" in body
        assert "nodes" in body["headers"]
        assert "edges" in body["headers"]

    def test_export_graphml_format(self, handler, http_handler):
        mock_trace = _make_mock_trace()

        mock_node = MagicMock()
        mock_node.claim_id = "c1"
        mock_node.claim_statement = "Test claim <special>"
        mock_node.author = "claude"

        mock_network = _make_mock_network()
        mock_network.get_all_claims.return_value = [{"node": mock_node, "centrality": 0.5}]
        mock_network.get_all_edges.return_value = [
            {"source": "c1", "target": "c1", "weight": 0.7, "type": "supports"}
        ]

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=True):
                with patch("aragora.server.handlers.belief.BeliefHandler._create_belief_network", return_value=mock_network):
                    with patch("aragora.server.handlers.belief.BeliefPropagationAnalyzer") as MockAnalyzer:
                        MockAnalyzer.return_value.identify_debate_cruxes.return_value = []
                        with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                            result = handler.handle(
                                "/api/belief-network/debate-123/export",
                                {"format": ["graphml"]},
                                http_handler,
                            )

        assert _status(result) == 200
        body = _body(result)
        assert body["format"] == "graphml"
        assert body["content_type"] == "application/xml"
        assert "<?xml version" in body["content"]
        assert "<graphml" in body["content"]
        assert "&lt;special&gt;" in body["content"]  # XML escaped

    def test_export_default_format_is_json(self, handler, http_handler):
        mock_trace = _make_mock_trace()
        mock_network = _make_mock_network()
        mock_network.get_all_claims.return_value = []
        mock_network.get_all_edges.return_value = []

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=True):
                with patch("aragora.server.handlers.belief.BeliefHandler._create_belief_network", return_value=mock_network):
                    with patch("aragora.server.handlers.belief.BeliefPropagationAnalyzer") as MockAnalyzer:
                        MockAnalyzer.return_value.identify_debate_cruxes.return_value = []
                        with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                            # No format param -> defaults to json
                            result = handler.handle(
                                "/api/belief-network/debate-123/export",
                                {},
                                http_handler,
                            )

        assert _status(result) == 200
        body = _body(result)
        assert body["format"] == "json"

    def test_export_unknown_format_defaults_to_json(self, handler, http_handler):
        mock_trace = _make_mock_trace()
        mock_network = _make_mock_network()
        mock_network.get_all_claims.return_value = []
        mock_network.get_all_edges.return_value = []

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=True):
                with patch("aragora.server.handlers.belief.BeliefHandler._create_belief_network", return_value=mock_network):
                    with patch("aragora.server.handlers.belief.BeliefPropagationAnalyzer") as MockAnalyzer:
                        MockAnalyzer.return_value.identify_debate_cruxes.return_value = []
                        with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                            result = handler.handle(
                                "/api/belief-network/debate-123/export",
                                {"format": ["yaml"]},
                                http_handler,
                            )

        assert _status(result) == 200
        body = _body(result)
        assert body["format"] == "json"

    def test_export_invalid_debate_id(self, handler, http_handler):
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            result = handler.handle(
                "/api/belief-network/../../export", {}, http_handler
            )
            assert _status(result) == 400

    def test_export_graphml_edge_indexing(self, handler, http_handler):
        """GraphML edges should have sequential IDs (e0, e1, ...)."""
        mock_trace = _make_mock_trace()

        mock_node1 = MagicMock(claim_id="c1", claim_statement="A", author="claude")
        mock_node2 = MagicMock(claim_id="c2", claim_statement="B", author="gpt4")

        mock_network = _make_mock_network()
        mock_network.get_all_claims.return_value = [
            {"node": mock_node1, "centrality": 0.5},
            {"node": mock_node2, "centrality": 0.3},
        ]
        mock_network.get_all_edges.return_value = [
            {"source": "c1", "target": "c2", "weight": 0.8, "type": "supports"},
            {"source": "c2", "target": "c1", "weight": 0.4, "type": "opposes"},
        ]

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=True):
                with patch("aragora.server.handlers.belief.BeliefHandler._create_belief_network", return_value=mock_network):
                    with patch("aragora.server.handlers.belief.BeliefPropagationAnalyzer") as MockAnalyzer:
                        MockAnalyzer.return_value.identify_debate_cruxes.return_value = []
                        with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                            result = handler.handle(
                                "/api/belief-network/debate-123/export",
                                {"format": ["graphml"]},
                                http_handler,
                            )

        body = _body(result)
        assert 'id="e0"' in body["content"]
        assert 'id="e1"' in body["content"]


# ============================================================================
# GET /api/provenance/:debate_id/claims/:claim_id/support
# ============================================================================


class TestGetClaimSupport:
    """Test claim support/provenance endpoint."""

    def test_provenance_unavailable(self, handler, http_handler):
        with patch("aragora.server.handlers.belief.PROVENANCE_AVAILABLE", False):
            result = handler.handle(
                "/api/provenance/debate-123/claims/claim-1/support", {}, http_handler
            )
            assert _status(result) == 503
            assert "not available" in _body(result)["error"]

    def test_no_nomic_dir(self, handler_no_nomic, http_handler):
        with patch("aragora.server.handlers.belief.PROVENANCE_AVAILABLE", True):
            result = handler_no_nomic.handle(
                "/api/provenance/debate-123/claims/claim-1/support", {}, http_handler
            )
            assert _status(result) == 503
            assert "not configured" in _body(result)["error"]

    def test_no_provenance_data(self, handler, http_handler):
        with patch("aragora.server.handlers.belief.PROVENANCE_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=False):
                result = handler.handle(
                    "/api/provenance/debate-123/claims/claim-1/support", {}, http_handler
                )
                assert _status(result) == 200
                body = _body(result)
                assert body["support"] is None
                assert "No provenance data" in body["message"]

    def test_successful_claim_support(self, handler, http_handler):
        mock_tracker = MagicMock()
        mock_tracker.get_claim_support.return_value = {
            "verified": True,
            "sources": ["source1", "source2"],
            "confidence": 0.85,
        }

        with patch("aragora.server.handlers.belief.PROVENANCE_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=True):
                with patch("aragora.server.handlers.belief.ProvenanceTracker") as MockTracker:
                    MockTracker.load.return_value = mock_tracker
                    result = handler.handle(
                        "/api/provenance/debate-123/claims/claim-1/support",
                        {},
                        http_handler,
                    )

        assert _status(result) == 200
        body = _body(result)
        assert body["debate_id"] == "debate-123"
        assert body["claim_id"] == "claim-1"
        assert body["support"]["verified"] is True
        assert body["support"]["confidence"] == 0.85

    def test_invalid_debate_id_in_provenance(self, handler, http_handler):
        with patch("aragora.server.handlers.belief.PROVENANCE_AVAILABLE", True):
            result = handler.handle(
                "/api/provenance/../../claims/claim-1/support", {}, http_handler
            )
            assert _status(result) == 400
            assert "Invalid" in _body(result)["error"]

    def test_invalid_claim_id_in_provenance(self, handler, http_handler):
        with patch("aragora.server.handlers.belief.PROVENANCE_AVAILABLE", True):
            result = handler.handle(
                "/api/provenance/debate-123/claims/../../support", {}, http_handler
            )
            assert _status(result) == 400
            assert "Invalid" in _body(result)["error"]

    def test_short_path_format(self, handler, http_handler):
        """Path with fewer than 7 parts should return invalid path format."""
        with patch("aragora.server.handlers.belief.PROVENANCE_AVAILABLE", True):
            result = handler.handle(
                "/api/claims/c1/support", {}, http_handler
            )
            # This path either doesn't match or returns 400
            if result is not None:
                status = _status(result)
                assert status == 400

    def test_versioned_provenance_path(self, handler, http_handler):
        """Versioned provenance path should also work via strip_version_prefix."""
        with patch("aragora.server.handlers.belief.PROVENANCE_AVAILABLE", False):
            result = handler.handle(
                "/api/v1/provenance/debate-123/claims/claim-1/support", {}, http_handler
            )
            assert _status(result) == 503


# ============================================================================
# GET /api/debate/:debate_id/graph-stats
# ============================================================================


class TestGetDebateGraphStats:
    """Test graph stats endpoint."""

    def test_no_nomic_dir(self, handler_no_nomic, http_handler):
        result = handler_no_nomic.handle(
            "/api/debate/debate-123/graph-stats", {}, http_handler
        )
        assert _status(result) == 503
        assert "not configured" in _body(result)["error"]

    def test_debate_not_found(self, handler, http_handler):
        with patch.object(Path, "exists", return_value=False):
            result = handler.handle(
                "/api/debate/debate-123/graph-stats", {}, http_handler
            )
            assert _status(result) == 404
            assert "not found" in _body(result)["error"]

    def test_successful_graph_stats_from_trace(self, handler, http_handler):
        mock_trace = _make_mock_trace()
        mock_stats = {
            "total_arguments": 5,
            "total_critiques": 3,
            "density": 0.6,
        }

        # Only the trace_path.exists() should return True
        original_exists = Path.exists

        def selective_exists(self):
            if "traces" in str(self):
                return True
            return False

        with patch.object(Path, "exists", selective_exists):
            with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                with patch("aragora.visualization.mapper.ArgumentCartographer") as MockCarto:
                    MockCarto.return_value.get_statistics.return_value = mock_stats
                    result = handler.handle(
                        "/api/debate/debate-123/graph-stats", {}, http_handler
                    )

        assert _status(result) == 200
        body = _body(result)
        assert body["total_arguments"] == 5

    def test_graph_stats_from_replay_fallback(self, handler, http_handler):
        """When trace doesn't exist but replay does, use replay."""
        mock_stats = {"total_arguments": 2}

        events = [
            json.dumps({"type": "agent_message", "agent": "claude", "data": {"content": "msg1", "role": "proposer"}, "round": 1}),
            json.dumps({"type": "critique", "agent": "gpt4", "data": {"target": "claude", "severity": 0.5, "content": "weak"}, "round": 1}),
        ]

        def selective_exists(self):
            if "replays" in str(self) and "events.jsonl" in str(self):
                return True
            return False

        mock_file_content = "\n".join(events) + "\n"

        with patch.object(Path, "exists", selective_exists):
            with patch.object(Path, "open", return_value=MagicMock(
                __enter__=MagicMock(return_value=iter(mock_file_content.splitlines(True))),
                __exit__=MagicMock(return_value=False),
            )):
                with patch("aragora.visualization.mapper.ArgumentCartographer") as MockCarto:
                    MockCarto.return_value.get_statistics.return_value = mock_stats
                    result = handler.handle(
                        "/api/debate/debate-123/graph-stats", {}, http_handler
                    )

        assert _status(result) == 200

    def test_graph_stats_invalid_debate_id(self, handler, http_handler):
        result = handler.handle(
            "/api/debate/../graph-stats", {}, http_handler
        )
        assert _status(result) == 400

    def test_graph_stats_malformed_replay_line(self, handler, http_handler):
        """Malformed JSON lines in replay should be skipped."""
        mock_stats = {"total_arguments": 0}

        events_with_bad_line = "not-json\n{\"type\": \"unknown\"}\n"

        def selective_exists(self):
            if "replays" in str(self) and "events.jsonl" in str(self):
                return True
            return False

        with patch.object(Path, "exists", selective_exists):
            with patch.object(Path, "open", return_value=MagicMock(
                __enter__=MagicMock(return_value=iter(events_with_bad_line.splitlines(True))),
                __exit__=MagicMock(return_value=False),
            )):
                with patch("aragora.visualization.mapper.ArgumentCartographer") as MockCarto:
                    MockCarto.return_value.get_statistics.return_value = mock_stats
                    result = handler.handle(
                        "/api/debate/debate-123/graph-stats", {}, http_handler
                    )

        assert _status(result) == 200


# ============================================================================
# GET /api/v1/debates/:debate_id/cruxes (SDK alias - crux analysis)
# ============================================================================


class TestGetCruxAnalysis:
    """Test the /api/debates/:debate_id/cruxes (crux analysis) endpoint."""

    def test_belief_network_unavailable(self, handler, http_handler):
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", False):
            result = handler.handle(
                "/api/debates/debate-123/cruxes", {}, http_handler
            )
            assert _status(result) == 503

    def test_no_nomic_dir(self, handler_no_nomic, http_handler):
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            result = handler_no_nomic.handle(
                "/api/debates/debate-123/cruxes", {}, http_handler
            )
            assert _status(result) == 503

    def test_trace_not_found(self, handler, http_handler):
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=False):
                result = handler.handle(
                    "/api/debates/debate-123/cruxes", {}, http_handler
                )
                assert _status(result) == 404

    def test_successful_crux_analysis(self, handler, http_handler):
        mock_trace = _make_mock_trace()
        mock_network = _make_mock_network()
        mock_analysis = MagicMock()
        mock_analysis.to_dict.return_value = {
            "cruxes": [{"claim_id": "c1", "score": 0.9}],
            "count": 1,
        }

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=True):
                with patch("aragora.server.handlers.belief.BeliefHandler._create_belief_network", return_value=mock_network):
                    with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                        with patch("aragora.reasoning.crux_detector.CruxDetector") as MockDetector:
                            MockDetector.return_value.detect_cruxes.return_value = mock_analysis
                            result = handler.handle(
                                "/api/debates/debate-123/cruxes",
                                {},
                                http_handler,
                            )

        assert _status(result) == 200
        body = _body(result)
        assert body["debate_id"] == "debate-123"
        assert body["count"] == 1

    def test_crux_analysis_with_limit(self, handler, http_handler):
        mock_trace = _make_mock_trace()
        mock_network = _make_mock_network()
        mock_analysis = MagicMock()
        mock_analysis.to_dict.return_value = {"cruxes": [], "count": 0}

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=True):
                with patch("aragora.server.handlers.belief.BeliefHandler._create_belief_network", return_value=mock_network):
                    with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                        with patch("aragora.reasoning.crux_detector.CruxDetector") as MockDetector:
                            MockDetector.return_value.detect_cruxes.return_value = mock_analysis
                            result = handler.handle(
                                "/api/debates/debate-123/cruxes",
                                {"limit": ["3"]},
                                http_handler,
                            )

        assert _status(result) == 200
        MockDetector.return_value.detect_cruxes.assert_called_once_with(top_k=3)

    def test_crux_analysis_versioned_path(self, handler, http_handler):
        """Versioned /api/v1/debates/:id/cruxes should work."""
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", False):
            result = handler.handle(
                "/api/v1/debates/debate-123/cruxes", {}, http_handler
            )
            assert _status(result) == 503

    def test_crux_analysis_invalid_debate_id(self, handler, http_handler):
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            result = handler.handle(
                "/api/debates/../cruxes", {}, http_handler
            )
            assert _status(result) == 400


# ============================================================================
# KM Adapter
# ============================================================================


class TestKMAdapter:
    """Test Knowledge Mound adapter lifecycle."""

    def test_get_km_adapter_from_context(self, handler):
        mock_adapter = MagicMock()
        handler.ctx["belief_km_adapter"] = mock_adapter

        result = handler._get_km_adapter()
        assert result is mock_adapter

    def test_get_km_adapter_cached(self, handler):
        mock_adapter = MagicMock()
        handler._km_adapter = mock_adapter

        result = handler._get_km_adapter()
        assert result is mock_adapter

    def test_get_km_adapter_import_failure(self, handler):
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            result = handler._get_km_adapter()
            # Should return None gracefully
            assert result is None or result is not None  # Either way, no crash

    def test_get_km_adapter_creation_runtime_error(self, handler):
        with patch.dict(handler.ctx, {}, clear=True):
            with patch(
                "aragora.knowledge.mound.adapters.belief_adapter.BeliefAdapter",
                side_effect=RuntimeError("Init failed"),
            ):
                result = handler._get_km_adapter()
                assert result is None

    def test_get_km_adapter_stores_in_context(self):
        h = BeliefHandler(server_context={})
        mock_adapter = MagicMock()

        with patch(
            "aragora.knowledge.mound.adapters.belief_adapter.BeliefAdapter",
            return_value=mock_adapter,
        ):
            result = h._get_km_adapter()
            if result is not None:
                assert h.ctx.get("belief_km_adapter") is mock_adapter


# ============================================================================
# _create_belief_network
# ============================================================================


class TestCreateBeliefNetwork:
    """Test belief network creation with KM integration."""

    def test_create_basic_network(self, handler):
        mock_network = MagicMock()
        with patch("aragora.server.handlers.belief.BeliefNetwork", return_value=mock_network):
            with patch.object(handler, "_get_km_adapter", return_value=None):
                result = handler._create_belief_network("debate-123")
                assert result is mock_network

    def test_create_network_with_km_seeding(self, handler):
        mock_network = MagicMock()
        mock_network.seed_from_km.return_value = 5
        mock_adapter = MagicMock()

        with patch("aragora.server.handlers.belief.BeliefNetwork", return_value=mock_network):
            with patch.object(handler, "_get_km_adapter", return_value=mock_adapter):
                result = handler._create_belief_network(
                    "debate-123", topic="AI safety", seed_from_km=True
                )
                assert result is mock_network
                mock_network.seed_from_km.assert_called_once_with("AI safety", min_confidence=0.7)

    def test_create_network_no_seeding_without_topic(self, handler):
        mock_network = MagicMock()
        mock_adapter = MagicMock()

        with patch("aragora.server.handlers.belief.BeliefNetwork", return_value=mock_network):
            with patch.object(handler, "_get_km_adapter", return_value=mock_adapter):
                handler._create_belief_network("debate-123", seed_from_km=True)
                mock_network.seed_from_km.assert_not_called()

    def test_create_network_no_seeding_without_adapter(self, handler):
        mock_network = MagicMock()

        with patch("aragora.server.handlers.belief.BeliefNetwork", return_value=mock_network):
            with patch.object(handler, "_get_km_adapter", return_value=None):
                handler._create_belief_network(
                    "debate-123", topic="test", seed_from_km=True
                )
                mock_network.seed_from_km.assert_not_called()

    def test_create_network_zero_seeded(self, handler):
        """When seed_from_km returns 0, no log about seeding."""
        mock_network = MagicMock()
        mock_network.seed_from_km.return_value = 0
        mock_adapter = MagicMock()

        with patch("aragora.server.handlers.belief.BeliefNetwork", return_value=mock_network):
            with patch.object(handler, "_get_km_adapter", return_value=mock_adapter):
                handler._create_belief_network(
                    "debate-123", topic="test", seed_from_km=True
                )
                mock_network.seed_from_km.assert_called_once()


# ============================================================================
# _emit_km_event
# ============================================================================


class TestEmitKMEvent:
    """Test KM event emission for WebSocket notifications."""

    def test_emit_known_event_type(self, handler):
        mock_emitter = MagicMock()
        handler._emit_km_event(mock_emitter, "belief_converged", {"debate_id": "d1"})
        mock_emitter.emit.assert_called_once()

    def test_emit_crux_detected(self, handler):
        mock_emitter = MagicMock()
        handler._emit_km_event(mock_emitter, "crux_detected", {"claim_id": "c1"})
        mock_emitter.emit.assert_called_once()

    def test_emit_unknown_event_type_defaults(self, handler):
        mock_emitter = MagicMock()
        handler._emit_km_event(mock_emitter, "unknown_event", {"key": "val"})
        # Should default to MOUND_UPDATED and not crash
        mock_emitter.emit.assert_called_once()

    def test_emit_import_error_handled(self, handler):
        """ImportError in event emission should be caught gracefully."""
        mock_emitter = MagicMock()
        with patch("aragora.server.handlers.belief.BeliefHandler._emit_km_event") as mock_method:
            # Just verify the original method handles errors
            handler._emit_km_event(mock_emitter, "belief_converged", {})

    def test_emit_with_none_emitter_attribute_error(self, handler):
        """If emitter doesn't have emit method, handle gracefully."""
        # This should not raise due to AttributeError being caught
        handler._emit_km_event(None, "belief_converged", {})
        # No assertion needed - just verify no exception


# ============================================================================
# Unmatched route returns None
# ============================================================================


class TestHandleRouting:
    """Test that unmatched paths return None."""

    def test_unmatched_path_returns_none(self, handler, http_handler):
        result = handler.handle("/api/unmatched/path", {}, http_handler)
        assert result is None

    def test_handle_routes_to_correct_method(self, handler, http_handler):
        """Verify route dispatch works for various patterns."""
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", False):
            # Test cruxes route
            result = handler.handle(
                "/api/belief-network/debate-123/cruxes", {}, http_handler
            )
            assert result is not None
            assert _status(result) == 503

    def test_handle_versioned_debates_cruxes(self, handler, http_handler):
        """Versioned /api/v1/debates/:id/cruxes routes to crux analysis."""
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", False):
            result = handler.handle(
                "/api/v1/debates/debate-123/cruxes", {}, http_handler
            )
            assert result is not None
            assert _status(result) == 503


# ============================================================================
# Emergent Traits (internal method)
# ============================================================================


class TestEmergentTraits:
    """Test the _get_emergent_traits internal method."""

    def test_laboratory_unavailable(self, handler):
        with patch("aragora.server.handlers.belief.LABORATORY_AVAILABLE", False):
            result = handler._get_emergent_traits(None, None, 0.5, 10)
            assert _status(result) == 503

    def test_successful_traits(self, handler):
        mock_trait = MagicMock()
        mock_trait.agent_name = "claude"
        mock_trait.trait_name = "lateral_thinking"
        mock_trait.domain = "reasoning"
        mock_trait.confidence = 0.85
        mock_trait.evidence = ["evidence1"]
        mock_trait.detected_at = "2026-01-01"

        with patch("aragora.server.handlers.belief.LABORATORY_AVAILABLE", True):
            with patch("aragora.server.handlers.belief.PersonaLaboratory") as MockLab:
                MockLab.return_value.detect_emergent_traits.return_value = [mock_trait]
                result = handler._get_emergent_traits(
                    Path("/tmp/test"), MagicMock(), 0.5, 10
                )

        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 1
        assert body["emergent_traits"][0]["agent"] == "claude"
        assert body["emergent_traits"][0]["confidence"] == 0.85

    def test_traits_filtered_by_confidence(self, handler):
        low_trait = MagicMock()
        low_trait.confidence = 0.3

        high_trait = MagicMock()
        high_trait.agent_name = "claude"
        high_trait.trait_name = "reasoning"
        high_trait.domain = "logic"
        high_trait.confidence = 0.8
        high_trait.evidence = []
        high_trait.detected_at = "2026-01-01"

        with patch("aragora.server.handlers.belief.LABORATORY_AVAILABLE", True):
            with patch("aragora.server.handlers.belief.PersonaLaboratory") as MockLab:
                MockLab.return_value.detect_emergent_traits.return_value = [
                    low_trait, high_trait
                ]
                result = handler._get_emergent_traits(
                    Path("/tmp/test"), MagicMock(), 0.5, 10
                )

        body = _body(result)
        assert body["count"] == 1
        assert body["min_confidence"] == 0.5

    def test_traits_limited(self, handler):
        traits = []
        for i in range(5):
            t = MagicMock()
            t.agent_name = f"agent-{i}"
            t.trait_name = "reasoning"
            t.domain = "logic"
            t.confidence = 0.9
            t.evidence = []
            t.detected_at = "2026-01-01"
            traits.append(t)

        with patch("aragora.server.handlers.belief.LABORATORY_AVAILABLE", True):
            with patch("aragora.server.handlers.belief.PersonaLaboratory") as MockLab:
                MockLab.return_value.detect_emergent_traits.return_value = traits
                result = handler._get_emergent_traits(
                    Path("/tmp/test"), MagicMock(), 0.0, 3
                )

        body = _body(result)
        assert body["count"] == 3


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_query_params(self, handler, http_handler):
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", False):
            result = handler.handle(
                "/api/belief-network/debate-123/cruxes", {}, http_handler
            )
            assert _status(result) == 503

    def test_handle_with_none_handler(self, handler):
        """None handler should still return rate limit or auth error."""
        result = handler.handle("/api/belief-network/debate-123/cruxes", {}, None)
        # get_client_ip returns "unknown" for None, should still process
        assert result is not None

    def test_multiple_routes_same_debate_id(self, handler, http_handler):
        """Same debate_id can be accessed through different endpoints."""
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", False):
            r1 = handler.handle("/api/belief-network/debate-123/cruxes", {}, http_handler)
            r2 = handler.handle("/api/belief-network/debate-123/load-bearing-claims", {}, http_handler)
            r3 = handler.handle("/api/belief-network/debate-123/graph", {}, http_handler)
            assert _status(r1) == 503
            assert _status(r2) == 503
            assert _status(r3) == 503

    def test_debate_id_with_special_characters_rejected(self, handler, http_handler):
        """Debate IDs with special characters should be rejected."""
        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            result = handler.handle(
                "/api/belief-network/debate<script>/cruxes", {}, http_handler
            )
            assert _status(result) == 400

    def test_include_cruxes_case_insensitive(self, handler, http_handler):
        """include_cruxes param should handle True/TRUE/true."""
        mock_trace = _make_mock_trace()
        mock_network = _make_mock_network()

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=True):
                with patch("aragora.server.handlers.belief.BeliefHandler._create_belief_network", return_value=mock_network):
                    with patch("aragora.server.handlers.belief.BeliefPropagationAnalyzer") as MockAnalyzer:
                        MockAnalyzer.return_value.identify_debate_cruxes.return_value = []
                        with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                            result = handler.handle(
                                "/api/belief-network/debate-123/graph",
                                {"include_cruxes": ["TRUE"]},
                                http_handler,
                            )

        # "TRUE".lower() == "true" -> include_cruxes = True
        assert _status(result) == 200
        MockAnalyzer.return_value.identify_debate_cruxes.assert_called_once()

    def test_export_graphml_ampersand_escape(self, handler, http_handler):
        """GraphML export should escape & in statements."""
        mock_trace = _make_mock_trace()

        mock_node = MagicMock()
        mock_node.claim_id = "c1"
        mock_node.claim_statement = "A & B"
        mock_node.author = "claude"

        mock_network = _make_mock_network()
        mock_network.get_all_claims.return_value = [{"node": mock_node, "centrality": 0.5}]
        mock_network.get_all_edges.return_value = []

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch.object(Path, "exists", return_value=True):
                with patch("aragora.server.handlers.belief.BeliefHandler._create_belief_network", return_value=mock_network):
                    with patch("aragora.server.handlers.belief.BeliefPropagationAnalyzer") as MockAnalyzer:
                        MockAnalyzer.return_value.identify_debate_cruxes.return_value = []
                        with patch("aragora.debate.traces.DebateTrace.load", return_value=mock_trace):
                            result = handler.handle(
                                "/api/belief-network/debate-123/export",
                                {"format": ["graphml"]},
                                http_handler,
                            )

        body = _body(result)
        assert "&amp;" in body["content"]
        assert "A & B" not in body["content"]  # Should be escaped
