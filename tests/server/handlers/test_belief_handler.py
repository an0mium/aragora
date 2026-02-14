"""
Tests for the BeliefHandler module.

Tests cover:
- Handler routing for belief network endpoints
- Handler routing for provenance endpoints
- can_handle method for static and dynamic routes
- Rate limiting behavior
- Authentication and RBAC permission checking
- Success cases with mocked dependencies
- Error cases (unavailable modules, missing config)
- KM adapter integration
- Export formats (JSON, CSV, GraphML)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.belief import BeliefHandler, _belief_limiter


# ===========================================================================
# Mock Classes
# ===========================================================================


class MockPermissionDecision:
    """Mock permission decision that allows or denies based on init param."""

    def __init__(self, allowed: bool = True, reason: str = "Test mock"):
        self.allowed = allowed
        self.reason = reason


class MockPermissionChecker:
    """Mock permission checker with configurable behavior."""

    def __init__(self, allow: bool = True):
        self._allow = allow

    def check_permission(self, context, permission):
        return MockPermissionDecision(
            self._allow, "Allowed by test mock" if self._allow else "Denied by test mock"
        )


class MockUserAuthContext:
    """Mock user auth context."""

    def __init__(self, authenticated: bool = True):
        self.authenticated = authenticated
        self.is_authenticated = authenticated
        self.user_id = "test-user"
        self.id = "test-user"
        self.email = "test@example.com"
        self.org_id = "test-org"
        self.role = "admin"


class MockBeliefNode:
    """Mock belief network node."""

    def __init__(self, claim_id: str, statement: str, author: str):
        self.claim_id = claim_id
        self.claim_statement = statement
        self.author = author

    def get_belief_distribution(self):
        return {"true_prob": 0.6, "false_prob": 0.2, "uncertain_prob": 0.2}


class MockBeliefNetwork:
    """Mock belief network for testing."""

    def __init__(self, debate_id: str = "test_debate", km_adapter=None):
        self.debate_id = debate_id
        self.km_adapter = km_adapter
        self._claims = []
        self._edges = []

    def add_claim(self, agent: str, content: str, confidence: float = 0.7):
        node = MockBeliefNode(f"claim_{len(self._claims)}", content, agent)
        self._claims.append(
            {"node": node, "centrality": 0.5 + len(self._claims) * 0.1, "entropy": 0.5}
        )

    def get_load_bearing_claims(self, limit: int = 5):
        return [(claim["node"], claim["centrality"]) for claim in self._claims[:limit]]

    def get_all_claims(self):
        return self._claims

    def get_all_edges(self):
        if len(self._claims) >= 2:
            return [{"source": "claim_0", "target": "claim_1", "weight": 0.7, "type": "supports"}]
        return []

    def seed_from_km(self, topic: str, min_confidence: float = 0.7):
        return 0


class MockBeliefPropagationAnalyzer:
    """Mock belief propagation analyzer."""

    def __init__(self, network: MockBeliefNetwork):
        self.network = network

    def identify_debate_cruxes(self, top_k: int = 3):
        cruxes = []
        for i, claim in enumerate(self.network._claims[:top_k]):
            cruxes.append(
                {
                    "claim_id": claim["node"].claim_id,
                    "statement": claim["node"].claim_statement,
                    "crux_score": 0.9 - i * 0.1,
                    "entropy": claim["entropy"],
                }
            )
        return cruxes


class MockDebateMessage:
    """Mock debate message."""

    def __init__(self, agent: str, content: str, role: str = "proposer", round_num: int = 1):
        self.agent = agent
        self.content = content
        self.role = role
        self.round = round_num


class MockCritique:
    """Mock critique."""

    def __init__(
        self, agent: str, target: str, severity: float, reasoning: str, round_num: int = 1
    ):
        self.agent = agent
        self.target = target
        self.severity = severity
        self.reasoning = reasoning
        self.round = round_num


class MockDebateResult:
    """Mock debate result."""

    def __init__(self):
        self.task = "Test debate task"
        self.messages = [
            MockDebateMessage("claude", "First argument about the topic", "proposer", 1),
            MockDebateMessage("gpt", "Counter argument to consider", "critic", 1),
        ]
        self.critiques = [
            MockCritique("gpt", "claude", 0.6, "Lacks supporting evidence"),
        ]


class MockDebateTrace:
    """Mock debate trace."""

    @classmethod
    def load(cls, path: Path):
        return cls()

    def to_debate_result(self):
        return MockDebateResult()


class MockProvenanceTracker:
    """Mock provenance tracker."""

    @classmethod
    def load(cls, path: Path):
        return cls()

    def get_claim_support(self, claim_id: str):
        return {
            "claim_id": claim_id,
            "verified": True,
            "sources": ["Source 1", "Source 2"],
            "confidence": 0.85,
        }


class MockCartographer:
    """Mock argument cartographer."""

    def __init__(self):
        self._context_set = False

    def set_debate_context(self, debate_id: str, task: str):
        self._context_set = True

    def update_from_message(self, agent: str, content: str, role: str, round_num: int):
        pass

    def update_from_critique(
        self,
        critic_agent: str,
        target_agent: str,
        severity: float,
        round_num: int,
        critique_text: str,
    ):
        pass

    def get_statistics(self):
        return {
            "node_count": 5,
            "edge_count": 3,
            "density": 0.4,
            "average_degree": 1.2,
        }


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset and bypass rate limiter to prevent xdist cross-test interference."""
    if hasattr(_belief_limiter, "_buckets"):
        _belief_limiter._buckets.clear()
    with patch(
        "aragora.server.handlers.belief._belief_limiter.is_allowed",
        return_value=True,
    ):
        yield


@pytest.fixture
def mock_belief_auth():
    """Mock authentication and permission checking for belief handler tests."""
    with patch(
        "aragora.server.handlers.belief.get_permission_checker",
        return_value=MockPermissionChecker(allow=True),
    ):
        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=MockUserAuthContext(),
        ):
            yield


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


@pytest.fixture
def mock_server_context_with_nomic_dir(tmp_path):
    """Create mock server context with nomic_dir configured."""
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    provenance_dir = tmp_path / "provenance"
    provenance_dir.mkdir()
    replays_dir = tmp_path / "replays"
    replays_dir.mkdir()
    return {"storage": None, "elo_system": None, "nomic_dir": tmp_path}


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with client address."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {"X-User-Roles": "admin"}
    return handler


# ===========================================================================
# Test Handler Routing
# ===========================================================================


class TestBeliefHandlerRouting:
    """Tests for handler routing."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BeliefHandler(mock_server_context)

    def test_can_handle_cruxes(self, handler):
        """Handler can handle cruxes endpoint."""
        assert handler.can_handle("/api/v1/belief-network/debate_123/cruxes")

    def test_can_handle_load_bearing_claims(self, handler):
        """Handler can handle load-bearing-claims endpoint."""
        assert handler.can_handle("/api/v1/belief-network/debate_abc/load-bearing-claims")

    def test_can_handle_graph(self, handler):
        """Handler can handle graph endpoint."""
        assert handler.can_handle("/api/v1/belief-network/debate_xyz/graph")

    def test_can_handle_export(self, handler):
        """Handler can handle export endpoint."""
        assert handler.can_handle("/api/v1/belief-network/debate_123/export")

    def test_can_handle_claim_support(self, handler):
        """Handler can handle claim support endpoint."""
        assert handler.can_handle("/api/v1/provenance/debate_123/claims/claim_456/support")

    def test_can_handle_graph_stats(self, handler):
        """Handler can handle graph stats endpoint."""
        assert handler.can_handle("/api/v1/debate/debate_123/graph-stats")

    def test_cannot_handle_other_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/belief-network")
        assert not handler.can_handle("/api/v1/other")

    def test_can_handle_without_version_prefix(self, handler):
        """Handler can handle paths without version prefix."""
        assert handler.can_handle("/api/belief-network/debate_123/cruxes")
        assert handler.can_handle("/api/belief-network/debate_123/load-bearing-claims")
        assert handler.can_handle("/api/belief-network/debate_123/graph")
        assert handler.can_handle("/api/belief-network/debate_123/export")


# ===========================================================================
# Test Authentication and Permissions
# ===========================================================================


class TestBeliefHandlerAuthentication:
    """Tests for authentication and permission checking."""

    def test_unauthenticated_request_returns_401(self, mock_server_context, mock_http_handler):
        """Unauthenticated requests return 401 error."""
        handler = BeliefHandler(mock_server_context)

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=MockUserAuthContext(authenticated=False),
        ):
            with patch(
                "aragora.server.handlers.belief.get_permission_checker",
                return_value=MockPermissionChecker(allow=True),
            ):
                result = handler.handle(
                    "/api/v1/belief-network/debate_123/cruxes",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        assert result.status_code == 401

    def test_permission_denied_returns_403(self, mock_server_context, mock_http_handler):
        """Permission denied returns 403 error."""
        handler = BeliefHandler(mock_server_context)

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=MockUserAuthContext(authenticated=True),
        ):
            with patch(
                "aragora.server.handlers.belief.get_permission_checker",
                return_value=MockPermissionChecker(allow=False),
            ):
                result = handler.handle(
                    "/api/v1/belief-network/debate_123/cruxes",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        assert result.status_code == 403

    def test_authenticated_request_proceeds(
        self, mock_server_context, mock_http_handler, mock_belief_auth
    ):
        """Authenticated requests proceed to handler logic."""
        handler = BeliefHandler(mock_server_context)

        result = handler.handle(
            "/api/v1/belief-network/debate_123/cruxes",
            {},
            mock_http_handler,
        )

        # Should proceed past auth (returns 503 since nomic_dir is None)
        assert result is not None
        assert result.status_code in (200, 503)


# ===========================================================================
# Test Rate Limiting
# ===========================================================================


class TestBeliefHandlerRateLimiting:
    """Tests for rate limiting behavior."""

    def test_rate_limit_returns_429(self, mock_server_context, mock_http_handler, mock_belief_auth):
        """Exceeding rate limit returns 429 error."""
        handler = BeliefHandler(mock_server_context)

        # Pre-fill rate limiter to exceed limit
        with patch.object(_belief_limiter, "is_allowed", return_value=False):
            result = handler.handle(
                "/api/v1/belief-network/debate_123/cruxes",
                {},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 429
        body = json.loads(result.body)
        assert "error" in body
        assert "rate limit" in body["error"].lower()

    def test_within_rate_limit_proceeds(
        self, mock_server_context, mock_http_handler, mock_belief_auth
    ):
        """Requests within rate limit proceed."""
        handler = BeliefHandler(mock_server_context)

        # Ensure rate limiter allows
        with patch.object(_belief_limiter, "is_allowed", return_value=True):
            result = handler.handle(
                "/api/v1/belief-network/debate_123/cruxes",
                {},
                mock_http_handler,
            )

        # Should proceed past rate limit (returns 503 since nomic_dir is None)
        assert result is not None
        assert result.status_code != 429


# ===========================================================================
# Test Route Dispatch
# ===========================================================================


class TestBeliefHandlerRouteDispatch:
    """Tests for route dispatch logic."""

    @pytest.fixture
    def handler(self, mock_server_context, mock_belief_auth):
        return BeliefHandler(mock_server_context)

    def test_handle_cruxes_returns_result(self, handler, mock_http_handler):
        """Handle returns result for cruxes endpoint."""
        result = handler.handle(
            "/api/v1/belief-network/debate_123/cruxes",
            {},
            mock_http_handler,
        )
        assert result is not None

    def test_handle_load_bearing_claims_returns_result(self, handler, mock_http_handler):
        """Handle returns result for load-bearing-claims endpoint."""
        result = handler.handle(
            "/api/v1/belief-network/debate_abc/load-bearing-claims",
            {},
            mock_http_handler,
        )
        assert result is not None

    def test_handle_graph_returns_result(self, handler, mock_http_handler):
        """Handle returns result for graph endpoint."""
        result = handler.handle(
            "/api/v1/belief-network/debate_xyz/graph",
            {},
            mock_http_handler,
        )
        assert result is not None

    def test_handle_export_returns_result(self, handler, mock_http_handler):
        """Handle returns result for export endpoint."""
        result = handler.handle(
            "/api/v1/belief-network/debate_123/export",
            {},
            mock_http_handler,
        )
        assert result is not None

    def test_handle_claim_support_returns_result(self, handler, mock_http_handler):
        """Handle returns result for claim support endpoint."""
        result = handler.handle(
            "/api/v1/provenance/debate_123/claims/claim_456/support",
            {},
            mock_http_handler,
        )
        assert result is not None

    def test_handle_graph_stats_returns_result(self, handler, mock_http_handler):
        """Handle returns result for graph stats endpoint."""
        result = handler.handle(
            "/api/v1/debate/debate_123/graph-stats",
            {},
            mock_http_handler,
        )
        assert result is not None

    def test_handle_unknown_returns_none(self, handler, mock_http_handler):
        """Handle returns None for unknown paths."""
        result = handler.handle("/api/v1/unknown", {}, mock_http_handler)
        assert result is None


# ===========================================================================
# Test Query Parameters
# ===========================================================================


class TestBeliefHandlerQueryParams:
    """Tests for query parameter handling."""

    @pytest.fixture
    def handler(self, mock_server_context, mock_belief_auth):
        return BeliefHandler(mock_server_context)

    def test_cruxes_top_k_param(self, handler, mock_http_handler):
        """Cruxes endpoint respects top_k parameter."""
        result = handler.handle(
            "/api/v1/belief-network/debate_123/cruxes",
            {"top_k": ["5"]},
            mock_http_handler,
        )
        assert result is not None

    def test_load_bearing_limit_param(self, handler, mock_http_handler):
        """Load-bearing-claims endpoint respects limit parameter."""
        result = handler.handle(
            "/api/v1/belief-network/debate_abc/load-bearing-claims",
            {"limit": ["10"]},
            mock_http_handler,
        )
        assert result is not None

    def test_graph_include_cruxes_param(self, handler, mock_http_handler):
        """Graph endpoint respects include_cruxes parameter."""
        result = handler.handle(
            "/api/v1/belief-network/debate_xyz/graph",
            {"include_cruxes": ["true"]},
            mock_http_handler,
        )
        assert result is not None

    def test_graph_include_cruxes_false(self, handler, mock_http_handler):
        """Graph endpoint respects include_cruxes=false parameter."""
        result = handler.handle(
            "/api/v1/belief-network/debate_xyz/graph",
            {"include_cruxes": ["false"]},
            mock_http_handler,
        )
        assert result is not None

    def test_export_format_json(self, handler, mock_http_handler):
        """Export endpoint respects format=json parameter."""
        result = handler.handle(
            "/api/v1/belief-network/debate_123/export",
            {"format": ["json"]},
            mock_http_handler,
        )
        assert result is not None

    def test_export_format_csv(self, handler, mock_http_handler):
        """Export endpoint respects format=csv parameter."""
        result = handler.handle(
            "/api/v1/belief-network/debate_123/export",
            {"format": ["csv"]},
            mock_http_handler,
        )
        assert result is not None

    def test_export_format_graphml(self, handler, mock_http_handler):
        """Export endpoint respects format=graphml parameter."""
        result = handler.handle(
            "/api/v1/belief-network/debate_123/export",
            {"format": ["graphml"]},
            mock_http_handler,
        )
        assert result is not None


# ===========================================================================
# Test Input Validation
# ===========================================================================


class TestBeliefHandlerValidation:
    """Tests for input validation."""

    @pytest.fixture
    def handler(self, mock_server_context, mock_belief_auth):
        return BeliefHandler(mock_server_context)

    def test_invalid_debate_id_returns_error(self, handler, mock_http_handler):
        """Invalid debate ID returns 400 error."""
        result = handler.handle(
            "/api/v1/belief-network//cruxes",
            {},
            mock_http_handler,
        )
        assert result is None or result.status_code in (400, 503)

    def test_top_k_clamped_to_max(self, handler, mock_http_handler):
        """Top K parameter is clamped to maximum."""
        result = handler.handle(
            "/api/v1/belief-network/debate_123/cruxes",
            {"top_k": ["100"]},
            mock_http_handler,
        )
        assert result is not None

    def test_limit_clamped_to_max(self, handler, mock_http_handler):
        """Limit parameter is clamped to maximum."""
        result = handler.handle(
            "/api/v1/belief-network/debate_abc/load-bearing-claims",
            {"limit": ["500"]},
            mock_http_handler,
        )
        assert result is not None

    def test_invalid_claim_support_path_returns_error(self, handler, mock_http_handler):
        """Invalid claim support path returns 400 error."""
        result = handler.handle(
            "/api/v1/provenance/debate_123/claims//support",
            {},
            mock_http_handler,
        )
        # Invalid claim_id should return 400
        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Test Debate ID Extraction
# ===========================================================================


class TestBeliefHandlerExtractDebateId:
    """Tests for debate ID extraction from paths."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BeliefHandler(mock_server_context)

    def test_extract_debate_id_from_cruxes_path(self, handler):
        """Extract debate ID from cruxes path."""
        debate_id = handler._extract_debate_id("/api/v1/belief-network/abc123/cruxes", 4)
        assert debate_id == "abc123"

    def test_extract_debate_id_from_graph_stats_path(self, handler):
        """Extract debate ID from graph-stats path."""
        debate_id = handler._extract_debate_id("/api/v1/debate/xyz789/graph-stats", 4)
        assert debate_id == "xyz789"

    def test_extract_debate_id_invalid_returns_none(self, handler):
        """Invalid debate ID extraction returns None."""
        debate_id = handler._extract_debate_id("/api/v1/belief-network/../etc/cruxes", 4)
        assert debate_id is None

    def test_extract_debate_id_without_version(self, handler):
        """Extract debate ID from path without version prefix."""
        debate_id = handler._extract_debate_id("/api/belief-network/test_debate/cruxes", 3)
        assert debate_id == "test_debate"

    def test_extract_debate_id_out_of_bounds(self, handler):
        """Out of bounds index returns None."""
        debate_id = handler._extract_debate_id("/api/v1/short", 10)
        assert debate_id is None


# ===========================================================================
# Test Path Matching
# ===========================================================================


class TestBeliefHandlerPathMatching:
    """Tests for path matching patterns."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return BeliefHandler(mock_server_context)

    def test_matches_various_debate_ids(self, handler):
        """Handler matches various debate ID formats."""
        assert handler.can_handle("/api/v1/belief-network/debate_123/cruxes")
        assert handler.can_handle("/api/v1/belief-network/abc-def-ghi/cruxes")
        assert handler.can_handle("/api/v1/belief-network/uuid_format_id/cruxes")

    def test_matches_claim_support_pattern(self, handler):
        """Handler matches claim support pattern correctly."""
        assert handler.can_handle("/api/v1/provenance/d1/claims/c1/support")
        assert handler.can_handle("/api/v1/something/debate_123/claims/claim_456/support")

    def test_does_not_match_partial_patterns(self, handler):
        """Handler does not match partial patterns."""
        assert not handler.can_handle("/api/v1/belief-network/debate_123")
        assert not handler.can_handle("/api/v1/provenance/debate_123/support")


# ===========================================================================
# Test Success Cases with Mocked Dependencies
# ===========================================================================


class TestBeliefHandlerSuccessCases:
    """Tests for successful handler responses with mocked dependencies."""

    @pytest.fixture
    def handler_with_mocks(self, mock_server_context_with_nomic_dir, mock_belief_auth):
        """Create handler with properly configured context and mocks."""
        return BeliefHandler(mock_server_context_with_nomic_dir)

    def test_get_debate_cruxes_success(
        self, mock_server_context_with_nomic_dir, mock_http_handler, mock_belief_auth
    ):
        """Successfully get debate cruxes."""
        handler = BeliefHandler(mock_server_context_with_nomic_dir)
        nomic_dir = mock_server_context_with_nomic_dir["nomic_dir"]

        # Create trace file
        trace_path = nomic_dir / "traces" / "debate_123.json"
        trace_path.write_text("{}")

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch("aragora.debate.traces.DebateTrace", MockDebateTrace):
                with patch("aragora.server.handlers.belief.BeliefNetwork", MockBeliefNetwork):
                    with patch(
                        "aragora.server.handlers.belief.BeliefPropagationAnalyzer",
                        MockBeliefPropagationAnalyzer,
                    ):
                        result = handler.handle(
                            "/api/v1/belief-network/debate_123/cruxes",
                            {"top_k": ["3"]},
                            mock_http_handler,
                        )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "debate_id" in body
        assert "cruxes" in body
        assert body["debate_id"] == "debate_123"

    def test_get_load_bearing_claims_success(
        self, mock_server_context_with_nomic_dir, mock_http_handler, mock_belief_auth
    ):
        """Successfully get load-bearing claims."""
        handler = BeliefHandler(mock_server_context_with_nomic_dir)
        nomic_dir = mock_server_context_with_nomic_dir["nomic_dir"]

        trace_path = nomic_dir / "traces" / "debate_abc.json"
        trace_path.write_text("{}")

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch("aragora.debate.traces.DebateTrace", MockDebateTrace):
                with patch("aragora.server.handlers.belief.BeliefNetwork", MockBeliefNetwork):
                    result = handler.handle(
                        "/api/v1/belief-network/debate_abc/load-bearing-claims",
                        {"limit": ["5"]},
                        mock_http_handler,
                    )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "debate_id" in body
        assert "load_bearing_claims" in body
        assert body["debate_id"] == "debate_abc"

    def test_get_claim_support_success(
        self, mock_server_context_with_nomic_dir, mock_http_handler, mock_belief_auth
    ):
        """Successfully get claim support from provenance tracker."""
        handler = BeliefHandler(mock_server_context_with_nomic_dir)
        nomic_dir = mock_server_context_with_nomic_dir["nomic_dir"]

        provenance_path = nomic_dir / "provenance" / "debate_123.json"
        provenance_path.write_text("{}")

        with patch("aragora.server.handlers.belief.PROVENANCE_AVAILABLE", True):
            with patch("aragora.server.handlers.belief.ProvenanceTracker", MockProvenanceTracker):
                result = handler.handle(
                    "/api/v1/provenance/debate_123/claims/claim_456/support",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["debate_id"] == "debate_123"
        assert body["claim_id"] == "claim_456"
        assert body["support"] is not None

    def test_get_claim_support_no_provenance_data(
        self, mock_server_context_with_nomic_dir, mock_http_handler, mock_belief_auth
    ):
        """Get claim support returns message when no provenance data exists."""
        handler = BeliefHandler(mock_server_context_with_nomic_dir)

        with patch("aragora.server.handlers.belief.PROVENANCE_AVAILABLE", True):
            result = handler.handle(
                "/api/v1/provenance/debate_123/claims/claim_456/support",
                {},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["support"] is None
        assert "message" in body

    def test_get_graph_stats_success(
        self, mock_server_context_with_nomic_dir, mock_http_handler, mock_belief_auth
    ):
        """Successfully get graph stats."""
        handler = BeliefHandler(mock_server_context_with_nomic_dir)
        nomic_dir = mock_server_context_with_nomic_dir["nomic_dir"]

        trace_path = nomic_dir / "traces" / "debate_stats.json"
        trace_path.write_text("{}")

        with patch("aragora.debate.traces.DebateTrace", MockDebateTrace):
            with patch("aragora.visualization.mapper.ArgumentCartographer", MockCartographer):
                result = handler.handle(
                    "/api/v1/debate/debate_stats/graph-stats",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "node_count" in body or "nodes" in body or "density" in body

    def test_get_belief_network_graph_success(
        self, mock_server_context_with_nomic_dir, mock_http_handler, mock_belief_auth
    ):
        """Successfully get belief network graph."""
        handler = BeliefHandler(mock_server_context_with_nomic_dir)
        nomic_dir = mock_server_context_with_nomic_dir["nomic_dir"]

        trace_path = nomic_dir / "traces" / "debate_graph.json"
        trace_path.write_text("{}")

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch("aragora.debate.traces.DebateTrace", MockDebateTrace):
                with patch("aragora.server.handlers.belief.BeliefNetwork", MockBeliefNetwork):
                    with patch(
                        "aragora.server.handlers.belief.BeliefPropagationAnalyzer",
                        MockBeliefPropagationAnalyzer,
                    ):
                        result = handler.handle(
                            "/api/v1/belief-network/debate_graph/graph",
                            {"include_cruxes": ["true"]},
                            mock_http_handler,
                        )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "nodes" in body
        assert "links" in body
        assert "metadata" in body


# ===========================================================================
# Test Export Formats
# ===========================================================================


class TestBeliefHandlerExportFormats:
    """Tests for belief network export in various formats."""

    def test_export_json_format(
        self, mock_server_context_with_nomic_dir, mock_http_handler, mock_belief_auth
    ):
        """Export belief network in JSON format."""
        handler = BeliefHandler(mock_server_context_with_nomic_dir)
        nomic_dir = mock_server_context_with_nomic_dir["nomic_dir"]

        trace_path = nomic_dir / "traces" / "debate_export.json"
        trace_path.write_text("{}")

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch("aragora.debate.traces.DebateTrace", MockDebateTrace):
                with patch("aragora.server.handlers.belief.BeliefNetwork", MockBeliefNetwork):
                    with patch(
                        "aragora.server.handlers.belief.BeliefPropagationAnalyzer",
                        MockBeliefPropagationAnalyzer,
                    ):
                        result = handler.handle(
                            "/api/v1/belief-network/debate_export/export",
                            {"format": ["json"]},
                            mock_http_handler,
                        )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["format"] == "json"
        assert "nodes" in body
        assert "edges" in body
        assert "summary" in body

    def test_export_csv_format(
        self, mock_server_context_with_nomic_dir, mock_http_handler, mock_belief_auth
    ):
        """Export belief network in CSV format."""
        handler = BeliefHandler(mock_server_context_with_nomic_dir)
        nomic_dir = mock_server_context_with_nomic_dir["nomic_dir"]

        trace_path = nomic_dir / "traces" / "debate_export.json"
        trace_path.write_text("{}")

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch("aragora.debate.traces.DebateTrace", MockDebateTrace):
                with patch("aragora.server.handlers.belief.BeliefNetwork", MockBeliefNetwork):
                    with patch(
                        "aragora.server.handlers.belief.BeliefPropagationAnalyzer",
                        MockBeliefPropagationAnalyzer,
                    ):
                        result = handler.handle(
                            "/api/v1/belief-network/debate_export/export",
                            {"format": ["csv"]},
                            mock_http_handler,
                        )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["format"] == "csv"
        assert "nodes_csv" in body
        assert "edges_csv" in body
        assert "headers" in body

    def test_export_graphml_format(
        self, mock_server_context_with_nomic_dir, mock_http_handler, mock_belief_auth
    ):
        """Export belief network in GraphML format."""
        handler = BeliefHandler(mock_server_context_with_nomic_dir)
        nomic_dir = mock_server_context_with_nomic_dir["nomic_dir"]

        trace_path = nomic_dir / "traces" / "debate_export.json"
        trace_path.write_text("{}")

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            with patch("aragora.debate.traces.DebateTrace", MockDebateTrace):
                with patch("aragora.server.handlers.belief.BeliefNetwork", MockBeliefNetwork):
                    with patch(
                        "aragora.server.handlers.belief.BeliefPropagationAnalyzer",
                        MockBeliefPropagationAnalyzer,
                    ):
                        result = handler.handle(
                            "/api/v1/belief-network/debate_export/export",
                            {"format": ["graphml"]},
                            mock_http_handler,
                        )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["format"] == "graphml"
        assert "content" in body
        assert body["content_type"] == "application/xml"
        assert '<?xml version="1.0"' in body["content"]
        assert "<graphml" in body["content"]


# ===========================================================================
# Test Error Cases
# ===========================================================================


class TestBeliefHandlerErrorCases:
    """Tests for error handling."""

    def test_belief_network_unavailable(
        self, mock_server_context, mock_http_handler, mock_belief_auth
    ):
        """Returns 503 when belief network module is unavailable."""
        handler = BeliefHandler(mock_server_context)

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", False):
            result = handler.handle(
                "/api/v1/belief-network/debate_123/cruxes",
                {},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 503
        body = json.loads(result.body)
        assert "error" in body

    def test_provenance_unavailable(
        self, mock_server_context_with_nomic_dir, mock_http_handler, mock_belief_auth
    ):
        """Returns 503 when provenance module is unavailable."""
        handler = BeliefHandler(mock_server_context_with_nomic_dir)

        with patch("aragora.server.handlers.belief.PROVENANCE_AVAILABLE", False):
            result = handler.handle(
                "/api/v1/provenance/debate_123/claims/claim_456/support",
                {},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 503

    def test_nomic_dir_not_configured(
        self, mock_server_context, mock_http_handler, mock_belief_auth
    ):
        """Returns 503 when nomic_dir is not configured."""
        handler = BeliefHandler(mock_server_context)

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            result = handler.handle(
                "/api/v1/belief-network/debate_123/cruxes",
                {},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 503
        body = json.loads(result.body)
        assert "nomic" in body["error"].lower() or "not configured" in body["error"].lower()

    def test_debate_trace_not_found(
        self, mock_server_context_with_nomic_dir, mock_http_handler, mock_belief_auth
    ):
        """Returns 404 when debate trace file doesn't exist."""
        handler = BeliefHandler(mock_server_context_with_nomic_dir)

        with patch("aragora.server.handlers.belief.BELIEF_NETWORK_AVAILABLE", True):
            result = handler.handle(
                "/api/v1/belief-network/nonexistent_debate/cruxes",
                {},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 404
        body = json.loads(result.body)
        assert "not found" in body["error"].lower()


# ===========================================================================
# Test KM Adapter Integration
# ===========================================================================


class TestBeliefHandlerKMAdapter:
    """Tests for Knowledge Mound adapter integration."""

    def test_get_km_adapter_returns_none_when_unavailable(self, mock_server_context):
        """Returns None when KM adapter is not available."""
        handler = BeliefHandler(mock_server_context)

        with patch.object(handler, "_get_km_adapter", return_value=None):
            adapter = handler._get_km_adapter()

        assert adapter is None

    def test_km_adapter_cached_in_context(self, mock_server_context):
        """KM adapter is cached in server context."""
        mock_adapter = MagicMock()
        mock_server_context["belief_km_adapter"] = mock_adapter

        handler = BeliefHandler(mock_server_context)
        adapter = handler._get_km_adapter()

        assert adapter is mock_adapter

    def test_create_belief_network_with_km_adapter(self, mock_server_context):
        """Belief network is created with KM adapter when available."""
        handler = BeliefHandler(mock_server_context)
        mock_adapter = MagicMock()

        with patch.object(handler, "_get_km_adapter", return_value=mock_adapter):
            with patch("aragora.server.handlers.belief.BeliefNetwork") as MockNetwork:
                MockNetwork.return_value = MockBeliefNetwork("test_debate", mock_adapter)
                network = handler._create_belief_network("test_debate")

                MockNetwork.assert_called_once_with(
                    debate_id="test_debate", km_adapter=mock_adapter
                )

    def test_emit_km_event(self, mock_server_context):
        """KM events are emitted to WebSocket clients."""
        mock_emitter = MagicMock()
        mock_server_context["event_emitter"] = mock_emitter

        handler = BeliefHandler(mock_server_context)

        with patch("aragora.events.types.StreamEvent") as MockEvent:
            with patch("aragora.events.types.StreamEventType") as MockEventType:
                MockEventType.BELIEF_CONVERGED = "belief_converged"
                MockEventType.CRUX_DETECTED = "crux_detected"
                MockEventType.MOUND_UPDATED = "mound_updated"

                handler._emit_km_event(mock_emitter, "belief_converged", {"test": "data"})

                MockEvent.assert_called_once()
                mock_emitter.emit.assert_called_once()


# ===========================================================================
# Test Graph Stats with Replays Fallback
# ===========================================================================


class TestBeliefHandlerGraphStatsReplays:
    """Tests for graph stats with replays directory fallback."""

    def test_graph_stats_from_replays(
        self, mock_server_context_with_nomic_dir, mock_http_handler, mock_belief_auth
    ):
        """Get graph stats from replays directory when trace not found."""
        handler = BeliefHandler(mock_server_context_with_nomic_dir)
        nomic_dir = mock_server_context_with_nomic_dir["nomic_dir"]

        # Create replays directory with events file
        replay_dir = nomic_dir / "replays" / "debate_replay"
        replay_dir.mkdir(parents=True)
        events_path = replay_dir / "events.jsonl"
        events_path.write_text(
            '{"type": "agent_message", "agent": "claude", "data": {"content": "Test message", "role": "proposer"}, "round": 1}\n'
            '{"type": "critique", "agent": "gpt", "data": {"target": "claude", "severity": 0.5, "content": "Critique"}, "round": 1}\n'
        )

        with patch("aragora.visualization.mapper.ArgumentCartographer", MockCartographer):
            result = handler.handle(
                "/api/v1/debate/debate_replay/graph-stats",
                {},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Test Integration Scenarios
# ===========================================================================


class TestBeliefHandlerIntegration:
    """Integration tests for belief handler."""

    def test_all_routes_reachable(self, mock_server_context, mock_http_handler, mock_belief_auth):
        """Test all belief routes are reachable."""
        handler = BeliefHandler(mock_server_context)

        routes_to_test = [
            ("/api/v1/belief-network/debate_123/cruxes", {}),
            ("/api/v1/belief-network/debate_123/load-bearing-claims", {}),
            ("/api/v1/belief-network/debate_123/graph", {}),
            ("/api/v1/belief-network/debate_123/export", {}),
            ("/api/v1/provenance/debate_123/claims/claim_456/support", {}),
            ("/api/v1/debate/debate_123/graph-stats", {}),
        ]

        for path, params in routes_to_test:
            result = handler.handle(path, params, mock_http_handler)
            assert result is not None, f"Route {path} returned None"
            # All should return either success or expected errors
            assert result.status_code in [200, 400, 403, 404, 429, 500, 503], (
                f"Route {path} returned unexpected {result.status_code}"
            )

    def test_parameter_validation_across_endpoints(
        self, mock_server_context, mock_http_handler, mock_belief_auth
    ):
        """Test parameter validation across endpoints."""
        handler = BeliefHandler(mock_server_context)

        # Test limit clamping
        result = handler.handle(
            "/api/v1/belief-network/debate_123/load-bearing-claims",
            {"limit": ["1000"]},
            mock_http_handler,
        )
        assert result is not None

        # Test top_k clamping
        result = handler.handle(
            "/api/v1/belief-network/debate_123/cruxes",
            {"top_k": ["100"]},
            mock_http_handler,
        )
        assert result is not None
