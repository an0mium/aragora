"""Tests for graph debates handler.

Tests the graph debates API endpoints including:
- POST /api/debates/graph - Run a graph-structured debate with branching
- GET /api/debates/graph/{id} - Get graph debate by ID
- GET /api/debates/graph/{id}/branches - Get all branches for a debate
- GET /api/debates/graph/{id}/nodes - Get all nodes in debate graph
"""

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def graph_handler():
    """Create graph debates handler with mock context."""
    from aragora.server.handlers.debates.graph_debates import GraphDebatesHandler

    ctx = {}
    handler = GraphDebatesHandler(ctx)
    return handler


@pytest.fixture(autouse=True)
def reset_state():
    """Reset state before each test."""
    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters

        reset_rate_limiters()
    except ImportError:
        pass

    # Also reset the module-level rate limiter
    try:
        from aragora.server.handlers.debates import graph_debates

        graph_debates._graph_limiter = graph_debates.RateLimiter(requests_per_minute=5)
    except (ImportError, AttributeError):
        pass

    yield

    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters

        reset_rate_limiters()
    except ImportError:
        pass


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {}
    handler.event_emitter = None
    return handler


# =============================================================================
# Initialization Tests
# =============================================================================


class TestGraphDebatesHandlerInit:
    """Tests for handler initialization."""

    def test_routes_defined(self, graph_handler):
        """Test that handler routes are defined."""
        assert hasattr(graph_handler, "ROUTES")
        assert len(graph_handler.ROUTES) > 0

    def test_can_handle_graph_path(self, graph_handler):
        """Test can_handle recognizes graph paths."""
        assert graph_handler.can_handle("/api/debates/graph")
        assert graph_handler.can_handle("/api/debates/graph/")
        assert graph_handler.can_handle("/api/debates/graph/abc123")
        assert graph_handler.can_handle("/api/debates/graph/abc123/branches")
        assert graph_handler.can_handle("/api/debates/graph/abc123/nodes")

    def test_cannot_handle_other_paths(self, graph_handler):
        """Test can_handle rejects non-graph paths."""
        assert not graph_handler.can_handle("/api/debates")
        assert not graph_handler.can_handle("/api/debates/abc123")
        assert not graph_handler.can_handle("/api/debates/matrix")
        assert not graph_handler.can_handle("/api/users")


# =============================================================================
# POST Validation Tests
# =============================================================================


class TestGraphDebatePostValidation:
    """Tests for POST request validation."""

    @pytest.mark.asyncio
    async def test_returns_404_for_wrong_path(self, graph_handler, mock_http_handler):
        """Returns 404 for non-graph POST paths."""
        result = await graph_handler.handle_post(
            mock_http_handler, "/api/debates/other", {}
        )
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_400_without_task(self, graph_handler, mock_http_handler):
        """Returns 400 when task is missing."""
        result = await graph_handler.handle_post(
            mock_http_handler, "/api/debates/graph", {"agents": ["claude", "gpt4"]}
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "task" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_returns_400_for_non_string_task(self, graph_handler, mock_http_handler):
        """Returns 400 when task is not a string."""
        result = await graph_handler.handle_post(
            mock_http_handler,
            "/api/debates/graph",
            {"task": 12345, "agents": ["claude", "gpt4"]},
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "string" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_returns_400_for_short_task(self, graph_handler, mock_http_handler):
        """Returns 400 when task is too short."""
        result = await graph_handler.handle_post(
            mock_http_handler,
            "/api/debates/graph",
            {"task": "Short", "agents": ["claude", "gpt4"]},
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "10 characters" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_returns_400_for_long_task(self, graph_handler, mock_http_handler):
        """Returns 400 when task is too long."""
        result = await graph_handler.handle_post(
            mock_http_handler,
            "/api/debates/graph",
            {"task": "A" * 5001, "agents": ["claude", "gpt4"]},
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "5000 characters" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_returns_400_for_suspicious_task_script(
        self, graph_handler, mock_http_handler
    ):
        """Returns 400 when task contains script tag."""
        result = await graph_handler.handle_post(
            mock_http_handler,
            "/api/debates/graph",
            {"task": "What about <script>alert('xss')</script>?", "agents": ["claude", "gpt4"]},
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "invalid characters" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_returns_400_for_suspicious_task_template(
        self, graph_handler, mock_http_handler
    ):
        """Returns 400 when task contains template injection."""
        result = await graph_handler.handle_post(
            mock_http_handler,
            "/api/debates/graph",
            {"task": "Evaluate this: {{config.secret}}", "agents": ["claude", "gpt4"]},
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "invalid characters" in data.get("error", "").lower()


# =============================================================================
# Agent Validation Tests
# =============================================================================


class TestGraphDebateAgentValidation:
    """Tests for agent validation."""

    @pytest.mark.asyncio
    async def test_returns_400_for_non_array_agents(
        self, graph_handler, mock_http_handler
    ):
        """Returns 400 when agents is not an array."""
        result = await graph_handler.handle_post(
            mock_http_handler,
            "/api/debates/graph",
            {"task": "What is the meaning of life and existence?", "agents": "claude"},
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "array" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_returns_400_for_too_few_agents(
        self, graph_handler, mock_http_handler
    ):
        """Returns 400 when less than 2 agents provided."""
        result = await graph_handler.handle_post(
            mock_http_handler,
            "/api/debates/graph",
            {"task": "What is the meaning of life and existence?", "agents": ["claude"]},
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "2 agents" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_returns_400_for_too_many_agents(
        self, graph_handler, mock_http_handler
    ):
        """Returns 400 when more than 10 agents provided."""
        agents = [f"agent{i}" for i in range(11)]
        result = await graph_handler.handle_post(
            mock_http_handler,
            "/api/debates/graph",
            {"task": "What is the meaning of life and existence?", "agents": agents},
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "10 agents" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_returns_400_for_non_string_agent(
        self, graph_handler, mock_http_handler
    ):
        """Returns 400 when agent name is not a string."""
        result = await graph_handler.handle_post(
            mock_http_handler,
            "/api/debates/graph",
            {"task": "What is the meaning of life and existence?", "agents": ["claude", 123]},
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "agents[1]" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_returns_400_for_long_agent_name(
        self, graph_handler, mock_http_handler
    ):
        """Returns 400 when agent name exceeds 50 chars."""
        result = await graph_handler.handle_post(
            mock_http_handler,
            "/api/debates/graph",
            {
                "task": "What is the meaning of life and existence?",
                "agents": ["claude", "a" * 51],
            },
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "50 chars" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_returns_400_for_invalid_agent_name(
        self, graph_handler, mock_http_handler
    ):
        """Returns 400 when agent name contains invalid characters."""
        result = await graph_handler.handle_post(
            mock_http_handler,
            "/api/debates/graph",
            {
                "task": "What is the meaning of life and existence?",
                "agents": ["claude", "agent<script>"],
            },
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "invalid agent name" in data.get("error", "").lower()


# =============================================================================
# Max Rounds Validation Tests
# =============================================================================


class TestGraphDebateRoundsValidation:
    """Tests for max_rounds validation."""

    @pytest.mark.asyncio
    async def test_returns_400_for_invalid_max_rounds(
        self, graph_handler, mock_http_handler
    ):
        """Returns 400 when max_rounds is not a number."""
        result = await graph_handler.handle_post(
            mock_http_handler,
            "/api/debates/graph",
            {
                "task": "What is the meaning of life and existence?",
                "agents": ["claude", "gpt4"],
                "max_rounds": "invalid",
            },
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "max_rounds" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_returns_400_for_zero_max_rounds(
        self, graph_handler, mock_http_handler
    ):
        """Returns 400 when max_rounds is less than 1."""
        result = await graph_handler.handle_post(
            mock_http_handler,
            "/api/debates/graph",
            {
                "task": "What is the meaning of life and existence?",
                "agents": ["claude", "gpt4"],
                "max_rounds": 0,
            },
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "at least 1" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_returns_400_for_too_many_rounds(
        self, graph_handler, mock_http_handler
    ):
        """Returns 400 when max_rounds exceeds 20."""
        result = await graph_handler.handle_post(
            mock_http_handler,
            "/api/debates/graph",
            {
                "task": "What is the meaning of life and existence?",
                "agents": ["claude", "gpt4"],
                "max_rounds": 21,
            },
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "at most 20" in data.get("error", "")


# =============================================================================
# Branch Policy Validation Tests
# =============================================================================


class TestGraphDebateBranchPolicyValidation:
    """Tests for branch_policy validation."""

    @pytest.mark.asyncio
    async def test_returns_400_for_non_dict_branch_policy(
        self, graph_handler, mock_http_handler
    ):
        """Returns 400 when branch_policy is not an object."""
        result = await graph_handler.handle_post(
            mock_http_handler,
            "/api/debates/graph",
            {
                "task": "What is the meaning of life and existence?",
                "agents": ["claude", "gpt4"],
                "branch_policy": "invalid",
            },
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "object" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_returns_400_for_invalid_min_disagreement(
        self, graph_handler, mock_http_handler
    ):
        """Returns 400 when min_disagreement is out of range."""
        result = await graph_handler.handle_post(
            mock_http_handler,
            "/api/debates/graph",
            {
                "task": "What is the meaning of life and existence?",
                "agents": ["claude", "gpt4"],
                "branch_policy": {"min_disagreement": 1.5},
            },
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "min_disagreement" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_returns_400_for_invalid_max_branches(
        self, graph_handler, mock_http_handler
    ):
        """Returns 400 when max_branches is out of range."""
        result = await graph_handler.handle_post(
            mock_http_handler,
            "/api/debates/graph",
            {
                "task": "What is the meaning of life and existence?",
                "agents": ["claude", "gpt4"],
                "branch_policy": {"max_branches": 15},
            },
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "max_branches" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_returns_400_for_invalid_merge_strategy(
        self, graph_handler, mock_http_handler
    ):
        """Returns 400 when merge_strategy is invalid."""
        result = await graph_handler.handle_post(
            mock_http_handler,
            "/api/debates/graph",
            {
                "task": "What is the meaning of life and existence?",
                "agents": ["claude", "gpt4"],
                "branch_policy": {"merge_strategy": "invalid"},
            },
        )
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "merge_strategy" in data.get("error", "")


# =============================================================================
# GET Endpoint Tests
# =============================================================================


class TestGraphDebateGetEndpoints:
    """Tests for GET endpoints."""

    @pytest.mark.asyncio
    async def test_get_returns_404_for_base_path(
        self, graph_handler, mock_http_handler
    ):
        """Returns 404 for GET on base graph path."""
        result = await graph_handler.handle_get(
            mock_http_handler, "/api/debates/graph", {}
        )
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_get_debate_returns_503_without_storage(
        self, graph_handler, mock_http_handler
    ):
        """Returns 503 when storage is not configured."""
        mock_http_handler.storage = None
        result = await graph_handler.handle_get(
            mock_http_handler, "/api/debates/graph/test-123", {}
        )
        assert result.status_code == 503
        data = json.loads(result.body)
        assert "storage" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_get_debate_returns_404_when_not_found(
        self, graph_handler, mock_http_handler
    ):
        """Returns 404 when graph debate doesn't exist."""
        mock_storage = AsyncMock()
        mock_storage.get_graph_debate = AsyncMock(return_value=None)
        mock_http_handler.storage = mock_storage

        result = await graph_handler.handle_get(
            mock_http_handler, "/api/debates/graph/nonexistent", {}
        )
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_get_debate_returns_debate_data(
        self, graph_handler, mock_http_handler
    ):
        """Returns debate data when found."""
        debate_data = {"id": "test-123", "task": "Test task", "nodes": []}
        mock_storage = AsyncMock()
        mock_storage.get_graph_debate = AsyncMock(return_value=debate_data)
        mock_http_handler.storage = mock_storage

        result = await graph_handler.handle_get(
            mock_http_handler, "/api/debates/graph/test-123", {}
        )
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["id"] == "test-123"

    @pytest.mark.asyncio
    async def test_get_branches_returns_503_without_storage(
        self, graph_handler, mock_http_handler
    ):
        """Returns 503 when storage is not configured for branches."""
        mock_http_handler.storage = None
        result = await graph_handler.handle_get(
            mock_http_handler, "/api/debates/graph/test-123/branches", {}
        )
        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_get_branches_returns_branch_data(
        self, graph_handler, mock_http_handler
    ):
        """Returns branch data when found."""
        branches = [{"id": "branch-1"}, {"id": "branch-2"}]
        mock_storage = AsyncMock()
        mock_storage.get_debate_branches = AsyncMock(return_value=branches)
        mock_http_handler.storage = mock_storage

        result = await graph_handler.handle_get(
            mock_http_handler, "/api/debates/graph/test-123/branches", {}
        )
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["debate_id"] == "test-123"
        assert len(data["branches"]) == 2

    @pytest.mark.asyncio
    async def test_get_nodes_returns_503_without_storage(
        self, graph_handler, mock_http_handler
    ):
        """Returns 503 when storage is not configured for nodes."""
        mock_http_handler.storage = None
        result = await graph_handler.handle_get(
            mock_http_handler, "/api/debates/graph/test-123/nodes", {}
        )
        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_get_nodes_returns_node_data(
        self, graph_handler, mock_http_handler
    ):
        """Returns node data when found."""
        nodes = [{"id": "node-1", "content": "First"}, {"id": "node-2", "content": "Second"}]
        mock_storage = AsyncMock()
        mock_storage.get_debate_nodes = AsyncMock(return_value=nodes)
        mock_http_handler.storage = mock_storage

        result = await graph_handler.handle_get(
            mock_http_handler, "/api/debates/graph/test-123/nodes", {}
        )
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["debate_id"] == "test-123"
        assert len(data["nodes"]) == 2


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestGraphDebateRateLimiting:
    """Tests for rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_after_multiple_requests(self, graph_handler):
        """Returns 429 after exceeding rate limit."""
        # Make requests until rate limited
        for i in range(6):  # 5 allowed, 6th should fail
            mock_handler = MagicMock()
            mock_handler.client_address = ("192.168.1.100", 12345)
            mock_handler.headers = {}

            result = await graph_handler.handle_post(
                mock_handler,
                "/api/debates/graph",
                {
                    "task": f"What is the meaning of life and existence? Request {i}",
                    "agents": ["claude", "gpt4"],
                },
            )

            if i >= 5:  # After 5 requests, should be rate limited
                assert result.status_code == 429
                data = json.loads(result.body)
                assert "rate limit" in data.get("error", "").lower()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestGraphDebateErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_handles_import_error(self, graph_handler, mock_http_handler):
        """Returns 500 when graph module import fails."""
        with patch(
            "aragora.server.handlers.debates.graph_debates.GraphDebatesHandler._load_agents",
            new_callable=AsyncMock,
            return_value=[MagicMock(), MagicMock()],
        ):
            # Patch the import to fail
            with patch.dict("sys.modules", {"aragora.debate.graph": None}):
                result = await graph_handler.handle_post(
                    mock_http_handler,
                    "/api/debates/graph",
                    {
                        "task": "What is the meaning of life and existence?",
                        "agents": ["claude", "gpt4"],
                    },
                )
                # Will fail at import or with 500 error
                assert result.status_code in [400, 500]

    @pytest.mark.asyncio
    async def test_handles_no_valid_agents(self, graph_handler, mock_http_handler):
        """Returns 400 when no valid agents are found."""
        with patch.object(
            graph_handler, "_load_agents", new_callable=AsyncMock, return_value=[]
        ):
            result = await graph_handler.handle_post(
                mock_http_handler,
                "/api/debates/graph",
                {
                    "task": "What is the meaning of life and existence?",
                    "agents": ["invalid_agent_1", "invalid_agent_2"],
                },
            )
            assert result.status_code == 400
            data = json.loads(result.body)
            assert "no valid agents" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_get_debate_handles_storage_error(
        self, graph_handler, mock_http_handler
    ):
        """Returns 500 on storage error when getting debate."""
        mock_storage = AsyncMock()
        mock_storage.get_graph_debate = AsyncMock(
            side_effect=RuntimeError("Database error")
        )
        mock_http_handler.storage = mock_storage

        result = await graph_handler.handle_get(
            mock_http_handler, "/api/debates/graph/test-123", {}
        )
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_get_branches_handles_storage_error(
        self, graph_handler, mock_http_handler
    ):
        """Returns 500 on storage error when getting branches."""
        mock_storage = AsyncMock()
        mock_storage.get_debate_branches = AsyncMock(
            side_effect=RuntimeError("Database error")
        )
        mock_http_handler.storage = mock_storage

        result = await graph_handler.handle_get(
            mock_http_handler, "/api/debates/graph/test-123/branches", {}
        )
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_get_nodes_handles_storage_error(
        self, graph_handler, mock_http_handler
    ):
        """Returns 500 on storage error when getting nodes."""
        mock_storage = AsyncMock()
        mock_storage.get_debate_nodes = AsyncMock(
            side_effect=RuntimeError("Database error")
        )
        mock_http_handler.storage = mock_storage

        result = await graph_handler.handle_get(
            mock_http_handler, "/api/debates/graph/test-123/nodes", {}
        )
        assert result.status_code == 500
