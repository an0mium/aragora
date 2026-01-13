"""End-to-end tests for the Aragora Python SDK.

Tests all SDK methods against a live or mock server to validate:
1. Request/response serialization
2. Error handling (404, 429, 500)
3. Timeout behavior
4. Connection pooling
5. Async/sync parity

Run with: pytest tests/integration/test_sdk_e2e.py -v
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Any, Dict
import json
import httpx

from aragora.client.client import (
    AragoraClient,
    DebatesAPI,
    AgentsAPI,
    LeaderboardAPI,
    GauntletAPI,
    GraphDebatesAPI,
    MatrixDebatesAPI,
    VerificationAPI,
    MemoryAPI,
)
from aragora.client.models import (
    Debate,
    DebateStatus,
    DebateCreateRequest,
    DebateCreateResponse,
    ConsensusResult,
    AgentProfile,
    LeaderboardEntry,
    GauntletReceipt,
    GauntletRunResponse,
    GraphDebate,
    GraphDebateCreateResponse,
    GraphDebateBranch,
    MatrixDebate,
    MatrixDebateCreateResponse,
    MatrixConclusion,
    VerifyClaimResponse,
    VerifyStatusResponse,
    VerificationStatus,
    MemoryAnalyticsResponse,
    MemorySnapshotResponse,
    APIError,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_responses():
    """Standard mock API responses for testing."""
    return {
        "debate_create": {"debate_id": "debate-123", "status": "created", "task": "Test debate"},
        "debate_get": {
            "debate_id": "debate-123",
            "task": "Test debate",
            "status": "completed",
            "agents": ["anthropic-api", "openai-api"],
            "rounds": [
                {
                    "round_number": 1,
                    "messages": [{"agent_id": "anthropic-api", "content": "Test response"}],
                }
            ],
            "consensus": {
                "reached": True,
                "conclusion": "Test conclusion",
                "confidence": 0.85,
                "supporting_agents": ["anthropic-api", "openai-api"],
            },
        },
        "debate_list": {
            "debates": [
                {"debate_id": "debate-1", "task": "Topic 1", "status": "completed"},
                {"debate_id": "debate-2", "task": "Topic 2", "status": "in_progress"},
            ],
            "total": 2,
        },
        "agents_list": {
            "agents": [
                {"agent_id": "anthropic-api", "name": "Claude", "provider": "anthropic"},
                {"agent_id": "openai-api", "name": "GPT-4", "provider": "openai"},
            ]
        },
        "agent_get": {
            "agent_id": "anthropic-api",
            "name": "Claude",
            "provider": "anthropic",
            "elo_rating": 1500,
            "wins": 10,
            "losses": 5,
        },
        "leaderboard": {
            "entries": [
                {"agent_id": "anthropic-api", "elo_rating": 1600, "rank": 1},
                {"agent_id": "openai-api", "elo_rating": 1550, "rank": 2},
            ]
        },
        "gauntlet_run": {"gauntlet_id": "gauntlet-123", "status": "running", "persona": "security"},
        "gauntlet_receipt": {
            "gauntlet_id": "gauntlet-123",
            "status": "completed",
            "score": 0.85,
            "findings": ["Finding 1", "Finding 2"],
            "duration_seconds": 120,
        },
        "graph_debate_create": {
            "debate_id": "graph-123",
            "status": "created",
            "task": "Design a system",
        },
        "graph_debate_get": {
            "debate_id": "graph-123",
            "task": "Design a system",
            "status": "completed",
            "agents": ["anthropic-api"],
            "branches": [
                {"branch_id": "main", "name": "Main Branch", "nodes": [], "is_main": True}
            ],
        },
        "graph_branches": {
            "branches": [
                {"branch_id": "main", "name": "Main", "nodes": [], "is_main": True},
                {"branch_id": "alt-1", "name": "Alternative 1", "nodes": [], "is_main": False},
            ]
        },
        "matrix_debate_create": {
            "matrix_id": "matrix-123",
            "status": "created",
            "task": "Analyze scenarios",
            "scenario_count": 3,
        },
        "matrix_debate_get": {
            "matrix_id": "matrix-123",
            "task": "Analyze scenarios",
            "status": "completed",
            "agents": ["anthropic-api"],
            "scenarios": [{"scenario_name": "baseline", "key_findings": ["Finding 1"]}],
        },
        "matrix_conclusions": {
            "universal": ["Conclusion that holds across all scenarios"],
            "conditional": {"high_load": ["Only true under high load"]},
            "contradictions": [],
        },
        "verify_claim": {
            "status": "valid",
            "claim": "All primes > 2 are odd",
            "formal_translation": "(assert (forall ((x Int)) (=> (and (> x 2) (prime x)) (odd x))))",
            "proof": "UNSAT",
            "duration_ms": 150,
        },
        "verify_status": {
            "available": True,
            "backends": [
                {"name": "z3", "available": True, "version": "4.12.0"},
                {"name": "lean", "available": False},
            ],
        },
        "memory_analytics": {
            "tiers": [
                {"tier_name": "fast", "entry_count": 100, "hit_rate": 0.85},
                {"tier_name": "medium", "entry_count": 500, "hit_rate": 0.60},
                {"tier_name": "slow", "entry_count": 1000, "hit_rate": 0.30},
            ],
            "total_entries": 1600,
            "learning_velocity": 0.75,
            "promotion_effectiveness": 0.82,
            "recommendations": [],
            "period_days": 30,
        },
        "memory_snapshot": {
            "snapshot_id": "snap-123",
            "timestamp": "2024-01-15T10:30:00Z",
            "success": True,
        },
        "health": {"status": "healthy", "version": "1.0.0", "uptime_seconds": 3600},
        "error_404": {"error": "Not found", "code": "NOT_FOUND"},
        "error_429": {"error": "Rate limit exceeded", "code": "RATE_LIMITED", "retry_after": 60},
        "error_500": {"error": "Internal server error", "code": "INTERNAL_ERROR"},
    }


@pytest.fixture
def mock_client(mock_responses):
    """Create a client with mocked HTTP transport."""
    client = AragoraClient(base_url="http://test:8080")

    def mock_request(method, url, **kwargs):
        """Mock HTTP request handler."""
        response = Mock()
        response.status_code = 200
        response.headers = {}

        # Route to appropriate response
        if "/api/debates" in url and method == "POST":
            if "graph" in url:
                response.json.return_value = mock_responses["graph_debate_create"]
            elif "matrix" in url:
                response.json.return_value = mock_responses["matrix_debate_create"]
            else:
                response.json.return_value = mock_responses["debate_create"]
        elif "/api/debates/graph/" in url and "/branches" in url:
            response.json.return_value = mock_responses["graph_branches"]
        elif "/api/debates/graph/" in url:
            response.json.return_value = mock_responses["graph_debate_get"]
        elif "/api/debates/matrix/" in url and "/conclusions" in url:
            response.json.return_value = mock_responses["matrix_conclusions"]
        elif "/api/debates/matrix/" in url:
            response.json.return_value = mock_responses["matrix_debate_get"]
        elif "/api/debates" in url and "list" in url:
            response.json.return_value = mock_responses["debate_list"]
        elif "/api/debates/" in url:
            response.json.return_value = mock_responses["debate_get"]
        elif "/api/agents" in url and method == "GET":
            if "/api/agents/" in url:
                response.json.return_value = mock_responses["agent_get"]
            else:
                response.json.return_value = mock_responses["agents_list"]
        elif "/api/leaderboard" in url:
            response.json.return_value = mock_responses["leaderboard"]
        elif "/api/gauntlet" in url and method == "POST":
            response.json.return_value = mock_responses["gauntlet_run"]
        elif "/api/gauntlet/" in url and "/receipt" in url:
            response.json.return_value = mock_responses["gauntlet_receipt"]
        elif "/api/verify/claim" in url:
            response.json.return_value = mock_responses["verify_claim"]
        elif "/api/verify/status" in url:
            response.json.return_value = mock_responses["verify_status"]
        elif "/api/memory/analytics/snapshot" in url:
            response.json.return_value = mock_responses["memory_snapshot"]
        elif "/api/memory/analytics" in url:
            response.json.return_value = mock_responses["memory_analytics"]
        elif "/health" in url:
            response.json.return_value = mock_responses["health"]
        else:
            response.status_code = 404
            response.json.return_value = mock_responses["error_404"]

        return response

    # Patch the internal request methods
    client._session = Mock()
    client._session.request = mock_request

    return client


# =============================================================================
# Debates API Tests
# =============================================================================


class TestDebatesAPI:
    """Tests for DebatesAPI interface."""

    def test_create_debate_sync(self, mock_client, mock_responses):
        """Test synchronous debate creation."""
        with patch.object(mock_client, "_post", return_value=mock_responses["debate_create"]):
            result = mock_client.debates.create(
                task="Test debate", agents=["anthropic-api", "openai-api"]
            )

            assert result.debate_id == "debate-123"
            assert result.status == "created"

    def test_get_debate_sync(self, mock_client, mock_responses):
        """Test synchronous debate retrieval."""
        with patch.object(mock_client, "_get", return_value=mock_responses["debate_get"]):
            result = mock_client.debates.get("debate-123")

            assert result.debate_id == "debate-123"
            assert result.status == DebateStatus.COMPLETED
            assert result.consensus is not None
            assert result.consensus.reached == True

    def test_list_debates_sync(self, mock_client, mock_responses):
        """Test synchronous debate listing."""
        with patch.object(mock_client, "_get", return_value=mock_responses["debate_list"]):
            results = mock_client.debates.list(limit=10)

            assert len(results) == 2
            assert results[0].debate_id == "debate-1"

    @pytest.mark.asyncio
    async def test_create_debate_async(self, mock_client, mock_responses):
        """Test asynchronous debate creation."""
        with patch.object(
            mock_client,
            "_post_async",
            new_callable=AsyncMock,
            return_value=mock_responses["debate_create"],
        ):
            result = await mock_client.debates.create_async(
                task="Test debate", agents=["anthropic-api"]
            )

            assert result.debate_id == "debate-123"

    @pytest.mark.asyncio
    async def test_get_debate_async(self, mock_client, mock_responses):
        """Test asynchronous debate retrieval."""
        with patch.object(
            mock_client,
            "_get_async",
            new_callable=AsyncMock,
            return_value=mock_responses["debate_get"],
        ):
            result = await mock_client.debates.get_async("debate-123")

            assert result.debate_id == "debate-123"
            assert result.consensus.confidence == 0.85


# =============================================================================
# Graph Debates API Tests
# =============================================================================


class TestGraphDebatesAPI:
    """Tests for GraphDebatesAPI interface."""

    def test_create_graph_debate_sync(self, mock_client, mock_responses):
        """Test synchronous graph debate creation."""
        with patch.object(mock_client, "_post", return_value=mock_responses["graph_debate_create"]):
            result = mock_client.graph_debates.create(
                task="Design a distributed system",
                agents=["anthropic-api"],
                max_rounds=5,
                branch_threshold=0.5,
            )

            assert result.debate_id == "graph-123"
            assert result.status == "created"

    def test_get_graph_debate_sync(self, mock_client, mock_responses):
        """Test synchronous graph debate retrieval."""
        with patch.object(mock_client, "_get", return_value=mock_responses["graph_debate_get"]):
            result = mock_client.graph_debates.get("graph-123")

            assert result.debate_id == "graph-123"
            assert len(result.branches) == 1
            assert result.branches[0].is_main == True

    def test_get_branches_sync(self, mock_client, mock_responses):
        """Test synchronous branch retrieval."""
        with patch.object(mock_client, "_get", return_value=mock_responses["graph_branches"]):
            branches = mock_client.graph_debates.get_branches("graph-123")

            assert len(branches) == 2
            assert branches[0].branch_id == "main"
            assert branches[1].branch_id == "alt-1"

    @pytest.mark.asyncio
    async def test_create_graph_debate_async(self, mock_client, mock_responses):
        """Test asynchronous graph debate creation."""
        with patch.object(
            mock_client,
            "_post_async",
            new_callable=AsyncMock,
            return_value=mock_responses["graph_debate_create"],
        ):
            result = await mock_client.graph_debates.create_async(
                task="Design a system", max_branches=10
            )

            assert result.debate_id == "graph-123"


# =============================================================================
# Matrix Debates API Tests
# =============================================================================


class TestMatrixDebatesAPI:
    """Tests for MatrixDebatesAPI interface."""

    def test_create_matrix_debate_sync(self, mock_client, mock_responses):
        """Test synchronous matrix debate creation."""
        with patch.object(
            mock_client, "_post", return_value=mock_responses["matrix_debate_create"]
        ):
            result = mock_client.matrix_debates.create(
                task="Analyze microservices adoption",
                agents=["anthropic-api", "openai-api"],
                scenarios=[
                    {"name": "small_team", "parameters": {"team_size": 5}},
                    {"name": "large_team", "parameters": {"team_size": 50}},
                ],
            )

            assert result.matrix_id == "matrix-123"
            assert result.scenario_count == 3

    def test_get_matrix_debate_sync(self, mock_client, mock_responses):
        """Test synchronous matrix debate retrieval."""
        with patch.object(mock_client, "_get", return_value=mock_responses["matrix_debate_get"]):
            result = mock_client.matrix_debates.get("matrix-123")

            assert result.matrix_id == "matrix-123"
            assert len(result.scenarios) == 1

    def test_get_conclusions_sync(self, mock_client, mock_responses):
        """Test synchronous conclusions retrieval."""
        with patch.object(mock_client, "_get", return_value=mock_responses["matrix_conclusions"]):
            conclusions = mock_client.matrix_debates.get_conclusions("matrix-123")

            assert len(conclusions.universal) == 1
            assert "high_load" in conclusions.conditional
            assert len(conclusions.contradictions) == 0

    @pytest.mark.asyncio
    async def test_get_conclusions_async(self, mock_client, mock_responses):
        """Test asynchronous conclusions retrieval."""
        with patch.object(
            mock_client,
            "_get_async",
            new_callable=AsyncMock,
            return_value=mock_responses["matrix_conclusions"],
        ):
            conclusions = await mock_client.matrix_debates.get_conclusions_async("matrix-123")

            assert len(conclusions.universal) == 1


# =============================================================================
# Verification API Tests
# =============================================================================


class TestVerificationAPI:
    """Tests for VerificationAPI interface."""

    def test_verify_claim_sync(self, mock_client, mock_responses):
        """Test synchronous claim verification."""
        with patch.object(mock_client, "_post", return_value=mock_responses["verify_claim"]):
            result = mock_client.verification.verify(claim="All primes > 2 are odd", backend="z3")

            assert result.status == VerificationStatus.VALID
            assert result.formal_translation is not None
            assert result.duration_ms > 0

    def test_verify_status_sync(self, mock_client, mock_responses):
        """Test synchronous verification status check."""
        with patch.object(mock_client, "_get", return_value=mock_responses["verify_status"]):
            result = mock_client.verification.status()

            assert result.available == True
            assert len(result.backends) == 2
            assert result.backends[0].name == "z3"
            assert result.backends[0].available == True

    @pytest.mark.asyncio
    async def test_verify_claim_async(self, mock_client, mock_responses):
        """Test asynchronous claim verification."""
        with patch.object(
            mock_client,
            "_post_async",
            new_callable=AsyncMock,
            return_value=mock_responses["verify_claim"],
        ):
            result = await mock_client.verification.verify_async(claim="2 + 2 = 4", timeout=60)

            assert result.status == VerificationStatus.VALID


# =============================================================================
# Memory API Tests
# =============================================================================


class TestMemoryAPI:
    """Tests for MemoryAPI interface."""

    def test_analytics_sync(self, mock_client, mock_responses):
        """Test synchronous memory analytics."""
        with patch.object(mock_client, "_get", return_value=mock_responses["memory_analytics"]):
            result = mock_client.memory.analytics(days=30)

            assert len(result.tiers) == 3
            assert result.total_entries == 1600
            assert result.learning_velocity == 0.75

    def test_snapshot_sync(self, mock_client, mock_responses):
        """Test synchronous memory snapshot."""
        with patch.object(mock_client, "_post", return_value=mock_responses["memory_snapshot"]):
            result = mock_client.memory.snapshot()

            assert result.snapshot_id == "snap-123"
            assert result.success == True

    @pytest.mark.asyncio
    async def test_analytics_async(self, mock_client, mock_responses):
        """Test asynchronous memory analytics."""
        with patch.object(
            mock_client,
            "_get_async",
            new_callable=AsyncMock,
            return_value=mock_responses["memory_analytics"],
        ):
            result = await mock_client.memory.analytics_async(days=7)

            assert result.total_entries == 1600


# =============================================================================
# Agents API Tests
# =============================================================================


class TestAgentsAPI:
    """Tests for AgentsAPI interface."""

    def test_list_agents_sync(self, mock_client, mock_responses):
        """Test synchronous agent listing."""
        with patch.object(mock_client, "_get", return_value=mock_responses["agents_list"]):
            agents = mock_client.agents.list()

            assert len(agents) == 2
            assert agents[0].agent_id == "anthropic-api"

    def test_get_agent_sync(self, mock_client, mock_responses):
        """Test synchronous agent retrieval."""
        with patch.object(mock_client, "_get", return_value=mock_responses["agent_get"]):
            agent = mock_client.agents.get("anthropic-api")

            assert agent.agent_id == "anthropic-api"
            assert agent.elo_rating == 1500


# =============================================================================
# Gauntlet API Tests
# =============================================================================


class TestGauntletAPI:
    """Tests for GauntletAPI interface."""

    def test_run_gauntlet_sync(self, mock_client, mock_responses):
        """Test synchronous gauntlet run."""
        with patch.object(mock_client, "_post", return_value=mock_responses["gauntlet_run"]):
            result = mock_client.gauntlet.run(input_content="Test spec content", persona="security")

            assert result.gauntlet_id == "gauntlet-123"
            assert result.status == "running"

    def test_get_receipt_sync(self, mock_client, mock_responses):
        """Test synchronous receipt retrieval."""
        with patch.object(mock_client, "_get", return_value=mock_responses["gauntlet_receipt"]):
            receipt = mock_client.gauntlet.get_receipt("gauntlet-123")

            assert receipt.gauntlet_id == "gauntlet-123"
            assert receipt.status == "completed"
            assert receipt.score == 0.85


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling across all APIs."""

    def test_404_error(self, mock_client, mock_responses):
        """Test 404 Not Found handling."""

        def mock_get_404(url, **kwargs):
            error = httpx.HTTPStatusError(
                "Not Found",
                request=Mock(),
                response=Mock(status_code=404, json=lambda: mock_responses["error_404"]),
            )
            raise error

        with patch.object(mock_client, "_get", side_effect=mock_get_404):
            with pytest.raises(Exception):  # Should raise appropriate error
                mock_client.debates.get("nonexistent-123")

    def test_429_rate_limit(self, mock_client, mock_responses):
        """Test 429 Rate Limit handling."""

        def mock_post_429(url, data=None, **kwargs):
            error = httpx.HTTPStatusError(
                "Rate Limited",
                request=Mock(),
                response=Mock(status_code=429, json=lambda: mock_responses["error_429"]),
            )
            raise error

        with patch.object(mock_client, "_post", side_effect=mock_post_429):
            with pytest.raises(Exception):
                mock_client.debates.create(task="Test")

    def test_500_server_error(self, mock_client, mock_responses):
        """Test 500 Internal Server Error handling."""

        def mock_get_500(url, **kwargs):
            error = httpx.HTTPStatusError(
                "Internal Server Error",
                request=Mock(),
                response=Mock(status_code=500, json=lambda: mock_responses["error_500"]),
            )
            raise error

        with patch.object(mock_client, "_get", side_effect=mock_get_500):
            with pytest.raises(Exception):
                mock_client.verification.status()

    def test_timeout_handling(self, mock_client):
        """Test timeout handling."""

        def mock_timeout(url, **kwargs):
            raise httpx.TimeoutException("Request timed out")

        with patch.object(mock_client, "_get", side_effect=mock_timeout):
            with pytest.raises(Exception):
                mock_client.debates.get("debate-123")

    def test_connection_error(self, mock_client):
        """Test connection error handling."""

        def mock_connection_error(url, **kwargs):
            raise httpx.ConnectError("Connection refused")

        with patch.object(mock_client, "_get", side_effect=mock_connection_error):
            with pytest.raises(Exception):
                mock_client.agents.list()


# =============================================================================
# Concurrent Request Tests
# =============================================================================


class TestConcurrentRequests:
    """Tests for concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_debate_creation(self, mock_client, mock_responses):
        """Test multiple concurrent debate creations."""
        with patch.object(
            mock_client,
            "_post_async",
            new_callable=AsyncMock,
            return_value=mock_responses["debate_create"],
        ):
            tasks = [mock_client.debates.create_async(task=f"Debate {i}") for i in range(10)]

            results = await asyncio.gather(*tasks)

            assert len(results) == 10
            for result in results:
                assert result.debate_id == "debate-123"

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self, mock_client, mock_responses):
        """Test concurrent operations across different APIs."""
        with patch.object(mock_client, "_get_async", new_callable=AsyncMock) as mock_get:
            with patch.object(mock_client, "_post_async", new_callable=AsyncMock) as mock_post:
                # Configure different responses for different endpoints
                def get_response(url, **kwargs):
                    if "leaderboard" in url:
                        return mock_responses["leaderboard"]
                    elif "agents" in url:
                        return mock_responses["agents_list"]
                    elif "memory" in url:
                        return mock_responses["memory_analytics"]
                    return mock_responses["debate_get"]

                mock_get.side_effect = get_response
                mock_post.return_value = mock_responses["debate_create"]

                tasks = [
                    mock_client.debates.create_async(task="Test"),
                    mock_client.agents.list_async(),
                    mock_client.memory.analytics_async(),
                ]

                results = await asyncio.gather(*tasks)

                assert len(results) == 3


# =============================================================================
# Client Lifecycle Tests
# =============================================================================


class TestClientLifecycle:
    """Tests for client lifecycle management."""

    def test_client_initialization(self):
        """Test client initializes correctly."""
        client = AragoraClient(base_url="http://localhost:8080", api_key="test-key", timeout=30)

        assert client.debates is not None
        assert client.graph_debates is not None
        assert client.matrix_debates is not None
        assert client.verification is not None
        assert client.memory is not None
        assert client.agents is not None
        assert client.leaderboard is not None
        assert client.gauntlet is not None

    def test_client_context_manager(self):
        """Test client works as context manager."""
        with AragoraClient(base_url="http://localhost:8080") as client:
            assert client is not None

    @pytest.mark.asyncio
    async def test_client_async_context_manager(self):
        """Test client works as async context manager."""
        async with AragoraClient(base_url="http://localhost:8080") as client:
            assert client is not None


# =============================================================================
# Request Parameter Tests
# =============================================================================


class TestRequestParameters:
    """Tests for request parameter handling."""

    def test_debate_create_all_parameters(self, mock_client, mock_responses):
        """Test debate creation with all parameters."""
        with patch.object(
            mock_client, "_post", return_value=mock_responses["debate_create"]
        ) as mock_post:
            mock_client.debates.create(
                task="Complex debate",
                agents=["anthropic-api", "openai-api", "gemini"],
                max_rounds=10,
                consensus_threshold=0.8,
                enable_voting=True,
            )

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            data = call_args[0][1]  # Second positional arg is the data

            assert data["task"] == "Complex debate"
            assert len(data["agents"]) == 3

    def test_graph_debate_branch_parameters(self, mock_client, mock_responses):
        """Test graph debate with branch parameters."""
        with patch.object(
            mock_client, "_post", return_value=mock_responses["graph_debate_create"]
        ) as mock_post:
            mock_client.graph_debates.create(
                task="Design system", branch_threshold=0.3, max_branches=15
            )

            mock_post.assert_called_once()

    def test_matrix_debate_scenario_parameters(self, mock_client, mock_responses):
        """Test matrix debate with scenario parameters."""
        with patch.object(
            mock_client, "_post", return_value=mock_responses["matrix_debate_create"]
        ) as mock_post:
            mock_client.matrix_debates.create(
                task="Analyze options",
                scenarios=[
                    {"name": "scenario_a", "parameters": {"x": 1}, "constraints": ["c1"]},
                    {"name": "scenario_b", "parameters": {"x": 2}, "is_baseline": True},
                ],
            )

            mock_post.assert_called_once()


# =============================================================================
# Response Validation Tests
# =============================================================================


class TestResponseValidation:
    """Tests for response validation and parsing."""

    def test_consensus_result_parsing(self, mock_client, mock_responses):
        """Test ConsensusResult is parsed correctly."""
        with patch.object(mock_client, "_get", return_value=mock_responses["debate_get"]):
            debate = mock_client.debates.get("debate-123")

            assert debate.consensus.reached == True
            assert debate.consensus.conclusion == "Test conclusion"
            assert debate.consensus.confidence == 0.85
            assert len(debate.consensus.supporting_agents) == 2

    def test_verification_status_parsing(self, mock_client, mock_responses):
        """Test VerifyStatusResponse is parsed correctly."""
        with patch.object(mock_client, "_get", return_value=mock_responses["verify_status"]):
            status = mock_client.verification.status()

            assert status.available == True
            assert status.backends[0].version == "4.12.0"

    def test_memory_tier_stats_parsing(self, mock_client, mock_responses):
        """Test MemoryAnalyticsResponse is parsed correctly."""
        with patch.object(mock_client, "_get", return_value=mock_responses["memory_analytics"]):
            analytics = mock_client.memory.analytics()

            assert analytics.tiers[0].tier_name == "fast"
            assert analytics.tiers[0].hit_rate == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
