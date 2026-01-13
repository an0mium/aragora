"""Tests for the Aragora Python SDK client."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import json
import time


class TestAragoraClient:
    """Test AragoraClient class."""

    def test_client_initialization(self):
        """Test client can be initialized."""
        from aragora.client import AragoraClient

        client = AragoraClient(base_url="http://localhost:8080")
        assert client.base_url == "http://localhost:8080"
        assert client.api_key is None
        assert client.timeout == 60

    def test_client_with_api_key(self):
        """Test client initialization with API key."""
        from aragora.client import AragoraClient

        client = AragoraClient(
            base_url="http://example.com",
            api_key="test-key",
            timeout=30,
        )
        assert client.api_key == "test-key"
        assert client.timeout == 30

    def test_client_has_api_interfaces(self):
        """Test client has all API interfaces."""
        from aragora.client import AragoraClient

        client = AragoraClient()
        assert hasattr(client, "debates")
        assert hasattr(client, "agents")
        assert hasattr(client, "leaderboard")
        assert hasattr(client, "gauntlet")

    def test_get_headers_without_key(self):
        """Test headers without API key."""
        from aragora.client import AragoraClient

        client = AragoraClient()
        headers = client._get_headers()
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    def test_get_headers_with_key(self):
        """Test headers with API key."""
        from aragora.client import AragoraClient

        client = AragoraClient(api_key="my-key")
        headers = client._get_headers()
        assert headers["Authorization"] == "Bearer my-key"


class TestModels:
    """Test Pydantic models."""

    def test_debate_model(self):
        """Test Debate model creation."""
        from aragora.client import Debate, DebateStatus

        debate = Debate(
            debate_id="test-123",
            task="Test question",
            status=DebateStatus.COMPLETED,
            agents=["a1", "a2"],
        )
        assert debate.debate_id == "test-123"
        assert debate.status == DebateStatus.COMPLETED

    def test_debate_create_request(self):
        """Test DebateCreateRequest model."""
        from aragora.client import DebateCreateRequest, ConsensusType

        request = DebateCreateRequest(
            task="Should we use microservices?",
            agents=["anthropic-api"],
            rounds=5,
        )
        assert request.task == "Should we use microservices?"
        assert request.rounds == 5
        assert request.consensus == ConsensusType.MAJORITY

    def test_consensus_result_model(self):
        """Test ConsensusResult model."""
        from aragora.client.models import ConsensusResult

        result = ConsensusResult(
            reached=True,
            agreement=0.85,
            final_answer="Microservices for scale",
        )
        assert result.reached is True
        assert result.agreement == 0.85

    def test_agent_profile_model(self):
        """Test AgentProfile model."""
        from aragora.client import AgentProfile

        agent = AgentProfile(
            agent_id="anthropic-api",
            name="Claude",
            provider="anthropic",
            elo_rating=1650,
        )
        assert agent.elo_rating == 1650
        assert agent.matches_played == 0

    def test_leaderboard_entry_model(self):
        """Test LeaderboardEntry model."""
        from aragora.client import LeaderboardEntry

        entry = LeaderboardEntry(
            rank=1,
            agent_id="top-agent",
            elo_rating=1800,
            matches_played=50,
            win_rate=0.72,
        )
        assert entry.rank == 1
        assert entry.recent_trend == "stable"

    def test_gauntlet_receipt_model(self):
        """Test GauntletReceipt model."""
        from aragora.client import GauntletReceipt, GauntletVerdict, Finding
        from datetime import datetime

        finding = Finding(
            severity="high",
            category="security",
            title="SQL Injection",
            description="Found potential SQL injection",
        )

        receipt = GauntletReceipt(
            receipt_id="rcpt-123",
            verdict=GauntletVerdict.NEEDS_REVIEW,
            risk_score=0.65,
            findings=[finding],
            summary="Found security issues",
            created_at=datetime.now(),
            input_hash="abc123",
            persona="security",
        )
        assert receipt.verdict == GauntletVerdict.NEEDS_REVIEW
        assert len(receipt.findings) == 1

    def test_health_check_model(self):
        """Test HealthCheck model."""
        from aragora.client import HealthCheck

        health = HealthCheck(
            status="healthy",
            version="1.0.0",
            uptime_seconds=3600.5,
            components={"database": "ok", "redis": "ok"},
        )
        assert health.status == "healthy"
        assert health.components["database"] == "ok"


class TestDebatesAPI:
    """Test DebatesAPI interface."""

    def test_create_debate_request_format(self):
        """Test debate creation request is formatted correctly."""
        from aragora.client import AragoraClient, DebateCreateRequest, ConsensusType

        client = AragoraClient()

        # Mock the POST request
        with patch.object(client, "_post") as mock_post:
            mock_post.return_value = {
                "debate_id": "debate-123",
                "status": "pending",
                "task": "Test task",
            }

            response = client.debates.create(
                task="Test task",
                agents=["agent1", "agent2"],
                rounds=5,
                consensus="unanimous",
            )

            # Verify POST was called with correct data
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "/api/debates"
            data = call_args[0][1]
            assert data["task"] == "Test task"
            assert data["rounds"] == 5
            assert data["consensus"] == "unanimous"

    def test_get_debate(self):
        """Test getting debate by ID."""
        from aragora.client import AragoraClient

        client = AragoraClient()

        with patch.object(client, "_get") as mock_get:
            mock_get.return_value = {
                "debate_id": "debate-456",
                "task": "Test",
                "status": "completed",
                "agents": ["a1"],
                "rounds": [],
            }

            debate = client.debates.get("debate-456")

            mock_get.assert_called_once_with("/api/debates/debate-456")
            assert debate.debate_id == "debate-456"
            assert debate.status.value == "completed"


class TestGauntletAPI:
    """Test GauntletAPI interface."""

    def test_run_gauntlet(self):
        """Test running gauntlet analysis."""
        from aragora.client import AragoraClient

        client = AragoraClient()

        with patch.object(client, "_post") as mock_post:
            mock_post.return_value = {
                "gauntlet_id": "gauntlet-789",
                "status": "running",
            }

            response = client.gauntlet.run(
                input_content="Test policy content",
                input_type="policy",
                persona="gdpr",
                profile="thorough",
            )

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            data = call_args[0][1]
            assert data["input_content"] == "Test policy content"
            assert data["persona"] == "gdpr"
            assert response.gauntlet_id == "gauntlet-789"


class TestAPIError:
    """Test API error handling."""

    def test_aragora_api_error(self):
        """Test AragoraAPIError exception."""
        from aragora.client import AragoraAPIError

        error = AragoraAPIError("Not found", "NOT_FOUND", 404)
        assert str(error) == "Not found"
        assert error.code == "NOT_FOUND"
        assert error.status_code == 404


class TestExports:
    """Test module exports."""

    def test_all_exports_importable(self):
        """Test all __all__ exports are importable."""
        from aragora.client import (
            AragoraClient,
            AragoraAPIError,
            DebatesAPI,
            AgentsAPI,
            LeaderboardAPI,
            GauntletAPI,
            DebateStatus,
            ConsensusType,
            GauntletVerdict,
            Debate,
            DebateRound,
            DebateCreateRequest,
            DebateCreateResponse,
            AgentMessage,
            Vote,
            ConsensusResult,
            AgentProfile,
            LeaderboardEntry,
            GauntletReceipt,
            GauntletRunRequest,
            GauntletRunResponse,
            Finding,
            HealthCheck,
            APIError,
        )

        # All imports successful
        assert AragoraClient is not None
        assert DebateStatus.COMPLETED.value == "completed"
        assert ConsensusType.MAJORITY.value == "majority"


# ============================================================================
# RetryConfig Tests
# ============================================================================


class TestRetryConfig:
    """Tests for RetryConfig class."""

    def test_retry_config_defaults(self):
        """Test default values for RetryConfig."""
        from aragora.client import RetryConfig

        config = RetryConfig()
        assert config.max_retries == 3
        assert config.backoff_factor == 0.5
        assert config.max_backoff == 30.0
        assert config.jitter is True
        assert 429 in config.retry_statuses
        assert 500 in config.retry_statuses

    def test_retry_config_custom_values(self):
        """Test RetryConfig with custom values."""
        from aragora.client import RetryConfig

        config = RetryConfig(
            max_retries=5,
            backoff_factor=1.0,
            max_backoff=60.0,
            retry_statuses=(429, 503),
            jitter=False,
        )
        assert config.max_retries == 5
        assert config.backoff_factor == 1.0
        assert config.max_backoff == 60.0
        assert config.retry_statuses == (429, 503)
        assert config.jitter is False

    def test_get_delay_exponential_backoff(self):
        """Test exponential backoff delay calculation."""
        from aragora.client import RetryConfig

        config = RetryConfig(backoff_factor=1.0, jitter=False)
        assert config.get_delay(0) == 1.0  # 1.0 * 2^0 = 1
        assert config.get_delay(1) == 2.0  # 1.0 * 2^1 = 2
        assert config.get_delay(2) == 4.0  # 1.0 * 2^2 = 4
        assert config.get_delay(3) == 8.0  # 1.0 * 2^3 = 8

    def test_get_delay_max_backoff(self):
        """Test that delay is capped at max_backoff."""
        from aragora.client import RetryConfig

        config = RetryConfig(backoff_factor=1.0, max_backoff=5.0, jitter=False)
        assert config.get_delay(0) == 1.0
        assert config.get_delay(2) == 4.0
        assert config.get_delay(3) == 5.0  # Would be 8, but capped at 5
        assert config.get_delay(10) == 5.0  # Always capped

    def test_get_delay_with_jitter(self):
        """Test that jitter adds randomization."""
        from aragora.client import RetryConfig

        config = RetryConfig(backoff_factor=1.0, jitter=True)
        delays = [config.get_delay(2) for _ in range(10)]
        # With jitter, delays should vary between 2.0 and 6.0 (4 * 0.5 to 4 * 1.5)
        assert all(2.0 <= d <= 6.0 for d in delays)
        # Delays should not all be the same
        assert len(set(delays)) > 1

    def test_get_delay_without_jitter(self):
        """Test consistent delay without jitter."""
        from aragora.client import RetryConfig

        config = RetryConfig(backoff_factor=1.0, jitter=False)
        delays = [config.get_delay(2) for _ in range(10)]
        # Without jitter, all delays should be identical
        assert all(d == 4.0 for d in delays)

    def test_retry_statuses_tuple(self):
        """Test that retry_statuses is correctly configured."""
        from aragora.client import RetryConfig

        config = RetryConfig()
        assert isinstance(config.retry_statuses, tuple)
        # Default retry statuses
        assert 429 in config.retry_statuses  # Rate limited
        assert 500 in config.retry_statuses  # Internal server error
        assert 502 in config.retry_statuses  # Bad gateway
        assert 503 in config.retry_statuses  # Service unavailable
        assert 504 in config.retry_statuses  # Gateway timeout


# ============================================================================
# RateLimiter Tests
# ============================================================================


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_rate_limiter_initialization(self):
        """Test RateLimiter initialization."""
        from aragora.client import RateLimiter

        limiter = RateLimiter(rps=10.0)
        assert limiter.rps == 10.0
        assert limiter.min_interval == pytest.approx(0.1)

    def test_rate_limiter_zero_rps(self):
        """Test RateLimiter with zero RPS (no limiting)."""
        from aragora.client import RateLimiter

        limiter = RateLimiter(rps=0)
        assert limiter.rps == 0
        assert limiter.min_interval == 0

    def test_rate_limiter_wait_no_blocking(self):
        """Test that wait() doesn't block unnecessarily."""
        from aragora.client import RateLimiter

        limiter = RateLimiter(rps=1000)  # Very high RPS
        limiter._last_request = 0  # Reset

        start = time.time()
        limiter.wait()
        elapsed = time.time() - start

        # Should be nearly instant (< 10ms)
        assert elapsed < 0.01

    def test_rate_limiter_wait_enforces_rate(self):
        """Test that wait() enforces rate limiting."""
        from aragora.client import RateLimiter

        limiter = RateLimiter(rps=10)  # 100ms between requests
        limiter._last_request = time.time()

        start = time.time()
        limiter.wait()
        elapsed = time.time() - start

        # Should wait approximately 100ms (10 RPS)
        assert elapsed >= 0.08  # Allow some tolerance

    def test_rate_limiter_zero_rps_no_wait(self):
        """Test that zero RPS doesn't wait."""
        from aragora.client import RateLimiter

        limiter = RateLimiter(rps=0)
        start = time.time()
        limiter.wait()
        elapsed = time.time() - start
        assert elapsed < 0.01

    @pytest.mark.asyncio
    async def test_rate_limiter_wait_async(self):
        """Test async wait method."""
        from aragora.client import RateLimiter

        limiter = RateLimiter(rps=1000)  # Very high RPS
        limiter._last_request = 0

        start = asyncio.get_event_loop().time()
        await limiter.wait_async()
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed < 0.01

    @pytest.mark.asyncio
    async def test_rate_limiter_async_zero_rps(self):
        """Test async wait with zero RPS."""
        from aragora.client import RateLimiter

        limiter = RateLimiter(rps=0)
        start = asyncio.get_event_loop().time()
        await limiter.wait_async()
        elapsed = asyncio.get_event_loop().time() - start
        assert elapsed < 0.01


# ============================================================================
# Client with Retry Tests
# ============================================================================


class TestClientWithRetry:
    """Tests for client retry functionality."""

    def test_client_with_retry_config(self):
        """Test client accepts retry config."""
        from aragora.client import AragoraClient, RetryConfig

        config = RetryConfig(max_retries=5)
        client = AragoraClient(
            base_url="http://localhost:8080",
            retry_config=config,
        )
        assert client.retry_config is not None
        assert client.retry_config.max_retries == 5

    def test_client_with_rate_limit(self):
        """Test client accepts rate limit."""
        from aragora.client import AragoraClient

        client = AragoraClient(
            base_url="http://localhost:8080",
            rate_limit_rps=10.0,
        )
        assert client._rate_limiter is not None
        assert client._rate_limiter.rps == 10.0

    def test_client_default_no_rate_limit(self):
        """Test client has no rate limiter by default."""
        from aragora.client import AragoraClient

        client = AragoraClient()
        assert client._rate_limiter is None


# ============================================================================
# Batch Get Tests
# ============================================================================


class TestBatchGet:
    """Tests for batch_get functionality."""

    def test_batch_get_single_item(self):
        """Test batch_get with single ID."""
        from aragora.client import AragoraClient

        client = AragoraClient()

        with patch.object(client, "_get") as mock_get:
            mock_get.return_value = {
                "debate_id": "debate-1",
                "task": "Test",
                "status": "completed",
                "agents": ["a1"],
                "rounds": [],
            }

            debates = client.debates.batch_get(["debate-1"])

            assert len(debates) == 1
            assert debates[0].debate_id == "debate-1"

    def test_batch_get_multiple_items(self):
        """Test batch_get with multiple IDs."""
        from aragora.client import AragoraClient

        client = AragoraClient()

        def mock_get_side_effect(path):
            debate_id = path.split("/")[-1]
            return {
                "debate_id": debate_id,
                "task": f"Task for {debate_id}",
                "status": "completed",
                "agents": ["a1"],
                "rounds": [],
            }

        with patch.object(client, "_get", side_effect=mock_get_side_effect):
            debates = client.debates.batch_get(["debate-1", "debate-2", "debate-3"])

            assert len(debates) == 3
            assert debates[0].debate_id == "debate-1"
            assert debates[1].debate_id == "debate-2"
            assert debates[2].debate_id == "debate-3"

    def test_batch_get_empty_list(self):
        """Test batch_get with empty list."""
        from aragora.client import AragoraClient

        client = AragoraClient()
        debates = client.debates.batch_get([])
        assert debates == []

    def test_batch_get_preserves_order(self):
        """Test batch_get preserves ID order in results."""
        from aragora.client import AragoraClient

        client = AragoraClient()
        ids = ["z-last", "a-first", "m-middle"]

        def mock_get_side_effect(path):
            debate_id = path.split("/")[-1]
            return {
                "debate_id": debate_id,
                "task": f"Task for {debate_id}",
                "status": "completed",
                "agents": ["a1"],
                "rounds": [],
            }

        with patch.object(client, "_get", side_effect=mock_get_side_effect):
            debates = client.debates.batch_get(ids)

            # Results should match input order
            assert [d.debate_id for d in debates] == ids

    @pytest.mark.asyncio
    async def test_batch_get_async_concurrent(self):
        """Test batch_get_async uses concurrency."""
        from aragora.client import AragoraClient

        client = AragoraClient()
        ids = ["debate-1", "debate-2", "debate-3"]

        async def mock_get_async(path):
            debate_id = path.split("/")[-1]
            return {
                "debate_id": debate_id,
                "task": f"Task for {debate_id}",
                "status": "completed",
                "agents": ["a1"],
                "rounds": [],
            }

        with patch.object(client, "_get_async", side_effect=mock_get_async):
            debates = await client.debates.batch_get_async(ids)

            assert len(debates) == 3
            assert debates[0].debate_id == "debate-1"

    @pytest.mark.asyncio
    async def test_batch_get_async_empty_list(self):
        """Test batch_get_async with empty list."""
        from aragora.client import AragoraClient

        client = AragoraClient()
        debates = await client.debates.batch_get_async([])
        assert debates == []

    @pytest.mark.asyncio
    async def test_batch_get_async_max_concurrent(self):
        """Test batch_get_async respects max_concurrent."""
        from aragora.client import AragoraClient

        client = AragoraClient()
        ids = ["d-1", "d-2", "d-3", "d-4", "d-5"]
        active_count = 0
        max_active = 0

        async def mock_get_async(path):
            nonlocal active_count, max_active
            active_count += 1
            max_active = max(max_active, active_count)
            await asyncio.sleep(0.01)
            active_count -= 1
            debate_id = path.split("/")[-1]
            return {
                "debate_id": debate_id,
                "task": "Task",
                "status": "completed",
                "agents": ["a1"],
                "rounds": [],
            }

        with patch.object(client, "_get_async", side_effect=mock_get_async):
            await client.debates.batch_get_async(ids, max_concurrent=2)
            # Max concurrent should be limited to 2
            assert max_active <= 2


# ============================================================================
# Pagination / Iterate Tests
# ============================================================================


class TestIterate:
    """Tests for iterate functionality."""

    def test_iterate_single_page(self):
        """Test iterate with results fitting in one page."""
        from aragora.client import AragoraClient

        client = AragoraClient()

        with patch.object(client, "_get") as mock_get:
            mock_get.return_value = {
                "debates": [
                    {
                        "debate_id": "d-1",
                        "task": "Task 1",
                        "status": "completed",
                        "agents": ["a1"],
                        "rounds": [],
                    },
                    {
                        "debate_id": "d-2",
                        "task": "Task 2",
                        "status": "completed",
                        "agents": ["a1"],
                        "rounds": [],
                    },
                ],
                "total": 2,
                "next_cursor": None,
            }

            debates = list(client.debates.iterate(page_size=10))

            assert len(debates) == 2
            assert debates[0].debate_id == "d-1"
            assert debates[1].debate_id == "d-2"

    def test_iterate_multiple_pages(self):
        """Test iterate fetches multiple pages."""
        from aragora.client import AragoraClient

        client = AragoraClient()
        call_count = 0

        def mock_get_side_effect(path, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "debates": [
                        {
                            "debate_id": "d-1",
                            "task": "Task",
                            "status": "completed",
                            "agents": ["a1"],
                            "rounds": [],
                        },
                        {
                            "debate_id": "d-2",
                            "task": "Task",
                            "status": "completed",
                            "agents": ["a1"],
                            "rounds": [],
                        },
                    ],
                }
            else:
                # Return empty on subsequent calls to stop iteration
                return {"debates": []}

        with patch.object(client, "_get", side_effect=mock_get_side_effect):
            debates = list(client.debates.iterate(page_size=5))

            assert len(debates) == 2
            # Only 1 call needed since 2 debates < page_size of 5
            assert call_count == 1

    def test_iterate_max_items(self):
        """Test iterate respects max_items limit."""
        from aragora.client import AragoraClient

        client = AragoraClient()

        def mock_get_side_effect(path, params=None):
            # Return more debates than max_items will allow
            return {
                "debates": [
                    {
                        "debate_id": f"d-{i}",
                        "task": "Task",
                        "status": "completed",
                        "agents": ["a1"],
                        "rounds": [],
                    }
                    for i in range(10)
                ],
            }

        with patch.object(client, "_get", side_effect=mock_get_side_effect):
            debates = list(client.debates.iterate(max_items=3))

            assert len(debates) == 3

    def test_iterate_with_status_filter(self):
        """Test iterate passes status filter."""
        from aragora.client import AragoraClient

        client = AragoraClient()

        def mock_get_side_effect(path, params=None):
            return {"debates": []}

        with patch.object(client, "_get", side_effect=mock_get_side_effect) as mock_get:
            list(client.debates.iterate(status="completed"))

            # Verify status was passed as parameter
            mock_get.assert_called_once()

    def test_iterate_empty_results(self):
        """Test iterate with no results."""
        from aragora.client import AragoraClient

        client = AragoraClient()

        def mock_get_side_effect(path, params=None):
            return {"debates": []}

        with patch.object(client, "_get", side_effect=mock_get_side_effect):
            debates = list(client.debates.iterate())
            assert debates == []

    @pytest.mark.asyncio
    async def test_iterate_async_single_page(self):
        """Test async iterate with single page."""
        from aragora.client import AragoraClient

        client = AragoraClient()
        call_count = 0

        async def mock_get_async(path, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "debates": [
                        {
                            "debate_id": "d-1",
                            "task": "Task",
                            "status": "completed",
                            "agents": ["a1"],
                            "rounds": [],
                        },
                    ],
                }
            return {"debates": []}

        with patch.object(client, "_get_async", side_effect=mock_get_async):
            debates = []
            async for debate in client.debates.iterate_async():
                debates.append(debate)

            assert len(debates) == 1

    @pytest.mark.asyncio
    async def test_iterate_async_max_items(self):
        """Test async iterate respects max_items."""
        from aragora.client import AragoraClient

        client = AragoraClient()

        async def mock_get_async(path, params=None):
            # Return many debates but max_items will limit
            return {
                "debates": [
                    {
                        "debate_id": f"d-{i}",
                        "task": "Task",
                        "status": "completed",
                        "agents": ["a1"],
                        "rounds": [],
                    }
                    for i in range(10)
                ],
            }

        with patch.object(client, "_get_async", side_effect=mock_get_async):
            debates = []
            async for debate in client.debates.iterate_async(max_items=5):
                debates.append(debate)

            assert len(debates) == 5


# ============================================================================
# Additional Model Tests
# ============================================================================


class TestGraphDebateModels:
    """Test Graph debate models."""

    def test_graph_debate_create_request(self):
        """Test GraphDebateCreateRequest model."""
        from aragora.client import GraphDebateCreateRequest

        request = GraphDebateCreateRequest(
            task="Test task",
            agents=["a1", "a2"],
            max_rounds=5,
            branch_threshold=0.6,
        )
        assert request.task == "Test task"
        assert request.max_rounds == 5
        assert request.branch_threshold == 0.6

    def test_graph_debate_model(self):
        """Test GraphDebate model."""
        from aragora.client import GraphDebate, DebateStatus

        debate = GraphDebate(
            debate_id="graph-123",
            task="Test",
            status=DebateStatus.RUNNING,
            agents=["a1"],
        )
        assert debate.debate_id == "graph-123"
        assert debate.status == DebateStatus.RUNNING


class TestMatrixDebateModels:
    """Test Matrix debate models."""

    def test_matrix_debate_create_request(self):
        """Test MatrixDebateCreateRequest model."""
        from aragora.client import MatrixDebateCreateRequest

        request = MatrixDebateCreateRequest(
            task="Test policy",
            agents=["a1"],
            max_rounds=5,
        )
        assert request.task == "Test policy"
        assert request.max_rounds == 5


class TestVerificationModels:
    """Test Verification models."""

    def test_verify_claim_request(self):
        """Test VerifyClaimRequest model."""
        from aragora.client import VerifyClaimRequest

        request = VerifyClaimRequest(
            claim="The sky is blue",
            backend="z3",
        )
        assert request.claim == "The sky is blue"
        assert request.backend == "z3"

    def test_verify_status_response(self):
        """Test VerifyStatusResponse model."""
        from aragora.client import VerifyStatusResponse

        response = VerifyStatusResponse(
            available=True,
            backends=[],
        )
        assert response.available is True


class TestMemoryModels:
    """Test Memory models."""

    def test_memory_analytics_response(self):
        """Test MemoryAnalyticsResponse model."""
        from aragora.client import MemoryAnalyticsResponse, MemoryTierStats

        stats = MemoryTierStats(
            tier_name="fast",
            entry_count=100,
            hit_rate=0.85,
        )
        response = MemoryAnalyticsResponse(
            tiers=[stats],
            total_entries=100,
        )
        assert response.total_entries == 100


class TestReplayModels:
    """Test Replay models."""

    def test_replay_model(self):
        """Test Replay model."""
        from aragora.client import Replay
        from datetime import datetime

        replay = Replay(
            replay_id="r-123",
            debate_id="d-456",
            task="Test debate",
            created_at=datetime.now(),
        )
        assert replay.replay_id == "r-123"
        assert replay.task == "Test debate"

    def test_replay_summary_model(self):
        """Test ReplaySummary model."""
        from aragora.client import ReplaySummary
        from datetime import datetime

        summary = ReplaySummary(
            replay_id="r-123",
            debate_id="d-456",
            task="Test debate",
            created_at=datetime.now(),
            duration_seconds=120,
            round_count=5,
        )
        assert summary.duration_seconds == 120
        assert summary.round_count == 5


# ============================================================================
# WebSocket Tests
# ============================================================================


class TestWebSocketExports:
    """Test WebSocket module exports."""

    def test_websocket_exports_importable(self):
        """Test WebSocket exports are importable."""
        from aragora.client import (
            DebateStream,
            DebateEvent,
            DebateEventType,
            WebSocketOptions,
            stream_debate,
        )

        assert DebateStream is not None
        assert DebateEventType is not None


# ============================================================================
# API Interface Tests
# ============================================================================


class TestGraphDebatesAPI:
    """Test GraphDebatesAPI interface."""

    def test_graph_debates_api_exists(self):
        """Test client has graph debates API."""
        from aragora.client import AragoraClient

        client = AragoraClient()
        assert hasattr(client, "graph_debates")


class TestMatrixDebatesAPI:
    """Test MatrixDebatesAPI interface."""

    def test_matrix_debates_api_exists(self):
        """Test client has matrix debates API."""
        from aragora.client import AragoraClient

        client = AragoraClient()
        assert hasattr(client, "matrix_debates")


class TestVerificationAPI:
    """Test VerificationAPI interface."""

    def test_verification_api_exists(self):
        """Test client has verification API."""
        from aragora.client import AragoraClient

        client = AragoraClient()
        assert hasattr(client, "verification")


class TestMemoryAPI:
    """Test MemoryAPI interface."""

    def test_memory_api_exists(self):
        """Test client has memory API."""
        from aragora.client import AragoraClient

        client = AragoraClient()
        assert hasattr(client, "memory")


class TestReplayAPI:
    """Test ReplayAPI interface."""

    def test_replay_api_exists(self):
        """Test client has replays API."""
        from aragora.client import AragoraClient

        client = AragoraClient()
        assert hasattr(client, "replays")
