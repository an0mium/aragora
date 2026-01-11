"""Tests for the Aragora Python SDK client."""

import pytest
from unittest.mock import patch, MagicMock
import json


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
