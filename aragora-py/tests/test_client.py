"""Tests for the Aragora client."""

from __future__ import annotations

from aragora_client import (
    AragoraClient,
    AragoraError,
    AragoraNotFoundError,
    DebateStatus,
)


class TestAragoraClient:
    """Tests for AragoraClient."""

    def test_client_initialization(self) -> None:
        """Test client can be initialized."""
        client = AragoraClient("http://localhost:8080")
        assert client.base_url == "http://localhost:8080"

    def test_client_with_api_key(self) -> None:
        """Test client with API key."""
        client = AragoraClient(
            "http://localhost:8080",
            api_key="test-key",
        )
        assert client._api_key == "test-key"

    def test_client_strips_trailing_slash(self) -> None:
        """Test client strips trailing slash from base URL."""
        client = AragoraClient("http://localhost:8080/")
        assert client.base_url == "http://localhost:8080"


class TestDebateTypes:
    """Tests for debate types."""

    def test_debate_status_enum(self) -> None:
        """Test DebateStatus enum values."""
        assert DebateStatus.PENDING == "pending"
        assert DebateStatus.RUNNING == "running"
        assert DebateStatus.COMPLETED == "completed"


class TestExceptions:
    """Tests for custom exceptions."""

    def test_aragora_error(self) -> None:
        """Test AragoraError."""
        error = AragoraError("test error", code="TEST", status=500)
        assert error.message == "test error"
        assert error.code == "TEST"
        assert error.status == 500

    def test_aragora_not_found_error(self) -> None:
        """Test AragoraNotFoundError."""
        error = AragoraNotFoundError("Debate", "debate-123")
        assert error.resource == "Debate"
        assert error.resource_id == "debate-123"
        assert error.status == 404
