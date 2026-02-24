"""Tests for the Python SDK consensus detection methods.

Validates:
- ConsensusAPI.detect()
- ConsensusAPI.get_detection_status()
- AsyncConsensusAPI.detect()
- AsyncConsensusAPI.get_detection_status()
"""

from unittest.mock import MagicMock, AsyncMock

import pytest

from aragora_sdk.namespaces.consensus import AsyncConsensusAPI, ConsensusAPI


class TestConsensusAPIDetect:
    """Test ConsensusAPI.detect() method."""

    def test_detect_calls_post(self):
        """Test that detect makes a POST request with correct payload."""
        mock_client = MagicMock()
        mock_client.request.return_value = {
            "data": {
                "debate_id": "detect-abc",
                "consensus_reached": True,
                "confidence": 0.85,
            }
        }

        api = ConsensusAPI(mock_client)
        result = api.detect(
            task="Choose a database",
            proposals=[
                {"agent": "claude", "content": "Use PostgreSQL"},
                {"agent": "gpt-4", "content": "Use PostgreSQL with Redis"},
            ],
            threshold=0.7,
        )

        mock_client.request.assert_called_once_with(
            "POST",
            "/api/v1/consensus/detect",
            json={
                "task": "Choose a database",
                "proposals": [
                    {"agent": "claude", "content": "Use PostgreSQL"},
                    {"agent": "gpt-4", "content": "Use PostgreSQL with Redis"},
                ],
                "threshold": 0.7,
            },
        )
        assert result["data"]["consensus_reached"] is True

    def test_detect_default_threshold(self):
        """Test that detect uses default threshold of 0.7."""
        mock_client = MagicMock()
        mock_client.request.return_value = {"data": {}}

        api = ConsensusAPI(mock_client)
        api.detect(task="Test", proposals=[{"agent": "a", "content": "x"}])

        _, kwargs = mock_client.request.call_args
        assert kwargs["json"]["threshold"] == 0.7

    def test_detect_custom_threshold(self):
        """Test that detect passes custom threshold."""
        mock_client = MagicMock()
        mock_client.request.return_value = {"data": {}}

        api = ConsensusAPI(mock_client)
        api.detect(task="Test", proposals=[{"agent": "a", "content": "x"}], threshold=0.9)

        _, kwargs = mock_client.request.call_args
        assert kwargs["json"]["threshold"] == 0.9


class TestConsensusAPIGetDetectionStatus:
    """Test ConsensusAPI.get_detection_status() method."""

    def test_get_detection_status_calls_get(self):
        """Test that get_detection_status makes a GET request."""
        mock_client = MagicMock()
        mock_client.request.return_value = {
            "data": {
                "debate_id": "test-123",
                "consensus_reached": True,
            }
        }

        api = ConsensusAPI(mock_client)
        result = api.get_detection_status("test-123")

        mock_client.request.assert_called_once_with("GET", "/api/v1/consensus/status/test-123")
        assert result["data"]["debate_id"] == "test-123"

    def test_get_detection_status_encodes_id(self):
        """Test that debate_id is passed in the URL path."""
        mock_client = MagicMock()
        mock_client.request.return_value = {"data": {}}

        api = ConsensusAPI(mock_client)
        api.get_detection_status("my-debate-456")

        mock_client.request.assert_called_once_with("GET", "/api/v1/consensus/status/my-debate-456")


class TestAsyncConsensusAPIDetect:
    """Test AsyncConsensusAPI.detect() method."""

    @pytest.mark.asyncio
    async def test_async_detect(self):
        """Test async detect makes correct request."""
        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value={"data": {"consensus_reached": True}})

        api = AsyncConsensusAPI(mock_client)
        result = await api.detect(
            task="Test question",
            proposals=[{"agent": "a", "content": "Answer"}],
            threshold=0.5,
        )

        mock_client.request.assert_awaited_once_with(
            "POST",
            "/api/v1/consensus/detect",
            json={
                "task": "Test question",
                "proposals": [{"agent": "a", "content": "Answer"}],
                "threshold": 0.5,
            },
        )
        assert result["data"]["consensus_reached"] is True


class TestAsyncConsensusAPIGetDetectionStatus:
    """Test AsyncConsensusAPI.get_detection_status() method."""

    @pytest.mark.asyncio
    async def test_async_get_detection_status(self):
        """Test async get_detection_status makes correct request."""
        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value={"data": {"debate_id": "test-789"}})

        api = AsyncConsensusAPI(mock_client)
        result = await api.get_detection_status("test-789")

        mock_client.request.assert_awaited_once_with("GET", "/api/v1/consensus/status/test-789")
        assert result["data"]["debate_id"] == "test-789"
