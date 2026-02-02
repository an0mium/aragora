"""Tests for Deliberations namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestDeliberationsListActive:
    """Tests for listing active deliberations."""

    def test_list_active_deliberations(self) -> None:
        """List all active deliberation sessions."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "deliberations": [
                    {
                        "id": "delib_123",
                        "status": "active",
                        "current_round": 2,
                    }
                ],
                "count": 1,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.deliberations.list_active()

            mock_request.assert_called_once()
            assert result["count"] == 1
            client.close()


class TestDeliberationsStats:
    """Tests for getting deliberation statistics."""

    def test_get_stats(self) -> None:
        """Get deliberation statistics."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "active_count": 5,
                "completed_today": 15,
                "average_duration_seconds": 120,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.deliberations.get_stats()

            mock_request.assert_called_once()
            assert result["active_count"] == 5
            client.close()


class TestDeliberationsGet:
    """Tests for getting specific deliberations."""

    def test_get_deliberation(self) -> None:
        """Get a specific deliberation by ID."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "id": "delib_123",
                "status": "active",
                "current_round": 2,
                "total_rounds": 5,
                "agents": ["claude", "gpt-4", "gemini"],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.deliberations.get("delib_123")

            mock_request.assert_called_once()
            assert result["id"] == "delib_123"
            assert result["current_round"] == 2
            client.close()


class TestDeliberationsStreamConfig:
    """Tests for getting stream configuration."""

    def test_get_stream_config(self) -> None:
        """Get WebSocket stream configuration."""
        client = AragoraClient(base_url="https://api.aragora.ai")
        result = client.deliberations.get_stream_config()

        # Should return a static config dict, not raise NotImplementedError
        assert result["type"] == "websocket"
        assert result["path"] == "/api/v1/deliberations/stream"
        assert "events" in result
        assert "deliberation_started" in result["events"]
        assert "deliberation_completed" in result["events"]
        client.close()

    def test_stream_config_contains_all_events(self) -> None:
        """Stream config contains expected event types."""
        client = AragoraClient(base_url="https://api.aragora.ai")
        result = client.deliberations.get_stream_config()

        expected_events = [
            "deliberation_started",
            "round_started",
            "agent_response",
            "consensus_forming",
            "deliberation_completed",
            "deliberation_failed",
        ]

        for event in expected_events:
            assert event in result["events"], f"Missing event: {event}"

        client.close()


class TestAsyncDeliberations:
    """Tests for async deliberations API."""

    @pytest.mark.asyncio
    async def test_async_list_active(self) -> None:
        """List active deliberations asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"deliberations": [], "count": 0}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.deliberations.list_active()

                mock_request.assert_called_once()
                assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_async_get_stats(self) -> None:
        """Get deliberation stats asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"active_count": 3}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.deliberations.get_stats()

                mock_request.assert_called_once()
                assert result["active_count"] == 3

    @pytest.mark.asyncio
    async def test_async_get_deliberation(self) -> None:
        """Get a deliberation asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"id": "delib_123"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.deliberations.get("delib_123")

                mock_request.assert_called_once()
                assert result["id"] == "delib_123"

    @pytest.mark.asyncio
    async def test_async_get_stream_config(self) -> None:
        """Get stream config asynchronously."""
        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.deliberations.get_stream_config()

            # Should return the same static config
            assert result["type"] == "websocket"
            assert result["path"] == "/api/v1/deliberations/stream"
