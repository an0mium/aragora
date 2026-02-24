"""Tests for Gauntlet namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestGauntletRun:
    """Tests for running Gauntlet validations."""

    def test_run_with_debate_id(self) -> None:
        """Run Gauntlet on an existing debate."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "gauntlet_id": "gnt_123",
                "verdict": "PASS",
                "confidence": 0.92,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.gauntlet.run(debate_id="dbt_123")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/gauntlet/run",
                json={"attack_rounds": 3, "debate_id": "dbt_123"},
            )
            assert result["verdict"] == "PASS"
            client.close()

    def test_run_with_task(self) -> None:
        """Run Gauntlet with a new task."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"gauntlet_id": "gnt_123", "verdict": "CONDITIONAL"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gauntlet.run(task="Should we deploy to production?")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/gauntlet/run",
                json={"attack_rounds": 3, "task": "Should we deploy to production?"},
            )
            client.close()

    def test_run_with_custom_rounds(self) -> None:
        """Run Gauntlet with custom attack rounds."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"gauntlet_id": "gnt_123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gauntlet.run(debate_id="dbt_123", attack_rounds=5)

            call_args = mock_request.call_args
            assert call_args[1]["json"]["attack_rounds"] == 5
            client.close()

    def test_run_with_agents(self) -> None:
        """Run Gauntlet with specific agents."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"gauntlet_id": "gnt_123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.gauntlet.run(
                task="Test decision",
                proposer_agent="claude",
                attacker_agents=["gpt-4", "gemini"],
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["proposer_agent"] == "claude"
            assert call_args[1]["json"]["attacker_agents"] == ["gpt-4", "gemini"]
            client.close()


class TestGauntletResults:
    """Tests for Gauntlet result operations."""

    def test_get_result(self) -> None:
        """Get a Gauntlet result by ID."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "gauntlet_id": "gnt_123",
                "verdict": "PASS",
                "findings_count": 2,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.gauntlet.get_result("gnt_123")

            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/gnt_123")
            assert result["gauntlet_id"] == "gnt_123"
            client.close()


class TestGauntletReceipts:
    """Tests for Gauntlet receipt operations."""

    def test_get_receipt(self) -> None:
        """Get the decision receipt for a Gauntlet run."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "receipt_id": "rcp_123",
                "hash": "sha256:abc123",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.gauntlet.get_receipt("gnt_123")

            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/gnt_123/receipt")
            assert result["receipt_id"] == "rcp_123"
            client.close()

    def test_verify_receipt(self) -> None:
        """Verify a Gauntlet receipt's integrity."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "valid": True,
                "hash": "sha256:abc123",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.gauntlet.verify_receipt("gnt_123")

            mock_request.assert_called_once_with("POST", "/api/v1/gauntlet/gnt_123/receipt/verify")
            assert result["valid"] is True
            client.close()


class TestAsyncGauntlet:
    """Tests for async Gauntlet API."""

    @pytest.mark.asyncio
    async def test_async_run(self) -> None:
        """Run Gauntlet asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"gauntlet_id": "gnt_123", "verdict": "PASS"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.gauntlet.run(task="Test decision")

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/v1/gauntlet/run",
                    json={"attack_rounds": 3, "task": "Test decision"},
                )
                assert result["verdict"] == "PASS"

    @pytest.mark.asyncio
    async def test_async_get_result(self) -> None:
        """Get result asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"gauntlet_id": "gnt_123"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.gauntlet.get_result("gnt_123")

                mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/gnt_123")

    @pytest.mark.asyncio
    async def test_async_get_receipt(self) -> None:
        """Get receipt asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"receipt_id": "rcp_123"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.gauntlet.get_receipt("gnt_123")

                mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/gnt_123/receipt")

    @pytest.mark.asyncio
    async def test_async_verify_receipt(self) -> None:
        """Verify receipt asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"valid": True}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.gauntlet.verify_receipt("gnt_123")

                mock_request.assert_called_once_with(
                    "POST", "/api/v1/gauntlet/gnt_123/receipt/verify"
                )
                assert result["valid"] is True
