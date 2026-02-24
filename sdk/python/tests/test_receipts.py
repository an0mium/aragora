"""Tests for Receipts namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestReceiptsGauntlet:
    """Tests for gauntlet receipt operations."""

    def test_list_gauntlet(self) -> None:
        """List gauntlet results."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"results": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.receipts.list_gauntlet(verdict="PASS", limit=10)

            mock_request.assert_called_once_with(
                "GET",
                "/api/v2/gauntlet/results",
                params={"limit": 10, "offset": 0, "verdict": "PASS"},
            )
            client.close()

    def test_get_gauntlet(self) -> None:
        """Get a gauntlet receipt."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"receipt_id": "gnt_123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.receipts.get_gauntlet("gnt_123")

            mock_request.assert_called_once_with("GET", "/api/v2/gauntlet/gnt_123/receipt")
            client.close()

    def test_verify_gauntlet(self) -> None:
        """Verify a gauntlet receipt."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"valid": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.receipts.verify_gauntlet("gnt_123")

            mock_request.assert_called_once_with("POST", "/api/v2/gauntlet/gnt_123/receipt/verify")
            client.close()

    def test_export_gauntlet_markdown(self) -> None:
        """Export gauntlet receipt as markdown."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"content": "# Receipt"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.receipts.export_gauntlet("gnt_123", format="markdown")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v2/gauntlet/gnt_123/receipt",
                params={"format": "md"},
            )
            client.close()


class TestReceiptsHelpers:
    """Tests for static helper methods."""

    def test_has_dissent_true(self) -> None:
        """Check receipt with dissenting views."""
        from aragora_sdk.namespaces.receipts import ReceiptsAPI

        receipt = {"dissenting_agents": ["agent_1", "agent_2"]}
        assert ReceiptsAPI.has_dissent(receipt) is True

    def test_has_dissent_false(self) -> None:
        """Check receipt without dissenting views."""
        from aragora_sdk.namespaces.receipts import ReceiptsAPI

        receipt = {"dissenting_agents": []}
        assert ReceiptsAPI.has_dissent(receipt) is False

    def test_get_consensus_status(self) -> None:
        """Get consensus status from receipt."""
        from aragora_sdk.namespaces.receipts import ReceiptsAPI

        receipt = {
            "consensus_reached": True,
            "confidence": 0.95,
            "participating_agents": ["a1", "a2", "a3"],
            "dissenting_agents": ["a3"],
        }

        status = ReceiptsAPI.get_consensus_status(receipt)
        assert status["reached"] is True
        assert status["confidence"] == 0.95
        assert status["participating_agents"] == 3
        assert status["dissenting_agents"] == 1


class TestReceiptsDeliveryBridge:
    """Tests for the v1 delivery bridge endpoint."""

    def test_deliver_v1_with_modern_fields(self) -> None:
        """Deliver a receipt using modern channel field names."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"delivered": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.receipts.deliver_v1(
                "r_123",
                channel_type="slack",
                channel_id="C123",
                workspace_id="T123",
                message="FYI",
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/receipts/r_123/deliver",
                json={
                    "channel_type": "slack",
                    "channel_id": "C123",
                    "workspace_id": "T123",
                    "message": "FYI",
                },
            )
            client.close()

    @pytest.mark.asyncio
    async def test_async_deliver_v1_with_legacy_fields(self) -> None:
        """Deliver a receipt using legacy channel/destination fields."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"delivered": True}

            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            await client.receipts.deliver_v1(
                "r_456",
                channel="teams",
                destination="chat:19:abc",
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/receipts/r_456/deliver",
                json={
                    "channel": "teams",
                    "destination": "chat:19:abc",
                },
            )
            await client.close()
