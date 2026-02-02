"""Tests for Receipts namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestReceiptsList:
    """Tests for listing receipts."""

    def test_list_receipts_default(self) -> None:
        """List receipts with default pagination."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "receipts": [],
                "total": 0,
                "limit": 20,
                "offset": 0,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.receipts.list()

            mock_request.assert_called_once_with(
                "GET", "/api/v2/receipts", params={"limit": 20, "offset": 0}
            )
            assert result["total"] == 0
            client.close()

    def test_list_receipts_with_filters(self) -> None:
        """List receipts with various filters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"receipts": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.receipts.list(
                verdict="APPROVED",
                risk_level="high",
                from_date="2024-01-01",
                to_date="2024-12-31",
                signed_only=True,
                limit=50,
                offset=10,
            )

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["verdict"] == "APPROVED"
            assert params["risk_level"] == "high"
            assert params["from_date"] == "2024-01-01"
            assert params["to_date"] == "2024-12-31"
            assert params["signed_only"] is True
            assert params["limit"] == 50
            assert params["offset"] == 10
            client.close()


class TestReceiptsGet:
    """Tests for getting receipt details."""

    def test_get_receipt(self) -> None:
        """Get a receipt by ID."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "receipt_id": "rcp_123",
                "verdict": "APPROVED",
                "confidence": 0.95,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.receipts.get("rcp_123")

            mock_request.assert_called_once_with("GET", "/api/v2/receipts/rcp_123")
            assert result["receipt_id"] == "rcp_123"
            client.close()

    def test_search_receipts(self) -> None:
        """Full-text search across receipts."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"receipts": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.receipts.search("microservices", limit=10, offset=5)

            mock_request.assert_called_once_with(
                "GET",
                "/api/v2/receipts/search",
                params={"query": "microservices", "limit": 10, "offset": 5},
            )
            client.close()


class TestReceiptsVerify:
    """Tests for receipt verification."""

    def test_verify_receipt(self) -> None:
        """Verify a receipt's integrity."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "valid": True,
                "checksum": "sha256:abc123",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.receipts.verify("rcp_123")

            mock_request.assert_called_once_with("POST", "/api/v2/receipts/rcp_123/verify")
            assert result["valid"] is True
            client.close()

    def test_verify_signature(self) -> None:
        """Verify a receipt's cryptographic signature."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "valid": True,
                "algorithm": "RS256",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.receipts.verify_signature("rcp_123")

            mock_request.assert_called_once_with(
                "POST", "/api/v2/receipts/rcp_123/verify-signature"
            )
            assert result["valid"] is True
            client.close()

    def test_verify_batch(self) -> None:
        """Verify multiple receipts in batch."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "results": [
                    {"receipt_id": "rcp_1", "valid": True},
                    {"receipt_id": "rcp_2", "valid": True},
                ]
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.receipts.verify_batch(["rcp_1", "rcp_2"])

            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/receipts/verify-batch",
                json={"receipt_ids": ["rcp_1", "rcp_2"]},
            )
            assert len(result["results"]) == 2
            client.close()


class TestReceiptsExport:
    """Tests for receipt export."""

    def test_export_json(self) -> None:
        """Export receipt as JSON."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": {}}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.receipts.export("rcp_123", format="json")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v2/receipts/rcp_123/export",
                params={"format": "json"},
            )
            client.close()

    def test_export_pdf(self) -> None:
        """Export receipt as PDF."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"url": "https://cdn.aragora.ai/exports/..."}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.receipts.export("rcp_123", format="pdf")

            call_args = mock_request.call_args
            assert call_args[1]["params"]["format"] == "pdf"
            client.close()

    def test_batch_export(self) -> None:
        """Export multiple receipts as ZIP."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"url": "https://cdn.aragora.ai/bundles/..."}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.receipts.batch_export(["rcp_1", "rcp_2"], format="markdown")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/receipts/batch-export",
                json={"receipt_ids": ["rcp_1", "rcp_2"], "format": "markdown"},
            )
            client.close()


class TestReceiptsStats:
    """Tests for receipt statistics."""

    def test_get_stats(self) -> None:
        """Get receipt statistics."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "total": 100,
                "by_verdict": {"APPROVED": 80, "REJECTED": 20},
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.receipts.get_stats()

            mock_request.assert_called_once_with("GET", "/api/v2/receipts/stats", params={})
            assert result["total"] == 100
            client.close()

    def test_get_stats_with_dates(self) -> None:
        """Get receipt statistics with date range."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"total": 50}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.receipts.get_stats(from_date="2024-01-01", to_date="2024-06-30")

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["from_date"] == "2024-01-01"
            assert params["to_date"] == "2024-06-30"
            client.close()


class TestReceiptsShare:
    """Tests for receipt sharing."""

    def test_share_receipt(self) -> None:
        """Create a shareable link for a receipt."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "share_url": "https://share.aragora.ai/rcp_123",
                "expires_at": "2024-12-31T23:59:59Z",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.receipts.share("rcp_123", expires_in_hours=48)

            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/receipts/rcp_123/share",
                json={"expires_in_hours": 48},
            )
            assert "share_url" in result
            client.close()

    def test_share_receipt_with_max_accesses(self) -> None:
        """Create a shareable link with access limit."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"share_url": "https://share.aragora.ai/..."}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.receipts.share("rcp_123", max_accesses=5)

            call_args = mock_request.call_args
            assert call_args[1]["json"]["max_accesses"] == 5
            client.close()

    def test_send_to_channel(self) -> None:
        """Send a receipt to a channel."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"sent": True, "channel": "slack"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.receipts.send_to_channel(
                "rcp_123", channel_type="slack", channel_id="C12345"
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/receipts/rcp_123/send-to-channel",
                json={"channel_type": "slack", "channel_id": "C12345"},
            )
            assert result["sent"] is True
            client.close()


class TestReceiptsCompliance:
    """Tests for compliance-related operations."""

    def test_get_retention_status(self) -> None:
        """Get retention status for GDPR compliance."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "retention_policy": "90 days",
                "expiring_soon": 5,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.receipts.get_retention_status()

            mock_request.assert_called_once_with("GET", "/api/v2/receipts/retention-status")
            assert result["retention_policy"] == "90 days"
            client.close()

    def test_get_dsar(self) -> None:
        """Get receipts for a user (GDPR DSAR)."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "user_id": "user_123",
                "receipts": [],
                "total": 0,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.receipts.get_dsar("user_123")

            mock_request.assert_called_once_with("GET", "/api/v2/receipts/dsar/user_123")
            assert result["user_id"] == "user_123"
            client.close()


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
                "/api/v1/gauntlet/results",
                params={"limit": 10, "offset": 0, "verdict": "PASS"},
            )
            client.close()

    def test_get_gauntlet(self) -> None:
        """Get a gauntlet receipt."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"receipt_id": "gnt_123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.receipts.get_gauntlet("gnt_123")

            mock_request.assert_called_once_with("GET", "/api/v1/gauntlet/gnt_123/receipt")
            client.close()

    def test_verify_gauntlet(self) -> None:
        """Verify a gauntlet receipt."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"valid": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.receipts.verify_gauntlet("gnt_123")

            mock_request.assert_called_once_with("POST", "/api/v1/gauntlet/gnt_123/receipt/verify")
            client.close()

    def test_export_gauntlet_markdown(self) -> None:
        """Export gauntlet receipt as markdown."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"content": "# Receipt"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.receipts.export_gauntlet("gnt_123", format="markdown")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/gauntlet/gnt_123/receipt",
                params={"format": "md"},
            )
            client.close()


class TestReceiptsBatch:
    """Tests for batch operations."""

    def test_sign_batch(self) -> None:
        """Sign multiple receipts in batch."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "signed": ["rcp_1", "rcp_2"],
                "failed": [],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.receipts.sign_batch(["rcp_1", "rcp_2"])

            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/receipts/sign-batch",
                json={"receipt_ids": ["rcp_1", "rcp_2"]},
            )
            assert len(result["signed"]) == 2
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


class TestAsyncReceipts:
    """Tests for async receipts API."""

    @pytest.mark.asyncio
    async def test_async_list(self) -> None:
        """List receipts asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"receipts": [], "total": 0}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.receipts.list(verdict="APPROVED")

                call_args = mock_request.call_args
                assert call_args[1]["params"]["verdict"] == "APPROVED"

    @pytest.mark.asyncio
    async def test_async_get(self) -> None:
        """Get receipt asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"receipt_id": "rcp_123"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.receipts.get("rcp_123")

                mock_request.assert_called_once_with("GET", "/api/v2/receipts/rcp_123")
                assert result["receipt_id"] == "rcp_123"

    @pytest.mark.asyncio
    async def test_async_verify(self) -> None:
        """Verify receipt asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"valid": True}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.receipts.verify("rcp_123")

                mock_request.assert_called_once_with("POST", "/api/v2/receipts/rcp_123/verify")
                assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_async_export(self) -> None:
        """Export receipt asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"data": {}}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.receipts.export("rcp_123", format="sarif")

                mock_request.assert_called_once_with(
                    "GET",
                    "/api/v2/receipts/rcp_123/export",
                    params={"format": "sarif"},
                )

    @pytest.mark.asyncio
    async def test_async_share(self) -> None:
        """Share receipt asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"share_url": "https://..."}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.receipts.share("rcp_123", expires_in_hours=72)

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/v2/receipts/rcp_123/share",
                    json={"expires_in_hours": 72},
                )

    @pytest.mark.asyncio
    async def test_async_verify_batch(self) -> None:
        """Verify batch asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"results": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.receipts.verify_batch(["rcp_1", "rcp_2", "rcp_3"])

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/v2/receipts/verify-batch",
                    json={"receipt_ids": ["rcp_1", "rcp_2", "rcp_3"]},
                )
