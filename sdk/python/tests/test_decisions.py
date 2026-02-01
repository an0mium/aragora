"""Tests for Decisions namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


class TestDecisionsSubmit:
    """Tests for decision submission."""

    def test_submit_decision(self) -> None:
        """Submit a decision request."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "decision_id": "dec_123",
                "status": "pending",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.decisions.submit(input="Should we adopt microservices?")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/decisions",
                params=None,
                json={
                    "input": "Should we adopt microservices?",
                    "decision_type": "debate",
                    "priority": "normal",
                },
                headers=None,
            )
            assert result["decision_id"] == "dec_123"
            client.close()

    def test_submit_decision_with_type(self) -> None:
        """Submit a decision with specific type."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"decision_id": "dec_123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.decisions.submit(input="Question", decision_type="gauntlet")

            call_args = mock_request.call_args
            assert call_args[1]["json"]["decision_type"] == "gauntlet"
            client.close()

    def test_submit_decision_with_context(self) -> None:
        """Submit a decision with context."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"decision_id": "dec_123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            context = {"workspace_id": "ws_123"}
            client.decisions.submit(input="Question", context=context)

            call_args = mock_request.call_args
            assert call_args[1]["json"]["context"] == context
            client.close()


class TestDecisionsGet:
    """Tests for getting decision details."""

    def test_get_decision_by_id(self) -> None:
        """Get a decision by its ID."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "decision_id": "dec_123",
                "status": "completed",
                "answer": "Yes, adopt microservices",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.decisions.get("dec_123")

            mock_request.assert_called_once()
            assert result["decision_id"] == "dec_123"
            client.close()

    def test_get_decision_status(self) -> None:
        """Get the status of a decision."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "decision_id": "dec_123",
                "status": "processing",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.decisions.get_status("dec_123")

            mock_request.assert_called_once()
            assert result["status"] == "processing"
            client.close()


class TestDecisionsList:
    """Tests for listing decisions."""

    def test_list_decisions_default_pagination(self) -> None:
        """List decisions with default pagination."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "decisions": [],
                "total": 0,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.decisions.list()

            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[1]["params"]["limit"] == 50
            assert call_args[1]["params"]["offset"] == 0
            client.close()

    def test_list_decisions_with_filters(self) -> None:
        """List decisions with status filter."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"decisions": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.decisions.list(status="completed", decision_type="debate")

            call_args = mock_request.call_args
            assert call_args[1]["params"]["status"] == "completed"
            assert call_args[1]["params"]["decision_type"] == "debate"
            client.close()


class TestDecisionsCancel:
    """Tests for cancelling decisions."""

    def test_cancel_decision(self) -> None:
        """Cancel a decision."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "status": "cancelled",
                "cancelled_at": "2024-01-01T00:00:00Z",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.decisions.cancel("dec_123", reason="No longer needed")

            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[1]["json"]["reason"] == "No longer needed"
            assert result["status"] == "cancelled"
            client.close()

    def test_cancel_decision_no_reason(self) -> None:
        """Cancel a decision without reason."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"status": "cancelled"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.decisions.cancel("dec_123")

            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[1]["json"] is None
            client.close()


class TestDecisionsRetry:
    """Tests for retrying decisions."""

    def test_retry_decision(self) -> None:
        """Retry a failed decision."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "decision_id": "dec_124",
                "status": "completed",
                "retried_from": "dec_123",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.decisions.retry("dec_123")

            mock_request.assert_called_once()
            assert result["retried_from"] == "dec_123"
            client.close()


class TestDecisionsReceipt:
    """Tests for getting decision receipts."""

    def test_get_receipt(self) -> None:
        """Get receipt for a completed decision."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "receipt_id": "rec_123",
                "decision_id": "dec_123",
                "signature": "sig_abc",
                "timestamp": "2024-01-01T00:00:00Z",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.decisions.get_receipt("dec_123")

            mock_request.assert_called_once()
            # Verify the path includes /api/v2/receipts/
            call_args = mock_request.call_args
            assert "/api/v2/receipts/dec_123" in call_args[0]
            assert result["receipt_id"] == "rec_123"
            client.close()


class TestDecisionsExplanation:
    """Tests for getting decision explanations."""

    def test_get_explanation(self) -> None:
        """Get explanation for a decision."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "factors": [{"name": "evidence", "weight": 0.8}],
                "narrative": "The decision was made because...",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.decisions.get_explanation("dec_123")

            mock_request.assert_called_once()
            assert "factors" in result
            client.close()


class TestDecisionsFeedback:
    """Tests for submitting feedback on decisions."""

    def test_submit_feedback(self) -> None:
        """Submit feedback on a decision."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "success": True,
                "feedback_id": "fb_123",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.decisions.submit_feedback(
                decision_id="dec_123",
                rating=5,
                comment="Great decision!",
            )

            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[1]["json"]["type"] == "debate_quality"
            assert call_args[1]["json"]["score"] == 5
            assert call_args[1]["json"]["comment"] == "Great decision!"
            assert call_args[1]["json"]["context"]["decision_id"] == "dec_123"
            assert result["success"] is True
            client.close()

    def test_submit_feedback_no_comment(self) -> None:
        """Submit feedback without comment."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"success": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.decisions.submit_feedback(decision_id="dec_123", rating=4)

            call_args = mock_request.call_args
            assert call_args[1]["json"]["score"] == 4
            assert "decision_id" in call_args[1]["json"]["context"]
            # Default comment is generated
            assert "dec_123" in call_args[1]["json"]["comment"]
            client.close()


class TestAsyncDecisions:
    """Tests for async decisions API."""

    @pytest.mark.asyncio
    async def test_async_submit_decision(self) -> None:
        """Submit a decision asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"decision_id": "dec_123"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.decisions.submit(input="Async decision")

                mock_request.assert_called_once()
                assert result["decision_id"] == "dec_123"

    @pytest.mark.asyncio
    async def test_async_get_decision(self) -> None:
        """Get a decision asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"decision_id": "dec_123"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.decisions.get("dec_123")

                mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_get_receipt(self) -> None:
        """Get receipt asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"receipt_id": "rec_123"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.decisions.get_receipt("dec_123")

                mock_request.assert_called_once()
                assert result["receipt_id"] == "rec_123"

    @pytest.mark.asyncio
    async def test_async_submit_feedback(self) -> None:
        """Submit feedback asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"success": True}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.decisions.submit_feedback(
                    decision_id="dec_123",
                    rating=5,
                    comment="Async feedback",
                )

                mock_request.assert_called_once()
                call_args = mock_request.call_args
                assert call_args[1]["json"]["score"] == 5
                assert result["success"] is True
