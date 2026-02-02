"""Tests for Email Debate namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestEmailPrioritize:
    """Tests for email prioritization methods."""

    def test_prioritize_single_email(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "priority": "high",
                "score": 0.85,
                "confidence": 0.92,
                "reasoning": "Sender is CFO and subject requires action.",
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.email_debate.prioritize(
                email={
                    "subject": "Q4 Budget Review - Action Required",
                    "body": "Please review the attached budget proposal...",
                    "sender": "cfo@company.com",
                },
                user_id="user_123",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/email/prioritize",
                json={
                    "subject": "Q4 Budget Review - Action Required",
                    "body": "Please review the attached budget proposal...",
                    "sender": "cfo@company.com",
                    "user_id": "user_123",
                },
            )
            assert result["priority"] == "high"
            assert result["score"] == 0.85
            client.close()

    def test_prioritize_without_user_id(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"priority": "low", "score": 0.2}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.email_debate.prioritize(
                email={"sender": "newsletter@example.com", "subject": "Weekly Digest"},
            )
            call_json = mock_request.call_args[1]["json"]
            assert "user_id" not in call_json
            assert call_json["sender"] == "newsletter@example.com"
            client.close()

    def test_prioritize_batch(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "results": [
                    {"priority": "high", "score": 0.9},
                    {"priority": "low", "score": 0.1},
                ],
                "total_processed": 2,
                "processing_time_ms": 320,
                "summary": {"high": 1, "low": 1},
            }
            emails = [
                {"sender": "boss@company.com", "subject": "Urgent meeting"},
                {"sender": "spam@junk.com", "subject": "You won a prize"},
            ]
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.email_debate.prioritize_batch(
                emails=emails,
                user_id="user_456",
                parallel=False,
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/email/prioritize/batch",
                json={
                    "emails": emails,
                    "parallel": False,
                    "user_id": "user_456",
                },
            )
            assert result["total_processed"] == 2
            assert result["summary"]["high"] == 1
            client.close()


class TestEmailTriageAndHistory:
    """Tests for inbox triage and history retrieval."""

    def test_triage_inbox(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "results": [{"category": "action_required", "priority": "high"}],
                "total_triaged": 1,
                "processing_time_ms": 150,
            }
            emails = [
                {"sender": "manager@co.com", "subject": "Review PR", "body": "Please review."}
            ]
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.email_debate.triage_inbox(
                emails=emails,
                user_id="user_789",
                include_auto_replies=True,
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/email/triage",
                json={
                    "emails": emails,
                    "include_auto_replies": True,
                    "user_id": "user_789",
                },
            )
            assert result["total_triaged"] == 1
            assert result["results"][0]["category"] == "action_required"
            client.close()

    def test_get_history_basic(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"results": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.email_debate.get_history(user_id="user_123")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/email/prioritize/history",
                params={"user_id": "user_123"},
            )
            assert result["total"] == 0
            client.close()

    def test_get_history_with_filters(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"results": [{"priority": "high"}], "total": 1}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.email_debate.get_history(
                user_id="user_123",
                limit=10,
                since="2025-01-01T00:00:00Z",
            )
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/email/prioritize/history",
                params={
                    "user_id": "user_123",
                    "limit": 10,
                    "since": "2025-01-01T00:00:00Z",
                },
            )
            client.close()


class TestAsyncEmailDebate:
    """Tests for async email debate methods."""

    @pytest.mark.asyncio
    async def test_prioritize(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"priority": "critical", "score": 0.98}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.email_debate.prioritize(
                email={"sender": "alerts@monitoring.com", "subject": "Server Down"},
                user_id="user_ops",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/email/prioritize",
                json={
                    "sender": "alerts@monitoring.com",
                    "subject": "Server Down",
                    "user_id": "user_ops",
                },
            )
            assert result["priority"] == "critical"
            await client.close()

    @pytest.mark.asyncio
    async def test_triage_inbox(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "results": [{"category": "fyi"}],
                "total_triaged": 1,
            }
            emails = [{"sender": "info@news.com", "subject": "Weekly Update"}]
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.email_debate.triage_inbox(emails=emails)
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/email/triage",
                json={
                    "emails": emails,
                    "include_auto_replies": False,
                },
            )
            assert result["results"][0]["category"] == "fyi"
            await client.close()

    @pytest.mark.asyncio
    async def test_get_history(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"results": [{"priority": "medium"}], "total": 1}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.email_debate.get_history(user_id="user_abc", limit=5)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/email/prioritize/history",
                params={"user_id": "user_abc", "limit": 5},
            )
            assert result["total"] == 1
            await client.close()
