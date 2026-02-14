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

