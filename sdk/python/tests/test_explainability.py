"""Tests for Explainability namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient

class TestExplainabilityComparison:
    """Tests for decision comparison operations."""

    def test_compare_decisions(self) -> None:
        """Compare two decisions."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "decision_1": "decision_123",
                "decision_2": "decision_456",
                "differences": [
                    {"factor": "cost", "delta": 0.15},
                ],
                "similarity_score": 0.72,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.explainability.compare_decisions("decision_123", "decision_456")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/explainability/compare",
                params={"decision_1": "decision_123", "decision_2": "decision_456"},
            )
            assert result["similarity_score"] == 0.72
            client.close()

class TestAsyncExplainability:
    """Tests for async explainability API."""

    @pytest.mark.asyncio
    async def test_async_compare_decisions(self) -> None:
        """Compare decisions asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"similarity_score": 0.85}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.explainability.compare_decisions(
                    "decision_123", "decision_456"
                )

                mock_request.assert_called_once_with(
                    "GET",
                    "/api/v1/explainability/compare",
                    params={"decision_1": "decision_123", "decision_2": "decision_456"},
                )
                assert result["similarity_score"] == 0.85
