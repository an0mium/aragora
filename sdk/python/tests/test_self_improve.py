"""Tests for Self-Improve namespace API."""

from __future__ import annotations

from unittest.mock import call, patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestSelfImproveSync:
    """Synchronous self-improve endpoint tests."""

    def test_feedback_and_goals_routes(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"ok": True}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            client.self_improve.submit_feedback({"score": 5, "notes": "good"})
            client.self_improve.get_feedback_summary({"period": "30d"})
            client.self_improve.upsert_goals({"goals": ["reduce regressions"]})
            client.self_improve.get_metrics_summary({"period": "30d"})
            client.self_improve.get_regression_history({"period": "30d"})

            expected = [
                call("POST", "/api/v1/self-improve/feedback", json={"score": 5, "notes": "good"}),
                call("POST", "/api/v1/self-improve/feedback-summary", json={"period": "30d"}),
                call("POST", "/api/v1/self-improve/goals", json={"goals": ["reduce regressions"]}),
                call("POST", "/api/v1/self-improve/metrics/summary", json={"period": "30d"}),
                call("POST", "/api/v1/self-improve/regression-history", json={"period": "30d"}),
            ]
            mock_request.assert_has_calls(expected)
            assert mock_request.call_count == 5
            client.close()


class TestSelfImproveAsync:
    """Asynchronous self-improve endpoint tests."""

    @pytest.mark.asyncio
    async def test_feedback_and_goals_routes(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"ok": True}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")

            await client.self_improve.submit_feedback({"score": 4})
            await client.self_improve.get_feedback_summary({"period": "7d"})
            await client.self_improve.upsert_goals({"goals": ["increase consensus"]})
            await client.self_improve.get_metrics_summary({"period": "7d"})
            await client.self_improve.get_regression_history({"period": "7d"})

            expected = [
                call("POST", "/api/v1/self-improve/feedback", json={"score": 4}),
                call("POST", "/api/v1/self-improve/feedback-summary", json={"period": "7d"}),
                call("POST", "/api/v1/self-improve/goals", json={"goals": ["increase consensus"]}),
                call("POST", "/api/v1/self-improve/metrics/summary", json={"period": "7d"}),
                call("POST", "/api/v1/self-improve/regression-history", json={"period": "7d"}),
            ]
            mock_request.assert_has_calls(expected)
            assert mock_request.call_count == 5
            await client.close()
