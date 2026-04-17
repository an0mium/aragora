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

    def test_detail_and_improvement_queue_routes(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"ok": True}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            client.self_improve.get_meta_planner_goals()
            client.self_improve.get_execution_timeline()
            client.self_improve.get_learning_insights()
            client.self_improve.get_metrics_comparison()
            client.self_improve.get_cycle_trends()
            client.self_improve.add_improvement_queue_item(
                "reduce flaky checks", priority=80, source="operator"
            )
            client.self_improve.update_improvement_queue_priority("item-1", 60)
            client.self_improve.delete_improvement_queue_item("item-1")

            expected = [
                call("GET", "/api/v1/self-improve/meta-planner/goals"),
                call("GET", "/api/v1/self-improve/execution/timeline"),
                call("GET", "/api/v1/self-improve/learning/insights"),
                call("GET", "/api/v1/self-improve/metrics/comparison"),
                call("GET", "/api/v1/self-improve/trends/cycles"),
                call(
                    "POST",
                    "/api/v1/self-improve/improvement-queue",
                    json={"goal": "reduce flaky checks", "priority": 80, "source": "operator"},
                ),
                call(
                    "PUT",
                    "/api/v1/self-improve/improvement-queue/item-1/priority",
                    json={"priority": 60},
                ),
                call("DELETE", "/api/v1/self-improve/improvement-queue/item-1"),
            ]
            mock_request.assert_has_calls(expected)
            assert mock_request.call_count == 8
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

    @pytest.mark.asyncio
    async def test_detail_and_improvement_queue_routes(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"ok": True}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")

            await client.self_improve.get_meta_planner_goals()
            await client.self_improve.get_execution_timeline()
            await client.self_improve.get_learning_insights()
            await client.self_improve.get_metrics_comparison()
            await client.self_improve.get_cycle_trends()
            await client.self_improve.add_improvement_queue_item(
                "reduce flaky checks", priority=80, source="operator"
            )
            await client.self_improve.update_improvement_queue_priority("item-1", 60)
            await client.self_improve.delete_improvement_queue_item("item-1")

            expected = [
                call("GET", "/api/v1/self-improve/meta-planner/goals"),
                call("GET", "/api/v1/self-improve/execution/timeline"),
                call("GET", "/api/v1/self-improve/learning/insights"),
                call("GET", "/api/v1/self-improve/metrics/comparison"),
                call("GET", "/api/v1/self-improve/trends/cycles"),
                call(
                    "POST",
                    "/api/v1/self-improve/improvement-queue",
                    json={"goal": "reduce flaky checks", "priority": 80, "source": "operator"},
                ),
                call(
                    "PUT",
                    "/api/v1/self-improve/improvement-queue/item-1/priority",
                    json={"priority": 60},
                ),
                call("DELETE", "/api/v1/self-improve/improvement-queue/item-1"),
            ]
            mock_request.assert_has_calls(expected)
            assert mock_request.call_count == 8
            await client.close()
