"""Tests for task namespace route mappings."""

from __future__ import annotations

from unittest.mock import call, patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient
from aragora_sdk.namespaces.tasks import AsyncTasksAPI


class TestTaskQueueRoutes:
    """Tests for /api/v1/tasks queue and lease endpoints."""

    def test_task_queue_endpoints(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"data": {}}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            client.tasks.list_queue(status="pending", work_type="code", limit=10)
            client.tasks.get_queue_task("task:demo")
            client.tasks.get_queue_stats()
            client.tasks.sync_queue(include_pending=False)
            client.tasks.claim_queue_task("task:demo", owner_agent="codex")
            client.tasks.list_leases()
            client.tasks.heartbeat_lease("lease-1", ttl_hours=2.5)
            client.tasks.release_lease("lease-1")
            client.tasks.complete_lease("lease-1", outcome="completed")
            client.tasks.list_salvage()

            expected_calls = [
                call(
                    "GET",
                    "/api/v1/tasks/queue",
                    params={"status": "pending", "work_type": "code", "limit": 10},
                ),
                call("GET", "/api/v1/tasks/queue/task:demo"),
                call("GET", "/api/v1/tasks/queue/stats"),
                call(
                    "POST",
                    "/api/v1/tasks/queue/sync",
                    json={
                        "include_pending": False,
                        "include_developer_tasks": True,
                        "complete_missing": True,
                    },
                ),
                call("POST", "/api/v1/tasks/queue/task:demo/claim", json={"owner_agent": "codex"}),
                call("GET", "/api/v1/tasks/leases"),
                call("POST", "/api/v1/tasks/leases/lease-1/heartbeat", json={"ttl_hours": 2.5}),
                call("POST", "/api/v1/tasks/leases/lease-1/release"),
                call(
                    "POST", "/api/v1/tasks/leases/lease-1/complete", json={"outcome": "completed"}
                ),
                call("GET", "/api/v1/tasks/salvage"),
            ]
            mock_request.assert_has_calls(expected_calls)
            assert mock_request.call_count == len(expected_calls)
            client.close()


class TestAsyncTaskQueueRoutes:
    """Async tests for /api/v1/tasks queue and lease endpoints."""

    @pytest.mark.asyncio
    async def test_async_task_queue_endpoints(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"data": {}}
            async with AragoraAsyncClient(
                base_url="https://api.aragora.ai", api_key="test-key"
            ) as client:
                tasks = AsyncTasksAPI(client)

                await tasks.list_queue(limit=5)
                await tasks.get_queue_task("task:async")
                await tasks.get_queue_stats()
                await tasks.sync_queue(complete_missing=False)
                await tasks.claim_queue_task("task:async", owner_agent="codex")
                await tasks.list_leases()
                await tasks.heartbeat_lease("lease-async")
                await tasks.release_lease("lease-async")
                await tasks.complete_lease("lease-async", confidence=0.9)
                await tasks.list_salvage()

                expected_calls = [
                    call("GET", "/api/v1/tasks/queue", params={"limit": 5}),
                    call("GET", "/api/v1/tasks/queue/task:async"),
                    call("GET", "/api/v1/tasks/queue/stats"),
                    call(
                        "POST",
                        "/api/v1/tasks/queue/sync",
                        json={
                            "include_pending": True,
                            "include_developer_tasks": True,
                            "complete_missing": False,
                        },
                    ),
                    call(
                        "POST",
                        "/api/v1/tasks/queue/task:async/claim",
                        json={"owner_agent": "codex"},
                    ),
                    call("GET", "/api/v1/tasks/leases"),
                    call("POST", "/api/v1/tasks/leases/lease-async/heartbeat", json={}),
                    call("POST", "/api/v1/tasks/leases/lease-async/release"),
                    call(
                        "POST",
                        "/api/v1/tasks/leases/lease-async/complete",
                        json={"confidence": 0.9},
                    ),
                    call("GET", "/api/v1/tasks/salvage"),
                ]
                mock_request.assert_has_calls(expected_calls)
                assert mock_request.call_count == len(expected_calls)
