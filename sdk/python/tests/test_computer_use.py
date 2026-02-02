"""Tests for Computer Use namespace API.

Tests cover both sync (ComputerUseAPI) and async (AsyncComputerUseAPI) classes
for the computer use orchestration functionality.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient

# =========================================================================
# Task Operations - Sync
# =========================================================================


class TestComputerUseCreateTask:
    """Tests for creating computer use tasks."""

    def test_create_task_default(self) -> None:
        """Create a task with default parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "task_id": "cu_task_123",
                "status": "pending",
                "message": "Task created successfully",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.computer_use.create_task(
                goal="Open the settings page and enable dark mode"
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/computer-use/tasks",
                json={
                    "goal": "Open the settings page and enable dark mode",
                    "max_steps": 10,
                    "dry_run": False,
                },
            )
            assert result["task_id"] == "cu_task_123"
            assert result["status"] == "pending"
            client.close()

    def test_create_task_with_max_steps(self) -> None:
        """Create a task with custom max_steps."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"task_id": "cu_task_456", "status": "pending"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.computer_use.create_task(
                goal="Navigate to dashboard and export report", max_steps=25
            )

            call_args = mock_request.call_args
            assert call_args[1]["json"]["max_steps"] == 25
            assert result["task_id"] == "cu_task_456"
            client.close()

    def test_create_task_dry_run(self) -> None:
        """Create a task in dry run mode."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "task_id": "cu_task_789",
                "status": "pending",
                "dry_run": True,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.computer_use.create_task(
                goal="Click the submit button", max_steps=5, dry_run=True
            )

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            assert json_data["goal"] == "Click the submit button"
            assert json_data["max_steps"] == 5
            assert json_data["dry_run"] is True
            assert result["dry_run"] is True
            client.close()


class TestComputerUseListTasks:
    """Tests for listing computer use tasks."""

    def test_list_tasks_default(self) -> None:
        """List tasks with default parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "tasks": [
                    {"task_id": "cu_task_1", "status": "completed"},
                    {"task_id": "cu_task_2", "status": "running"},
                ],
                "total": 2,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.computer_use.list_tasks()

            mock_request.assert_called_once_with(
                "GET", "/api/v1/computer-use/tasks", params={"limit": 20}
            )
            assert len(result["tasks"]) == 2
            assert result["total"] == 2
            client.close()

    def test_list_tasks_with_limit(self) -> None:
        """List tasks with custom limit."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"tasks": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.computer_use.list_tasks(limit=50)

            call_args = mock_request.call_args
            assert call_args[1]["params"]["limit"] == 50
            client.close()

    def test_list_tasks_filtered_by_status(self) -> None:
        """List tasks filtered by status."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "tasks": [{"task_id": "cu_task_1", "status": "running"}],
                "total": 1,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.computer_use.list_tasks(status="running")

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["status"] == "running"
            assert result["tasks"][0]["status"] == "running"
            client.close()

    def test_list_tasks_completed_status(self) -> None:
        """List tasks filtered by completed status."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"tasks": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.computer_use.list_tasks(limit=10, status="completed")

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["limit"] == 10
            assert params["status"] == "completed"
            client.close()

    def test_list_tasks_failed_status(self) -> None:
        """List tasks filtered by failed status."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"tasks": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.computer_use.list_tasks(status="failed")

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["status"] == "failed"
            client.close()


class TestComputerUseGetTask:
    """Tests for getting computer use task details."""

    def test_get_task(self) -> None:
        """Get task status and details."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "task": {
                    "task_id": "cu_task_123",
                    "status": "completed",
                    "goal": "Open settings",
                    "steps_taken": 5,
                    "max_steps": 10,
                    "result": "Successfully opened settings page",
                }
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.computer_use.get_task("cu_task_123")

            mock_request.assert_called_once_with("GET", "/api/v1/computer-use/tasks/cu_task_123")
            assert result["task"]["task_id"] == "cu_task_123"
            assert result["task"]["status"] == "completed"
            assert result["task"]["steps_taken"] == 5
            client.close()

    def test_get_task_running(self) -> None:
        """Get a running task's status."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "task": {
                    "task_id": "cu_task_456",
                    "status": "running",
                    "goal": "Fill out form",
                    "steps_taken": 3,
                    "max_steps": 15,
                    "current_step": "Typing in email field",
                }
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.computer_use.get_task("cu_task_456")

            assert result["task"]["status"] == "running"
            assert result["task"]["current_step"] == "Typing in email field"
            client.close()


class TestComputerUseCancelTask:
    """Tests for cancelling computer use tasks."""

    def test_cancel_task(self) -> None:
        """Cancel a running task."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "message": "Task cancelled successfully",
                "task_id": "cu_task_123",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.computer_use.cancel_task("cu_task_123")

            mock_request.assert_called_once_with(
                "POST", "/api/v1/computer-use/tasks/cu_task_123/cancel"
            )
            assert "cancelled" in result["message"].lower()
            client.close()


# =========================================================================
# Action Operations - Sync
# =========================================================================


class TestComputerUseActionStats:
    """Tests for action statistics."""

    def test_get_action_stats(self) -> None:
        """Get action statistics."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "stats": {
                    "screenshot": {"count": 150, "avg_duration_ms": 250},
                    "click": {"count": 89, "avg_duration_ms": 50},
                    "type": {"count": 45, "avg_duration_ms": 120},
                    "scroll": {"count": 32, "avg_duration_ms": 80},
                    "wait": {"count": 28, "avg_duration_ms": 1000},
                },
                "total_actions": 344,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.computer_use.get_action_stats()

            mock_request.assert_called_once_with("GET", "/api/v1/computer-use/actions/stats")
            assert result["total_actions"] == 344
            assert result["stats"]["screenshot"]["count"] == 150
            assert result["stats"]["click"]["count"] == 89
            client.close()


# =========================================================================
# Policy Operations - Sync
# =========================================================================


class TestComputerUseListPolicies:
    """Tests for listing computer use policies."""

    def test_list_policies(self) -> None:
        """List active policies."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "policies": [
                    {
                        "policy_id": "cu_pol_1",
                        "name": "Production Safety",
                        "allowed_actions": ["screenshot", "click", "scroll"],
                        "blocked_domains": ["admin.example.com"],
                    },
                    {
                        "policy_id": "cu_pol_2",
                        "name": "Development",
                        "allowed_actions": ["screenshot", "click", "type", "scroll"],
                        "blocked_domains": [],
                    },
                ],
                "total": 2,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.computer_use.list_policies()

            mock_request.assert_called_once_with("GET", "/api/v1/computer-use/policies")
            assert len(result["policies"]) == 2
            assert result["policies"][0]["name"] == "Production Safety"
            client.close()


class TestComputerUseCreatePolicy:
    """Tests for creating computer use policies."""

    def test_create_policy_minimal(self) -> None:
        """Create a policy with only required fields."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "policy_id": "cu_pol_new",
                "message": "Policy created successfully",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.computer_use.create_policy(name="Basic Policy")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/computer-use/policies",
                json={"name": "Basic Policy"},
            )
            assert result["policy_id"] == "cu_pol_new"
            client.close()

    def test_create_policy_with_description(self) -> None:
        """Create a policy with description."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"policy_id": "cu_pol_desc"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.computer_use.create_policy(
                name="Documented Policy",
                description="This policy restricts actions for safety",
            )

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            assert json_data["name"] == "Documented Policy"
            assert json_data["description"] == "This policy restricts actions for safety"
            client.close()

    def test_create_policy_with_allowed_actions(self) -> None:
        """Create a policy with allowed actions."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"policy_id": "cu_pol_actions"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.computer_use.create_policy(
                name="Read-Only Policy",
                allowed_actions=["screenshot", "scroll"],
            )

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            assert json_data["allowed_actions"] == ["screenshot", "scroll"]
            client.close()

    def test_create_policy_with_blocked_domains(self) -> None:
        """Create a policy with blocked domains."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"policy_id": "cu_pol_domains"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.computer_use.create_policy(
                name="Domain Restricted Policy",
                blocked_domains=["admin.example.com", "internal.corp.net"],
            )

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            assert json_data["blocked_domains"] == [
                "admin.example.com",
                "internal.corp.net",
            ]
            client.close()

    def test_create_policy_full(self) -> None:
        """Create a policy with all options."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"policy_id": "cu_pol_full"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.computer_use.create_policy(
                name="Comprehensive Policy",
                description="Full-featured policy for production use",
                allowed_actions=["screenshot", "click", "type", "scroll", "wait"],
                blocked_domains=["admin.example.com", "sensitive.internal.com"],
            )

            call_args = mock_request.call_args
            json_data = call_args[1]["json"]
            assert json_data["name"] == "Comprehensive Policy"
            assert json_data["description"] == "Full-featured policy for production use"
            assert len(json_data["allowed_actions"]) == 5
            assert len(json_data["blocked_domains"]) == 2
            client.close()


# =========================================================================
# Async Tests
# =========================================================================


class TestAsyncComputerUseTasks:
    """Tests for async computer use task operations."""

    @pytest.mark.asyncio
    async def test_async_create_task(self) -> None:
        """Create a task asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "task_id": "cu_task_async",
                "status": "pending",
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.computer_use.create_task(
                    goal="Open browser and navigate to homepage"
                )

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/v1/computer-use/tasks",
                    json={
                        "goal": "Open browser and navigate to homepage",
                        "max_steps": 10,
                        "dry_run": False,
                    },
                )
                assert result["task_id"] == "cu_task_async"

    @pytest.mark.asyncio
    async def test_async_create_task_with_options(self) -> None:
        """Create a task asynchronously with all options."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"task_id": "cu_task_opts"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.computer_use.create_task(
                    goal="Complex automation task",
                    max_steps=50,
                    dry_run=True,
                )

                call_args = mock_request.call_args
                json_data = call_args[1]["json"]
                assert json_data["goal"] == "Complex automation task"
                assert json_data["max_steps"] == 50
                assert json_data["dry_run"] is True
                assert result["task_id"] == "cu_task_opts"

    @pytest.mark.asyncio
    async def test_async_list_tasks(self) -> None:
        """List tasks asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "tasks": [{"task_id": "cu_task_1"}],
                "total": 1,
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.computer_use.list_tasks()

                mock_request.assert_called_once_with(
                    "GET", "/api/v1/computer-use/tasks", params={"limit": 20}
                )
                assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_async_list_tasks_with_filters(self) -> None:
        """List tasks asynchronously with filters."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"tasks": [], "total": 0}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.computer_use.list_tasks(limit=100, status="cancelled")

                call_args = mock_request.call_args
                params = call_args[1]["params"]
                assert params["limit"] == 100
                assert params["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_async_get_task(self) -> None:
        """Get task details asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "task": {
                    "task_id": "cu_task_async",
                    "status": "completed",
                    "result": "Success",
                }
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.computer_use.get_task("cu_task_async")

                mock_request.assert_called_once_with(
                    "GET", "/api/v1/computer-use/tasks/cu_task_async"
                )
                assert result["task"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_async_cancel_task(self) -> None:
        """Cancel a task asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"message": "Task cancelled"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.computer_use.cancel_task("cu_task_to_cancel")

                mock_request.assert_called_once_with(
                    "POST", "/api/v1/computer-use/tasks/cu_task_to_cancel/cancel"
                )
                assert "cancelled" in result["message"].lower()


class TestAsyncComputerUseActions:
    """Tests for async computer use action operations."""

    @pytest.mark.asyncio
    async def test_async_get_action_stats(self) -> None:
        """Get action statistics asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "stats": {
                    "screenshot": {"count": 100},
                    "click": {"count": 50},
                },
                "total_actions": 150,
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.computer_use.get_action_stats()

                mock_request.assert_called_once_with("GET", "/api/v1/computer-use/actions/stats")
                assert result["total_actions"] == 150


class TestAsyncComputerUsePolicies:
    """Tests for async computer use policy operations."""

    @pytest.mark.asyncio
    async def test_async_list_policies(self) -> None:
        """List policies asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "policies": [{"policy_id": "cu_pol_1", "name": "Default"}],
                "total": 1,
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.computer_use.list_policies()

                mock_request.assert_called_once_with("GET", "/api/v1/computer-use/policies")
                assert len(result["policies"]) == 1

    @pytest.mark.asyncio
    async def test_async_create_policy_minimal(self) -> None:
        """Create a policy asynchronously with minimal fields."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"policy_id": "cu_pol_async"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.computer_use.create_policy(name="Async Policy")

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/v1/computer-use/policies",
                    json={"name": "Async Policy"},
                )
                assert result["policy_id"] == "cu_pol_async"

    @pytest.mark.asyncio
    async def test_async_create_policy_full(self) -> None:
        """Create a policy asynchronously with all options."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"policy_id": "cu_pol_full_async"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.computer_use.create_policy(
                    name="Full Async Policy",
                    description="Comprehensive async policy",
                    allowed_actions=["screenshot", "click", "type"],
                    blocked_domains=["restricted.com"],
                )

                call_args = mock_request.call_args
                json_data = call_args[1]["json"]
                assert json_data["name"] == "Full Async Policy"
                assert json_data["description"] == "Comprehensive async policy"
                assert json_data["allowed_actions"] == ["screenshot", "click", "type"]
                assert json_data["blocked_domains"] == ["restricted.com"]
                assert result["policy_id"] == "cu_pol_full_async"
