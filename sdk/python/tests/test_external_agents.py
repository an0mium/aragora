"""Tests for External Agents SDK namespace."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock client."""
    return MagicMock()


class TestExternalAgentsAPI:
    """Test synchronous ExternalAgentsAPI."""

    def test_init(self, mock_client: MagicMock) -> None:
        """Test API initialization."""
        from aragora_sdk.namespaces.external_agents import ExternalAgentsAPI

        api = ExternalAgentsAPI(mock_client)
        assert api._client is mock_client

    def test_list_adapters(self, mock_client: MagicMock) -> None:
        """Test list_adapters calls correct endpoint."""
        from aragora_sdk.namespaces.external_agents import ExternalAgentsAPI

        mock_client.request.return_value = {"adapters": [], "total": 0}

        api = ExternalAgentsAPI(mock_client)
        result = api.list_adapters()

        mock_client.request.assert_called_once_with("GET", "/api/v1/external-agents/adapters")
        assert result == {"adapters": [], "total": 0}

    def test_get_adapter_health(self, mock_client: MagicMock) -> None:
        """Test get_adapter_health calls correct endpoint."""
        from aragora_sdk.namespaces.external_agents import ExternalAgentsAPI

        mock_client.request.return_value = {"health": [], "total": 0}

        api = ExternalAgentsAPI(mock_client)
        result = api.get_adapter_health()

        mock_client.request.assert_called_once_with(
            "GET", "/api/v1/external-agents/health", params={}
        )
        assert result == {"health": [], "total": 0}

    def test_get_adapter_health_with_name(self, mock_client: MagicMock) -> None:
        """Test get_adapter_health with adapter name filter."""
        from aragora_sdk.namespaces.external_agents import ExternalAgentsAPI

        mock_client.request.return_value = {"health": [], "total": 0}

        api = ExternalAgentsAPI(mock_client)
        api.get_adapter_health(adapter_name="openhands")

        mock_client.request.assert_called_once_with(
            "GET", "/api/v1/external-agents/health", params={"adapter": "openhands"}
        )

    def test_submit_task(self, mock_client: MagicMock) -> None:
        """Test submit_task calls correct endpoint."""
        from aragora_sdk.namespaces.external_agents import ExternalAgentsAPI

        mock_client.request.return_value = {
            "task_id": "task-123",
            "status": "pending",
            "adapter": "openhands",
        }

        api = ExternalAgentsAPI(mock_client)
        result = api.submit_task(task_type="code_review", prompt="Review this code")

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/external-agents/tasks")
        assert call_args[1]["json"]["task_type"] == "code_review"
        assert call_args[1]["json"]["prompt"] == "Review this code"
        assert result["task_id"] == "task-123"

    def test_submit_task_with_options(self, mock_client: MagicMock) -> None:
        """Test submit_task with all options."""
        from aragora_sdk.namespaces.external_agents import ExternalAgentsAPI

        mock_client.request.return_value = {"task_id": "task-123"}

        api = ExternalAgentsAPI(mock_client)
        api.submit_task(
            task_type="analysis",
            prompt="Analyze security",
            adapter="openhands",
            tool_permissions=["shell:read", "fs:read"],
            timeout_seconds=300.0,
            max_steps=50,
            context={"repo": "test"},
            workspace_id="ws-123",
            metadata={"priority": "high"},
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["adapter"] == "openhands"
        assert json_body["tool_permissions"] == ["shell:read", "fs:read"]
        assert json_body["timeout_seconds"] == 300.0
        assert json_body["max_steps"] == 50
        assert json_body["context"] == {"repo": "test"}
        assert json_body["workspace_id"] == "ws-123"
        assert json_body["metadata"] == {"priority": "high"}

    def test_get_task(self, mock_client: MagicMock) -> None:
        """Test get_task calls correct endpoint."""
        from aragora_sdk.namespaces.external_agents import ExternalAgentsAPI

        mock_client.request.return_value = {
            "task_id": "task-123",
            "status": "completed",
        }

        api = ExternalAgentsAPI(mock_client)
        result = api.get_task("task-123")

        mock_client.request.assert_called_once_with("GET", "/api/v1/external-agents/tasks/task-123")
        assert result["status"] == "completed"

    def test_cancel_task(self, mock_client: MagicMock) -> None:
        """Test cancel_task calls correct endpoint."""
        from aragora_sdk.namespaces.external_agents import ExternalAgentsAPI

        mock_client.request.return_value = {"task_id": "task-123", "cancelled": True}

        api = ExternalAgentsAPI(mock_client)
        result = api.cancel_task("task-123")

        mock_client.request.assert_called_once_with(
            "DELETE", "/api/v1/external-agents/tasks/task-123"
        )
        assert result["cancelled"] is True


@pytest.fixture
def mock_async_client() -> MagicMock:
    """Create a mock async client."""
    from unittest.mock import AsyncMock

    client = MagicMock()
    client.request = AsyncMock()
    return client


class TestAsyncExternalAgentsAPI:
    """Test asynchronous AsyncExternalAgentsAPI."""

    @pytest.mark.asyncio
    async def test_list_adapters(self, mock_async_client: MagicMock) -> None:
        """Test list_adapters calls correct endpoint."""
        from aragora_sdk.namespaces.external_agents import AsyncExternalAgentsAPI

        mock_async_client.request.return_value = {"adapters": [], "total": 0}

        api = AsyncExternalAgentsAPI(mock_async_client)
        result = await api.list_adapters()

        mock_async_client.request.assert_called_once_with("GET", "/api/v1/external-agents/adapters")
        assert result == {"adapters": [], "total": 0}

    @pytest.mark.asyncio
    async def test_submit_task(self, mock_async_client: MagicMock) -> None:
        """Test submit_task calls correct endpoint."""
        from aragora_sdk.namespaces.external_agents import AsyncExternalAgentsAPI

        mock_async_client.request.return_value = {"task_id": "task-123"}

        api = AsyncExternalAgentsAPI(mock_async_client)
        result = await api.submit_task(task_type="test", prompt="Hello")

        mock_async_client.request.assert_called_once()
        assert result["task_id"] == "task-123"

    @pytest.mark.asyncio
    async def test_get_task(self, mock_async_client: MagicMock) -> None:
        """Test get_task calls correct endpoint."""
        from aragora_sdk.namespaces.external_agents import AsyncExternalAgentsAPI

        mock_async_client.request.return_value = {"task_id": "task-123", "status": "running"}

        api = AsyncExternalAgentsAPI(mock_async_client)
        result = await api.get_task("task-123")

        mock_async_client.request.assert_called_once_with(
            "GET", "/api/v1/external-agents/tasks/task-123"
        )
        assert result["status"] == "running"

    @pytest.mark.asyncio
    async def test_cancel_task(self, mock_async_client: MagicMock) -> None:
        """Test cancel_task calls correct endpoint."""
        from aragora_sdk.namespaces.external_agents import AsyncExternalAgentsAPI

        mock_async_client.request.return_value = {"cancelled": True}

        api = AsyncExternalAgentsAPI(mock_async_client)
        result = await api.cancel_task("task-123")

        mock_async_client.request.assert_called_once_with(
            "DELETE", "/api/v1/external-agents/tasks/task-123"
        )
        assert result["cancelled"] is True

    @pytest.mark.asyncio
    async def test_get_adapter_health(self, mock_async_client: MagicMock) -> None:
        """Test get_adapter_health calls correct endpoint."""
        from aragora_sdk.namespaces.external_agents import AsyncExternalAgentsAPI

        mock_async_client.request.return_value = {"health": []}

        api = AsyncExternalAgentsAPI(mock_async_client)
        await api.get_adapter_health(adapter_name="openhands")

        mock_async_client.request.assert_called_once_with(
            "GET", "/api/v1/external-agents/health", params={"adapter": "openhands"}
        )
