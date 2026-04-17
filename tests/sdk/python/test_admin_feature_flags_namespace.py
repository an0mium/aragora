"""Tests for Python SDK admin feature-flag helpers."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora_sdk.namespaces.admin import AdminAPI, AsyncAdminAPI


def test_list_feature_flags_uses_collection_route() -> None:
    mock_client = MagicMock()
    mock_client.request.return_value = {"flags": []}

    api = AdminAPI(mock_client)
    result = api.list_feature_flags()

    mock_client.request.assert_called_once_with("GET", "/api/v1/admin/feature-flags")
    assert result == {"flags": []}


def test_update_feature_flags_uses_detail_route() -> None:
    mock_client = MagicMock()
    mock_client.request.side_effect = [
        {"name": "enable_checkpointing", "updated": True},
        {"name": "max_agent_retries", "updated": True},
    ]

    api = AdminAPI(mock_client)
    result = api.update_feature_flags({"enable_checkpointing": True, "max_agent_retries": 7})

    assert mock_client.request.call_args_list == [
        (
            ("PUT", "/api/v1/admin/feature-flags/enable_checkpointing"),
            {"json": {"value": True}},
        ),
        (
            ("PUT", "/api/v1/admin/feature-flags/max_agent_retries"),
            {"json": {"value": 7}},
        ),
    ]
    assert result == {
        "enable_checkpointing": {"name": "enable_checkpointing", "updated": True},
        "max_agent_retries": {"name": "max_agent_retries", "updated": True},
    }


@pytest.mark.asyncio
async def test_async_list_feature_flags_uses_collection_route() -> None:
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value={"flags": []})

    api = AsyncAdminAPI(mock_client)
    result = await api.list_feature_flags()

    mock_client.request.assert_awaited_once_with("GET", "/api/v1/admin/feature-flags")
    assert result == {"flags": []}


@pytest.mark.asyncio
async def test_async_update_feature_flags_uses_detail_route() -> None:
    mock_client = MagicMock()
    mock_client.request = AsyncMock(
        side_effect=[
            {"name": "enable_checkpointing", "updated": True},
            {"name": "max_agent_retries", "updated": True},
        ]
    )

    api = AsyncAdminAPI(mock_client)
    result = await api.update_feature_flags({"enable_checkpointing": True, "max_agent_retries": 7})

    assert mock_client.request.await_args_list == [
        (
            ("PUT", "/api/v1/admin/feature-flags/enable_checkpointing"),
            {"json": {"value": True}},
        ),
        (
            ("PUT", "/api/v1/admin/feature-flags/max_agent_retries"),
            {"json": {"value": 7}},
        ),
    ]
    assert result == {
        "enable_checkpointing": {"name": "enable_checkpointing", "updated": True},
        "max_agent_retries": {"name": "max_agent_retries", "updated": True},
    }
