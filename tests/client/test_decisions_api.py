"""Tests for DecisionsAPI client resource."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.client.client import AragoraClient
from aragora.client.resources.decisions import (
    DecisionConfig,
    DecisionContext,
    DecisionResult,
    DecisionStatus,
    DecisionsAPI,
    ResponseChannel,
)


@pytest.fixture
def mock_client() -> AragoraClient:
    client = MagicMock(spec=AragoraClient)
    return client


@pytest.fixture
def api(mock_client: AragoraClient) -> DecisionsAPI:
    return DecisionsAPI(mock_client)


SAMPLE_RESULT = {
    "request_id": "dec-123",
    "status": "completed",
    "decision_type": "debate",
    "content": "Should we use Redis?",
    "result": {"consensus": "yes"},
    "created_at": "2026-01-15T10:00:00Z",
    "completed_at": "2026-01-15T10:05:00Z",
    "metadata": {"source": "api"},
}

SAMPLE_STATUS = {
    "request_id": "dec-123",
    "status": "processing",
    "progress": 0.6,
    "current_stage": "round_2",
    "estimated_remaining_seconds": 30,
}


class TestDecisionsCreate:
    def test_create_minimal(self, api: DecisionsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_RESULT
        result = api.create("Should we use Redis?")
        assert isinstance(result, DecisionResult)
        assert result.request_id == "dec-123"
        assert result.status == "completed"
        mock_client._post.assert_called_once()
        body = mock_client._post.call_args[0][1]
        assert body["content"] == "Should we use Redis?"
        assert body["decision_type"] == "auto"

    def test_create_with_config(self, api: DecisionsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_RESULT
        config = DecisionConfig(agents=["a", "b"], rounds=5, consensus="unanimous")
        api.create("topic", config=config)
        body = mock_client._post.call_args[0][1]
        assert body["config"]["agents"] == ["a", "b"]
        assert body["config"]["rounds"] == 5
        assert body["config"]["consensus"] == "unanimous"

    def test_create_with_context(self, api: DecisionsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_RESULT
        ctx = DecisionContext(user_id="u1", workspace_id="w1", metadata={"key": "val"})
        api.create("topic", context=ctx)
        body = mock_client._post.call_args[0][1]
        assert body["context"]["user_id"] == "u1"
        assert body["context"]["key"] == "val"

    def test_create_with_response_channels(
        self, api: DecisionsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = SAMPLE_RESULT
        channels = [ResponseChannel(platform="slack", target="#general")]
        api.create("topic", response_channels=channels)
        body = mock_client._post.call_args[0][1]
        assert body["response_channels"][0]["platform"] == "slack"
        assert body["response_channels"][0]["target"] == "#general"

    @pytest.mark.asyncio
    async def test_create_async(self, api: DecisionsAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value=SAMPLE_RESULT)
        result = await api.create_async("topic")
        assert result.request_id == "dec-123"


class TestDecisionsGet:
    def test_get(self, api: DecisionsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_RESULT
        result = api.get("dec-123")
        assert result.request_id == "dec-123"
        assert result.decision_type == "debate"
        mock_client._get.assert_called_once_with("/api/v1/decisions/dec-123")

    @pytest.mark.asyncio
    async def test_get_async(self, api: DecisionsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_RESULT)
        result = await api.get_async("dec-123")
        assert result.status == "completed"


class TestDecisionsGetStatus:
    def test_get_status(self, api: DecisionsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_STATUS
        status = api.get_status("dec-123")
        assert isinstance(status, DecisionStatus)
        assert status.progress == 0.6
        assert status.current_stage == "round_2"
        assert status.estimated_remaining_seconds == 30

    @pytest.mark.asyncio
    async def test_get_status_async(self, api: DecisionsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_STATUS)
        status = await api.get_status_async("dec-123")
        assert status.status == "processing"


class TestDecisionsList:
    def test_list_default(self, api: DecisionsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"decisions": [SAMPLE_RESULT], "total": 1}
        results, total = api.list()
        assert len(results) == 1
        assert total == 1
        assert results[0].request_id == "dec-123"

    def test_list_with_filters(self, api: DecisionsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"decisions": [], "total": 0}
        api.list(status="completed", decision_type="debate", limit=10, offset=5)
        params = mock_client._get.call_args[1]["params"]
        assert params["status"] == "completed"
        assert params["decision_type"] == "debate"
        assert params["limit"] == 10
        assert params["offset"] == 5

    @pytest.mark.asyncio
    async def test_list_async(self, api: DecisionsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={"decisions": [SAMPLE_RESULT], "total": 1})
        results, total = await api.list_async()
        assert total == 1


class TestDecisionsConvenience:
    def test_quick_decision(self, api: DecisionsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_RESULT
        result = api.quick_decision("Is Redis better than Memcached?")
        assert result.request_id == "dec-123"
        body = mock_client._post.call_args[0][1]
        assert body["decision_type"] == "quick"
        assert body["config"]["rounds"] == 2
        assert body["config"]["timeout_seconds"] == 60

    def test_start_debate(self, api: DecisionsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = SAMPLE_RESULT
        result = api.start_debate("Redis vs Memcached", agents=["agent1"])
        assert result.request_id == "dec-123"
        body = mock_client._post.call_args[0][1]
        assert body["decision_type"] == "debate"
        assert body["config"]["agents"] == ["agent1"]


class TestParseResult:
    def test_parse_datetime_iso(self, api: DecisionsAPI) -> None:
        result = api._parse_result(SAMPLE_RESULT)
        assert result.created_at is not None
        assert result.created_at.year == 2026

    def test_parse_missing_datetimes(self, api: DecisionsAPI) -> None:
        data = {"request_id": "x", "status": "pending", "content": "q"}
        result = api._parse_result(data)
        assert result.created_at is None
        assert result.completed_at is None

    def test_parse_invalid_datetime(self, api: DecisionsAPI) -> None:
        data = {**SAMPLE_RESULT, "created_at": "not-a-date"}
        result = api._parse_result(data)
        assert result.created_at is None

    def test_parse_falls_back_to_id(self, api: DecisionsAPI) -> None:
        data = {"id": "fallback-id", "status": "ok", "content": "q"}
        result = api._parse_result(data)
        assert result.request_id == "fallback-id"

    def test_parse_status(self, api: DecisionsAPI) -> None:
        status = api._parse_status(SAMPLE_STATUS)
        assert status.request_id == "dec-123"
        assert status.progress == 0.6


class TestDataclasses:
    def test_decision_config_defaults(self) -> None:
        config = DecisionConfig()
        assert isinstance(config.agents, list)
        assert config.rounds > 0
        assert config.timeout_seconds == 300

    def test_decision_context(self) -> None:
        ctx = DecisionContext(user_id="u1")
        assert ctx.user_id == "u1"
        assert ctx.metadata == {}

    def test_response_channel(self) -> None:
        ch = ResponseChannel(platform="email", target="test@example.com")
        assert ch.platform == "email"
        assert ch.options == {}
