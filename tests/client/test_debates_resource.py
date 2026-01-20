"""Tests for the extracted DebatesAPI resource."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from aragora.client.resources.debates import DebatesAPI
from aragora.client.models import (
    Debate,
    DebateStatus,
    DebateCreateResponse,
)


@pytest.fixture
def mock_client():
    """Create a mock AragoraClient."""
    client = MagicMock()
    client._get = MagicMock()
    client._post = MagicMock()
    client._get_async = AsyncMock()
    client._post_async = AsyncMock()
    return client


@pytest.fixture
def debates_api(mock_client):
    """Create a DebatesAPI instance with mock client."""
    return DebatesAPI(mock_client)


class TestDebatesAPICreate:
    """Tests for DebatesAPI.create()."""

    def test_create_with_defaults(self, debates_api, mock_client):
        """Test creating a debate with default parameters."""
        mock_client._post.return_value = {
            "debate_id": "test-123",
            "status": "pending",
        }

        response = debates_api.create(task="Should we use microservices?")

        assert response.debate_id == "test-123"
        mock_client._post.assert_called_once()
        call_args = mock_client._post.call_args
        assert call_args[0][0] == "/api/debates"

    def test_create_with_custom_agents(self, debates_api, mock_client):
        """Test creating a debate with custom agents."""
        mock_client._post.return_value = {
            "debate_id": "test-456",
            "status": "pending",
        }

        response = debates_api.create(
            task="Test task",
            agents=["claude", "gpt-4"],
            rounds=5,
            consensus="unanimous",
        )

        assert response.debate_id == "test-456"
        call_data = mock_client._post.call_args[0][1]
        assert call_data["agents"] == ["claude", "gpt-4"]
        assert call_data["rounds"] == 5
        assert call_data["consensus"] == "unanimous"

    @pytest.mark.asyncio
    async def test_create_async(self, debates_api, mock_client):
        """Test async debate creation."""
        mock_client._post_async.return_value = {
            "debate_id": "async-123",
            "status": "pending",
        }

        response = await debates_api.create_async(task="Async test")

        assert response.debate_id == "async-123"
        mock_client._post_async.assert_called_once()


class TestDebatesAPIGet:
    """Tests for DebatesAPI.get()."""

    def test_get_debate(self, debates_api, mock_client):
        """Test getting a debate by ID."""
        mock_client._get.return_value = {
            "debate_id": "test-123",
            "task": "Test question",
            "status": "completed",
            "rounds": [],
        }

        debate = debates_api.get("test-123")

        assert debate.debate_id == "test-123"
        mock_client._get.assert_called_with("/api/debates/test-123")

    @pytest.mark.asyncio
    async def test_get_async(self, debates_api, mock_client):
        """Test async debate get."""
        mock_client._get_async.return_value = {
            "debate_id": "async-123",
            "task": "Async question",
            "status": "running",
            "rounds": [],
        }

        debate = await debates_api.get_async("async-123")

        assert debate.debate_id == "async-123"


class TestDebatesAPIList:
    """Tests for DebatesAPI.list()."""

    def test_list_debates(self, debates_api, mock_client):
        """Test listing debates."""
        mock_client._get.return_value = {
            "debates": [
                {"debate_id": "1", "task": "Q1", "status": "completed", "rounds": []},
                {"debate_id": "2", "task": "Q2", "status": "running", "rounds": []},
            ]
        }

        debates = debates_api.list(limit=10)

        assert len(debates) == 2
        assert debates[0].debate_id == "1"

    def test_list_with_status_filter(self, debates_api, mock_client):
        """Test listing debates with status filter."""
        mock_client._get.return_value = []

        debates_api.list(status="completed")

        call_params = mock_client._get.call_args[1]["params"]
        assert call_params["status"] == "completed"

    @pytest.mark.asyncio
    async def test_list_async(self, debates_api, mock_client):
        """Test async debate listing."""
        mock_client._get_async.return_value = []

        debates = await debates_api.list_async()

        assert debates == []


class TestDebatesAPIWait:
    """Tests for wait_for_completion methods."""

    def test_wait_for_completion_immediate(self, debates_api, mock_client):
        """Test waiting for a debate that's already completed."""
        mock_client._get.return_value = {
            "debate_id": "test-123",
            "task": "Test",
            "status": "completed",
            "rounds": [],
        }

        debate = debates_api.wait_for_completion("test-123", timeout=10)

        assert debate.status == DebateStatus.COMPLETED

    def test_wait_for_completion_failed(self, debates_api, mock_client):
        """Test waiting for a debate that failed."""
        mock_client._get.return_value = {
            "debate_id": "test-123",
            "task": "Test",
            "status": "failed",
            "rounds": [],
        }

        debate = debates_api.wait_for_completion("test-123", timeout=10)

        assert debate.status == DebateStatus.FAILED


class TestDebatesAPICompare:
    """Tests for compare methods."""

    def test_compare_debates(self, debates_api, mock_client):
        """Test comparing multiple debates."""
        mock_client._get.side_effect = [
            {"debate_id": "1", "task": "Q1", "status": "completed", "rounds": []},
            {"debate_id": "2", "task": "Q2", "status": "completed", "rounds": []},
        ]

        debates = debates_api.compare(["1", "2"])

        assert len(debates) == 2
        assert mock_client._get.call_count == 2

    @pytest.mark.asyncio
    async def test_compare_async(self, debates_api, mock_client):
        """Test async comparison."""
        mock_client._get_async.side_effect = [
            {"debate_id": "1", "task": "Q1", "status": "completed", "rounds": []},
            {"debate_id": "2", "task": "Q2", "status": "completed", "rounds": []},
        ]

        debates = await debates_api.compare_async(["1", "2"])

        assert len(debates) == 2


class TestDebatesAPIBatchGet:
    """Tests for batch_get methods."""

    def test_batch_get(self, debates_api, mock_client):
        """Test batch fetching debates."""
        mock_client._get.side_effect = [
            {"debate_id": str(i), "task": f"Q{i}", "status": "completed", "rounds": []}
            for i in range(3)
        ]

        debates = debates_api.batch_get(["0", "1", "2"])

        assert len(debates) == 3
        assert mock_client._get.call_count == 3

    @pytest.mark.asyncio
    async def test_batch_get_async(self, debates_api, mock_client):
        """Test async batch fetching."""
        mock_client._get_async.side_effect = [
            {"debate_id": str(i), "task": f"Q{i}", "status": "completed", "rounds": []}
            for i in range(3)
        ]

        debates = await debates_api.batch_get_async(["0", "1", "2"], max_concurrent=2)

        assert len(debates) == 3


class TestDebatesAPIIterate:
    """Tests for iterate methods."""

    def test_iterate_pagination(self, debates_api, mock_client):
        """Test iteration with pagination."""
        mock_client._get.side_effect = [
            {"debates": [
                {"debate_id": "1", "task": "Q1", "status": "completed", "rounds": []},
                {"debate_id": "2", "task": "Q2", "status": "completed", "rounds": []},
            ]},
            {"debates": []},  # End of pagination
        ]

        debates = list(debates_api.iterate(page_size=2))

        assert len(debates) == 2

    def test_iterate_with_max_items(self, debates_api, mock_client):
        """Test iteration with max items limit."""
        mock_client._get.return_value = {
            "debates": [
                {"debate_id": str(i), "task": f"Q{i}", "status": "completed", "rounds": []}
                for i in range(10)
            ]
        }

        debates = list(debates_api.iterate(max_items=3))

        assert len(debates) == 3

    @pytest.mark.asyncio
    async def test_iterate_async(self, debates_api, mock_client):
        """Test async iteration."""
        mock_client._get_async.side_effect = [
            {"debates": [
                {"debate_id": "1", "task": "Q1", "status": "completed", "rounds": []},
            ]},
            {"debates": []},
        ]

        debates = []
        async for debate in debates_api.iterate_async():
            debates.append(debate)

        assert len(debates) == 1
