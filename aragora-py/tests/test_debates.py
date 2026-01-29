"""Tests for Aragora SDK Debates API.

Comprehensive tests covering:
- DebatesAPI CRUD operations
- GraphDebatesAPI operations
- MatrixDebatesAPI operations
- Batch operations
- Export functionality
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora_client.client import (
    AragoraClient,
    DebatesAPI,
    GraphDebatesAPI,
    MatrixDebatesAPI,
)
from aragora_client.exceptions import AragoraTimeoutError
from aragora_client.types import (
    Debate,
    GraphBranch,
    GraphDebate,
    MatrixConclusion,
    MatrixDebate,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock AragoraClient."""
    client = MagicMock(spec=AragoraClient)
    client._get = AsyncMock()
    client._post = AsyncMock()
    client._patch = AsyncMock()
    client._delete = AsyncMock()
    client._get_raw = AsyncMock()
    return client


@pytest.fixture
def debates_api(mock_client: MagicMock) -> DebatesAPI:
    """Create DebatesAPI with mock client."""
    return DebatesAPI(mock_client)


@pytest.fixture
def graph_debates_api(mock_client: MagicMock) -> GraphDebatesAPI:
    """Create GraphDebatesAPI with mock client."""
    return GraphDebatesAPI(mock_client)


@pytest.fixture
def matrix_debates_api(mock_client: MagicMock) -> MatrixDebatesAPI:
    """Create MatrixDebatesAPI with mock client."""
    return MatrixDebatesAPI(mock_client)


@pytest.fixture
def debate_response() -> dict[str, Any]:
    """Standard debate response."""
    return {
        "id": "debate-123",
        "task": "Should we adopt microservices?",
        "status": "completed",
        "agents": ["claude", "gpt4", "gemini"],
        "rounds": [
            [
                {
                    "agent_id": "claude",
                    "content": "I propose we adopt microservices.",
                    "round_number": 1,
                    "timestamp": "2026-01-01T00:00:00Z",
                    "metadata": {},
                }
            ]
        ],
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T12:00:00Z",
        "consensus": {
            "reached": True,
            "confidence": 0.85,
            "conclusion": "Microservices are recommended for this scale.",
        },
        "metadata": {"requested_by": "team-lead"},
    }


@pytest.fixture
def graph_debate_response() -> dict[str, Any]:
    """Standard graph debate response."""
    return {
        "id": "graph-debate-123",
        "task": "Explore alternatives to SQL",
        "status": "completed",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T12:00:00Z",
        "branches": [
            {
                "id": "branch-1",
                "parent_id": None,
                "approach": "NoSQL options",
                "agents": ["claude", "gpt4"],
            },
            {
                "id": "branch-2",
                "parent_id": None,
                "approach": "Graph databases",
                "agents": ["claude", "gpt4"],
            },
        ],
        "metadata": {},
    }


@pytest.fixture
def matrix_debate_response() -> dict[str, Any]:
    """Standard matrix debate response."""
    return {
        "id": "matrix-debate-123",
        "task": "Best cloud provider",
        "status": "completed",
        "scenarios": [
            {"name": "cost", "parameters": {"weight": 0.4}, "is_baseline": False},
            {
                "name": "performance",
                "parameters": {"weight": 0.3},
                "is_baseline": False,
            },
            {
                "name": "scalability",
                "parameters": {"weight": 0.3},
                "is_baseline": True,
            },
        ],
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T12:00:00Z",
        "metadata": {},
    }


@pytest.fixture
def branch_response() -> dict[str, Any]:
    """Standard branch response."""
    return {
        "id": "branch-123",
        "parent_id": None,
        "approach": "NoSQL options",
        "agents": ["claude", "gpt4"],
    }


@pytest.fixture
def conclusion_response() -> dict[str, Any]:
    """Standard matrix conclusion response."""
    return {
        "universal": ["All providers support auto-scaling"],
        "conditional": {
            "cost-sensitive": ["GCP is recommended for cost optimization"],
            "performance-critical": ["AWS offers best compute performance"],
        },
        "contradictions": [],
    }


# =============================================================================
# DebatesAPI Tests
# =============================================================================


class TestDebatesAPICreate:
    """Tests for DebatesAPI.create()."""

    @pytest.mark.asyncio
    async def test_create_basic(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test creating a basic debate."""
        mock_client._post.return_value = {"id": "debate-123", "task": "Test debate"}

        result = await debates_api.create("Test debate")

        mock_client._post.assert_called_once()
        call_args = mock_client._post.call_args
        assert call_args[0][0] == "/api/v1/debates"
        assert call_args[0][1]["task"] == "Test debate"
        assert result["id"] == "debate-123"

    @pytest.mark.asyncio
    async def test_create_with_agents(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test creating a debate with specific agents."""
        mock_client._post.return_value = {"id": "debate-123"}

        await debates_api.create("Test", agents=["claude", "gpt4"])

        call_args = mock_client._post.call_args[0][1]
        assert call_args["agents"] == ["claude", "gpt4"]

    @pytest.mark.asyncio
    async def test_create_with_options(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test creating a debate with all options."""
        mock_client._post.return_value = {"id": "debate-123"}

        await debates_api.create(
            "Test debate",
            agents=["claude"],
            max_rounds=10,
            consensus_threshold=0.9,
            metadata={"priority": "high"},
        )

        call_args = mock_client._post.call_args[0][1]
        assert call_args["max_rounds"] == 10
        assert call_args["consensus_threshold"] == 0.9
        assert call_args["metadata"]["priority"] == "high"


class TestDebatesAPIGet:
    """Tests for DebatesAPI.get()."""

    @pytest.mark.asyncio
    async def test_get_debate(
        self,
        debates_api: DebatesAPI,
        mock_client: MagicMock,
        debate_response: dict[str, Any],
    ) -> None:
        """Test getting a debate by ID."""
        mock_client._get.return_value = debate_response

        result = await debates_api.get("debate-123")

        mock_client._get.assert_called_once_with("/api/v1/debates/debate-123")
        assert isinstance(result, Debate)
        assert result.id == "debate-123"
        assert result.task == "Should we adopt microservices?"


class TestDebatesAPIList:
    """Tests for DebatesAPI.list()."""

    @pytest.mark.asyncio
    async def test_list_debates(
        self,
        debates_api: DebatesAPI,
        mock_client: MagicMock,
        debate_response: dict[str, Any],
    ) -> None:
        """Test listing debates."""
        mock_client._get.return_value = {"debates": [debate_response]}

        result = await debates_api.list()

        mock_client._get.assert_called_once()
        assert len(result) == 1
        assert isinstance(result[0], Debate)

    @pytest.mark.asyncio
    async def test_list_with_pagination(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test listing debates with pagination."""
        mock_client._get.return_value = {"debates": []}

        await debates_api.list(limit=20, offset=40)

        call_args = mock_client._get.call_args
        assert call_args[1]["params"]["limit"] == 20
        assert call_args[1]["params"]["offset"] == 40

    @pytest.mark.asyncio
    async def test_list_with_status_filter(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test listing debates with status filter."""
        mock_client._get.return_value = {"debates": []}

        await debates_api.list(status="completed")

        call_args = mock_client._get.call_args
        assert call_args[1]["params"]["status"] == "completed"


class TestDebatesAPIRun:
    """Tests for DebatesAPI.run() - synchronous debate execution."""

    @pytest.mark.asyncio
    async def test_run_completes(
        self,
        debates_api: DebatesAPI,
        mock_client: MagicMock,
        debate_response: dict[str, Any],
    ) -> None:
        """Test running a debate that completes successfully."""
        mock_client._post.return_value = {"id": "debate-123"}
        mock_client._get.return_value = debate_response

        result = await debates_api.run("Test debate", poll_interval=0.01)

        assert isinstance(result, Debate)
        assert result.status.value == "completed"

    @pytest.mark.asyncio
    async def test_run_timeout(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test run timeout when debate doesn't complete."""
        mock_client._post.return_value = {"id": "debate-123"}
        mock_client._get.return_value = {
            "id": "debate-123",
            "task": "Test",
            "status": "running",
            "agents": [],
            "rounds": [],
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }

        with pytest.raises(AragoraTimeoutError):
            await debates_api.run("Test", poll_interval=0.01, timeout=0.05)


class TestDebatesAPIUpdate:
    """Tests for DebatesAPI.update()."""

    @pytest.mark.asyncio
    async def test_update_title(
        self,
        debates_api: DebatesAPI,
        mock_client: MagicMock,
        debate_response: dict[str, Any],
    ) -> None:
        """Test updating debate title."""
        mock_client._patch.return_value = debate_response

        await debates_api.update("debate-123", title="New Title")

        mock_client._patch.assert_called_once()
        call_args = mock_client._patch.call_args
        assert call_args[0][1]["title"] == "New Title"

    @pytest.mark.asyncio
    async def test_update_status(
        self,
        debates_api: DebatesAPI,
        mock_client: MagicMock,
        debate_response: dict[str, Any],
    ) -> None:
        """Test updating debate status."""
        mock_client._patch.return_value = debate_response

        await debates_api.update("debate-123", status="archived")

        call_args = mock_client._patch.call_args[0][1]
        assert call_args["status"] == "archived"


class TestDebatesAPICancel:
    """Tests for DebatesAPI.cancel()."""

    @pytest.mark.asyncio
    async def test_cancel_debate(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test cancelling a debate."""
        mock_client._post.return_value = {"cancelled": True}

        result = await debates_api.cancel("debate-123")

        mock_client._post.assert_called_once_with(
            "/api/v1/debates/debate-123/cancel", {}
        )
        assert result["cancelled"] is True


class TestDebatesAPIMessages:
    """Tests for DebatesAPI message operations."""

    @pytest.mark.asyncio
    async def test_get_messages(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test getting debate messages."""
        mock_client._get.return_value = {
            "messages": [{"id": "msg-1", "content": "Hello"}]
        }

        result = await debates_api.get_messages("debate-123")

        mock_client._get.assert_called_once()
        assert len(result) == 1
        assert result[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_get_messages_pagination(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test getting messages with pagination."""
        mock_client._get.return_value = {"messages": []}

        await debates_api.get_messages("debate-123", limit=10, offset=20)

        call_args = mock_client._get.call_args
        assert call_args[1]["params"]["limit"] == 10
        assert call_args[1]["params"]["offset"] == 20


class TestDebatesAPIAnalysis:
    """Tests for DebatesAPI analysis endpoints."""

    @pytest.mark.asyncio
    async def test_get_convergence(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test getting convergence status."""
        mock_client._get.return_value = {"converging": True, "rate": 0.85}

        result = await debates_api.get_convergence("debate-123")

        mock_client._get.assert_called_once_with(
            "/api/v1/debates/debate-123/convergence"
        )
        assert result["converging"] is True

    @pytest.mark.asyncio
    async def test_get_citations(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test getting debate citations."""
        mock_client._get.return_value = {"citations": [{"source": "paper.pdf"}]}

        result = await debates_api.get_citations("debate-123")

        assert len(result) == 1
        assert result[0]["source"] == "paper.pdf"

    @pytest.mark.asyncio
    async def test_get_evidence(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test getting evidence trail."""
        mock_client._get.return_value = {"chain": [], "verified": True}

        result = await debates_api.get_evidence("debate-123")

        assert result["verified"] is True

    @pytest.mark.asyncio
    async def test_get_impasse(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test detecting impasse."""
        mock_client._get.return_value = {"impasse_detected": False}

        result = await debates_api.get_impasse("debate-123")

        assert result["impasse_detected"] is False

    @pytest.mark.asyncio
    async def test_get_summary(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test getting debate summary."""
        mock_client._get.return_value = {"summary": "The debate concluded..."}

        result = await debates_api.get_summary("debate-123")

        assert "summary" in result

    @pytest.mark.asyncio
    async def test_get_rhetorical(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test getting rhetorical patterns."""
        mock_client._get.return_value = {"patterns": ["appeal_to_authority"]}

        result = await debates_api.get_rhetorical("debate-123")

        assert "patterns" in result

    @pytest.mark.asyncio
    async def test_get_trickster(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test getting trickster status."""
        mock_client._get.return_value = {"hollow_consensus": False}

        result = await debates_api.get_trickster("debate-123")

        assert result["hollow_consensus"] is False


class TestDebatesAPIForks:
    """Tests for DebatesAPI fork operations."""

    @pytest.mark.asyncio
    async def test_fork_debate(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test forking a debate."""
        mock_client._post.return_value = {"id": "fork-123", "parent_id": "debate-123"}

        await debates_api.fork("debate-123", branch_point=3)

        mock_client._post.assert_called_once()
        call_args = mock_client._post.call_args[0][1]
        assert call_args["branch_point"] == 3

    @pytest.mark.asyncio
    async def test_fork_with_context(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test forking with modified context."""
        mock_client._post.return_value = {"id": "fork-123"}

        await debates_api.fork(
            "debate-123", branch_point=3, modified_context="Alternative premise"
        )

        call_args = mock_client._post.call_args[0][1]
        assert call_args["modified_context"] == "Alternative premise"

    @pytest.mark.asyncio
    async def test_list_forks(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test listing debate forks."""
        mock_client._get.return_value = {"forks": [{"id": "fork-1"}]}

        result = await debates_api.list_forks("debate-123")

        assert len(result) == 1


class TestDebatesAPIFollowup:
    """Tests for DebatesAPI follow-up operations."""

    @pytest.mark.asyncio
    async def test_create_followup(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test creating follow-up debate."""
        mock_client._post.return_value = {"id": "followup-123"}

        await debates_api.create_followup(
            "debate-123",
            crux="Key disagreement",
        )

        mock_client._post.assert_called_once()
        call_args = mock_client._post.call_args[0][1]
        assert call_args["crux"] == "Key disagreement"

    @pytest.mark.asyncio
    async def test_get_followup_suggestions(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test getting follow-up suggestions."""
        mock_client._get.return_value = {"suggestions": [{"topic": "Explore X"}]}

        result = await debates_api.get_followup_suggestions("debate-123")

        assert len(result) == 1


class TestDebatesAPIExport:
    """Tests for DebatesAPI export functionality."""

    @pytest.mark.asyncio
    async def test_export_json(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test exporting debate as JSON."""
        mock_client._get_raw.return_value = b'{"debate": "data"}'

        result = await debates_api.export("debate-123", format="json")

        mock_client._get_raw.assert_called_once()
        assert b"debate" in result

    @pytest.mark.asyncio
    async def test_export_csv(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test exporting debate as CSV."""
        mock_client._get_raw.return_value = b"col1,col2\nval1,val2"

        result = await debates_api.export("debate-123", format="csv")

        assert b"col1" in result

    @pytest.mark.asyncio
    async def test_export_with_tables(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test exporting with specific tables."""
        mock_client._get_raw.return_value = b"{}"

        await debates_api.export("debate-123", tables=["messages", "votes"])

        call_args = mock_client._get_raw.call_args
        assert "messages,votes" in call_args[1]["params"]["tables"]


class TestDebatesAPIBatch:
    """Tests for DebatesAPI batch operations."""

    @pytest.mark.asyncio
    async def test_batch_submit(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test batch debate submission."""
        mock_client._post.return_value = {"batch_id": "batch-123", "count": 3}

        items = [
            {"task": "Debate 1"},
            {"task": "Debate 2"},
            {"task": "Debate 3"},
        ]
        result = await debates_api.batch_submit(items)

        assert result["batch_id"] == "batch-123"
        assert result["count"] == 3

    @pytest.mark.asyncio
    async def test_batch_submit_with_webhook(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test batch submission with webhook notification."""
        mock_client._post.return_value = {"batch_id": "batch-123"}

        await debates_api.batch_submit(
            [{"task": "Test"}],
            webhook_url="https://example.com/hook",
        )

        call_args = mock_client._post.call_args[0][1]
        assert call_args["webhook_url"] == "https://example.com/hook"

    @pytest.mark.asyncio
    async def test_get_batch_status(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test getting batch status."""
        mock_client._get.return_value = {
            "status": "running",
            "completed": 2,
            "total": 5,
        }

        result = await debates_api.get_batch_status("batch-123")

        assert result["status"] == "running"
        assert result["completed"] == 2


class TestDebatesAPISearch:
    """Tests for DebatesAPI search functionality."""

    @pytest.mark.asyncio
    async def test_search(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test searching debates."""
        mock_client._get.return_value = {"results": [{"id": "debate-123"}]}

        await debates_api.search("microservices")

        mock_client._get.assert_called_once()
        call_args = mock_client._get.call_args
        assert call_args[1]["params"]["query"] == "microservices"

    @pytest.mark.asyncio
    async def test_search_pagination(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test search with pagination."""
        mock_client._get.return_value = {"results": []}

        await debates_api.search("test", limit=10, offset=20)

        call_args = mock_client._get.call_args
        assert call_args[1]["params"]["limit"] == 10
        assert call_args[1]["params"]["offset"] == 20


class TestDebatesAPIQueue:
    """Tests for DebatesAPI queue operations."""

    @pytest.mark.asyncio
    async def test_get_queue_status(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test getting queue status."""
        mock_client._get.return_value = {"pending": 5, "running": 2, "completed": 100}

        result = await debates_api.get_queue_status()

        mock_client._get.assert_called_once_with("/api/v1/debates/queue/status")
        assert result["pending"] == 5


# =============================================================================
# GraphDebatesAPI Tests
# =============================================================================


class TestGraphDebatesAPI:
    """Tests for GraphDebatesAPI."""

    @pytest.mark.asyncio
    async def test_create(
        self, graph_debates_api: GraphDebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test creating a graph debate."""
        mock_client._post.return_value = {"id": "graph-123"}

        await graph_debates_api.create(
            "Explore alternatives",
            branch_threshold=0.6,
            max_branches=15,
        )

        mock_client._post.assert_called_once()
        call_args = mock_client._post.call_args[0][1]
        assert call_args["task"] == "Explore alternatives"
        assert call_args["branch_threshold"] == 0.6
        assert call_args["max_branches"] == 15

    @pytest.mark.asyncio
    async def test_get(
        self,
        graph_debates_api: GraphDebatesAPI,
        mock_client: MagicMock,
        graph_debate_response: dict[str, Any],
    ) -> None:
        """Test getting a graph debate."""
        mock_client._get.return_value = graph_debate_response

        result = await graph_debates_api.get("graph-123")

        assert isinstance(result, GraphDebate)
        assert result.id == "graph-debate-123"

    @pytest.mark.asyncio
    async def test_get_branches(
        self,
        graph_debates_api: GraphDebatesAPI,
        mock_client: MagicMock,
        branch_response: dict[str, Any],
    ) -> None:
        """Test getting branches."""
        mock_client._get.return_value = {"branches": [branch_response]}

        result = await graph_debates_api.get_branches("graph-123")

        assert len(result) == 1
        assert isinstance(result[0], GraphBranch)


# =============================================================================
# MatrixDebatesAPI Tests
# =============================================================================


class TestMatrixDebatesAPI:
    """Tests for MatrixDebatesAPI."""

    @pytest.mark.asyncio
    async def test_create(
        self, matrix_debates_api: MatrixDebatesAPI, mock_client: MagicMock
    ) -> None:
        """Test creating a matrix debate."""
        scenarios = [
            {"name": "cost", "weight": 0.5},
            {"name": "performance", "weight": 0.5},
        ]
        mock_client._post.return_value = {"id": "matrix-123"}

        await matrix_debates_api.create(
            "Compare options",
            scenarios=scenarios,
            max_rounds=3,
        )

        mock_client._post.assert_called_once()
        call_args = mock_client._post.call_args[0][1]
        assert call_args["scenarios"] == scenarios
        assert call_args["max_rounds"] == 3

    @pytest.mark.asyncio
    async def test_get(
        self,
        matrix_debates_api: MatrixDebatesAPI,
        mock_client: MagicMock,
        matrix_debate_response: dict[str, Any],
    ) -> None:
        """Test getting a matrix debate."""
        mock_client._get.return_value = matrix_debate_response

        result = await matrix_debates_api.get("matrix-123")

        assert isinstance(result, MatrixDebate)
        assert result.id == "matrix-debate-123"

    @pytest.mark.asyncio
    async def test_get_conclusions(
        self,
        matrix_debates_api: MatrixDebatesAPI,
        mock_client: MagicMock,
        conclusion_response: dict[str, Any],
    ) -> None:
        """Test getting matrix conclusions."""
        mock_client._get.return_value = conclusion_response

        result = await matrix_debates_api.get_conclusions("matrix-123")

        assert isinstance(result, MatrixConclusion)
        assert len(result.universal) == 1
        assert result.contradictions == []
