"""
Tests for Debates API resource.

Tests cover:
- DebatesAPI.create() and create_async() for starting debates
- DebatesAPI.get() and get_async() for retrieving debates
- DebatesAPI.list() and list_async() for listing debates
- DebatesAPI.run() and run_async() for synchronous execution
- DebatesAPI.wait_for_completion() for polling
- DebatesAPI.compare() for side-by-side comparison
- DebatesAPI.batch_get() for bulk retrieval
- DebatesAPI.iterate() for pagination
- DebatesAPI.update() for metadata updates
- DebatesAPI.get_verification_report() for verification
- DebatesAPI.search() for searching debates
- Model validation for Debate-related models
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.client.client import AragoraClient
from aragora.client.models import (
    AgentMessage,
    ConsensusResult,
    ConsensusType,
    Debate,
    DebateCreateRequest,
    DebateCreateResponse,
    DebateRound,
    DebateStatus,
    DebateUpdateRequest,
    SearchResponse,
    SearchResult,
    VerificationReport,
    VerificationReportClaimDetail,
    Vote,
)
from aragora.client.resources.debates import DebatesAPI


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_client() -> AragoraClient:
    """Create a mock AragoraClient."""
    client = MagicMock(spec=AragoraClient)
    return client


@pytest.fixture
def debates_api(mock_client: AragoraClient) -> DebatesAPI:
    """Create a DebatesAPI with mock client."""
    return DebatesAPI(mock_client)


@pytest.fixture
def sample_timestamp() -> str:
    """Sample ISO timestamp for tests."""
    return datetime.now(timezone.utc).isoformat()


# ============================================================================
# DebateStatus Tests
# ============================================================================


class TestDebateStatus:
    """Tests for DebateStatus enum."""

    def test_canonical_values(self):
        """Test canonical status values."""
        assert DebateStatus.PENDING.value == "pending"
        assert DebateStatus.RUNNING.value == "running"
        assert DebateStatus.COMPLETED.value == "completed"
        assert DebateStatus.FAILED.value == "failed"
        assert DebateStatus.CANCELLED.value == "cancelled"
        assert DebateStatus.PAUSED.value == "paused"

    def test_legacy_values(self):
        """Test legacy status values."""
        assert DebateStatus.CREATED.value == "created"
        assert DebateStatus.IN_PROGRESS.value == "in_progress"
        assert DebateStatus.STARTING.value == "starting"

    def test_missing_maps_legacy(self):
        """Test _missing_ maps legacy server values."""
        assert DebateStatus("active") == DebateStatus.RUNNING
        assert DebateStatus("concluded") == DebateStatus.COMPLETED
        assert DebateStatus("archived") == DebateStatus.COMPLETED


# ============================================================================
# ConsensusType Tests
# ============================================================================


class TestConsensusType:
    """Tests for ConsensusType enum."""

    def test_all_consensus_types(self):
        """Test all consensus type values."""
        assert ConsensusType.UNANIMOUS.value == "unanimous"
        assert ConsensusType.MAJORITY.value == "majority"
        assert ConsensusType.SUPERMAJORITY.value == "supermajority"
        assert ConsensusType.HYBRID.value == "hybrid"
        assert ConsensusType.JUDGE.value == "judge"


# ============================================================================
# AgentMessage Model Tests
# ============================================================================


class TestAgentMessageModel:
    """Tests for AgentMessage model."""

    def test_message_basic(self):
        """Test AgentMessage with basic fields."""
        msg = AgentMessage(agent_id="claude", content="Hello")
        assert msg.agent_id == "claude"
        assert msg.content == "Hello"
        assert msg.round is None

    def test_message_with_alias(self):
        """Test AgentMessage with alias field."""
        msg = AgentMessage(agent="gpt-4", content="Hi", round_number=1)
        assert msg.agent_id == "gpt-4"
        assert msg.round == 1


# ============================================================================
# Vote Model Tests
# ============================================================================


class TestVoteModel:
    """Tests for Vote model."""

    def test_vote_basic(self):
        """Test Vote with basic fields."""
        vote = Vote(agent_id="claude", position="agree", confidence=0.9)
        assert vote.agent_id == "claude"
        assert vote.position == "agree"
        assert vote.confidence == 0.9

    def test_vote_with_reasoning(self):
        """Test Vote with reasoning."""
        vote = Vote(
            agent_id="gpt-4",
            position="disagree",
            confidence=0.7,
            reasoning="Insufficient evidence",
        )
        assert vote.reasoning == "Insufficient evidence"


# ============================================================================
# ConsensusResult Model Tests
# ============================================================================


class TestConsensusResultModel:
    """Tests for ConsensusResult model."""

    def test_consensus_minimal(self):
        """Test ConsensusResult with minimal data."""
        result = ConsensusResult(reached=True)
        assert result.reached is True
        assert result.votes == []

    def test_consensus_full(self):
        """Test ConsensusResult with all fields."""
        result = ConsensusResult(
            reached=True,
            agreement=0.85,
            confidence=0.85,
            final_answer="Yes",
            conclusion="Yes",
            supporting_agents=["claude", "gpt-4"],
            dissenting_agents=["gemini"],
            votes=[Vote(agent_id="claude", position="agree", confidence=0.9)],
        )
        assert result.agreement == 0.85
        assert len(result.supporting_agents) == 2
        assert len(result.votes) == 1

    def test_consensus_sync_fields(self):
        """Test ConsensusResult syncs agreement and confidence."""
        result = ConsensusResult(reached=True, agreement=0.75)
        assert result.confidence == 0.75

    def test_consensus_sync_conclusion(self):
        """Test ConsensusResult syncs final_answer and conclusion."""
        result = ConsensusResult(reached=True, final_answer="Agreed")
        assert result.conclusion == "Agreed"


# ============================================================================
# DebateRound Model Tests
# ============================================================================


class TestDebateRoundModel:
    """Tests for DebateRound model."""

    def test_round_basic(self):
        """Test DebateRound with basic data."""
        round_ = DebateRound(round_number=1)
        assert round_.round_number == 1
        assert round_.messages == []
        assert round_.critiques == []

    def test_round_with_alias(self):
        """Test DebateRound with alias."""
        round_ = DebateRound(round=2)
        assert round_.round_number == 2

    def test_round_with_messages(self):
        """Test DebateRound with messages."""
        round_ = DebateRound(
            round_number=1,
            messages=[
                AgentMessage(agent_id="claude", content="Proposal"),
            ],
            critiques=[
                AgentMessage(agent_id="gpt-4", content="Critique"),
            ],
        )
        assert len(round_.messages) == 1
        assert len(round_.critiques) == 1


# ============================================================================
# Debate Model Tests
# ============================================================================


class TestDebateModel:
    """Tests for Debate model."""

    def test_debate_minimal(self):
        """Test Debate with required fields."""
        debate = Debate(
            debate_id="deb-123",
            task="What is 2+2?",
            status=DebateStatus.PENDING,
        )
        assert debate.debate_id == "deb-123"
        assert debate.task == "What is 2+2?"
        assert debate.status == DebateStatus.PENDING
        assert debate.agents == []
        assert debate.rounds == []

    def test_debate_with_alias(self):
        """Test Debate with id alias."""
        debate = Debate(
            id="deb-alias",
            task="Test",
            status=DebateStatus.RUNNING,
        )
        assert debate.debate_id == "deb-alias"

    def test_debate_coerces_rounds(self):
        """Test Debate coerces int rounds to empty list."""
        debate = Debate(
            debate_id="deb-coerce",
            task="Test",
            status=DebateStatus.COMPLETED,
            rounds=3,  # type: ignore[arg-type] - Server sometimes sends int
        )
        assert debate.rounds == []

    def test_debate_derives_consensus(self):
        """Test Debate derives consensus from consensus_proof."""
        debate = Debate(
            debate_id="deb-proof",
            task="Test",
            status=DebateStatus.COMPLETED,
            consensus_proof={
                "reached": True,
                "confidence": 0.9,
                "final_answer": "Yes",
                "vote_breakdown": {"claude": True, "gpt-4": True, "gemini": False},
            },
        )
        assert debate.consensus is not None
        assert debate.consensus.reached is True
        assert debate.consensus.confidence == 0.9
        assert "claude" in debate.consensus.supporting_agents
        assert "gemini" in debate.consensus.dissenting_agents


# ============================================================================
# DebateCreateRequest/Response Model Tests
# ============================================================================


class TestDebateCreateRequestModel:
    """Tests for DebateCreateRequest model."""

    def test_request_minimal(self):
        """Test DebateCreateRequest with required fields."""
        request = DebateCreateRequest(task="What is the best approach?")
        assert request.task == "What is the best approach?"
        # Default consensus type comes from config
        assert request.consensus in list(ConsensusType)

    def test_request_full(self):
        """Test DebateCreateRequest with all fields."""
        request = DebateCreateRequest(
            task="Debate topic",
            agents=["claude", "gpt-4"],
            rounds=5,
            consensus=ConsensusType.UNANIMOUS,
            context="Additional context",
            metadata={"key": "value"},
        )
        assert request.agents == ["claude", "gpt-4"]
        assert request.rounds == 5
        assert request.consensus == ConsensusType.UNANIMOUS


class TestDebateCreateResponseModel:
    """Tests for DebateCreateResponse model."""

    def test_response_basic(self):
        """Test DebateCreateResponse with required fields."""
        response = DebateCreateResponse(debate_id="deb-new")
        assert response.debate_id == "deb-new"
        assert response.status is None

    def test_response_full(self):
        """Test DebateCreateResponse with all fields."""
        response = DebateCreateResponse(
            debate_id="deb-full",
            status=DebateStatus.PENDING,
            task="The task",
        )
        assert response.status == DebateStatus.PENDING
        assert response.task == "The task"


# ============================================================================
# DebateUpdateRequest Model Tests
# ============================================================================


class TestDebateUpdateRequestModel:
    """Tests for DebateUpdateRequest model."""

    def test_update_request_empty(self):
        """Test DebateUpdateRequest with no fields."""
        request = DebateUpdateRequest()
        assert request.status is None
        assert request.metadata is None

    def test_update_request_full(self):
        """Test DebateUpdateRequest with all fields."""
        request = DebateUpdateRequest(
            status=DebateStatus.PAUSED,
            metadata={"key": "value"},
            tags=["important"],
            archived=True,
            notes="Some notes",
        )
        assert request.status == DebateStatus.PAUSED
        assert request.tags == ["important"]


# ============================================================================
# VerificationReport Model Tests
# ============================================================================


class TestVerificationReportModel:
    """Tests for VerificationReport model."""

    def test_report_minimal(self):
        """Test VerificationReport with minimal data."""
        report = VerificationReport(
            debate_id="deb-123",
            verified=True,
        )
        assert report.debate_id == "deb-123"
        assert report.verified is True
        assert report.claim_details == []

    def test_report_full(self):
        """Test VerificationReport with all fields."""
        report = VerificationReport(
            debate_id="deb-456",
            verified=True,
            verification_method="fact_check",
            claims_verified=5,
            claims_failed=1,
            claims_skipped=2,
            claim_details=[
                VerificationReportClaimDetail(
                    claim="Water boils at 100C",
                    verified=True,
                    confidence=0.95,
                    evidence="Scientific consensus",
                ),
            ],
            overall_confidence=0.9,
            verification_duration_ms=1500,
        )
        assert report.claims_verified == 5
        assert len(report.claim_details) == 1


# ============================================================================
# SearchResponse Model Tests
# ============================================================================


class TestSearchResponseModel:
    """Tests for SearchResponse model."""

    def test_search_empty(self):
        """Test SearchResponse with empty results."""
        response = SearchResponse(query="test")
        assert response.query == "test"
        assert response.results == []
        assert response.total_count == 0

    def test_search_with_results(self):
        """Test SearchResponse with results."""
        response = SearchResponse(
            query="debate AI",
            results=[
                SearchResult(
                    type="debate",
                    id="deb-1",
                    title="AI Safety Debate",
                    snippet="Discussion about AI safety...",
                    score=0.95,
                ),
            ],
            total_count=1,
            suggestions=["AI safety", "AI ethics"],
        )
        assert len(response.results) == 1
        assert response.results[0].type == "debate"
        assert len(response.suggestions) == 2


# ============================================================================
# DebatesAPI.create() Tests
# ============================================================================


class TestDebatesAPICreate:
    """Tests for DebatesAPI.create() method."""

    def test_create_basic(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test basic create() call."""
        mock_client._post.return_value = {
            "debate_id": "deb-new",
            "status": "pending",
        }

        result = debates_api.create("What is the meaning of life?")

        assert result.debate_id == "deb-new"
        assert result.status == DebateStatus.PENDING
        mock_client._post.assert_called_once()

    def test_create_with_options(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test create() with custom options."""
        mock_client._post.return_value = {
            "debate_id": "deb-custom",
            "status": "pending",
            "task": "Custom task",
        }

        result = debates_api.create(
            task="Custom task",
            agents=["claude", "gpt-4", "gemini"],
            rounds=7,
            consensus="unanimous",
            context="Extra context",
        )

        assert result.debate_id == "deb-custom"
        call_args = mock_client._post.call_args
        assert call_args[0][0] == "/api/debates"
        payload = call_args[0][1]
        assert payload["task"] == "Custom task"
        assert payload["agents"] == ["claude", "gpt-4", "gemini"]
        assert payload["rounds"] == 7
        assert payload["consensus"] == "unanimous"


class TestDebatesAPICreateAsync:
    """Tests for DebatesAPI.create_async() method."""

    @pytest.mark.asyncio
    async def test_create_async(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test create_async() call."""
        mock_client._post_async = AsyncMock(
            return_value={
                "debate_id": "deb-async",
                "status": "pending",
            }
        )

        result = await debates_api.create_async("Async debate")

        assert result.debate_id == "deb-async"


# ============================================================================
# DebatesAPI.get() Tests
# ============================================================================


class TestDebatesAPIGet:
    """Tests for DebatesAPI.get() method."""

    def test_get_debate(
        self, debates_api: DebatesAPI, mock_client: MagicMock, sample_timestamp: str
    ):
        """Test get() retrieves debate."""
        mock_client._get.return_value = {
            "debate_id": "deb-get",
            "task": "Test task",
            "status": "completed",
            "agents": ["claude", "gpt-4"],
            "rounds": [],
            "created_at": sample_timestamp,
        }

        result = debates_api.get("deb-get")

        assert result.debate_id == "deb-get"
        assert result.status == DebateStatus.COMPLETED
        assert len(result.agents) == 2
        mock_client._get.assert_called_once_with("/api/debates/deb-get")


class TestDebatesAPIGetAsync:
    """Tests for DebatesAPI.get_async() method."""

    @pytest.mark.asyncio
    async def test_get_async(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test get_async() call."""
        mock_client._get_async = AsyncMock(
            return_value={
                "debate_id": "deb-async-get",
                "task": "Async task",
                "status": "running",
            }
        )

        result = await debates_api.get_async("deb-async-get")

        assert result.debate_id == "deb-async-get"


# ============================================================================
# DebatesAPI.list() Tests
# ============================================================================


class TestDebatesAPIList:
    """Tests for DebatesAPI.list() method."""

    def test_list_basic(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test list() basic call."""
        mock_client._get.return_value = {
            "debates": [
                {"debate_id": "deb-1", "task": "Task 1", "status": "completed"},
                {"debate_id": "deb-2", "task": "Task 2", "status": "running"},
            ]
        }

        result = debates_api.list()

        assert len(result) == 2
        assert result[0].debate_id == "deb-1"

    def test_list_with_params(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test list() with parameters."""
        mock_client._get.return_value = {"debates": []}

        debates_api.list(limit=10, offset=5, status="completed")

        call_args = mock_client._get.call_args
        params = call_args[1]["params"]
        assert params["limit"] == 10
        assert params["offset"] == 5
        assert params["status"] == "completed"

    def test_list_handles_array_response(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test list() handles array response (no wrapper)."""
        mock_client._get.return_value = [
            {"debate_id": "deb-arr", "task": "Array task", "status": "pending"}
        ]

        result = debates_api.list()

        assert len(result) == 1


class TestDebatesAPIListAsync:
    """Tests for DebatesAPI.list_async() method."""

    @pytest.mark.asyncio
    async def test_list_async(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test list_async() call."""
        mock_client._get_async = AsyncMock(
            return_value={
                "debates": [{"debate_id": "deb-async", "task": "Async", "status": "completed"}]
            }
        )

        result = await debates_api.list_async()

        assert len(result) == 1


# ============================================================================
# DebatesAPI.run() Tests
# ============================================================================


class TestDebatesAPIRun:
    """Tests for DebatesAPI.run() method."""

    def test_run_immediate_complete(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test run() when debate completes immediately."""
        mock_client._post.return_value = {
            "debate_id": "deb-run",
            "status": "pending",
        }
        mock_client._get.return_value = {
            "debate_id": "deb-run",
            "task": "Run task",
            "status": "completed",
            "consensus": {"reached": True, "final_answer": "Yes"},
        }

        result = debates_api.run("Quick debate")

        assert result.status == DebateStatus.COMPLETED

    def test_run_polls_until_complete(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test run() polls until completion."""
        mock_client._post.return_value = {
            "debate_id": "deb-poll",
            "status": "pending",
        }
        mock_client._get.side_effect = [
            {"debate_id": "deb-poll", "task": "Poll", "status": "running"},
            {"debate_id": "deb-poll", "task": "Poll", "status": "running"},
            {"debate_id": "deb-poll", "task": "Poll", "status": "completed"},
        ]

        with patch("time.sleep"):
            result = debates_api.run("Polling debate", timeout=60)

        assert result.status == DebateStatus.COMPLETED
        assert mock_client._get.call_count == 3

    def test_run_timeout(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test run() raises TimeoutError."""
        mock_client._post.return_value = {
            "debate_id": "deb-timeout",
            "status": "pending",
        }
        mock_client._get.return_value = {
            "debate_id": "deb-timeout",
            "task": "Timeout",
            "status": "running",
        }

        with patch("time.sleep"):
            with patch("time.time", side_effect=[0, 0, 100]):
                with pytest.raises(TimeoutError):
                    debates_api.run("Timeout debate", timeout=10)


# ============================================================================
# DebatesAPI.wait_for_completion() Tests
# ============================================================================


class TestDebatesAPIWaitForCompletion:
    """Tests for DebatesAPI.wait_for_completion() method."""

    def test_wait_immediate(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test wait_for_completion() when already complete."""
        mock_client._get.return_value = {
            "debate_id": "deb-wait",
            "task": "Wait",
            "status": "completed",
        }

        result = debates_api.wait_for_completion("deb-wait")

        assert result.status == DebateStatus.COMPLETED

    def test_wait_polls(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test wait_for_completion() polls."""
        mock_client._get.side_effect = [
            {"debate_id": "deb-poll", "task": "Poll", "status": "running"},
            {"debate_id": "deb-poll", "task": "Poll", "status": "completed"},
        ]

        with patch("time.sleep"):
            result = debates_api.wait_for_completion("deb-poll")

        assert result.status == DebateStatus.COMPLETED

    def test_wait_returns_on_failed(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test wait_for_completion() returns on failed status."""
        mock_client._get.return_value = {
            "debate_id": "deb-fail",
            "task": "Fail",
            "status": "failed",
        }

        result = debates_api.wait_for_completion("deb-fail")

        assert result.status == DebateStatus.FAILED


# ============================================================================
# DebatesAPI.compare() Tests
# ============================================================================


class TestDebatesAPICompare:
    """Tests for DebatesAPI.compare() method."""

    def test_compare_debates(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test compare() fetches multiple debates."""
        mock_client._get.side_effect = [
            {"debate_id": "deb-1", "task": "Task 1", "status": "completed"},
            {"debate_id": "deb-2", "task": "Task 2", "status": "completed"},
        ]

        result = debates_api.compare(["deb-1", "deb-2"])

        assert len(result) == 2
        assert result[0].debate_id == "deb-1"
        assert result[1].debate_id == "deb-2"


class TestDebatesAPICompareAsync:
    """Tests for DebatesAPI.compare_async() method."""

    @pytest.mark.asyncio
    async def test_compare_async(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test compare_async() call."""
        mock_client._get_async = AsyncMock(
            side_effect=[
                {"debate_id": "deb-a", "task": "A", "status": "completed"},
                {"debate_id": "deb-b", "task": "B", "status": "completed"},
            ]
        )

        result = await debates_api.compare_async(["deb-a", "deb-b"])

        assert len(result) == 2


# ============================================================================
# DebatesAPI.batch_get() Tests
# ============================================================================


class TestDebatesAPIBatchGet:
    """Tests for DebatesAPI.batch_get() method."""

    def test_batch_get(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test batch_get() fetches in order."""
        mock_client._get.side_effect = [
            {"debate_id": f"deb-{i}", "task": f"Task {i}", "status": "completed"} for i in range(3)
        ]

        result = debates_api.batch_get(["deb-0", "deb-1", "deb-2"])

        assert len(result) == 3
        assert result[0].debate_id == "deb-0"
        assert result[2].debate_id == "deb-2"

    def test_batch_get_with_pacing(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test batch_get() adds delay for pacing."""
        mock_client._get.side_effect = [
            {"debate_id": f"deb-{i}", "task": f"Task {i}", "status": "completed"} for i in range(12)
        ]

        with patch("time.sleep") as mock_sleep:
            debates_api.batch_get([f"deb-{i}" for i in range(12)], max_concurrent=5)

        # Should have slept twice (after 5 and after 10)
        assert mock_sleep.call_count == 2


# ============================================================================
# DebatesAPI.iterate() Tests
# ============================================================================


class TestDebatesAPIIterate:
    """Tests for DebatesAPI.iterate() method."""

    def test_iterate_single_page(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test iterate() with single page."""
        mock_client._get.return_value = {
            "debates": [
                {"debate_id": "deb-1", "task": "Task 1", "status": "completed"},
                {"debate_id": "deb-2", "task": "Task 2", "status": "completed"},
            ]
        }

        result = list(debates_api.iterate(page_size=10))

        assert len(result) == 2

    def test_iterate_multiple_pages(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test iterate() paginates."""
        mock_client._get.side_effect = [
            {
                "debates": [
                    {"debate_id": f"deb-{i}", "task": f"Task {i}", "status": "completed"}
                    for i in range(2)
                ]
            },
            {
                "debates": [
                    {"debate_id": f"deb-{i}", "task": f"Task {i}", "status": "completed"}
                    for i in range(2, 3)
                ]
            },
        ]

        result = list(debates_api.iterate(page_size=2))

        assert len(result) == 3

    def test_iterate_with_max_items(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test iterate() respects max_items."""
        mock_client._get.return_value = {
            "debates": [
                {"debate_id": f"deb-{i}", "task": f"Task {i}", "status": "completed"}
                for i in range(5)
            ]
        }

        result = list(debates_api.iterate(max_items=3))

        assert len(result) == 3


# ============================================================================
# DebatesAPI.update() Tests
# ============================================================================


class TestDebatesAPIUpdate:
    """Tests for DebatesAPI.update() method."""

    def test_update_status(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test update() changes status."""
        mock_client._patch.return_value = {
            "debate_id": "deb-update",
            "task": "Task",
            "status": "paused",
        }

        result = debates_api.update("deb-update", status="paused")

        assert result.status == DebateStatus.PAUSED
        call_args = mock_client._patch.call_args
        assert call_args[0][0] == "/api/v1/debates/deb-update"

    def test_update_metadata(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test update() changes metadata."""
        mock_client._patch.return_value = {
            "debate_id": "deb-meta",
            "task": "Task",
            "status": "completed",
            "metadata": {"priority": "high"},
        }

        result = debates_api.update(
            "deb-meta",
            metadata={"priority": "high"},
            tags=["important"],
            notes="Review needed",
        )

        call_args = mock_client._patch.call_args
        body = call_args[0][1]
        assert body["metadata"] == {"priority": "high"}
        assert body["tags"] == ["important"]


class TestDebatesAPIUpdateAsync:
    """Tests for DebatesAPI.update_async() method."""

    @pytest.mark.asyncio
    async def test_update_async(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test update_async() call."""
        mock_client._patch_async = AsyncMock(
            return_value={
                "debate_id": "deb-async",
                "task": "Async",
                "status": "completed",
            }
        )

        result = await debates_api.update_async("deb-async", archived=True)

        assert result.debate_id == "deb-async"


# ============================================================================
# DebatesAPI.get_verification_report() Tests
# ============================================================================


class TestDebatesAPIGetVerificationReport:
    """Tests for DebatesAPI.get_verification_report() method."""

    def test_get_verification_report(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test get_verification_report() call."""
        mock_client._get.return_value = {
            "debate_id": "deb-verify",
            "verified": True,
            "verification_method": "fact_check",
            "claims_verified": 5,
            "claims_failed": 0,
            "overall_confidence": 0.95,
        }

        result = debates_api.get_verification_report("deb-verify")

        assert result.debate_id == "deb-verify"
        assert result.verified is True
        assert result.claims_verified == 5
        mock_client._get.assert_called_once_with("/api/v1/debates/deb-verify/verification-report")


class TestDebatesAPIGetVerificationReportAsync:
    """Tests for DebatesAPI.get_verification_report_async() method."""

    @pytest.mark.asyncio
    async def test_get_verification_report_async(
        self, debates_api: DebatesAPI, mock_client: MagicMock
    ):
        """Test get_verification_report_async() call."""
        mock_client._get_async = AsyncMock(
            return_value={
                "debate_id": "deb-async-verify",
                "verified": True,
            }
        )

        result = await debates_api.get_verification_report_async("deb-async-verify")

        assert result.verified is True


# ============================================================================
# DebatesAPI.search() Tests
# ============================================================================


class TestDebatesAPISearch:
    """Tests for DebatesAPI.search() method."""

    def test_search_basic(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test search() basic call."""
        mock_client._get.return_value = {
            "query": "AI safety",
            "results": [
                {
                    "type": "debate",
                    "id": "deb-search",
                    "title": "AI Safety Debate",
                    "score": 0.9,
                }
            ],
            "total_count": 1,
        }

        result = debates_api.search("AI safety")

        assert result.query == "AI safety"
        assert len(result.results) == 1
        assert result.results[0].type == "debate"

    def test_search_with_pagination(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test search() with pagination."""
        mock_client._get.return_value = {
            "query": "test",
            "results": [],
            "total_count": 0,
        }

        debates_api.search("test", limit=10, offset=20)

        call_args = mock_client._get.call_args
        params = call_args[1]["params"]
        assert params["q"] == "test"
        assert params["limit"] == 10
        assert params["offset"] == 20


class TestDebatesAPISearchAsync:
    """Tests for DebatesAPI.search_async() method."""

    @pytest.mark.asyncio
    async def test_search_async(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test search_async() call."""
        mock_client._get_async = AsyncMock(
            return_value={
                "query": "async search",
                "results": [],
                "total_count": 0,
            }
        )

        result = await debates_api.search_async("async search")

        assert result.query == "async search"


# ============================================================================
# Integration-like Tests
# ============================================================================


class TestDebatesAPIIntegration:
    """Integration-like tests for DebatesAPI."""

    def test_full_debate_workflow(
        self, debates_api: DebatesAPI, mock_client: MagicMock, sample_timestamp: str
    ):
        """Test full debate workflow: create -> poll -> get."""
        # Create debate
        mock_client._post.return_value = {
            "debate_id": "deb-workflow",
            "status": "pending",
        }
        create_response = debates_api.create(
            "Should we use microservices?",
            agents=["claude", "gpt-4"],
            rounds=3,
            consensus="majority",
        )
        assert create_response.debate_id == "deb-workflow"

        # Get debate (completed)
        mock_client._get.return_value = {
            "debate_id": "deb-workflow",
            "task": "Should we use microservices?",
            "status": "completed",
            "agents": ["claude", "gpt-4"],
            "rounds": [
                {
                    "round_number": 1,
                    "messages": [
                        {"agent_id": "claude", "content": "I propose..."},
                        {"agent_id": "gpt-4", "content": "I agree..."},
                    ],
                }
            ],
            "consensus": {
                "reached": True,
                "agreement": 0.85,
                "final_answer": "Yes, for this scale",
            },
            "created_at": sample_timestamp,
        }
        debate = debates_api.get("deb-workflow")
        assert debate.status == DebateStatus.COMPLETED
        assert debate.consensus is not None
        assert debate.consensus.reached is True
        assert len(debate.rounds) == 1

    def test_search_and_compare_workflow(self, debates_api: DebatesAPI, mock_client: MagicMock):
        """Test search and compare workflow."""
        # Search for debates
        mock_client._get.return_value = {
            "query": "architecture",
            "results": [
                {"type": "debate", "id": "deb-arch-1", "score": 0.95},
                {"type": "debate", "id": "deb-arch-2", "score": 0.90},
            ],
            "total_count": 2,
        }
        search_result = debates_api.search("architecture")
        debate_ids = [r.id for r in search_result.results]

        # Compare the found debates
        mock_client._get.side_effect = [
            {
                "debate_id": "deb-arch-1",
                "task": "Microservices vs Monolith",
                "status": "completed",
            },
            {
                "debate_id": "deb-arch-2",
                "task": "Event Sourcing",
                "status": "completed",
            },
        ]
        compared = debates_api.compare(debate_ids)
        assert len(compared) == 2
        assert compared[0].task == "Microservices vs Monolith"
