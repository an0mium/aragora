"""Tests for GraphQL resolvers."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from aragora.server.graphql.resolvers import (
    QueryResolvers,
    MutationResolvers,
    ResolverContext,
    ResolverResult,
    QUERY_RESOLVERS,
    MUTATION_RESOLVERS,
    _normalize_debate_status,
    _normalize_agent_status,
    _normalize_task_status,
    _normalize_priority,
    _normalize_health_status,
    _to_iso_datetime,
    _transform_debate,
    _transform_agent,
    _transform_task,
)


class TestStatusNormalization:
    """Tests for status normalization functions."""

    def test_normalize_debate_status(self):
        """Test debate status normalization."""
        assert _normalize_debate_status("starting") == "PENDING"
        assert _normalize_debate_status("running") == "RUNNING"
        assert _normalize_debate_status("active") == "RUNNING"
        assert _normalize_debate_status("completed") == "COMPLETED"
        assert _normalize_debate_status("concluded") == "COMPLETED"
        assert _normalize_debate_status("failed") == "FAILED"
        assert _normalize_debate_status("cancelled") == "CANCELLED"
        assert _normalize_debate_status(None) == "PENDING"
        assert _normalize_debate_status("UNKNOWN") == "UNKNOWN"

    def test_normalize_agent_status(self):
        """Test agent status normalization."""
        assert _normalize_agent_status("available") == "AVAILABLE"
        assert _normalize_agent_status("idle") == "AVAILABLE"
        assert _normalize_agent_status("busy") == "BUSY"
        assert _normalize_agent_status("running") == "BUSY"
        assert _normalize_agent_status("offline") == "OFFLINE"
        assert _normalize_agent_status("degraded") == "DEGRADED"
        assert _normalize_agent_status(None) == "OFFLINE"

    def test_normalize_task_status(self):
        """Test task status normalization."""
        assert _normalize_task_status("pending") == "PENDING"
        assert _normalize_task_status("queued") == "PENDING"
        assert _normalize_task_status("running") == "RUNNING"
        assert _normalize_task_status("in_progress") == "RUNNING"
        assert _normalize_task_status("completed") == "COMPLETED"
        assert _normalize_task_status("done") == "COMPLETED"
        assert _normalize_task_status("failed") == "FAILED"
        assert _normalize_task_status("cancelled") == "CANCELLED"
        assert _normalize_task_status(None) == "PENDING"

    def test_normalize_priority(self):
        """Test priority normalization."""
        assert _normalize_priority("low") == "LOW"
        assert _normalize_priority("normal") == "NORMAL"
        assert _normalize_priority("medium") == "NORMAL"
        assert _normalize_priority("high") == "HIGH"
        assert _normalize_priority("urgent") == "URGENT"
        assert _normalize_priority("critical") == "URGENT"
        assert _normalize_priority(None) == "NORMAL"

    def test_normalize_health_status(self):
        """Test health status normalization."""
        assert _normalize_health_status("healthy") == "HEALTHY"
        assert _normalize_health_status("ok") == "HEALTHY"
        assert _normalize_health_status("degraded") == "DEGRADED"
        assert _normalize_health_status("warning") == "DEGRADED"
        assert _normalize_health_status("unhealthy") == "UNHEALTHY"
        assert _normalize_health_status("error") == "UNHEALTHY"
        assert _normalize_health_status(None) == "HEALTHY"


class TestDatetimeConversion:
    """Tests for datetime conversion."""

    def test_to_iso_datetime_string(self):
        """Test ISO datetime conversion from string."""
        result = _to_iso_datetime("2024-01-15T10:30:00Z")
        assert result == "2024-01-15T10:30:00Z"

    def test_to_iso_datetime_datetime(self):
        """Test ISO datetime conversion from datetime object."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = _to_iso_datetime(dt)
        assert "2024-01-15" in result
        assert "10:30:00" in result

    def test_to_iso_datetime_timestamp(self):
        """Test ISO datetime conversion from Unix timestamp."""
        timestamp = 1705315800  # 2024-01-15 10:30:00 UTC
        result = _to_iso_datetime(timestamp)
        assert result is not None
        assert "2024" in result

    def test_to_iso_datetime_none(self):
        """Test ISO datetime conversion with None."""
        assert _to_iso_datetime(None) is None


class TestTransformFunctions:
    """Tests for data transformation functions."""

    def test_transform_debate_basic(self):
        """Test basic debate transformation."""
        debate = {
            "id": "debate-123",
            "task": "Test question",
            "status": "completed",
            "messages": [],
            "agents": ["claude", "gpt4"],
            "rounds": 3,
            "consensus_reached": True,
            "confidence": 0.85,
        }

        result = _transform_debate(debate)

        assert result["id"] == "debate-123"
        assert result["topic"] == "Test question"
        assert result["status"] == "COMPLETED"
        assert result["consensusReached"] is True
        assert result["confidence"] == 0.85
        assert len(result["participants"]) == 2

    def test_transform_debate_with_messages(self):
        """Test debate transformation with messages."""
        debate = {
            "id": "debate-123",
            "task": "Test",
            "status": "running",
            "messages": [
                {"role": "agent", "agent": "claude", "content": "Yes", "round": 1},
                {"role": "agent", "agent": "gpt4", "content": "No", "round": 1},
                {"role": "agent", "agent": "claude", "content": "Still yes", "round": 2},
            ],
            "agents": ["claude", "gpt4"],
            "rounds": 3,
        }

        result = _transform_debate(debate)

        assert len(result["rounds"]) == 2  # Rounds 1 and 2
        assert len(result["rounds"][0]["messages"]) == 2
        assert len(result["rounds"][1]["messages"]) == 1

    def test_transform_debate_with_consensus(self):
        """Test debate transformation with consensus."""
        debate = {
            "id": "debate-123",
            "task": "Test",
            "status": "completed",
            "messages": [],
            "agents": ["claude", "gpt4"],
            "consensus_reached": True,
            "final_answer": "The answer is 42",
            "agreeing_agents": ["claude", "gpt4"],
            "confidence": 0.95,
            "consensus_method": "unanimous",
        }

        result = _transform_debate(debate)

        assert result["consensus"] is not None
        assert result["consensus"]["reached"] is True
        assert result["consensus"]["answer"] == "The answer is 42"
        assert result["consensus"]["confidence"] == 0.95

    def test_transform_agent_dict(self):
        """Test agent transformation from dict."""
        agent = {
            "name": "claude",
            "elo": 1600,
            "wins": 10,
            "losses": 5,
            "draws": 2,
            "status": "available",
        }

        result = _transform_agent(agent)

        assert result["id"] == "claude"
        assert result["name"] == "claude"
        assert result["elo"] == 1600
        assert result["stats"]["wins"] == 10
        assert result["stats"]["losses"] == 5

    def test_transform_agent_object(self):
        """Test agent transformation from object."""
        agent = MagicMock()
        agent.name = "gpt4"
        agent.elo = 1550
        agent.wins = 8
        agent.losses = 6
        agent.draws = 3
        agent.capabilities = ["reasoning", "coding"]

        result = _transform_agent(agent)

        assert result["id"] == "gpt4"
        assert result["name"] == "gpt4"
        assert result["elo"] == 1550
        assert result["stats"]["totalGames"] == 17

    def test_transform_task_dict(self):
        """Test task transformation from dict."""
        task = {
            "id": "task-123",
            "task_type": "debate",
            "status": "running",
            "priority": "high",
            "assigned_agent": "claude",
            "created_at": "2024-01-15T10:00:00Z",
        }

        result = _transform_task(task)

        assert result["id"] == "task-123"
        assert result["type"] == "debate"
        assert result["status"] == "RUNNING"
        assert result["priority"] == "HIGH"

    def test_transform_task_object(self):
        """Test task transformation from object."""
        task = MagicMock()
        task.id = "task-456"
        task.task_type = "analysis"
        task.status = MagicMock(value="pending")
        task.priority = MagicMock(name="NORMAL")
        task.assigned_agent = None
        task.result = None
        task.created_at = 1705315800
        task.completed_at = None
        task.payload = {"data": "test"}
        task.metadata = {}

        result = _transform_task(task)

        assert result["id"] == "task-456"
        assert result["type"] == "analysis"
        assert result["status"] == "PENDING"


class TestResolverContext:
    """Tests for ResolverContext."""

    def test_resolver_context_creation(self):
        """Test creating a resolver context."""
        ctx = ResolverContext(
            server_context={"storage": MagicMock()},
            user_id="user-123",
            org_id="org-456",
        )

        assert ctx.user_id == "user-123"
        assert ctx.org_id == "org-456"
        assert ctx.variables == {}

    def test_resolver_context_with_variables(self):
        """Test resolver context with variables."""
        ctx = ResolverContext(
            server_context={},
            variables={"id": "123", "limit": 10},
        )

        assert ctx.variables["id"] == "123"
        assert ctx.variables["limit"] == 10


class TestResolverResult:
    """Tests for ResolverResult."""

    def test_resolver_result_success(self):
        """Test successful resolver result."""
        result = ResolverResult(data={"id": "123"})

        assert result.success is True
        assert result.data == {"id": "123"}
        assert result.errors == []

    def test_resolver_result_error(self):
        """Test resolver result with errors."""
        result = ResolverResult(errors=["Something went wrong"])

        assert result.success is False
        assert result.data is None
        assert len(result.errors) == 1


class TestQueryResolvers:
    """Tests for Query resolvers."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock resolver context."""
        server_ctx = {
            "storage": MagicMock(),
            "elo_system": MagicMock(),
            "control_plane_coordinator": None,
            "_start_time": 1000000,
        }
        return ResolverContext(server_context=server_ctx)

    @pytest.mark.asyncio
    async def test_resolve_debate(self, mock_context):
        """Test resolving a single debate."""
        mock_context.server_context["storage"].get_debate.return_value = {
            "id": "debate-123",
            "task": "Test question",
            "status": "completed",
            "messages": [],
            "agents": ["claude"],
            "rounds": 3,
        }

        result = await QueryResolvers.resolve_debate(mock_context, id="debate-123")

        assert result.success
        assert result.data["id"] == "debate-123"

    @pytest.mark.asyncio
    async def test_resolve_debate_not_found(self, mock_context):
        """Test resolving a non-existent debate."""
        mock_context.server_context["storage"].get_debate.return_value = None

        result = await QueryResolvers.resolve_debate(mock_context, id="nonexistent")

        assert not result.success
        assert any("not found" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_resolve_debates(self, mock_context):
        """Test resolving a list of debates."""
        mock_context.server_context["storage"].list_recent.return_value = [
            {"id": "d1", "task": "Q1", "status": "completed", "agents": [], "messages": []},
            {"id": "d2", "task": "Q2", "status": "running", "agents": [], "messages": []},
        ]

        result = await QueryResolvers.resolve_debates(mock_context, limit=10)

        assert result.success
        assert result.data["total"] == 2
        assert len(result.data["debates"]) == 2

    @pytest.mark.asyncio
    async def test_resolve_debates_with_status_filter(self, mock_context):
        """Test resolving debates with status filter."""
        mock_context.server_context["storage"].list_recent.return_value = [
            {"id": "d1", "task": "Q1", "status": "completed", "agents": [], "messages": []},
            {"id": "d2", "task": "Q2", "status": "running", "agents": [], "messages": []},
        ]

        result = await QueryResolvers.resolve_debates(mock_context, status="COMPLETED", limit=10)

        assert result.success
        # Only completed debates should be returned
        for debate in result.data["debates"]:
            assert debate["status"] == "COMPLETED"

    @pytest.mark.asyncio
    async def test_resolve_agent(self, mock_context):
        """Test resolving a single agent."""
        mock_rating = MagicMock()
        mock_rating.name = "claude"
        mock_rating.elo = 1600
        mock_rating.wins = 10
        mock_rating.losses = 5
        mock_rating.draws = 2
        mock_rating.capabilities = []
        mock_context.server_context["elo_system"].get_rating.return_value = mock_rating

        result = await QueryResolvers.resolve_agent(mock_context, id="claude")

        assert result.success
        assert result.data["name"] == "claude"
        assert result.data["elo"] == 1600

    @pytest.mark.asyncio
    async def test_resolve_leaderboard(self, mock_context):
        """Test resolving the leaderboard."""
        mock_ratings = []
        for name, elo in [("claude", 1600), ("gpt4", 1550), ("gemini", 1500)]:
            r = MagicMock()
            r.name = name
            r.elo = elo
            r.wins = 10
            r.losses = 5
            r.draws = 2
            mock_ratings.append(r)

        mock_context.server_context["elo_system"].get_cached_leaderboard.return_value = mock_ratings

        result = await QueryResolvers.resolve_leaderboard(mock_context, limit=10)

        assert result.success
        assert len(result.data) == 3

    @pytest.mark.asyncio
    async def test_resolve_system_health(self, mock_context):
        """Test resolving system health."""
        result = await QueryResolvers.resolve_system_health(mock_context)

        assert result.success
        assert "status" in result.data
        assert "components" in result.data
        assert "version" in result.data

    @pytest.mark.asyncio
    async def test_resolve_stats(self, mock_context):
        """Test resolving system stats."""
        result = await QueryResolvers.resolve_stats(mock_context)

        assert result.success
        assert "activeJobs" in result.data
        assert "totalAgents" in result.data


class TestMutationResolvers:
    """Tests for Mutation resolvers."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock resolver context."""
        server_ctx = {
            "storage": MagicMock(),
            "elo_system": MagicMock(),
            "control_plane_coordinator": AsyncMock(),
        }
        return ResolverContext(
            server_context=server_ctx,
            user_id="user-123",
            org_id="org-456",
        )

    @pytest.mark.asyncio
    async def test_resolve_start_debate_missing_question(self, mock_context):
        """Test starting debate without question fails."""
        result = await MutationResolvers.resolve_start_debate(
            mock_context,
            input={},
        )

        assert not result.success
        assert any("question" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_resolve_submit_vote_missing_agent(self, mock_context):
        """Test submitting vote without agent fails."""
        result = await MutationResolvers.resolve_submit_vote(
            mock_context,
            debate_id="debate-123",
            vote={},
        )

        assert not result.success
        assert any("agent" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_resolve_submit_vote(self, mock_context):
        """Test submitting a vote."""
        mock_context.server_context["storage"].get_debate.return_value = {
            "id": "debate-123",
            "task": "Test",
            "status": "running",
        }

        result = await MutationResolvers.resolve_submit_vote(
            mock_context,
            debate_id="debate-123",
            vote={"agentId": "claude", "reason": "Best argument"},
        )

        assert result.success
        assert result.data["agentId"] == "claude"

    @pytest.mark.asyncio
    async def test_resolve_submit_task_missing_type(self, mock_context):
        """Test submitting task without type fails."""
        result = await MutationResolvers.resolve_submit_task(
            mock_context,
            input={},
        )

        assert not result.success
        assert any("task type" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_resolve_register_agent_missing_id(self, mock_context):
        """Test registering agent without ID fails."""
        result = await MutationResolvers.resolve_register_agent(
            mock_context,
            input={},
        )

        assert not result.success
        assert any("agent id" in e.lower() for e in result.errors)


class TestResolverRegistry:
    """Tests for resolver registry."""

    def test_query_resolvers_registered(self):
        """Test that all query resolvers are registered."""
        expected = {
            "debate",
            "debates",
            "searchDebates",
            "agent",
            "agents",
            "leaderboard",
            "task",
            "tasks",
            "systemHealth",
            "stats",
        }
        assert expected.issubset(set(QUERY_RESOLVERS.keys()))

    def test_mutation_resolvers_registered(self):
        """Test that all mutation resolvers are registered."""
        expected = {
            "startDebate",
            "submitVote",
            "cancelDebate",
            "submitTask",
            "cancelTask",
            "registerAgent",
            "unregisterAgent",
        }
        assert expected.issubset(set(MUTATION_RESOLVERS.keys()))

    def test_all_resolvers_callable(self):
        """Test that all registered resolvers are callable."""
        for name, resolver in QUERY_RESOLVERS.items():
            assert callable(resolver), f"Query resolver {name} is not callable"

        for name, resolver in MUTATION_RESOLVERS.items():
            assert callable(resolver), f"Mutation resolver {name} is not callable"
