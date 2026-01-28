"""
Tests for Debate Intervention Handler.

Tests cover:
- Pause/resume debate functionality
- Argument injection
- Agent weight adjustment
- Consensus threshold updates
- State and log retrieval
"""

import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

from aragora.rbac.models import AuthorizationContext
from aragora.server.handlers.debates import intervention


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_auth_context():
    """Create a mock authorization context with all required attributes."""
    ctx = Mock()
    ctx.user_id = "test-user-123"
    ctx.org_id = "test-org"
    ctx.roles = ["admin"]
    ctx.permissions = frozenset(["debates:read", "debates:write", "admin:*"])
    ctx.is_authenticated = True
    ctx.tenant_id = None
    ctx.resource_scopes = {}
    return ctx


@pytest.fixture(autouse=True)
def clear_state():
    """Clear intervention state before each test."""
    intervention._debate_state.clear()
    intervention._intervention_log.clear()
    yield
    intervention._debate_state.clear()
    intervention._intervention_log.clear()


@pytest.fixture(autouse=True)
def mock_permission_check():
    """Mock the permission checker to always allow access."""
    with patch("aragora.rbac.decorators.get_permission_checker") as mock_get:
        mock_checker = Mock()
        mock_checker.check_permission.return_value = Mock(
            allowed=True,
            permission="test",
            reason="test allowed",
        )
        mock_get.return_value = mock_checker
        yield mock_checker


# ============================================================================
# State Management Tests
# ============================================================================


class TestDebateState:
    """Tests for debate state management."""

    def test_get_debate_state_creates_new(self):
        """Test get_debate_state creates new state if not exists."""
        state = intervention.get_debate_state("debate-123")

        assert state["is_paused"] is False
        assert state["agent_weights"] == {}
        assert state["consensus_threshold"] == 0.75
        assert state["injected_arguments"] == []
        assert state["follow_up_questions"] == []

    def test_get_debate_state_returns_existing(self):
        """Test get_debate_state returns existing state."""
        # Create initial state
        intervention._debate_state["debate-123"] = {
            "is_paused": True,
            "agent_weights": {"claude": 1.5},
            "consensus_threshold": 0.8,
            "injected_arguments": [],
            "follow_up_questions": [],
        }

        state = intervention.get_debate_state("debate-123")

        assert state["is_paused"] is True
        assert state["agent_weights"] == {"claude": 1.5}
        assert state["consensus_threshold"] == 0.8

    def test_log_intervention_adds_entry(self):
        """Test log_intervention adds audit entry."""
        intervention.log_intervention(
            debate_id="debate-123",
            intervention_type="pause",
            data={"reason": "user requested"},
            user_id="user-456",
        )

        assert len(intervention._intervention_log) == 1
        entry = intervention._intervention_log[0]
        assert entry["debate_id"] == "debate-123"
        assert entry["type"] == "pause"
        assert entry["data"] == {"reason": "user requested"}
        assert entry["user_id"] == "user-456"
        assert "timestamp" in entry


# ============================================================================
# Pause/Resume Tests
# ============================================================================


class TestPauseResume:
    """Tests for pause/resume functionality."""

    @pytest.mark.asyncio
    async def test_pause_debate_success(self, mock_auth_context):
        """Test successful debate pause."""
        result = await intervention.handle_pause_debate("debate-123", mock_auth_context)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["success"] is True
        assert data["is_paused"] is True
        assert "paused_at" in data

        # Verify state updated
        state = intervention.get_debate_state("debate-123")
        assert state["is_paused"] is True

    @pytest.mark.asyncio
    async def test_pause_already_paused(self, mock_auth_context):
        """Test pause fails if already paused."""
        # Pause first
        await intervention.handle_pause_debate("debate-123", mock_auth_context)

        # Try to pause again
        result = await intervention.handle_pause_debate("debate-123", mock_auth_context)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["success"] is False
        assert "already paused" in data["error"]

    @pytest.mark.asyncio
    async def test_resume_debate_success(self, mock_auth_context):
        """Test successful debate resume."""
        # Pause first
        await intervention.handle_pause_debate("debate-123", mock_auth_context)

        # Resume
        result = await intervention.handle_resume_debate("debate-123", mock_auth_context)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["success"] is True
        assert data["is_paused"] is False
        assert "resumed_at" in data
        assert "pause_duration_seconds" in data

    @pytest.mark.asyncio
    async def test_resume_not_paused(self, mock_auth_context):
        """Test resume fails if not paused."""
        result = await intervention.handle_resume_debate("debate-123", mock_auth_context)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["success"] is False
        assert "not paused" in data["error"]


# ============================================================================
# Argument Injection Tests
# ============================================================================


class TestArgumentInjection:
    """Tests for argument injection."""

    @pytest.mark.asyncio
    async def test_inject_argument_success(self, mock_auth_context):
        """Test successful argument injection."""
        result = await intervention.handle_inject_argument(
            debate_id="debate-123",
            context=mock_auth_context,
            content="Consider the environmental impact.",
            injection_type="argument",
            source="user",
        )

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["success"] is True
        assert "injection_id" in data
        assert data["type"] == "argument"

        # Verify state
        state = intervention.get_debate_state("debate-123")
        assert len(state["injected_arguments"]) == 1
        assert state["injected_arguments"][0]["content"] == "Consider the environmental impact."

    @pytest.mark.asyncio
    async def test_inject_follow_up(self, mock_auth_context):
        """Test follow-up question injection."""
        result = await intervention.handle_inject_argument(
            debate_id="debate-123",
            context=mock_auth_context,
            content="What about long-term costs?",
            injection_type="follow_up",
            source="user",
        )

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["type"] == "follow_up"

        # Verify state
        state = intervention.get_debate_state("debate-123")
        assert len(state["follow_up_questions"]) == 1

    @pytest.mark.asyncio
    async def test_inject_empty_content(self, mock_auth_context):
        """Test injection fails with empty content."""
        result = await intervention.handle_inject_argument(
            debate_id="debate-123",
            context=mock_auth_context,
            content="",
            injection_type="argument",
            source="user",
        )

        assert result.status_code == 400
        data = json.loads(result.body)
        assert data["success"] is False
        assert "empty" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_inject_whitespace_only(self, mock_auth_context):
        """Test injection fails with whitespace-only content."""
        result = await intervention.handle_inject_argument(
            debate_id="debate-123",
            context=mock_auth_context,
            content="   ",
            injection_type="argument",
            source="user",
        )

        assert result.status_code == 400


# ============================================================================
# Weight Adjustment Tests
# ============================================================================


class TestWeightAdjustment:
    """Tests for agent weight adjustment."""

    @pytest.mark.asyncio
    async def test_update_weight_success(self, mock_auth_context):
        """Test successful weight update."""
        result = await intervention.handle_update_weights(
            debate_id="debate-123",
            context=mock_auth_context,
            agent="claude",
            weight=1.5,
        )

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["success"] is True
        assert data["agent"] == "claude"
        assert data["old_weight"] == 1.0
        assert data["new_weight"] == 1.5

        # Verify state
        state = intervention.get_debate_state("debate-123")
        assert state["agent_weights"]["claude"] == 1.5

    @pytest.mark.asyncio
    async def test_mute_agent(self, mock_auth_context):
        """Test muting agent with weight 0."""
        result = await intervention.handle_update_weights(
            debate_id="debate-123",
            context=mock_auth_context,
            agent="gpt4",
            weight=0.0,
        )

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["new_weight"] == 0.0

    @pytest.mark.asyncio
    async def test_weight_out_of_range_low(self, mock_auth_context):
        """Test weight below minimum fails."""
        result = await intervention.handle_update_weights(
            debate_id="debate-123",
            context=mock_auth_context,
            agent="claude",
            weight=-0.5,
        )

        assert result.status_code == 400
        data = json.loads(result.body)
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_weight_out_of_range_high(self, mock_auth_context):
        """Test weight above maximum fails."""
        result = await intervention.handle_update_weights(
            debate_id="debate-123",
            context=mock_auth_context,
            agent="claude",
            weight=3.0,
        )

        assert result.status_code == 400


# ============================================================================
# Threshold Update Tests
# ============================================================================


class TestThresholdUpdate:
    """Tests for consensus threshold updates."""

    @pytest.mark.asyncio
    async def test_update_threshold_success(self, mock_auth_context):
        """Test successful threshold update."""
        result = await intervention.handle_update_threshold(
            debate_id="debate-123",
            context=mock_auth_context,
            threshold=0.9,
        )

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["success"] is True
        assert data["old_threshold"] == 0.75
        assert data["new_threshold"] == 0.9

        # Verify state
        state = intervention.get_debate_state("debate-123")
        assert state["consensus_threshold"] == 0.9

    @pytest.mark.asyncio
    async def test_threshold_minimum(self, mock_auth_context):
        """Test threshold at minimum (simple majority)."""
        result = await intervention.handle_update_threshold(
            debate_id="debate-123",
            context=mock_auth_context,
            threshold=0.5,
        )

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["new_threshold"] == 0.5

    @pytest.mark.asyncio
    async def test_threshold_maximum(self, mock_auth_context):
        """Test threshold at maximum (unanimous)."""
        result = await intervention.handle_update_threshold(
            debate_id="debate-123",
            context=mock_auth_context,
            threshold=1.0,
        )

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["new_threshold"] == 1.0

    @pytest.mark.asyncio
    async def test_threshold_below_minimum(self, mock_auth_context):
        """Test threshold below minimum fails."""
        result = await intervention.handle_update_threshold(
            debate_id="debate-123",
            context=mock_auth_context,
            threshold=0.3,
        )

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_threshold_above_maximum(self, mock_auth_context):
        """Test threshold above maximum fails."""
        result = await intervention.handle_update_threshold(
            debate_id="debate-123",
            context=mock_auth_context,
            threshold=1.5,
        )

        assert result.status_code == 400


# ============================================================================
# State Retrieval Tests
# ============================================================================


class TestStateRetrieval:
    """Tests for state retrieval endpoints."""

    @pytest.mark.asyncio
    async def test_get_intervention_state(self, mock_auth_context):
        """Test getting intervention state."""
        # Set up some state
        await intervention.handle_pause_debate("debate-123", mock_auth_context)
        await intervention.handle_update_weights("debate-123", mock_auth_context, "claude", 1.5)
        await intervention.handle_inject_argument(
            "debate-123", mock_auth_context, "Test argument", "argument", "user"
        )

        result = await intervention.handle_get_intervention_state("debate-123", mock_auth_context)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["debate_id"] == "debate-123"
        assert data["is_paused"] is True
        assert data["consensus_threshold"] == 0.75
        assert "claude" in data["agent_weights"]
        assert data["pending_injections"] == 1

    @pytest.mark.asyncio
    async def test_get_intervention_log(self, mock_auth_context):
        """Test getting intervention log."""
        # Create some interventions
        await intervention.handle_pause_debate("debate-123", mock_auth_context)
        await intervention.handle_resume_debate("debate-123", mock_auth_context)
        await intervention.handle_update_threshold("debate-123", mock_auth_context, 0.9)

        result = await intervention.handle_get_intervention_log(
            "debate-123", mock_auth_context, limit=50
        )

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["debate_id"] == "debate-123"
        assert data["total_interventions"] == 3
        assert len(data["interventions"]) == 3

    @pytest.mark.asyncio
    async def test_get_intervention_log_with_limit(self, mock_auth_context):
        """Test intervention log respects limit."""
        # Create multiple interventions
        for i in range(10):
            await intervention.handle_inject_argument(
                "debate-123", mock_auth_context, f"Arg {i}", "argument", "user"
            )

        result = await intervention.handle_get_intervention_log(
            "debate-123", mock_auth_context, limit=5
        )

        data = json.loads(result.body)
        assert len(data["interventions"]) == 5
        assert data["total_interventions"] == 10


# ============================================================================
# Audit Trail Tests
# ============================================================================


class TestAuditTrail:
    """Tests for audit trail functionality."""

    @pytest.mark.asyncio
    async def test_pause_logs_intervention(self, mock_auth_context):
        """Test pause creates audit entry."""
        await intervention.handle_pause_debate("debate-123", mock_auth_context)

        assert len(intervention._intervention_log) == 1
        entry = intervention._intervention_log[0]
        assert entry["type"] == "pause"
        assert entry["user_id"] == mock_auth_context.user_id

    @pytest.mark.asyncio
    async def test_weight_change_logs_old_and_new(self, mock_auth_context):
        """Test weight change logs both values."""
        await intervention.handle_update_weights("debate-123", mock_auth_context, "claude", 1.5)

        entry = intervention._intervention_log[0]
        assert entry["type"] == "weight_change"
        assert entry["data"]["old_weight"] == 1.0
        assert entry["data"]["new_weight"] == 1.5
        assert entry["data"]["agent"] == "claude"

    @pytest.mark.asyncio
    async def test_inject_logs_content_preview(self, mock_auth_context):
        """Test injection logs content preview."""
        long_content = "A" * 200
        await intervention.handle_inject_argument(
            "debate-123", mock_auth_context, long_content, "argument", "user"
        )

        entry = intervention._intervention_log[0]
        assert entry["type"] == "inject_argument"
        assert len(entry["data"]["content_preview"]) == 100
        assert entry["data"]["full_length"] == 200


# ============================================================================
# Route Registration Tests
# ============================================================================


class TestRouteRegistration:
    """Tests for route registration."""

    def test_register_intervention_routes(self):
        """Test routes are registered correctly."""
        mock_router = Mock()

        intervention.register_intervention_routes(mock_router)

        # Verify all routes registered
        assert mock_router.add_route.call_count == 7

        # Check route paths
        calls = mock_router.add_route.call_args_list
        routes = [(call[0][0], call[0][1]) for call in calls]

        assert ("POST", "/api/debates/{debate_id}/intervention/pause") in routes
        assert ("POST", "/api/debates/{debate_id}/intervention/resume") in routes
        assert ("POST", "/api/debates/{debate_id}/intervention/inject") in routes
        assert ("POST", "/api/debates/{debate_id}/intervention/weights") in routes
        assert ("POST", "/api/debates/{debate_id}/intervention/threshold") in routes
        assert ("GET", "/api/debates/{debate_id}/intervention/state") in routes
        assert ("GET", "/api/debates/{debate_id}/intervention/log") in routes
