"""Tests for debate intervention handler.

Tests the intervention API endpoints including:
- POST /api/debates/{id}/intervention/pause - Pause active debate
- POST /api/debates/{id}/intervention/resume - Resume paused debate
- POST /api/debates/{id}/intervention/inject - Inject user argument
- POST /api/debates/{id}/intervention/weights - Update agent weights
- POST /api/debates/{id}/intervention/threshold - Update consensus threshold
- GET /api/debates/{id}/intervention/state - Get intervention state
- GET /api/debates/{id}/intervention/log - Get intervention log
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Import the module-level state for cleanup
from aragora.server.handlers.debates import intervention
from aragora.rbac.models import AuthorizationContext


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_ctx():
    """Provide a mock authorization context for handler calls."""
    return AuthorizationContext(
        user_id="test-user-001",
        user_email="test@example.com",
        org_id="test-org-001",
        roles={"admin", "owner"},
        permissions={"*"},
    )


@pytest.fixture(autouse=True)
def reset_intervention_state():
    """Reset intervention state before each test."""
    # Clear the module-level state
    intervention._debate_state.clear()
    intervention._intervention_log.clear()
    yield
    # Clean up after test
    intervention._debate_state.clear()
    intervention._intervention_log.clear()


# =============================================================================
# Pause/Resume Tests
# =============================================================================


class TestPauseResume:
    """Tests for debate pause/resume functionality."""

    @pytest.mark.asyncio
    async def test_pause_debate_success(self, mock_ctx):
        """Test pausing a debate successfully."""
        result = await intervention.handle_pause_debate("debate-123", mock_ctx)

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["success"] is True
        assert body["debate_id"] == "debate-123"
        assert body["is_paused"] is True
        assert "paused_at" in body

    @pytest.mark.asyncio
    async def test_pause_already_paused(self, mock_ctx):
        """Test pausing an already paused debate returns error."""
        # First pause
        await intervention.handle_pause_debate("debate-123", mock_ctx)

        # Try to pause again
        result = await intervention.handle_pause_debate("debate-123", mock_ctx)

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["success"] is False
        assert "already paused" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_resume_debate_success(self, mock_ctx):
        """Test resuming a paused debate successfully."""
        # First pause the debate
        await intervention.handle_pause_debate("debate-123", mock_ctx)

        # Now resume
        result = await intervention.handle_resume_debate("debate-123", mock_ctx)

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["success"] is True
        assert body["is_paused"] is False
        assert "resumed_at" in body

    @pytest.mark.asyncio
    async def test_resume_not_paused(self, mock_ctx):
        """Test resuming a non-paused debate returns error."""
        result = await intervention.handle_resume_debate("debate-123", mock_ctx)

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["success"] is False
        assert "not paused" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_pause_duration_calculated(self, mock_ctx):
        """Test that pause duration is calculated on resume."""
        # Pause the debate
        await intervention.handle_pause_debate("debate-123", mock_ctx)

        # Resume immediately
        result = await intervention.handle_resume_debate("debate-123", mock_ctx)

        import json

        body = json.loads(result.body)
        assert "pause_duration_seconds" in body
        # Duration should be very small (near 0)
        assert body["pause_duration_seconds"] is not None


# =============================================================================
# Inject Argument Tests
# =============================================================================


class TestInjectArgument:
    """Tests for argument injection functionality."""

    @pytest.mark.asyncio
    async def test_inject_argument_success(self, mock_ctx):
        """Test injecting an argument successfully."""
        result = await intervention.handle_inject_argument(
            debate_id="debate-123",
            content="This is an important point to consider",
            injection_type="argument",
            source="user",
            user_id="user-456",
            context=mock_ctx,
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["success"] is True
        assert body["debate_id"] == "debate-123"
        assert "injection_id" in body
        assert body["type"] == "argument"

    @pytest.mark.asyncio
    async def test_inject_follow_up(self, mock_ctx):
        """Test injecting a follow-up question."""
        result = await intervention.handle_inject_argument(
            debate_id="debate-123",
            content="Can you elaborate on this point?",
            injection_type="follow_up",
            source="user",
            context=mock_ctx,
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["success"] is True
        assert body["type"] == "follow_up"

    @pytest.mark.asyncio
    async def test_inject_empty_content(self, mock_ctx):
        """Test injecting empty content returns 400."""
        result = await intervention.handle_inject_argument(
            debate_id="debate-123",
            content="",
            injection_type="argument",
            source="user",
            context=mock_ctx,
        )

        assert result.status_code == 400
        import json

        body = json.loads(result.body)
        assert body["success"] is False
        assert "empty" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_inject_whitespace_content(self, mock_ctx):
        """Test injecting whitespace-only content returns 400."""
        result = await intervention.handle_inject_argument(
            debate_id="debate-123",
            content="   ",
            injection_type="argument",
            source="user",
            context=mock_ctx,
        )

        assert result.status_code == 400
        import json

        body = json.loads(result.body)
        assert body["success"] is False

    @pytest.mark.asyncio
    async def test_inject_logged_to_audit(self, mock_ctx):
        """Test that injection is logged to audit trail."""
        await intervention.handle_inject_argument(
            debate_id="debate-123",
            content="Test argument",
            injection_type="argument",
            source="user",
            user_id="user-456",
            context=mock_ctx,
        )

        # Check the intervention log
        logs = [log for log in intervention._intervention_log if log["debate_id"] == "debate-123"]
        assert len(logs) == 1
        assert logs[0]["type"] == "inject_argument"
        assert logs[0]["user_id"] == "user-456"


# =============================================================================
# Weight Update Tests
# =============================================================================


class TestWeightUpdate:
    """Tests for agent weight update functionality."""

    @pytest.mark.asyncio
    async def test_update_weight_success(self, mock_ctx):
        """Test updating agent weight successfully."""
        result = await intervention.handle_update_weights(
            debate_id="debate-123",
            agent="claude",
            weight=1.5,
            user_id="admin-789",
            context=mock_ctx,
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["success"] is True
        assert body["agent"] == "claude"
        assert body["new_weight"] == 1.5
        assert body["old_weight"] == 1.0  # Default weight

    @pytest.mark.asyncio
    async def test_update_weight_invalid_too_high(self, mock_ctx):
        """Test updating weight above 2.0 returns 400."""
        result = await intervention.handle_update_weights(
            debate_id="debate-123",
            agent="claude",
            weight=2.5,
            context=mock_ctx,
        )

        assert result.status_code == 400
        import json

        body = json.loads(result.body)
        assert body["success"] is False
        assert "between" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_update_weight_invalid_negative(self, mock_ctx):
        """Test updating weight below 0 returns 400."""
        result = await intervention.handle_update_weights(
            debate_id="debate-123",
            agent="claude",
            weight=-0.5,
            context=mock_ctx,
        )

        assert result.status_code == 400
        import json

        body = json.loads(result.body)
        assert body["success"] is False

    @pytest.mark.asyncio
    async def test_update_weight_muted(self, mock_ctx):
        """Test setting weight to 0 (muted) works."""
        result = await intervention.handle_update_weights(
            debate_id="debate-123",
            agent="claude",
            weight=0.0,
            context=mock_ctx,
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["success"] is True
        assert body["new_weight"] == 0.0

    @pytest.mark.asyncio
    async def test_update_weight_double_influence(self, mock_ctx):
        """Test setting weight to 2.0 (double influence) works."""
        result = await intervention.handle_update_weights(
            debate_id="debate-123",
            agent="claude",
            weight=2.0,
            context=mock_ctx,
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["success"] is True
        assert body["new_weight"] == 2.0


# =============================================================================
# Threshold Update Tests
# =============================================================================


class TestThresholdUpdate:
    """Tests for consensus threshold update functionality."""

    @pytest.mark.asyncio
    async def test_update_threshold_success(self, mock_ctx):
        """Test updating consensus threshold successfully."""
        result = await intervention.handle_update_threshold(
            debate_id="debate-123",
            threshold=0.8,
            user_id="admin-789",
            context=mock_ctx,
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["success"] is True
        assert body["new_threshold"] == 0.8
        assert body["old_threshold"] == 0.75  # Default threshold

    @pytest.mark.asyncio
    async def test_update_threshold_unanimous(self, mock_ctx):
        """Test setting threshold to 1.0 (unanimous) works."""
        result = await intervention.handle_update_threshold(
            debate_id="debate-123",
            threshold=1.0,
            context=mock_ctx,
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["success"] is True
        assert body["new_threshold"] == 1.0

    @pytest.mark.asyncio
    async def test_update_threshold_simple_majority(self, mock_ctx):
        """Test setting threshold to 0.5 (simple majority) works."""
        result = await intervention.handle_update_threshold(
            debate_id="debate-123",
            threshold=0.5,
            context=mock_ctx,
        )

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["success"] is True
        assert body["new_threshold"] == 0.5

    @pytest.mark.asyncio
    async def test_update_threshold_invalid_too_low(self, mock_ctx):
        """Test threshold below 0.5 returns 400."""
        result = await intervention.handle_update_threshold(
            debate_id="debate-123",
            threshold=0.4,
            context=mock_ctx,
        )

        assert result.status_code == 400
        import json

        body = json.loads(result.body)
        assert body["success"] is False
        assert "between" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_update_threshold_invalid_too_high(self, mock_ctx):
        """Test threshold above 1.0 returns 400."""
        result = await intervention.handle_update_threshold(
            debate_id="debate-123",
            threshold=1.5,
            context=mock_ctx,
        )

        assert result.status_code == 400
        import json

        body = json.loads(result.body)
        assert body["success"] is False


# =============================================================================
# Intervention State Tests
# =============================================================================


class TestInterventionState:
    """Tests for intervention state retrieval."""

    @pytest.mark.asyncio
    async def test_get_state_new_debate(self, mock_ctx):
        """Test getting state for a new debate returns defaults."""
        result = await intervention.handle_get_intervention_state("debate-new", mock_ctx)

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["debate_id"] == "debate-new"
        assert body["is_paused"] is False
        assert body["consensus_threshold"] == 0.75
        assert body["agent_weights"] == {}
        assert body["pending_injections"] == 0
        assert body["pending_follow_ups"] == 0

    @pytest.mark.asyncio
    async def test_get_state_after_interventions(self, mock_ctx):
        """Test getting state after making interventions."""
        # Make some interventions
        await intervention.handle_pause_debate("debate-123", mock_ctx)
        await intervention.handle_update_weights(
            debate_id="debate-123", agent="claude", weight=1.5, context=mock_ctx
        )
        await intervention.handle_update_threshold(
            debate_id="debate-123", threshold=0.9, context=mock_ctx
        )
        await intervention.handle_inject_argument(
            debate_id="debate-123",
            content="Test",
            injection_type="argument",
            source="user",
            context=mock_ctx,
        )

        result = await intervention.handle_get_intervention_state("debate-123", mock_ctx)

        import json

        body = json.loads(result.body)
        assert body["is_paused"] is True
        assert body["consensus_threshold"] == 0.9
        assert body["agent_weights"]["claude"] == 1.5
        assert body["pending_injections"] == 1

    @pytest.mark.asyncio
    async def test_get_intervention_log(self, mock_ctx):
        """Test getting intervention log entries."""
        # Make some interventions
        await intervention.handle_pause_debate("debate-123", mock_ctx)
        await intervention.handle_resume_debate("debate-123", mock_ctx)
        await intervention.handle_inject_argument(
            debate_id="debate-123",
            content="Test",
            injection_type="argument",
            source="user",
            context=mock_ctx,
        )

        result = await intervention.handle_get_intervention_log("debate-123", context=mock_ctx)

        assert result.status_code == 200
        import json

        body = json.loads(result.body)
        assert body["debate_id"] == "debate-123"
        assert body["total_interventions"] == 3
        assert len(body["interventions"]) == 3

    @pytest.mark.asyncio
    async def test_get_intervention_log_limit(self, mock_ctx):
        """Test intervention log respects limit parameter."""
        # Make several interventions
        for i in range(10):
            await intervention.handle_inject_argument(
                debate_id="debate-123",
                content=f"Test {i}",
                injection_type="argument",
                source="user",
                context=mock_ctx,
            )

        result = await intervention.handle_get_intervention_log(
            "debate-123", limit=5, context=mock_ctx
        )

        import json

        body = json.loads(result.body)
        assert body["total_interventions"] == 10
        assert len(body["interventions"]) == 5

    @pytest.mark.asyncio
    async def test_get_intervention_log_sorted(self, mock_ctx):
        """Test intervention log is sorted by timestamp descending."""
        await intervention.handle_pause_debate("debate-123", mock_ctx)
        await intervention.handle_resume_debate("debate-123", mock_ctx)

        result = await intervention.handle_get_intervention_log("debate-123", context=mock_ctx)

        import json

        body = json.loads(result.body)
        interventions = body["interventions"]

        # Should be sorted descending by timestamp
        assert interventions[0]["type"] == "resume"  # Most recent
        assert interventions[1]["type"] == "pause"


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_debate_state_creates_default(self):
        """Test get_debate_state creates default state for new debate."""
        state = intervention.get_debate_state("new-debate")

        assert state["is_paused"] is False
        assert state["agent_weights"] == {}
        assert state["consensus_threshold"] == 0.75
        assert state["injected_arguments"] == []
        assert state["follow_up_questions"] == []

    def test_get_debate_state_returns_existing(self):
        """Test get_debate_state returns existing state."""
        # Create state
        state1 = intervention.get_debate_state("debate-123")
        state1["is_paused"] = True

        # Get it again
        state2 = intervention.get_debate_state("debate-123")

        assert state2["is_paused"] is True

    def test_log_intervention(self):
        """Test log_intervention creates audit entry."""
        intervention.log_intervention(
            debate_id="debate-123",
            intervention_type="test",
            data={"key": "value"},
            user_id="user-456",
        )

        assert len(intervention._intervention_log) == 1
        entry = intervention._intervention_log[0]
        assert entry["debate_id"] == "debate-123"
        assert entry["type"] == "test"
        assert entry["data"]["key"] == "value"
        assert entry["user_id"] == "user-456"
        assert "timestamp" in entry
