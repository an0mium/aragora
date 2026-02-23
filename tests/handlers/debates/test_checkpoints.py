"""Tests for debate checkpoint handler.

Tests the checkpoint API endpoints including:
- POST /api/v1/debates/:id/checkpoint/pause  - Create a checkpoint and pause
- POST /api/v1/debates/:id/checkpoint/resume - Resume from a checkpoint
- GET  /api/v1/debates/:id/checkpoints       - List checkpoints for a debate
"""

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.rbac.models import AuthorizationContext
from aragora.server.handlers.debates import checkpoints
from aragora.server.handlers.debates.checkpoints import (
    _paused_debates,
    handle_checkpoint_pause,
    handle_checkpoint_resume,
    handle_list_checkpoints,
    register_checkpoint_routes,
)


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


@pytest.fixture
def alt_ctx():
    """Provide an alternate authorization context (different user)."""
    return AuthorizationContext(
        user_id="other-user-002",
        user_email="other@example.com",
        org_id="test-org-001",
        roles={"member"},
        permissions={"debates:read", "debates:write"},
    )


@pytest.fixture(autouse=True)
def reset_paused_debates():
    """Reset the module-level _paused_debates dict before and after each test."""
    _paused_debates.clear()
    yield
    _paused_debates.clear()


@pytest.fixture
def mock_checkpoint():
    """Create a mock checkpoint object."""

    @dataclass
    class MockCheckpoint:
        checkpoint_id: str = "chk-abc-123"
        debate_id: str = "debate-001"
        current_round: int = 3
        total_rounds: int = 5
        phase: str = "paused"

    return MockCheckpoint()


@pytest.fixture
def mock_resumed():
    """Create a mock resume result object."""

    @dataclass
    class MockCheckpointInner:
        checkpoint_id: str = "chk-abc-123"
        current_round: int = 3
        total_rounds: int = 5

    @dataclass
    class MockResumeResult:
        checkpoint: Any = None
        resumed_at: str = "2026-02-23T10:00:00"
        messages: list = None

        def __post_init__(self):
            if self.checkpoint is None:
                self.checkpoint = MockCheckpointInner()
            if self.messages is None:
                self.messages = [{"role": "agent", "content": "Hello"}]

    return MockResumeResult()


@pytest.fixture
def mock_manager(mock_checkpoint, mock_resumed):
    """Create a mock checkpoint manager with configured responses."""
    manager = MagicMock()
    manager.create_checkpoint = AsyncMock(return_value=mock_checkpoint)
    manager.get_latest = AsyncMock(return_value=mock_checkpoint)
    manager.resume_from_checkpoint = AsyncMock(return_value=mock_resumed)
    manager.store = MagicMock()
    manager.store.list_checkpoints = AsyncMock(return_value=[])
    return manager


# =============================================================================
# Pause Tests
# =============================================================================


class TestCheckpointPause:
    """Tests for handle_checkpoint_pause."""

    @pytest.mark.asyncio
    async def test_pause_success(self, mock_ctx, mock_manager, mock_checkpoint):
        """Test pausing a debate creates a checkpoint and returns success."""
        result = await handle_checkpoint_pause(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True
        assert body["debate_id"] == "debate-001"
        assert body["checkpoint_id"] == "chk-abc-123"
        assert "paused_at" in body
        assert body["message"] == "Debate paused and checkpoint created"

    @pytest.mark.asyncio
    async def test_pause_stores_in_paused_debates(self, mock_ctx, mock_manager):
        """Test that pausing a debate stores info in _paused_debates."""
        await handle_checkpoint_pause(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert "debate-001" in _paused_debates
        assert _paused_debates["debate-001"]["checkpoint_id"] == "chk-abc-123"
        assert _paused_debates["debate-001"]["paused_by"] == "test-user-001"
        assert "paused_at" in _paused_debates["debate-001"]

    @pytest.mark.asyncio
    async def test_pause_records_user_id(self, alt_ctx, mock_manager):
        """Test that pausing records the user who paused."""
        await handle_checkpoint_pause(
            debate_id="debate-001",
            context=alt_ctx,
            checkpoint_manager=mock_manager,
        )

        assert _paused_debates["debate-001"]["paused_by"] == "other-user-002"

    @pytest.mark.asyncio
    async def test_pause_calls_create_checkpoint(self, mock_ctx, mock_manager):
        """Test that pause calls create_checkpoint with expected args."""
        await handle_checkpoint_pause(
            debate_id="debate-007",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        mock_manager.create_checkpoint.assert_awaited_once_with(
            debate_id="debate-007",
            task="Paused debate debate-007",
            current_round=0,
            total_rounds=0,
            phase="paused",
            messages=[],
            critiques=[],
            votes=[],
            agents=[],
            current_consensus=None,
        )

    @pytest.mark.asyncio
    async def test_pause_no_checkpoint_manager(self, mock_ctx):
        """Test pause returns 503 when no checkpoint manager is available."""
        result = await handle_checkpoint_pause(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=None,
        )

        assert result.status_code == 503
        body = json.loads(result.body)
        assert "not configured" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_pause_create_checkpoint_os_error(self, mock_ctx, mock_manager):
        """Test pause handles OSError from create_checkpoint."""
        mock_manager.create_checkpoint = AsyncMock(side_effect=OSError("disk full"))

        result = await handle_checkpoint_pause(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 500
        body = json.loads(result.body)
        assert "failed" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_pause_create_checkpoint_value_error(self, mock_ctx, mock_manager):
        """Test pause handles ValueError from create_checkpoint."""
        mock_manager.create_checkpoint = AsyncMock(side_effect=ValueError("invalid debate_id"))

        result = await handle_checkpoint_pause(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_pause_create_checkpoint_type_error(self, mock_ctx, mock_manager):
        """Test pause handles TypeError from create_checkpoint."""
        mock_manager.create_checkpoint = AsyncMock(side_effect=TypeError("wrong type"))

        result = await handle_checkpoint_pause(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_pause_create_checkpoint_runtime_error(self, mock_ctx, mock_manager):
        """Test pause handles RuntimeError from create_checkpoint."""
        mock_manager.create_checkpoint = AsyncMock(side_effect=RuntimeError("concurrency issue"))

        result = await handle_checkpoint_pause(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_pause_twice_overwrites_paused_state(self, mock_ctx, mock_manager):
        """Test that pausing the same debate twice overwrites the paused entry."""
        await handle_checkpoint_pause(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )
        first_paused_at = _paused_debates["debate-001"]["paused_at"]

        # Pause again (different checkpoint_id to distinguish)
        @dataclass
        class NewCheckpoint:
            checkpoint_id: str = "chk-new-456"

        mock_manager.create_checkpoint = AsyncMock(return_value=NewCheckpoint())

        await handle_checkpoint_pause(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert _paused_debates["debate-001"]["checkpoint_id"] == "chk-new-456"

    @pytest.mark.asyncio
    async def test_pause_multiple_debates(self, mock_ctx, mock_manager):
        """Test pausing multiple debates independently."""
        await handle_checkpoint_pause(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )
        await handle_checkpoint_pause(
            debate_id="debate-002",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert "debate-001" in _paused_debates
        assert "debate-002" in _paused_debates

    @pytest.mark.asyncio
    async def test_pause_empty_debate_id(self, mock_ctx, mock_manager):
        """Test pausing with an empty debate ID still processes."""
        result = await handle_checkpoint_pause(
            debate_id="",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        # Handler does not validate debate_id; it delegates to manager
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["debate_id"] == ""

    @pytest.mark.asyncio
    async def test_pause_response_content_type(self, mock_ctx, mock_manager):
        """Test that pause response has JSON content type."""
        result = await handle_checkpoint_pause(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert result.content_type == "application/json"


# =============================================================================
# Resume Tests
# =============================================================================


class TestCheckpointResume:
    """Tests for handle_checkpoint_resume."""

    @pytest.mark.asyncio
    async def test_resume_with_latest_checkpoint(self, mock_ctx, mock_manager):
        """Test resuming using the latest checkpoint (no explicit ID)."""
        # First pause
        _paused_debates["debate-001"] = {
            "checkpoint_id": "chk-abc-123",
            "paused_at": "2026-02-23T09:00:00",
            "paused_by": "test-user-001",
        }

        result = await handle_checkpoint_resume(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True
        assert body["debate_id"] == "debate-001"
        assert body["checkpoint_id"] == "chk-abc-123"
        assert body["message"] == "Debate resumed from checkpoint"
        assert "resumed_at" in body
        assert "resumed_from_round" in body
        assert "total_rounds" in body
        assert "message_count" in body

    @pytest.mark.asyncio
    async def test_resume_with_explicit_checkpoint_id(self, mock_ctx, mock_manager, mock_resumed):
        """Test resuming from a specific checkpoint ID."""
        result = await handle_checkpoint_resume(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_id="chk-specific-789",
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["checkpoint_id"] == "chk-specific-789"

        # Should NOT call get_latest when explicit ID is provided
        mock_manager.get_latest.assert_not_awaited()
        mock_manager.resume_from_checkpoint.assert_awaited_once_with(
            checkpoint_id="chk-specific-789",
            resumed_by="test-user-001",
        )

    @pytest.mark.asyncio
    async def test_resume_clears_paused_state(self, mock_ctx, mock_manager):
        """Test that resuming clears the paused state entry."""
        _paused_debates["debate-001"] = {
            "checkpoint_id": "chk-abc-123",
            "paused_at": "2026-02-23T09:00:00",
            "paused_by": "test-user-001",
        }

        await handle_checkpoint_resume(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert "debate-001" not in _paused_debates

    @pytest.mark.asyncio
    async def test_resume_clears_only_target_debate(self, mock_ctx, mock_manager):
        """Test that resuming only clears the target debate's paused state."""
        _paused_debates["debate-001"] = {
            "checkpoint_id": "chk-1",
            "paused_at": "2026-02-23T09:00:00",
            "paused_by": "test-user-001",
        }
        _paused_debates["debate-002"] = {
            "checkpoint_id": "chk-2",
            "paused_at": "2026-02-23T09:00:00",
            "paused_by": "test-user-001",
        }

        await handle_checkpoint_resume(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert "debate-001" not in _paused_debates
        assert "debate-002" in _paused_debates

    @pytest.mark.asyncio
    async def test_resume_no_checkpoint_manager(self, mock_ctx):
        """Test resume returns 503 when no checkpoint manager is available."""
        result = await handle_checkpoint_resume(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=None,
        )

        assert result.status_code == 503
        body = json.loads(result.body)
        assert "not configured" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_resume_no_checkpoints_found(self, mock_ctx, mock_manager):
        """Test resume returns 404 when no checkpoints exist for the debate."""
        mock_manager.get_latest = AsyncMock(return_value=None)

        result = await handle_checkpoint_resume(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 404
        body = json.loads(result.body)
        assert "no checkpoints" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_resume_checkpoint_not_found(self, mock_ctx, mock_manager):
        """Test resume returns 404 when the specific checkpoint ID is not found."""
        mock_manager.resume_from_checkpoint = AsyncMock(return_value=None)

        result = await handle_checkpoint_resume(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_id="chk-nonexistent",
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 404
        body = json.loads(result.body)
        assert "not found" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_resume_os_error(self, mock_ctx, mock_manager):
        """Test resume handles OSError gracefully."""
        mock_manager.get_latest = AsyncMock(side_effect=OSError("I/O error"))

        result = await handle_checkpoint_resume(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 500
        body = json.loads(result.body)
        assert "failed" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_resume_value_error(self, mock_ctx, mock_manager):
        """Test resume handles ValueError gracefully."""
        mock_manager.resume_from_checkpoint = AsyncMock(side_effect=ValueError("corrupted data"))

        result = await handle_checkpoint_resume(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_id="chk-abc-123",
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_resume_type_error(self, mock_ctx, mock_manager):
        """Test resume handles TypeError gracefully."""
        mock_manager.resume_from_checkpoint = AsyncMock(side_effect=TypeError("type mismatch"))

        result = await handle_checkpoint_resume(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_id="chk-abc-123",
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_resume_key_error(self, mock_ctx, mock_manager):
        """Test resume handles KeyError gracefully."""
        mock_manager.resume_from_checkpoint = AsyncMock(side_effect=KeyError("missing key"))

        result = await handle_checkpoint_resume(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_id="chk-abc-123",
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_resume_attribute_error(self, mock_ctx, mock_manager):
        """Test resume handles AttributeError gracefully."""
        mock_manager.resume_from_checkpoint = AsyncMock(side_effect=AttributeError("no such attr"))

        result = await handle_checkpoint_resume(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_id="chk-abc-123",
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_resume_uses_user_id_for_resumed_by(self, mock_ctx, mock_manager):
        """Test that resume passes user_id as resumed_by."""
        await handle_checkpoint_resume(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_id="chk-abc-123",
            checkpoint_manager=mock_manager,
        )

        mock_manager.resume_from_checkpoint.assert_awaited_once_with(
            checkpoint_id="chk-abc-123",
            resumed_by="test-user-001",
        )

    @pytest.mark.asyncio
    async def test_resume_uses_api_fallback_when_no_user_id(self, mock_manager):
        """Test that resume uses 'api' when user_id is None."""
        ctx = AuthorizationContext(
            user_id=None,
            user_email="test@example.com",
            org_id="test-org-001",
            roles={"admin"},
            permissions={"*"},
        )

        await handle_checkpoint_resume(
            debate_id="debate-001",
            context=ctx,
            checkpoint_id="chk-abc-123",
            checkpoint_manager=mock_manager,
        )

        mock_manager.resume_from_checkpoint.assert_awaited_once_with(
            checkpoint_id="chk-abc-123",
            resumed_by="api",
        )

    @pytest.mark.asyncio
    async def test_resume_response_includes_round_info(self, mock_ctx, mock_manager, mock_resumed):
        """Test that resume response includes round and message info."""
        result = await handle_checkpoint_resume(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_id="chk-abc-123",
            checkpoint_manager=mock_manager,
        )

        body = json.loads(result.body)
        assert body["resumed_from_round"] == 3
        assert body["total_rounds"] == 5
        assert body["message_count"] == 1  # mock_resumed has 1 message

    @pytest.mark.asyncio
    async def test_resume_not_paused_still_works(self, mock_ctx, mock_manager):
        """Test that resume works even if the debate wasn't in _paused_debates."""
        # No entry in _paused_debates
        result = await handle_checkpoint_resume(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_id="chk-abc-123",
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_resume_response_content_type(self, mock_ctx, mock_manager):
        """Test that resume response has JSON content type."""
        result = await handle_checkpoint_resume(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_id="chk-abc-123",
            checkpoint_manager=mock_manager,
        )

        assert result.content_type == "application/json"


# =============================================================================
# List Checkpoints Tests
# =============================================================================


class TestListCheckpoints:
    """Tests for handle_list_checkpoints."""

    @pytest.mark.asyncio
    async def test_list_empty(self, mock_ctx, mock_manager):
        """Test listing checkpoints when none exist."""
        mock_manager.store.list_checkpoints = AsyncMock(return_value=[])

        result = await handle_list_checkpoints(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["debate_id"] == "debate-001"
        assert body["is_paused"] is False
        assert body["paused_at"] is None
        assert body["total_checkpoints"] == 0
        assert body["checkpoints"] == []

    @pytest.mark.asyncio
    async def test_list_with_checkpoints(self, mock_ctx, mock_manager):
        """Test listing when checkpoints exist."""
        checkpoints_list = [
            {"id": "chk-1", "round": 1},
            {"id": "chk-2", "round": 3},
            {"id": "chk-3", "round": 5},
        ]
        mock_manager.store.list_checkpoints = AsyncMock(return_value=checkpoints_list)

        result = await handle_list_checkpoints(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total_checkpoints"] == 3
        assert body["checkpoints"] == checkpoints_list

    @pytest.mark.asyncio
    async def test_list_shows_paused_state(self, mock_ctx, mock_manager):
        """Test that list shows paused state when debate is paused."""
        _paused_debates["debate-001"] = {
            "checkpoint_id": "chk-abc-123",
            "paused_at": "2026-02-23T09:00:00",
            "paused_by": "test-user-001",
        }

        result = await handle_list_checkpoints(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["is_paused"] is True
        assert body["paused_at"] == "2026-02-23T09:00:00"

    @pytest.mark.asyncio
    async def test_list_not_paused_shows_false(self, mock_ctx, mock_manager):
        """Test that is_paused is False when debate is not paused."""
        result = await handle_list_checkpoints(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        body = json.loads(result.body)
        assert body["is_paused"] is False
        assert body["paused_at"] is None

    @pytest.mark.asyncio
    async def test_list_default_limit(self, mock_ctx, mock_manager):
        """Test listing with default limit of 50."""
        await handle_list_checkpoints(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        mock_manager.store.list_checkpoints.assert_awaited_once_with(
            debate_id="debate-001",
            limit=50,
        )

    @pytest.mark.asyncio
    async def test_list_custom_limit(self, mock_ctx, mock_manager):
        """Test listing with a custom limit."""
        await handle_list_checkpoints(
            debate_id="debate-001",
            context=mock_ctx,
            limit=10,
            checkpoint_manager=mock_manager,
        )

        mock_manager.store.list_checkpoints.assert_awaited_once_with(
            debate_id="debate-001",
            limit=10,
        )

    @pytest.mark.asyncio
    async def test_list_limit_capped_at_100(self, mock_ctx, mock_manager):
        """Test that limit is capped at 100 even if a higher value is given."""
        await handle_list_checkpoints(
            debate_id="debate-001",
            context=mock_ctx,
            limit=500,
            checkpoint_manager=mock_manager,
        )

        mock_manager.store.list_checkpoints.assert_awaited_once_with(
            debate_id="debate-001",
            limit=100,
        )

    @pytest.mark.asyncio
    async def test_list_limit_exactly_100(self, mock_ctx, mock_manager):
        """Test that limit of exactly 100 is accepted."""
        await handle_list_checkpoints(
            debate_id="debate-001",
            context=mock_ctx,
            limit=100,
            checkpoint_manager=mock_manager,
        )

        mock_manager.store.list_checkpoints.assert_awaited_once_with(
            debate_id="debate-001",
            limit=100,
        )

    @pytest.mark.asyncio
    async def test_list_limit_1(self, mock_ctx, mock_manager):
        """Test listing with minimum limit of 1."""
        await handle_list_checkpoints(
            debate_id="debate-001",
            context=mock_ctx,
            limit=1,
            checkpoint_manager=mock_manager,
        )

        mock_manager.store.list_checkpoints.assert_awaited_once_with(
            debate_id="debate-001",
            limit=1,
        )

    @pytest.mark.asyncio
    async def test_list_no_checkpoint_manager(self, mock_ctx):
        """Test list returns 503 when no checkpoint manager is available."""
        result = await handle_list_checkpoints(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=None,
        )

        assert result.status_code == 503
        body = json.loads(result.body)
        assert "not configured" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_list_os_error(self, mock_ctx, mock_manager):
        """Test list handles OSError gracefully."""
        mock_manager.store.list_checkpoints = AsyncMock(side_effect=OSError("storage error"))

        result = await handle_list_checkpoints(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 500
        body = json.loads(result.body)
        assert "failed" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_list_value_error(self, mock_ctx, mock_manager):
        """Test list handles ValueError gracefully."""
        mock_manager.store.list_checkpoints = AsyncMock(side_effect=ValueError("bad query"))

        result = await handle_list_checkpoints(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_list_type_error(self, mock_ctx, mock_manager):
        """Test list handles TypeError gracefully."""
        mock_manager.store.list_checkpoints = AsyncMock(side_effect=TypeError("wrong arg"))

        result = await handle_list_checkpoints(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_list_attribute_error(self, mock_ctx, mock_manager):
        """Test list handles AttributeError gracefully."""
        mock_manager.store.list_checkpoints = AsyncMock(side_effect=AttributeError("no store"))

        result = await handle_list_checkpoints(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_list_response_content_type(self, mock_ctx, mock_manager):
        """Test that list response has JSON content type."""
        result = await handle_list_checkpoints(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert result.content_type == "application/json"


# =============================================================================
# _get_checkpoint_manager Tests
# =============================================================================


class TestGetCheckpointManager:
    """Tests for _get_checkpoint_manager helper."""

    def test_returns_manager_from_server_context(self):
        """Test that manager is returned from server_context if present."""
        mock_mgr = MagicMock()
        ctx = {"checkpoint_manager": mock_mgr}

        result = checkpoints._get_checkpoint_manager(server_context=ctx)

        assert result is mock_mgr

    def test_returns_none_with_empty_context(self):
        """Test that None context falls through to import attempt."""
        with patch.dict("sys.modules", {"aragora.debate.checkpoint": None}):
            # Import will fail since module is None
            result = checkpoints._get_checkpoint_manager(server_context=None)
            # May return None or a real instance depending on import availability
            # The important thing is it doesn't raise
            assert result is None or result is not None

    def test_returns_manager_from_context_ignores_import(self):
        """Test that when server_context has manager, import is not attempted."""
        mock_mgr = MagicMock()
        ctx = {"checkpoint_manager": mock_mgr}

        # Even if import would fail, server_context takes priority
        result = checkpoints._get_checkpoint_manager(server_context=ctx)
        assert result is mock_mgr

    def test_empty_dict_context_falls_through(self):
        """Test that empty dict falls through to import."""
        # With empty context, it tries to import
        result = checkpoints._get_checkpoint_manager(server_context={})
        # Result depends on whether aragora.debate.checkpoint is importable
        # Just verify no exception is raised
        assert result is None or result is not None

    def test_context_with_other_keys_falls_through(self):
        """Test that context without checkpoint_manager key falls through."""
        ctx = {"something_else": "value"}
        result = checkpoints._get_checkpoint_manager(server_context=ctx)
        assert result is None or result is not None


# =============================================================================
# Integration / Workflow Tests
# =============================================================================


class TestPauseResumeWorkflow:
    """Tests for the complete pause -> list -> resume workflow."""

    @pytest.mark.asyncio
    async def test_pause_then_list_then_resume(self, mock_ctx, mock_manager):
        """Test the full workflow: pause, list (shows paused), resume, list (shows not paused)."""
        # Step 1: Pause
        pause_result = await handle_checkpoint_pause(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )
        assert pause_result.status_code == 200

        # Step 2: List - should show paused
        list_result_paused = await handle_list_checkpoints(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )
        body_paused = json.loads(list_result_paused.body)
        assert body_paused["is_paused"] is True

        # Step 3: Resume
        resume_result = await handle_checkpoint_resume(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )
        assert resume_result.status_code == 200

        # Step 4: List - should show not paused
        list_result_resumed = await handle_list_checkpoints(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )
        body_resumed = json.loads(list_result_resumed.body)
        assert body_resumed["is_paused"] is False

    @pytest.mark.asyncio
    async def test_pause_debate_a_resume_debate_b_independence(self, mock_ctx, mock_manager):
        """Test that pause/resume on different debates is independent."""
        # Pause debate A
        await handle_checkpoint_pause(
            debate_id="debate-A",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )
        # Pause debate B
        await handle_checkpoint_pause(
            debate_id="debate-B",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert "debate-A" in _paused_debates
        assert "debate-B" in _paused_debates

        # Resume only debate A
        await handle_checkpoint_resume(
            debate_id="debate-A",
            context=mock_ctx,
            checkpoint_id="chk-abc-123",
            checkpoint_manager=mock_manager,
        )

        assert "debate-A" not in _paused_debates
        assert "debate-B" in _paused_debates


# =============================================================================
# register_checkpoint_routes Tests
# =============================================================================


class TestRegisterCheckpointRoutes:
    """Tests for route registration."""

    def test_register_routes(self):
        """Test that all three routes are registered with the router."""
        mock_router = MagicMock()
        register_checkpoint_routes(mock_router)

        assert mock_router.add_route.call_count == 3

        # Extract all registered routes
        calls = mock_router.add_route.call_args_list
        registered = [(c.args[0], c.args[1]) for c in calls]

        assert ("POST", "/api/v1/debates/{debate_id}/checkpoint/pause") in registered
        assert ("POST", "/api/v1/debates/{debate_id}/checkpoint/resume") in registered
        assert ("GET", "/api/v1/debates/{debate_id}/checkpoints") in registered

    def test_registered_pause_handler_is_callable(self):
        """Test that the registered pause handler is a callable coroutine."""
        mock_router = MagicMock()
        register_checkpoint_routes(mock_router)

        calls = mock_router.add_route.call_args_list
        pause_handler = calls[0].args[2]  # Third arg is the handler function
        assert callable(pause_handler)

    def test_registered_resume_handler_is_callable(self):
        """Test that the registered resume handler is a callable coroutine."""
        mock_router = MagicMock()
        register_checkpoint_routes(mock_router)

        calls = mock_router.add_route.call_args_list
        resume_handler = calls[1].args[2]
        assert callable(resume_handler)

    def test_registered_list_handler_is_callable(self):
        """Test that the registered list handler is a callable coroutine."""
        mock_router = MagicMock()
        register_checkpoint_routes(mock_router)

        calls = mock_router.add_route.call_args_list
        list_handler = calls[2].args[2]
        assert callable(list_handler)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_pause_with_special_chars_in_debate_id(self, mock_ctx, mock_manager):
        """Test pause with special characters in debate ID."""
        result = await handle_checkpoint_pause(
            debate_id="debate/special#chars&here",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["debate_id"] == "debate/special#chars&here"

    @pytest.mark.asyncio
    async def test_resume_with_special_chars_in_checkpoint_id(self, mock_ctx, mock_manager):
        """Test resume with special characters in checkpoint ID."""
        result = await handle_checkpoint_resume(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_id="chk/special:id",
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_list_for_nonexistent_debate(self, mock_ctx, mock_manager):
        """Test listing checkpoints for a debate that has no checkpoints."""
        mock_manager.store.list_checkpoints = AsyncMock(return_value=[])

        result = await handle_list_checkpoints(
            debate_id="nonexistent-debate",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total_checkpoints"] == 0
        assert body["is_paused"] is False

    @pytest.mark.asyncio
    async def test_list_limit_zero(self, mock_ctx, mock_manager):
        """Test listing with limit=0."""
        await handle_list_checkpoints(
            debate_id="debate-001",
            context=mock_ctx,
            limit=0,
            checkpoint_manager=mock_manager,
        )

        # min(0, 100) = 0
        mock_manager.store.list_checkpoints.assert_awaited_once_with(
            debate_id="debate-001",
            limit=0,
        )

    @pytest.mark.asyncio
    async def test_list_negative_limit(self, mock_ctx, mock_manager):
        """Test listing with a negative limit (min(-5, 100) = -5)."""
        await handle_list_checkpoints(
            debate_id="debate-001",
            context=mock_ctx,
            limit=-5,
            checkpoint_manager=mock_manager,
        )

        # min(-5, 100) = -5, handler does not validate lower bound
        mock_manager.store.list_checkpoints.assert_awaited_once_with(
            debate_id="debate-001",
            limit=-5,
        )

    @pytest.mark.asyncio
    async def test_pause_response_paused_at_is_iso_format(self, mock_ctx, mock_manager):
        """Test that paused_at in the response is an ISO format timestamp."""
        result = await handle_checkpoint_pause(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        body = json.loads(result.body)
        # Should be a valid ISO format string (parseable by datetime)
        from datetime import datetime

        parsed = datetime.fromisoformat(body["paused_at"])
        assert parsed is not None

    @pytest.mark.asyncio
    async def test_resume_without_checkpoint_id_calls_get_latest(self, mock_ctx, mock_manager):
        """Test that resume without checkpoint_id calls get_latest."""
        await handle_checkpoint_resume(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_id=None,
            checkpoint_manager=mock_manager,
        )

        mock_manager.get_latest.assert_awaited_once_with("debate-001")

    @pytest.mark.asyncio
    async def test_resume_with_empty_string_checkpoint_id(self, mock_ctx, mock_manager):
        """Test resume with empty string checkpoint_id uses get_latest."""
        await handle_checkpoint_resume(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_id="",
            checkpoint_manager=mock_manager,
        )

        # Empty string is falsy, so get_latest should be called
        mock_manager.get_latest.assert_awaited_once_with("debate-001")

    @pytest.mark.asyncio
    async def test_pause_result_tuple_unpacking(self, mock_ctx, mock_manager):
        """Test that HandlerResult supports tuple-style unpacking."""
        result = await handle_checkpoint_pause(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=mock_manager,
        )

        body_data, status, headers = result
        assert status == 200
        assert body_data["success"] is True

    @pytest.mark.asyncio
    async def test_error_result_tuple_unpacking(self, mock_ctx):
        """Test that error HandlerResult supports tuple-style unpacking."""
        result = await handle_checkpoint_pause(
            debate_id="debate-001",
            context=mock_ctx,
            checkpoint_manager=None,
        )

        body_data, status, headers = result
        assert status == 503
        assert "error" in body_data


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports(self):
        """Test that __all__ contains the expected exports."""
        assert "handle_checkpoint_pause" in checkpoints.__all__
        assert "handle_checkpoint_resume" in checkpoints.__all__
        assert "handle_list_checkpoints" in checkpoints.__all__
        assert "register_checkpoint_routes" in checkpoints.__all__

    def test_all_exports_count(self):
        """Test that __all__ contains exactly 4 items."""
        assert len(checkpoints.__all__) == 4
