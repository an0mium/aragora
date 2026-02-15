"""
Tests for debate checkpoint HTTP endpoints.

Tests the handlers in aragora/server/handlers/debates/checkpoints.py:
- POST /api/v1/debates/:id/checkpoint/pause
- POST /api/v1/debates/:id/checkpoint/resume
- GET  /api/v1/debates/:id/checkpoints
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.rbac.models import AuthorizationContext
from aragora.server.handlers.base import HandlerResult

# Import handlers under test
from aragora.server.handlers.debates.checkpoints import (
    _paused_debates,
    handle_checkpoint_pause,
    handle_checkpoint_resume,
    handle_list_checkpoints,
)


def _body(result) -> dict:
    """Parse JSON body from HandlerResult."""
    raw = result["body"]
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, bytes):
        return json.loads(raw.decode("utf-8"))
    return raw


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_paused_state():
    """Clear paused debate state between tests."""
    _paused_debates.clear()
    yield
    _paused_debates.clear()


@pytest.fixture
def auth_context():
    """Create a mock authorization context."""
    ctx = MagicMock(spec=AuthorizationContext)
    ctx.user_id = "user-123"
    ctx.org_id = "org-456"
    ctx.permissions = {"debates:write", "debates:read"}
    return ctx


@pytest.fixture
def mock_checkpoint():
    """Create a mock checkpoint object."""
    cp = MagicMock()
    cp.checkpoint_id = "cp-debate1-001-abcd"
    cp.debate_id = "debate-1"
    cp.current_round = 3
    cp.total_rounds = 5
    cp.phase = "revision"
    cp.created_at = "2026-02-14T12:00:00"
    cp.status = MagicMock()
    cp.status.value = "complete"
    return cp


@pytest.fixture
def mock_checkpoint_manager(mock_checkpoint):
    """Create a mock CheckpointManager."""
    manager = MagicMock()
    manager.create_checkpoint = AsyncMock(return_value=mock_checkpoint)
    manager.resume_from_checkpoint = AsyncMock()
    manager.get_latest = AsyncMock()
    manager.store = MagicMock()
    manager.store.list_checkpoints = AsyncMock(return_value=[])
    return manager


# ---------------------------------------------------------------------------
# Pause Endpoint Tests
# ---------------------------------------------------------------------------


class TestCheckpointPause:
    """Tests for handle_checkpoint_pause."""

    @pytest.mark.asyncio
    async def test_pause_creates_checkpoint(self, auth_context, mock_checkpoint_manager):
        """Pausing creates a checkpoint and returns success."""
        result = await handle_checkpoint_pause(
            "debate-1", auth_context, checkpoint_manager=mock_checkpoint_manager
        )

        assert result["status"] == 200
        body = _body(result)
        assert body["success"] is True
        assert body["debate_id"] == "debate-1"
        assert body["checkpoint_id"] == "cp-debate1-001-abcd"
        assert "paused_at" in body

    @pytest.mark.asyncio
    async def test_pause_updates_paused_state(self, auth_context, mock_checkpoint_manager):
        """Pausing tracks the debate in _paused_debates."""
        await handle_checkpoint_pause(
            "debate-1", auth_context, checkpoint_manager=mock_checkpoint_manager
        )

        assert "debate-1" in _paused_debates
        assert _paused_debates["debate-1"]["checkpoint_id"] == "cp-debate1-001-abcd"
        assert _paused_debates["debate-1"]["paused_by"] == "user-123"

    @pytest.mark.asyncio
    async def test_pause_without_manager_returns_503(self, auth_context):
        """Pausing without a checkpoint manager returns 503."""
        result = await handle_checkpoint_pause("debate-1", auth_context, checkpoint_manager=None)

        assert result["status"] == 503

    @pytest.mark.asyncio
    async def test_pause_handles_save_error(self, auth_context):
        """Pausing handles checkpoint creation errors gracefully."""
        manager = MagicMock()
        manager.create_checkpoint = AsyncMock(side_effect=RuntimeError("Store unavailable"))

        result = await handle_checkpoint_pause("debate-1", auth_context, checkpoint_manager=manager)

        assert result["status"] == 500
        assert _body(result)["error"]  # Sanitized error message present


# ---------------------------------------------------------------------------
# Resume Endpoint Tests
# ---------------------------------------------------------------------------


class TestCheckpointResume:
    """Tests for handle_checkpoint_resume."""

    @pytest.mark.asyncio
    async def test_resume_with_checkpoint_id(self, auth_context, mock_checkpoint_manager):
        """Resume with explicit checkpoint_id loads that checkpoint."""
        resumed = MagicMock()
        resumed.resumed_at = "2026-02-14T12:30:00"
        resumed.checkpoint = MagicMock()
        resumed.checkpoint.current_round = 3
        resumed.checkpoint.total_rounds = 5
        resumed.messages = [MagicMock(), MagicMock()]
        mock_checkpoint_manager.resume_from_checkpoint = AsyncMock(return_value=resumed)

        result = await handle_checkpoint_resume(
            "debate-1",
            auth_context,
            checkpoint_id="cp-specific",
            checkpoint_manager=mock_checkpoint_manager,
        )

        assert result["status"] == 200
        body = _body(result)
        assert body["success"] is True
        assert body["checkpoint_id"] == "cp-specific"
        assert body["resumed_from_round"] == 3
        assert body["message_count"] == 2

    @pytest.mark.asyncio
    async def test_resume_uses_latest_when_no_id(self, auth_context, mock_checkpoint_manager):
        """Resume without checkpoint_id uses latest checkpoint."""
        latest_cp = MagicMock()
        latest_cp.checkpoint_id = "cp-latest"
        mock_checkpoint_manager.get_latest = AsyncMock(return_value=latest_cp)

        resumed = MagicMock()
        resumed.resumed_at = "2026-02-14T12:30:00"
        resumed.checkpoint = MagicMock()
        resumed.checkpoint.current_round = 4
        resumed.checkpoint.total_rounds = 5
        resumed.messages = []
        mock_checkpoint_manager.resume_from_checkpoint = AsyncMock(return_value=resumed)

        result = await handle_checkpoint_resume(
            "debate-1",
            auth_context,
            checkpoint_id=None,
            checkpoint_manager=mock_checkpoint_manager,
        )

        assert result["status"] == 200
        body = _body(result)
        assert body["checkpoint_id"] == "cp-latest"

    @pytest.mark.asyncio
    async def test_resume_no_checkpoints_returns_404(self, auth_context, mock_checkpoint_manager):
        """Resume returns 404 when no checkpoints exist for debate."""
        mock_checkpoint_manager.get_latest = AsyncMock(return_value=None)

        result = await handle_checkpoint_resume(
            "debate-1",
            auth_context,
            checkpoint_id=None,
            checkpoint_manager=mock_checkpoint_manager,
        )

        assert result["status"] == 404
        assert "No checkpoints found" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_resume_corrupted_checkpoint_returns_404(
        self, auth_context, mock_checkpoint_manager
    ):
        """Resume returns 404 when checkpoint is corrupted."""
        mock_checkpoint_manager.resume_from_checkpoint = AsyncMock(return_value=None)

        result = await handle_checkpoint_resume(
            "debate-1",
            auth_context,
            checkpoint_id="cp-corrupted",
            checkpoint_manager=mock_checkpoint_manager,
        )

        assert result["status"] == 404
        assert "not found or corrupted" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_resume_clears_paused_state(self, auth_context, mock_checkpoint_manager):
        """Resuming clears the debate from _paused_debates."""
        _paused_debates["debate-1"] = {
            "checkpoint_id": "cp-old",
            "paused_at": "2026-02-14T12:00:00",
            "paused_by": "user-123",
        }

        resumed = MagicMock()
        resumed.resumed_at = "2026-02-14T12:30:00"
        resumed.checkpoint = MagicMock()
        resumed.checkpoint.current_round = 2
        resumed.checkpoint.total_rounds = 5
        resumed.messages = []
        mock_checkpoint_manager.resume_from_checkpoint = AsyncMock(return_value=resumed)

        await handle_checkpoint_resume(
            "debate-1",
            auth_context,
            checkpoint_id="cp-old",
            checkpoint_manager=mock_checkpoint_manager,
        )

        assert "debate-1" not in _paused_debates

    @pytest.mark.asyncio
    async def test_resume_without_manager_returns_503(self, auth_context):
        """Resume without checkpoint manager returns 503."""
        result = await handle_checkpoint_resume("debate-1", auth_context, checkpoint_manager=None)

        assert result["status"] == 503

    @pytest.mark.asyncio
    async def test_resume_handles_error(self, auth_context):
        """Resume handles errors from checkpoint manager."""
        manager = MagicMock()
        manager.resume_from_checkpoint = AsyncMock(
            side_effect=ValueError("Invalid checkpoint format")
        )

        result = await handle_checkpoint_resume(
            "debate-1",
            auth_context,
            checkpoint_id="cp-bad",
            checkpoint_manager=manager,
        )

        assert result["status"] == 500
        assert _body(result)["error"]  # Sanitized error message present


# ---------------------------------------------------------------------------
# List Checkpoints Endpoint Tests
# ---------------------------------------------------------------------------


class TestListCheckpoints:
    """Tests for handle_list_checkpoints."""

    @pytest.mark.asyncio
    async def test_list_returns_checkpoints(self, auth_context, mock_checkpoint_manager):
        """List returns checkpoints for a debate."""
        mock_checkpoint_manager.store.list_checkpoints = AsyncMock(
            return_value=[
                {
                    "checkpoint_id": "cp-1",
                    "debate_id": "debate-1",
                    "current_round": 1,
                    "created_at": "2026-02-14T10:00:00",
                    "status": "complete",
                },
                {
                    "checkpoint_id": "cp-2",
                    "debate_id": "debate-1",
                    "current_round": 2,
                    "created_at": "2026-02-14T11:00:00",
                    "status": "complete",
                },
            ]
        )

        result = await handle_list_checkpoints(
            "debate-1", auth_context, checkpoint_manager=mock_checkpoint_manager
        )

        assert result["status"] == 200
        body = _body(result)
        assert body["debate_id"] == "debate-1"
        assert body["total_checkpoints"] == 2
        assert len(body["checkpoints"]) == 2
        assert body["is_paused"] is False

    @pytest.mark.asyncio
    async def test_list_shows_paused_state(self, auth_context, mock_checkpoint_manager):
        """List shows is_paused=True when debate is paused."""
        _paused_debates["debate-1"] = {
            "checkpoint_id": "cp-pause",
            "paused_at": "2026-02-14T12:00:00",
            "paused_by": "user-123",
        }

        mock_checkpoint_manager.store.list_checkpoints = AsyncMock(return_value=[])

        result = await handle_list_checkpoints(
            "debate-1", auth_context, checkpoint_manager=mock_checkpoint_manager
        )

        body = _body(result)
        assert body["is_paused"] is True
        assert body["paused_at"] == "2026-02-14T12:00:00"

    @pytest.mark.asyncio
    async def test_list_empty_checkpoints(self, auth_context, mock_checkpoint_manager):
        """List returns empty array when no checkpoints exist."""
        mock_checkpoint_manager.store.list_checkpoints = AsyncMock(return_value=[])

        result = await handle_list_checkpoints(
            "debate-1", auth_context, checkpoint_manager=mock_checkpoint_manager
        )

        body = _body(result)
        assert body["total_checkpoints"] == 0
        assert body["checkpoints"] == []

    @pytest.mark.asyncio
    async def test_list_without_manager_returns_503(self, auth_context):
        """List without checkpoint manager returns 503."""
        result = await handle_list_checkpoints("debate-1", auth_context, checkpoint_manager=None)

        assert result["status"] == 503

    @pytest.mark.asyncio
    async def test_list_respects_limit(self, auth_context, mock_checkpoint_manager):
        """List passes limit parameter (capped at 100)."""
        mock_checkpoint_manager.store.list_checkpoints = AsyncMock(return_value=[])

        await handle_list_checkpoints(
            "debate-1", auth_context, limit=200, checkpoint_manager=mock_checkpoint_manager
        )

        # Verify the store was called with capped limit
        mock_checkpoint_manager.store.list_checkpoints.assert_called_once_with(
            debate_id="debate-1", limit=100
        )

    @pytest.mark.asyncio
    async def test_list_handles_store_error(self, auth_context):
        """List handles store errors gracefully."""
        manager = MagicMock()
        manager.store = MagicMock()
        manager.store.list_checkpoints = AsyncMock(side_effect=OSError("Store connection lost"))

        result = await handle_list_checkpoints("debate-1", auth_context, checkpoint_manager=manager)

        assert result["status"] == 500
        assert _body(result)["error"]  # Sanitized error message present


# ---------------------------------------------------------------------------
# Integration: Pause then Resume
# ---------------------------------------------------------------------------


class TestPauseResumeFlow:
    """Tests for the complete pause -> resume flow."""

    @pytest.mark.asyncio
    async def test_pause_then_resume_flow(self, auth_context, mock_checkpoint_manager):
        """Full flow: pause creates checkpoint, resume loads it."""
        # Step 1: Pause
        pause_result = await handle_checkpoint_pause(
            "debate-flow", auth_context, checkpoint_manager=mock_checkpoint_manager
        )
        assert pause_result["status"] == 200
        assert "debate-flow" in _paused_debates

        # Step 2: Resume
        resumed = MagicMock()
        resumed.resumed_at = "2026-02-14T13:00:00"
        resumed.checkpoint = MagicMock()
        resumed.checkpoint.current_round = 0
        resumed.checkpoint.total_rounds = 0
        resumed.messages = []
        mock_checkpoint_manager.resume_from_checkpoint = AsyncMock(return_value=resumed)

        resume_result = await handle_checkpoint_resume(
            "debate-flow",
            auth_context,
            checkpoint_id=_body(pause_result)["checkpoint_id"],
            checkpoint_manager=mock_checkpoint_manager,
        )
        assert resume_result["status"] == 200
        assert "debate-flow" not in _paused_debates

    @pytest.mark.asyncio
    async def test_pause_multiple_debates(self, auth_context, mock_checkpoint_manager):
        """Can pause multiple debates independently."""
        for debate_id in ["debate-a", "debate-b", "debate-c"]:
            cp = MagicMock()
            cp.checkpoint_id = f"cp-{debate_id}"
            mock_checkpoint_manager.create_checkpoint = AsyncMock(return_value=cp)

            result = await handle_checkpoint_pause(
                debate_id, auth_context, checkpoint_manager=mock_checkpoint_manager
            )
            assert result["status"] == 200

        assert len(_paused_debates) == 3
        assert "debate-a" in _paused_debates
        assert "debate-b" in _paused_debates
        assert "debate-c" in _paused_debates
