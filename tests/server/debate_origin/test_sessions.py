"""Comprehensive tests for debate origin session management.

Tests cover:
1. Session creation and linking
2. Session lookup for debate
3. Error handling when session manager unavailable
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


from aragora.server.debate_origin.sessions import (
    _create_and_link_session,
    get_sessions_for_debate,
)


# =============================================================================
# Test: Session Creation and Linking
# =============================================================================


class TestCreateAndLinkSession:
    """Tests for _create_and_link_session function."""

    @pytest.mark.asyncio
    async def test_creates_and_links_session(self):
        """_create_and_link_session creates session and links to debate."""
        mock_session = MagicMock()
        mock_session.session_id = "session-123"

        mock_manager = MagicMock()
        mock_manager.create_session = AsyncMock(return_value=mock_session)
        mock_manager.link_debate = AsyncMock()

        await _create_and_link_session(
            manager=mock_manager,
            platform="telegram",
            user_id="user-456",
            metadata={"key": "value"},
            debate_id="debate-789",
        )

        mock_manager.create_session.assert_called_once_with(
            "telegram", "user-456", {"key": "value"}
        )
        mock_manager.link_debate.assert_called_once_with("session-123", "debate-789")

    @pytest.mark.asyncio
    async def test_handles_create_session_error(self):
        """_create_and_link_session handles session creation errors."""
        mock_manager = MagicMock()
        mock_manager.create_session = AsyncMock(side_effect=RuntimeError("Failed"))

        # Should not raise
        await _create_and_link_session(
            manager=mock_manager,
            platform="slack",
            user_id="U123",
            metadata=None,
            debate_id="debate-001",
        )

    @pytest.mark.asyncio
    async def test_handles_link_debate_error(self):
        """_create_and_link_session handles link errors."""
        mock_session = MagicMock()
        mock_session.session_id = "session-abc"

        mock_manager = MagicMock()
        mock_manager.create_session = AsyncMock(return_value=mock_session)
        mock_manager.link_debate = AsyncMock(side_effect=ValueError("Invalid debate"))

        # Should not raise
        await _create_and_link_session(
            manager=mock_manager,
            platform="discord",
            user_id="user-xyz",
            metadata={},
            debate_id="bad-debate",
        )

    @pytest.mark.asyncio
    async def test_handles_connection_error(self):
        """_create_and_link_session handles connection errors."""
        mock_manager = MagicMock()
        mock_manager.create_session = AsyncMock(side_effect=ConnectionError("Network down"))

        # Should not raise
        await _create_and_link_session(
            manager=mock_manager,
            platform="teams",
            user_id="T123",
            metadata=None,
            debate_id="debate-net",
        )

    @pytest.mark.asyncio
    async def test_handles_timeout_error(self):
        """_create_and_link_session handles timeout errors."""
        mock_manager = MagicMock()
        mock_manager.create_session = AsyncMock(side_effect=TimeoutError("Timed out"))

        # Should not raise
        await _create_and_link_session(
            manager=mock_manager,
            platform="whatsapp",
            user_id="+1234567890",
            metadata=None,
            debate_id="debate-timeout",
        )


# =============================================================================
# Test: Session Lookup
# =============================================================================


class TestGetSessionsForDebate:
    """Tests for get_sessions_for_debate function."""

    @pytest.mark.asyncio
    async def test_returns_sessions_from_manager(self):
        """get_sessions_for_debate returns sessions from manager."""
        mock_session1 = MagicMock()
        mock_session1.session_id = "session-1"
        mock_session2 = MagicMock()
        mock_session2.session_id = "session-2"

        mock_manager = MagicMock()
        mock_manager.find_sessions_for_debate = AsyncMock(
            return_value=[mock_session1, mock_session2]
        )

        with patch(
            "aragora.server.debate_origin.sessions.get_debate_session_manager",
            return_value=mock_manager,
        ):
            sessions = await get_sessions_for_debate("debate-multi")

        assert len(sessions) == 2
        assert sessions[0].session_id == "session-1"
        assert sessions[1].session_id == "session-2"

    @pytest.mark.asyncio
    async def test_returns_empty_when_import_error(self):
        """get_sessions_for_debate returns empty list on ImportError."""
        with patch(
            "aragora.server.debate_origin.sessions.get_debate_session_manager",
            side_effect=ImportError("Module not found"),
        ):
            sessions = await get_sessions_for_debate("debate-no-module")

        assert sessions == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_runtime_error(self):
        """get_sessions_for_debate returns empty list on RuntimeError."""
        mock_manager = MagicMock()
        mock_manager.find_sessions_for_debate = AsyncMock(
            side_effect=RuntimeError("Manager not initialized")
        )

        with patch(
            "aragora.server.debate_origin.sessions.get_debate_session_manager",
            return_value=mock_manager,
        ):
            sessions = await get_sessions_for_debate("debate-runtime")

        assert sessions == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_connection_error(self):
        """get_sessions_for_debate returns empty list on ConnectionError."""
        mock_manager = MagicMock()
        mock_manager.find_sessions_for_debate = AsyncMock(
            side_effect=ConnectionError("DB unavailable")
        )

        with patch(
            "aragora.server.debate_origin.sessions.get_debate_session_manager",
            return_value=mock_manager,
        ):
            sessions = await get_sessions_for_debate("debate-conn")

        assert sessions == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_timeout_error(self):
        """get_sessions_for_debate returns empty list on TimeoutError."""
        mock_manager = MagicMock()
        mock_manager.find_sessions_for_debate = AsyncMock(side_effect=TimeoutError("Query timeout"))

        with patch(
            "aragora.server.debate_origin.sessions.get_debate_session_manager",
            return_value=mock_manager,
        ):
            sessions = await get_sessions_for_debate("debate-timeout")

        assert sessions == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_value_error(self):
        """get_sessions_for_debate returns empty list on ValueError."""
        mock_manager = MagicMock()
        mock_manager.find_sessions_for_debate = AsyncMock(
            side_effect=ValueError("Invalid debate ID")
        )

        with patch(
            "aragora.server.debate_origin.sessions.get_debate_session_manager",
            return_value=mock_manager,
        ):
            sessions = await get_sessions_for_debate("invalid-id")

        assert sessions == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_key_error(self):
        """get_sessions_for_debate returns empty list on KeyError."""
        mock_manager = MagicMock()
        mock_manager.find_sessions_for_debate = AsyncMock(side_effect=KeyError("debate_id"))

        with patch(
            "aragora.server.debate_origin.sessions.get_debate_session_manager",
            return_value=mock_manager,
        ):
            sessions = await get_sessions_for_debate("missing-key")

        assert sessions == []

    @pytest.mark.asyncio
    async def test_handles_empty_result(self):
        """get_sessions_for_debate handles empty result."""
        mock_manager = MagicMock()
        mock_manager.find_sessions_for_debate = AsyncMock(return_value=[])

        with patch(
            "aragora.server.debate_origin.sessions.get_debate_session_manager",
            return_value=mock_manager,
        ):
            sessions = await get_sessions_for_debate("debate-no-sessions")

        assert sessions == []

    @pytest.mark.asyncio
    async def test_returns_all_linked_sessions(self):
        """get_sessions_for_debate returns all sessions linked to debate."""
        sessions_data = [
            {"session_id": f"sess-{i}", "channel": "telegram", "user_id": f"user-{i}"}
            for i in range(5)
        ]
        mock_sessions = []
        for data in sessions_data:
            mock_session = MagicMock()
            mock_session.session_id = data["session_id"]
            mock_session.channel = data["channel"]
            mock_session.user_id = data["user_id"]
            mock_sessions.append(mock_session)

        mock_manager = MagicMock()
        mock_manager.find_sessions_for_debate = AsyncMock(return_value=mock_sessions)

        with patch(
            "aragora.server.debate_origin.sessions.get_debate_session_manager",
            return_value=mock_manager,
        ):
            sessions = await get_sessions_for_debate("debate-many")

        assert len(sessions) == 5
        for i, session in enumerate(sessions):
            assert session.session_id == f"sess-{i}"
