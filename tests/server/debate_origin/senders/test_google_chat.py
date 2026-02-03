"""Tests for Google Chat sender for debate origin result routing.

Tests cover:
1. Result message sending via Google Chat connector
2. Rich Card v2 formatting
3. Vote button integration
4. Receipt posting
5. Thread handling
6. Connector availability handling
7. Error handling for API failures
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.debate_origin.models import DebateOrigin
from aragora.server.debate_origin.senders.google_chat import (
    _send_google_chat_result,
    _send_google_chat_receipt,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_origin() -> DebateOrigin:
    """Create a sample Google Chat debate origin for testing."""
    return DebateOrigin(
        debate_id="debate-gchat-123",
        platform="google_chat",
        channel_id="spaces/ABCDEFGHIJ",
        user_id="users/123456789",
        thread_id="spaces/ABCDEFGHIJ/threads/KLMNOPQRST",
        metadata={
            "topic": "Google Chat Integration Test",
            "thread_name": "spaces/ABCDEFGHIJ/threads/KLMNOPQRST",
        },
    )


@pytest.fixture
def sample_origin_no_thread() -> DebateOrigin:
    """Create a Google Chat origin without thread info."""
    return DebateOrigin(
        debate_id="debate-gchat-456",
        platform="google_chat",
        channel_id="spaces/ZYXWVUTSRQ",
        user_id="users/987654321",
        metadata={"topic": "No Thread Test"},
    )


@pytest.fixture
def sample_result() -> dict[str, Any]:
    """Create a sample debate result for testing."""
    return {
        "id": "result-gchat-789",
        "consensus_reached": True,
        "final_answer": "The team reached agreement on the approach.",
        "confidence": 0.85,
        "participants": ["claude", "gpt-4", "gemini"],
        "task": "Evaluate the Google Chat proposal",
    }


@pytest.fixture
def mock_connector():
    """Create a mock Google Chat connector."""
    connector = MagicMock()
    connector.send_message = AsyncMock()
    return connector


@pytest.fixture
def mock_response_success():
    """Create a successful mock response."""
    response = MagicMock()
    response.success = True
    response.error = None
    return response


@pytest.fixture
def mock_response_failure():
    """Create a failed mock response."""
    response = MagicMock()
    response.success = False
    response.error = "Permission denied"
    return response


# =============================================================================
# Test: Send Google Chat Result
# =============================================================================


class TestSendGoogleChatResult:
    """Tests for _send_google_chat_result function."""

    @pytest.mark.asyncio
    async def test_sends_result_via_connector(
        self, sample_origin, sample_result, mock_connector, mock_response_success
    ):
        """_send_google_chat_result sends message via connector."""
        mock_connector.send_message.return_value = mock_response_success
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            result = await _send_google_chat_result(sample_origin, sample_result)

        assert result is True
        mock_connector.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_includes_space_name(
        self, sample_origin, sample_result, mock_connector, mock_response_success
    ):
        """_send_google_chat_result sends to correct space."""
        mock_connector.send_message.return_value = mock_response_success
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            await _send_google_chat_result(sample_origin, sample_result)

        call_args = mock_connector.send_message.call_args
        assert call_args[0][0] == sample_origin.channel_id  # space_name

    @pytest.mark.asyncio
    async def test_includes_thread_id(
        self, sample_origin, sample_result, mock_connector, mock_response_success
    ):
        """_send_google_chat_result includes thread_id when available."""
        mock_connector.send_message.return_value = mock_response_success
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            await _send_google_chat_result(sample_origin, sample_result)

        call_kwargs = mock_connector.send_message.call_args[1]
        assert call_kwargs["thread_id"] == sample_origin.thread_id

    @pytest.mark.asyncio
    async def test_uses_metadata_thread_name_as_fallback(
        self, sample_origin_no_thread, sample_result, mock_connector, mock_response_success
    ):
        """_send_google_chat_result uses metadata thread_name when thread_id is None."""
        sample_origin_no_thread.metadata["thread_name"] = "spaces/X/threads/Y"
        mock_connector.send_message.return_value = mock_response_success
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            await _send_google_chat_result(sample_origin_no_thread, sample_result)

        call_kwargs = mock_connector.send_message.call_args[1]
        assert call_kwargs["thread_id"] == "spaces/X/threads/Y"

    @pytest.mark.asyncio
    async def test_builds_card_with_consensus_status(
        self, sample_origin, sample_result, mock_connector, mock_response_success
    ):
        """_send_google_chat_result includes consensus status in card."""
        mock_connector.send_message.return_value = mock_response_success
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            await _send_google_chat_result(sample_origin, sample_result)

        call_args = mock_connector.send_message.call_args
        blocks = call_args[1]["blocks"]

        # Should have consensus emoji in header
        header_section = blocks[0]
        assert "header" in header_section
        # Checkmark for consensus reached
        assert "\u2705" in header_section["header"]

    @pytest.mark.asyncio
    async def test_shows_no_consensus_status(
        self, sample_origin, sample_result, mock_connector, mock_response_success
    ):
        """_send_google_chat_result shows X when no consensus."""
        sample_result["consensus_reached"] = False
        mock_connector.send_message.return_value = mock_response_success
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            await _send_google_chat_result(sample_origin, sample_result)

        call_args = mock_connector.send_message.call_args
        blocks = call_args[1]["blocks"]

        # Should have X emoji for no consensus
        header_section = blocks[0]
        assert "\u274c" in header_section["header"]

    @pytest.mark.asyncio
    async def test_includes_confidence_bar(
        self, sample_origin, sample_result, mock_connector, mock_response_success
    ):
        """_send_google_chat_result includes visual confidence bar."""
        mock_connector.send_message.return_value = mock_response_success
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            await _send_google_chat_result(sample_origin, sample_result)

        call_args = mock_connector.send_message.call_args
        blocks = call_args[1]["blocks"]

        # Look for confidence bar characters
        blocks_str = str(blocks)
        assert "85%" in blocks_str

    @pytest.mark.asyncio
    async def test_includes_vote_buttons(
        self, sample_origin, sample_result, mock_connector, mock_response_success
    ):
        """_send_google_chat_result includes vote buttons."""
        mock_connector.send_message.return_value = mock_response_success
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            await _send_google_chat_result(sample_origin, sample_result)

        call_args = mock_connector.send_message.call_args
        blocks = call_args[1]["blocks"]
        blocks_str = str(blocks)

        # Should have Agree and Disagree buttons
        assert "Agree" in blocks_str
        assert "Disagree" in blocks_str
        assert "vote_agree" in blocks_str
        assert "vote_disagree" in blocks_str

    @pytest.mark.asyncio
    async def test_passes_debate_id_to_vote_buttons(
        self, sample_origin, sample_result, mock_connector, mock_response_success
    ):
        """_send_google_chat_result includes debate_id in vote button parameters."""
        mock_connector.send_message.return_value = mock_response_success
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            await _send_google_chat_result(sample_origin, sample_result)

        call_args = mock_connector.send_message.call_args
        blocks = call_args[1]["blocks"]
        blocks_str = str(blocks)

        # Should include the result/debate ID in button parameters
        assert sample_result["id"] in blocks_str

    @pytest.mark.asyncio
    async def test_truncates_long_answer(
        self, sample_origin, sample_result, mock_connector, mock_response_success
    ):
        """_send_google_chat_result truncates answers over 500 chars."""
        sample_result["final_answer"] = "X" * 700
        mock_connector.send_message.return_value = mock_response_success
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            await _send_google_chat_result(sample_origin, sample_result)

        call_args = mock_connector.send_message.call_args
        blocks = call_args[1]["blocks"]
        blocks_str = str(blocks)

        # Answer should be truncated with ellipsis
        assert "X" * 500 in blocks_str
        assert "..." in blocks_str
        assert "X" * 501 not in blocks_str

    @pytest.mark.asyncio
    async def test_truncates_long_topic(
        self, sample_origin, sample_result, mock_connector, mock_response_success
    ):
        """_send_google_chat_result truncates topics over 200 chars."""
        sample_result["task"] = "T" * 300
        mock_connector.send_message.return_value = mock_response_success
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            await _send_google_chat_result(sample_origin, sample_result)

        call_args = mock_connector.send_message.call_args
        blocks = call_args[1]["blocks"]
        blocks_str = str(blocks)

        # Topic should be truncated
        assert "T" * 200 in blocks_str
        assert "T" * 201 not in blocks_str

    @pytest.mark.asyncio
    async def test_returns_false_when_connector_unavailable(self, sample_origin, sample_result):
        """_send_google_chat_result returns False when connector not configured."""
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=None))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            result = await _send_google_chat_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_module_unavailable(self, sample_origin, sample_result):
        """_send_google_chat_result returns False when google_chat module not available."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "google_chat" in name:
                raise ImportError("No google_chat module")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", mock_import):
            result = await _send_google_chat_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_send_failure(
        self, sample_origin, sample_result, mock_connector, mock_response_failure
    ):
        """_send_google_chat_result returns False when send fails."""
        mock_connector.send_message.return_value = mock_response_failure
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            result = await _send_google_chat_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_os_error(self, sample_origin, sample_result, mock_connector):
        """_send_google_chat_result returns False on OS errors."""
        mock_connector.send_message = AsyncMock(side_effect=OSError("Network error"))
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            result = await _send_google_chat_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_runtime_error(
        self, sample_origin, sample_result, mock_connector
    ):
        """_send_google_chat_result returns False on runtime errors."""
        mock_connector.send_message = AsyncMock(side_effect=RuntimeError("Runtime failure"))
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            result = await _send_google_chat_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_value_error(self, sample_origin, sample_result, mock_connector):
        """_send_google_chat_result returns False on value errors."""
        mock_connector.send_message = AsyncMock(side_effect=ValueError("Invalid value"))
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            result = await _send_google_chat_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_type_error(self, sample_origin, sample_result, mock_connector):
        """_send_google_chat_result returns False on type errors."""
        mock_connector.send_message = AsyncMock(side_effect=TypeError("Type mismatch"))
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            result = await _send_google_chat_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_uses_origin_topic_as_fallback(
        self, sample_origin, sample_result, mock_connector, mock_response_success
    ):
        """_send_google_chat_result uses origin metadata topic when task missing."""
        del sample_result["task"]
        mock_connector.send_message.return_value = mock_response_success
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            await _send_google_chat_result(sample_origin, sample_result)

        call_args = mock_connector.send_message.call_args
        blocks = call_args[1]["blocks"]
        blocks_str = str(blocks)

        assert "Google Chat Integration Test" in blocks_str

    @pytest.mark.asyncio
    async def test_uses_debate_id_from_origin_as_fallback(
        self, sample_origin, sample_result, mock_connector, mock_response_success
    ):
        """_send_google_chat_result uses origin debate_id when result has no id."""
        del sample_result["id"]
        mock_connector.send_message.return_value = mock_response_success
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            await _send_google_chat_result(sample_origin, sample_result)

        call_args = mock_connector.send_message.call_args
        blocks = call_args[1]["blocks"]
        blocks_str = str(blocks)

        # Should use origin's debate_id in vote buttons
        assert sample_origin.debate_id in blocks_str


# =============================================================================
# Test: Send Google Chat Receipt
# =============================================================================


class TestSendGoogleChatReceipt:
    """Tests for _send_google_chat_receipt function."""

    @pytest.mark.asyncio
    async def test_posts_receipt_summary(
        self, sample_origin, mock_connector, mock_response_success
    ):
        """_send_google_chat_receipt posts receipt summary."""
        mock_connector.send_message.return_value = mock_response_success
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            result = await _send_google_chat_receipt(
                sample_origin, "Receipt: Approved with 95% confidence"
            )

        assert result is True
        call_args = mock_connector.send_message.call_args
        assert call_args[0][1] == "Receipt: Approved with 95% confidence"

    @pytest.mark.asyncio
    async def test_includes_space_name(self, sample_origin, mock_connector, mock_response_success):
        """_send_google_chat_receipt sends to correct space."""
        mock_connector.send_message.return_value = mock_response_success
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            await _send_google_chat_receipt(sample_origin, "Receipt summary")

        call_args = mock_connector.send_message.call_args
        assert call_args[0][0] == sample_origin.channel_id

    @pytest.mark.asyncio
    async def test_includes_thread_id(self, sample_origin, mock_connector, mock_response_success):
        """_send_google_chat_receipt includes thread_id when available."""
        mock_connector.send_message.return_value = mock_response_success
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            await _send_google_chat_receipt(sample_origin, "Receipt")

        call_kwargs = mock_connector.send_message.call_args[1]
        assert call_kwargs["thread_id"] == sample_origin.thread_id

    @pytest.mark.asyncio
    async def test_uses_metadata_thread_name_as_fallback(
        self, sample_origin_no_thread, mock_connector, mock_response_success
    ):
        """_send_google_chat_receipt uses metadata thread_name when thread_id is None."""
        sample_origin_no_thread.metadata["thread_name"] = "spaces/X/threads/Z"
        mock_connector.send_message.return_value = mock_response_success
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            await _send_google_chat_receipt(sample_origin_no_thread, "Receipt")

        call_kwargs = mock_connector.send_message.call_args[1]
        assert call_kwargs["thread_id"] == "spaces/X/threads/Z"

    @pytest.mark.asyncio
    async def test_returns_false_when_connector_unavailable(self, sample_origin):
        """_send_google_chat_receipt returns False when connector not configured."""
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=None))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            result = await _send_google_chat_receipt(sample_origin, "Receipt")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_module_unavailable(self, sample_origin):
        """_send_google_chat_receipt returns False when google_chat module not available."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "google_chat" in name:
                raise ImportError("No google_chat module")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", mock_import):
            result = await _send_google_chat_receipt(sample_origin, "Receipt")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_send_failure(
        self, sample_origin, mock_connector, mock_response_failure
    ):
        """_send_google_chat_receipt returns False when send fails."""
        mock_connector.send_message.return_value = mock_response_failure
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            result = await _send_google_chat_receipt(sample_origin, "Receipt")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_error(self, sample_origin, mock_connector):
        """_send_google_chat_receipt returns False on errors."""
        mock_connector.send_message = AsyncMock(side_effect=RuntimeError("Error"))
        mock_module = MagicMock(get_google_chat_connector=MagicMock(return_value=mock_connector))

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.google_chat": mock_module}):
            result = await _send_google_chat_receipt(sample_origin, "Receipt")

        assert result is False
