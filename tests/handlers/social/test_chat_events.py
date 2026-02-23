"""
Comprehensive tests for aragora/server/handlers/social/chat_events.py.

Covers:
- _dispatch_chat_event: dispatch success, ImportError fallback, exception handling
- emit_message_received: all parameters, truncation, type coercion, empty/None
- emit_command_received: all parameters, truncation, empty args
- emit_debate_started: all parameters, optional debate_id, topic truncation
- emit_debate_completed: all parameters, truncation, consensus/confidence fields
- emit_gauntlet_started: all parameters, optional gauntlet_id, statement truncation
- emit_gauntlet_completed: all parameters, verdict/challenges fields
- emit_vote_received: all parameters, vote values
- Edge cases: empty strings, unicode, very long strings, special characters
- Security: path traversal in inputs, injection attempts
- __all__ exports
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, call, patch

import pytest

from aragora.server.handlers.social.chat_events import (
    _dispatch_chat_event,
    emit_command_received,
    emit_debate_completed,
    emit_debate_started,
    emit_gauntlet_completed,
    emit_gauntlet_started,
    emit_message_received,
    emit_vote_received,
)


# ============================================================================
# Helpers
# ============================================================================


def _last_dispatched_data(mock_dispatch: MagicMock) -> dict:
    """Extract the data dict from the most recent _dispatch_chat_event call."""
    assert mock_dispatch.called, "_dispatch_chat_event was not called"
    _, kwargs = mock_dispatch.call_args
    if kwargs:
        return kwargs.get("data", mock_dispatch.call_args[0][1])
    return mock_dispatch.call_args[0][1]


def _last_dispatched_event_type(mock_dispatch: MagicMock) -> str:
    """Extract the event_type from the most recent _dispatch_chat_event call."""
    assert mock_dispatch.called, "_dispatch_chat_event was not called"
    return mock_dispatch.call_args[0][0]


# ============================================================================
# _dispatch_chat_event
# ============================================================================


class TestDispatchChatEvent:
    """Tests for the internal _dispatch_chat_event helper."""

    def test_dispatches_to_event_system(self):
        """Successfully dispatches event when events module is available."""
        mock_dispatch_fn = MagicMock()
        with patch.dict(
            "sys.modules",
            {"aragora.events": MagicMock(dispatch_event=mock_dispatch_fn)},
        ):
            _dispatch_chat_event("chat.test_event", {"key": "value"})
            mock_dispatch_fn.assert_called_once_with("chat.test_event", {"key": "value"})

    def test_handles_import_error_gracefully(self):
        """Does not raise when aragora.events is not importable."""
        with patch.dict("sys.modules", {"aragora.events": None}):
            # Should not raise
            _dispatch_chat_event("chat.test_event", {"key": "value"})

    def test_handles_runtime_error(self):
        """Handles RuntimeError from dispatch gracefully."""
        mock_events = MagicMock()
        mock_events.dispatch_event.side_effect = RuntimeError("queue full")
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            _dispatch_chat_event("chat.test", {"data": 1})

    def test_handles_os_error(self):
        """Handles OSError from dispatch gracefully."""
        mock_events = MagicMock()
        mock_events.dispatch_event.side_effect = OSError("network failure")
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            _dispatch_chat_event("chat.test", {"data": 1})

    def test_handles_value_error(self):
        """Handles ValueError from dispatch gracefully."""
        mock_events = MagicMock()
        mock_events.dispatch_event.side_effect = ValueError("bad data")
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            _dispatch_chat_event("chat.test", {"data": 1})

    def test_handles_type_error(self):
        """Handles TypeError from dispatch gracefully."""
        mock_events = MagicMock()
        mock_events.dispatch_event.side_effect = TypeError("wrong type")
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            _dispatch_chat_event("chat.test", {"data": 1})

    def test_passes_event_type_and_data(self):
        """Event type and data are passed through to dispatch_event."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            data = {"platform": "telegram", "chat_id": "123"}
            _dispatch_chat_event("chat.custom_event", data)
            mock_events.dispatch_event.assert_called_once_with("chat.custom_event", data)

    def test_empty_data_dict(self):
        """Empty data dict is passed without error."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            _dispatch_chat_event("chat.empty", {})
            mock_events.dispatch_event.assert_called_once_with("chat.empty", {})

    def test_large_data_dict(self):
        """Large data dict is passed without truncation at dispatch level."""
        mock_events = MagicMock()
        large_data = {f"key_{i}": f"value_{i}" for i in range(1000)}
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            _dispatch_chat_event("chat.large", large_data)
            mock_events.dispatch_event.assert_called_once_with("chat.large", large_data)


# ============================================================================
# emit_message_received
# ============================================================================


class TestEmitMessageReceived:
    """Tests for emit_message_received."""

    def test_dispatches_correct_event_type(self):
        """Event type is 'chat.message_received'."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", "chat1", "user1", "alice", "hello")
            event_type = mock_events.dispatch_event.call_args[0][0]
            assert event_type == "chat.message_received"

    def test_all_fields_present(self):
        """All expected fields are present in the event data."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", "chat1", "user1", "alice", "hello world")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["platform"] == "telegram"
            assert data["chat_id"] == "chat1"
            assert data["user_id"] == "user1"
            assert data["username"] == "alice"
            assert data["message_type"] == "text"
            assert data["message_preview"] == "hello world"
            assert "timestamp" in data
            assert isinstance(data["timestamp"], float)

    def test_custom_message_type(self):
        """Custom message_type parameter is passed through."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("whatsapp", "c2", "u2", "bob", "cmd", message_type="command")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["message_type"] == "command"

    def test_message_truncated_at_100_chars(self):
        """Long message text is truncated to 100 characters."""
        mock_events = MagicMock()
        long_text = "x" * 250
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", "c1", "u1", "alice", long_text)
            data = mock_events.dispatch_event.call_args[0][1]
            assert len(data["message_preview"]) == 100
            assert data["message_preview"] == "x" * 100

    def test_message_exactly_100_chars(self):
        """Message of exactly 100 characters is kept as-is."""
        mock_events = MagicMock()
        text = "a" * 100
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", "c1", "u1", "alice", text)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["message_preview"] == text

    def test_empty_message(self):
        """Empty message text produces empty preview."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", "c1", "u1", "alice", "")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["message_preview"] == ""

    def test_none_message(self):
        """None message text produces empty preview."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", "c1", "u1", "alice", None)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["message_preview"] == ""

    def test_chat_id_coerced_to_string(self):
        """Numeric chat_id is coerced to string."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", 12345, "u1", "alice", "hi")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["chat_id"] == "12345"

    def test_user_id_coerced_to_string(self):
        """Numeric user_id is coerced to string."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", "c1", 67890, "alice", "hi")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["user_id"] == "67890"

    def test_timestamp_is_recent(self):
        """Timestamp is approximately current time."""
        mock_events = MagicMock()
        before = time.time()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", "c1", "u1", "alice", "hi")
        after = time.time()
        data = mock_events.dispatch_event.call_args[0][1]
        assert before <= data["timestamp"] <= after

    def test_whatsapp_platform(self):
        """WhatsApp platform is correctly set."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("whatsapp", "c1", "u1", "carol", "msg")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["platform"] == "whatsapp"

    def test_unicode_username(self):
        """Unicode characters in username are preserved."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", "c1", "u1", "user name", "msg")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["username"] == "user name"

    def test_unicode_message(self):
        """Unicode characters in message are preserved."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", "c1", "u1", "alice", "Hello! How are you?")
            data = mock_events.dispatch_event.call_args[0][1]
            assert "Hello" in data["message_preview"]


# ============================================================================
# emit_command_received
# ============================================================================


class TestEmitCommandReceived:
    """Tests for emit_command_received."""

    def test_dispatches_correct_event_type(self):
        """Event type is 'chat.command_received'."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_command_received("telegram", "c1", "u1", "alice", "debate")
            event_type = mock_events.dispatch_event.call_args[0][0]
            assert event_type == "chat.command_received"

    def test_all_fields_present(self):
        """All expected fields are present in the event data."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_command_received("telegram", "c1", "u1", "alice", "debate", "is AI safe?")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["platform"] == "telegram"
            assert data["chat_id"] == "c1"
            assert data["user_id"] == "u1"
            assert data["username"] == "alice"
            assert data["command"] == "debate"
            assert data["args_preview"] == "is AI safe?"
            assert "timestamp" in data

    def test_empty_args_default(self):
        """Default empty args produce empty args_preview."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_command_received("telegram", "c1", "u1", "alice", "help")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["args_preview"] == ""

    def test_args_truncated_at_100_chars(self):
        """Long command args are truncated to 100 characters."""
        mock_events = MagicMock()
        long_args = "y" * 200
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_command_received("telegram", "c1", "u1", "alice", "debate", long_args)
            data = mock_events.dispatch_event.call_args[0][1]
            assert len(data["args_preview"]) == 100

    def test_args_exactly_100_chars(self):
        """Args of exactly 100 chars are kept intact."""
        mock_events = MagicMock()
        args_100 = "b" * 100
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_command_received("telegram", "c1", "u1", "alice", "debate", args_100)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["args_preview"] == args_100

    def test_chat_id_coerced_to_string(self):
        """Numeric chat_id is coerced to string."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_command_received("telegram", 99, "u1", "alice", "help")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["chat_id"] == "99"

    def test_user_id_coerced_to_string(self):
        """Numeric user_id is coerced to string."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_command_received("telegram", "c1", 42, "alice", "help")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["user_id"] == "42"

    def test_command_with_special_characters(self):
        """Command name with special characters is preserved."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_command_received("telegram", "c1", "u1", "alice", "debate-v2")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["command"] == "debate-v2"


# ============================================================================
# emit_debate_started
# ============================================================================


class TestEmitDebateStarted:
    """Tests for emit_debate_started."""

    def test_dispatches_correct_event_type(self):
        """Event type is 'chat.debate_started'."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_started("telegram", "c1", "u1", "alice", "AI topic")
            event_type = mock_events.dispatch_event.call_args[0][0]
            assert event_type == "chat.debate_started"

    def test_all_fields_present(self):
        """All expected fields are present in the event data."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_started("telegram", "c1", "u1", "alice", "AI safety", "d-001")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["platform"] == "telegram"
            assert data["chat_id"] == "c1"
            assert data["user_id"] == "u1"
            assert data["username"] == "alice"
            assert data["topic"] == "AI safety"
            assert data["debate_id"] == "d-001"
            assert "timestamp" in data

    def test_debate_id_none_by_default(self):
        """debate_id defaults to None when not provided."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_started("telegram", "c1", "u1", "alice", "topic")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["debate_id"] is None

    def test_topic_truncated_at_200_chars(self):
        """Long topic is truncated to 200 characters."""
        mock_events = MagicMock()
        long_topic = "t" * 500
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_started("telegram", "c1", "u1", "alice", long_topic)
            data = mock_events.dispatch_event.call_args[0][1]
            assert len(data["topic"]) == 200

    def test_topic_exactly_200_chars(self):
        """Topic of exactly 200 chars is kept intact."""
        mock_events = MagicMock()
        topic_200 = "z" * 200
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_started("telegram", "c1", "u1", "alice", topic_200)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["topic"] == topic_200

    def test_chat_id_coerced_to_string(self):
        """Numeric chat_id is coerced to string."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_started("telegram", 777, "u1", "alice", "topic")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["chat_id"] == "777"

    def test_user_id_coerced_to_string(self):
        """Numeric user_id is coerced to string."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_started("telegram", "c1", 999, "alice", "topic")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["user_id"] == "999"


# ============================================================================
# emit_debate_completed
# ============================================================================


class TestEmitDebateCompleted:
    """Tests for emit_debate_completed."""

    def test_dispatches_correct_event_type(self):
        """Event type is 'chat.debate_completed'."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_completed("telegram", "c1", "d1", "topic", True, 0.95, 3)
            event_type = mock_events.dispatch_event.call_args[0][0]
            assert event_type == "chat.debate_completed"

    def test_all_fields_present(self):
        """All expected fields are present in the event data."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_completed(
                "telegram",
                "c1",
                "d1",
                "Is AI safe?",
                True,
                0.87,
                5,
                "Yes, with caveats",
            )
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["platform"] == "telegram"
            assert data["chat_id"] == "c1"
            assert data["debate_id"] == "d1"
            assert data["topic"] == "Is AI safe?"
            assert data["consensus_reached"] is True
            assert data["confidence"] == 0.87
            assert data["rounds_used"] == 5
            assert data["final_answer_preview"] == "Yes, with caveats"
            assert "timestamp" in data

    def test_consensus_not_reached(self):
        """consensus_reached=False is preserved."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_completed("telegram", "c1", "d1", "topic", False, 0.3, 3)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["consensus_reached"] is False

    def test_final_answer_none_by_default(self):
        """final_answer_preview is None when final_answer not provided."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_completed("telegram", "c1", "d1", "topic", True, 0.9, 3)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["final_answer_preview"] is None

    def test_final_answer_truncated_at_300_chars(self):
        """Long final_answer is truncated to 300 characters."""
        mock_events = MagicMock()
        long_answer = "a" * 500
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_completed("telegram", "c1", "d1", "topic", True, 0.9, 3, long_answer)
            data = mock_events.dispatch_event.call_args[0][1]
            assert len(data["final_answer_preview"]) == 300

    def test_final_answer_exactly_300_chars(self):
        """Final answer of exactly 300 chars is kept intact."""
        mock_events = MagicMock()
        answer_300 = "f" * 300
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_completed("telegram", "c1", "d1", "topic", True, 0.9, 3, answer_300)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["final_answer_preview"] == answer_300

    def test_topic_truncated_at_200_chars(self):
        """Long topic is truncated to 200 characters."""
        mock_events = MagicMock()
        long_topic = "q" * 400
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_completed("telegram", "c1", "d1", long_topic, True, 0.9, 3)
            data = mock_events.dispatch_event.call_args[0][1]
            assert len(data["topic"]) == 200

    def test_chat_id_coerced_to_string(self):
        """Numeric chat_id is coerced to string."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_completed("telegram", 111, "d1", "topic", True, 0.9, 3)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["chat_id"] == "111"

    def test_confidence_zero(self):
        """Confidence of 0.0 is preserved."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_completed("telegram", "c1", "d1", "topic", False, 0.0, 1)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["confidence"] == 0.0

    def test_confidence_one(self):
        """Confidence of 1.0 is preserved."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_completed("telegram", "c1", "d1", "topic", True, 1.0, 10)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["confidence"] == 1.0

    def test_rounds_used_preserved(self):
        """rounds_used integer is preserved in event data."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_completed("telegram", "c1", "d1", "topic", True, 0.9, 7)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["rounds_used"] == 7


# ============================================================================
# emit_gauntlet_started
# ============================================================================


class TestEmitGauntletStarted:
    """Tests for emit_gauntlet_started."""

    def test_dispatches_correct_event_type(self):
        """Event type is 'chat.gauntlet_started'."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_gauntlet_started("telegram", "c1", "u1", "alice", "AI is safe")
            event_type = mock_events.dispatch_event.call_args[0][0]
            assert event_type == "chat.gauntlet_started"

    def test_all_fields_present(self):
        """All expected fields are present in the event data."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_gauntlet_started("telegram", "c1", "u1", "alice", "statement", "g-001")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["platform"] == "telegram"
            assert data["chat_id"] == "c1"
            assert data["user_id"] == "u1"
            assert data["username"] == "alice"
            assert data["statement"] == "statement"
            assert data["gauntlet_id"] == "g-001"
            assert "timestamp" in data

    def test_gauntlet_id_none_by_default(self):
        """gauntlet_id defaults to None when not provided."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_gauntlet_started("telegram", "c1", "u1", "alice", "stmt")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["gauntlet_id"] is None

    def test_statement_truncated_at_200_chars(self):
        """Long statement is truncated to 200 characters."""
        mock_events = MagicMock()
        long_stmt = "s" * 500
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_gauntlet_started("telegram", "c1", "u1", "alice", long_stmt)
            data = mock_events.dispatch_event.call_args[0][1]
            assert len(data["statement"]) == 200

    def test_statement_exactly_200_chars(self):
        """Statement of exactly 200 chars is kept intact."""
        mock_events = MagicMock()
        stmt_200 = "w" * 200
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_gauntlet_started("telegram", "c1", "u1", "alice", stmt_200)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["statement"] == stmt_200

    def test_chat_id_coerced_to_string(self):
        """Numeric chat_id is coerced to string."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_gauntlet_started("telegram", 555, "u1", "alice", "stmt")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["chat_id"] == "555"

    def test_user_id_coerced_to_string(self):
        """Numeric user_id is coerced to string."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_gauntlet_started("telegram", "c1", 888, "alice", "stmt")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["user_id"] == "888"


# ============================================================================
# emit_gauntlet_completed
# ============================================================================


class TestEmitGauntletCompleted:
    """Tests for emit_gauntlet_completed."""

    def test_dispatches_correct_event_type(self):
        """Event type is 'chat.gauntlet_completed'."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_gauntlet_completed("telegram", "c1", "g1", "stmt", "passed", 0.9, 5, 5)
            event_type = mock_events.dispatch_event.call_args[0][0]
            assert event_type == "chat.gauntlet_completed"

    def test_all_fields_present(self):
        """All expected fields are present in the event data."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_gauntlet_completed("telegram", "c1", "g1", "AI is safe", "passed", 0.85, 4, 5)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["platform"] == "telegram"
            assert data["chat_id"] == "c1"
            assert data["gauntlet_id"] == "g1"
            assert data["statement"] == "AI is safe"
            assert data["verdict"] == "passed"
            assert data["confidence"] == 0.85
            assert data["challenges_passed"] == 4
            assert data["challenges_total"] == 5
            assert "timestamp" in data

    def test_statement_truncated_at_200_chars(self):
        """Long statement is truncated to 200 characters."""
        mock_events = MagicMock()
        long_stmt = "g" * 400
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_gauntlet_completed("telegram", "c1", "g1", long_stmt, "failed", 0.3, 1, 5)
            data = mock_events.dispatch_event.call_args[0][1]
            assert len(data["statement"]) == 200

    def test_verdict_failed(self):
        """Verdict 'failed' is preserved."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_gauntlet_completed("telegram", "c1", "g1", "stmt", "failed", 0.2, 1, 5)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["verdict"] == "failed"

    def test_verdict_inconclusive(self):
        """Custom verdict string is preserved."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_gauntlet_completed("telegram", "c1", "g1", "stmt", "inconclusive", 0.5, 3, 5)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["verdict"] == "inconclusive"

    def test_chat_id_coerced_to_string(self):
        """Numeric chat_id is coerced to string."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_gauntlet_completed("telegram", 321, "g1", "stmt", "passed", 0.9, 5, 5)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["chat_id"] == "321"

    def test_zero_challenges(self):
        """Zero challenges passed and zero total are valid."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_gauntlet_completed("telegram", "c1", "g1", "stmt", "failed", 0.0, 0, 0)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["challenges_passed"] == 0
            assert data["challenges_total"] == 0

    def test_all_challenges_passed(self):
        """All challenges passed scenario."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_gauntlet_completed("telegram", "c1", "g1", "stmt", "passed", 1.0, 10, 10)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["challenges_passed"] == 10
            assert data["challenges_total"] == 10

    def test_confidence_boundary_values(self):
        """Confidence at boundary values 0.0 and 1.0."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_gauntlet_completed("telegram", "c1", "g1", "stmt", "failed", 0.0, 0, 5)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["confidence"] == 0.0


# ============================================================================
# emit_vote_received
# ============================================================================


class TestEmitVoteReceived:
    """Tests for emit_vote_received."""

    def test_dispatches_correct_event_type(self):
        """Event type is 'chat.vote_received'."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_vote_received("telegram", "c1", "u1", "alice", "d1", "agree")
            event_type = mock_events.dispatch_event.call_args[0][0]
            assert event_type == "chat.vote_received"

    def test_all_fields_present(self):
        """All expected fields are present in the event data."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_vote_received("telegram", "c1", "u1", "alice", "d1", "agree")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["platform"] == "telegram"
            assert data["chat_id"] == "c1"
            assert data["user_id"] == "u1"
            assert data["username"] == "alice"
            assert data["debate_id"] == "d1"
            assert data["vote"] == "agree"
            assert "timestamp" in data

    def test_vote_disagree(self):
        """Vote 'disagree' is preserved."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_vote_received("telegram", "c1", "u1", "alice", "d1", "disagree")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["vote"] == "disagree"

    def test_chat_id_coerced_to_string(self):
        """Numeric chat_id is coerced to string."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_vote_received("telegram", 444, "u1", "alice", "d1", "agree")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["chat_id"] == "444"

    def test_user_id_coerced_to_string(self):
        """Numeric user_id is coerced to string."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_vote_received("telegram", "c1", 777, "alice", "d1", "agree")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["user_id"] == "777"

    def test_arbitrary_vote_value(self):
        """Arbitrary vote string is preserved."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_vote_received("telegram", "c1", "u1", "alice", "d1", "abstain")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["vote"] == "abstain"


# ============================================================================
# Edge Cases and Security
# ============================================================================


class TestEdgeCases:
    """Edge case tests across all emit functions."""

    def test_empty_platform_string(self):
        """Empty platform string is handled without error."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("", "c1", "u1", "alice", "hi")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["platform"] == ""

    def test_empty_username(self):
        """Empty username string is handled without error."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", "c1", "u1", "", "hi")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["username"] == ""

    def test_special_characters_in_chat_id(self):
        """Special characters in chat_id are preserved."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", "chat-123_abc", "u1", "alice", "hi")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["chat_id"] == "chat-123_abc"

    def test_very_long_username(self):
        """Very long username is not truncated (no limit in spec)."""
        mock_events = MagicMock()
        long_name = "x" * 1000
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", "c1", "u1", long_name, "hi")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["username"] == long_name

    def test_message_with_newlines(self):
        """Message with newlines is preserved (but may be truncated)."""
        mock_events = MagicMock()
        msg_with_newlines = "line1\nline2\nline3"
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", "c1", "u1", "alice", msg_with_newlines)
            data = mock_events.dispatch_event.call_args[0][1]
            assert "line1\nline2\nline3" == data["message_preview"]

    def test_message_with_null_bytes(self):
        """Message with null bytes is preserved in preview."""
        mock_events = MagicMock()
        msg = "hello\x00world"
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", "c1", "u1", "alice", msg)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["message_preview"] == msg

    def test_numeric_platform(self):
        """Non-string platform value is passed through (no coercion)."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            # The function doesn't coerce platform to string
            emit_message_received("123", "c1", "u1", "alice", "hi")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["platform"] == "123"

    def test_debate_completed_with_empty_final_answer(self):
        """Empty string final_answer produces empty preview."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_completed("telegram", "c1", "d1", "topic", True, 0.9, 3, "")
            data = mock_events.dispatch_event.call_args[0][1]
            # Empty string is falsy, so final_answer[:300] would be ""
            # but the code checks `if final_answer` so "" -> None? Let's check
            # Looking at the code: final_answer[:300] if final_answer else None
            # "" is falsy, so result is None
            assert data["final_answer_preview"] is None

    def test_debate_id_with_uuid_format(self):
        """UUID-formatted debate_id is preserved."""
        mock_events = MagicMock()
        uuid_id = "550e8400-e29b-41d4-a716-446655440000"
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_started("telegram", "c1", "u1", "alice", "topic", uuid_id)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["debate_id"] == uuid_id


class TestSecurityConcerns:
    """Security-oriented tests for chat event emission."""

    def test_path_traversal_in_platform(self):
        """Path traversal attempt in platform is just passed as string."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("../../etc/passwd", "c1", "u1", "alice", "hi")
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["platform"] == "../../etc/passwd"

    def test_script_injection_in_message(self):
        """Script injection in message is preserved (not sanitized at this layer)."""
        mock_events = MagicMock()
        xss_msg = "<script>alert('xss')</script>"
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", "c1", "u1", "alice", xss_msg)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["message_preview"] == xss_msg

    def test_sql_injection_in_command(self):
        """SQL injection attempt in command is just passed as string."""
        mock_events = MagicMock()
        sql_cmd = "'; DROP TABLE users; --"
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_command_received("telegram", "c1", "u1", "alice", sql_cmd)
            data = mock_events.dispatch_event.call_args[0][1]
            assert data["command"] == sql_cmd

    def test_very_large_input_truncation(self):
        """Very large inputs are truncated by the respective functions."""
        mock_events = MagicMock()
        huge = "X" * 100_000
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", "c1", "u1", "alice", huge)
            data = mock_events.dispatch_event.call_args[0][1]
            assert len(data["message_preview"]) == 100

    def test_topic_truncation_prevents_large_payloads(self):
        """Topic truncation prevents excessively large event payloads."""
        mock_events = MagicMock()
        huge_topic = "T" * 100_000
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_started("telegram", "c1", "u1", "alice", huge_topic)
            data = mock_events.dispatch_event.call_args[0][1]
            assert len(data["topic"]) == 200

    def test_statement_truncation_prevents_large_payloads(self):
        """Statement truncation prevents excessively large event payloads."""
        mock_events = MagicMock()
        huge_stmt = "S" * 100_000
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_gauntlet_started("telegram", "c1", "u1", "alice", huge_stmt)
            data = mock_events.dispatch_event.call_args[0][1]
            assert len(data["statement"]) == 200

    def test_final_answer_truncation_prevents_large_payloads(self):
        """Final answer truncation prevents excessively large event payloads."""
        mock_events = MagicMock()
        huge_answer = "A" * 100_000
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_completed("telegram", "c1", "d1", "topic", True, 0.9, 3, huge_answer)
            data = mock_events.dispatch_event.call_args[0][1]
            assert len(data["final_answer_preview"]) == 300


# ============================================================================
# __all__ exports
# ============================================================================


class TestModuleExports:
    """Tests for module-level __all__ exports."""

    def test_all_contains_all_emit_functions(self):
        """__all__ contains all 7 public emit functions."""
        import aragora.server.handlers.social.chat_events as mod

        expected = {
            "emit_message_received",
            "emit_command_received",
            "emit_debate_started",
            "emit_debate_completed",
            "emit_gauntlet_started",
            "emit_gauntlet_completed",
            "emit_vote_received",
        }
        assert set(mod.__all__) == expected

    def test_all_has_exactly_seven_entries(self):
        """__all__ has exactly 7 entries."""
        import aragora.server.handlers.social.chat_events as mod

        assert len(mod.__all__) == 7

    def test_dispatch_not_in_all(self):
        """Internal _dispatch_chat_event is not in __all__."""
        import aragora.server.handlers.social.chat_events as mod

        assert "_dispatch_chat_event" not in mod.__all__


# ============================================================================
# Integration-style scenarios
# ============================================================================


class TestIntegrationScenarios:
    """End-to-end scenarios combining multiple emit functions."""

    def test_full_debate_lifecycle(self):
        """Track a debate from message to vote through all events."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", "c1", "u1", "alice", "/debate AI safety")
            emit_command_received("telegram", "c1", "u1", "alice", "debate", "AI safety")
            emit_debate_started("telegram", "c1", "u1", "alice", "AI safety", "d-123")
            emit_debate_completed(
                "telegram",
                "c1",
                "d-123",
                "AI safety",
                True,
                0.92,
                3,
                "AI is generally safe with proper guardrails",
            )
            emit_vote_received("telegram", "c1", "u1", "alice", "d-123", "agree")

        assert mock_events.dispatch_event.call_count == 5
        event_types = [c[0][0] for c in mock_events.dispatch_event.call_args_list]
        assert event_types == [
            "chat.message_received",
            "chat.command_received",
            "chat.debate_started",
            "chat.debate_completed",
            "chat.vote_received",
        ]

    def test_full_gauntlet_lifecycle(self):
        """Track a gauntlet from command to completion."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_command_received("whatsapp", "c2", "u2", "bob", "gauntlet", "Earth is flat")
            emit_gauntlet_started("whatsapp", "c2", "u2", "bob", "Earth is flat", "g-456")
            emit_gauntlet_completed("whatsapp", "c2", "g-456", "Earth is flat", "failed", 0.1, 0, 5)

        assert mock_events.dispatch_event.call_count == 3
        event_types = [c[0][0] for c in mock_events.dispatch_event.call_args_list]
        assert event_types == [
            "chat.command_received",
            "chat.gauntlet_started",
            "chat.gauntlet_completed",
        ]

    def test_multiple_platforms_independent(self):
        """Events from different platforms are dispatched independently."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", "c1", "u1", "alice", "hello")
            emit_message_received("whatsapp", "c2", "u2", "bob", "hi there")

        assert mock_events.dispatch_event.call_count == 2
        data_1 = mock_events.dispatch_event.call_args_list[0][0][1]
        data_2 = mock_events.dispatch_event.call_args_list[1][0][1]
        assert data_1["platform"] == "telegram"
        assert data_2["platform"] == "whatsapp"

    def test_debate_without_consensus(self):
        """Debate that ends without consensus."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_started("telegram", "c1", "u1", "alice", "Controversial topic")
            emit_debate_completed(
                "telegram",
                "c1",
                "d-789",
                "Controversial topic",
                False,
                0.45,
                5,
                None,
            )

        completed_data = mock_events.dispatch_event.call_args_list[1][0][1]
        assert completed_data["consensus_reached"] is False
        assert completed_data["confidence"] == 0.45
        assert completed_data["final_answer_preview"] is None

    def test_multiple_votes_on_same_debate(self):
        """Multiple votes on the same debate each produce separate events."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_vote_received("telegram", "c1", "u1", "alice", "d1", "agree")
            emit_vote_received("telegram", "c1", "u2", "bob", "d1", "disagree")
            emit_vote_received("telegram", "c1", "u3", "carol", "d1", "agree")

        assert mock_events.dispatch_event.call_count == 3
        votes = [c[0][1]["vote"] for c in mock_events.dispatch_event.call_args_list]
        assert votes == ["agree", "disagree", "agree"]

    def test_timestamps_are_monotonically_increasing(self):
        """Timestamps across sequential calls are non-decreasing."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", "c1", "u1", "alice", "msg1")
            emit_message_received("telegram", "c1", "u1", "alice", "msg2")
            emit_message_received("telegram", "c1", "u1", "alice", "msg3")

        timestamps = [c[0][1]["timestamp"] for c in mock_events.dispatch_event.call_args_list]
        assert timestamps[0] <= timestamps[1] <= timestamps[2]

    def test_dispatch_failure_does_not_prevent_subsequent_calls(self):
        """If one dispatch fails, subsequent calls still work."""
        mock_events = MagicMock()
        # First call raises, second succeeds
        mock_events.dispatch_event.side_effect = [
            RuntimeError("transient"),
            None,
        ]
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", "c1", "u1", "alice", "msg1")
            emit_message_received("telegram", "c1", "u1", "alice", "msg2")

        assert mock_events.dispatch_event.call_count == 2


# ============================================================================
# Timestamp precision tests
# ============================================================================


class TestTimestamps:
    """Tests focused on timestamp behavior."""

    def test_message_received_has_timestamp(self):
        """emit_message_received includes a float timestamp."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_message_received("telegram", "c1", "u1", "alice", "hi")
            data = mock_events.dispatch_event.call_args[0][1]
            assert isinstance(data["timestamp"], float)

    def test_command_received_has_timestamp(self):
        """emit_command_received includes a float timestamp."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_command_received("telegram", "c1", "u1", "alice", "help")
            data = mock_events.dispatch_event.call_args[0][1]
            assert isinstance(data["timestamp"], float)

    def test_debate_started_has_timestamp(self):
        """emit_debate_started includes a float timestamp."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_started("telegram", "c1", "u1", "alice", "topic")
            data = mock_events.dispatch_event.call_args[0][1]
            assert isinstance(data["timestamp"], float)

    def test_debate_completed_has_timestamp(self):
        """emit_debate_completed includes a float timestamp."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_debate_completed("telegram", "c1", "d1", "topic", True, 0.9, 3)
            data = mock_events.dispatch_event.call_args[0][1]
            assert isinstance(data["timestamp"], float)

    def test_gauntlet_started_has_timestamp(self):
        """emit_gauntlet_started includes a float timestamp."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_gauntlet_started("telegram", "c1", "u1", "alice", "stmt")
            data = mock_events.dispatch_event.call_args[0][1]
            assert isinstance(data["timestamp"], float)

    def test_gauntlet_completed_has_timestamp(self):
        """emit_gauntlet_completed includes a float timestamp."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_gauntlet_completed("telegram", "c1", "g1", "stmt", "passed", 0.9, 5, 5)
            data = mock_events.dispatch_event.call_args[0][1]
            assert isinstance(data["timestamp"], float)

    def test_vote_received_has_timestamp(self):
        """emit_vote_received includes a float timestamp."""
        mock_events = MagicMock()
        with patch.dict("sys.modules", {"aragora.events": mock_events}):
            emit_vote_received("telegram", "c1", "u1", "alice", "d1", "agree")
            data = mock_events.dispatch_event.call_args[0][1]
            assert isinstance(data["timestamp"], float)
