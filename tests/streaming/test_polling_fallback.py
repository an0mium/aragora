"""Tests for the polling fallback endpoint (_get_debate_events).

Covers:
- Events returned after given seq
- Empty buffer returns empty list
- Limit cap works
- Ordering is preserved
- Missing debate_id returns empty events
- No replay buffer returns empty events
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from aragora.server.handlers.debates.crud import CrudOperationsMixin


def _make_handler_mixin():
    """Create a CrudOperationsMixin instance with mock protocol methods."""

    class FakeHandler(CrudOperationsMixin):
        def __init__(self):
            self.ctx = {}

        def get_storage(self):
            return MagicMock()

        def read_json_body(self, handler, max_size=None):
            return {}

        def get_current_user(self, handler):
            return None

    return FakeHandler()


def _make_event_json(seq: int, debate_id: str = "debate-1") -> str:
    """Create a JSON string resembling a serialized StreamEvent."""
    return json.dumps(
        {
            "type": "agent_message",
            "seq": seq,
            "loop_id": debate_id,
            "data": {"content": f"message-{seq}"},
        }
    )


def _parse_result(result):
    """Parse a HandlerResult into (status_code, body_dict)."""
    return result.status_code, json.loads(result.body)


_PATCH_TARGET = "aragora.server.stream.debate_stream_server.get_global_replay_buffer"


class TestGetDebateEventsPolling:
    """Tests for the _get_debate_events handler method."""

    def test_events_returned_after_given_seq(self):
        mixin = _make_handler_mixin()
        mock_buffer = MagicMock()
        mock_buffer.replay_since.return_value = [
            _make_event_json(3),
            _make_event_json(4),
            _make_event_json(5),
        ]

        with patch(_PATCH_TARGET, return_value=mock_buffer):
            result = mixin._get_debate_events(None, "debate-1", since_seq=2, limit=100)

        status, body = _parse_result(result)
        assert status == 200
        assert len(body["events"]) == 3
        assert body["next_seq"] == 6  # last_seq(5) + 1
        mock_buffer.replay_since.assert_called_once_with("debate-1", 2)

    def test_empty_buffer_returns_empty_list(self):
        mixin = _make_handler_mixin()
        mock_buffer = MagicMock()
        mock_buffer.replay_since.return_value = []

        with patch(_PATCH_TARGET, return_value=mock_buffer):
            result = mixin._get_debate_events(None, "debate-1", since_seq=0, limit=100)

        status, body = _parse_result(result)
        assert status == 200
        assert body["events"] == []
        assert body["next_seq"] == 0  # since_seq when no events

    def test_limit_cap_works(self):
        mixin = _make_handler_mixin()
        mock_buffer = MagicMock()
        mock_buffer.replay_since.return_value = [_make_event_json(i) for i in range(1, 11)]

        with patch(_PATCH_TARGET, return_value=mock_buffer):
            result = mixin._get_debate_events(None, "debate-1", since_seq=0, limit=3)

        _, body = _parse_result(result)
        assert len(body["events"]) == 3
        seqs = [e["seq"] for e in body["events"]]
        assert seqs == [1, 2, 3]

    def test_limit_clamped_to_max_500(self):
        mixin = _make_handler_mixin()
        mock_buffer = MagicMock()
        mock_buffer.replay_since.return_value = [_make_event_json(i) for i in range(1, 600)]

        with patch(_PATCH_TARGET, return_value=mock_buffer):
            result = mixin._get_debate_events(None, "debate-1", since_seq=0, limit=9999)

        _, body = _parse_result(result)
        assert len(body["events"]) == 500

    def test_ordering_is_preserved(self):
        mixin = _make_handler_mixin()
        mock_buffer = MagicMock()
        mock_buffer.replay_since.return_value = [
            _make_event_json(5),
            _make_event_json(3),
            _make_event_json(7),
        ]

        with patch(_PATCH_TARGET, return_value=mock_buffer):
            result = mixin._get_debate_events(None, "debate-1", since_seq=0, limit=100)

        _, body = _parse_result(result)
        seqs = [e["seq"] for e in body["events"]]
        assert seqs == [5, 3, 7]  # insertion order preserved

    def test_missing_debate_returns_empty(self):
        mixin = _make_handler_mixin()
        mock_buffer = MagicMock()
        mock_buffer.replay_since.return_value = []

        with patch(_PATCH_TARGET, return_value=mock_buffer):
            result = mixin._get_debate_events(None, "nonexistent-debate", since_seq=0, limit=100)

        _, body = _parse_result(result)
        assert body["events"] == []
        mock_buffer.replay_since.assert_called_once_with("nonexistent-debate", 0)

    def test_no_replay_buffer_returns_empty(self):
        mixin = _make_handler_mixin()

        with patch(_PATCH_TARGET, return_value=None):
            result = mixin._get_debate_events(None, "debate-1", since_seq=0, limit=100)

        _, body = _parse_result(result)
        assert body["events"] == []
        assert body["next_seq"] == 0

    def test_default_params(self):
        mixin = _make_handler_mixin()
        mock_buffer = MagicMock()
        mock_buffer.replay_since.return_value = []

        with patch(_PATCH_TARGET, return_value=mock_buffer):
            result = mixin._get_debate_events(None, "debate-1")

        mock_buffer.replay_since.assert_called_once_with("debate-1", 0)
        _, body = _parse_result(result)
        assert body["events"] == []

    def test_malformed_json_in_buffer_skipped(self):
        mixin = _make_handler_mixin()
        mock_buffer = MagicMock()
        mock_buffer.replay_since.return_value = [
            _make_event_json(1),
            "not-valid-json{{{",
            _make_event_json(3),
        ]

        with patch(_PATCH_TARGET, return_value=mock_buffer):
            result = mixin._get_debate_events(None, "debate-1", since_seq=0, limit=100)

        _, body = _parse_result(result)
        assert len(body["events"]) == 2
        seqs = [e["seq"] for e in body["events"]]
        assert seqs == [1, 3]
