"""Unit tests for debate origin data models."""

from __future__ import annotations

import pytest

from aragora.server.debate_origin import models
from aragora.server.debate_origin.models import DebateOrigin


def test_debate_origin_defaults_are_populated() -> None:
    before = models.time.time()
    origin = DebateOrigin(
        debate_id="debate-1",
        platform="slack",
        channel_id="channel-1",
        user_id="user-1",
    )
    after = models.time.time()

    assert before <= origin.created_at <= after
    assert origin.metadata == {}
    assert origin.thread_id is None
    assert origin.message_id is None
    assert origin.session_id is None
    assert origin.result_sent is False
    assert origin.result_sent_at is None


def test_to_dict_includes_required_and_optional_fields() -> None:
    origin = DebateOrigin(
        debate_id="debate-1",
        platform="teams",
        channel_id="channel-1",
        user_id="user-1",
        created_at=42.0,
        metadata={"priority": "high"},
        thread_id="thread-1",
        message_id="message-1",
        session_id="session-1",
        result_sent=True,
        result_sent_at=99.0,
    )

    assert origin.to_dict() == {
        "debate_id": "debate-1",
        "platform": "teams",
        "channel_id": "channel-1",
        "user_id": "user-1",
        "created_at": 42.0,
        "metadata": {"priority": "high"},
        "thread_id": "thread-1",
        "message_id": "message-1",
        "session_id": "session-1",
        "result_sent": True,
        "result_sent_at": 99.0,
    }


def test_to_dict_returns_metadata_copy() -> None:
    origin = DebateOrigin(
        debate_id="debate-1",
        platform="email",
        channel_id="inbox",
        user_id="user-1",
        metadata={"source": "triage"},
    )

    serialized = origin.to_dict()
    serialized["metadata"]["source"] = "changed"

    assert origin.metadata == {"source": "triage"}


def test_from_dict_restores_full_payload() -> None:
    origin = DebateOrigin.from_dict(
        {
            "debate_id": "debate-1",
            "platform": "discord",
            "channel_id": "channel-1",
            "user_id": "user-1",
            "created_at": 10.0,
            "metadata": {"guild": "guild-1"},
            "thread_id": "thread-1",
            "message_id": "message-1",
            "session_id": "session-1",
            "result_sent": True,
            "result_sent_at": 12.5,
        }
    )

    assert origin == DebateOrigin(
        debate_id="debate-1",
        platform="discord",
        channel_id="channel-1",
        user_id="user-1",
        created_at=10.0,
        metadata={"guild": "guild-1"},
        thread_id="thread-1",
        message_id="message-1",
        session_id="session-1",
        result_sent=True,
        result_sent_at=12.5,
    )


def test_from_dict_defaults_missing_optional_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models.time, "time", lambda: 5678.0)

    origin = DebateOrigin.from_dict(
        {
            "debate_id": "debate-1",
            "platform": "web",
            "channel_id": "browser",
            "user_id": "user-1",
        }
    )

    assert origin.created_at == 5678.0
    assert origin.metadata == {}
    assert origin.thread_id is None
    assert origin.message_id is None
    assert origin.session_id is None
    assert origin.result_sent is False
    assert origin.result_sent_at is None


def test_from_dict_preserves_false_result_sent() -> None:
    origin = DebateOrigin.from_dict(
        {
            "debate_id": "debate-1",
            "platform": "telegram",
            "channel_id": "chat-1",
            "user_id": "user-1",
            "created_at": 10.0,
            "result_sent": False,
        }
    )

    assert origin.result_sent is False


def test_from_dict_converts_none_metadata_to_empty_dict() -> None:
    origin = DebateOrigin.from_dict(
        {
            "debate_id": "debate-1",
            "platform": "slack",
            "channel_id": "channel-1",
            "user_id": "user-1",
            "created_at": 10.0,
            "metadata": None,
        }
    )

    assert origin.metadata == {}


def test_from_dict_copies_metadata() -> None:
    payload = {
        "debate_id": "debate-1",
        "platform": "slack",
        "channel_id": "channel-1",
        "user_id": "user-1",
        "created_at": 10.0,
        "metadata": {"key": "value"},
    }

    origin = DebateOrigin.from_dict(payload)
    payload["metadata"]["key"] = "changed"

    assert origin.metadata == {"key": "value"}


def test_from_dict_missing_required_field_raises_key_error() -> None:
    with pytest.raises(KeyError, match="debate_id"):
        DebateOrigin.from_dict(
            {
                "platform": "slack",
                "channel_id": "channel-1",
                "user_id": "user-1",
            }
        )


def test_from_dict_invalid_metadata_raises_type_error() -> None:
    with pytest.raises(TypeError):
        DebateOrigin.from_dict(
            {
                "debate_id": "debate-1",
                "platform": "slack",
                "channel_id": "channel-1",
                "user_id": "user-1",
                "metadata": 42,
            }
        )
