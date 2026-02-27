"""Tests for interrogation WebSocket event types."""

import pytest

from aragora.events.types import StreamEventType


class TestInterrogationEvents:
    def test_interrogation_event_types_exist(self):
        assert hasattr(StreamEventType, "INTERROGATION_STARTED")
        assert hasattr(StreamEventType, "INTERROGATION_QUESTION")
        assert hasattr(StreamEventType, "INTERROGATION_ANSWER")
        assert hasattr(StreamEventType, "INTERROGATION_CRYSTALLIZED")

    def test_interrogation_event_values(self):
        assert StreamEventType.INTERROGATION_STARTED.value == "interrogation_started"
        assert StreamEventType.INTERROGATION_QUESTION.value == "interrogation_question"
        assert StreamEventType.INTERROGATION_ANSWER.value == "interrogation_answer"
        assert StreamEventType.INTERROGATION_CRYSTALLIZED.value == "interrogation_crystallized"
