"""Tests for the RescueEvent ledger."""

from __future__ import annotations

from pathlib import Path

import pytest

from aragora.swarm.rescue_events import (
    RescueEvent,
    RescueEventLedger,
    RescueEventType,
    record_rescue,
)


@pytest.fixture()
def ledger(tmp_path: Path) -> RescueEventLedger:
    return RescueEventLedger(path=tmp_path / "rescue_events.jsonl")


class TestRescueEvent:
    def test_event_has_timestamp(self) -> None:
        event = RescueEvent(event_type="followup_prompt", reason="session stalled")
        assert event.created_at
        assert "T" in event.created_at

    def test_event_to_dict(self) -> None:
        event = RescueEvent(
            event_type="issue_rewrite",
            reason="sanitizer false positive",
            issue_number=5348,
        )
        d = event.to_dict()
        assert d["event_type"] == "issue_rewrite"
        assert d["issue_number"] == 5348
        assert "created_at" in d

    def test_event_to_dict_omits_none(self) -> None:
        event = RescueEvent(event_type="other", reason="test")
        d = event.to_dict()
        assert "issue_number" not in d
        assert "pr_number" not in d


class TestRescueEventLedger:
    def test_record_and_read(self, ledger: RescueEventLedger) -> None:
        ledger.record(RescueEvent(event_type="followup_prompt", reason="stalled"))
        ledger.record(RescueEvent(event_type="permission_approval", reason="git fetch"))
        events = ledger.recent()
        assert len(events) == 2
        assert events[0].event_type == "followup_prompt"
        assert events[1].event_type == "permission_approval"

    def test_recent_limits(self, ledger: RescueEventLedger) -> None:
        for i in range(10):
            ledger.record(RescueEvent(event_type="other", reason=f"reason {i}"))
        events = ledger.recent(limit=3)
        assert len(events) == 3
        assert events[0].reason == "reason 7"

    def test_count_by_type(self, ledger: RescueEventLedger) -> None:
        ledger.record(RescueEvent(event_type="followup_prompt", reason="a"))
        ledger.record(RescueEvent(event_type="followup_prompt", reason="b"))
        ledger.record(RescueEvent(event_type="issue_rewrite", reason="c"))
        counts = ledger.count_by_type()
        assert counts["followup_prompt"] == 2
        assert counts["issue_rewrite"] == 1

    def test_repeated_classes(self, ledger: RescueEventLedger) -> None:
        ledger.record(RescueEvent(event_type="followup_prompt", reason="session stalled"))
        ledger.record(RescueEvent(event_type="followup_prompt", reason="session stalled"))
        ledger.record(RescueEvent(event_type="followup_prompt", reason="session stalled"))
        ledger.record(RescueEvent(event_type="issue_rewrite", reason="unique reason"))
        repeated = ledger.repeated_classes(threshold=2)
        assert len(repeated) == 1
        assert repeated[0]["class"] == "followup_prompt:session stalled"
        assert repeated[0]["count"] == 3

    def test_empty_ledger_returns_empty(self, ledger: RescueEventLedger) -> None:
        assert ledger.recent() == []
        assert ledger.count_by_type() == {}
        assert ledger.repeated_classes() == []


class TestRecordRescueConvenience:
    def test_convenience_function(self, tmp_path: Path) -> None:
        path = tmp_path / "rescue.jsonl"
        record_rescue(
            "issue_requeue",
            "sanitizer false positive on contradictory_scope",
            issue_number=5348,
            ledger_path=path,
        )
        ledger = RescueEventLedger(path=path)
        events = ledger.recent()
        assert len(events) == 1
        assert events[0].event_type == "issue_requeue"
        assert events[0].issue_number == 5348


class TestRescueEventType:
    def test_enum_values(self) -> None:
        assert RescueEventType.FOLLOWUP_PROMPT.value == "followup_prompt"
        assert RescueEventType.COPY_PASTE_RELAY.value == "copy_paste_relay"
