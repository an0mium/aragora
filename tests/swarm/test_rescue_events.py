"""Tests for the RescueEvent ledger."""

from __future__ import annotations

import threading
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

    def test_concurrent_writes(self, ledger: RescueEventLedger) -> None:
        """Two threads writing in rapid succession should not lose events."""
        barrier = threading.Barrier(2)

        def _write(idx: int) -> None:
            barrier.wait()
            ledger.record(RescueEvent(event_type="other", reason=f"thread-{idx}"))

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        events = ledger.recent()
        assert len(events) == 2
        reasons = {e.reason for e in events}
        assert reasons == {"thread-0", "thread-1"}

    def test_malformed_jsonl_lines_skipped(self, ledger: RescueEventLedger) -> None:
        """Malformed lines in the JSONL file should be silently skipped."""
        ledger.record(RescueEvent(event_type="followup_prompt", reason="good"))
        # Inject malformed lines directly into the file
        with ledger.path.open("a", encoding="utf-8") as f:
            f.write("NOT VALID JSON\n")
            f.write("{}\n")  # missing required fields — still parseable
            f.write('{"event_type": "other", "reason": "also good"}\n')
        events = ledger.recent()
        # First good event + empty-dict event (defaults) + last good event
        assert any(e.reason == "good" for e in events)
        assert any(e.reason == "also good" for e in events)

    def test_large_ledger_performance(self, ledger: RescueEventLedger) -> None:
        """Ledger with 1000+ events should still return results correctly."""
        count = 1200
        # Write all events in bulk for speed
        ledger.path.parent.mkdir(parents=True, exist_ok=True)
        with ledger.path.open("a", encoding="utf-8") as f:
            for i in range(count):
                import json as _json

                evt = RescueEvent(event_type="other", reason=f"evt-{i}")
                f.write(_json.dumps(evt.to_dict(), sort_keys=True) + "\n")

        # Default limit 50
        events = ledger.recent()
        assert len(events) == 50
        assert events[-1].reason == f"evt-{count - 1}"

        # Explicit large limit
        all_events = ledger.recent(limit=2000)
        assert len(all_events) == count


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

    def test_enum_completeness(self) -> None:
        """All expected rescue event types should be present."""
        expected = {
            "followup_prompt",
            "permission_approval",
            "issue_rewrite",
            "issue_requeue",
            "session_restart",
            "session_kill",
            "pr_shepherd",
            "blocked_escalate",
            "manual_merge",
            "worktree_cleanup",
            "copy_paste_relay",
            "other",
        }
        actual = {member.value for member in RescueEventType}
        assert actual == expected

    def test_enum_is_str_subclass(self) -> None:
        """RescueEventType members should be usable as plain strings."""
        assert isinstance(RescueEventType.OTHER, str)
        assert RescueEventType.OTHER == "other"
