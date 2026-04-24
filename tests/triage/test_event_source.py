"""Tests for :mod:`aragora.triage.event_source`.

Exercises the adapter that reads settlement receipts from disk and
yields :class:`TriageDecisionEvent` instances. Uses a fixture tmp
directory as the review-queue store root so nothing depends on the
caller's actual ``.aragora/`` tree.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aragora.triage import event_source as es

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store_root(tmp_path: Path, monkeypatch) -> Path:
    """Redirect the store root at a temp dir via env var."""
    monkeypatch.setenv("ARAGORA_REVIEW_QUEUE_ROOT", str(tmp_path))
    (tmp_path / "receipts").mkdir()
    return tmp_path


def _write_receipt(root: Path, name: str, payload: dict) -> Path:
    path = root / "receipts" / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _receipt_payload(
    *,
    pr_number: int = 42,
    session_id: str = "20260422T000000Z",
    action: str = "approve",
    reviewed_at: str | None = None,
    machine_recommendation: str = "approve_candidate",
    queue_bucket: str = "ready_now",
    elapsed_seconds: float | None = 12.5,
) -> dict:
    payload: dict = {
        "session_id": session_id,
        "reviewed_at": reviewed_at or datetime.now(UTC).isoformat(),
        "actor": "armand",
        "action": action,
        "reason": "",
        "pr_number": pr_number,
        "pr_url": f"https://example.com/pr/{pr_number}",
        "head_sha": "a" * 40,
        "base_sha": "b" * 40,
        "packet_sha": "sha256:deadbeef",
        "queue_bucket": queue_bucket,
        "machine_recommendation": machine_recommendation,
        "github_event": action.upper(),
    }
    if elapsed_seconds is not None:
        payload["elapsed_seconds"] = elapsed_seconds
    return payload


# ---------------------------------------------------------------------------
# Single-payload adapter
# ---------------------------------------------------------------------------


class TestSinglePayload:
    def test_approve_with_default_queue_bucket(self):
        payload = _receipt_payload()
        event = es.event_from_settlement_receipt(payload)
        assert event is not None
        assert event.was_escalated is False
        assert event.was_auto_handled is False
        assert event.was_human_override is False
        assert event.ensemble_recommendation == "approve_candidate"
        assert event.settlement_duration_seconds == pytest.approx(12.5)
        assert event.final_outcome is None

    def test_request_changes_counts_as_escalation(self):
        payload = _receipt_payload(action="request_changes", queue_bucket="ready_now")
        event = es.event_from_settlement_receipt(payload)
        assert event is not None
        assert event.was_escalated is True

    def test_needs_attention_bucket_escalates(self):
        payload = _receipt_payload(queue_bucket="needs_attention")
        event = es.event_from_settlement_receipt(payload)
        assert event is not None
        assert event.was_escalated is True

    def test_approve_vs_reject_recommendation_is_override(self):
        payload = _receipt_payload(action="approve", machine_recommendation="needs_human_attention")
        event = es.event_from_settlement_receipt(payload)
        assert event is not None
        assert event.was_human_override is True

    def test_request_changes_over_approve_recommendation_is_override(self):
        payload = _receipt_payload(
            action="request_changes", machine_recommendation="approve_candidate"
        )
        event = es.event_from_settlement_receipt(payload)
        assert event is not None
        assert event.was_human_override is True

    def test_defer_is_not_override(self):
        payload = _receipt_payload(action="defer", machine_recommendation="approve_candidate")
        event = es.event_from_settlement_receipt(payload)
        assert event is not None
        assert event.was_human_override is False

    def test_missing_reviewed_at_returns_none(self):
        payload = _receipt_payload()
        payload.pop("reviewed_at")
        assert es.event_from_settlement_receipt(payload) is None

    def test_invalid_reviewed_at_returns_none(self):
        payload = _receipt_payload(reviewed_at="not-a-timestamp")
        assert es.event_from_settlement_receipt(payload) is None

    def test_negative_elapsed_seconds_dropped(self):
        payload = _receipt_payload(elapsed_seconds=-5.0)
        event = es.event_from_settlement_receipt(payload)
        assert event is not None
        assert event.settlement_duration_seconds is None

    def test_trailing_z_in_iso_timestamp_is_parsed(self):
        payload = _receipt_payload(reviewed_at="2026-04-22T12:00:00Z")
        event = es.event_from_settlement_receipt(payload)
        assert event is not None
        assert event.ts.tzinfo is not None


# ---------------------------------------------------------------------------
# Directory iteration
# ---------------------------------------------------------------------------


class TestIterFromStore:
    def test_iterates_every_json_receipt(self, store_root: Path):
        payload_a = _receipt_payload(pr_number=42, action="approve")
        payload_b = _receipt_payload(pr_number=43, action="request_changes")
        _write_receipt(store_root, "pr-42-a-approve.json", payload_a)
        _write_receipt(store_root, "pr-43-b-request_changes.json", payload_b)

        events = list(es.iter_events_from_store())
        assert len(events) == 2
        by_id = {e.decision_id: e for e in events}
        assert any("pr-42" in d for d in by_id)
        assert any("pr-43" in d for d in by_id)

    def test_skips_corrupt_receipt(self, store_root: Path):
        _write_receipt(
            store_root,
            "pr-42-a-approve.json",
            _receipt_payload(pr_number=42, action="approve"),
        )
        (store_root / "receipts" / "pr-99-broken.json").write_text(
            "{not valid json", encoding="utf-8"
        )
        events = list(es.iter_events_from_store())
        assert len(events) == 1

    def test_missing_receipts_dir_returns_nothing(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("ARAGORA_REVIEW_QUEUE_ROOT", str(tmp_path))
        assert list(es.iter_events_from_store()) == []

    def test_skips_non_json_files(self, store_root: Path):
        _write_receipt(
            store_root,
            "pr-42-a-approve.json",
            _receipt_payload(pr_number=42),
        )
        (store_root / "receipts" / "readme.txt").write_text("noop", encoding="utf-8")
        events = list(es.iter_events_from_store())
        assert len(events) == 1

    def test_explicit_override_takes_precedence_over_env(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("ARAGORA_REVIEW_QUEUE_ROOT", "/nonexistent")
        rooted = tmp_path / "explicit-store"
        (rooted / "receipts").mkdir(parents=True)
        _write_receipt(
            rooted,
            "pr-1-s-approve.json",
            _receipt_payload(pr_number=1, action="approve"),
        )
        events = list(es.iter_events_from_store(store_root=rooted))
        assert len(events) == 1


# ---------------------------------------------------------------------------
# Events feed compute_window end-to-end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_events_feed_metrics(self, store_root: Path):
        """Integration smoke: write 10 escalated receipts, ensure
        compute_window produces a non-None escalation rate."""
        from aragora.triage.metrics import compute_window

        now = datetime.now(UTC)
        for i in range(10):
            payload = _receipt_payload(
                pr_number=100 + i,
                session_id=f"s{i}",
                action="request_changes",
                reviewed_at=(now - timedelta(hours=i + 1)).isoformat(),
                queue_bucket="needs_attention",
                elapsed_seconds=30.0 + i,
            )
            _write_receipt(store_root, f"pr-{100 + i}-s{i}-request_changes.json", payload)

        events = list(es.iter_events_from_store())
        snapshot = compute_window(events, window_end=now, window_days=7)
        assert snapshot.total_decisions == 10
        assert snapshot.escalation_rate == 1.0
        assert snapshot.settlement_samples == 10
        # Median of durations 30..39 = 34.5
        assert snapshot.settlement_duration_median_s == pytest.approx(34.5)
