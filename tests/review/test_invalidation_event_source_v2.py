"""Tests for v2 outcome-field handling in invalidation_event_source (#6375 phase 4)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aragora.review.invalidation import (
    INVALIDATION_HUMAN_OVERRIDE_REDO,
    INVALIDATION_POST_MERGE_INCIDENT,
    INVALIDATION_REOPENED_PR,
    INVALIDATION_REVERT_WITHIN_WINDOW,
    INVALIDATION_ROLLBACK,
)
from aragora.review.invalidation_event_source import (
    _any_receipt_has_v2_outcome_fields,
    _invalidation_from_settlement_receipt as _settlement_payload_to_invalidated_decision,
)

UTC = timezone.utc
RECEIPTS_SUBDIR = "receipts"


def _write_receipt(receipts_dir: Path, name: str, payload: dict) -> Path:
    receipts_dir.mkdir(parents=True, exist_ok=True)
    path = receipts_dir / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _base_payload(
    *,
    pr_number: int = 100,
    reviewed_at: str = "2026-04-15T12:00:00+00:00",
) -> dict:
    return {
        "session_id": "sess-1",
        "reviewed_at": reviewed_at,
        "actor": "armand",
        "action": "settle",
        "reason": "test",
        "pr_number": pr_number,
        "pr_url": f"https://github.com/org/repo/pull/{pr_number}",
        "head_sha": "abc1234",
        "base_sha": "000",
        "packet_sha": "p",
        "queue_bucket": "ready",
        "machine_recommendation": "fire_and_forget",
        "github_event": "merged",
    }


class TestPayloadToInvalidatedDecisionV2Fields:
    def test_v2_revert_signal_fires(self) -> None:
        payload = _base_payload()
        payload["outcome_revert_within_window"] = True
        payload["outcome_observed_at"] = "2026-04-22T00:00:00Z"
        decision = _settlement_payload_to_invalidated_decision(payload)
        assert decision is not None
        assert INVALIDATION_REVERT_WITHIN_WINDOW in decision.signals
        assert decision.was_human_settled is True

    def test_v2_incident_signal_fires(self) -> None:
        payload = _base_payload()
        payload["outcome_post_merge_incident"] = True
        payload["outcome_observed_at"] = "2026-04-22T00:00:00Z"
        decision = _settlement_payload_to_invalidated_decision(payload)
        assert decision is not None
        assert INVALIDATION_POST_MERGE_INCIDENT in decision.signals

    def test_v2_redo_signal_fires(self) -> None:
        payload = _base_payload()
        payload["outcome_human_override_redo"] = True
        decision = _settlement_payload_to_invalidated_decision(payload)
        assert decision is not None
        assert INVALIDATION_HUMAN_OVERRIDE_REDO in decision.signals

    def test_v2_rollback_signal_fires(self) -> None:
        payload = _base_payload()
        payload["outcome_rollback"] = True
        decision = _settlement_payload_to_invalidated_decision(payload)
        assert decision is not None
        assert INVALIDATION_ROLLBACK in decision.signals

    def test_v2_reopened_signal_fires(self) -> None:
        payload = _base_payload()
        payload["outcome_reopened_pr"] = True
        decision = _settlement_payload_to_invalidated_decision(payload)
        assert decision is not None
        assert INVALIDATION_REOPENED_PR in decision.signals

    def test_v2_all_false_returns_none(self) -> None:
        payload = _base_payload()
        payload["outcome_revert_within_window"] = False
        payload["outcome_post_merge_incident"] = False
        payload["outcome_human_override_redo"] = False
        payload["outcome_rollback"] = False
        payload["outcome_reopened_pr"] = False
        payload["outcome_observed_at"] = "2026-04-22T00:00:00Z"
        decision = _settlement_payload_to_invalidated_decision(payload)
        assert decision is None

    def test_v2_all_none_returns_none(self) -> None:
        # No outcome fields = no signals = None (denominator only)
        payload = _base_payload()
        decision = _settlement_payload_to_invalidated_decision(payload)
        assert decision is None

    def test_v2_multiple_signals_combine(self) -> None:
        payload = _base_payload()
        payload["outcome_revert_within_window"] = True
        payload["outcome_post_merge_incident"] = True
        decision = _settlement_payload_to_invalidated_decision(payload)
        assert decision is not None
        assert INVALIDATION_REVERT_WITHIN_WINDOW in decision.signals
        assert INVALIDATION_POST_MERGE_INCIDENT in decision.signals
        assert len(decision.signals) == 2

    def test_v2_takes_precedence_over_v1(self) -> None:
        # If both v1 and v2 fields are present, v2 wins (no double-counting).
        payload = _base_payload()
        payload["outcome_revert_within_window"] = True
        payload["reverted_at"] = "2026-04-20T10:00:00Z"  # v1 legacy field
        decision = _settlement_payload_to_invalidated_decision(payload)
        assert decision is not None
        revert_signals = [s for s in decision.signals if s == INVALIDATION_REVERT_WITHIN_WINDOW]
        assert len(revert_signals) == 1

    def test_v2_false_takes_precedence_over_matching_v1_fields(self) -> None:
        # Explicit v2 False means "observed and did not fire"; legacy v1 fields
        # must not resurrect the matching invalidation signals.
        payload = _base_payload()
        payload["outcome_revert_within_window"] = False
        payload["outcome_post_merge_incident"] = False
        payload["outcome_human_override_redo"] = False
        payload["outcome_observed_at"] = "2026-04-22T00:00:00Z"
        payload["reverted_at"] = "2026-04-20T10:00:00Z"
        payload["post_merge_incident"] = True
        payload["redo_pr"] = 123
        decision = _settlement_payload_to_invalidated_decision(payload)
        assert decision is None


class TestAnyReceiptHasV2OutcomeFields:
    def test_returns_false_when_no_receipts(self, tmp_path: Path) -> None:
        result = _any_receipt_has_v2_outcome_fields(
            store_root=tmp_path,
            window_end=datetime(2026, 4, 30, tzinfo=UTC),
            window_days=30,
        )
        assert result is False

    def test_returns_false_when_only_v1_receipts(self, tmp_path: Path) -> None:
        receipts_dir = tmp_path / RECEIPTS_SUBDIR
        _write_receipt(receipts_dir, "r1", _base_payload())
        _write_receipt(receipts_dir, "r2", _base_payload(pr_number=101))
        result = _any_receipt_has_v2_outcome_fields(
            store_root=tmp_path,
            window_end=datetime(2026, 4, 30, tzinfo=UTC),
            window_days=30,
        )
        assert result is False

    def test_returns_true_when_v2_field_populated(self, tmp_path: Path) -> None:
        receipts_dir = tmp_path / RECEIPTS_SUBDIR
        v2_payload = _base_payload()
        v2_payload["outcome_revert_within_window"] = False  # explicit False still counts
        v2_payload["outcome_observed_at"] = "2026-04-22T00:00:00Z"
        _write_receipt(receipts_dir, "r1", v2_payload)
        result = _any_receipt_has_v2_outcome_fields(
            store_root=tmp_path,
            window_end=datetime(2026, 4, 30, tzinfo=UTC),
            window_days=30,
        )
        assert result is True

    def test_v2_field_present_but_none_does_not_count(self, tmp_path: Path) -> None:
        receipts_dir = tmp_path / RECEIPTS_SUBDIR
        v2_payload = _base_payload()
        v2_payload["outcome_revert_within_window"] = None  # explicit None
        v2_payload["outcome_observed_at"] = None
        _write_receipt(receipts_dir, "r1", v2_payload)
        result = _any_receipt_has_v2_outcome_fields(
            store_root=tmp_path,
            window_end=datetime(2026, 4, 30, tzinfo=UTC),
            window_days=30,
        )
        assert result is False

    def test_outside_window_does_not_count(self, tmp_path: Path) -> None:
        receipts_dir = tmp_path / RECEIPTS_SUBDIR
        old_payload = _base_payload(reviewed_at="2025-12-01T12:00:00+00:00")
        old_payload["outcome_revert_within_window"] = True
        _write_receipt(receipts_dir, "old", old_payload)
        result = _any_receipt_has_v2_outcome_fields(
            store_root=tmp_path,
            window_end=datetime(2026, 4, 30, tzinfo=UTC),
            window_days=30,
        )
        assert result is False

    def test_probe_cap_bounds_scan(self, tmp_path: Path) -> None:
        receipts_dir = tmp_path / RECEIPTS_SUBDIR
        # 5 v1-only receipts, then 1 v2 receipt; deterministic name sorting means
        # a small cap scans only the first v1 receipts.
        for i in range(5):
            _write_receipt(receipts_dir, f"v1_{i}", _base_payload(pr_number=i))
        v2_payload = _base_payload(pr_number=99)
        v2_payload["outcome_revert_within_window"] = True
        _write_receipt(receipts_dir, "v2", v2_payload)
        result = _any_receipt_has_v2_outcome_fields(
            store_root=tmp_path,
            window_end=datetime(2026, 4, 30, tzinfo=UTC),
            window_days=30,
            probe_cap=3,
        )
        assert result is False

    def test_rejects_nonpositive_window_days(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            _any_receipt_has_v2_outcome_fields(
                store_root=tmp_path,
                window_end=datetime(2026, 4, 30, tzinfo=UTC),
                window_days=0,
            )
