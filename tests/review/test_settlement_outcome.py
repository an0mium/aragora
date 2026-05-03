"""Tests for settlement-receipt outcome observation (#6375 phase 4)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aragora.cli.commands.review_queue import SettlementReceipt
from aragora.review.invalidation import (
    INVALIDATION_HUMAN_OVERRIDE_REDO,
    INVALIDATION_POST_MERGE_INCIDENT,
    INVALIDATION_REOPENED_PR,
    INVALIDATION_REVERT_WITHIN_WINDOW,
    INVALIDATION_ROLLBACK,
    INVALIDATION_SIGNALS,
)
from aragora.review.settlement_outcome import (
    INCIDENT_LABELS,
    ROLLBACK_LABELS,
    ObservationWindow,
    observe_outcome,
)

UTC = timezone.utc


def _base_receipt(*, reviewed_at: str = "2026-04-15T12:00:00+00:00") -> SettlementReceipt:
    return SettlementReceipt(
        session_id="sess-1",
        reviewed_at=reviewed_at,
        actor="armand",
        action="settle",
        reason="test",
        pr_number=1234,
        pr_url="https://github.com/org/repo/pull/1234",
        head_sha="abc1234567890def1234567890abc1234567890d",
        base_sha="000000000000000000000000000000000000",
        packet_sha="packet-abc",
        queue_bucket="ready",
        machine_recommendation="fire_and_forget",
        github_event="merged",
    )


# --- ObservationWindow ---------------------------------------------------


class TestObservationWindow:
    def test_constructs_with_positive_days(self) -> None:
        settled = datetime(2026, 4, 15, 12, tzinfo=UTC)
        window = ObservationWindow(settled, 14)
        assert window.start == settled
        assert window.end == settled + timedelta(days=14)

    def test_rejects_naive_datetime(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            ObservationWindow(datetime(2026, 4, 15, 12), 14)

    def test_rejects_nonpositive_days(self) -> None:
        settled = datetime(2026, 4, 15, 12, tzinfo=UTC)
        with pytest.raises(ValueError, match="must be positive"):
            ObservationWindow(settled, 0)
        with pytest.raises(ValueError, match="must be positive"):
            ObservationWindow(settled, -3)


# --- observe_outcome ----------------------------------------------------


class TestObserveOutcomeNoSignals:
    def test_empty_timeline_yields_all_false(self) -> None:
        receipt = _base_receipt()
        observed = datetime(2026, 4, 30, 12, tzinfo=UTC)
        out = observe_outcome(receipt, github_timeline=[], observed_at=observed)
        assert out.outcome_revert_within_window is False
        assert out.outcome_post_merge_incident is False
        assert out.outcome_human_override_redo is False
        assert out.outcome_rollback is False
        assert out.outcome_reopened_pr is False
        assert out.outcome_observed_at == "2026-04-30T12:00:00Z"

    def test_observed_at_normalizes_aware_timestamp_to_utc(self) -> None:
        receipt = _base_receipt()
        observed = datetime(2026, 4, 30, 7, tzinfo=timezone(timedelta(hours=-5)))
        out = observe_outcome(receipt, github_timeline=[], observed_at=observed)
        assert out.outcome_observed_at == "2026-04-30T12:00:00Z"

    def test_does_not_mutate_input(self) -> None:
        receipt = _base_receipt()
        observe_outcome(receipt, github_timeline=[])
        assert receipt.outcome_revert_within_window is None
        assert receipt.outcome_observed_at is None


class TestObserveOutcomeRevertSignal:
    def test_revert_commit_in_window(self) -> None:
        receipt = _base_receipt()
        timeline = [
            {
                "type": "commit",
                "at": "2026-04-20T08:00:00+00:00",
                "message": 'Revert "feat: thing" — refs abc1234',
            },
        ]
        out = observe_outcome(receipt, github_timeline=timeline)
        assert out.outcome_revert_within_window is True

    def test_revert_outside_window_does_not_fire(self) -> None:
        receipt = _base_receipt()
        timeline = [
            {
                "type": "commit",
                "at": "2026-05-15T08:00:00+00:00",
                "message": 'Revert "feat: thing" — refs abc1234',
            },
        ]
        out = observe_outcome(receipt, github_timeline=timeline, window_days=14)
        assert out.outcome_revert_within_window is False

    def test_revert_without_sha_match_does_not_fire(self) -> None:
        receipt = _base_receipt()
        timeline = [
            {
                "type": "commit",
                "at": "2026-04-20T08:00:00+00:00",
                "message": 'Revert "unrelated change" — refs deadbee',
            },
        ]
        out = observe_outcome(receipt, github_timeline=timeline)
        assert out.outcome_revert_within_window is False


class TestObserveOutcomeIncidentSignal:
    def test_incident_label_with_pr_ref(self) -> None:
        receipt = _base_receipt()
        timeline = [
            {
                "type": "issue_opened",
                "at": "2026-04-22T10:00:00+00:00",
                "labels": ["incident"],
                "title": "API outage attributed to #1234",
                "body": "rollback recommended",
            },
        ]
        out = observe_outcome(receipt, github_timeline=timeline)
        assert out.outcome_post_merge_incident is True

    def test_incident_label_without_ref_does_not_fire(self) -> None:
        receipt = _base_receipt()
        timeline = [
            {
                "type": "issue_opened",
                "at": "2026-04-22T10:00:00+00:00",
                "labels": ["incident"],
                "title": "Unrelated outage",
                "body": "no merge ref",
            },
        ]
        out = observe_outcome(receipt, github_timeline=timeline)
        assert out.outcome_post_merge_incident is False

    def test_non_incident_label_does_not_fire(self) -> None:
        receipt = _base_receipt()
        timeline = [
            {
                "type": "issue_opened",
                "at": "2026-04-22T10:00:00+00:00",
                "labels": ["question"],
                "title": "ref to #1234",
                "body": "",
            },
        ]
        out = observe_outcome(receipt, github_timeline=timeline)
        assert out.outcome_post_merge_incident is False


class TestObserveOutcomeRedoSignal:
    def test_follow_up_pr_with_closes_keyword(self) -> None:
        receipt = _base_receipt()
        timeline = [
            {
                "type": "pr_opened",
                "at": "2026-04-20T15:00:00+00:00",
                "title": "fix follow-up",
                "body": "Closes #1234",
                "labels": [],
            },
        ]
        out = observe_outcome(receipt, github_timeline=timeline)
        assert out.outcome_human_override_redo is True

    def test_follow_up_with_supersedes_keyword(self) -> None:
        receipt = _base_receipt()
        timeline = [
            {
                "type": "pr_opened",
                "at": "2026-04-21T09:00:00+00:00",
                "title": "supersede prior PR",
                "body": "supersedes #1234",
                "labels": [],
            },
        ]
        out = observe_outcome(receipt, github_timeline=timeline)
        assert out.outcome_human_override_redo is True


class TestObserveOutcomeRollbackSignal:
    def test_rollback_label(self) -> None:
        receipt = _base_receipt()
        timeline = [
            {
                "type": "pr_opened",
                "at": "2026-04-22T11:00:00+00:00",
                "title": "feature flag off",
                "body": "",
                "labels": ["rollback"],
            },
        ]
        out = observe_outcome(receipt, github_timeline=timeline)
        assert out.outcome_rollback is True

    def test_revert_title_prefix(self) -> None:
        receipt = _base_receipt()
        timeline = [
            {
                "type": "pr_opened",
                "at": "2026-04-22T11:00:00+00:00",
                "title": "Revert feature flag for safety",
                "body": "",
                "labels": [],
            },
        ]
        out = observe_outcome(receipt, github_timeline=timeline)
        assert out.outcome_rollback is True

    def test_feature_flag_rollback_label(self) -> None:
        receipt = _base_receipt()
        timeline = [
            {
                "type": "pr_opened",
                "at": "2026-04-22T11:00:00+00:00",
                "title": "Disable launch",
                "body": "",
                "labels": ["feature-flag-rollback"],
            },
        ]
        out = observe_outcome(receipt, github_timeline=timeline)
        assert out.outcome_rollback is True


class TestObserveOutcomeReopenedSignal:
    def test_reopened_event_for_pr(self) -> None:
        receipt = _base_receipt()
        timeline = [
            {
                "type": "pr_reopened",
                "at": "2026-04-20T10:00:00+00:00",
                "pr_number": 1234,
            },
        ]
        out = observe_outcome(receipt, github_timeline=timeline)
        assert out.outcome_reopened_pr is True

    def test_reopened_for_other_pr_does_not_fire(self) -> None:
        receipt = _base_receipt()
        timeline = [
            {
                "type": "pr_reopened",
                "at": "2026-04-20T10:00:00+00:00",
                "pr_number": 9999,
            },
        ]
        out = observe_outcome(receipt, github_timeline=timeline)
        assert out.outcome_reopened_pr is False


class TestObserveOutcomeMultipleSignals:
    def test_pr_can_fire_multiple_signals(self) -> None:
        receipt = _base_receipt()
        timeline = [
            {
                "type": "commit",
                "at": "2026-04-20T08:00:00+00:00",
                "message": 'Revert "feat: thing" — refs abc1234',
            },
            {
                "type": "issue_opened",
                "at": "2026-04-22T10:00:00+00:00",
                "labels": ["incident"],
                "title": "regression in #1234",
                "body": "",
            },
            {
                "type": "pr_reopened",
                "at": "2026-04-21T10:00:00+00:00",
                "pr_number": 1234,
            },
        ]
        out = observe_outcome(receipt, github_timeline=timeline)
        assert out.outcome_revert_within_window is True
        assert out.outcome_post_merge_incident is True
        assert out.outcome_reopened_pr is True
        assert out.outcome_rollback is False
        assert out.outcome_human_override_redo is False


class TestSchemaInvariants:
    def test_module_canonical_signal_set_matches_invalidation(self) -> None:
        from aragora.review import settlement_outcome

        # The module-level assert at import time should pass; this test ensures
        # the contract is checked even if pytest is run with -O.
        expected = {
            INVALIDATION_REVERT_WITHIN_WINDOW,
            INVALIDATION_POST_MERGE_INCIDENT,
            INVALIDATION_HUMAN_OVERRIDE_REDO,
            INVALIDATION_ROLLBACK,
            INVALIDATION_REOPENED_PR,
        }
        assert expected == INVALIDATION_SIGNALS
        assert settlement_outcome._EXPECTED_SIGNALS == INVALIDATION_SIGNALS

    def test_incident_labels_pinned(self) -> None:
        # Pin INCIDENT_LABELS so adding a new label is an explicit acknowledgement.
        assert INCIDENT_LABELS == frozenset(
            {"incident", "regression", "revert-target", "boss-stuck"}
        )

    def test_rollback_labels_pinned(self) -> None:
        assert ROLLBACK_LABELS == frozenset({"rollback", "feature-flag-rollback"})


class TestSettlementReceiptSchemaV2:
    def test_default_outcome_fields_are_none(self) -> None:
        receipt = _base_receipt()
        assert receipt.outcome_revert_within_window is None
        assert receipt.outcome_post_merge_incident is None
        assert receipt.outcome_human_override_redo is None
        assert receipt.outcome_rollback is None
        assert receipt.outcome_reopened_pr is None
        assert receipt.outcome_observed_at is None

    def test_to_dict_round_trip_preserves_outcome_fields(self) -> None:
        receipt = _base_receipt()
        out = observe_outcome(
            receipt,
            github_timeline=[],
            observed_at=datetime(2026, 4, 30, 12, tzinfo=UTC),
        )
        d = out.to_dict()
        assert d["outcome_revert_within_window"] is False
        assert d["outcome_observed_at"] == "2026-04-30T12:00:00Z"

    def test_legacy_receipt_without_outcome_fields_still_parses(self) -> None:
        # SettlementReceipt has all-keyword defaults for the v2 fields; old
        # callers that construct receipts positionally up through receipt_path
        # must still work.
        receipt = SettlementReceipt(
            session_id="sess-1",
            reviewed_at="2026-04-15T12:00:00+00:00",
            actor="armand",
            action="settle",
            reason="test",
            pr_number=1234,
            pr_url="https://github.com/org/repo/pull/1234",
            head_sha="abc",
            base_sha="000",
            packet_sha="p",
            queue_bucket="ready",
            machine_recommendation="fire_and_forget",
            github_event="merged",
            elapsed_seconds=1.5,
            receipt_path="/tmp/x.json",
        )
        d = receipt.to_dict()
        assert d["outcome_revert_within_window"] is None
        assert d["outcome_observed_at"] is None
