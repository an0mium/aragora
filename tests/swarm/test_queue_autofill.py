"""Tests for the pure queue-autofill data types.

The autonomous dispatcher (``maybe_autofill_queue``) that lived in this module
was removed as part of the batched-triage alignment
(``docs/plans/2026-04-19-batched-pr-review-triage.md``).  These tests now only
cover the passive data types that survived the declaw.
"""

from __future__ import annotations

from aragora.swarm.queue_autofill import (
    ALLOWED_CATEGORIES,
    DEFAULT_EMPTY_TICK_THRESHOLD,
    DEFAULT_MAX_ISSUES,
    DEFAULT_MIN_INTERVAL_SECONDS,
    AutofillCandidate,
    AutofillResult,
)


class TestAllowedCategories:
    def test_includes_the_two_well_proven_categories(self):
        assert "test_coverage" in ALLOWED_CATEGORIES
        assert "broad_exception" in ALLOWED_CATEGORIES

    def test_is_immutable(self):
        assert isinstance(ALLOWED_CATEGORIES, frozenset)


class TestDefaults:
    def test_defaults_are_conservative(self):
        assert DEFAULT_EMPTY_TICK_THRESHOLD >= 1
        assert DEFAULT_MAX_ISSUES >= 1
        assert DEFAULT_MIN_INTERVAL_SECONDS > 0


class TestAutofillCandidate:
    def test_to_dict_roundtrips_required_fields(self):
        candidate = AutofillCandidate(
            title="add missing coverage for tranche_queue",
            category="test_coverage",
            fingerprint="abc123",
            file_scope=("aragora/swarm/tranche_queue.py",),
            lane="test_authoring",
        )
        payload = candidate.to_dict()
        assert payload == {
            "title": "add missing coverage for tranche_queue",
            "category": "test_coverage",
            "fingerprint": "abc123",
            "file_scope": ["aragora/swarm/tranche_queue.py"],
            "lane": "test_authoring",
        }


class TestAutofillResult:
    def test_created_count_reflects_created_tuple_length(self):
        candidate = AutofillCandidate(
            title="t",
            category="test_coverage",
            fingerprint="f",
            file_scope=(),
            lane="lane",
        )
        result = AutofillResult(
            attempted=True,
            reason="advisory_only",
            consecutive_empty_ticks=3,
            threshold=3,
            created=(candidate,),
        )
        assert result.created_count == 1

    def test_to_dict_includes_event_tag(self):
        result = AutofillResult(
            attempted=False,
            reason="below_threshold",
            consecutive_empty_ticks=1,
            threshold=3,
        )
        payload = result.to_dict()
        assert payload["event"] == "queue_autofill"
        assert payload["attempted"] is False
        assert payload["created_count"] == 0
        assert payload["created"] == []

    def test_defaults_mirror_advisory_only_model(self):
        result = AutofillResult(
            attempted=False,
            reason="flag_removed",
            consecutive_empty_ticks=0,
            threshold=3,
        )
        assert result.rate_limited is False
        assert result.seconds_since_last is None
        assert result.scanned_count == 0
        assert result.eligible_count == 0
        assert result.created == ()
        assert result.errors == ()
