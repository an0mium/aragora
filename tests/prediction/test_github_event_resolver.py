"""Tests for the GitHub-event resolution adapter (AGT-04 sub-deliverable 2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aragora.prediction.github_event_resolver import (
    GitHubEventPayload,
    GitHubEventResolver,
    ResolutionResult,
)
from aragora.prediction.stakeable_claim import (
    InMemoryStakeableClaimStore,
    QuestionType,
    ResolutionStatus,
    StakeableClaim,
)

_FLAG = "ARAGORA_PREDICTION_MARKETS_ENABLED"
_NOW = datetime.now(tz=UTC)
_EVENT_TIME = _NOW.isoformat()
_FUTURE = (_NOW + timedelta(days=30)).isoformat()
_PAST = (_NOW - timedelta(days=1)).isoformat()
_AFTER_EXPIRY = (_NOW + timedelta(days=31)).isoformat()


@pytest.fixture(autouse=True)
def enable_flag(monkeypatch):
    monkeypatch.setenv(_FLAG, "1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_claim(
    claim_id: str = "c1",
    question_type: QuestionType = QuestionType.PR_MERGE,
    target_ref: str = "owner/repo#42",
) -> StakeableClaim:
    return StakeableClaim(
        claim_id=claim_id,
        question=f"Will {target_ref} happen?",
        question_type=question_type,
        target_ref=target_ref,
        expiry=_FUTURE,
    )


# ---------------------------------------------------------------------------
# can_resolve
# ---------------------------------------------------------------------------


class TestCanResolve:
    def test_pr_merge_event_matches(self):
        r = GitHubEventResolver()
        claim = _open_claim(question_type=QuestionType.PR_MERGE, target_ref="a/b#1")
        event = GitHubEventPayload(
            event_type="pull_request",
            action="closed",
            target_ref="a/b#1",
            occurred_at=_EVENT_TIME,
            merged=True,
        )
        assert r.can_resolve(claim, event)

    def test_target_ref_mismatch_returns_false(self):
        r = GitHubEventResolver()
        claim = _open_claim(target_ref="a/b#1")
        event = GitHubEventPayload(event_type="pull_request", action="closed", target_ref="a/b#99")
        assert not r.can_resolve(claim, event)

    def test_wrong_event_type_returns_false(self):
        r = GitHubEventResolver()
        claim = _open_claim(question_type=QuestionType.PR_MERGE)
        event = GitHubEventPayload(
            event_type="issues", action="closed", target_ref=claim.target_ref
        )
        assert not r.can_resolve(claim, event)

    def test_issue_close_event_matches(self):
        r = GitHubEventResolver()
        claim = _open_claim(question_type=QuestionType.ISSUE_CLOSE, target_ref="a/b#7")
        event = GitHubEventPayload(event_type="issues", action="closed", target_ref="a/b#7")
        assert r.can_resolve(claim, event)

    def test_ci_pass_check_run_matches(self):
        r = GitHubEventResolver()
        claim = _open_claim(question_type=QuestionType.CI_PASS, target_ref="a/b#5")
        event = GitHubEventPayload(
            event_type="check_run",
            action="completed",
            target_ref="a/b#5",
            occurred_at=_EVENT_TIME,
            conclusion="success",
        )
        assert r.can_resolve(claim, event)

    def test_workflow_run_matches_ci_pass(self):
        r = GitHubEventResolver()
        claim = _open_claim(question_type=QuestionType.CI_PASS, target_ref="a/b#5")
        event = GitHubEventPayload(
            event_type="workflow_run",
            action="completed",
            target_ref="a/b#5",
            occurred_at=_EVENT_TIME,
            conclusion="success",
        )
        assert r.can_resolve(claim, event)

    def test_unsupported_question_type_returns_false(self):
        r = GitHubEventResolver()
        claim = _open_claim(question_type=QuestionType.DEPENDENCY_RELEASE)
        event = GitHubEventPayload(
            event_type="release", action="published", target_ref=claim.target_ref
        )
        assert not r.can_resolve(claim, event)


# ---------------------------------------------------------------------------
# Flag gate
# ---------------------------------------------------------------------------


class TestFlagGate:
    def test_resolve_raises_when_flag_off(self, monkeypatch):
        monkeypatch.delenv(_FLAG, raising=False)
        r = GitHubEventResolver()
        claim = _open_claim()
        event = GitHubEventPayload(
            event_type="pull_request",
            action="closed",
            target_ref=claim.target_ref,
            occurred_at=_EVENT_TIME,
            merged=True,
        )
        with pytest.raises(RuntimeError, match="Prediction markets are disabled"):
            r.resolve_from_event(claim, event)

    def test_can_resolve_does_not_require_flag(self, monkeypatch):
        monkeypatch.delenv(_FLAG, raising=False)
        r = GitHubEventResolver()
        claim = _open_claim(question_type=QuestionType.PR_MERGE, target_ref="a/b#1")
        event = GitHubEventPayload(event_type="pull_request", action="closed", target_ref="a/b#1")
        # can_resolve is pure logic — must not raise
        assert r.can_resolve(claim, event)


# ---------------------------------------------------------------------------
# PR merge resolution
# ---------------------------------------------------------------------------


class TestPRMergeResolution:
    def test_merged_pr_resolves_yes(self):
        r = GitHubEventResolver()
        claim = _open_claim(target_ref="a/b#10")
        event = GitHubEventPayload(
            event_type="pull_request",
            action="closed",
            target_ref="a/b#10",
            occurred_at=_EVENT_TIME,
            merged=True,
        )
        result = r.resolve_from_event(claim, event)
        assert result.resolved is True
        assert result.resolution_value is True
        assert "merged" in result.evidence

    def test_closed_without_merge_before_expiry_waits(self):
        r = GitHubEventResolver()
        claim = _open_claim(target_ref="a/b#11")
        event = GitHubEventPayload(
            event_type="pull_request",
            action="closed",
            target_ref="a/b#11",
            occurred_at=_EVENT_TIME,
            merged=False,
        )
        result = r.resolve_from_event(claim, event)
        assert result.resolved is False
        assert result.resolution_value is False
        assert "can reopen" in result.evidence

    def test_opened_action_not_terminal(self):
        r = GitHubEventResolver()
        claim = _open_claim(target_ref="a/b#12")
        event = GitHubEventPayload(
            event_type="pull_request",
            action="opened",
            target_ref="a/b#12",
            occurred_at=_EVENT_TIME,
        )
        result = r.resolve_from_event(claim, event)
        assert result.resolved is False

    def test_already_resolved_claim_skipped(self):
        r = GitHubEventResolver()
        claim = StakeableClaim(
            claim_id="x1",
            question="?",
            question_type=QuestionType.PR_MERGE,
            target_ref="a/b#1",
            expiry=_FUTURE,
            resolution_status=ResolutionStatus.RESOLVED_YES,
            resolution_value=True,
        )
        event = GitHubEventPayload(
            event_type="pull_request",
            action="closed",
            target_ref="a/b#1",
            occurred_at=_EVENT_TIME,
            merged=True,
        )
        result = r.resolve_from_event(claim, event)
        assert result.resolved is False
        assert "already" in result.evidence

    def test_target_ref_mismatch_not_resolved(self):
        r = GitHubEventResolver()
        claim = _open_claim(target_ref="a/b#1")
        event = GitHubEventPayload(
            event_type="pull_request",
            action="closed",
            target_ref="a/b#999",
            occurred_at=_EVENT_TIME,
            merged=True,
        )
        result = r.resolve_from_event(claim, event)
        assert result.resolved is False

    def test_event_after_expiry_does_not_resolve(self):
        r = GitHubEventResolver()
        claim = StakeableClaim(
            claim_id="expired-pr",
            question="Will a/b#13 merge?",
            question_type=QuestionType.PR_MERGE,
            target_ref="a/b#13",
            expiry=_PAST,
        )
        event = GitHubEventPayload(
            event_type="pull_request",
            action="closed",
            target_ref="a/b#13",
            occurred_at=_EVENT_TIME,
            merged=True,
        )
        result = r.resolve_from_event(claim, event)
        assert result.resolved is False
        assert "after claim expiry" in result.evidence

    def test_missing_event_timestamp_does_not_resolve(self):
        r = GitHubEventResolver()
        claim = _open_claim(target_ref="a/b#14")
        event = GitHubEventPayload(
            event_type="pull_request",
            action="closed",
            target_ref="a/b#14",
            merged=True,
        )
        result = r.resolve_from_event(claim, event)
        assert result.resolved is False
        assert "timestamp is missing" in result.evidence


# ---------------------------------------------------------------------------
# Issue close resolution
# ---------------------------------------------------------------------------


class TestIssueCloseResolution:
    def test_issue_closed_resolves_yes(self):
        r = GitHubEventResolver()
        claim = _open_claim(question_type=QuestionType.ISSUE_CLOSE, target_ref="x/y#3")
        event = GitHubEventPayload(
            event_type="issues",
            action="closed",
            target_ref="x/y#3",
            occurred_at=_EVENT_TIME,
        )
        result = r.resolve_from_event(claim, event)
        assert result.resolved is True
        assert result.resolution_value is True
        assert "closed" in result.evidence

    def test_issue_reopened_not_terminal(self):
        r = GitHubEventResolver()
        claim = _open_claim(question_type=QuestionType.ISSUE_CLOSE, target_ref="x/y#4")
        event = GitHubEventPayload(
            event_type="issues",
            action="reopened",
            target_ref="x/y#4",
            occurred_at=_EVENT_TIME,
        )
        result = r.resolve_from_event(claim, event)
        assert result.resolved is False

    def test_issue_labeled_not_terminal(self):
        r = GitHubEventResolver()
        claim = _open_claim(question_type=QuestionType.ISSUE_CLOSE, target_ref="x/y#5")
        event = GitHubEventPayload(
            event_type="issues",
            action="labeled",
            target_ref="x/y#5",
            occurred_at=_EVENT_TIME,
        )
        result = r.resolve_from_event(claim, event)
        assert result.resolved is False


# ---------------------------------------------------------------------------
# CI pass resolution
# ---------------------------------------------------------------------------


class TestCIPassResolution:
    def test_check_run_success_resolves_yes(self):
        r = GitHubEventResolver()
        claim = _open_claim(question_type=QuestionType.CI_PASS, target_ref="p/q#5")
        event = GitHubEventPayload(
            event_type="check_run",
            action="completed",
            target_ref="p/q#5",
            occurred_at=_EVENT_TIME,
            conclusion="success",
            raw={"aggregate": True, "run_attempt": 1},
        )
        result = r.resolve_from_event(claim, event)
        assert result.resolved is True
        assert result.resolution_value is True
        assert "pass" in result.evidence

    def test_check_run_failure_resolves_no(self):
        r = GitHubEventResolver()
        claim = _open_claim(question_type=QuestionType.CI_PASS, target_ref="p/q#6")
        event = GitHubEventPayload(
            event_type="check_run",
            action="completed",
            target_ref="p/q#6",
            occurred_at=_EVENT_TIME,
            conclusion="failure",
            raw={"aggregate": True, "run_attempt": 1},
        )
        result = r.resolve_from_event(claim, event)
        assert result.resolved is True
        assert result.resolution_value is False
        assert "fail" in result.evidence

    def test_workflow_run_success_resolves_yes(self):
        r = GitHubEventResolver()
        claim = _open_claim(question_type=QuestionType.CI_PASS, target_ref="p/q#7")
        event = GitHubEventPayload(
            event_type="workflow_run",
            action="completed",
            target_ref="p/q#7",
            occurred_at=_EVENT_TIME,
            conclusion="success",
            raw={"aggregate": True, "run_attempt": 1},
        )
        result = r.resolve_from_event(claim, event)
        assert result.resolved is True
        assert result.resolution_value is True

    def test_check_run_queued_not_terminal(self):
        r = GitHubEventResolver()
        claim = _open_claim(question_type=QuestionType.CI_PASS, target_ref="p/q#8")
        event = GitHubEventPayload(
            event_type="check_run",
            action="queued",
            target_ref="p/q#8",
            occurred_at=_EVENT_TIME,
        )
        result = r.resolve_from_event(claim, event)
        assert result.resolved is False

    def test_check_run_cancelled_resolves_no(self):
        r = GitHubEventResolver()
        claim = _open_claim(question_type=QuestionType.CI_PASS, target_ref="p/q#9")
        event = GitHubEventPayload(
            event_type="check_run",
            action="completed",
            target_ref="p/q#9",
            occurred_at=_EVENT_TIME,
            conclusion="cancelled",
            raw={"aggregate": True, "run_attempt": 1},
        )
        result = r.resolve_from_event(claim, event)
        assert result.resolved is True
        assert result.resolution_value is False

    def test_single_check_run_without_aggregate_marker_waits(self):
        r = GitHubEventResolver()
        claim = _open_claim(question_type=QuestionType.CI_PASS, target_ref="p/q#10")
        event = GitHubEventPayload(
            event_type="check_run",
            action="completed",
            target_ref="p/q#10",
            occurred_at=_EVENT_TIME,
            conclusion="success",
        )
        result = r.resolve_from_event(claim, event)
        assert result.resolved is False
        assert "not marked as an aggregate" in result.evidence

    def test_rerun_ci_event_does_not_resolve_first_run_claim(self):
        r = GitHubEventResolver()
        claim = _open_claim(question_type=QuestionType.CI_PASS, target_ref="p/q#11")
        event = GitHubEventPayload(
            event_type="workflow_run",
            action="completed",
            target_ref="p/q#11",
            occurred_at=_EVENT_TIME,
            conclusion="success",
            raw={"aggregate": True, "run_attempt": 2},
        )
        result = r.resolve_from_event(claim, event)
        assert result.resolved is False
        assert "run_attempt=2" in result.evidence

    def test_ci_event_after_expiry_does_not_resolve(self):
        r = GitHubEventResolver()
        claim = StakeableClaim(
            claim_id="expired-ci",
            question="Will p/q#12 pass?",
            question_type=QuestionType.CI_PASS,
            target_ref="p/q#12",
            expiry=_FUTURE,
        )
        event = GitHubEventPayload(
            event_type="workflow_run",
            action="completed",
            target_ref="p/q#12",
            occurred_at=_AFTER_EXPIRY,
            conclusion="success",
            raw={"aggregate": True, "run_attempt": 1},
        )
        result = r.resolve_from_event(claim, event)
        assert result.resolved is False
        assert "after claim expiry" in result.evidence


# ---------------------------------------------------------------------------
# End-to-end: resolver + store
# ---------------------------------------------------------------------------


class TestResolverWithStore:
    def test_full_roundtrip_merge(self):
        r = GitHubEventResolver()
        store = InMemoryStakeableClaimStore()
        claim = _open_claim(claim_id="e2e-1", target_ref="a/b#99")
        store.add(claim)

        event = GitHubEventPayload(
            event_type="pull_request",
            action="closed",
            target_ref="a/b#99",
            occurred_at=_EVENT_TIME,
            merged=True,
        )
        result = r.resolve_from_event(store.get("e2e-1"), event)
        assert result.resolved
        store.resolve("e2e-1", result.resolution_value, result.evidence)
        resolved = store.get("e2e-1")
        assert resolved.resolution_status == ResolutionStatus.RESOLVED_YES
        assert resolved.resolution_value is True

    def test_full_roundtrip_ci_fail(self):
        r = GitHubEventResolver()
        store = InMemoryStakeableClaimStore()
        claim = _open_claim(
            claim_id="e2e-2",
            question_type=QuestionType.CI_PASS,
            target_ref="a/b#100",
        )
        store.add(claim)

        event = GitHubEventPayload(
            event_type="check_run",
            action="completed",
            target_ref="a/b#100",
            occurred_at=_EVENT_TIME,
            conclusion="failure",
            raw={"aggregate": True, "run_attempt": 1},
        )
        result = r.resolve_from_event(store.get("e2e-2"), event)
        assert result.resolved
        store.resolve("e2e-2", result.resolution_value, result.evidence)
        resolved = store.get("e2e-2")
        assert resolved.resolution_status == ResolutionStatus.RESOLVED_NO
        assert resolved.resolution_value is False
