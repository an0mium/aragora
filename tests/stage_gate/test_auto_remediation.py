"""Tests for :mod:`aragora.stage_gate.auto_remediation`.

The dispatcher is intentionally conservative: it should only attempt a
repair when (a) the feature flag is on, (b) the drift pattern is
recognised, and (c) the evidence contains the expected keys.  These tests
exercise both the happy paths and every guard rail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from aragora.stage_gate.auto_remediation import (
    AUTO_REMEDIATION_FLAG_ENV,
    RemediationResult,
    auto_remediation_enabled,
    remediate_drift,
)


# ---------------------------------------------------------------------------
# Recording stub for the RemediationActions protocol
# ---------------------------------------------------------------------------


@dataclass
class RecordingActions:
    """Collects every action call for assertions, returns configurable results."""

    remove_label_result: bool = True
    post_comment_result: bool = True
    trigger_workflow_result: bool = True
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    def remove_label(self, issue_number: int, label: str) -> bool:
        self.calls.append(("remove_label", (issue_number, label), {}))
        return self.remove_label_result

    def post_comment(self, issue_number: int, body: str) -> bool:
        self.calls.append(("post_comment", (issue_number, body), {}))
        return self.post_comment_result

    def trigger_workflow(self, workflow: str, *, ref: str | None = None) -> bool:
        self.calls.append(("trigger_workflow", (workflow,), {"ref": ref}))
        return self.trigger_workflow_result


FLAG_ENV = {AUTO_REMEDIATION_FLAG_ENV: "1"}


# ---------------------------------------------------------------------------
# Feature-flag guard
# ---------------------------------------------------------------------------


def test_auto_remediation_enabled_respects_truthy_values() -> None:
    assert auto_remediation_enabled({AUTO_REMEDIATION_FLAG_ENV: "1"}) is True
    assert auto_remediation_enabled({AUTO_REMEDIATION_FLAG_ENV: "true"}) is True
    assert auto_remediation_enabled({AUTO_REMEDIATION_FLAG_ENV: "YES"}) is True


def test_auto_remediation_enabled_defaults_off() -> None:
    assert auto_remediation_enabled({}) is False
    assert auto_remediation_enabled({AUTO_REMEDIATION_FLAG_ENV: "0"}) is False
    assert auto_remediation_enabled({AUTO_REMEDIATION_FLAG_ENV: "false"}) is False
    assert auto_remediation_enabled({AUTO_REMEDIATION_FLAG_ENV: "  "}) is False


def test_flag_disabled_short_circuits_dispatcher() -> None:
    actions = RecordingActions()
    result = remediate_drift(
        "boss_ready_on_deferred_track",
        {"issue_number": 42, "track": "deferred-cleanup"},
        actions=actions,
        env={},  # flag missing ⇒ disabled
    )

    assert isinstance(result, RemediationResult)
    assert result.applied is False
    assert result.pattern == "flag_disabled"
    assert actions.calls == []


# ---------------------------------------------------------------------------
# Recognised pattern: boss-ready on deferred track
# ---------------------------------------------------------------------------


def test_boss_ready_on_deferred_track_strips_label_and_comments() -> None:
    actions = RecordingActions()
    evidence = {
        "issue_number": 501,
        "track": "AGT-deferred",
        "deferred_label": "track:agt-deferred",
    }

    result = remediate_drift(
        "boss_ready_on_deferred_track",
        evidence,
        actions=actions,
        env=FLAG_ENV,
    )

    assert result.applied is True
    assert result.drift_type == "boss_ready_on_deferred_track"
    assert result.pattern == "strip_boss_ready_and_comment"
    action_kinds = [action.kind for action in result.actions]
    assert action_kinds == ["remove_label", "post_comment"]
    assert ("remove_label", (501, "boss-ready"), {}) in actions.calls
    comment_calls = [call for call in actions.calls if call[0] == "post_comment"]
    assert comment_calls
    assert "AGT-deferred" in comment_calls[0][1][1]
    assert "boss-ready" in comment_calls[0][1][1]


def test_boss_ready_on_deferred_track_thin_evidence_is_noop() -> None:
    actions = RecordingActions()
    result = remediate_drift(
        "boss_ready_on_deferred_track",
        {"issue_number": 501},  # track missing
        actions=actions,
        env=FLAG_ENV,
    )

    assert result.applied is False
    assert result.pattern == "thin_evidence"
    assert actions.calls == []


def test_boss_ready_on_deferred_track_adapter_rejection_is_noop() -> None:
    actions = RecordingActions(remove_label_result=False, post_comment_result=False)
    result = remediate_drift(
        "boss_ready_on_deferred_track",
        {"issue_number": 501, "track": "deferred"},
        actions=actions,
        env=FLAG_ENV,
    )

    assert result.applied is False
    assert result.pattern == "no_actions_taken"
    # Both calls were attempted but returned False
    kinds = [call[0] for call in actions.calls]
    assert kinds == ["remove_label", "post_comment"]


# ---------------------------------------------------------------------------
# Recognised pattern: duplicate epic
# ---------------------------------------------------------------------------


def test_duplicate_epic_cross_links_both_sides() -> None:
    actions = RecordingActions()
    result = remediate_drift(
        "duplicate_epic",
        {"epic_numbers": (700, 701)},
        actions=actions,
        env=FLAG_ENV,
    )

    assert result.applied is True
    assert result.pattern == "cross_link_and_flag"
    post_calls = [call for call in actions.calls if call[0] == "post_comment"]
    assert len(post_calls) == 2
    targets = {call[1][0] for call in post_calls}
    assert targets == {700, 701}
    # Each comment must reference the sibling epic
    assert "#701" in post_calls[0][1][1] or "#701" in post_calls[1][1][1]
    assert "#700" in post_calls[0][1][1] or "#700" in post_calls[1][1][1]


def test_duplicate_epic_requires_pair() -> None:
    actions = RecordingActions()
    result = remediate_drift(
        "duplicate_epic",
        {"epic_numbers": (700,)},  # only one number
        actions=actions,
        env=FLAG_ENV,
    )

    assert result.applied is False
    assert result.pattern == "thin_evidence"
    assert actions.calls == []


def test_duplicate_epic_rejects_non_iterable_evidence() -> None:
    actions = RecordingActions()
    result = remediate_drift(
        "duplicate_epic",
        {"epic_numbers": "700,701"},
        actions=actions,
        env=FLAG_ENV,
    )

    assert result.applied is False
    assert result.pattern == "thin_evidence"


# ---------------------------------------------------------------------------
# Recognised pattern: doc staleness
# ---------------------------------------------------------------------------


def test_doc_staleness_triggers_workflow_with_default_ref() -> None:
    actions = RecordingActions()
    result = remediate_drift(
        "doc_staleness",
        {"issue_number": 42},
        actions=actions,
        env=FLAG_ENV,
    )

    assert result.applied is True
    assert result.pattern == "rerun_publication_workflow"
    trigger_calls = [call for call in actions.calls if call[0] == "trigger_workflow"]
    assert len(trigger_calls) == 1
    assert trigger_calls[0][1] == ("benchmark-publication",)
    assert trigger_calls[0][2] == {"ref": None}
    # Audit comment posted on the referenced issue
    comment_calls = [call for call in actions.calls if call[0] == "post_comment"]
    assert comment_calls
    assert comment_calls[0][1][0] == 42


def test_doc_staleness_uses_custom_workflow_and_ref() -> None:
    actions = RecordingActions()
    result = remediate_drift(
        "doc_staleness",
        {"workflow": "corpus-publish", "ref": "release/april"},
        actions=actions,
        env=FLAG_ENV,
    )

    assert result.applied is True
    trigger_calls = [call for call in actions.calls if call[0] == "trigger_workflow"]
    assert trigger_calls[0][1] == ("corpus-publish",)
    assert trigger_calls[0][2] == {"ref": "release/april"}


def test_doc_staleness_noop_when_workflow_refuses() -> None:
    actions = RecordingActions(trigger_workflow_result=False)
    result = remediate_drift(
        "doc_staleness",
        {},  # no issue number, so comment is skipped
        actions=actions,
        env=FLAG_ENV,
    )

    assert result.applied is False
    assert result.pattern == "no_actions_taken"


# ---------------------------------------------------------------------------
# Unknown / malformed drift input
# ---------------------------------------------------------------------------


def test_unrecognised_drift_is_noop() -> None:
    actions = RecordingActions()
    result = remediate_drift(
        "mystery_drift",
        {"any": "thing"},
        actions=actions,
        env=FLAG_ENV,
    )

    assert result.applied is False
    assert result.pattern == "unrecognised_drift"
    assert actions.calls == []


@pytest.mark.parametrize("bad", ["", None, "   "])
def test_empty_drift_type_is_noop(bad: Any) -> None:
    actions = RecordingActions()
    result = remediate_drift(bad, {}, actions=actions, env=FLAG_ENV)

    assert result.applied is False
    assert result.pattern == "missing_drift_type"
    assert actions.calls == []


def test_missing_actions_adapter_is_noop_even_when_flag_is_on() -> None:
    result = remediate_drift(
        "boss_ready_on_deferred_track",
        {"issue_number": 1, "track": "deferred"},
        actions=None,
        env=FLAG_ENV,
    )

    assert result.applied is False
    assert result.pattern == "no_actions_adapter"


def test_result_serialisation_round_trips_to_dict() -> None:
    actions = RecordingActions()
    result = remediate_drift(
        "boss_ready_on_deferred_track",
        {"issue_number": 1, "track": "deferred"},
        actions=actions,
        env=FLAG_ENV,
    )
    payload = result.to_dict()

    assert payload["applied"] is True
    assert payload["drift_type"] == "boss_ready_on_deferred_track"
    assert payload["pattern"] == "strip_boss_ready_and_comment"
    assert payload["evidence_keys"] == ["issue_number", "track"]
    for action in payload["actions"]:
        assert set(action.keys()) == {"kind", "target", "detail"}
