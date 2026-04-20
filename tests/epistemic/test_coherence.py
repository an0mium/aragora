"""Unit tests for DIC-26 coherence monitor (aragora.epistemic.coherence)."""

from __future__ import annotations

import pytest

from aragora.epistemic.coherence import (
    BeliefEntry,
    CoherenceIssue,
    CoherenceReport,
    IncoherenceKind,
    coherence_monitor_enabled,
    from_belief_node,
    scan_coherence,
)


def _e(
    bid: str,
    subject: str = "claim.default",
    confidence: float = 0.8,
    status: str = "pass",
    evidence_paths: tuple[str, ...] = (),
) -> BeliefEntry:
    return BeliefEntry(bid, subject, confidence, status, evidence_paths)


# --- flag gate ---


def test_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARAGORA_COHERENCE_MONITOR_ENABLED", raising=False)
    assert not coherence_monitor_enabled()


def test_enabled_by_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    for val in ("1", "true", "yes", "on"):
        monkeypatch.setenv("ARAGORA_COHERENCE_MONITOR_ENABLED", val)
        assert coherence_monitor_enabled()


def test_disabled_returns_empty_report() -> None:
    report = scan_coherence([_e("b1")], enabled=False)
    assert report.enabled is False and report.scanned == 1 and report.coherent


def test_enabled_kwarg_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARAGORA_COHERENCE_MONITOR_ENABLED", raising=False)
    assert scan_coherence([], enabled=True).enabled is True


# --- empty / single ---


def test_empty_ledger_is_coherent() -> None:
    report = scan_coherence([], enabled=True)
    assert report.coherent and report.scanned == 0 and report.to_dict()["issue_count"] == 0


def test_single_entry_no_issues() -> None:
    assert scan_coherence([_e("b1")], enabled=True).coherent


# --- contradiction ---


def test_contradiction_opposing_confidence() -> None:
    hi = _e("b-hi", subject="claim.x", confidence=0.95)
    lo = _e("b-lo", subject="claim.x", confidence=0.05)
    report = scan_coherence([hi, lo], enabled=True)
    assert report.contradiction_count == 1
    issue = next(i for i in report.issues if i.kind == IncoherenceKind.CONTRADICTION)
    assert "b-hi" in issue.belief_ids and "b-lo" in issue.belief_ids
    assert issue.severity == "error"


def test_no_contradiction_consistent_or_different_subjects() -> None:
    # same subject, both high confidence → no contradiction
    a = _e("b-a", subject="claim.y", confidence=0.8)
    b = _e("b-b", subject="claim.y", confidence=0.75)
    assert scan_coherence([a, b], enabled=True).contradiction_count == 0
    # different subjects, one high one low → no contradiction
    hi = _e("b1", subject="alpha", confidence=0.95)
    lo = _e("b2", subject="beta", confidence=0.05)
    assert scan_coherence([hi, lo], enabled=True).contradiction_count == 0


# --- evidence conflict ---


def test_evidence_conflict_mixed_outcomes() -> None:
    passing = _e("b-pass", status="pass", evidence_paths=("docs/status/foo.md",))
    failing = _e("b-fail", status="fail", evidence_paths=("docs/status/foo.md",))
    report = scan_coherence([passing, failing], enabled=True)
    assert report.evidence_conflict_count >= 1
    issue = next(i for i in report.issues if i.kind == IncoherenceKind.EVIDENCE_CONFLICT)
    assert "b-pass" in issue.belief_ids and "b-fail" in issue.belief_ids


def test_no_conflict_consistent_evidence() -> None:
    a = _e("b-a", status="pass", evidence_paths=("docs/status/bar.md",))
    b = _e("b-b", status="pass", evidence_paths=("docs/status/bar.md",))
    assert scan_coherence([a, b], enabled=True).evidence_conflict_count == 0


# --- confidence rot ---


def test_confidence_rot_below_threshold() -> None:
    low = _e("b-low", confidence=0.1)
    report = scan_coherence([low], min_confidence=0.3, enabled=True)
    rot = [i for i in report.issues if i.kind == IncoherenceKind.CONFIDENCE_ROT]
    assert len(rot) == 1 and rot[0].belief_ids == ("b-low",)


def test_no_rot_above_threshold_and_error_severity_below_half() -> None:
    ok = _e("b-ok", confidence=0.8)
    assert scan_coherence([ok], min_confidence=0.3, enabled=True).confidence_rot_count == 0
    very_low = _e("b-vl", confidence=0.05)
    report = scan_coherence([very_low], min_confidence=0.3, enabled=True)
    rot = next(i for i in report.issues if i.kind == IncoherenceKind.CONFIDENCE_ROT)
    assert rot.severity == "error"


# --- aggregate + serialisation ---


def test_report_not_coherent_with_issues() -> None:
    hi = _e("b-hi", subject="s", confidence=0.95)
    lo = _e("b-lo", subject="s", confidence=0.05)
    report = scan_coherence([hi, lo], enabled=True)
    assert not report.coherent and report.contradiction_count >= 1


def test_coherence_issue_to_dict() -> None:
    issue = CoherenceIssue(IncoherenceKind.CONTRADICTION, ("a", "b"), "test", "error")
    d = issue.to_dict()
    assert (
        d["kind"] == "contradiction" and d["belief_ids"] == ["a", "b"] and d["severity"] == "error"
    )


def test_belief_entry_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        BeliefEntry("b", "s", confidence=1.5)


def test_from_belief_node_duck_type() -> None:
    class _P:
        p_true = 0.72

    class _S:
        value = "updated"

    class _N:
        belief_id = "node-001"
        claim_id = "claim.abc"
        status = _S()
        posterior = _P()
        evidence_paths = ["docs/status/abc.md"]

    entry = from_belief_node(_N())
    assert entry.belief_id == "node-001"
    assert abs(entry.confidence - 0.72) < 1e-6
    assert entry.evidence_paths == ("docs/status/abc.md",)
