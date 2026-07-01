"""M0 Review Adjudicator (#8748) — acceptance tests.

The adjudicator fires ONLY on a stalled review (quorum not satisfied, dissent
present, at least one supportive signal). It scores each disputed finding's
groundedness and decides:

- any [P0]/[P1] present            -> BLOCK      (hard bar; never suppressed)
- all dissent thin (below bar)     -> SETTLE     (the treadmill escape)
- grounded dissent, no grounded support -> BLOCK  (evidence-backed nit stands)
- grounded dissent AND grounded support -> ESCALATE (material two-sided crux)
- no supportive signal / no dissent -> NOT_APPLICABLE (not a stall)

Default is flag-OFF (byte-identical to today).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from aragora.swarm.review_adjudicator import (
    AdjudicationVerdict,
    adjudicate,
    review_adjudicator_enabled,
    score_groundedness,
)


@dataclass
class _Item:
    """Minimal EvidenceItem-shaped stand-in (family, body, verdict, supportive)."""

    family: str
    body: str
    verdict: str  # "pass" | "changes_requested"

    @property
    def supportive(self) -> bool:
        return self.verdict == "pass"


def _pass(family: str, body: str = "Verdict: PASS\nLGTM.") -> _Item:
    return _Item(family=family, body=body, verdict="pass")


_GROUNDED_P2 = (
    "Verdict: CHANGES-REQUESTED\n"
    "- [P2] The retry loop in aragora/foo.py:42 has no ceiling; a persistent 500 "
    "response spins forever (e.g. 12 retries in 3s). Repro: mock a 500, observe "
    "unbounded retries. Fix: cap at max_retries."
)
_THIN_P3 = (
    "Verdict: CHANGES-REQUESTED\n"
    "- [P3] Consider adding more tests here in a follow-up; it would generally be "
    "nice to have and might help."
)
_BLOCKING_P1 = (
    "Verdict: CHANGES-REQUESTED\n"
    "- [P1] aragora/gate.py:88 bypasses the auth check when tenant is None; a null "
    "tenant merges without settlement."
)


class TestGroundednessScorer:
    def test_grounded_scores_above_thin(self) -> None:
        assert score_groundedness(_GROUNDED_P2) > score_groundedness(_THIN_P3)

    def test_thin_below_default_bar(self) -> None:
        assert score_groundedness(_THIN_P3) < 0.5

    def test_grounded_above_default_bar(self) -> None:
        assert score_groundedness(_GROUNDED_P2) >= 0.5


class TestAdjudicate:
    def test_all_thin_dissent_settles(self) -> None:
        # #8748 acceptance: advisory-only, thin dissent, a supportive signal -> SETTLE.
        items = [_pass("claude"), _Item("grok", _THIN_P3, "changes_requested")]
        r = adjudicate(items)
        assert r.verdict is AdjudicationVerdict.SETTLE
        assert r.settled_findings  # the thin finding is filed for follow-up
        assert not r.blocking_findings

    def test_evidence_backed_advisory_blocks(self) -> None:
        # #8748 acceptance: a grounded [P2] (concrete repro) is NOT suppressed.
        items = [_pass("claude"), _Item("openai", _GROUNDED_P2, "changes_requested")]
        r = adjudicate(items)
        assert r.verdict is AdjudicationVerdict.BLOCK
        assert r.blocking_findings

    def test_blocking_p1_always_blocks(self) -> None:
        # The hard bar is inviolate: any [P0]/[P1] -> BLOCK regardless of scores.
        items = [_pass("claude"), _Item("openai", _BLOCKING_P1, "changes_requested")]
        r = adjudicate(items)
        assert r.verdict is AdjudicationVerdict.BLOCK

    def test_material_disagreement_escalates(self) -> None:
        # #8748 acceptance: grounded dissent AND grounded support -> ESCALATE.
        grounded_support = _pass(
            "claude",
            "Verdict: PASS\nVerified aragora/foo.py:42 caps retries at max_retries=5; "
            "reproduced the 500 path and it terminates after 5 attempts in 1.2s.",
        )
        items = [grounded_support, _Item("openai", _GROUNDED_P2, "changes_requested")]
        r = adjudicate(items)
        assert r.verdict is AdjudicationVerdict.ESCALATE
        assert r.escalated_findings

    def test_no_supportive_signal_is_not_applicable(self) -> None:
        # Zero support is genuine rejection, not a stall — adjudicator abstains.
        items = [_Item("openai", _GROUNDED_P2, "changes_requested")]
        r = adjudicate(items)
        assert r.verdict is AdjudicationVerdict.NOT_APPLICABLE

    def test_no_dissent_is_not_applicable(self) -> None:
        items = [_pass("claude"), _pass("openai")]
        r = adjudicate(items)
        assert r.verdict is AdjudicationVerdict.NOT_APPLICABLE

    def test_receipt_dict_is_serializable(self) -> None:
        import json

        items = [_pass("claude"), _Item("grok", _THIN_P3, "changes_requested")]
        r = adjudicate(items)
        payload = json.dumps(r.to_receipt_dict())
        assert "adjudicated_settle" in payload
        assert "assessments" in payload


class TestFlag:
    def test_flag_default_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ARAGORA_ENABLE_REVIEW_ADJUDICATOR", raising=False)
        assert review_adjudicator_enabled() is False

    def test_flag_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_ENABLE_REVIEW_ADJUDICATOR", "1")
        assert review_adjudicator_enabled() is True
