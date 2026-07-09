"""M0 Review Adjudicator (#8748) — acceptance tests.

The adjudicator fires ONLY on a stalled review (quorum not satisfied, dissent
present, at least one supportive signal). It scores each disputed finding's
groundedness and decides:

- any [P0]/[P1] present            -> BLOCK      (hard bar; never suppressed)
- all dissent thin (below bar)     -> SETTLE     (the treadmill escape)
- grounded advisory dissent defaults to SETTLE as follow-up, unless callers
  explicitly opt into promoting grounded advisory findings to BLOCK
- grounded dissent AND grounded support -> ESCALATE (material two-sided crux)
- no supportive signal / no dissent -> NOT_APPLICABLE (not a stall)

Default is flag-OFF (byte-identical to today).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import pytest

from aragora.swarm.review_adjudicator import (
    AdjudicationVerdict,
    AdvisorySeverityPolicy,
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


@dataclass
class _ConflictingItem:
    """EvidenceItem-shaped stand-in with a contradictory supportive flag."""

    family: str
    body: str
    verdict: str
    supportive: bool


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

    def test_default_scorer_falls_back_to_in_tree_analyzer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "aragora_debate", None)
        monkeypatch.setitem(sys.modules, "aragora_debate.evidence", None)

        r = adjudicate([_pass("claude"), _Item("openai", _GROUNDED_P2, "changes_requested")])
        dissent = next(a for a in r.assessments if a.family == "openai")

        assert r.verdict is AdjudicationVerdict.SETTLE
        assert dissent.groundedness >= 0.5


class TestAdjudicate:
    def test_all_thin_dissent_settles(self) -> None:
        # #8748 acceptance: advisory-only, thin dissent, a supportive signal -> SETTLE.
        items = [_pass("claude"), _Item("grok", _THIN_P3, "changes_requested")]
        r = adjudicate(items)
        assert r.verdict is AdjudicationVerdict.SETTLE
        assert r.settled_findings  # the thin finding is filed for follow-up
        assert not r.blocking_findings

    def test_evidence_backed_advisory_can_be_promoted_to_block_explicitly(self) -> None:
        # #8748 acceptance remains available as an explicit policy: a grounded
        # [P2] (concrete repro) is NOT suppressed.
        items = [_pass("claude"), _Item("openai", _GROUNDED_P2, "changes_requested")]
        r = adjudicate(
            items,
            advisory_severity_policy=AdvisorySeverityPolicy.PROMOTE_GROUNDED_TO_BLOCK,
        )
        assert r.verdict is AdjudicationVerdict.BLOCK
        assert r.blocking_findings

    def test_grounded_advisory_dissent_defaults_to_advisory_followup(self) -> None:
        # #8752: the default must not re-promote [P2]/[P3]-only dissent into a
        # hard block after severity-gated dissent made those findings advisory.
        items = [_pass("claude"), _Item("openai", _GROUNDED_P2, "changes_requested")]
        r = adjudicate(items, scorer=lambda body: 0.9 if "CHANGES-REQUESTED" in body else 0.1)
        assert r.verdict is AdjudicationVerdict.SETTLE
        assert r.settled_findings == [_GROUNDED_P2]
        assert not r.blocking_findings
        assert r.advisory_severity_policy is AdvisorySeverityPolicy.CAP_AT_ADVISORY
        assert (
            r.to_receipt_dict()["advisory_severity_policy"]
            == AdvisorySeverityPolicy.CAP_AT_ADVISORY.value
        )

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


class TestReviewFindingsFixes:
    """#8749 frontier-review fixes: single-score consistency + hard-bar robustness."""

    def test_verdict_consistent_with_assessments_under_nondeterministic_scorer(self) -> None:
        # claude [P2]: score each item ONCE. A scorer that would return different
        # values on repeated calls must not make the verdict contradict the
        # assessments embedded in the same receipt.
        calls: dict[str, int] = {}

        def flaky(body: str) -> float:
            calls[body] = calls.get(body, 0) + 1
            # high on the 1st call, low on any subsequent call for the same body
            return 0.9 if calls[body] == 1 else 0.0

        items = [_pass("claude"), _Item("openai", _GROUNDED_P2, "changes_requested")]
        r = adjudicate(items, scorer=flaky)
        # each body scored exactly once (no re-scoring in the verdict paths)
        assert all(v == 1 for v in calls.values())
        # both bodies score grounded on their single call → the verdict must be
        # consistent with those assessments (ESCALATE). The OLD re-scoring code
        # would have re-scored to 0.0 and returned SETTLE — contradicting a
        # receipt that records grounded=True.
        openai_a = next(a for a in r.assessments if a.family == "openai")
        assert openai_a.grounded is True
        assert r.verdict is AdjudicationVerdict.ESCALATE
        assert r.verdict is not AdjudicationVerdict.SETTLE

    def test_hard_bar_survives_raising_scorer(self) -> None:
        # openai [P2]: a [P0]/[P1] must hard-block even if the scorer raises.
        def boom(_body: str) -> float:
            raise RuntimeError("scorer exploded")

        items = [_pass("claude"), _Item("openai", _BLOCKING_P1, "changes_requested")]
        r = adjudicate(items, scorer=boom)  # must not raise
        assert r.verdict is AdjudicationVerdict.BLOCK

    def test_scorer_failure_fails_closed_to_escalate(self) -> None:
        # openai [P2] round 2: on an ADVISORY stall, a scorer failure must NOT
        # fail open to SETTLE (which would suppress a possibly-real finding) — it
        # must fail closed to ESCALATE for human settlement.
        def boom(_body: str) -> float:
            raise RuntimeError("analyzer down")

        items = [_pass("claude"), _Item("openai", _THIN_P3, "changes_requested")]
        r = adjudicate(items, scorer=boom)
        assert r.verdict is AdjudicationVerdict.ESCALATE
        assert r.verdict is not AdjudicationVerdict.SETTLE

    def test_hard_bar_never_imports_default_analyzer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # claude [P3] round 2: a definite [P0]/[P1] block must not construct (or
        # import) the default analyzer, so a missing aragora_debate never crashes
        # a hard block.
        monkeypatch.setitem(sys.modules, "aragora_debate", None)
        monkeypatch.setitem(sys.modules, "aragora_debate.evidence", None)
        items = [_pass("claude"), _Item("openai", _BLOCKING_P1, "changes_requested")]
        r = adjudicate(items)  # scorer=None default; must not raise
        assert r.verdict is AdjudicationVerdict.BLOCK

    def test_not_applicable_never_invokes_scorer(self) -> None:
        # openai [P2] r3: no-dissent / no-support cases must not build the scorer
        # or score any finding (no import, no side effects on the not-a-stall path).
        def boom(_body: str) -> float:
            raise RuntimeError("scorer must not be called here")

        no_dissent = [_pass("claude"), _pass("openai")]
        assert adjudicate(no_dissent, scorer=boom).verdict is AdjudicationVerdict.NOT_APPLICABLE
        no_support = [_Item("openai", _THIN_P3, "changes_requested")]
        assert adjudicate(no_support, scorer=boom).verdict is AdjudicationVerdict.NOT_APPLICABLE


class TestReviewFindingsAdvisoryRefinements:
    """#8752 advisory refinements deferred from #8749 review."""

    def test_custom_scorer_values_are_clamped_before_verdict(self) -> None:
        def out_of_range(body: str) -> float:
            return 2.0 if "CHANGES-REQUESTED" in body else -3.0

        items = [_pass("claude"), _Item("openai", _GROUNDED_P2, "changes_requested")]
        r = adjudicate(
            items,
            scorer=out_of_range,
            advisory_severity_policy=AdvisorySeverityPolicy.PROMOTE_GROUNDED_TO_BLOCK,
        )

        support = next(a for a in r.assessments if a.family == "claude")
        dissent = next(a for a in r.assessments if a.family == "openai")
        assert support.groundedness == 0.0
        assert dissent.groundedness == 1.0
        assert r.verdict is AdjudicationVerdict.BLOCK

    def test_non_finite_custom_scorer_output_fails_closed(self) -> None:
        items = [_pass("claude"), _Item("openai", _THIN_P3, "changes_requested")]
        r = adjudicate(items, scorer=lambda _body: float("nan"))
        assert r.verdict is AdjudicationVerdict.ESCALATE
        assert "scoring failed" in r.reason

    def test_groundedness_bar_is_clamped_for_receipts(self) -> None:
        items = [_pass("claude"), _pass("openai")]
        assert adjudicate(items, groundedness_bar=2.5).groundedness_bar == 1.0
        assert adjudicate(items, groundedness_bar=-0.5).groundedness_bar == 0.0

    def test_changes_requested_item_is_not_also_supportive(self) -> None:
        item = _ConflictingItem(
            family="openai",
            body=_GROUNDED_P2,
            verdict="changes_requested",
            supportive=True,
        )
        r = adjudicate([item], scorer=lambda _body: 0.9)
        assert r.verdict is AdjudicationVerdict.NOT_APPLICABLE
        assert "no supportive signal" in r.reason
