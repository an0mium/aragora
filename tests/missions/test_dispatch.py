"""Tests for BossLoopDispatch — the merge-gate driving contract.

Validates every hard-won rule with a fake gate (no live main touched):
idempotency, the foreign-commit guard, Tier-3 escalation, dissent, head-bound
merge, and head-moved-under-us.
"""

from __future__ import annotations

from aragora.missions.dispatch import BossLoopDispatch, GateVerdict
from aragora.missions.state import Feature


class FakeGate:
    """Configurable in-memory stand-in for the live review_queue gate."""

    def __init__(
        self,
        *,
        merged: bool = False,
        head: str = "abc123",
        foreign: list[str] | None = None,
        verdict: GateVerdict | None = None,
        merge_ok: bool = True,
    ) -> None:
        self.merged = merged
        self.head = head
        self.foreign = foreign or []
        self.verdict = verdict or GateVerdict(satisfied=True)
        self.merge_ok = merge_ok
        self.merge_calls: list[tuple[str, str]] = []
        self.evidence_calls = 0

    def branch_for(self, feature: Feature) -> str:
        return f"mission/{feature.id}"

    def already_merged(self, branch: str) -> bool:
        return self.merged

    def head_of(self, branch: str) -> str:
        return self.head

    def foreign_commits(self, branch, base, allowed_prefixes):
        return list(self.foreign)

    def collect_evidence(self, branch, head):
        self.evidence_calls += 1
        return self.verdict

    def merge_head_bound(self, branch, head):
        self.merge_calls.append((branch, head))
        return self.merge_ok


def _feat() -> Feature:
    return Feature(id="a5", description="rehome x", milestone="phase-a-spine")


def test_clean_quorum_merges_head_bound():
    gate = FakeGate(head="deadbeef", verdict=GateVerdict(satisfied=True, tier=0))
    handoff = BossLoopDispatch(gate)(_feat())
    assert handoff.success
    assert gate.merge_calls == [("mission/a5", "deadbeef")]


def test_already_merged_is_idempotent_success():
    gate = FakeGate(merged=True)
    handoff = BossLoopDispatch(gate)(_feat())
    assert handoff.success
    assert gate.evidence_calls == 0  # never re-collected
    assert gate.merge_calls == []  # never double-merged


def test_foreign_commit_guard_blocks_before_evidence():
    gate = FakeGate(foreign=["aa008da5 fix(memory): cache", "6f4a7222 test"])
    handoff = BossLoopDispatch(gate)(_feat())
    assert not handoff.success
    assert "re-derive" in handoff.blocked_reason
    assert gate.evidence_calls == 0  # the #8616 lesson: never collect on contamination
    assert gate.merge_calls == []


def test_tier3_escalates_even_on_clean_quorum():
    gate = FakeGate(verdict=GateVerdict(satisfied=True, tier=3))
    handoff = BossLoopDispatch(gate)(_feat())
    assert not handoff.success
    assert "operator settlement" in handoff.blocked_reason
    assert gate.merge_calls == []  # never auto-settles a Tier-3 surface


def test_dissent_blocks_without_merge():
    gate = FakeGate(verdict=GateVerdict(satisfied=False, tier=1, dissent=["[P1] real bug"]))
    handoff = BossLoopDispatch(gate)(_feat())
    assert not handoff.success
    assert "[P1] real bug" in handoff.blocked_reason
    assert gate.merge_calls == []


def test_head_moved_under_us_does_not_falsely_succeed():
    gate = FakeGate(verdict=GateVerdict(satisfied=True), merge_ok=False)
    handoff = BossLoopDispatch(gate)(_feat())
    assert not handoff.success
    assert "did not land" in handoff.blocked_reason
