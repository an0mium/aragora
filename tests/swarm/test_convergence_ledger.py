"""Tests for the per-PR ConvergenceLedger — the round budget that survives head drift.

The load-bearing test is ``test_distinct_heads_count_as_rounds_not_rereviews``: the
churn metric is *distinct heads put through review*, so a repair (new head) counts
and a re-review of the same head does not — that is exactly what the per-head gate
budget cannot see.
"""

from __future__ import annotations

from aragora.swarm.convergence_ledger import (
    DEFAULT_ROUND_BUDGET,
    MAX_TRACKED_PRS,
    ConvergenceLedger,
    PRConvergence,
)


def _ledger(tmp_path):
    return ConvergenceLedger(tmp_path / "conv.json")


def test_record_round_returns_count_and_persists(tmp_path):
    led = _ledger(tmp_path)
    assert led.record_round(8628, "head-a", verdict="changes_requested", now="t0") == 1
    assert led.record_round(8628, "head-b", verdict="changes_requested", now="t1") == 2
    # A fresh instance reads the same persisted state.
    assert ConvergenceLedger(tmp_path / "conv.json").rounds(8628) == 2


def test_distinct_heads_count_as_rounds_not_rereviews(tmp_path):
    led = _ledger(tmp_path)
    led.record_round(8595, "h1", verdict="changes_requested", now="t0")
    led.record_round(8595, "h1", verdict="changes_requested", now="t1")  # same head re-review
    led.record_round(8595, "h2", verdict="changes_requested", now="t2")  # a repair → new head
    rec = led.get(8595)
    assert rec.rounds == 2  # two distinct heads, not three records
    assert rec.verdicts == ["changes_requested"] * 3  # full verdict history retained


def test_budget_disabled_when_zero(tmp_path):
    led = _ledger(tmp_path)
    led.record_round(1, "h1", now="t0")
    assert led.is_exhausted(1, budget=0) is False  # 0 = disabled


def test_budget_exhaustion_and_remaining(tmp_path):
    led = _ledger(tmp_path)
    for i in range(6):
        led.record_round(8628, f"h{i}", now=f"t{i}")
    assert led.rounds(8628) == 6
    assert led.budget_remaining(8628, budget=6) == 0
    assert led.is_exhausted(8628, budget=6) is True
    assert led.is_exhausted(8628, budget=8) is False


def test_unknown_pr_is_zero_not_exhausted(tmp_path):
    led = _ledger(tmp_path)
    assert led.rounds(404) == 0
    assert led.is_exhausted(404, budget=6) is False
    assert led.budget_remaining(404, budget=6) == 6


def test_record_adjudication_audit_trail(tmp_path):
    led = _ledger(tmp_path)
    led.record_round(8628, "h1", now="t0")
    led.record_adjudication(
        8628, verdict="CLOSE", rationale="superseded; churned 7 rounds", now="t9"
    )
    rec = ConvergenceLedger(tmp_path / "conv.json").get(8628)
    assert rec.adjudication == {
        "verdict": "CLOSE",
        "rationale": "superseded; churned 7 rounds",
        "at": "t9",
    }


def test_summarize_orders_by_rounds_desc(tmp_path):
    led = _ledger(tmp_path)
    led.record_round(100, "a", verdict="pass", now="t0")
    for i in range(4):
        led.record_round(200, f"h{i}", verdict="changes_requested", now=f"t{i}")
    rows = led.summarize(budget=6)
    assert [r["pr"] for r in rows] == [200, 100]  # most-churned first
    top = rows[0]
    assert top["rounds"] == 4 and top["budget_remaining"] == 2
    assert top["last_verdict"] == "changes_requested" and top["adjudicated"] is False


def test_default_budget_constant_is_sane(tmp_path):
    assert DEFAULT_ROUND_BUDGET >= 3  # enough for real polishing, bounded against churn


def test_prune_caps_tracked_prs(tmp_path):
    led = _ledger(tmp_path)
    # Record MAX_TRACKED_PRS + 5; oldest-by-last_round_at evicted.
    for i in range(MAX_TRACKED_PRS + 5):
        led.record_round(i, "h", now=f"{i:06d}")
    data = led._load()
    assert len(data) == MAX_TRACKED_PRS
    assert 0 not in data and 4 not in data  # five oldest evicted
    assert (MAX_TRACKED_PRS + 4) in data  # newest retained


def test_corrupt_ledger_degrades_to_empty(tmp_path):
    p = tmp_path / "conv.json"
    p.write_text("{ this is not json", encoding="utf-8")
    led = ConvergenceLedger(p)
    assert led.rounds(1) == 0  # unreadable → empty, not a crash
    # ...and a subsequent write recovers cleanly.
    led.record_round(1, "h1", now="t0")
    assert ConvergenceLedger(p).rounds(1) == 1


def test_record_dataclass_round_property():
    rec = PRConvergence(pr_number=1, heads=["a", "b", "c"])
    assert rec.rounds == 3
    assert rec.is_exhausted(3) is True
    assert rec.budget_remaining(5) == 2
