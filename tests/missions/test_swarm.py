"""Tests for the swarm worker loop — pheromone wired to the gate.

The money test is ``test_worker_parks_persistent_blocker_and_continues``: a unit
that always blocks gets parked in the shared ledger after N attempts and the
swarm finishes everything else — the treadmill escape that a stateless prompt
loop cannot achieve.

``dispatch`` here is a plain closure; BossLoopDispatch (tested in test_dispatch)
plugs in identically — it is just a ``Feature -> Handoff`` callable.
"""

from __future__ import annotations

from aragora.missions.ledger import Ledger
from aragora.missions.orchestrator import Handoff
from aragora.missions.state import Feature, MissionState
from aragora.missions.swarm import run_worker


def _mission(tmp_path, n=4):
    p = tmp_path / "state.json"
    MissionState(
        mission_id="t", goal="g", milestones=["m1"],
        features=[Feature(id=f"f{i}", description="", milestone="m1") for i in range(1, n + 1)],
    ).save(p)
    return p, tmp_path / "ledger.json"


def test_worker_drains_clean_mission(tmp_path):
    sp, lp = _mission(tmp_path, 4)
    res = run_worker(sp, lp, "w1", lambda feat: Handoff(success=True))
    assert sorted(res.done) == ["f1", "f2", "f3", "f4"]
    assert Ledger(lp).done_units() == {"f1", "f2", "f3", "f4"}


def test_worker_parks_persistent_blocker_and_continues(tmp_path):
    sp, lp = _mission(tmp_path, 4)

    def dispatch(feat: Feature) -> Handoff:
        if feat.id == "f2":  # always fails (e.g. a Tier-3 surface needing the operator)
            return Handoff(success=False, blocked_reason="tier-3 operator settlement required")
        return Handoff(success=True)

    res = run_worker(sp, lp, "w1", dispatch, park_threshold=2)

    # f2 is parked after exactly 2 blocked attempts; everything else completes.
    assert sorted(res.done) == ["f1", "f3", "f4"]
    assert res.parked == ["f2"]
    assert res.blocked.count("f2") == 2  # tried twice, then parked — not an unbounded treadmill
    led = Ledger(lp)
    assert led.is_excluded("feature:f2") is True
    assert led.done_units() == {"f1", "f3", "f4"}


def test_two_workers_partition_and_collectively_park(tmp_path):
    sp, lp = _mission(tmp_path, 6)

    def dispatch(feat: Feature) -> Handoff:
        if feat.id == "f4":
            return Handoff(success=False, blocked_reason="persistent blocker")
        return Handoff(success=True)

    a = run_worker(sp, lp, "wA", dispatch, park_threshold=2)
    b = run_worker(sp, lp, "wB", dispatch, park_threshold=2)

    led = Ledger(lp)
    # The 5 good units are each done exactly once across the two workers.
    all_done = a.done + b.done
    assert sorted(all_done) == ["f1", "f2", "f3", "f5", "f6"]
    assert len(all_done) == len(set(all_done))  # no unit done twice
    # f4 was parked (collectively) and never falsely completed.
    assert led.is_excluded("feature:f4") is True
    assert "f4" not in led.done_units()


def test_already_done_units_are_skipped(tmp_path):
    sp, lp = _mission(tmp_path, 3)
    Ledger(lp).record_done("f1")  # pre-completed by some earlier run

    seen: list[str] = []

    def dispatch(feat: Feature) -> Handoff:
        seen.append(feat.id)
        return Handoff(success=True)

    run_worker(sp, lp, "w1", dispatch)
    assert "f1" not in seen  # never re-dispatched
    assert sorted(seen) == ["f2", "f3"]
