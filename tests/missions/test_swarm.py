"""Tests for the swarm worker loop — pheromone wired to the gate.

The money test is ``test_worker_parks_persistent_blocker_and_continues``: a unit
that always blocks gets parked in the shared ledger after N attempts and the
swarm finishes everything else — the treadmill escape that a stateless prompt
loop cannot achieve.

``dispatch`` here is a plain closure; BossLoopDispatch (tested in test_dispatch)
plugs in identically — it is just a ``Feature -> Handoff`` callable.
"""

from __future__ import annotations

import time

from aragora.missions.ledger import Ledger
from aragora.missions.orchestrator import Handoff, MissionOrchestrator
from aragora.missions.state import Feature, MissionState, Status
from aragora.missions.swarm import _LeaseHeartbeat, reconcile_from_ledger, run_worker


def _mission(tmp_path, n=4):
    p = tmp_path / "state.json"
    MissionState(
        mission_id="t",
        goal="g",
        milestones=["m1"],
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


def test_reconcile_folds_ledger_done_back_into_state(tmp_path):
    """Fixes the two-sources-of-truth finding: after a swarm run the ledger holds
    'done'; reconcile makes MissionState.progress() consistent."""
    sp, lp = _mission(tmp_path, 3)
    run_worker(sp, lp, "w1", lambda feat: Handoff(success=True))
    assert MissionState.load(sp).progress() == (0, 3)  # state untouched by the swarm...
    n = reconcile_from_ledger(sp, lp)
    assert n == 3
    assert MissionState.load(sp).progress() == (3, 3)  # ...until reconciled


def test_swarm_bounds_a_raising_dispatch_and_releases_lease(tmp_path):
    """The merge-gate-caught gap: a raising dispatch must be counted + bounded,
    and the lease must always be released (try/finally), not leaked for a TTL."""
    sp, lp = _mission(tmp_path, 2)
    calls: list[str] = []

    def raises_on_f1(feat):
        calls.append(feat.id)
        if feat.id == "f1":
            raise RuntimeError("boom")
        return Handoff(success=True)

    res = run_worker(sp, lp, "w1", raises_on_f1, park_threshold=2)
    led = Ledger(lp)
    assert res.parked == ["f1"]  # bounded after park_threshold raising attempts
    assert "f2" in res.done  # worker survived the poison unit and continued
    assert led.is_excluded("feature:f1") is True
    assert led.active_claims() == {}  # every lease released, even on the raise


def test_swarm_terminal_handoff_parks_immediately(tmp_path):
    sp, lp = _mission(tmp_path, 2)

    def dispatch(feat):
        if feat.id == "f1":
            return Handoff(success=False, terminal=True, blocked_reason="operator-gated")
        return Handoff(success=True)

    res = run_worker(sp, lp, "w1", dispatch, park_threshold=5)
    assert res.parked == ["f1"]
    assert res.blocked.count("f1") == 1  # terminal: parked on the first block, not retried
    assert "f2" in res.done


def test_swarm_discovered_work_survives_as_advisory_notes(tmp_path):
    """Propose/accept boundary: swarm RECORDS discovered work (notes + proposed
    follow-ups) so it is never silently dropped, but never INSERTS a Feature from
    ledger data (no gate-bypass injection). reconcile folds the notes in; the
    backlog gains no executable feature from the swarm."""
    sp, lp = _mission(tmp_path, 1)

    def dispatch(feat):
        return Handoff(
            success=True,
            follow_ups=[Feature(id="f1-follow", description="found it", milestone="m1")],
            discovered=["a stale assertion on f1"],
        )

    run_worker(sp, lp, "w1", dispatch)
    reconcile_from_ledger(sp, lp)
    final = MissionState.load(sp)
    # No feature was created from ledger JSON — the injection surface is gone.
    assert {f.id for f in final.features} == {"f1"}
    notes = final.get("f1").notes
    assert "discovered: a stale assertion on f1" in notes  # discovered note folded
    assert "discovered: follow-up proposed: f1-follow" in notes  # proposal kept as advisory


def test_swarm_records_discoveries_on_failed_handoff(tmp_path):
    """claude [P3]: discovered notes must be recorded even when the handoff fails —
    the orchestrator path records them regardless of success."""
    sp, lp = _mission(tmp_path, 1)

    def dispatch(feat):
        return Handoff(success=False, blocked_reason="blocked", discovered=["seen on failure"])

    run_worker(sp, lp, "w1", dispatch, park_threshold=1)  # parks after 1 block
    assert Ledger(lp).discoveries() == {"f1": ["seen on failure"]}


def test_orchestrator_auto_reconciles_ledger_before_driving(tmp_path):
    """grok [P2]: switching swarm→orchestrator must not re-dispatch ledger-done work.
    An orchestrator given a ledger_path folds the swarm's results in before ticking,
    so already-done units are COMPLETED and never re-dispatched."""
    sp, lp = _mission(tmp_path, 3)
    run_worker(
        sp, lp, "w1", lambda feat: Handoff(success=True)
    )  # swarm finishes all 3 in the ledger

    redispatched: list[str] = []

    def dispatch(feat):
        redispatched.append(feat.id)
        return Handoff(success=True)

    # Orchestrator wired to the same ledger: reconciles first, finds nothing to do.
    done, total = MissionOrchestrator(sp, ledger_path=lp).run(dispatch)
    assert (done, total) == (3, 3)
    assert redispatched == []  # ledger-done units were reconciled, not re-dispatched


def test_lease_heartbeat_keeps_a_long_dispatch_claim_alive(tmp_path):
    """claude [P2]: a dispatch outliving the lease TTL would be reclaimable. The
    heartbeat re-claims in the background so a live worker's lease never lapses."""
    _, lp = _mission(tmp_path, 1)
    led = Ledger(lp)
    led.claim("u1", "w1", ttl=0.3)  # short TTL — would expire in 0.3s without a beat
    with _LeaseHeartbeat(led, "u1", "w1", ttl=0.3):
        time.sleep(0.6)  # outlive two TTLs
        assert led.active_claims() == {"u1": "w1"}  # still held — heartbeat refreshed it
    # After the heartbeat stops, the lease lapses normally.
    time.sleep(0.4)
    assert led.active_claims() == {}


def test_worker_discards_success_after_losing_lease(tmp_path):
    """If a long-running dispatch loses its lease, its eventual success is stale and
    must not mark the unit done."""
    sp, lp = _mission(tmp_path, 1)

    def dispatch(feat):
        led = Ledger(lp)
        led.claim(feat.id, "other", now=time.time() + 3600.0)  # steal expired lease
        return Handoff(success=True, discovered=["stale success"])

    res = run_worker(sp, lp, "w1", dispatch, max_units=1)
    assert res.done == []
    led = Ledger(lp)
    assert led.is_done("f1") is False
    assert led.discoveries() == {}


def test_reconcile_does_not_revert_completed_to_blocked(tmp_path):
    """claude [P3]: a feature already COMPLETED in state must not be downgraded to
    BLOCKED by a stale active park constraint on the same id."""
    sp, lp = _mission(tmp_path, 2)
    state = MissionState.load(sp)
    state.mark_completed("f1")  # f1 finished (e.g. by the orchestrator path)
    state.save(sp)
    Ledger(lp).record_constraint("feature:f1", "stale park from a prior cross-mode run")

    reconcile_from_ledger(sp, lp)
    # f1 stays COMPLETED — only PENDING features are parked to BLOCKED.
    assert MissionState.load(sp).get("f1").status == Status.COMPLETED


def test_reconcile_folds_parks_to_blocked(tmp_path):
    sp, lp = _mission(tmp_path, 3)

    def dispatch(feat):
        if feat.id == "f2":
            return Handoff(success=False, terminal=True, blocked_reason="op")
        return Handoff(success=True)

    run_worker(sp, lp, "w1", dispatch)
    reconcile_from_ledger(sp, lp)
    final = MissionState.load(sp)
    assert final.get("f1").status == Status.COMPLETED
    assert final.get("f2").status == Status.BLOCKED  # parked -> BLOCKED, won't be re-dispatched
    assert final.get("f3").status == Status.COMPLETED
