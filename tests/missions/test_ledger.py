"""Tests for the stigmergic Ledger — atomic claims, evaporation, and the
constraint-driven anti-treadmill that the worker swarm relies on.

Time is injected via ``now=`` so the lease/constraint TTL logic is deterministic.
"""

from __future__ import annotations

from aragora.missions.ledger import Ledger, select_for
from aragora.missions.state import Feature, MissionState


def _ledger(tmp_path):
    return Ledger(tmp_path / "ledger.json")


def test_atomic_claim_excludes_other_worker(tmp_path):
    led = _ledger(tmp_path)
    assert led.claim("u1", "w1", now=100.0) is True
    assert led.claim("u1", "w2", now=101.0) is False  # w2 cannot steal a live claim
    assert led.active_claims(now=101.0) == {"u1": "w1"}


def test_reclaim_own_lease_is_idempotent(tmp_path):
    led = _ledger(tmp_path)
    assert led.claim("u1", "w1", now=100.0) is True
    assert led.claim("u1", "w1", now=200.0) is True  # same worker re-claims fine


def test_expired_lease_evaporates_and_is_reclaimable(tmp_path):
    led = _ledger(tmp_path)
    led.claim("u1", "w1", ttl=30.0, now=100.0)
    # 40s later the lease is dead -> a different worker can claim it
    assert led.claim("u1", "w2", now=140.0) is True
    assert led.active_claims(now=140.0) == {"u1": "w2"}


def test_release(tmp_path):
    led = _ledger(tmp_path)
    led.claim("u1", "w1", now=100.0)
    led.release("u1", "w1")
    assert led.active_claims(now=100.0) == {}
    # a non-owner release is a no-op
    led.claim("u1", "w1", now=100.0)
    led.release("u1", "w2")
    assert led.active_claims(now=100.0) == {"u1": "w1"}


def test_constraint_excludes_then_invalidates(tmp_path):
    led = _ledger(tmp_path)
    led.record_constraint("feature:x", "tier-3 needs operator", now=100.0)
    assert led.is_excluded("feature:x", now=100.0) is True
    led.invalidate_constraint("feature:x")  # live state changed
    assert led.is_excluded("feature:x", now=100.0) is False


def test_constraint_ttl_evaporates(tmp_path):
    led = _ledger(tmp_path)
    led.record_constraint("feature:x", "transient", ttl=60.0, now=100.0)
    assert led.is_excluded("feature:x", now=130.0) is True
    assert led.is_excluded("feature:x", now=200.0) is False


def test_attempts_and_prune(tmp_path):
    led = _ledger(tmp_path)
    assert led.bump_attempt("feature:x") == 1
    assert led.bump_attempt("feature:x") == 2
    assert led.attempts("feature:x") == 2
    led.claim("u1", "w1", ttl=10.0, now=100.0)
    led.record_constraint("c1", "old", ttl=5.0, now=100.0)
    dead_leases, dead_constraints = led.prune(now=200.0)
    assert (dead_leases, dead_constraints) == (1, 1)


def _mission(n=4):
    return MissionState(
        mission_id="t", goal="g", milestones=["m1"],
        features=[Feature(id=f"f{i}", description="", milestone="m1") for i in range(1, n + 1)],
    )


def test_two_workers_self_partition_no_overlap(tmp_path):
    """Non-overlapping fronts emerge with no dispatcher: each worker claims a
    distinct unit off the same queue."""
    state = _mission(4)
    led = _ledger(tmp_path)
    a = select_for(state, led, "wA", now=100.0)
    b = select_for(state, led, "wB", now=100.0)
    c = select_for(state, led, "wC", now=100.0)
    assert a == "f1" and b == "f2" and c == "f3"  # three workers, three distinct units
    assert len({a, b, c}) == 3  # zero collisions


def test_select_skips_parked_feature(tmp_path):
    """The anti-treadmill rule your Codex prompt couldn't enforce: a parked unit
    is skipped because the exclusion lives in the shared environment."""
    state = _mission(2)
    led = _ledger(tmp_path)
    led.record_constraint("feature:f1", "parked: repeated blocker", now=100.0)
    assert select_for(state, led, "wA", now=100.0) == "f2"  # f1 is skipped, not re-attempted


def test_select_respects_preconditions(tmp_path):
    state = MissionState(
        mission_id="t", goal="g", milestones=["m1"],
        features=[
            Feature(id="a", description="", milestone="m1", preconditions=["feature:b"]),
            Feature(id="b", description="", milestone="m1"),
        ],
    )
    led = _ledger(tmp_path)
    assert select_for(state, led, "wA", now=100.0) == "b"  # gated 'a' is not claimable yet
