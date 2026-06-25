"""Tests for the Phase-A mission spine: state + the survivable tick loop.

The load-bearing test is ``test_crash_mid_feature_resumes_no_loss`` — it proves
the whole reason the engine exists: kill it mid-feature, relaunch, finish with no
lost and no double-done work.
"""

from __future__ import annotations

import pytest

from aragora.missions import Feature, Handoff, MissionOrchestrator, MissionState, Status


def _mission(n: int = 3, milestone: str = "m1") -> MissionState:
    return MissionState(
        mission_id="test",
        goal="prove the spine",
        milestones=[milestone],
        features=[
            Feature(id=f"f{i}", description=f"feat {i}", milestone=milestone)
            for i in range(1, n + 1)
        ],
    )


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    _mission().save(p)
    loaded = MissionState.load(p)
    assert loaded.mission_id == "test"
    assert [f.id for f in loaded.features] == ["f1", "f2", "f3"]
    assert all(f.status == Status.PENDING for f in loaded.features)


def test_next_pending_is_array_order():
    m = _mission()
    m.mark_completed("f1")
    assert m.next_pending().id == "f2"


def test_precondition_gating():
    m = MissionState(
        mission_id="t",
        goal="g",
        milestones=["m1"],
        features=[
            Feature(id="a", description="", milestone="m1", preconditions=["feature:b"]),
            Feature(id="b", description="", milestone="m1"),
        ],
    )
    # 'a' is gated on 'b'; b must come first even though a is earlier in the list.
    assert m.next_pending().id == "b"
    m.mark_completed("b")
    assert m.next_pending().id == "a"


def test_insert_followup_extends_queue():
    m = _mission(2)
    m.insert_feature(Feature(id="f1.5", description="follow", milestone="m1"), before="f2")
    assert [f.id for f in m.features] == ["f1", "f1.5", "f2"]
    with pytest.raises(ValueError, match="duplicate"):
        m.insert_feature(Feature(id="f1", description="dup", milestone="m1"))


def test_milestone_complete():
    m = _mission(2)
    assert not m.milestone_complete("m1")
    m.mark_completed("f1")
    m.mark_completed("f2")
    assert m.milestone_complete("m1")


def test_reclaim_orphaned_in_progress():
    m = _mission(2)
    m.mark_in_progress("f1", session_id="dead-worker")
    assert m.reclaim_in_progress() == ["f1"]
    assert m.get("f1").status == Status.PENDING
    # session id is retained for a future resume policy
    assert "dead-worker" in m.get("f1").worker_session_ids


def test_invalid_status_rejected():
    with pytest.raises(ValueError, match="invalid status"):
        Feature(id="x", description="", milestone="m1", status="bogus")


def test_run_drains_queue(tmp_path):
    p = tmp_path / "state.json"
    _mission(3).save(p)
    orch = MissionOrchestrator(p)
    done, total = orch.run(lambda feat: Handoff(success=True, session_id="w"))
    assert (done, total) == (3, 3)
    assert all(f.status == Status.COMPLETED for f in MissionState.load(p).features)


def test_handoff_followups_and_discoveries(tmp_path):
    p = tmp_path / "state.json"
    _mission(1).save(p)

    def dispatch(feat: Feature) -> Handoff:
        return Handoff(
            success=True,
            discovered=["found a stale assertion"],
            follow_ups=[Feature(id="f1-followup", description="fix it", milestone="m1")],
        )

    MissionOrchestrator(p).run(dispatch)
    final = MissionState.load(p)
    assert {f.id for f in final.features} == {"f1", "f1-followup"}
    assert "found a stale assertion" in final.get("f1").notes


def test_crash_mid_feature_resumes_no_loss(tmp_path):
    """THE exit test: kill mid-feature, relaunch, finish with no loss/dup."""
    p = tmp_path / "state.json"
    _mission(4).save(p)

    completed_calls: list[str] = []

    class _Boom(RuntimeError):
        pass

    def crashing_dispatch(feat: Feature) -> Handoff:
        if feat.id == "f3":
            raise _Boom("worker died mid-feature")  # simulate kill -9 during f3
        completed_calls.append(feat.id)
        return Handoff(success=True, session_id="w1")

    # First run: completes f1, f2, then dies inside f3.
    orch = MissionOrchestrator(p)
    with pytest.raises(_Boom):
        orch.run(crashing_dispatch)

    mid = MissionState.load(p)
    assert mid.get("f1").status == Status.COMPLETED
    assert mid.get("f2").status == Status.COMPLETED
    # f3 was checkpointed IN_PROGRESS before the crash — not lost, not completed.
    assert mid.get("f3").status == Status.IN_PROGRESS
    assert mid.get("f4").status == Status.PENDING

    # Second run on the SAME disk state with a healthy worker resumes cleanly.
    def healthy_dispatch(feat: Feature) -> Handoff:
        completed_calls.append(feat.id)
        return Handoff(success=True, session_id="w2")

    done, total = MissionOrchestrator(p).run(healthy_dispatch)
    assert (done, total) == (4, 4)

    # f1/f2 done exactly once; f3 retried once after the crash; f4 once. No dupes.
    assert completed_calls == ["f1", "f2", "f3", "f4"]


def test_orchestrator_caps_retries_instead_of_spinning(tmp_path):
    """A success=False handoff with no reason must NOT re-pick the same feature
    forever (the merge-gate-caught [P1])."""
    p = tmp_path / "state.json"
    _mission(2).save(p)
    calls: list[str] = []

    def always_fail_no_reason(feat):
        calls.append(feat.id)
        return Handoff(success=False)  # default blocked_reason=None — the dangerous case

    MissionOrchestrator(p, max_retries=3).run(always_fail_no_reason)
    final = MissionState.load(p)
    assert final.get("f1").status == Status.BLOCKED  # parked after the cap, not spinning
    assert final.get("f1").retry_count == 3
    assert calls == ["f1", "f1", "f1", "f2", "f2", "f2"]  # bounded: 3 each, not 10_000


def test_orchestrator_operator_block_is_terminal_immediately(tmp_path):
    p = tmp_path / "state.json"
    _mission(1).save(p)
    calls: list[str] = []

    def op_block(feat):
        calls.append(feat.id)
        return Handoff(success=False, blocked_reason="tier-3 operator settlement required")

    MissionOrchestrator(p, max_retries=5).run(op_block)
    assert MissionState.load(p).get("f1").status == Status.BLOCKED
    assert calls == ["f1"]  # terminal fork — not retried 5x pointlessly
