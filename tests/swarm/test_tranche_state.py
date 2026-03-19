from pathlib import Path

from aragora.swarm.tranche_state import LaneRunState, TrancheRunState


def test_tranche_run_state_round_trip() -> None:
    state = TrancheRunState(
        manifest_id="test-manifest",
        status="planned",
        autonomy_mode="adaptive",
    )
    state.lane_states["lane_a"] = LaneRunState(lane_id="lane_a", status="pending")
    restored = TrancheRunState.from_dict(state.to_dict())
    assert restored.manifest_id == "test-manifest"
    assert restored.status == "planned"
    assert restored.lane_states["lane_a"].status == "pending"


def test_lane_run_state_defaults() -> None:
    lane = LaneRunState(lane_id="x", status="pending")
    assert lane.run_id is None
    assert lane.receipt_id is None
    assert lane.lease_id is None
    assert lane.retry_count == 0


def test_tranche_run_state_persistence(tmp_path: Path) -> None:
    state = TrancheRunState(
        manifest_id="persist-test",
        status="running",
        autonomy_mode="fire_and_forget",
    )
    path = tmp_path / "run_state.yaml"
    state.save(path)
    loaded = TrancheRunState.load(path)
    assert loaded.manifest_id == "persist-test"
    assert loaded.status == "running"
