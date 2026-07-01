"""Tests for the intake→decomposition bridge (#8758).

A seeded mission (``aragora mission seed --goal ...``) starts as a single
"mission-intake" feature with no ``metadata.branch``. Before the bridge, the
live dispatch parked it terminally. These tests pin the bridge contract:

* intake-shaped features are decomposed via TaskDecomposer into 1+ child
  Features, each branch-backed (``metadata.branch``) and claimable by the
  existing dispatch/lease machinery;
* the intake feature completes (non-terminal) with provenance notes;
* empty decompositions mirror the goal into one child instead of parking;
* a raising decomposer parks the intake with a diagnostic — the tick loop
  never crashes;
* re-ticking after a crash never duplicates children (idempotency);
* non-intake features pass through to the inner dispatch untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from aragora.missions import (
    Feature,
    Handoff,
    Ledger,
    MissionOrchestrator,
    MissionState,
    Status,
    select_for,
)
from aragora.missions.dispatch import BossLoopDispatch
from aragora.missions.intake import (
    IntakeBridgeDispatch,
    intake_bridge_enabled,
    is_intake_feature,
)
from aragora.nomic.task_decomposer import SubTask

GOAL = "Add a rate limiter to the API server"


def _intake_feature(goal: str = GOAL) -> Feature:
    return Feature(
        id="mission-intake",
        description=goal,
        milestone="mission",
        metadata={"paths": ["aragora/server"], "tracks": [], "autonomy": "auto-drain"},
    )


def _seeded_state(tmp_path: Path, goal: str = GOAL) -> Path:
    state_path = tmp_path / "state.json"
    state = MissionState(
        mission_id="mission-test",
        goal=goal,
        milestones=["mission"],
        features=[_intake_feature(goal)],
    )
    state.save(state_path)
    return state_path


def _refusing_inner(feature: Feature) -> Handoff:
    raise AssertionError(f"inner dispatch must never see intake feature {feature.id}")


def _two_subtasks(goal: str, paths: list[str]) -> list[SubTask]:
    return [
        SubTask(
            id="subtask_1",
            title="Server Changes",
            description="Wire the rate limiter into the server",
            file_scope=["aragora/server"],
        ),
        SubTask(
            id="subtask_2",
            title="Tests",
            description="Cover the rate limiter with tests",
            dependencies=["subtask_1"],
        ),
    ]


# ---- intake detection ---------------------------------------------------------


def test_branch_backed_feature_is_not_intake():
    feat = Feature(id="a1", description="x", milestone="m", metadata={"branch": "mission/a1"})
    assert not is_intake_feature(feat)


def test_seeded_intake_feature_is_intake():
    assert is_intake_feature(_intake_feature())


def test_kind_marker_makes_any_branchless_feature_intake():
    feat = Feature(id="other-id", description="x", milestone="m", metadata={"kind": "intake"})
    assert is_intake_feature(feat)


# ---- successful decomposition -------------------------------------------------


def test_intake_decomposes_into_branch_backed_follow_ups():
    bridge = IntakeBridgeDispatch(_refusing_inner, decompose=_two_subtasks)

    handoff = bridge(_intake_feature())

    assert handoff.success
    assert not handoff.terminal
    assert handoff.accept_follow_ups
    assert len(handoff.follow_ups) == 2
    for child in handoff.follow_ups:
        assert child.status == Status.PENDING
        assert not is_intake_feature(child)
        branch = child.metadata["branch"]
        assert branch.startswith("mission/")
        assert child.metadata["intake_parent"] == "mission-intake"
        assert child.metadata["decomposer"]
        assert isinstance(child.metadata["decomposed_at"], float)
    first, second = handoff.follow_ups
    assert second.preconditions == [f"feature:{first.id}"]
    # Subtask file scope narrows the child's path allowlist.
    assert first.metadata["paths"] == ["aragora/server"]
    assert any("decomposed" in note for note in handoff.discovered)


def test_tick_converts_intake_into_children_instead_of_parking(tmp_path):
    state_path = _seeded_state(tmp_path)
    bridge = IntakeBridgeDispatch(_refusing_inner, decompose=_two_subtasks)

    assert MissionOrchestrator(state_path).tick(bridge) is True

    state = MissionState.load(state_path)
    intake = state.get("mission-intake")
    assert intake.status == Status.COMPLETED  # non-terminal: never BLOCKED
    assert "decomposed" in intake.notes
    children = [f for f in state.features if f.id != "mission-intake"]
    assert len(children) == 2
    assert all(f.status == Status.PENDING for f in children)
    assert all(str(f.metadata["branch"]).startswith("mission/") for f in children)


def test_children_are_claimable_by_existing_lease_machinery(tmp_path):
    state_path = _seeded_state(tmp_path)
    bridge = IntakeBridgeDispatch(_refusing_inner, decompose=_two_subtasks)
    MissionOrchestrator(state_path).tick(bridge)

    state = MissionState.load(state_path)
    ledger = Ledger(tmp_path / "ledger.json")
    claimed = select_for(state, ledger, "worker-1")

    children = [f.id for f in state.features if f.id != "mission-intake"]
    assert claimed == children[0]
    # Second worker skips the claimed unit; the dependent child is gated.
    assert select_for(state, ledger, "worker-2") is None


# ---- empty decomposition ------------------------------------------------------


def test_empty_decomposition_falls_back_to_single_mirrored_child():
    bridge = IntakeBridgeDispatch(_refusing_inner, decompose=lambda goal, paths: [])

    handoff = bridge(_intake_feature("Bump the httpx pin"))

    assert handoff.success
    assert len(handoff.follow_ups) == 1
    child = handoff.follow_ups[0]
    assert "Bump the httpx pin" in child.description
    assert child.metadata["branch"].startswith("mission/")
    assert child.metadata["intake_parent"] == "mission-intake"


# ---- failure paths ------------------------------------------------------------


def test_decomposer_exception_parks_with_diagnostic_and_never_crashes_tick(tmp_path):
    def boom(goal: str, paths: list[str]) -> list[SubTask]:
        raise RuntimeError("no decomposition today")

    state_path = _seeded_state(tmp_path)
    bridge = IntakeBridgeDispatch(_refusing_inner, decompose=boom)

    assert MissionOrchestrator(state_path).tick(bridge) is True  # no raise

    state = MissionState.load(state_path)
    intake = state.get("mission-intake")
    assert intake.status == Status.BLOCKED
    assert "intake decomposition failed" in intake.notes
    assert "no decomposition today" in intake.notes
    assert len(state.features) == 1  # no half-inserted children


def test_blank_goal_parks_terminal():
    bridge = IntakeBridgeDispatch(_refusing_inner, decompose=_two_subtasks)

    handoff = bridge(_intake_feature("   "))

    assert not handoff.success
    assert handoff.terminal
    assert "goal" in (handoff.blocked_reason or "")
    assert handoff.follow_ups == []


# ---- idempotency --------------------------------------------------------------


def test_crash_retick_does_not_duplicate_children(tmp_path):
    state_path = _seeded_state(tmp_path)
    bridge = IntakeBridgeDispatch(_refusing_inner, decompose=_two_subtasks)
    orch = MissionOrchestrator(state_path)
    orch.tick(bridge)

    # Simulate a crash after dispatch but before triage persisted completion:
    # the intake is re-picked and re-decomposed on the next tick.
    state = MissionState.load(state_path)
    state.get("mission-intake").status = Status.PENDING
    state.save(state_path)
    orch.tick(bridge)

    state = MissionState.load(state_path)
    ids = [f.id for f in state.features]
    assert len(ids) == len(set(ids)) == 3  # intake + 2 children, no duplicates
    assert state.get("mission-intake").status == Status.COMPLETED


# ---- pass-through -------------------------------------------------------------


def test_non_intake_features_pass_through_to_inner_dispatch():
    seen: list[str] = []

    def inner(feature: Feature) -> Handoff:
        seen.append(feature.id)
        return Handoff(success=True)

    bridge = IntakeBridgeDispatch(inner, decompose=_two_subtasks)
    feat = Feature(id="a1", description="x", milestone="m", metadata={"branch": "mission/a1"})

    assert bridge(feat).success
    assert seen == ["a1"]


# ---- default decomposer (heuristic, no API keys) ------------------------------


def test_default_decomposer_requires_no_api_keys(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "ARAGORA_USE_SECRETS_MANAGER"):
        monkeypatch.delenv(var, raising=False)
    bridge = IntakeBridgeDispatch(_refusing_inner)

    handoff = bridge(
        _intake_feature("Harden the api server and add tests for the storage handlers")
    )

    assert handoff.success
    assert len(handoff.follow_ups) >= 1
    assert all(c.metadata["branch"].startswith("mission/") for c in handoff.follow_ups)


# ---- feature flag + CLI wiring -------------------------------------------------


def test_intake_bridge_enabled_env_kill_switch(monkeypatch):
    monkeypatch.delenv("ARAGORA_DISABLE_MISSION_INTAKE_BRIDGE", raising=False)
    assert intake_bridge_enabled()
    monkeypatch.setenv("ARAGORA_DISABLE_MISSION_INTAKE_BRIDGE", "1")
    assert not intake_bridge_enabled()
    monkeypatch.setenv("ARAGORA_DISABLE_MISSION_INTAKE_BRIDGE", "0")
    assert intake_bridge_enabled()


@pytest.mark.parametrize("disabled", [False, True])
def test_auto_drain_dispatch_wires_intake_bridge_by_default(tmp_path, monkeypatch, disabled):
    from aragora.cli.commands.mission import _dispatch_for

    if disabled:
        monkeypatch.setenv("ARAGORA_DISABLE_MISSION_INTAKE_BRIDGE", "1")
    else:
        monkeypatch.delenv("ARAGORA_DISABLE_MISSION_INTAKE_BRIDGE", raising=False)
    args = argparse.Namespace(
        autonomy="auto-drain",
        repo_root=str(tmp_path),
        auto_settle_max_tier=2,
        operator_tier=3,
    )

    dispatch = _dispatch_for(args, state_path=None)

    if disabled:
        assert isinstance(dispatch, BossLoopDispatch)
    else:
        assert isinstance(dispatch, IntakeBridgeDispatch)
