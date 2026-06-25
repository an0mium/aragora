from __future__ import annotations

import sys
from pathlib import Path

import yaml

from aragora.nomic.mission_bridge import decomposition_to_mission, write_mission_yaml
from aragora.nomic.task_decomposer import SubTask, TaskDecomposition

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


def _load_goal_conductor_module():
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import goal_conductor

        return goal_conductor
    finally:
        sys.path.remove(str(SCRIPTS_DIR))


def _decomposition() -> TaskDecomposition:
    return TaskDecomposition(
        original_task="publish H1-01 rev-4 benchmark result",
        complexity_score=6,
        complexity_level="medium",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="bench",
                title="Run the benchmark harness",
                description="Run the existing benchmark publication scripts.",
                file_scope=["scripts/build_benchmark_truth_artifact.py"],
                success_criteria={"artifact": "generated"},
            ),
            SubTask(
                id="docs",
                title="Write the public result",
                description="Publish the benchmark result as a status artifact.",
                file_scope=["docs/status/B0_BENCHMARK_TRUTH_STATUS.md"],
            ),
            SubTask(
                id="extra",
                title="Extra lane should be capped",
                description="This should not become an implementation lane.",
            ),
        ],
    )


def test_decomposition_to_mission_is_goal_conductor_compatible() -> None:
    mod = _load_goal_conductor_module()

    mission = decomposition_to_mission(
        _decomposition(),
        objective="publish H1-01 rev-4 benchmark result",
    )

    parsed = mod.Mission.from_dict(mission)
    assert parsed.name == "publish-h1-01-rev-4-benchmark-result"
    assert parsed.limits.queue_cap == 6
    assert [lane.mode for lane in parsed.lanes] == ["implementation", "implementation", "panel"]
    assert [lane.agent for lane in parsed.lanes[:2]] == ["codex", "claude"]
    assert parsed.lanes[2].agents_spec == "heterogeneous"


def test_decomposition_to_mission_can_omit_panel_review() -> None:
    mission = decomposition_to_mission(
        _decomposition(),
        objective="publish H1-01 rev-4 benchmark result",
        include_panel_review=False,
    )

    assert len(mission["lanes"]) == 2
    assert all(lane["mode"] == "implementation" for lane in mission["lanes"])


def test_decomposition_to_mission_does_not_default_to_repo_wide_pytest() -> None:
    mission = decomposition_to_mission(
        _decomposition(),
        objective="publish H1-01 rev-4 benchmark result",
        include_panel_review=False,
    )

    assert mission["lanes"][0]["tests"] == []
    assert mission["lanes"][1]["tests"] == []


def test_decomposition_to_mission_uses_scoped_test_paths_as_validation() -> None:
    decomp = TaskDecomposition(
        original_task="repair a focused test",
        complexity_score=2,
        complexity_level="small",
        should_decompose=False,
        subtasks=[
            SubTask(
                id="focused",
                title="Repair focused test",
                description="Patch only the focused test.",
                file_scope=["tests/nomic/test_mission_bridge.py"],
            )
        ],
    )

    mission = decomposition_to_mission(
        decomp,
        objective="repair a focused test",
        include_panel_review=False,
    )

    assert mission["lanes"][0]["tests"] == [
        "python3 -m pytest -q tests/nomic/test_mission_bridge.py"
    ]


def test_decomposition_to_mission_does_not_emit_unscoped_codex_implementation_lane() -> None:
    mod = _load_goal_conductor_module()
    decomp = TaskDecomposition(
        original_task="investigate unclear repo behavior",
        complexity_score=2,
        complexity_level="small",
        should_decompose=False,
        subtasks=[
            SubTask(
                id="unclear",
                title="Investigate unclear behavior",
                description="No file scope is known yet.",
            )
        ],
    )

    mission = decomposition_to_mission(
        decomp,
        objective="investigate unclear repo behavior",
        agents=["codex"],
        include_panel_review=False,
    )

    parsed = mod.Mission.from_dict(mission)
    assert parsed.lanes[0].mode == "panel"
    assert parsed.lanes[0].agent == "panel"
    assert parsed.lanes[0].lease_blocker() is None


def test_write_mission_yaml_round_trips(tmp_path: Path) -> None:
    mod = _load_goal_conductor_module()
    mission = decomposition_to_mission(
        _decomposition(),
        objective="publish H1-01 rev-4 benchmark result",
    )
    path = write_mission_yaml(mission, tmp_path / "mission.yaml")

    loaded = yaml.safe_load(path.read_text())
    parsed = mod.Mission.from_dict(loaded)
    assert parsed.objective == "publish H1-01 rev-4 benchmark result"
