"""Tests for ``scripts/self_develop.py --to-mission``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from aragora.nomic.task_decomposer import SubTask, TaskDecomposition

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


@pytest.fixture(autouse=True)
def _setup_path():
    sys.path.insert(0, str(SCRIPTS_DIR))
    yield
    sys.path.remove(str(SCRIPTS_DIR))


def _decomposition(goal: str) -> TaskDecomposition:
    return TaskDecomposition(
        original_task=goal,
        complexity_score=5,
        complexity_level="medium",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="bench",
                title="Publish benchmark result",
                description="Run the benchmark publication path.",
            ),
            SubTask(
                id="docs",
                title="Write public artifact",
                description="Document the benchmark result.",
            ),
        ],
    )


def test_self_develop_to_mission_writes_yaml_without_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import self_develop as mod

    mission_path = tmp_path / "mission.yaml"

    monkeypatch.setattr(mod, "run_heuristic_decomposition", lambda goal: _decomposition(goal))

    async def _fail_run_orchestration(**_kwargs):
        raise AssertionError("self_develop --to-mission must not execute orchestration")

    monkeypatch.setattr(mod, "run_orchestration", _fail_run_orchestration)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "self_develop.py",
            "--goal",
            "publish H1-01 rev-4 benchmark result",
            "--to-mission",
            str(mission_path),
        ],
    )

    rc = mod.main()

    out = capsys.readouterr().out
    mission = yaml.safe_load(mission_path.read_text())
    assert rc == 0
    assert "Conductor mission saved to:" in out
    assert mission["objective"] == "publish H1-01 rev-4 benchmark result"
    assert [lane["mode"] for lane in mission["lanes"]] == [
        "implementation",
        "implementation",
        "panel",
    ]
