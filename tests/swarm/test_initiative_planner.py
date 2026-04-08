from __future__ import annotations

from aragora.nomic.task_decomposer import SubTask, TaskDecomposition
from aragora.pipeline.decision_plan.core import PlanStatus
from aragora.swarm.initiative_planner import InitiativePlanner


class _FakeDecomposer:
    def __init__(self) -> None:
        self.analyze_calls: list[dict[str, object]] = []
        self.model_calls: list[dict[str, object]] = []

    def analyze(self, task_description: str, **kwargs):
        self.analyze_calls.append({"task_description": task_description, **kwargs})
        return TaskDecomposition(
            original_task=task_description,
            complexity_score=6,
            complexity_level="medium",
            should_decompose=True,
            rationale="heuristic decomposition",
            subtasks=[
                SubTask(
                    id="slice-1",
                    title="Registry store",
                    description="Persist initiative JSON records.",
                    dependencies=[],
                    estimated_complexity="medium",
                    file_scope=["aragora/swarm/initiative_store.py"],
                    success_criteria={
                        "tests": ["python3 -m pytest tests/swarm/test_initiative_store.py -q"],
                        "behavior": ["initiative records survive reload"],
                    },
                ),
                SubTask(
                    id="slice-2",
                    title="CLI surfacing",
                    description="Expose initiative plan/show/list.",
                    dependencies=["slice-1"],
                    estimated_complexity="medium",
                    file_scope=["aragora/cli/commands/swarm.py"],
                    success_criteria={
                        "tests": [
                            "python3 -m pytest tests/cli/test_swarm_command.py -q -k initiative"
                        ],
                    },
                ),
            ],
        )

    def analyze_with_model_sync(self, task_description: str, **kwargs):
        self.model_calls.append({"task_description": task_description, **kwargs})
        return self.analyze(task_description, **kwargs)


def test_initiative_planner_builds_persistent_initiative_from_subtasks(tmp_path) -> None:
    planner = InitiativePlanner(repo_root=tmp_path, decomposer=_FakeDecomposer())

    initiative = planner.plan(
        goal="Create an initiative registry and planner",
        rationale="Roadmap work needs a durable planning object instead of recycled issues.",
        dependencies=["decision-plan-core"],
        validations=["python3 -m pytest tests/swarm/test_initiative_store.py -q"],
        feature_flag_name="initiative_registry",
        milestone_titles=["Planning ready"],
        checkpoint_titles=["Registry validated", "CLI validated"],
    )

    assert initiative.title == "Create an initiative registry and planner"
    assert initiative.status == PlanStatus.CREATED.value
    assert initiative.feature_flag_name == "initiative_registry"
    assert [item.slice_id for item in initiative.slices] == ["slice-1", "slice-2"]
    assert initiative.slices[1].dependencies == ["slice-1"]
    assert initiative.slices[0].acceptance_criteria == [
        "behavior: initiative records survive reload"
    ]
    assert initiative.checkpoints[0].title == "Registry validated"
    assert initiative.milestones[0].title == "Planning ready"
    assert initiative.metadata["planner_strategy"] == "heuristic"


def test_initiative_planner_uses_model_strategy_when_requested(tmp_path) -> None:
    decomposer = _FakeDecomposer()
    planner = InitiativePlanner(repo_root=tmp_path, decomposer=decomposer)

    initiative = planner.plan(
        goal="Use model planning",
        rationale="Planner should defer to the configured model strategy.",
        planner_strategy="model",
        planner_model="claude",
    )

    assert initiative.metadata["planner_strategy"] == "model"
    assert initiative.metadata["planner_model"] == "claude"
    assert len(decomposer.model_calls) == 1
    assert decomposer.model_calls[0]["planner_model"] == "claude"
