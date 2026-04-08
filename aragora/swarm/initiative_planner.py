from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from aragora.nomic.task_decomposer import SubTask, TaskDecomposer
from aragora.pipeline.decision_plan.core import PlanStatus
from aragora.swarm.initiative_models import (
    InitiativeCheckpoint,
    InitiativeMilestone,
    InitiativeRecord,
    InitiativeSlice,
)
from aragora.swarm.spec import SwarmSpec


_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def _text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in [_text(value) for value in values] if item))


def _slugify(value: str) -> str:
    normalized = _SLUG_PATTERN.sub("-", value.lower()).strip("-")
    return normalized or f"initiative-{uuid4().hex[:8]}"


def _initiative_id(title: str) -> str:
    return _slugify(title)


def _derive_title(goal: str) -> str:
    cleaned = " ".join(str(goal or "").split()).strip(" -:_")
    if not cleaned:
        return "Untitled initiative"
    return cleaned[:120]


def _feature_flag_name_for_title(title: str) -> str:
    return f"initiative_{_slugify(title).replace('-', '_')}"


def _success_criteria_lines(success_criteria: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key, value in success_criteria.items():
        lowered = str(key).strip().lower()
        if lowered == "tests":
            continue
        if isinstance(value, list):
            lines.extend(f"{key}: {str(item).strip()}" for item in value if str(item).strip())
            continue
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                nested_text = _text(nested_value)
                if nested_text:
                    lines.append(f"{key}.{nested_key}: {nested_text}")
            continue
        text = _text(value)
        if text:
            lines.append(f"{key}: {text}")
    return _dedupe(lines)


def _validation_commands(success_criteria: dict[str, Any], fallback: list[str]) -> list[str]:
    tests_value = success_criteria.get("tests")
    commands: list[str] = []
    if isinstance(tests_value, str) and tests_value.strip():
        commands.append(tests_value.strip())
    elif isinstance(tests_value, list):
        commands.extend(str(item).strip() for item in tests_value if str(item).strip())
    return _dedupe(commands or fallback)


def _slice_from_subtask(
    subtask: SubTask,
    *,
    fallback_validations: list[str],
) -> InitiativeSlice:
    acceptance_criteria = _success_criteria_lines(subtask.success_criteria)
    validations = _validation_commands(subtask.success_criteria, fallback_validations)
    if not acceptance_criteria and validations:
        acceptance_criteria = [f"Run and satisfy: {command}" for command in validations]
    if not acceptance_criteria:
        acceptance_criteria = [f"Complete the bounded slice '{subtask.title}'."]
    return InitiativeSlice(
        slice_id=_text(subtask.id) or f"slice-{uuid4().hex[:8]}",
        title=_text(subtask.title) or "Untitled slice",
        description=_text(subtask.description) or (_text(subtask.title) or "Untitled slice"),
        dependencies=_dedupe(list(subtask.dependencies)),
        file_scope=_dedupe(list(subtask.file_scope)),
        acceptance_criteria=acceptance_criteria,
        validations=validations,
        estimated_complexity=_text(subtask.estimated_complexity) or "medium",
        status=PlanStatus.CREATED.value,
        metadata={"depth": int(getattr(subtask, "depth", 0) or 0)},
    )


def _default_checkpoint_title(slice_title: str) -> str:
    return f"Checkpoint: {slice_title}"


class InitiativePlanner:
    """Builds persistent roadmap initiatives from bounded decomposition."""

    def __init__(
        self, *, repo_root: Path | None = None, decomposer: TaskDecomposer | None = None
    ) -> None:
        self.repo_root = Path(repo_root or Path.cwd()).resolve()
        self.decomposer = decomposer or TaskDecomposer()

    def plan(
        self,
        *,
        goal: str,
        rationale: str = "",
        title: str | None = None,
        dependencies: list[str] | None = None,
        validations: list[str] | None = None,
        feature_flag_name: str | None = None,
        milestone_titles: list[str] | None = None,
        checkpoint_titles: list[str] | None = None,
        planner_strategy: str = "heuristic",
        planner_model: str = "claude",
        status: str = PlanStatus.CREATED.value,
    ) -> InitiativeRecord:
        goal_text = _text(goal)
        if not goal_text:
            raise ValueError("goal is required")
        title_text = _text(title) or _derive_title(goal_text)
        rationale_text = _text(rationale) or goal_text
        dependency_values = _dedupe(list(dependencies or []))
        validation_values = _dedupe(list(validations or []))
        file_scope_hints = SwarmSpec.infer_file_scope_hints("\n".join([goal_text, rationale_text]))

        decomposition = self._decompose(
            rationale_text=rationale_text,
            validation_values=validation_values,
            dependency_values=dependency_values,
            file_scope_hints=file_scope_hints,
            planner_strategy=planner_strategy,
            planner_model=planner_model,
        )
        subtasks = list(decomposition.subtasks)
        if not subtasks:
            subtasks = [
                SubTask(
                    id="slice-1",
                    title=title_text,
                    description=rationale_text,
                    file_scope=list(file_scope_hints),
                    success_criteria={"tests": list(validation_values)},
                )
            ]
        slices = [
            _slice_from_subtask(subtask, fallback_validations=validation_values)
            for subtask in subtasks
        ]
        if not slices:
            raise ValueError("initiative planner produced no slices")

        combined_validations = _dedupe(
            validation_values + [command for item in slices for command in item.validations]
        )
        checkpoints = self._build_checkpoints(slices, checkpoint_titles or [])
        milestones = self._build_milestones(slices, checkpoints, milestone_titles or [])

        return InitiativeRecord(
            initiative_id=_initiative_id(title_text),
            title=title_text,
            goal=goal_text,
            rationale=rationale_text,
            slices=slices,
            dependencies=dependency_values,
            validations=combined_validations,
            feature_flag_name=_text(feature_flag_name) or _feature_flag_name_for_title(title_text),
            milestones=milestones,
            checkpoints=checkpoints,
            status=_text(status) or PlanStatus.CREATED.value,
            planner_rationale=_text(decomposition.rationale) or "",
            metadata={
                "planner_strategy": planner_strategy,
                "planner_model": planner_model if planner_strategy == "model" else None,
                "complexity_score": decomposition.complexity_score,
                "complexity_level": decomposition.complexity_level,
                "should_decompose": decomposition.should_decompose,
            },
        )

    def _decompose(
        self,
        rationale_text: str,
        validation_values: list[str],
        dependency_values: list[str],
        file_scope_hints: list[str],
        planner_strategy: str,
        planner_model: str,
    ):
        acceptance = [f"Run and satisfy: {command}" for command in validation_values]
        constraints = [f"Respect dependency: {item}" for item in dependency_values]
        if planner_strategy == "model":
            return self.decomposer.analyze_with_model_sync(
                rationale_text,
                planner_model=planner_model,
                file_scope_hints=file_scope_hints or None,
                acceptance_criteria=acceptance or None,
                constraints=constraints or None,
            )
        return self.decomposer.analyze(
            rationale_text,
            file_scope_hints=file_scope_hints or None,
            acceptance_criteria=acceptance or None,
            constraints=constraints or None,
        )

    def _build_checkpoints(
        self,
        slices: list[InitiativeSlice],
        checkpoint_titles: list[str],
    ) -> list[InitiativeCheckpoint]:
        if checkpoint_titles:
            checkpoints: list[InitiativeCheckpoint] = []
            for index, title in enumerate(_dedupe(checkpoint_titles), start=1):
                slice_ref = slices[min(index - 1, len(slices) - 1)]
                checkpoints.append(
                    InitiativeCheckpoint(
                        checkpoint_id=f"checkpoint-{index}",
                        title=title,
                        description=f"Checkpoint for {slice_ref.title}.",
                        dependencies=[slice_ref.slice_id],
                        validations=list(slice_ref.validations),
                        status=PlanStatus.CREATED.value,
                    )
                )
            return checkpoints
        return [
            InitiativeCheckpoint(
                checkpoint_id=f"checkpoint-{index}",
                title=_default_checkpoint_title(item.title),
                description=f"Validate completion for slice '{item.title}'.",
                dependencies=[item.slice_id],
                validations=list(item.validations),
                status=PlanStatus.CREATED.value,
            )
            for index, item in enumerate(slices, start=1)
        ]

    def _build_milestones(
        self,
        slices: list[InitiativeSlice],
        checkpoints: list[InitiativeCheckpoint],
        milestone_titles: list[str],
    ) -> list[InitiativeMilestone]:
        titles = _dedupe(milestone_titles)
        if titles:
            milestones: list[InitiativeMilestone] = []
            slice_groups = [item.slice_id for item in slices]
            checkpoint_groups = [item.checkpoint_id for item in checkpoints]
            for index, title in enumerate(titles, start=1):
                milestones.append(
                    InitiativeMilestone(
                        milestone_id=f"milestone-{index}",
                        title=title,
                        description=f"Milestone {index} for the initiative roadmap.",
                        slice_ids=list(slice_groups),
                        checkpoint_ids=list(checkpoint_groups),
                        status=PlanStatus.CREATED.value,
                    )
                )
            return milestones
        return [
            InitiativeMilestone(
                milestone_id="milestone-1",
                title="Planned slices ready for execution",
                description="Initial initiative decomposition persisted for follow-on execution.",
                slice_ids=[item.slice_id for item in slices],
                checkpoint_ids=[item.checkpoint_id for item in checkpoints],
                status=PlanStatus.CREATED.value,
            )
        ]
