"""
Evaluation helpers for comparing Nomic multi-agent output vs single-agent baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class EvalTask:
    """A task specification for evaluation runs."""

    task_id: str
    title: str
    description: str
    acceptance_criteria: list[str] = field(default_factory=list)
    scope: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    test_commands: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalTask":
        if not isinstance(data, dict):
            raise ValueError("Task data must be a dict")
        task_id = str(data.get("task_id") or data.get("id") or "").strip()
        title = str(data.get("title") or "").strip()
        description = str(data.get("description") or "").strip()
        if not task_id:
            raise ValueError("Task requires task_id")
        if not title:
            raise ValueError(f"Task {task_id} requires title")
        if not description:
            raise ValueError(f"Task {task_id} requires description")
        return cls(
            task_id=task_id,
            title=title,
            description=description,
            acceptance_criteria=list(data.get("acceptance_criteria") or []),
            scope=list(data.get("scope") or []),
            constraints=list(data.get("constraints") or []),
            test_commands=list(data.get("test_commands") or []),
            labels=list(data.get("labels") or []),
        )


def load_tasks(path: Path) -> list[EvalTask]:
    """Load evaluation tasks from a JSON file."""
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        tasks = data.get("tasks", [])
    else:
        tasks = data
    if not isinstance(tasks, list):
        raise ValueError('Tasks file must contain a list or {"tasks": [...]}')
    return [EvalTask.from_dict(item) for item in tasks]


def build_task_prompt(task: EvalTask) -> str:
    """Build a structured prompt from a task specification."""
    lines = [
        f"TASK ID: {task.task_id}",
        f"TITLE: {task.title}",
        "",
        "DESCRIPTION:",
        task.description.strip(),
    ]
    if task.acceptance_criteria:
        lines.append("\nACCEPTANCE CRITERIA:")
        for item in task.acceptance_criteria:
            lines.append(f"- {item}")
    if task.scope:
        lines.append("\nSCOPE:")
        for item in task.scope:
            lines.append(f"- {item}")
    if task.constraints:
        lines.append("\nCONSTRAINTS:")
        for item in task.constraints:
            lines.append(f"- {item}")
    if task.test_commands:
        lines.append("\nTEST COMMANDS:")
        for item in task.test_commands:
            lines.append(f"- {item}")
    if task.labels:
        lines.append("\nLABELS:")
        for item in task.labels:
            lines.append(f"- {item}")
    return "\n".join(lines).strip() + "\n"


__all__ = ["EvalTask", "load_tasks", "build_task_prompt"]
