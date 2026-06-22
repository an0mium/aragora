"""Bridge Nomic task decompositions into goal-conductor mission YAML."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


DEFAULT_GOAL_CONDUCTOR_OUTPUT_DIR = ".aragora/goal-conductor"
DEFAULT_AGENTS = ["codex", "claude"]
DEFAULT_STOP_CONDITION = (
    "Stop when every lane reaches a draft PR, a precise blocker report, or a handoff."
)


def _slug(value: str, *, fallback: str = "goal") -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-").lower()
    return slug[:80] or fallback


def _text(value: Any, *, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def _subtasks_from_decomposition(decomp: Any) -> list[Any]:
    subtasks = getattr(decomp, "subtasks", None)
    if isinstance(subtasks, list):
        return subtasks
    return []


def _subtask_title(subtask: Any, *, index: int) -> str:
    return _text(getattr(subtask, "title", ""), fallback=f"Subtask {index}")


def _subtask_description(subtask: Any, *, title: str) -> str:
    return _text(getattr(subtask, "description", ""), fallback=title)


def _subtask_prompt(subtask: Any, *, title: str, description: str) -> str:
    parts = [
        "You are an Aragora implementation worker.",
        "",
        f"Goal: {title}",
        "",
        description,
        "",
        "Rules:",
        "- Rebuild live repo and queue truth before editing.",
        "- Keep the change bounded to this lane.",
        "- Do not merge PRs.",
        "- Do not run observe-outcomes --write.",
        "- Do not install launchd jobs, close issues, or clean worktrees.",
        "- Stop after one draft PR or one precise blocker report with validation evidence.",
    ]
    file_scope = getattr(subtask, "file_scope", None) or []
    if file_scope:
        parts.extend(["", "Suggested file scope:"])
        parts.extend(f"- {path}" for path in file_scope)
    success_criteria = getattr(subtask, "success_criteria", None) or {}
    if success_criteria:
        parts.extend(["", "Success criteria:"])
        if isinstance(success_criteria, dict):
            parts.extend(f"- {key}: {value}" for key, value in success_criteria.items())
        else:
            parts.append(f"- {success_criteria}")
    return "\n".join(parts)


def decomposition_to_mission(
    decomp: Any,
    *,
    objective: str,
    stop_condition: str = "",
    agents: list[str] | None = None,
    include_panel_review: bool = True,
    queue_cap: int = 6,
) -> dict[str, Any]:
    """Convert a TaskDecomposition-like object into a conductor mission dict.

    The returned mapping intentionally matches ``scripts/goal_conductor.py``
    ``Mission.from_dict()`` so the user can feed it directly to the conductor
    after writing YAML.
    """
    objective_text = _text(objective, fallback=getattr(decomp, "original_task", ""))
    if not objective_text:
        objective_text = "Aragora goal-mode mission"
    selected_agents = [agent.strip().lower() for agent in (agents or DEFAULT_AGENTS) if agent]
    if not selected_agents:
        selected_agents = list(DEFAULT_AGENTS)

    subtasks = _subtasks_from_decomposition(decomp)
    if not subtasks:
        subtasks = [
            type(
                "_MissionBridgeSubtask",
                (),
                {
                    "title": objective_text[:80],
                    "description": objective_text,
                    "file_scope": [],
                    "success_criteria": {},
                },
            )()
        ]

    lanes: list[dict[str, Any]] = []
    for index, subtask in enumerate(subtasks[:2], start=1):
        title = _subtask_title(subtask, index=index)
        description = _subtask_description(subtask, title=title)
        agent = selected_agents[(index - 1) % len(selected_agents)]
        lanes.append(
            {
                "id": f"implementation-{index}",
                "agent": agent,
                "mode": "implementation",
                "goal": title,
                "cwd": ".",
                "autonomous": True,
                "source": "mission-bridge",
                "status": "active",
                "next_action": "Open one draft PR or report one blocker, then stop.",
                "prompt": _subtask_prompt(subtask, title=title, description=description),
            }
        )

    if include_panel_review:
        lanes.append(
            {
                "id": "adversarial-review",
                "mode": "panel",
                "agents_spec": "heterogeneous",
                "goal": f"Review mission plan: {objective_text[:120]}",
                "round_id": f"{_slug(objective_text)}-mission-review",
                "prompt": (
                    "Review this Aragora conductor mission before execution.\n\n"
                    f"Objective: {objective_text}\n\n"
                    "Focus on hidden safety, settlement, evidence, queue-cap, "
                    "validation, and wedge-alignment risks. Output concise findings "
                    "and a proceed/hold/narrow recommendation."
                ),
            }
        )

    return {
        "name": _slug(objective_text),
        "objective": objective_text,
        "stop_condition": stop_condition or DEFAULT_STOP_CONDITION,
        "checkpoints": [
            "Snapshot root, open PRs, publisher, proof-loop health, and agent lanes.",
            "Run at most two implementation lanes and one review lane.",
            "Stop at hard gates and write a handoff.",
        ],
        "base_branch": "main",
        "output_dir": DEFAULT_GOAL_CONDUCTOR_OUTPUT_DIR,
        "limits": {
            "queue_cap": int(queue_cap),
            "max_implementation_lanes": 2,
            "max_review_lanes": 1,
        },
        "allowed_mutations": [
            "draft_pr",
            "bounded_repo_patch",
            "status_artifact",
        ],
        "stop_conditions": [
            "root_dirty",
            "queue_at_cap",
            "tier4_gate",
            "destructive_or_live_spend_request",
        ],
        "collect_merge_packets": True,
        "max_merge_packets": 5,
        "lanes": lanes,
    }


def write_mission_yaml(mission: dict[str, Any], path: str | Path) -> Path:
    """Write a conductor mission dict to YAML and return the path."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - repo dependency missing
        raise RuntimeError("PyYAML is required to write mission YAML") from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(mission, sort_keys=False), encoding="utf-8")
    return output_path
