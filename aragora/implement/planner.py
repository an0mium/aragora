"""
Gemini-based implementation planner.

Analyzes a design and decomposes it into discrete implementation tasks
with complexity scoring and dependency tracking.
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from aragora.agents.cli_agents import GeminiCLIAgent

from .types import ImplementPlan, ImplementTask

# Feature flag for task decomposition (default OFF until tested)
DECOMPOSE_FAILED = os.environ.get("IMPL_DECOMPOSE_FAILED", "0") == "1"


PLAN_PROMPT_TEMPLATE = """Analyze this implementation design and create a detailed execution plan.

## Design to Implement

{design}

## Repository Context

Working directory: {repo_path}

## Your Task

Break this design into discrete implementation tasks. Each task should be:
1. Atomic - can be completed independently (after dependencies)
2. Testable - produces verifiable output
3. Sized correctly - neither too large nor too small

## Output Format

Output ONLY valid JSON with this structure:

{{
  "tasks": [
    {{
      "id": "task-1",
      "description": "Clear description of what to implement",
      "files": ["path/to/file1.py", "path/to/file2.py"],
      "complexity": "simple|moderate|complex",
      "dependencies": []
    }},
    {{
      "id": "task-2",
      "description": "Second task description",
      "files": ["path/to/file3.py"],
      "complexity": "simple",
      "dependencies": ["task-1"]
    }}
  ]
}}

## Complexity Guidelines

- **simple**: Single file, <50 lines, straightforward logic (e.g., add a function, create a dataclass)
- **moderate**: 2-3 files, some coordination needed (e.g., add a feature with tests)
- **complex**: 4+ files, architectural changes, or intricate logic (e.g., new module, refactor)

## Important Rules

1. Order tasks by dependencies (tasks with no dependencies first)
2. Include test files if tests are mentioned in the design
3. Be specific about file paths relative to the repository root
4. Each task should produce a working, non-breaking change

Output ONLY the JSON, no explanation or markdown formatting.
"""


def extract_json(text: str) -> str:
    """Extract JSON from a response that might contain other text."""
    # Try to find JSON in code blocks first
    code_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if code_block_match:
        return code_block_match.group(1)

    # Try to find raw JSON object
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        return json_match.group(0)

    return text


def validate_plan(plan_data: dict) -> list[str]:
    """Validate plan structure and return list of errors."""
    errors = []

    if "tasks" not in plan_data:
        errors.append("Missing 'tasks' key in plan")
        return errors

    if not isinstance(plan_data["tasks"], list):
        errors.append("'tasks' must be a list")
        return errors

    if len(plan_data["tasks"]) == 0:
        errors.append("Plan has no tasks")
        return errors

    task_ids = set()
    for i, task in enumerate(plan_data["tasks"]):
        if not isinstance(task, dict):
            errors.append(f"Task {i} is not a dict")
            continue

        if "id" not in task:
            errors.append(f"Task {i} missing 'id'")
        else:
            if task["id"] in task_ids:
                errors.append(f"Duplicate task id: {task['id']}")
            task_ids.add(task["id"])

        if "description" not in task:
            errors.append(f"Task {i} missing 'description'")

        if "files" not in task or not isinstance(task.get("files"), list):
            errors.append(f"Task {i} missing or invalid 'files'")

        complexity = task.get("complexity", "")
        if complexity not in ("simple", "moderate", "complex"):
            errors.append(f"Task {i} has invalid complexity: {complexity}")

        # Check dependencies reference valid task IDs
        for dep in task.get("dependencies", []):
            if dep not in task_ids and dep not in [t.get("id") for t in plan_data["tasks"]]:
                # Allow forward references for now, but check later
                pass

    return errors


async def generate_implement_plan(
    design: str,
    repo_path: Path,
    timeout: int = 180,
    gemini_model: str = "gemini-3-pro-preview",
) -> ImplementPlan:
    """
    Use Gemini to decompose a design into implementation tasks.

    Args:
        design: The implementation design text
        repo_path: Path to the repository root
        timeout: Timeout for Gemini API call
        gemini_model: Gemini model to use

    Returns:
        ImplementPlan with decomposed tasks

    Raises:
        ValueError: If the plan cannot be parsed or is invalid
    """
    gemini = GeminiCLIAgent(
        name="implement-planner",
        model=gemini_model,
        role="planner",
        timeout=timeout,
    )

    prompt = PLAN_PROMPT_TEMPLATE.format(
        design=design,
        repo_path=str(repo_path),
    )

    logger.info("  Generating implementation plan with Gemini...")
    response = await gemini.generate(prompt)

    # Extract and parse JSON
    json_str = extract_json(response)

    try:
        plan_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse plan JSON: {e}\nResponse: {response[:500]}")

    # Validate
    errors = validate_plan(plan_data)
    if errors:
        raise ValueError(f"Invalid plan: {'; '.join(errors)}")

    # Build ImplementPlan
    design_hash = hashlib.sha256(design.encode()).hexdigest()
    tasks = [ImplementTask.from_dict(t) for t in plan_data["tasks"]]

    logger.info(f"  Plan generated: {len(tasks)} tasks")
    for task in tasks:
        logger.info(f"    - [{task.complexity}] {task.id}: {task.description[:50]}...")

    return ImplementPlan(design_hash=design_hash, tasks=tasks)


DECOMPOSE_PROMPT_TEMPLATE = """A complex implementation task failed. Break it into smaller subtasks.

## Original Task
ID: {task_id}
Description: {description}
Files: {files}
Original Complexity: {complexity}

## Error
{error}

## Instructions
Split this into 2-3 smaller tasks, each handling a subset of files.
Each subtask should:
1. Be completable independently
2. Handle 1-2 files max
3. Have complexity "simple" or "moderate"

## Output Format
Output ONLY valid JSON:
{{
  "subtasks": [
    {{
      "id": "{task_id}-a",
      "description": "First part of the original task",
      "files": ["file1.py"],
      "complexity": "simple",
      "dependencies": []
    }},
    {{
      "id": "{task_id}-b",
      "description": "Second part of the original task",
      "files": ["file2.py"],
      "complexity": "simple",
      "dependencies": ["{task_id}-a"]
    }}
  ]
}}

Output ONLY the JSON, no explanation.
"""


async def decompose_failed_task(
    task: ImplementTask,
    error: str,
    repo_path: Path,
    gemini_model: str = "gemini-2.0-flash",
) -> list[ImplementTask]:
    """
    Decompose a failed complex task into smaller subtasks.

    Uses DECOMPOSE_FAILED feature flag (default OFF).
    Only decomposes tasks that are complex and have more than 2 files.

    Args:
        task: The failed task to decompose
        error: Error message from the failure
        repo_path: Path to the repository
        gemini_model: Gemini model to use for decomposition

    Returns:
        List of subtasks (or original task if not worth decomposing)
    """
    if not DECOMPOSE_FAILED:
        return [task]

    # Only decompose complex tasks with many files
    if task.complexity != "complex" or len(task.files) <= 2:
        logger.info(f"    Task {task.id} not suitable for decomposition")
        return [task]

    logger.info(f"    Decomposing failed task {task.id} into subtasks...")

    gemini = GeminiCLIAgent(
        name="task-decomposer",
        model=gemini_model,
        role="decomposer",
        timeout=120,
    )

    prompt = DECOMPOSE_PROMPT_TEMPLATE.format(
        task_id=task.id,
        description=task.description,
        files=", ".join(task.files),
        complexity=task.complexity,
        error=error[:500] if error else "Unknown error",
    )

    try:
        response = await gemini.generate(prompt)
        json_str = extract_json(response)
        data = json.loads(json_str)

        if "subtasks" not in data or not data["subtasks"]:
            logger.info("    No subtasks generated, keeping original")
            return [task]

        subtasks = []
        for st in data["subtasks"]:
            # Inherit dependencies from original task
            deps = list(task.dependencies) + st.get("dependencies", [])
            subtask = ImplementTask(
                id=st["id"],
                description=st["description"],
                files=st.get("files", []),
                complexity=st.get("complexity", "moderate"),
                dependencies=deps,
            )
            subtasks.append(subtask)

        logger.info(f"    Decomposed into {len(subtasks)} subtasks:")
        for st in subtasks:
            logger.info(f"      - [{st.complexity}] {st.id}: {st.description[:40]}...")

        return subtasks

    except Exception as e:
        logger.warning(f"    Decomposition failed: {e}, keeping original task")
        return [task]


def create_single_task_plan(design: str, repo_path: Path) -> ImplementPlan:
    """
    Create a fallback single-task plan when Gemini planning fails.

    This preserves the legacy behavior of treating the entire design
    as a single implementation task.
    """
    design_hash = hashlib.sha256(design.encode()).hexdigest()

    task = ImplementTask(
        id="task-1",
        description="Implement the complete design",
        files=[],  # Unknown - let the executor figure it out
        complexity="complex",  # Assume complex for safety
        dependencies=[],
    )

    return ImplementPlan(design_hash=design_hash, tasks=[task])
