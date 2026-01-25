"""
Workflow Presets - Pre-defined workflow configurations.

Provides ready-to-use workflow definitions for common use cases:
- debate_workflow: Standard multi-agent debate
- nomic_workflow: Self-improvement cycle with human checkpoints
- knowledge_pipeline: Document ingestion workflow
- code_review: Code review with multiple perspectives
- compliance_check: Compliance validation workflow

Usage:
    from aragora.workflow.presets import load_preset, list_presets

    # List available presets
    presets = list_presets()

    # Load a specific preset
    workflow = load_preset("nomic_workflow")

    # Execute the workflow
    engine = WorkflowEngine()
    result = await engine.execute(workflow, inputs)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from aragora.workflow.types import WorkflowDefinition

logger = logging.getLogger(__name__)

# Directory containing preset YAML files
PRESETS_DIR = Path(__file__).parent


def list_presets() -> List[str]:
    """
    List all available workflow presets.

    Returns:
        List of preset names (without .yaml extension)
    """
    return [f.stem for f in PRESETS_DIR.glob("*.yaml") if not f.name.startswith("_")]


def load_preset(
    name: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> WorkflowDefinition:
    """
    Load a workflow preset by name.

    Args:
        name: Preset name (with or without .yaml extension)
        overrides: Optional configuration overrides

    Returns:
        WorkflowDefinition instance

    Raises:
        FileNotFoundError: If preset doesn't exist
        ValueError: If preset is invalid
    """
    # Normalize name
    preset_name = name.replace(".yaml", "")
    preset_file = PRESETS_DIR / f"{preset_name}.yaml"

    if not preset_file.exists():
        available = list_presets()
        raise FileNotFoundError(f"Preset '{preset_name}' not found. Available presets: {available}")

    # Load YAML
    with preset_file.open() as f:
        preset_data = yaml.safe_load(f)

    # Apply overrides
    if overrides:
        preset_data = _deep_merge(preset_data, overrides)

    # Convert to WorkflowDefinition
    return _parse_workflow_definition(preset_data)


def get_preset_info(name: str) -> Dict[str, Any]:
    """
    Get metadata about a preset without fully loading it.

    Args:
        name: Preset name

    Returns:
        Dict with id, name, description, version
    """
    preset_name = name.replace(".yaml", "")
    preset_file = PRESETS_DIR / f"{preset_name}.yaml"

    if not preset_file.exists():
        raise FileNotFoundError(f"Preset '{preset_name}' not found")

    with preset_file.open() as f:
        preset_data = yaml.safe_load(f)

    return {
        "id": preset_data.get("id", preset_name),
        "name": preset_data.get("name", preset_name),
        "description": preset_data.get("description", ""),
        "version": preset_data.get("version", "1.0"),
        "entry_point": preset_data.get("entry_point", ""),
        "step_count": len(preset_data.get("steps", [])),
    }


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _parse_workflow_definition(data: Dict[str, Any]) -> WorkflowDefinition:
    """Parse YAML data into WorkflowDefinition."""
    from aragora.workflow.types import StepDefinition, TransitionRule

    steps = []
    transitions = []

    for step_data in data.get("steps", []):
        step_id = step_data.get("id", "")
        step = StepDefinition(
            id=step_id,
            name=step_data.get("name", step_id),  # Use id as fallback for name
            step_type=step_data.get("type", "task"),
            config=step_data.get("config", {}),
            next_steps=[step_data.get("next")] if step_data.get("next") else [],
        )

        # Parse transitions - add to workflow-level transitions list
        if "on_success" in step_data:
            step.next_steps = [step_data["on_success"]]
        if "on_failure" in step_data:
            transitions.append(
                TransitionRule(
                    id=f"{step_id}_failure",
                    from_step=step_id,
                    to_step=step_data["on_failure"],
                    condition="failure",
                )
            )
        if "on_reject" in step_data:
            transitions.append(
                TransitionRule(
                    id=f"{step_id}_reject",
                    from_step=step_id,
                    to_step=step_data["on_reject"],
                    condition="rejected",
                )
            )

        steps.append(step)

    return WorkflowDefinition(
        id=data.get("id", ""),
        name=data.get("name", ""),
        description=data.get("description", ""),
        version=data.get("version", "1.0"),
        steps=steps,
        transitions=transitions,
        entry_step=data.get("entry_point", steps[0].id if steps else ""),
        metadata=data.get("metadata", {}),
    )


__all__ = [
    "list_presets",
    "load_preset",
    "get_preset_info",
    "PRESETS_DIR",
]
