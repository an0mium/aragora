"""Tests for the self-development workflow template."""

from aragora.workflow.engine import WorkflowEngine
from aragora.workflow.templates.self_development import create_self_development_workflow


def test_self_development_steps_are_registered() -> None:
    """Self-development template should only use step types known to the engine."""
    workflow = create_self_development_workflow(
        objective="Improve reliability",
        tracks=["sme", "qa"],
    )
    engine = WorkflowEngine()

    missing_types = {
        step.step_type for step in workflow.steps if step.step_type not in engine._step_types
    }
    assert not missing_types


def test_parallel_nomic_template_step_type_is_registered() -> None:
    """Parallel branch step template should resolve to a registered step type."""
    workflow = create_self_development_workflow(
        objective="Improve reliability",
        tracks=["sme", "qa"],
    )
    engine = WorkflowEngine()

    parallel_steps = [step for step in workflow.steps if step.step_type == "parallel"]
    assert parallel_steps

    for step in parallel_steps:
        template_type = step.config.get("step_template", {}).get("type")
        assert template_type in engine._step_types
