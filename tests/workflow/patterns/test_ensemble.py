"""Tests for the Ensemble workflow pattern."""

from aragora.workflow.patterns.ensemble import EnsemblePattern


def test_ensemble_pattern_builds_agent_pool_step():
    workflow = EnsemblePattern.create(
        name="Ensemble Test",
        agents=["claude", "gpt4", "gemini"],
        task="Analyze {input}",
        selection_strategy="best_score",
        samples_per_agent=2,
        include_candidates=True,
    )

    assert workflow.entry_step == "ensemble_select"
    step = workflow.get_step("ensemble_select")
    assert step is not None
    assert step.config["agent_pool"] == ["claude", "gpt4", "gemini"]
    assert step.config["selection_strategy"] == "best_score"
    assert step.config["samples_per_agent"] == 2
    assert step.config["include_candidates"] is True
