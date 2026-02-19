"""Tests for the SME pipeline templates library.

Verifies that each template returns a valid PipelineConfig with the
correct settings pre-filled, and that seed ideas are non-empty and
meaningful for the template's use case.
"""

from __future__ import annotations

import pytest

from aragora.pipeline.idea_to_execution import PipelineConfig
from aragora.pipeline.templates import (
    PipelineTemplate,
    TEMPLATE_REGISTRY,
    get_template,
    get_template_config,
    list_templates,
)


# =========================================================================
# The 6 core SME templates
# =========================================================================

SME_TEMPLATES = [
    "product_launch",
    "bug_triage",
    "content_calendar",
    "strategic_review",
    "hiring_pipeline",
    "compliance_audit",
]


class TestSMETemplatesExist:
    """All 6 SME templates are registered and retrievable."""

    @pytest.mark.parametrize("name", SME_TEMPLATES)
    def test_template_registered(self, name: str):
        assert name in TEMPLATE_REGISTRY, f"{name} not in registry"

    @pytest.mark.parametrize("name", SME_TEMPLATES)
    def test_template_retrievable(self, name: str):
        t = get_template(name)
        assert t is not None
        assert isinstance(t, PipelineTemplate)
        assert t.name == name


class TestTemplateReturnsValidConfig:
    """Each template produces a valid PipelineConfig."""

    @pytest.mark.parametrize("name", SME_TEMPLATES)
    def test_to_pipeline_config_returns_pipeline_config(self, name: str):
        t = get_template(name)
        assert t is not None
        config = t.to_pipeline_config()
        assert isinstance(config, PipelineConfig)

    @pytest.mark.parametrize("name", SME_TEMPLATES)
    def test_config_workflow_mode_is_valid(self, name: str):
        t = get_template(name)
        assert t is not None
        config = t.to_pipeline_config()
        assert config.workflow_mode in ("quick", "debate")

    @pytest.mark.parametrize("name", SME_TEMPLATES)
    def test_config_booleans_are_bool(self, name: str):
        t = get_template(name)
        assert t is not None
        config = t.to_pipeline_config()
        assert isinstance(config.enable_smart_goals, bool)
        assert isinstance(config.enable_elo_assignment, bool)
        assert isinstance(config.enable_km_precedents, bool)
        assert isinstance(config.dry_run, bool)


class TestSeedIdeas:
    """Each template has meaningful, non-empty seed ideas."""

    @pytest.mark.parametrize("name", SME_TEMPLATES)
    def test_seed_ideas_non_empty(self, name: str):
        t = get_template(name)
        assert t is not None
        assert len(t.seed_ideas) >= 3, f"{name} needs at least 3 seed ideas"

    @pytest.mark.parametrize("name", SME_TEMPLATES)
    def test_seed_ideas_are_strings(self, name: str):
        t = get_template(name)
        assert t is not None
        for idea in t.seed_ideas:
            assert isinstance(idea, str)
            assert len(idea) > 10, f"Idea too short: {idea!r}"

    @pytest.mark.parametrize("name", SME_TEMPLATES)
    def test_seed_ideas_alias_matches_stage_1(self, name: str):
        """seed_ideas property is an alias for stage_1_ideas."""
        t = get_template(name)
        assert t is not None
        assert t.seed_ideas is t.stage_1_ideas

    @pytest.mark.parametrize("name", SME_TEMPLATES)
    def test_seed_ideas_are_unique(self, name: str):
        t = get_template(name)
        assert t is not None
        assert len(t.seed_ideas) == len(set(t.seed_ideas)), (
            f"{name} has duplicate seed ideas"
        )


# =========================================================================
# Template-specific settings
# =========================================================================


class TestProductLaunchSettings:
    """product_launch: smart goals ON, human approval ON."""

    def test_smart_goals_enabled(self):
        t = get_template("product_launch")
        assert t is not None
        assert t.enable_smart_goals is True

    def test_human_approval_required(self):
        t = get_template("product_launch")
        assert t is not None
        assert t.human_approval_required is True

    def test_workflow_mode_quick(self):
        t = get_template("product_launch")
        assert t is not None
        assert t.workflow_mode == "quick"

    def test_config_reflects_template(self):
        t = get_template("product_launch")
        assert t is not None
        config = t.to_pipeline_config()
        assert config.enable_smart_goals is True
        assert config.enable_elo_assignment is True


class TestBugTriageSettings:
    """bug_triage: quick mode, ELO ON for best debugging agents."""

    def test_workflow_mode_quick(self):
        t = get_template("bug_triage")
        assert t is not None
        assert t.workflow_mode == "quick"

    def test_elo_assignment_enabled(self):
        t = get_template("bug_triage")
        assert t is not None
        assert t.enable_elo_assignment is True

    def test_smart_goals_off(self):
        """Bug triage does not need SMART goals."""
        t = get_template("bug_triage")
        assert t is not None
        assert t.enable_smart_goals is False

    def test_no_human_approval(self):
        """Bugs flow fast -- no human gate."""
        t = get_template("bug_triage")
        assert t is not None
        assert t.human_approval_required is False

    def test_config_reflects_template(self):
        t = get_template("bug_triage")
        assert t is not None
        config = t.to_pipeline_config()
        assert config.workflow_mode == "quick"
        assert config.enable_elo_assignment is True
        assert config.enable_smart_goals is False


class TestContentCalendarSettings:
    """content_calendar: smart goals ON."""

    def test_smart_goals_enabled(self):
        t = get_template("content_calendar")
        assert t is not None
        assert t.enable_smart_goals is True

    def test_km_precedents_enabled(self):
        t = get_template("content_calendar")
        assert t is not None
        assert t.enable_km_precedents is True

    def test_config_reflects_template(self):
        t = get_template("content_calendar")
        assert t is not None
        config = t.to_pipeline_config()
        assert config.enable_smart_goals is True
        assert config.enable_km_precedents is True


class TestStrategicReviewSettings:
    """strategic_review: debate mode, KM ON, everything maximized."""

    def test_workflow_mode_debate(self):
        t = get_template("strategic_review")
        assert t is not None
        assert t.workflow_mode == "debate"

    def test_km_precedents_enabled(self):
        t = get_template("strategic_review")
        assert t is not None
        assert t.enable_km_precedents is True

    def test_all_features_enabled(self):
        t = get_template("strategic_review")
        assert t is not None
        assert t.enable_smart_goals is True
        assert t.enable_elo_assignment is True
        assert t.enable_km_precedents is True
        assert t.human_approval_required is True

    def test_config_reflects_template(self):
        t = get_template("strategic_review")
        assert t is not None
        config = t.to_pipeline_config()
        assert config.workflow_mode == "debate"
        assert config.enable_smart_goals is True
        assert config.enable_elo_assignment is True
        assert config.enable_km_precedents is True


class TestHiringPipelineSettings:
    """hiring_pipeline: human approval ON."""

    def test_human_approval_required(self):
        t = get_template("hiring_pipeline")
        assert t is not None
        assert t.human_approval_required is True

    def test_smart_goals_enabled(self):
        t = get_template("hiring_pipeline")
        assert t is not None
        assert t.enable_smart_goals is True

    def test_config_reflects_template(self):
        t = get_template("hiring_pipeline")
        assert t is not None
        config = t.to_pipeline_config()
        assert config.enable_smart_goals is True
        assert config.enable_km_precedents is True


class TestComplianceAuditSettings:
    """compliance_audit: debate mode, human approval ON."""

    def test_workflow_mode_debate(self):
        t = get_template("compliance_audit")
        assert t is not None
        assert t.workflow_mode == "debate"

    def test_human_approval_required(self):
        t = get_template("compliance_audit")
        assert t is not None
        assert t.human_approval_required is True

    def test_vertical_profile(self):
        t = get_template("compliance_audit")
        assert t is not None
        assert t.vertical_profile == "compliance_sox"

    def test_config_reflects_template(self):
        t = get_template("compliance_audit")
        assert t is not None
        config = t.to_pipeline_config()
        assert config.workflow_mode == "debate"
        assert config.enable_smart_goals is True
        assert config.enable_elo_assignment is True


# =========================================================================
# Convenience API
# =========================================================================


class TestGetTemplateConfig:
    """Test the get_template_config() convenience function."""

    def test_returns_tuple_for_valid_template(self):
        result = get_template_config("strategic_review")
        assert result is not None
        config, ideas = result
        assert isinstance(config, PipelineConfig)
        assert isinstance(ideas, list)
        assert len(ideas) > 0

    def test_returns_none_for_unknown_template(self):
        assert get_template_config("nonexistent") is None

    @pytest.mark.parametrize("name", SME_TEMPLATES)
    def test_config_and_ideas_for_all_sme(self, name: str):
        result = get_template_config(name)
        assert result is not None
        config, ideas = result
        assert isinstance(config, PipelineConfig)
        assert len(ideas) >= 3


class TestToDictIncludesConfigFields:
    """to_dict() includes PipelineConfig-relevant fields."""

    @pytest.mark.parametrize("name", SME_TEMPLATES)
    def test_to_dict_has_config_fields(self, name: str):
        t = get_template(name)
        assert t is not None
        d = t.to_dict()
        assert "workflow_mode" in d
        assert "enable_smart_goals" in d
        assert "enable_elo_assignment" in d
        assert "enable_km_precedents" in d
        assert "human_approval_required" in d
        assert d["workflow_mode"] in ("quick", "debate")


class TestFromDemoIntegration:
    """from_demo() returns valid pipeline output with flywheel features.

    The demo may use the strategic_review template or bespoke ideas;
    these tests verify the output is well-formed either way.
    """

    def test_demo_returns_tuple(self):
        from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

        result = IdeaToExecutionPipeline.from_demo()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_demo_config_has_smart_goals(self):
        from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

        _, config = IdeaToExecutionPipeline.from_demo()
        assert config.enable_smart_goals is True

    def test_demo_config_has_elo(self):
        from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

        _, config = IdeaToExecutionPipeline.from_demo()
        assert config.enable_elo_assignment is True

    def test_demo_config_has_km_precedents(self):
        from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

        _, config = IdeaToExecutionPipeline.from_demo()
        assert config.enable_km_precedents is True

    def test_demo_config_has_dry_run(self):
        from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

        _, config = IdeaToExecutionPipeline.from_demo()
        assert config.dry_run is True

    def test_demo_has_ideas_canvas(self):
        from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

        result, _ = IdeaToExecutionPipeline.from_demo()
        assert result.ideas_canvas is not None
        assert len(result.ideas_canvas.nodes) >= 4

    def test_demo_has_goal_graph(self):
        from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline

        result, _ = IdeaToExecutionPipeline.from_demo()
        assert result.goal_graph is not None
        assert len(result.goal_graph.goals) >= 1

    def test_strategic_review_template_is_demo_ready(self):
        """The strategic_review template should produce valid demo output."""
        template = get_template("strategic_review")
        assert template is not None
        config = template.to_pipeline_config()
        # strategic_review uses all features -- ideal for demo
        assert config.workflow_mode == "debate"
        assert config.enable_smart_goals is True
        assert config.enable_elo_assignment is True
        assert config.enable_km_precedents is True
        assert template.human_approval_required is True
