"""Tests for aragora.pipeline.templates -- Pipeline template library."""

import pytest

from aragora.pipeline.templates import (
    PipelineTemplate,
    TEMPLATE_REGISTRY,
    get_template,
    get_template_config,
    list_templates,
)


class TestTemplateRegistry:
    """Test template registration and lookup."""

    def test_all_templates_registered(self):
        """All 9 templates should be registered (6 SME + 3 legacy)."""
        assert len(TEMPLATE_REGISTRY) >= 9
        expected_sme = {
            "product_launch",
            "bug_triage",
            "content_calendar",
            "strategic_review",
            "hiring_pipeline",
            "compliance_audit",
        }
        expected_legacy = {
            "hiring_decision",
            "market_entry",
            "vendor_selection",
        }
        registered = set(TEMPLATE_REGISTRY.keys())
        assert expected_sme.issubset(registered)
        assert expected_legacy.issubset(registered)

    def test_get_template_found(self):
        t = get_template("hiring_decision")
        assert t is not None
        assert t.name == "hiring_decision"
        assert t.display_name == "Hiring Decision"

    def test_get_template_not_found(self):
        assert get_template("nonexistent") is None

    def test_list_templates_all(self):
        templates = list_templates()
        assert len(templates) >= 9
        assert all(isinstance(t, PipelineTemplate) for t in templates)

    def test_list_templates_filtered_by_category(self):
        templates = list_templates(category="compliance")
        assert len(templates) == 1
        assert templates[0].name == "compliance_audit"

    def test_list_templates_empty_category(self):
        templates = list_templates(category="nonexistent_category")
        assert len(templates) == 0


class TestPipelineTemplate:
    """Test PipelineTemplate dataclass."""

    def test_to_dict(self):
        t = get_template("product_launch")
        assert t is not None
        d = t.to_dict()
        assert d["name"] == "product_launch"
        assert d["display_name"] == "Product Launch"
        assert d["category"] == "product"
        assert d["idea_count"] == len(t.stage_1_ideas)
        assert "stage_1_ideas" in d
        assert "tags" in d
        # New PipelineConfig fields present in to_dict
        assert "workflow_mode" in d
        assert "enable_smart_goals" in d
        assert "enable_elo_assignment" in d
        assert "enable_km_precedents" in d
        assert "human_approval_required" in d

    def test_each_template_has_ideas(self):
        """Every template must have at least 3 pre-populated ideas."""
        for name, t in TEMPLATE_REGISTRY.items():
            assert len(t.stage_1_ideas) >= 3, f"{name} has too few ideas"

    def test_each_template_has_agent_config(self):
        """Every template must specify agent configuration."""
        for name, t in TEMPLATE_REGISTRY.items():
            assert "min_agents" in t.agent_config, f"{name} missing min_agents"
            assert t.agent_config["min_agents"] >= 2, f"{name} needs >=2 agents"

    def test_each_template_has_tags(self):
        """Every template must have at least one tag."""
        for name, t in TEMPLATE_REGISTRY.items():
            assert len(t.tags) > 0, f"{name} has no tags"

    def test_compliance_template_has_vertical_profile(self):
        t = get_template("compliance_audit")
        assert t is not None
        assert t.vertical_profile == "compliance_sox"


class TestTemplateCreatePipeline:
    """Test creating pipelines from templates."""

    def test_create_pipeline_returns_result(self):
        t = get_template("vendor_selection")
        assert t is not None
        result = t.create_pipeline()
        assert result is not None
        assert result.pipeline_id.startswith("pipe-vendor_selection-")

    def test_create_pipeline_populates_ideas(self):
        t = get_template("hiring_decision")
        assert t is not None
        result = t.create_pipeline()
        # The pipeline should have ideas and goals populated
        assert result.ideas_canvas is not None
        assert result.goal_graph is not None

    def test_create_pipeline_attaches_template_metadata(self):
        t = get_template("market_entry")
        assert t is not None
        result = t.create_pipeline()
        assert result.stage_status.get("_template") == "market_entry"

    def test_create_pipeline_unique_ids(self):
        t = get_template("product_launch")
        assert t is not None
        r1 = t.create_pipeline()
        r2 = t.create_pipeline()
        assert r1.pipeline_id != r2.pipeline_id
