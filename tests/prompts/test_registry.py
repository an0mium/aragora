"""Tests for the prompt registry module."""

from __future__ import annotations

import pytest

from aragora.prompts.registry import (
    PromptMetrics,
    PromptRegistry,
    PromptTemplate,
    get_prompt_registry,
)


class TestPromptMetrics:
    """Tests for PromptMetrics dataclass."""

    def test_defaults(self):
        m = PromptMetrics()
        assert m.uses == 0
        assert m.success_rate == 0.0
        assert m.avg_score == 0.0

    def test_success_rate(self):
        m = PromptMetrics(uses=10, successes=7, failures=3)
        assert m.success_rate == 0.7

    def test_avg_score(self):
        m = PromptMetrics(scores=[0.5, 0.7, 0.9])
        assert abs(m.avg_score - 0.7) < 0.01

    def test_to_dict(self):
        m = PromptMetrics(uses=5, successes=4, failures=1, scores=[0.8, 0.9])
        d = m.to_dict()
        assert d["uses"] == 5
        assert d["success_rate"] == 0.8
        assert abs(d["avg_score"] - 0.85) < 0.01


class TestPromptTemplate:
    """Tests for PromptTemplate dataclass."""

    def test_defaults(self):
        t = PromptTemplate(
            id="test-1",
            domain="pipeline",
            stage="goals",
            template="Extract goals from: {ideas}",
        )
        assert t.version == 1
        assert t.active is True
        assert t.created_at  # Auto-generated

    def test_auto_detect_variables(self):
        t = PromptTemplate(
            id="test-1",
            domain="test",
            stage="test",
            template="Hello {name}, your task is {task}",
        )
        assert "name" in t.variables
        assert "task" in t.variables

    def test_render(self):
        t = PromptTemplate(
            id="test-1",
            domain="test",
            stage="test",
            template="Goals: {goals}\nContext: {context}",
        )
        result = t.render(goals="improve UX", context="web app")
        assert "improve UX" in result
        assert "web app" in result

    def test_render_missing_variable(self):
        t = PromptTemplate(
            id="test-1",
            domain="test",
            stage="test",
            template="Goals: {goals}",
        )
        result = t.render()
        assert "{goals}" in result  # Left as placeholder

    def test_content_hash(self):
        t = PromptTemplate(
            id="test-1",
            domain="test",
            stage="test",
            template="Hello world",
        )
        assert len(t.content_hash) == 16

    def test_to_dict(self):
        t = PromptTemplate(
            id="test-1",
            domain="pipeline",
            stage="goals",
            template="test",
            version=2,
        )
        d = t.to_dict()
        assert d["id"] == "test-1"
        assert d["domain"] == "pipeline"
        assert d["version"] == 2


class TestPromptRegistry:
    """Tests for PromptRegistry."""

    @pytest.fixture
    def registry(self):
        return PromptRegistry()

    @pytest.fixture
    def sample_templates(self, registry):
        t1 = PromptTemplate(
            id="goals_v1",
            domain="pipeline",
            stage="goals",
            template="Extract goals: {ideas}",
            version=1,
        )
        t2 = PromptTemplate(
            id="goals_v2",
            domain="pipeline",
            stage="goals",
            template="Extract SMART goals: {ideas}",
            version=2,
        )
        registry.register(t1)
        registry.register(t2)
        return t1, t2

    def test_register_and_get(self, registry):
        t = PromptTemplate(
            id="test-1",
            domain="test",
            stage="test",
            template="Hello",
        )
        registry.register(t)
        assert registry.get("test-1") is t

    def test_get_nonexistent(self, registry):
        assert registry.get("nonexistent") is None

    def test_get_best_single_template(self, registry):
        t = PromptTemplate(
            id="test-1",
            domain="pipeline",
            stage="goals",
            template="Hello",
        )
        registry.register(t)
        assert registry.get_best("pipeline", "goals") is t

    def test_get_best_no_templates(self, registry):
        assert registry.get_best("nonexistent", "stage") is None

    def test_get_best_with_metrics(self, registry, sample_templates):
        t1, t2 = sample_templates

        # Give t1 many uses with good score
        for _ in range(10):
            registry.record_outcome("goals_v1", success=True, score=0.9)
        for _ in range(3):
            registry.record_outcome("goals_v2", success=True, score=0.7)

        best = registry.get_best("pipeline", "goals")
        assert best is t1  # Higher avg_score with enough data

    def test_get_best_explores_undertested(self, registry, sample_templates):
        t1, t2 = sample_templates

        # Give t1 few uses — not enough data
        registry.record_outcome("goals_v1", success=True, score=0.9)

        best = registry.get_best("pipeline", "goals")
        # Should explore least-used template since t1 has < 5 uses
        assert best.metrics.uses <= 1

    def test_record_outcome(self, registry):
        t = PromptTemplate(
            id="test-1",
            domain="test",
            stage="test",
            template="Hello",
        )
        registry.register(t)

        registry.record_outcome("test-1", success=True, score=0.8)
        registry.record_outcome("test-1", success=True, score=0.9)
        registry.record_outcome("test-1", success=False, score=0.3)

        assert t.metrics.uses == 3
        assert t.metrics.successes == 2
        assert t.metrics.failures == 1
        assert len(t.metrics.scores) == 3

    def test_record_outcome_unknown_template(self, registry):
        # Should not raise
        registry.record_outcome("nonexistent", success=True)

    def test_record_outcome_score_window(self, registry):
        t = PromptTemplate(
            id="test-1",
            domain="test",
            stage="test",
            template="Hello",
        )
        registry.register(t)

        # Record 150 outcomes — only last 100 should be kept
        for i in range(150):
            registry.record_outcome("test-1", success=True, score=float(i) / 150)

        assert len(t.metrics.scores) == 100

    def test_list_templates(self, registry, sample_templates):
        templates = registry.list_templates()
        assert len(templates) == 2

    def test_list_templates_by_domain(self, registry, sample_templates):
        # Add a template from different domain
        registry.register(
            PromptTemplate(
                id="debate-1",
                domain="debate",
                stage="review",
                template="Review",
            )
        )

        pipeline_templates = registry.list_templates(domain="pipeline")
        assert len(pipeline_templates) == 2

        debate_templates = registry.list_templates(domain="debate")
        assert len(debate_templates) == 1

    def test_list_active_only(self, registry, sample_templates):
        registry.deactivate("goals_v1")
        active = registry.list_templates(active_only=True)
        assert len(active) == 1
        assert active[0].id == "goals_v2"

    def test_deactivate(self, registry, sample_templates):
        assert registry.deactivate("goals_v1") is True
        assert registry.get("goals_v1").active is False

    def test_deactivate_nonexistent(self, registry):
        assert registry.deactivate("nonexistent") is False

    def test_get_for_ab_test(self, registry, sample_templates):
        result = registry.get_for_ab_test("pipeline", "goals")
        assert result is not None
        a, b = result
        assert a.id != b.id

    def test_get_for_ab_test_insufficient(self, registry):
        registry.register(
            PromptTemplate(
                id="solo",
                domain="test",
                stage="test",
                template="Hello",
            )
        )
        assert registry.get_for_ab_test("test", "test") is None

    def test_metrics_report(self, registry, sample_templates):
        registry.record_outcome("goals_v1", success=True, score=0.8)
        report = registry.get_metrics_report()
        assert report["total_templates"] == 2
        assert report["active_templates"] == 2
        assert "pipeline" in report["domains"]
        assert report["domains"]["pipeline"]["stages"]["goals"]["total_uses"] == 1


class TestGlobalRegistry:
    """Tests for the global registry singleton."""

    def test_get_prompt_registry_returns_instance(self):
        # Reset singleton for test
        import aragora.prompts.registry as mod

        mod._global_registry = None

        registry = get_prompt_registry()
        assert isinstance(registry, PromptRegistry)

        # Should return same instance
        assert get_prompt_registry() is registry

    def test_builtin_templates_registered(self):
        import aragora.prompts.registry as mod

        mod._global_registry = None

        registry = get_prompt_registry()
        templates = registry.list_templates()
        assert len(templates) >= 5  # At least 5 built-in templates

    def test_builtin_pipeline_templates(self):
        import aragora.prompts.registry as mod

        mod._global_registry = None

        registry = get_prompt_registry()

        # Check pipeline templates exist
        goals_template = registry.get("pipeline_ideas_to_goals_v1")
        assert goals_template is not None
        assert goals_template.domain == "pipeline"

        actions_template = registry.get("pipeline_goals_to_actions_v1")
        assert actions_template is not None

    def test_builtin_agent_templates(self):
        import aragora.prompts.registry as mod

        mod._global_registry = None

        registry = get_prompt_registry()

        proposer = registry.get("agent_proposer_v1")
        assert proposer is not None
        assert "proposal" in proposer.stage

        critic = registry.get("agent_critic_v1")
        assert critic is not None
