"""Tests for the prompt optimizer module."""

from __future__ import annotations

import pytest

from aragora.prompts.optimizer import (
    OptimizationCandidate,
    OptimizationResult,
    PromptOptimizer,
    _variance,
)
from aragora.prompts.registry import PromptMetrics, PromptRegistry, PromptTemplate


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_defaults(self):
        r = OptimizationResult(domain="pipeline", stage="goals")
        assert r.candidates_generated == 0
        assert r.candidates_promoted == 0
        assert r.candidates_retired == 0
        assert r.timestamp  # Auto-generated

    def test_to_dict(self):
        r = OptimizationResult(
            domain="pipeline",
            stage="goals",
            candidates_generated=3,
            best_score=0.85,
        )
        d = r.to_dict()
        assert d["domain"] == "pipeline"
        assert d["best_score"] == 0.85


class TestVariance:
    """Tests for _variance helper."""

    def test_empty(self):
        assert _variance([]) == 0.0

    def test_single(self):
        assert _variance([0.5]) == 0.0

    def test_uniform(self):
        assert _variance([0.5, 0.5, 0.5]) == 0.0

    def test_varied(self):
        v = _variance([0.0, 1.0])
        assert abs(v - 0.25) < 0.01  # mean=0.5, var=(0.25+0.25)/2=0.25


class TestIdentifyUnderperformers:
    """Tests for identifying underperforming templates."""

    @pytest.fixture
    def optimizer(self):
        registry = PromptRegistry()
        return PromptOptimizer(registry, min_uses=5, score_threshold=0.7)

    def test_no_templates(self, optimizer):
        result = optimizer.identify_underperformers()
        assert result == []

    def test_not_enough_uses(self, optimizer):
        t = PromptTemplate(id="test-1", domain="test", stage="test", template="Hello")
        t.metrics = PromptMetrics(uses=3, scores=[0.3, 0.4, 0.5])
        optimizer.registry.register(t)

        result = optimizer.identify_underperformers()
        assert result == []  # Only 3 uses, need 5

    def test_good_performance(self, optimizer):
        t = PromptTemplate(id="test-1", domain="test", stage="test", template="Hello")
        t.metrics = PromptMetrics(uses=10, successes=9, scores=[0.8, 0.9, 0.85, 0.88, 0.92])
        optimizer.registry.register(t)

        result = optimizer.identify_underperformers()
        assert result == []  # avg_score > 0.7

    def test_finds_underperformer(self, optimizer):
        good = PromptTemplate(id="good", domain="test", stage="test", template="Good prompt")
        good.metrics = PromptMetrics(uses=10, scores=[0.8, 0.9, 0.85])
        optimizer.registry.register(good)

        bad = PromptTemplate(id="bad", domain="test", stage="test", template="Bad prompt")
        bad.metrics = PromptMetrics(uses=10, scores=[0.3, 0.4, 0.5])
        optimizer.registry.register(bad)

        result = optimizer.identify_underperformers()
        assert len(result) == 1
        assert result[0].id == "bad"

    def test_filter_by_domain(self, optimizer):
        t1 = PromptTemplate(id="pipe-1", domain="pipeline", stage="goals", template="Hello")
        t1.metrics = PromptMetrics(uses=10, scores=[0.3, 0.4])
        optimizer.registry.register(t1)

        t2 = PromptTemplate(id="debate-1", domain="debate", stage="review", template="Hello")
        t2.metrics = PromptMetrics(uses=10, scores=[0.3, 0.4])
        optimizer.registry.register(t2)

        result = optimizer.identify_underperformers(domain="pipeline")
        assert len(result) == 1
        assert result[0].id == "pipe-1"


class TestGenerateVariant:
    """Tests for generating template variants."""

    @pytest.fixture
    def registry(self):
        reg = PromptRegistry()
        t = PromptTemplate(
            id="test-v1",
            domain="test",
            stage="goals",
            template="Extract goals from: {ideas}\nMake them SMART.\nPrioritize by impact.",
            version=1,
        )
        t.metrics = PromptMetrics(uses=10, scores=[0.4, 0.5, 0.3])
        reg.register(t)
        return reg

    @pytest.fixture
    def optimizer(self, registry):
        return PromptOptimizer(registry, min_uses=5, score_threshold=0.7)

    def test_generates_variant(self, optimizer):
        candidate = optimizer.generate_variant("test-v1")
        assert candidate is not None
        assert candidate.source_id == "test-v1"
        assert candidate.template.domain == "test"
        assert candidate.template.stage == "goals"
        assert candidate.template.version == 2  # Incremented

    def test_variant_has_different_template(self, optimizer):
        candidate = optimizer.generate_variant("test-v1", mutation_type="restructure")
        assert candidate is not None
        # Restructured should have numbered steps
        assert candidate.mutation_type == "restructure"

    def test_unknown_template(self, optimizer):
        result = optimizer.generate_variant("nonexistent")
        assert result is None

    def test_mutation_type_respected(self, optimizer):
        for mt in ["restructure", "simplify", "add_examples", "add_constraints"]:
            candidate = optimizer.generate_variant("test-v1", mutation_type=mt)
            assert candidate is not None
            assert candidate.mutation_type == mt


class TestMutationStrategies:
    """Tests for individual mutation strategies."""

    @pytest.fixture
    def optimizer(self):
        return PromptOptimizer(PromptRegistry())

    def test_restructure_adds_numbers(self, optimizer):
        template = "Do this.\n- Step one\n- Step two\n- Step three"
        result = optimizer._restructure(template)
        assert "1." in result
        assert "2." in result

    def test_simplify_deduplicates(self, optimizer):
        template = "Extract goals.\n\n\nExtract goals.\nDo stuff."
        result = optimizer._simplify(template)
        assert result.count("Extract goals") == 1

    def test_add_examples_appends(self, optimizer):
        template = "Extract goals."
        result = optimizer._add_examples(template, "goals")
        assert "Example output format" in result
        assert "goals" in result

    def test_add_constraints_appends(self, optimizer):
        template = "Extract goals."
        result = optimizer._add_constraints(template)
        assert "Quality constraints" in result
        assert "Do NOT" in result


class TestMutationSelection:
    """Tests for automatic mutation type selection."""

    @pytest.fixture
    def optimizer(self):
        return PromptOptimizer(PromptRegistry(), min_uses=5)

    def test_low_score_selects_restructure(self, optimizer):
        t = PromptTemplate(id="test", domain="test", stage="test", template="x" * 300)
        t.metrics = PromptMetrics(uses=10, scores=[0.2, 0.3, 0.4])
        mt = optimizer._select_mutation_type(t)
        assert mt == "restructure"

    def test_high_variance_selects_constraints(self, optimizer):
        t = PromptTemplate(id="test", domain="test", stage="test", template="x" * 300)
        t.metrics = PromptMetrics(uses=10, scores=[0.1, 0.9, 0.2, 0.8])
        mt = optimizer._select_mutation_type(t)
        assert mt == "add_constraints"

    def test_short_template_selects_examples(self, optimizer):
        t = PromptTemplate(id="test", domain="test", stage="test", template="Short prompt")
        t.metrics = PromptMetrics(uses=10, scores=[0.6, 0.65, 0.7])
        mt = optimizer._select_mutation_type(t)
        assert mt == "add_examples"

    def test_default_simplify(self, optimizer):
        t = PromptTemplate(id="test", domain="test", stage="test", template="x" * 300)
        t.metrics = PromptMetrics(uses=10, scores=[0.6, 0.65, 0.6])
        mt = optimizer._select_mutation_type(t)
        assert mt == "simplify"


class TestPromoteAndRetire:
    """Tests for promoting candidates and retiring underperformers."""

    @pytest.fixture
    def registry(self):
        reg = PromptRegistry()
        for i in range(4):
            t = PromptTemplate(
                id=f"test-v{i + 1}",
                domain="test",
                stage="goals",
                template=f"Template version {i + 1}",
                version=i + 1,
            )
            t.metrics = PromptMetrics(
                uses=20,
                scores=[0.3 + i * 0.15] * 5,  # v1=0.3, v2=0.45, v3=0.6, v4=0.75
            )
            reg.register(t)
        return reg

    @pytest.fixture
    def optimizer(self, registry):
        return PromptOptimizer(registry, min_uses=5, score_threshold=0.7)

    def test_promote_registers_template(self, optimizer):
        candidate = OptimizationCandidate(
            template=PromptTemplate(
                id="new-variant",
                domain="test",
                stage="goals",
                template="New variant",
            ),
            source_id="test-v1",
            mutation_type="restructure",
        )
        result_id = optimizer.promote_candidate(candidate)
        assert result_id == "new-variant"
        assert optimizer.registry.get("new-variant") is not None

    def test_retire_keeps_best(self, optimizer):
        retired = optimizer.retire_underperformers("test", "goals", keep_best=2)
        assert len(retired) == 2
        # Best two (v4=0.75, v3=0.6) should survive
        assert optimizer.registry.get("test-v4").active is True
        assert optimizer.registry.get("test-v3").active is True
        assert "test-v1" in retired
        assert "test-v2" in retired

    def test_retire_no_action_when_few_templates(self, optimizer):
        retired = optimizer.retire_underperformers("test", "goals", keep_best=10)
        assert retired == []


class TestRunCycle:
    """Tests for full optimization cycles."""

    @pytest.fixture
    def registry(self):
        reg = PromptRegistry()
        # Add a template that underperforms
        t = PromptTemplate(
            id="pipeline_goals_v1",
            domain="pipeline",
            stage="goals",
            template="Extract goals from: {ideas}\nMake them actionable.",
            version=1,
        )
        t.metrics = PromptMetrics(uses=15, successes=5, scores=[0.4, 0.5, 0.3, 0.45, 0.35])
        reg.register(t)
        return reg

    @pytest.fixture
    def optimizer(self, registry):
        return PromptOptimizer(registry, min_uses=5, score_threshold=0.7)

    def test_cycle_generates_candidates(self, optimizer):
        result = optimizer.run_cycle("pipeline", "goals")
        assert result.domain == "pipeline"
        assert result.stage == "goals"
        assert result.candidates_generated >= 1
        assert result.timestamp

    def test_cycle_tracks_history(self, optimizer):
        optimizer.run_cycle("pipeline", "goals")
        history = optimizer.get_optimization_history()
        assert len(history) == 1
        assert history[0]["domain"] == "pipeline"

    def test_multiple_cycles(self, optimizer):
        optimizer.run_cycle("pipeline", "goals")
        optimizer.run_cycle("pipeline", "goals")
        assert len(optimizer.get_optimization_history()) == 2


class TestFeedbackPrompts:
    """Tests for user feedback prompt generation."""

    @pytest.fixture
    def optimizer(self):
        reg = PromptRegistry()
        return PromptOptimizer(reg, min_uses=5, score_threshold=0.7)

    def test_no_templates_no_prompts(self, optimizer):
        prompts = optimizer.suggest_feedback_prompts("test", "goals")
        assert prompts == []

    def test_new_template_asks_for_rating(self, optimizer):
        t = PromptTemplate(id="test-1", domain="test", stage="goals", template="Hello")
        t.metrics = PromptMetrics(uses=2, scores=[0.5])
        optimizer.registry.register(t)

        prompts = optimizer.suggest_feedback_prompts("test", "goals")
        assert any("rate" in p.lower() or "quality" in p.lower() for p in prompts)

    def test_underperformer_asks_whats_missing(self, optimizer):
        t = PromptTemplate(id="test-1", domain="test", stage="goals", template="Hello")
        t.metrics = PromptMetrics(uses=10, scores=[0.3, 0.4, 0.5])
        optimizer.registry.register(t)

        prompts = optimizer.suggest_feedback_prompts("test", "goals")
        assert any("missing" in p.lower() for p in prompts)
