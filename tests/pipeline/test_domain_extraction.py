"""Tests for domain-specific goal extraction.

Tests the GoalExtractor domain parameter that injects vertical-specific
goals for healthcare, financial, and legal domains.
"""

import pytest

from aragora.goals.extractor import (
    GoalExtractor,
    GoalGraph,
    GoalNode,
    _get_domain_goals,
    _DOMAIN_GOALS,
)
from aragora.canvas.stages import GoalNodeType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_ideas():
    return [
        "Build a customer portal with self-service features",
        "Add automated reporting for monthly metrics",
        "Improve data security and access controls",
    ]


@pytest.fixture
def sample_canvas_data():
    return {
        "nodes": [
            {"id": "n1", "label": "Build a customer portal", "data": {"idea_type": "concept"}},
            {"id": "n2", "label": "Add automated reporting", "data": {"idea_type": "concept"}},
            {"id": "n3", "label": "Improve data security", "data": {"idea_type": "concept"}},
        ],
        "edges": [],
    }


# =============================================================================
# Domain goal template tests
# =============================================================================


class TestDomainGoalTemplates:
    """Test the domain goal template definitions."""

    def test_healthcare_domain_has_goals(self):
        goals = _get_domain_goals("healthcare")
        assert len(goals) == 3

    def test_financial_domain_has_goals(self):
        goals = _get_domain_goals("financial")
        assert len(goals) == 3

    def test_legal_domain_has_goals(self):
        goals = _get_domain_goals("legal")
        assert len(goals) == 3

    def test_unknown_domain_returns_empty(self):
        goals = _get_domain_goals("unknown_domain")
        assert goals == []

    def test_healthcare_goals_have_required_fields(self):
        for goal in _get_domain_goals("healthcare"):
            assert "title" in goal
            assert "description" in goal
            assert "type" in goal
            assert "priority" in goal

    def test_financial_goals_have_required_fields(self):
        for goal in _get_domain_goals("financial"):
            assert "title" in goal
            assert "description" in goal

    def test_legal_goals_have_required_fields(self):
        for goal in _get_domain_goals("legal"):
            assert "title" in goal
            assert "description" in goal

    def test_healthcare_includes_hipaa_goal(self):
        goals = _get_domain_goals("healthcare")
        titles = [g["title"] for g in goals]
        assert any("HIPAA" in t for t in titles)

    def test_healthcare_includes_patient_safety_goal(self):
        goals = _get_domain_goals("healthcare")
        titles = [g["title"] for g in goals]
        assert any("patient safety" in t.lower() for t in titles)

    def test_financial_includes_risk_assessment(self):
        goals = _get_domain_goals("financial")
        titles = [g["title"] for g in goals]
        assert any("risk" in t.lower() for t in titles)

    def test_financial_includes_regulatory_compliance(self):
        goals = _get_domain_goals("financial")
        titles = [g["title"] for g in goals]
        assert any("regulatory" in t.lower() or "compliance" in t.lower() for t in titles)

    def test_legal_includes_contract_review(self):
        goals = _get_domain_goals("legal")
        titles = [g["title"] for g in goals]
        assert any("contract" in t.lower() for t in titles)

    def test_legal_includes_due_diligence(self):
        goals = _get_domain_goals("legal")
        titles = [g["title"] for g in goals]
        assert any("due diligence" in t.lower() for t in titles)


# =============================================================================
# GoalExtractor domain parameter tests
# =============================================================================


class TestGoalExtractorWithDomain:
    """Test GoalExtractor with domain parameter."""

    def test_extractor_without_domain_has_no_domain_goals(self, sample_canvas_data):
        extractor = GoalExtractor()
        result = extractor.extract_from_ideas(sample_canvas_data)
        domain_goals = [g for g in result.goals if g.metadata.get("domain_injected")]
        assert len(domain_goals) == 0

    def test_extractor_with_healthcare_domain(self, sample_canvas_data):
        extractor = GoalExtractor(domain="healthcare")
        result = extractor.extract_from_ideas(sample_canvas_data)
        domain_goals = [g for g in result.goals if g.metadata.get("domain_injected")]
        assert len(domain_goals) == 3
        assert all(g.metadata.get("domain") == "healthcare" for g in domain_goals)

    def test_extractor_with_financial_domain(self, sample_canvas_data):
        extractor = GoalExtractor(domain="financial")
        result = extractor.extract_from_ideas(sample_canvas_data)
        domain_goals = [g for g in result.goals if g.metadata.get("domain_injected")]
        assert len(domain_goals) == 3
        assert all(g.metadata.get("domain") == "financial" for g in domain_goals)

    def test_extractor_with_legal_domain(self, sample_canvas_data):
        extractor = GoalExtractor(domain="legal")
        result = extractor.extract_from_ideas(sample_canvas_data)
        domain_goals = [g for g in result.goals if g.metadata.get("domain_injected")]
        assert len(domain_goals) == 3
        assert all(g.metadata.get("domain") == "legal" for g in domain_goals)

    def test_extractor_with_unknown_domain_adds_no_extras(self, sample_canvas_data):
        extractor = GoalExtractor(domain="unknown")
        result = extractor.extract_from_ideas(sample_canvas_data)
        domain_goals = [g for g in result.goals if g.metadata.get("domain_injected")]
        assert len(domain_goals) == 0

    def test_domain_goals_have_high_confidence(self, sample_canvas_data):
        extractor = GoalExtractor(domain="healthcare")
        result = extractor.extract_from_ideas(sample_canvas_data)
        domain_goals = [g for g in result.goals if g.metadata.get("domain_injected")]
        for g in domain_goals:
            assert g.confidence == 0.9

    def test_domain_goals_have_correct_types(self, sample_canvas_data):
        extractor = GoalExtractor(domain="healthcare")
        result = extractor.extract_from_ideas(sample_canvas_data)
        domain_goals = [g for g in result.goals if g.metadata.get("domain_injected")]
        types = {g.goal_type for g in domain_goals}
        # Healthcare should have goal, risk, and principle types
        assert GoalNodeType.GOAL in types
        assert GoalNodeType.RISK in types
        assert GoalNodeType.PRINCIPLE in types

    def test_domain_goals_from_raw_ideas(self, sample_ideas):
        extractor = GoalExtractor(domain="financial")
        result = extractor.extract_from_raw_ideas(sample_ideas)
        domain_goals = [g for g in result.goals if g.metadata.get("domain_injected")]
        assert len(domain_goals) == 3

    def test_domain_goals_combined_with_structural(self, sample_canvas_data):
        extractor = GoalExtractor(domain="legal")
        result = extractor.extract_from_ideas(sample_canvas_data)
        structural_goals = [g for g in result.goals if not g.metadata.get("domain_injected")]
        domain_goals = [g for g in result.goals if g.metadata.get("domain_injected")]
        # Should have both structural and domain goals
        assert len(structural_goals) > 0
        assert len(domain_goals) == 3
        assert len(result.goals) == len(structural_goals) + len(domain_goals)

    def test_domain_goal_ids_are_unique(self, sample_canvas_data):
        extractor = GoalExtractor(domain="healthcare")
        result = extractor.extract_from_ideas(sample_canvas_data)
        ids = [g.id for g in result.goals]
        assert len(ids) == len(set(ids)), "Goal IDs should be unique"

    def test_domain_goal_priorities(self, sample_canvas_data):
        extractor = GoalExtractor(domain="financial")
        result = extractor.extract_from_ideas(sample_canvas_data)
        domain_goals = [g for g in result.goals if g.metadata.get("domain_injected")]
        priorities = {g.priority for g in domain_goals}
        # Financial goals should be critical or high priority
        assert all(p in ("critical", "high") for p in priorities)
