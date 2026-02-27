"""Tests for the interrogation crystallizer module."""

import pytest

from aragora.interrogation.crystallizer import (
    CrystallizedSpec,
    MoSCoWItem,
    Requirement,
    RequirementLevel,
    Spec,
)


class TestSpec:
    def test_spec_defaults(self):
        spec = Spec()
        assert spec.problem_statement == ""
        assert spec.requirements == []
        assert spec.success_criteria == []
        assert spec.non_requirements == []
        assert spec.risks == []
        assert spec.context_summary == ""

    def test_spec_with_requirements(self):
        spec = Spec(
            problem_statement="Make it faster",
            requirements=[
                Requirement(
                    description="Reduce latency",
                    level=RequirementLevel.MUST,
                    dimension="performance",
                ),
                Requirement(
                    description="Add caching",
                    level=RequirementLevel.SHOULD,
                    dimension="performance",
                ),
            ],
            success_criteria=["API response < 200ms"],
        )
        assert spec.problem_statement == "Make it faster"
        assert len(spec.requirements) == 2
        assert spec.requirements[0].level == RequirementLevel.MUST
        assert spec.requirements[1].level == RequirementLevel.SHOULD

    def test_to_goal_text(self):
        spec = Spec(
            problem_statement="Fix performance",
            requirements=[
                Requirement(
                    description="Reduce latency",
                    level=RequirementLevel.MUST,
                    dimension="performance",
                ),
            ],
        )
        goal = spec.to_goal_text()
        assert "Fix performance" in goal
        assert "Reduce latency" in goal
        assert "[MUST]" in goal

    def test_spec_has_problem_statement(self):
        spec = Spec(problem_statement="Improve UX")
        assert spec.problem_statement == "Improve UX"

    def test_spec_has_requirements(self):
        spec = Spec(
            problem_statement="Test",
            requirements=[
                Requirement(
                    description="Reduce latency",
                    level=RequirementLevel.MUST,
                    dimension="performance",
                ),
            ],
        )
        assert len(spec.requirements) >= 1
        assert all(isinstance(r, Requirement) for r in spec.requirements)

    def test_spec_has_success_criteria(self):
        spec = Spec(
            problem_statement="Test",
            success_criteria=["Response time < 200ms"],
        )
        assert len(spec.success_criteria) >= 1


class TestRequirementLevel:
    def test_levels_exist(self):
        assert RequirementLevel.MUST.value == "must"
        assert RequirementLevel.SHOULD.value == "should"
        assert RequirementLevel.COULD.value == "could"
        assert RequirementLevel.WONT.value == "wont"

    def test_requirement_fields(self):
        r = Requirement(
            description="Add tests",
            level=RequirementLevel.SHOULD,
            dimension="quality",
        )
        assert r.description == "Add tests"
        assert r.level == RequirementLevel.SHOULD
        assert r.dimension == "quality"


class TestMoSCoWItem:
    def test_moscow_item(self):
        item = MoSCoWItem(description="A", priority="must", rationale="R1")
        assert item.description == "A"
        assert item.priority == "must"


class TestCrystallizedSpec:
    def test_moscow_properties(self):
        spec = CrystallizedSpec(
            title="Test",
            problem_statement="Test problem",
            requirements=[
                MoSCoWItem(description="A", priority="must", rationale="R1"),
                MoSCoWItem(description="B", priority="should", rationale="R2"),
                MoSCoWItem(description="C", priority="could", rationale="R3"),
                MoSCoWItem(description="D", priority="wont", rationale="R4"),
            ],
        )
        assert len(spec.musts) == 1
        assert len(spec.shoulds) == 1
        assert len(spec.coulds) == 1
        assert len(spec.wonts) == 1

    def test_to_dict(self):
        spec = CrystallizedSpec(
            title="Test",
            problem_statement="Test problem",
            requirements=[MoSCoWItem(description="A", priority="must", rationale="R")],
            success_criteria=["Tests pass"],
        )
        d = spec.to_dict()
        assert d["title"] == "Test"
        assert len(d["requirements"]) == 1
        assert d["requirements"][0]["priority"] == "must"
