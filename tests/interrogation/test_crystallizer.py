"""Tests for the interrogation crystallizer module."""

import pytest

from aragora.interrogation.crystallizer import Crystallizer, Spec, Requirement, RequirementLevel
from aragora.interrogation.decomposer import Dimension, DecompositionResult
from aragora.interrogation.questioner import Question, QuestionSet
from aragora.interrogation.researcher import ResearchResult


class TestCrystallizer:
    @pytest.fixture
    def crystallizer(self):
        return Crystallizer()

    @pytest.fixture
    def answered_questions(self):
        return QuestionSet(
            questions=[
                Question(
                    text="What performance aspects?",
                    why="Clarification needed",
                    dimension_name="performance",
                    priority=0.8,
                    options=["Latency", "Throughput", "Both"],
                    context="",
                ),
            ]
        )

    @pytest.fixture
    def user_answers(self):
        return {"What performance aspects?": "Latency reduction for API calls"}

    @pytest.fixture
    def decomposition(self):
        return DecompositionResult(
            original_prompt="Make it faster",
            dimensions=[
                Dimension(
                    name="performance", description="Speed", vagueness_score=0.7, keywords=["fast"]
                )
            ],
            overall_vagueness=0.7,
        )

    def test_crystallize_returns_spec(
        self, crystallizer, decomposition, answered_questions, user_answers
    ):
        spec = crystallizer.crystallize(
            decomposition, answered_questions, user_answers, ResearchResult()
        )
        assert isinstance(spec, Spec)

    def test_spec_has_problem_statement(
        self, crystallizer, decomposition, answered_questions, user_answers
    ):
        spec = crystallizer.crystallize(
            decomposition, answered_questions, user_answers, ResearchResult()
        )
        assert spec.problem_statement

    def test_spec_has_requirements(
        self, crystallizer, decomposition, answered_questions, user_answers
    ):
        spec = crystallizer.crystallize(
            decomposition, answered_questions, user_answers, ResearchResult()
        )
        assert len(spec.requirements) >= 1
        assert all(isinstance(r, Requirement) for r in spec.requirements)

    def test_spec_has_success_criteria(
        self, crystallizer, decomposition, answered_questions, user_answers
    ):
        spec = crystallizer.crystallize(
            decomposition, answered_questions, user_answers, ResearchResult()
        )
        assert len(spec.success_criteria) >= 1

    def test_spec_includes_original_prompt(
        self, crystallizer, decomposition, answered_questions, user_answers
    ):
        spec = crystallizer.crystallize(
            decomposition, answered_questions, user_answers, ResearchResult()
        )
        assert decomposition.original_prompt in spec.problem_statement
