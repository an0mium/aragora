"""Tests for the Interrogation Engine."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aragora.interrogation.crystallizer import Crystallizer, CrystallizedSpec, MoSCoWItem
from aragora.interrogation.engine import (
    InterrogationConfig,
    InterrogationEngine,
    PrioritizedQuestion,
)


# ── Helpers ───────────────────────────────────────────────────────


def mock_agent(responses: list[str]) -> AsyncMock:
    agent = AsyncMock()
    agent.generate = AsyncMock(side_effect=responses)
    return agent


# ── Crystallizer Tests ────────────────────────────────────────────


class TestCrystallizer:
    @pytest.mark.asyncio
    async def test_crystallize_basic(self):
        agent = mock_agent(
            [
                "TITLE: Improve Onboarding Flow\n\n"
                "PROBLEM_STATEMENT: Users drop off during onboarding.\n\n"
                "MUST:\n"
                "- Reduce steps to 3: Users complete faster\n"
                "- Add progress bar: Shows completion percentage\n\n"
                "SHOULD:\n"
                "- Add tooltips: Guide first-time users\n\n"
                "COULD:\n"
                "- Add video walkthrough: Visual learners benefit\n\n"
                "WONT:\n"
                "- Mobile onboarding: Desktop only for V1\n\n"
                "NON_REQUIREMENTS:\n"
                "- Multi-language support\n"
                "- A/B testing framework\n\n"
                "SUCCESS_CRITERIA:\n"
                "- Onboarding completion rate > 80%\n"
                "- Time to first action < 5 minutes\n\n"
                "RISKS:\n"
                "- Oversimplification: May lose power users\n\n"
                "IMPLICATIONS:\n"
                "- Need to audit existing user flow analytics\n\n"
                "CONSTRAINTS:\n"
                "- Must work with existing auth system\n\n"
                "PRIOR_ART:\n"
                "- Slack's progressive onboarding pattern\n"
            ]
        )
        crystallizer = Crystallizer(agent)
        spec = await crystallizer.crystallize(
            prompt="Improve onboarding",
            research_context="Current completion rate is 45%",
        )
        assert spec.title == "Improve Onboarding Flow"
        assert "drop off" in spec.problem_statement
        assert len(spec.musts) == 2
        assert len(spec.shoulds) == 1
        assert len(spec.coulds) == 1
        assert len(spec.wonts) == 1
        assert len(spec.non_requirements) == 2
        assert len(spec.success_criteria) == 2
        assert len(spec.risks) == 1
        assert len(spec.implications) == 1
        assert len(spec.constraints) == 1
        assert len(spec.prior_art) == 1

    def test_moscow_item_properties(self):
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


# ── Dimension Parsing Tests ───────────────────────────────────────


class TestDimensionParsing:
    def test_parse_dimensions(self):
        engine = InterrogationEngine(agents=[mock_agent([])])
        text = (
            "DIMENSION: Scope\n"
            "DESCRIPTION: What features are included\n"
            "CATEGORY: scope\n\n"
            "DIMENSION: Timeline\n"
            "DESCRIPTION: When should it ship\n"
            "CATEGORY: business\n"
        )
        result = engine._parse_dimensions(text)
        assert len(result) == 2
        assert "Scope" in result[0]
        assert "Timeline" in result[1]


# ── Question Parsing Tests ────────────────────────────────────────


class TestQuestionParsing:
    def test_parse_questions(self):
        engine = InterrogationEngine(agents=[mock_agent([])])
        text = (
            "QUESTION: Should we support mobile?\n"
            "WHY: Changes the entire frontend architecture\n"
            "ASSUMPTION: Desktop-only is acceptable\n"
            "OPTIONS: Desktop only, Mobile responsive, Native app\n"
            "CATEGORY: technical\n\n"
            "QUESTION: What's the target audience?\n"
            "WHY: Affects UX complexity\n"
            "ASSUMPTION: Technical users\n"
            "OPTIONS: Developers, Product managers, Executives\n"
            "CATEGORY: business\n"
        )
        result = engine._parse_questions(text)
        assert len(result) == 2
        assert "mobile" in result[0].question.lower()
        assert result[0].category == "technical"
        assert len(result[0].options) == 3
        assert result[1].hidden_assumption == "Technical users"

    def test_parse_empty_returns_empty(self):
        engine = InterrogationEngine(agents=[mock_agent([])])
        result = engine._parse_questions("No structured output here")
        assert result == []


# ── Ranking Parsing Tests ─────────────────────────────────────────


class TestRankingParsing:
    def test_parse_ranking(self):
        engine = InterrogationEngine(agents=[mock_agent([])])
        text = (
            "RANK 1: 3 - Most impactful question\n"
            "RANK 2: 1 - Second most important\n"
            "RANK 3: 2 - Least important\n"
        )
        result = engine._parse_ranking(text, 3)
        assert result[2] == 1  # Question 3 ranked 1st
        assert result[0] == 2  # Question 1 ranked 2nd
        assert result[1] == 3  # Question 2 ranked 3rd

    def test_parse_ranking_with_missing(self):
        engine = InterrogationEngine(agents=[mock_agent([])])
        text = "RANK 1: 2 - Only one ranked\n"
        result = engine._parse_ranking(text, 3)
        assert result[1] == 1  # Question 2 ranked 1st
        assert result[0] == 3  # Question 1 unranked → default
        assert result[2] == 3  # Question 3 unranked → default


# ── Full Pipeline Tests ───────────────────────────────────────────


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_no_agents_returns_empty(self):
        engine = InterrogationEngine(agents=[])
        result = await engine.interrogate("Make things better")
        assert result.prioritized_questions == []
        assert "error" in result.metadata

    @pytest.mark.asyncio
    async def test_single_agent_no_debate(self):
        """Single agent: generates questions without debate prioritization."""
        agent = mock_agent(
            [
                # Decompose response
                "DIMENSION: Scope\nDESCRIPTION: What to improve\nCATEGORY: scope\n",
                # Research response
                "SUMMARY: Current system has known performance issues.",
                # Question generation
                (
                    "QUESTION: Which component is slowest?\n"
                    "WHY: Focuses optimization effort\n"
                    "ASSUMPTION: All components equally slow\n"
                    "OPTIONS: API, Database, Frontend\n"
                    "CATEGORY: technical\n"
                ),
            ]
        )
        engine = InterrogationEngine(
            agents=[agent],
            config=InterrogationConfig(skip_debate=True),
        )
        result = await engine.interrogate("Improve performance")
        assert len(result.dimensions) >= 1
        assert len(result.prioritized_questions) >= 1
        assert result.research_summary != ""

    @pytest.mark.asyncio
    async def test_multi_agent_debate(self):
        """Two agents debate question priorities."""
        agent1 = mock_agent(
            [
                # Decompose
                "DIMENSION: Architecture\nDESCRIPTION: How to structure\nCATEGORY: technical\n",
                # Research
                "SUMMARY: System uses monolithic architecture.",
                # Question generation
                (
                    "QUESTION: Microservices or monolith?\n"
                    "WHY: Fundamental architecture choice\n"
                    "ASSUMPTION: Current monolith works\n"
                    "OPTIONS: Keep monolith, Microservices, Modular monolith\n"
                    "CATEGORY: technical\n\n"
                    "QUESTION: What's the team size?\n"
                    "WHY: Affects architecture choice\n"
                    "ASSUMPTION: Small team\n"
                    "OPTIONS: 1-3, 4-10, 10+\n"
                    "CATEGORY: business\n"
                ),
                # Agent 1's ranking
                "RANK 1: 1 - Architecture is fundamental\nRANK 2: 2 - Team size matters less\n"
                "REASONING: Architecture constrains everything downstream.",
            ]
        )
        agent2 = mock_agent(
            [
                # Agent 2's ranking
                "RANK 1: 2 - Team determines what's feasible\nRANK 2: 1 - Architecture follows team\n"
                "REASONING: Team capability determines architecture viability.",
            ]
        )
        engine = InterrogationEngine(
            agents=[agent1, agent2],
            config=InterrogationConfig(debate_rounds=1),
        )
        result = await engine.interrogate("Scale the system")
        assert len(result.prioritized_questions) == 2
        assert result.debate_reasoning != ""
        # Both questions should have priority scores
        assert all(q.priority_score > 0 for q in result.prioritized_questions)

    @pytest.mark.asyncio
    async def test_with_crystallization(self):
        """Full pipeline with crystallization after user answers."""
        main_agent = mock_agent(
            [
                # Decompose
                "DIMENSION: Feature\nDESCRIPTION: What to build\nCATEGORY: scope\n",
                # Research
                "SUMMARY: No existing implementation.",
                # Questions
                "QUESTION: What's the priority?\nWHY: Determines scope\n"
                "ASSUMPTION: Everything is equal\n"
                "OPTIONS: Speed, Quality\nCATEGORY: business\n",
            ]
        )
        crystal_agent = mock_agent(
            [
                "TITLE: Speed-First Feature\n\n"
                "PROBLEM_STATEMENT: Need fast delivery.\n\n"
                "MUST:\n- Ship in 1 week: Time-constrained\n\n"
                "SHOULD:\n- Include tests: Quality baseline\n\n"
                "COULD:\n\n"
                "WONT:\n\n"
                "NON_REQUIREMENTS:\n\n"
                "SUCCESS_CRITERIA:\n- Deployed within 7 days\n\n"
                "RISKS:\n- Technical debt: May need refactor\n\n"
                "IMPLICATIONS:\n\n"
                "CONSTRAINTS:\n\n"
                "PRIOR_ART:\n"
            ]
        )

        # Simulate user answering questions
        async def answer_questions(
            questions: list[PrioritizedQuestion],
        ) -> list[PrioritizedQuestion]:
            for q in questions:
                q.answer = "Speed"
            return questions

        engine = InterrogationEngine(
            agents=[main_agent],
            crystallizer_agent=crystal_agent,
            config=InterrogationConfig(skip_debate=True),
            on_questions=answer_questions,
        )
        result = await engine.interrogate("Build a new feature")
        assert result.crystallized_spec is not None
        assert result.crystallized_spec.title == "Speed-First Feature"
        assert len(result.crystallized_spec.musts) == 1

    @pytest.mark.asyncio
    async def test_min_priority_filter(self):
        """Questions below min_priority threshold are filtered out."""
        agent = mock_agent(
            [
                "DIMENSION: A\nDESCRIPTION: X\nCATEGORY: scope\n",
                "SUMMARY: context",
                (
                    "QUESTION: Important question?\n"
                    "WHY: Matters a lot\nASSUMPTION: None\n"
                    "OPTIONS: Yes, No\nCATEGORY: scope\n\n"
                    "QUESTION: Trivial question?\n"
                    "WHY: Not very important\nASSUMPTION: None\n"
                    "OPTIONS: A, B\nCATEGORY: scope\n"
                ),
            ]
        )
        engine = InterrogationEngine(
            agents=[agent],
            config=InterrogationConfig(skip_debate=True, min_priority=0.6),
        )
        result = await engine.interrogate("Do something")
        # Default priority is 0.5, so all questions below 0.6 are filtered
        assert len(result.prioritized_questions) == 0
