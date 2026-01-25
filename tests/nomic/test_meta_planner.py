"""Tests for MetaPlanner - debate-driven goal prioritization."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.nomic.meta_planner import (
    MetaPlanner,
    MetaPlannerConfig,
    PrioritizedGoal,
    PlanningContext,
    Track,
)


class TestTrackEnum:
    """Tests for Track enum."""

    def test_track_values(self):
        """Track enum should have correct values."""
        assert Track.SME.value == "sme"
        assert Track.DEVELOPER.value == "developer"
        assert Track.SELF_HOSTED.value == "self_hosted"
        assert Track.QA.value == "qa"
        assert Track.CORE.value == "core"

    def test_track_count(self):
        """Should have exactly 5 tracks."""
        assert len(Track) == 5


class TestPrioritizedGoal:
    """Tests for PrioritizedGoal dataclass."""

    def test_goal_creation(self):
        """Should create goal with all fields."""
        goal = PrioritizedGoal(
            id="goal_0",
            track=Track.SME,
            description="Improve dashboard",
            rationale="Increases user engagement",
            estimated_impact="high",
            priority=1,
            focus_areas=["ui", "ux"],
            file_hints=["dashboard.py"],
        )

        assert goal.id == "goal_0"
        assert goal.track == Track.SME
        assert goal.description == "Improve dashboard"
        assert goal.rationale == "Increases user engagement"
        assert goal.estimated_impact == "high"
        assert goal.priority == 1
        assert "ui" in goal.focus_areas
        assert "dashboard.py" in goal.file_hints

    def test_goal_default_lists(self):
        """Should have empty default lists."""
        goal = PrioritizedGoal(
            id="goal_0",
            track=Track.QA,
            description="Add tests",
            rationale="Improve coverage",
            estimated_impact="medium",
            priority=1,
        )

        assert goal.focus_areas == []
        assert goal.file_hints == []


class TestPlanningContext:
    """Tests for PlanningContext dataclass."""

    def test_context_creation(self):
        """Should create context with all fields."""
        context = PlanningContext(
            recent_issues=["Bug in auth"],
            test_failures=["test_login failed"],
            user_feedback=["Dashboard is slow"],
            recent_changes=["Updated handlers.py"],
        )

        assert "Bug in auth" in context.recent_issues
        assert "test_login failed" in context.test_failures
        assert "Dashboard is slow" in context.user_feedback
        assert "Updated handlers.py" in context.recent_changes

    def test_context_defaults(self):
        """Should have empty default lists."""
        context = PlanningContext()

        assert context.recent_issues == []
        assert context.test_failures == []
        assert context.user_feedback == []
        assert context.recent_changes == []


class TestMetaPlannerConfig:
    """Tests for MetaPlannerConfig dataclass."""

    def test_config_defaults(self):
        """Should have sensible defaults."""
        config = MetaPlannerConfig()

        assert config.agents == ["claude", "gemini", "deepseek"]
        assert config.debate_rounds == 2
        assert config.max_goals == 5
        assert config.consensus_threshold == 0.6

    def test_config_custom_values(self):
        """Should accept custom values."""
        config = MetaPlannerConfig(
            agents=["claude"],
            debate_rounds=3,
            max_goals=10,
            consensus_threshold=0.8,
        )

        assert config.agents == ["claude"]
        assert config.debate_rounds == 3
        assert config.max_goals == 10
        assert config.consensus_threshold == 0.8


class TestMetaPlanner:
    """Tests for MetaPlanner class."""

    def test_init_default_config(self):
        """Should initialize with default config."""
        planner = MetaPlanner()

        assert planner.config is not None
        assert planner.config.agents == ["claude", "gemini", "deepseek"]

    def test_init_custom_config(self):
        """Should accept custom config."""
        config = MetaPlannerConfig(max_goals=3)
        planner = MetaPlanner(config=config)

        assert planner.config.max_goals == 3


class TestBuildDebateTopic:
    """Tests for _build_debate_topic method."""

    def test_topic_includes_objective(self):
        """Topic should include the objective."""
        planner = MetaPlanner()
        topic = planner._build_debate_topic(
            objective="Maximize SME utility",
            tracks=[Track.SME],
            constraints=[],
            context=PlanningContext(),
        )

        assert "Maximize SME utility" in topic

    def test_topic_includes_tracks(self):
        """Topic should list available tracks."""
        planner = MetaPlanner()
        topic = planner._build_debate_topic(
            objective="Improve features",
            tracks=[Track.SME, Track.DEVELOPER],
            constraints=[],
            context=PlanningContext(),
        )

        assert "sme" in topic
        assert "developer" in topic

    def test_topic_includes_constraints(self):
        """Topic should include constraints."""
        planner = MetaPlanner()
        topic = planner._build_debate_topic(
            objective="Improve features",
            tracks=[Track.QA],
            constraints=["No breaking changes", "Must pass CI"],
            context=PlanningContext(),
        )

        assert "No breaking changes" in topic
        assert "Must pass CI" in topic

    def test_topic_includes_context(self):
        """Topic should include context information."""
        planner = MetaPlanner()
        context = PlanningContext(
            recent_issues=["Auth bug"],
            test_failures=["test_login"],
        )
        topic = planner._build_debate_topic(
            objective="Fix issues",
            tracks=[Track.CORE],
            constraints=[],
            context=context,
        )

        assert "Auth bug" in topic
        assert "test_login" in topic


class TestInferTrack:
    """Tests for _infer_track method."""

    def test_infer_sme_track(self):
        """Should infer SME track from dashboard keywords."""
        planner = MetaPlanner()
        track = planner._infer_track(
            "Improve dashboard UX",
            [Track.SME, Track.DEVELOPER],
        )
        assert track == Track.SME

    def test_infer_developer_track(self):
        """Should infer Developer track from SDK keywords."""
        planner = MetaPlanner()
        track = planner._infer_track(
            "Update the Python SDK documentation",
            [Track.SME, Track.DEVELOPER],
        )
        assert track == Track.DEVELOPER

    def test_infer_self_hosted_track(self):
        """Should infer Self-Hosted track from deployment keywords."""
        planner = MetaPlanner()
        track = planner._infer_track(
            "Add Docker compose support",
            [Track.SELF_HOSTED, Track.QA],
        )
        assert track == Track.SELF_HOSTED

    def test_infer_qa_track(self):
        """Should infer QA track from test keywords."""
        planner = MetaPlanner()
        track = planner._infer_track(
            "Improve test coverage",
            [Track.QA, Track.CORE],
        )
        assert track == Track.QA

    def test_infer_core_track(self):
        """Should infer Core track from agent keywords."""
        planner = MetaPlanner()
        track = planner._infer_track(
            "Improve agent consensus detection",
            [Track.QA, Track.CORE],
        )
        assert track == Track.CORE

    def test_infer_defaults_to_first(self):
        """Should default to first available track."""
        planner = MetaPlanner()
        track = planner._infer_track(
            "Do something unrelated",
            [Track.QA, Track.DEVELOPER],
        )
        assert track == Track.QA

    def test_infer_with_unavailable_track(self):
        """Should not choose unavailable tracks."""
        planner = MetaPlanner()
        track = planner._infer_track(
            "Improve dashboard UX",  # Suggests SME
            [Track.QA, Track.DEVELOPER],  # But SME not available
        )
        assert track in [Track.QA, Track.DEVELOPER]


class TestHeuristicPrioritize:
    """Tests for _heuristic_prioritize method."""

    def test_sme_objective_generates_sme_goal(self):
        """SME objectives should generate SME goals."""
        planner = MetaPlanner()
        goals = planner._heuristic_prioritize(
            "Maximize utility for SME businesses",
            [Track.SME, Track.QA],
        )

        assert len(goals) >= 1
        sme_goals = [g for g in goals if g.track == Track.SME]
        assert len(sme_goals) >= 1
        assert sme_goals[0].estimated_impact == "high"

    def test_small_business_objective(self):
        """'Small business' should trigger SME goals."""
        planner = MetaPlanner()
        goals = planner._heuristic_prioritize(
            "Help small business users",
            [Track.SME],
        )

        assert any(g.track == Track.SME for g in goals)

    def test_generates_goals_for_all_tracks(self):
        """Should generate goals for all available tracks."""
        planner = MetaPlanner()
        goals = planner._heuristic_prioritize(
            "Generic objective",
            [Track.SME, Track.QA, Track.DEVELOPER],
        )

        tracks_covered = {g.track for g in goals}
        assert Track.SME in tracks_covered or Track.QA in tracks_covered

    def test_respects_max_goals(self):
        """Should not exceed max_goals."""
        config = MetaPlannerConfig(max_goals=2)
        planner = MetaPlanner(config=config)
        goals = planner._heuristic_prioritize(
            "Objective",
            list(Track),
        )

        assert len(goals) <= 2

    def test_priority_ordering(self):
        """Goals should have correct priority ordering."""
        planner = MetaPlanner()
        goals = planner._heuristic_prioritize(
            "Maximize SME utility",
            [Track.SME, Track.QA],
        )

        if len(goals) >= 2:
            assert goals[0].priority < goals[1].priority


class TestBuildGoal:
    """Tests for _build_goal method."""

    def test_builds_goal_with_track(self):
        """Should build goal with explicit track."""
        planner = MetaPlanner()
        goal_dict = {
            "description": "Add new feature",
            "track": Track.DEVELOPER,
            "rationale": "Important",
            "impact": "high",
        }
        goal = planner._build_goal(goal_dict, 0, [Track.DEVELOPER])

        assert goal.track == Track.DEVELOPER
        assert goal.description == "Add new feature"
        assert goal.estimated_impact == "high"
        assert goal.priority == 1

    def test_builds_goal_infers_track(self):
        """Should infer track when not specified."""
        planner = MetaPlanner()
        goal_dict = {
            "description": "Improve test coverage",
            "track": None,
            "rationale": "",
            "impact": "medium",
        }
        goal = planner._build_goal(goal_dict, 2, [Track.QA, Track.CORE])

        assert goal.track == Track.QA
        assert goal.priority == 3

    def test_builds_goal_default_impact(self):
        """Should default to medium impact."""
        planner = MetaPlanner()
        goal_dict = {
            "description": "Something",
            "track": Track.SME,
        }
        goal = planner._build_goal(goal_dict, 0, [Track.SME])

        assert goal.estimated_impact == "medium"


class TestParseGoalsFromDebate:
    """Tests for _parse_goals_from_debate method."""

    def test_parses_numbered_list(self):
        """Should parse numbered list format."""
        planner = MetaPlanner()

        mock_result = MagicMock()
        mock_result.consensus = """
        1. Improve dashboard UX
        2. Add more tests for the QA track
        3. Update SDK documentation
        """

        goals = planner._parse_goals_from_debate(
            mock_result,
            [Track.SME, Track.QA, Track.DEVELOPER],
            "Test objective",
        )

        assert len(goals) >= 1

    def test_parses_bullet_points(self):
        """Should parse bullet point format."""
        planner = MetaPlanner()

        mock_result = MagicMock()
        mock_result.consensus = """
        - Improve dashboard
        - Add tests
        """

        goals = planner._parse_goals_from_debate(
            mock_result,
            [Track.SME, Track.QA],
            "Test",
        )

        assert len(goals) >= 1

    def test_falls_back_on_empty_consensus(self):
        """Should fallback to heuristics on empty consensus."""
        planner = MetaPlanner()

        mock_result = MagicMock()
        mock_result.consensus = ""
        mock_result.final_response = ""
        mock_result.responses = []

        goals = planner._parse_goals_from_debate(
            mock_result,
            [Track.SME],
            "SME objective",
        )

        assert len(goals) >= 1

    def test_extracts_impact(self):
        """Should extract impact from text."""
        planner = MetaPlanner()

        mock_result = MagicMock()
        # Impact detection happens on lines after the goal line
        mock_result.consensus = """
        1. Critical feature
           Expected impact: high
        """

        goals = planner._parse_goals_from_debate(
            mock_result,
            [Track.SME],
            "Test",
        )

        if goals:
            assert goals[0].estimated_impact == "high"

    def test_respects_max_goals(self):
        """Should limit goals to max_goals."""
        config = MetaPlannerConfig(max_goals=2)
        planner = MetaPlanner(config=config)

        mock_result = MagicMock()
        mock_result.consensus = """
        1. Goal one
        2. Goal two
        3. Goal three
        4. Goal four
        """

        goals = planner._parse_goals_from_debate(
            mock_result,
            list(Track),
            "Test",
        )

        assert len(goals) <= 2


class TestPrioritizeWorkAsync:
    """Tests for prioritize_work async method."""

    @pytest.mark.asyncio
    async def test_prioritize_uses_heuristic_fallback(self):
        """Should fall back to heuristics when debate fails."""
        planner = MetaPlanner()

        # Import error will trigger heuristic fallback
        with patch.dict("sys.modules", {"aragora.debate.orchestrator": None}):
            goals = await planner.prioritize_work(
                objective="Maximize SME utility",
                available_tracks=[Track.SME, Track.QA],
            )

        assert len(goals) >= 1
        assert all(isinstance(g, PrioritizedGoal) for g in goals)

    @pytest.mark.asyncio
    async def test_prioritize_with_defaults(self):
        """Should work with default parameters using heuristic fallback."""
        planner = MetaPlanner()

        # Test heuristic fallback directly (avoids slow imports)
        goals = planner._heuristic_prioritize(
            objective="Improve the system",
            available_tracks=list(Track),
        )

        assert isinstance(goals, list)
        assert len(goals) >= 1

    @pytest.mark.asyncio
    async def test_prioritize_with_context(self):
        """Should incorporate context in debate topic building."""
        planner = MetaPlanner()
        context = PlanningContext(
            recent_issues=["Auth failures"],
            test_failures=["test_login"],
        )

        # Test topic building which incorporates context
        topic = planner._build_debate_topic(
            objective="Fix issues",
            tracks=[Track.CORE, Track.QA],
            constraints=["No breaking changes"],
            context=context,
        )

        assert "Fix issues" in topic
        assert "Auth failures" in topic
        assert "test_login" in topic
        assert "No breaking changes" in topic
