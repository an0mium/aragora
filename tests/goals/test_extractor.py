"""Tests for the GoalExtractor — structural goal extraction from idea graphs."""

from __future__ import annotations

import pytest

from aragora.canvas.stages import GoalNodeType, PipelineStage
from aragora.goals.extractor import GoalExtractor, GoalGraph, GoalNode, _STOP_WORDS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def extractor():
    return GoalExtractor()


def _make_idea_canvas(
    nodes: list[dict] | None = None,
    edges: list[dict] | None = None,
) -> dict:
    """Helper to build a minimal canvas dict."""
    return {"nodes": nodes or [], "edges": edges or []}


# ---------------------------------------------------------------------------
# GoalNode dataclass
# ---------------------------------------------------------------------------


class TestGoalNode:
    def test_defaults(self):
        g = GoalNode(id="g1", title="T", description="D")
        assert g.goal_type == GoalNodeType.GOAL
        assert g.priority == "medium"
        assert g.dependencies == []
        assert g.source_idea_ids == []
        assert g.confidence == 0.0
        assert g.metadata == {}

    def test_to_dict(self):
        g = GoalNode(
            id="g1",
            title="T",
            description="D",
            goal_type=GoalNodeType.STRATEGY,
            priority="high",
            confidence=0.75,
        )
        d = g.to_dict()
        assert d["id"] == "g1"
        assert d["type"] == "strategy"
        assert d["priority"] == "high"
        assert d["confidence"] == 0.75


# ---------------------------------------------------------------------------
# GoalGraph dataclass
# ---------------------------------------------------------------------------


class TestGoalGraph:
    def test_empty_graph(self):
        gg = GoalGraph(id="gg1")
        assert gg.goals == []
        assert gg.provenance == []
        assert gg.transition is None

    def test_to_dict_empty(self):
        d = GoalGraph(id="gg1").to_dict()
        assert d["id"] == "gg1"
        assert d["goals"] == []
        assert d["provenance"] == []
        assert d["transition"] is None

    def test_to_dict_with_goals(self):
        g = GoalNode(id="g1", title="T", description="D")
        gg = GoalGraph(id="gg1", goals=[g])
        d = gg.to_dict()
        assert len(d["goals"]) == 1
        assert d["goals"][0]["id"] == "g1"


# ---------------------------------------------------------------------------
# GoalExtractor — empty / trivial inputs
# ---------------------------------------------------------------------------


class TestExtractorEmpty:
    def test_empty_canvas(self, extractor):
        result = extractor.extract_from_ideas(_make_idea_canvas())
        assert isinstance(result, GoalGraph)
        assert result.goals == []
        assert result.id.startswith("goals-")

    def test_no_nodes_key(self, extractor):
        result = extractor.extract_from_ideas({})
        assert result.goals == []

    def test_empty_raw_ideas(self, extractor):
        result = extractor.extract_from_raw_ideas([])
        assert result.goals == []


# ---------------------------------------------------------------------------
# GoalExtractor — structural extraction
# ---------------------------------------------------------------------------


class TestExtractFromIdeas:
    def test_single_node(self, extractor):
        canvas = _make_idea_canvas(
            nodes=[{"id": "n1", "label": "improve performance", "data": {"idea_type": "concept"}}],
        )
        result = extractor.extract_from_ideas(canvas)
        assert len(result.goals) == 1
        goal = result.goals[0]
        assert goal.id.startswith("goal-")
        assert "Improve performance" in goal.title
        assert goal.goal_type == GoalNodeType.GOAL

    def test_priority_from_support(self, extractor):
        """Nodes with >= 5 support edges get 'critical' priority."""
        nodes = [
            {"id": "main", "label": "central idea", "data": {"idea_type": "concept"}},
        ]
        # Add 5 support nodes
        for i in range(5):
            nodes.append({"id": f"s{i}", "label": f"supporter {i}", "data": {}})
        edges = [
            {"source": f"s{i}", "target": "main", "type": "support"} for i in range(5)
        ]
        canvas = _make_idea_canvas(nodes=nodes, edges=edges)
        result = extractor.extract_from_ideas(canvas)
        # The main node should rank highest and get critical priority
        main_goal = next((g for g in result.goals if "Central idea" in g.title), None)
        assert main_goal is not None
        assert main_goal.priority == "critical"

    def test_priority_high(self, extractor):
        """Nodes with 3-4 support get 'high' priority."""
        nodes = [
            {"id": "main", "label": "the idea", "data": {"idea_type": "concept"}},
        ]
        for i in range(3):
            nodes.append({"id": f"s{i}", "label": f"s{i}", "data": {}})
        edges = [{"source": f"s{i}", "target": "main", "type": "support"} for i in range(3)]
        canvas = _make_idea_canvas(nodes=nodes, edges=edges)
        result = extractor.extract_from_ideas(canvas)
        main_goal = next((g for g in result.goals if "The idea" in g.title), None)
        assert main_goal is not None
        assert main_goal.priority == "high"

    def test_constraint_becomes_principle(self, extractor):
        canvas = _make_idea_canvas(
            nodes=[{"id": "c1", "label": "security first", "data": {"idea_type": "constraint"}}],
        )
        result = extractor.extract_from_ideas(canvas)
        assert result.goals[0].goal_type == GoalNodeType.PRINCIPLE
        assert "Maintain" in result.goals[0].title

    def test_question_becomes_milestone(self, extractor):
        canvas = _make_idea_canvas(
            nodes=[{"id": "q1", "label": "how to scale?", "data": {"idea_type": "question"}}],
        )
        result = extractor.extract_from_ideas(canvas)
        assert result.goals[0].goal_type == GoalNodeType.MILESTONE
        assert "Complete" in result.goals[0].title

    def test_provenance_links(self, extractor):
        canvas = _make_idea_canvas(
            nodes=[{"id": "n1", "label": "test idea", "data": {"idea_type": "concept"}}],
        )
        result = extractor.extract_from_ideas(canvas)
        assert len(result.provenance) >= 1
        link = result.provenance[0]
        assert link.source_stage == PipelineStage.IDEAS
        assert link.target_stage == PipelineStage.GOALS
        assert link.method == "structural_extraction"

    def test_transition_record(self, extractor):
        canvas = _make_idea_canvas(
            nodes=[{"id": "n1", "label": "idea", "data": {"idea_type": "concept"}}],
        )
        result = extractor.extract_from_ideas(canvas)
        assert result.transition is not None
        assert result.transition.from_stage == PipelineStage.IDEAS
        assert result.transition.to_stage == PipelineStage.GOALS
        assert result.transition.status == "pending"
        assert "1 goals" in result.transition.ai_rationale

    def test_max_goals_capped(self, extractor):
        """At most 1 goal per 3 ideas."""
        nodes = [
            {"id": f"n{i}", "label": f"idea {i}", "data": {"idea_type": "concept"}}
            for i in range(9)
        ]
        canvas = _make_idea_canvas(nodes=nodes)
        result = extractor.extract_from_ideas(canvas)
        assert len(result.goals) <= 3

    def test_confidence_from_score(self, extractor):
        """Confidence = min(1.0, score / 10.0)."""
        canvas = _make_idea_canvas(
            nodes=[{"id": "n1", "label": "idea", "data": {"idea_type": "concept"}}],
        )
        result = extractor.extract_from_ideas(canvas)
        # concept type_bonus = 2.0, 0 support, 0 evidence → score = 2.0
        assert result.goals[0].confidence == pytest.approx(0.2)

    def test_source_idea_ids(self, extractor):
        canvas = _make_idea_canvas(
            nodes=[{"id": "n1", "label": "idea", "data": {"idea_type": "concept"}}],
        )
        result = extractor.extract_from_ideas(canvas)
        assert "n1" in result.goals[0].source_idea_ids

    def test_metadata_contains_rank_and_score(self, extractor):
        canvas = _make_idea_canvas(
            nodes=[{"id": "n1", "label": "idea", "data": {"idea_type": "concept"}}],
        )
        result = extractor.extract_from_ideas(canvas)
        meta = result.goals[0].metadata
        assert "rank" in meta
        assert "support_score" in meta

    def test_edge_support_and_supports_both_counted(self, extractor):
        """Both 'support' and 'supports' edge types increment support_count."""
        nodes = [
            {"id": "main", "label": "main", "data": {"idea_type": "concept"}},
            {"id": "s1", "label": "s1", "data": {}},
            {"id": "s2", "label": "s2", "data": {}},
        ]
        edges = [
            {"source": "s1", "target": "main", "type": "support"},
            {"source": "s2", "target": "main", "type": "supports"},
        ]
        canvas = _make_idea_canvas(nodes=nodes, edges=edges)
        result = extractor.extract_from_ideas(canvas)
        main_goal = next((g for g in result.goals if "Main" in g.title), None)
        assert main_goal is not None
        # 2 support → total_support=2 → "medium" priority
        assert main_goal.priority == "medium"


# ---------------------------------------------------------------------------
# GoalExtractor — raw ideas
# ---------------------------------------------------------------------------


class TestExtractFromRawIdeas:
    def test_single_idea(self, extractor):
        result = extractor.extract_from_raw_ideas(["improve database performance"])
        assert len(result.goals) == 1
        assert "Improve database performance" in result.goals[0].title

    def test_multiple_ideas(self, extractor):
        result = extractor.extract_from_raw_ideas([
            "optimize database queries",
            "add caching layer",
            "improve API response times",
        ])
        assert len(result.goals) >= 1

    def test_keyword_linking(self, extractor):
        """Ideas sharing >= 3 non-stop words get a relates_to edge."""
        result = extractor.extract_from_raw_ideas([
            "improve database query performance significantly fast",
            "enhance database query optimization performance fast",
        ])
        # With edges, connectivity affects scoring
        assert len(result.goals) >= 1

    def test_label_truncated_to_80(self, extractor):
        long_idea = "x " * 100  # 200 chars
        result = extractor.extract_from_raw_ideas([long_idea])
        assert len(result.goals) == 1
        # Label was truncated before synthesis, but title prepends prefix
        assert len(result.goals[0].title) < 200


# ---------------------------------------------------------------------------
# Internal methods
# ---------------------------------------------------------------------------


class TestRankCandidates:
    def test_scoring_formula(self, extractor):
        """Score = 3*support + 2*evidence + type_bonus."""
        node_map = {
            "n1": {"data": {"idea_type": "concept"}},
        }
        support = {"n1": 2}
        evidence = {"n1": 1}
        result = extractor._rank_candidates(node_map, support, evidence)
        # 3*2 + 2*1 + 2.0 (concept bonus) = 10.0
        assert result[0] == ("n1", 10.0)

    def test_cluster_has_highest_bonus(self, extractor):
        node_map = {
            "a": {"data": {"idea_type": "concept"}},
            "b": {"data": {"idea_type": "cluster"}},
        }
        support = {"a": 0, "b": 0}
        evidence = {"a": 0, "b": 0}
        result = extractor._rank_candidates(node_map, support, evidence)
        # cluster bonus=3.0 > concept bonus=2.0
        assert result[0][0] == "b"

    def test_unknown_type_gets_default_bonus(self, extractor):
        node_map = {"n1": {"data": {"idea_type": "unknown_type"}}}
        result = extractor._rank_candidates(node_map, {"n1": 0}, {"n1": 0})
        # default bonus = 1.0
        assert result[0][1] == 1.0


class TestIdeaTypeToGoalType:
    def test_cluster(self, extractor):
        assert extractor._idea_type_to_goal_type("cluster", 1.0) == GoalNodeType.GOAL

    def test_constraint(self, extractor):
        assert extractor._idea_type_to_goal_type("constraint", 1.0) == GoalNodeType.PRINCIPLE

    def test_insight_high_score(self, extractor):
        assert extractor._idea_type_to_goal_type("insight", 5.0) == GoalNodeType.STRATEGY

    def test_insight_low_score(self, extractor):
        assert extractor._idea_type_to_goal_type("insight", 4.0) == GoalNodeType.GOAL

    def test_question(self, extractor):
        assert extractor._idea_type_to_goal_type("question", 1.0) == GoalNodeType.MILESTONE

    def test_default(self, extractor):
        assert extractor._idea_type_to_goal_type("concept", 1.0) == GoalNodeType.GOAL


class TestSynthesizeTitle:
    def test_goal_prefix(self, extractor):
        title = extractor._synthesize_goal_title("optimize perf", GoalNodeType.GOAL)
        assert title == "Achieve: Optimize perf"

    def test_principle_prefix(self, extractor):
        title = extractor._synthesize_goal_title("security first", GoalNodeType.PRINCIPLE)
        assert title == "Maintain: Security first"

    def test_strategy_prefix(self, extractor):
        title = extractor._synthesize_goal_title("caching", GoalNodeType.STRATEGY)
        assert title == "Implement: Caching"

    def test_milestone_prefix(self, extractor):
        title = extractor._synthesize_goal_title("resolve scaling", GoalNodeType.MILESTONE)
        assert title == "Complete: Resolve scaling"

    def test_metric_prefix(self, extractor):
        title = extractor._synthesize_goal_title("latency", GoalNodeType.METRIC)
        assert title == "Measure: Latency"

    def test_risk_prefix(self, extractor):
        title = extractor._synthesize_goal_title("data loss", GoalNodeType.RISK)
        assert title == "Mitigate: Data loss"

    def test_trailing_dot_stripped(self, extractor):
        title = extractor._synthesize_goal_title("idea.", GoalNodeType.GOAL)
        assert not title.endswith(".")

    def test_capitalization(self, extractor):
        title = extractor._synthesize_goal_title("lowercase start", GoalNodeType.GOAL)
        assert "Lowercase start" in title


class TestSynthesizeDescription:
    def test_basic_description(self, extractor):
        desc = extractor._synthesize_goal_description(
            "label", {"full_content": "the full idea"}, GoalNodeType.GOAL
        )
        assert "the full idea" in desc

    def test_with_agent(self, extractor):
        desc = extractor._synthesize_goal_description(
            "label", {"agent": "claude-3-opus"}, GoalNodeType.GOAL
        )
        assert "claude-3-opus" in desc

    def test_fallback_to_label(self, extractor):
        desc = extractor._synthesize_goal_description("the label", {}, GoalNodeType.GOAL)
        assert "the label" in desc


class TestStopWords:
    def test_contains_common_words(self):
        assert "the" in _STOP_WORDS
        assert "and" in _STOP_WORDS
        assert "is" in _STOP_WORDS

    def test_does_not_contain_content_words(self):
        assert "database" not in _STOP_WORDS
        assert "performance" not in _STOP_WORDS
