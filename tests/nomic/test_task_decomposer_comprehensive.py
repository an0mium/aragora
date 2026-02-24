"""Comprehensive tests for TaskDecomposer -- deep coverage of decomposition
logic, AI extraction, KM enrichment, debate parsing, goal tree operations,
and edge cases.

This supplements the existing test_task_decomposer.py with additional
scenarios focused on:
- AI extraction function integration
- Hierarchical goal tree (build_tree, flatten_tree, get_children, max_depth)
- _parse_debate_subtasks JSON extraction
- _is_specific_goal classification
- _score_codebase_relevance module mapping
- Custom extract_subtasks_fn override
- _estimate_concept_complexity mapping
- Generic phase fallback when no concepts found
- enrich_subtasks_from_km async overlay
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.task_decomposer import (
    COMPLEXITY_INDICATORS,
    DECOMPOSITION_CONCEPTS,
    DecomposerConfig,
    SubTask,
    TaskDecomposer,
    TaskDecomposition,
    analyze_task,
    get_task_decomposer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def decomposer() -> TaskDecomposer:
    return TaskDecomposer()


@pytest.fixture
def low_threshold_decomposer() -> TaskDecomposer:
    return TaskDecomposer(DecomposerConfig(complexity_threshold=2))


# ---------------------------------------------------------------------------
# AI extraction function
# ---------------------------------------------------------------------------


class TestAIExtractionFunction:
    def test_extract_subtasks_fn_used_when_provided(self):
        """Custom AI extraction function should override heuristic."""
        extracted = [
            {"title": "AI Step 1", "description": "Do step 1", "complexity": "low"},
            {"title": "AI Step 2", "description": "Do step 2", "complexity": "high"},
        ]
        decomposer = TaskDecomposer(
            config=DecomposerConfig(complexity_threshold=1),
            extract_subtasks_fn=lambda task: extracted,
        )
        result = decomposer.analyze(
            "Refactor the authentication system with security checks"
        )

        if result.should_decompose:
            assert any("AI Step 1" in st.title for st in result.subtasks)
            assert any("AI Step 2" in st.title for st in result.subtasks)

    def test_extract_subtasks_fn_fallback_on_error(self):
        """If AI extraction raises, fallback to heuristic decomposition."""

        def failing_extractor(task: str) -> list[dict]:
            raise RuntimeError("AI unavailable")

        decomposer = TaskDecomposer(
            config=DecomposerConfig(complexity_threshold=1),
            extract_subtasks_fn=failing_extractor,
        )
        result = decomposer.analyze(
            "Major system-wide refactor of database and api and security layers"
        )

        # Should still produce subtasks from heuristic fallback
        if result.should_decompose:
            assert len(result.subtasks) >= 1

    def test_extract_subtasks_fn_empty_result_falls_back(self):
        """If AI extraction returns empty list, fall back to heuristic."""
        decomposer = TaskDecomposer(
            config=DecomposerConfig(complexity_threshold=1),
            extract_subtasks_fn=lambda task: [],
        )
        result = decomposer.analyze(
            "Refactor the database and security and api layers system-wide"
        )

        if result.should_decompose:
            # Fallback to heuristic should still generate subtasks
            assert len(result.subtasks) >= 1

    def test_extract_subtasks_capped_at_max(self):
        """Extraction should respect max_subtasks config."""
        extracted = [
            {"title": f"Step {i}", "description": f"Desc {i}"}
            for i in range(20)
        ]
        decomposer = TaskDecomposer(
            config=DecomposerConfig(
                complexity_threshold=1, max_subtasks=3
            ),
            extract_subtasks_fn=lambda task: extracted,
        )
        result = decomposer.analyze(
            "Refactor the entire database system with security and api changes"
        )

        if result.should_decompose:
            assert len(result.subtasks) <= 3


# ---------------------------------------------------------------------------
# Hierarchical goal tree
# ---------------------------------------------------------------------------


class TestGoalTree:
    def test_build_tree_creates_hierarchy(self):
        """build_tree should populate children from parent_id."""
        decomp = TaskDecomposition(
            original_task="test",
            complexity_score=7,
            complexity_level="high",
            should_decompose=True,
            subtasks=[
                SubTask(id="root", title="Root", description="Root task", parent_id=None, depth=0),
                SubTask(id="child1", title="Child 1", description="C1", parent_id="root", depth=1),
                SubTask(id="child2", title="Child 2", description="C2", parent_id="root", depth=1),
                SubTask(id="grandchild", title="GC", description="GC", parent_id="child1", depth=2),
            ],
        )

        roots = decomp.build_tree()
        assert len(roots) == 1
        assert roots[0].id == "root"
        assert len(roots[0].children) == 2
        child1 = next(c for c in roots[0].children if c.id == "child1")
        assert len(child1.children) == 1
        assert child1.children[0].id == "grandchild"

    def test_get_roots_returns_only_root_level(self):
        """get_roots should return subtasks with parent_id=None."""
        decomp = TaskDecomposition(
            original_task="test",
            complexity_score=5,
            complexity_level="medium",
            should_decompose=True,
            subtasks=[
                SubTask(id="a", title="A", description="", parent_id=None),
                SubTask(id="b", title="B", description="", parent_id="a"),
                SubTask(id="c", title="C", description="", parent_id=None),
            ],
        )
        roots = decomp.get_roots()
        assert len(roots) == 2
        assert {r.id for r in roots} == {"a", "c"}

    def test_get_children(self):
        """get_children should return direct children of a parent."""
        decomp = TaskDecomposition(
            original_task="test",
            complexity_score=5,
            complexity_level="medium",
            should_decompose=True,
            subtasks=[
                SubTask(id="root", title="Root", description=""),
                SubTask(id="c1", title="C1", description="", parent_id="root"),
                SubTask(id="c2", title="C2", description="", parent_id="root"),
                SubTask(id="gc1", title="GC1", description="", parent_id="c1"),
            ],
        )
        children = decomp.get_children("root")
        assert len(children) == 2
        assert {c.id for c in children} == {"c1", "c2"}

    def test_max_depth(self):
        """max_depth should return the maximum nesting level."""
        decomp = TaskDecomposition(
            original_task="test",
            complexity_score=5,
            complexity_level="medium",
            should_decompose=True,
            subtasks=[
                SubTask(id="a", title="A", description="", depth=0),
                SubTask(id="b", title="B", description="", depth=1),
                SubTask(id="c", title="C", description="", depth=3),
            ],
        )
        assert decomp.max_depth() == 3

    def test_max_depth_empty(self):
        """max_depth with no subtasks should return 0."""
        decomp = TaskDecomposition(
            original_task="test",
            complexity_score=1,
            complexity_level="low",
            should_decompose=False,
        )
        assert decomp.max_depth() == 0

    def test_flatten_tree(self):
        """flatten_tree should return depth-first ordering."""
        decomp = TaskDecomposition(
            original_task="test",
            complexity_score=7,
            complexity_level="high",
            should_decompose=True,
            subtasks=[
                SubTask(id="root", title="Root", description="", parent_id=None),
                SubTask(id="child1", title="C1", description="", parent_id="root"),
                SubTask(id="grandchild", title="GC", description="", parent_id="child1"),
                SubTask(id="child2", title="C2", description="", parent_id="root"),
            ],
        )
        flat = decomp.flatten_tree()
        ids = [s.id for s in flat]
        # Depth-first: root -> child1 -> grandchild -> child2
        assert ids == ["root", "child1", "grandchild", "child2"]


# ---------------------------------------------------------------------------
# Debate subtask parsing
# ---------------------------------------------------------------------------


class TestParseDebateSubtasks:
    def test_parse_json_code_block(self, decomposer: TaskDecomposer):
        """Should parse JSON from ```json code block."""
        subtasks_json = json.dumps([
            {"title": "Step A", "description": "Do A", "complexity": "low", "files": ["a.py"]},
            {"title": "Step B", "description": "Do B", "complexity": "high", "dependencies": ["subtask_1"]},
        ])
        text = f"Here are the subtasks:\n```json\n{subtasks_json}\n```"

        result = decomposer._parse_debate_subtasks(text)
        assert len(result) == 2
        assert result[0].title == "Step A"
        assert result[0].file_scope == ["a.py"]
        assert result[1].dependencies == ["subtask_1"]

    def test_parse_raw_json_array(self, decomposer: TaskDecomposer):
        """Should parse raw JSON array without code block."""
        subtasks_json = json.dumps([
            {"title": "Only Step", "description": "Single step"},
        ])
        text = f"The result is: {subtasks_json}"

        result = decomposer._parse_debate_subtasks(text)
        assert len(result) == 1
        assert result[0].title == "Only Step"

    def test_parse_no_json_returns_empty(self, decomposer: TaskDecomposer):
        """No JSON in text should return empty list."""
        result = decomposer._parse_debate_subtasks("Just some plain text without JSON")
        assert result == []

    def test_parse_invalid_json_returns_empty(self, decomposer: TaskDecomposer):
        """Invalid JSON should return empty list."""
        result = decomposer._parse_debate_subtasks("```json\n{broken\n```")
        assert result == []

    def test_parse_single_object_wrapped_in_list(self, decomposer: TaskDecomposer):
        """A single JSON object (not array) should be wrapped in a list."""
        obj = json.dumps({"title": "Solo", "description": "Only one"})
        text = f"```json\n{obj}\n```"

        result = decomposer._parse_debate_subtasks(text)
        assert len(result) == 1
        assert result[0].title == "Solo"

    def test_parse_non_dict_items_skipped(self, decomposer: TaskDecomposer):
        """Non-dict items in the parsed array should be skipped."""
        text = '```json\n["not a dict", {"title": "Valid", "description": "ok"}]\n```'
        result = decomposer._parse_debate_subtasks(text)
        assert len(result) == 1
        assert result[0].title == "Valid"


# ---------------------------------------------------------------------------
# _is_specific_goal classification
# ---------------------------------------------------------------------------


class TestIsSpecificGoal:
    def test_specific_action_plus_technical_term(self, decomposer: TaskDecomposer):
        """Action verb + technical term = specific."""
        assert decomposer._is_specific_goal("add retry logic to connectors") is True

    def test_specific_action_plus_module(self, decomposer: TaskDecomposer):
        """Action verb + module area = specific."""
        assert decomposer._is_specific_goal("improve test coverage") is True

    def test_vague_strategic_goal(self, decomposer: TaskDecomposer):
        """Strategic goals without technical terms = not specific."""
        assert decomposer._is_specific_goal("maximize utility for SMEs") is False

    def test_only_action_verb_not_specific(self, decomposer: TaskDecomposer):
        """Action verb alone without technical or module = not specific."""
        assert decomposer._is_specific_goal("fix the problem") is False


# ---------------------------------------------------------------------------
# _score_codebase_relevance
# ---------------------------------------------------------------------------


class TestCodebaseRelevance:
    def test_debate_module_recognized(self, decomposer: TaskDecomposer):
        """Goal mentioning 'debate' should resolve to aragora/debate/."""
        paths = decomposer._score_codebase_relevance("improve the debate orchestrator")
        assert "aragora/debate/" in paths

    def test_multiple_modules_recognized(self, decomposer: TaskDecomposer):
        """Goal mentioning multiple modules should return multiple paths."""
        paths = decomposer._score_codebase_relevance("improve workflow and security")
        assert "aragora/workflow/" in paths
        assert "aragora/security/" in paths

    def test_no_modules_returns_empty(self, decomposer: TaskDecomposer):
        """Goal with no module keywords returns empty."""
        paths = decomposer._score_codebase_relevance("maximize business value")
        assert paths == []

    def test_capped_at_five(self, decomposer: TaskDecomposer):
        """Result should be capped at 5 paths."""
        # mention many modules
        goal = "update debate agents analytics audit billing cli compliance connectors"
        paths = decomposer._score_codebase_relevance(goal)
        assert len(paths) <= 5


# ---------------------------------------------------------------------------
# _estimate_concept_complexity
# ---------------------------------------------------------------------------


class TestEstimateConceptComplexity:
    @pytest.mark.parametrize(
        "concept,expected",
        [
            ("database", "high"),
            ("security", "high"),
            ("api", "medium"),
            ("backend", "medium"),
            ("frontend", "medium"),
            ("testing", "low"),
            ("documentation", "low"),
        ],
    )
    def test_concept_complexity_mapping(
        self, decomposer: TaskDecomposer, concept: str, expected: str
    ):
        assert decomposer._estimate_concept_complexity(concept) == expected


# ---------------------------------------------------------------------------
# Generic phase fallback
# ---------------------------------------------------------------------------


class TestGenericPhaseFallback:
    def test_no_concepts_produces_generic_phases(self, decomposer: TaskDecomposer):
        """When no concept keywords match, generic phases are created."""
        phases = decomposer._create_generic_phases("Do something vague")
        assert len(phases) == 3
        assert phases[0].title == "Analysis & Design"
        assert phases[1].title == "Core Implementation"
        assert phases[2].title == "Testing & Integration"
        # Check dependency chain
        assert phases[0].dependencies == []
        assert "subtask_1" in phases[1].dependencies
        assert "subtask_2" in phases[2].dependencies


# ---------------------------------------------------------------------------
# enrich_subtasks_from_km (async overlay)
# ---------------------------------------------------------------------------


class TestEnrichSubtasksFromKm:
    @pytest.mark.asyncio
    async def test_enrichment_adds_km_warnings(self):
        """KM enrichment should add warnings from recurring failures."""
        decomposer = TaskDecomposer()
        subtasks = [
            SubTask(id="s1", title="Improve tests", description="Better coverage"),
        ]

        mock_adapter = AsyncMock()
        mock_adapter.find_recurring_failures = AsyncMock(return_value=[
            {
                "pattern": "tests flaky on CI",
                "affected_tracks": ["qa"],
                "occurrences": 3,
            }
        ])
        mock_adapter.find_high_roi_goal_types = AsyncMock(return_value=[])

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=mock_adapter,
        ):
            enriched = await decomposer.enrich_subtasks_from_km("improve tests", subtasks)

        assert len(enriched) >= 1
        # Warning should be injected into success_criteria
        assert "km_warnings" in enriched[0].success_criteria

    @pytest.mark.asyncio
    async def test_enrichment_adds_high_roi_subtask(self):
        """KM enrichment should add subtask from high-ROI patterns."""
        decomposer = TaskDecomposer(DecomposerConfig(max_subtasks=5))
        subtasks = [
            SubTask(id="s1", title="Fix bugs", description="Fix things"),
        ]

        mock_adapter = AsyncMock()
        mock_adapter.find_recurring_failures = AsyncMock(return_value=[])
        mock_adapter.find_high_roi_goal_types = AsyncMock(return_value=[
            {
                "pattern": "lint coverage",
                "avg_improvement_score": 0.8,
                "example_objectives": ["Run linter on all modules"],
            }
        ])

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=mock_adapter,
        ):
            enriched = await decomposer.enrich_subtasks_from_km("improve quality", subtasks)

        assert len(enriched) == 2
        assert "KM-suggested" in enriched[1].title

    @pytest.mark.asyncio
    async def test_enrichment_graceful_on_import_error(self):
        """If NomicCycleAdapter is not available, enrichment returns original subtasks."""
        decomposer = TaskDecomposer()
        subtasks = [
            SubTask(id="s1", title="Test", description="Test"),
        ]

        with patch.dict("sys.modules", {
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter": None,
        }):
            enriched = await decomposer.enrich_subtasks_from_km("test", subtasks)

        assert enriched is subtasks  # Should return the same list unchanged


# ---------------------------------------------------------------------------
# Complexity indicator constants
# ---------------------------------------------------------------------------


class TestComplexityConstants:
    def test_high_indicators_contain_refactor(self):
        assert "refactor" in COMPLEXITY_INDICATORS["high"]

    def test_medium_indicators_contain_implement(self):
        assert "implement" in COMPLEXITY_INDICATORS["medium"]

    def test_low_indicators_contain_fix(self):
        assert "fix" in COMPLEXITY_INDICATORS["low"]

    def test_decomposition_concepts_contain_expected_areas(self):
        expected = {"database", "api", "security", "testing", "frontend", "backend"}
        assert expected.issubset(set(DECOMPOSITION_CONCEPTS))
