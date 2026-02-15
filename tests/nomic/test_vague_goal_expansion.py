"""Tests for TaskDecomposer vague goal expansion.

Verifies that abstract goals like "maximize utility for SMEs" produce
concrete subtasks by cross-referencing deliberation templates and
development track configs.
"""

import pytest

from aragora.nomic.task_decomposer import (
    TaskDecomposer,
    TaskDecomposition,
    DecomposerConfig,
)


class TestVagueGoalExpansion:
    """Tests for _expand_vague_goal and its integration into analyze()."""

    def test_sme_goal_produces_subtasks(self):
        """'maximize utility for SMEs' should produce 3+ concrete subtasks."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze("maximize utility for SMEs")

        assert result.should_decompose is True
        assert len(result.subtasks) >= 3
        assert result.complexity_score >= 5
        assert "expanded" in result.rationale.lower() or "vague" in result.rationale.lower()

    def test_improve_security_produces_subtasks(self):
        """Security-related vague goals should match security templates and tracks."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze("harden security posture")

        assert result.should_decompose is True
        assert len(result.subtasks) >= 2

        # Should reference security-related content
        all_text = " ".join(
            f"{st.title} {st.description}" for st in result.subtasks
        ).lower()
        assert "security" in all_text or "audit" in all_text

    def test_improve_developer_experience(self):
        """Developer-focused vague goals should match developer templates."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze("improve developer experience")

        assert result.should_decompose is True
        assert len(result.subtasks) >= 2

    def test_concrete_goal_not_expanded(self):
        """Concrete goals with files shouldn't trigger vague expansion."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze(
            "Refactor database layer in models.py, db.py, and handlers.py "
            "to support multi-tenancy with migration scripts"
        )

        # This should score high on heuristic complexity, bypassing expansion
        if result.should_decompose:
            # Should not say "vague" -- it was decomposed normally
            assert "vague" not in result.rationale.lower() or result.complexity_score >= 3

    def test_expansion_respects_max_subtasks(self):
        """Expanded subtasks should not exceed config.max_subtasks."""
        config = DecomposerConfig(max_subtasks=3)
        decomposer = TaskDecomposer(config=config)
        result = decomposer.analyze("maximize utility for SMEs")

        if result.should_decompose:
            assert len(result.subtasks) <= 3

    def test_expansion_sets_elevated_complexity(self):
        """Expanded vague goals should have elevated complexity score."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze("maximize utility for SMEs")

        # Vague goals are inherently complex even if heuristic says otherwise
        assert result.complexity_score >= 5
        assert result.complexity_level in ("medium", "high")

    def test_expansion_subtasks_have_descriptions(self):
        """Each expanded subtask should have meaningful description."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze("maximize utility for SMEs")

        for subtask in result.subtasks:
            assert subtask.title, "Subtask should have a title"
            assert subtask.description, "Subtask should have a description"
            assert len(subtask.description) > 10, "Description should be meaningful"

    def test_expansion_includes_template_sources(self):
        """Rationale should mention template or track sources."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze("improve quality and testing")

        if "expanded" in result.rationale.lower():
            assert "template:" in result.rationale or "track:" in result.rationale

    def test_single_word_goal_still_works(self):
        """Even very short goals should not crash."""
        decomposer = TaskDecomposer()
        # "security" alone should match security templates
        result = decomposer.analyze("security")

        # May or may not expand, but should not crash
        assert isinstance(result, TaskDecomposition)

    def test_empty_goal_not_expanded(self):
        """Empty goals should return early, not attempt expansion."""
        decomposer = TaskDecomposer()
        result = decomposer.analyze("")

        assert result.complexity_score == 0
        assert result.should_decompose is False


class TestExpandVagueGoalDirect:
    """Tests for _expand_vague_goal method directly."""

    def test_returns_none_for_no_matches(self):
        """Should return None if no templates or tracks match."""
        decomposer = TaskDecomposer()
        result = decomposer._expand_vague_goal("xyzzy foobar nonsense")

        # No templates or tracks should match random words
        assert result is None

    def test_sme_matches_track(self):
        """'SME' keyword should match the SME track."""
        decomposer = TaskDecomposer()
        result = decomposer._expand_vague_goal("maximize utility for SMEs")

        assert result is not None
        assert result.should_decompose is True
        # Should have track-based subtask
        track_titles = [st.title for st in result.subtasks]
        assert any("SME" in t for t in track_titles)

    def test_quality_matches_qa_track(self):
        """Quality/testing goals should match QA track."""
        decomposer = TaskDecomposer()
        result = decomposer._expand_vague_goal("improve test coverage and reliability")

        assert result is not None
        subtask_text = " ".join(st.title + " " + st.description for st in result.subtasks)
        assert "QA" in subtask_text or "test" in subtask_text.lower()

    def test_subtask_ids_are_sequential(self):
        """Subtask IDs should be sequential."""
        decomposer = TaskDecomposer()
        result = decomposer._expand_vague_goal("maximize utility for SMEs")

        if result is not None:
            for i, subtask in enumerate(result.subtasks):
                assert subtask.id == f"subtask_{i + 1}"

    def test_track_subtasks_include_folders(self):
        """Track-derived subtasks should include file_scope from track config."""
        decomposer = TaskDecomposer()
        result = decomposer._expand_vague_goal("improve developer SDK experience")

        if result is not None:
            track_subtasks = [st for st in result.subtasks if st.file_scope]
            assert len(track_subtasks) >= 1


class TestMatchTemplates:
    """Tests for the match_templates registry function."""

    def test_match_templates_import(self):
        """match_templates should be importable from the templates package."""
        from aragora.deliberation.templates import match_templates

        results = match_templates("code review security")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_match_templates_returns_relevant(self):
        """Should return templates relevant to the query."""
        from aragora.deliberation.templates import match_templates

        results = match_templates("security audit vulnerabilities")
        assert len(results) > 0
        names = [t.name for t in results]
        assert "security_audit" in names

    def test_match_templates_limit(self):
        """Should respect the limit parameter."""
        from aragora.deliberation.templates import match_templates

        results = match_templates("review", limit=2)
        assert len(results) <= 2

    def test_match_templates_empty_query(self):
        """Empty query should return empty list."""
        from aragora.deliberation.templates import match_templates

        results = match_templates("")
        assert results == []


class TestMetaPlannerTemplateInjection:
    """Tests for MetaPlanner template injection into debate topic."""

    def test_debate_topic_includes_templates(self):
        """Planning debate topic should include relevant templates."""
        from aragora.nomic.meta_planner import MetaPlanner, Track, PlanningContext

        planner = MetaPlanner()
        topic = planner._build_debate_topic(
            objective="Improve security posture",
            tracks=[Track.SECURITY, Track.QA],
            constraints=[],
            context=PlanningContext(),
        )

        assert "RELEVANT DELIBERATION TEMPLATES" in topic
        assert "security" in topic.lower()

    def test_debate_topic_no_crash_without_templates(self):
        """Should not crash if template matching finds nothing."""
        from aragora.nomic.meta_planner import MetaPlanner, Track, PlanningContext

        planner = MetaPlanner()
        topic = planner._build_debate_topic(
            objective="xyzzy foobar",
            tracks=[Track.DEVELOPER],
            constraints=[],
            context=PlanningContext(),
        )

        # Should still have YOUR TASK section
        assert "YOUR TASK" in topic
