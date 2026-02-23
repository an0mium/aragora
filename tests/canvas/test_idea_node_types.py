"""Tests for IdeaNodeType enum, colors, and edge types."""

from __future__ import annotations

import pytest

from aragora.canvas.stages import (
    IdeaNodeType,
    NODE_TYPE_COLORS,
    PipelineStage,
    ProvenanceLink,
    StageEdgeType,
    content_hash,
)


class TestIdeaNodeType:
    """IdeaNodeType enum membership and value tests."""

    def test_all_9_types_exist(self):
        expected = {
            "concept",
            "cluster",
            "question",
            "insight",
            "evidence",
            "assumption",
            "constraint",
            "observation",
            "hypothesis",
        }
        actual = {t.value for t in IdeaNodeType}
        assert actual == expected

    def test_observation_value(self):
        assert IdeaNodeType.OBSERVATION.value == "observation"

    def test_hypothesis_value(self):
        assert IdeaNodeType.HYPOTHESIS.value == "hypothesis"

    def test_original_7_still_present(self):
        for name in [
            "CONCEPT",
            "CLUSTER",
            "QUESTION",
            "INSIGHT",
            "EVIDENCE",
            "ASSUMPTION",
            "CONSTRAINT",
        ]:
            assert hasattr(IdeaNodeType, name)

    def test_str_enum_behavior(self):
        assert IdeaNodeType.CONCEPT == "concept"
        assert isinstance(IdeaNodeType.HYPOTHESIS, str)


class TestNodeTypeColors:
    """NODE_TYPE_COLORS has entries for all idea types."""

    def test_observation_color(self):
        assert NODE_TYPE_COLORS["observation"] == "#34d399"

    def test_hypothesis_color(self):
        assert NODE_TYPE_COLORS["hypothesis"] == "#c084fc"

    def test_all_idea_types_have_colors(self):
        for t in IdeaNodeType:
            assert t.value in NODE_TYPE_COLORS, f"Missing color for {t.value}"

    def test_colors_are_hex(self):
        for idea_type in IdeaNodeType:
            color = NODE_TYPE_COLORS[idea_type.value]
            assert color.startswith("#"), f"Color for {idea_type.value} not hex: {color}"
            assert len(color) == 7, f"Color for {idea_type.value} wrong length: {color}"


class TestStageEdgeType:
    """StageEdgeType includes idea-specific edge types."""

    def test_inspires(self):
        assert StageEdgeType.INSPIRES.value == "inspires"

    def test_refines(self):
        assert StageEdgeType.REFINES.value == "refines"

    def test_challenges(self):
        assert StageEdgeType.CHALLENGES.value == "challenges"

    def test_exemplifies(self):
        assert StageEdgeType.EXEMPLIFIES.value == "exemplifies"

    def test_cross_stage_still_present(self):
        assert StageEdgeType.DERIVED_FROM.value == "derived_from"
        assert StageEdgeType.IMPLEMENTS.value == "implements"
        assert StageEdgeType.EXECUTES.value == "executes"


class TestContentHash:
    """content_hash utility."""

    def test_deterministic(self):
        assert content_hash("hello") == content_hash("hello")

    def test_different_inputs(self):
        assert content_hash("a") != content_hash("b")

    def test_returns_16_chars(self):
        assert len(content_hash("test")) == 16


class TestProvenanceLink:
    """ProvenanceLink serialization."""

    def test_to_dict(self):
        link = ProvenanceLink(
            source_node_id="idea-1",
            source_stage=PipelineStage.IDEAS,
            target_node_id="goal-1",
            target_stage=PipelineStage.GOALS,
            content_hash="abc123",
            method="manual_promotion",
        )
        d = link.to_dict()
        assert d["source_node_id"] == "idea-1"
        assert d["source_stage"] == "ideas"
        assert d["target_stage"] == "goals"
        assert d["method"] == "manual_promotion"
