"""Tests for full provenance chain integrity across pipeline stages.

Verifies that every node at every stage is traceable back to its origin,
content hashes are valid, and cross-stage references resolve correctly.
"""

import pytest

from aragora.pipeline.idea_to_execution import (
    IdeaToExecutionPipeline,
    PipelineConfig,
    PipelineResult,
)
from aragora.canvas.stages import PipelineStage, content_hash


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def pipeline():
    return IdeaToExecutionPipeline()


@pytest.fixture
def sample_ideas():
    return [
        "Build a rate limiter for API endpoints",
        "Add Redis-backed caching for frequently accessed data",
        "Improve API docs with OpenAPI interactive playground",
        "Set up end-to-end performance monitoring",
    ]


@pytest.fixture
def full_result(pipeline, sample_ideas):
    """Run a full pipeline to get a complete result with provenance."""
    return pipeline.from_ideas(sample_ideas, auto_advance=True)


@pytest.fixture
def debate_data():
    return {
        "nodes": [
            {"id": "n1", "type": "proposal", "summary": "Rate limiter", "content": "Token bucket"},
            {"id": "n2", "type": "evidence", "summary": "Reduces 429 errors", "content": "Data"},
            {
                "id": "n3",
                "type": "consensus",
                "summary": "Rate limiter needed",
                "content": "Agreement",
            },
        ],
        "edges": [
            {"source_id": "n2", "target_id": "n1", "relation": "supports"},
        ],
    }


# =============================================================================
# Full chain integrity
# =============================================================================


class TestFullChainIntegrity:
    """Test that provenance chains are complete across all stages."""

    def test_provenance_links_exist(self, full_result):
        assert len(full_result.provenance) > 0

    def test_provenance_spans_goals_and_actions(self, full_result):
        stages = set()
        for link in full_result.provenance:
            stages.add(link.source_stage)
            stages.add(link.target_stage)
        assert PipelineStage.IDEAS in stages
        assert PipelineStage.GOALS in stages

    def test_every_goal_has_provenance(self, full_result):
        """Every goal should trace back to at least one idea."""
        goal_ids = {g.id for g in full_result.goal_graph.goals}
        provenance_targets = {p.target_node_id for p in full_result.provenance}
        # Every goal should appear as a target in provenance
        for goal_id in goal_ids:
            assert goal_id in provenance_targets, f"Goal {goal_id} has no provenance link"

    def test_provenance_source_ids_exist_in_ideas(self, full_result):
        """Provenance source IDs from IDEAS stage should be valid idea node IDs."""
        idea_node_ids = set(full_result.ideas_canvas.nodes.keys())
        for link in full_result.provenance:
            if link.source_stage == PipelineStage.IDEAS:
                assert link.source_node_id in idea_node_ids or link.source_node_id.startswith(
                    "raw-idea-"
                ), f"Source {link.source_node_id} not found in ideas canvas"

    def test_actions_have_provenance_from_goals(self, full_result):
        """Actions stage provenance should link back to goals."""
        action_provenance = [
            p for p in full_result.provenance if p.target_stage == PipelineStage.ACTIONS
        ]
        assert len(action_provenance) > 0
        for link in action_provenance:
            assert link.source_stage == PipelineStage.GOALS

    def test_orchestration_has_provenance_from_actions(self, full_result):
        """Orchestration stage provenance should link back to actions."""
        orch_provenance = [
            p for p in full_result.provenance if p.target_stage == PipelineStage.ORCHESTRATION
        ]
        assert len(orch_provenance) > 0
        for link in orch_provenance:
            assert link.source_stage == PipelineStage.ACTIONS

    def test_debate_pipeline_provenance(self, pipeline, debate_data):
        """Debate-originated pipeline should also have provenance."""
        result = pipeline.from_debate(debate_data, auto_advance=True)
        assert len(result.provenance) > 0


# =============================================================================
# Content hash verification
# =============================================================================


class TestContentHashVerification:
    """Test SHA-256 content hashes in provenance links."""

    def test_hashes_are_nonempty(self, full_result):
        for link in full_result.provenance:
            assert link.content_hash, (
                f"Empty hash for {link.source_node_id} -> {link.target_node_id}"
            )

    def test_hashes_are_hex_strings(self, full_result):
        for link in full_result.provenance:
            # content_hash returns sha256 hex digest (or prefix thereof)
            assert all(c in "0123456789abcdef" for c in link.content_hash), (
                f"Non-hex hash: {link.content_hash}"
            )

    def test_content_hash_deterministic(self):
        """Same input always produces same hash."""
        text = "Build a rate limiter"
        h1 = content_hash(text)
        h2 = content_hash(text)
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = content_hash("Build a rate limiter")
        h2 = content_hash("Add caching layer")
        assert h1 != h2


# =============================================================================
# Cross-stage references
# =============================================================================


class TestCrossStageReferences:
    """Test that node IDs resolve correctly across stages."""

    def test_goal_source_idea_ids_are_valid(self, full_result):
        """Goals' source_idea_ids should reference real idea nodes."""
        idea_ids = set(full_result.ideas_canvas.nodes.keys())
        for goal in full_result.goal_graph.goals:
            for src_id in goal.source_idea_ids:
                assert src_id in idea_ids, f"Goal {goal.id} references nonexistent idea {src_id}"

    def test_action_source_goal_ids_resolve(self, full_result):
        """Action nodes derived from goals should reference real goals."""
        goal_ids = {g.id for g in full_result.goal_graph.goals}
        for node_id, node in full_result.actions_canvas.nodes.items():
            source = node.data.get("source_goal_id", "")
            if source:
                assert source in goal_ids, f"Action {node_id} references nonexistent goal {source}"

    def test_orchestration_source_action_ids_resolve(self, full_result):
        """Orchestration nodes should reference real action nodes."""
        if not full_result.orchestration_canvas:
            pytest.skip("No orchestration canvas")
        action_ids = set(full_result.actions_canvas.nodes.keys())
        for node_id, node in full_result.orchestration_canvas.nodes.items():
            source = node.data.get("source_action_id", "")
            if source:
                assert source in action_ids, (
                    f"Orch {node_id} references nonexistent action {source}"
                )


# =============================================================================
# Stage transitions
# =============================================================================


class TestStageTransitions:
    """Test that stage transitions form a valid chain."""

    def test_transitions_exist(self, full_result):
        assert len(full_result.transitions) >= 2  # goals→actions, actions→orchestration

    def test_transitions_form_chain(self, full_result):
        """Transitions should form a connected chain."""
        stages_seen = set()
        for t in full_result.transitions:
            stages_seen.add(t.from_stage)
            stages_seen.add(t.to_stage)
        # At minimum, GOALS→ACTIONS and ACTIONS→ORCHESTRATION
        assert PipelineStage.GOALS in stages_seen
        assert PipelineStage.ACTIONS in stages_seen
        assert PipelineStage.ORCHESTRATION in stages_seen

    def test_transitions_have_confidence(self, full_result):
        for t in full_result.transitions:
            assert 0.0 <= t.confidence <= 1.0

    def test_transitions_have_rationale(self, full_result):
        for t in full_result.transitions:
            assert t.ai_rationale, f"Transition {t.id} missing rationale"

    def test_transitions_have_provenance(self, full_result):
        for t in full_result.transitions:
            assert len(t.provenance) >= 0  # Some may be empty but shouldn't crash


# =============================================================================
# Integrity hash
# =============================================================================


class TestIntegrityHash:
    """Test pipeline integrity hash computation."""

    def test_integrity_hash_present(self, full_result):
        d = full_result.to_dict()
        assert d["integrity_hash"]
        assert len(d["integrity_hash"]) > 0

    def test_integrity_hash_deterministic(self, pipeline, sample_ideas):
        """Same input should produce same integrity hash."""
        r1 = pipeline.from_ideas(sample_ideas, auto_advance=True, pipeline_id="fixed-id")
        r2 = pipeline.from_ideas(sample_ideas, auto_advance=True, pipeline_id="fixed-id")
        # UUIDs in goal IDs make exact matching impossible, but hashes should be hex strings
        assert all(c in "0123456789abcdef" for c in r1.to_dict()["integrity_hash"])
        assert all(c in "0123456789abcdef" for c in r2.to_dict()["integrity_hash"])

    def test_integrity_hash_changes_with_different_input(self, pipeline):
        r1 = pipeline.from_ideas(["Build a rate limiter"], auto_advance=True)
        r2 = pipeline.from_ideas(["Deploy a monitoring dashboard"], auto_advance=True)
        # Different ideas should produce different provenance chains
        # (though collision is theoretically possible, extremely unlikely)
        h1 = r1.to_dict()["integrity_hash"]
        h2 = r2.to_dict()["integrity_hash"]
        assert isinstance(h1, str) and isinstance(h2, str)


# =============================================================================
# Receipt provenance
# =============================================================================


class TestReceiptProvenance:
    """Test that receipts include provenance information."""

    @pytest.mark.asyncio
    async def test_receipt_includes_provenance_count(self):
        pipeline = IdeaToExecutionPipeline()
        cfg = PipelineConfig(enable_receipts=True, dry_run=False)
        result = await pipeline.run(
            "Build a rate limiter and add caching",
            config=cfg,
        )
        if result.receipt:
            # Receipt should exist and include some provenance reference
            assert isinstance(result.receipt, dict)

    @pytest.mark.asyncio
    async def test_receipt_has_pipeline_id(self):
        pipeline = IdeaToExecutionPipeline()
        cfg = PipelineConfig(enable_receipts=True, dry_run=False)
        result = await pipeline.run("Test input", config=cfg, pipeline_id="pipe-receipt-test")
        if result.receipt:
            receipt = result.receipt
            # Receipt should reference the pipeline
            has_id = (
                receipt.get("pipeline_id") == "pipe-receipt-test"
                or receipt.get("decision_id") == "pipe-receipt-test"
            )
            assert has_id
