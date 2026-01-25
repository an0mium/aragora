"""
Tests for provenance handler endpoints.

Tests the provenance API handlers for:
- Retrieving debate provenance graphs
- Timeline views
- Chain verification
- Export functionality
- Claim provenance
- Agent contributions
"""

import pytest

from aragora.reasoning.provenance import (
    ProvenanceManager,
    SourceType,
    TransformationType,
)
from aragora.server.handlers.features.provenance import (
    _build_graph_edges,
    _build_graph_nodes,
    _compute_max_depth,
    _map_source_to_node_type,
    _truncate,
    get_provenance_manager,
    handle_export_provenance_report,
    handle_get_agent_contributions,
    handle_get_claim_provenance,
    handle_get_debate_provenance,
    handle_get_provenance_timeline,
    handle_verify_provenance_chain,
    register_provenance_manager,
)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_truncate_short_text(self):
        """Truncate returns text unchanged if short enough."""
        text = "short text"
        assert _truncate(text, 20) == text

    def test_truncate_long_text(self):
        """Truncate adds ellipsis for long text."""
        text = "this is a very long text that should be truncated"
        result = _truncate(text, 20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_truncate_exact_length(self):
        """Truncate handles exact length text."""
        text = "exactly twenty chars"
        assert _truncate(text, 20) == text

    def test_map_source_to_node_type_agent(self):
        """Map agent source type correctly."""
        assert _map_source_to_node_type(SourceType.AGENT_GENERATED) == "argument"

    def test_map_source_to_node_type_synthesis(self):
        """Map synthesis source type correctly."""
        assert _map_source_to_node_type(SourceType.SYNTHESIS) == "synthesis"

    def test_map_source_to_node_type_evidence(self):
        """Map evidence source types correctly."""
        assert _map_source_to_node_type(SourceType.WEB_SEARCH) == "evidence"
        assert _map_source_to_node_type(SourceType.DOCUMENT) == "evidence"
        assert _map_source_to_node_type(SourceType.DATABASE) == "evidence"

    def test_map_source_to_node_type_unknown(self):
        """Map unknown source type correctly."""
        assert _map_source_to_node_type(SourceType.UNKNOWN) == "unknown"


class TestProvenanceManagerStore:
    """Tests for provenance manager store functions."""

    def test_get_provenance_manager_creates_new(self):
        """get_provenance_manager creates new manager for unknown debate."""
        manager = get_provenance_manager("test-debate-123")
        assert manager is not None
        assert manager.debate_id == "test-debate-123"

    def test_get_provenance_manager_returns_cached(self):
        """get_provenance_manager returns same manager for same debate."""
        manager1 = get_provenance_manager("test-debate-456")
        manager2 = get_provenance_manager("test-debate-456")
        assert manager1 is manager2

    def test_register_provenance_manager(self):
        """register_provenance_manager stores external manager."""
        custom_manager = ProvenanceManager(debate_id="custom-debate")
        register_provenance_manager("custom-debate", custom_manager)

        retrieved = get_provenance_manager("custom-debate")
        assert retrieved is custom_manager


class TestBuildGraphNodes:
    """Tests for _build_graph_nodes function."""

    def test_empty_manager_returns_empty_list(self):
        """Empty manager returns empty nodes list."""
        manager = ProvenanceManager()
        nodes = _build_graph_nodes(manager)
        assert nodes == []

    def test_single_record_returns_node(self):
        """Single record creates one node."""
        manager = ProvenanceManager()
        manager.record_evidence(
            content="Test evidence",
            source_type=SourceType.AGENT_GENERATED,
            source_id="agent-1",
        )

        nodes = _build_graph_nodes(manager)
        assert len(nodes) == 1
        assert nodes[0]["source_id"] == "agent-1"
        assert nodes[0]["type"] == "argument"
        assert "Test evidence" in nodes[0]["label"]

    def test_node_has_required_fields(self):
        """Node contains all required fields."""
        manager = ProvenanceManager()
        manager.record_evidence(
            content="Test content",
            source_type=SourceType.WEB_SEARCH,
            source_id="search-1",
        )

        nodes = _build_graph_nodes(manager)
        node = nodes[0]

        assert "id" in node
        assert "type" in node
        assert "label" in node
        assert "content" in node
        assert "source_type" in node
        assert "source_id" in node
        assert "timestamp" in node
        assert "content_hash" in node
        assert "verified" in node
        assert "confidence" in node
        assert "transformation" in node


class TestBuildGraphEdges:
    """Tests for _build_graph_edges function."""

    def test_empty_manager_returns_empty_list(self):
        """Empty manager returns empty edges list."""
        manager = ProvenanceManager()
        edges = _build_graph_edges(manager)
        assert edges == []

    def test_chain_creates_edges(self):
        """Multiple records create chain edges."""
        manager = ProvenanceManager()
        manager.record_evidence(
            content="First",
            source_type=SourceType.USER_PROVIDED,
            source_id="user-1",
        )
        manager.record_evidence(
            content="Second",
            source_type=SourceType.AGENT_GENERATED,
            source_id="agent-1",
        )

        edges = _build_graph_edges(manager)
        chain_edges = [e for e in edges if e["type"] == "chain"]
        assert len(chain_edges) == 1

    def test_citations_create_edges(self):
        """Citations create support/contradict edges."""
        manager = ProvenanceManager()
        record = manager.record_evidence(
            content="Evidence",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
        )
        manager.cite_evidence(
            claim_id="claim-1",
            evidence_id=record.id,
            support_type="supports",
        )

        edges = _build_graph_edges(manager)
        cite_edges = [e for e in edges if e["type"] == "supports"]
        assert len(cite_edges) == 1


class TestComputeMaxDepth:
    """Tests for _compute_max_depth function."""

    def test_empty_manager_depth_zero(self):
        """Empty manager has depth 0."""
        manager = ProvenanceManager()
        assert _compute_max_depth(manager) == 0

    def test_single_record_depth_one(self):
        """Single record has depth 1."""
        manager = ProvenanceManager()
        manager.record_evidence(
            content="Test",
            source_type=SourceType.USER_PROVIDED,
            source_id="user",
        )
        assert _compute_max_depth(manager) == 1

    def test_multiple_records_depth_equals_count(self):
        """Multiple records depth equals record count."""
        manager = ProvenanceManager()
        for i in range(5):
            manager.record_evidence(
                content=f"Record {i}",
                source_type=SourceType.AGENT_GENERATED,
                source_id=f"agent-{i}",
            )
        assert _compute_max_depth(manager) == 5


@pytest.mark.asyncio
class TestHandleGetDebateProvenance:
    """Tests for handle_get_debate_provenance endpoint."""

    async def test_returns_graph_structure(self):
        """Returns complete graph structure."""
        result = await handle_get_debate_provenance("test-graph-debate")

        assert result.status == 200
        body = result.body
        assert "debate_id" in body
        assert "nodes" in body
        assert "edges" in body
        assert "metadata" in body

    async def test_metadata_contains_status(self):
        """Metadata contains status field."""
        result = await handle_get_debate_provenance("test-status-debate")

        assert result.body["metadata"]["status"] == "ready"

    async def test_returns_verification_status(self):
        """Returns chain verification status."""
        result = await handle_get_debate_provenance("test-verify-debate")

        assert "verified" in result.body["metadata"]
        assert isinstance(result.body["metadata"]["verified"], bool)


@pytest.mark.asyncio
class TestHandleGetProvenanceTimeline:
    """Tests for handle_get_provenance_timeline endpoint."""

    async def test_returns_timeline_structure(self):
        """Returns timeline structure."""
        result = await handle_get_provenance_timeline("test-timeline-debate")

        assert result.status == 200
        body = result.body
        assert "debate_id" in body
        assert "rounds" in body
        assert "agent_positions" in body

    async def test_round_filter_works(self):
        """Round filter parameter works."""
        # Create manager with records in different rounds
        manager = ProvenanceManager(debate_id="test-round-debate")
        manager.record_evidence(
            content="Round 0",
            source_type=SourceType.AGENT_GENERATED,
            source_id="agent-1",
            metadata={"round_number": 0},
        )
        manager.record_evidence(
            content="Round 1",
            source_type=SourceType.AGENT_GENERATED,
            source_id="agent-1",
            metadata={"round_number": 1},
        )
        register_provenance_manager("test-round-debate", manager)

        result = await handle_get_provenance_timeline("test-round-debate", round_number=1)

        # Should only include round 1 records
        for r in result.body["rounds"]:
            if r["round"] == 1:
                assert len(r["records"]) == 1


@pytest.mark.asyncio
class TestHandleVerifyProvenanceChain:
    """Tests for handle_verify_provenance_chain endpoint."""

    async def test_returns_verification_result(self):
        """Returns verification result structure."""
        result = await handle_verify_provenance_chain("test-chain-debate")

        assert result.status == 200
        body = result.body
        assert "chain_valid" in body
        assert "content_valid" in body
        assert "citations_complete" in body
        assert "verified_at" in body

    async def test_empty_chain_is_valid(self):
        """Empty chain is considered valid."""
        result = await handle_verify_provenance_chain("test-empty-chain-debate")

        assert result.body["chain_valid"] is True
        assert result.body["content_valid"] is True

    async def test_returns_error_list(self):
        """Returns errors list (empty for valid chain)."""
        result = await handle_verify_provenance_chain("test-errors-debate")

        assert "errors" in result.body
        assert isinstance(result.body["errors"], list)


@pytest.mark.asyncio
class TestHandleExportProvenanceReport:
    """Tests for handle_export_provenance_report endpoint."""

    async def test_json_format_includes_records(self):
        """JSON format includes records when include_evidence is True."""
        manager = ProvenanceManager(debate_id="test-export-json")
        manager.record_evidence(
            content="Export test",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
        )
        register_provenance_manager("test-export-json", manager)

        result = await handle_export_provenance_report(
            "test-export-json", format="json", include_evidence=True
        )

        assert "records" in result.body
        assert len(result.body["records"]) == 1

    async def test_audit_format_includes_hash_chain(self):
        """Audit format includes hash chain."""
        result = await handle_export_provenance_report("test-export-audit", format="audit")

        assert "hash_chain" in result.body

    async def test_summary_format_includes_summary(self):
        """Summary format includes summary object."""
        result = await handle_export_provenance_report("test-export-summary", format="summary")

        assert "summary" in result.body
        assert "total_evidence_pieces" in result.body["summary"]

    async def test_returns_verification_status(self):
        """Returns verification status."""
        result = await handle_export_provenance_report("test-export-status")

        assert "verification_status" in result.body
        assert result.body["verification_status"] in ["verified", "failed"]


@pytest.mark.asyncio
class TestHandleGetClaimProvenance:
    """Tests for handle_get_claim_provenance endpoint."""

    async def test_returns_claim_structure(self):
        """Returns claim provenance structure."""
        result = await handle_get_claim_provenance("test-claim-debate", "claim-1")

        assert result.status == 200
        body = result.body
        assert "debate_id" in body
        assert "claim_id" in body
        assert "supporting_evidence" in body
        assert "contradicting_evidence" in body
        assert "verification_status" in body

    async def test_includes_evidence_with_provenance(self):
        """Includes evidence with provenance details."""
        manager = ProvenanceManager(debate_id="test-claim-evidence")
        record = manager.record_evidence(
            content="Supporting evidence",
            source_type=SourceType.DOCUMENT,
            source_id="doc-1",
        )
        manager.cite_evidence(
            claim_id="claim-123",
            evidence_id=record.id,
            support_type="supports",
            relevance=0.9,
        )
        register_provenance_manager("test-claim-evidence", manager)

        result = await handle_get_claim_provenance("test-claim-evidence", "claim-123")

        assert len(result.body["supporting_evidence"]) == 1
        evidence = result.body["supporting_evidence"][0]
        assert "relevance" in evidence
        assert "content_hash" in evidence


@pytest.mark.asyncio
class TestHandleGetAgentContributions:
    """Tests for handle_get_agent_contributions endpoint."""

    async def test_returns_contributions_structure(self):
        """Returns contributions structure."""
        result = await handle_get_agent_contributions("test-contrib-debate")

        assert result.status == 200
        body = result.body
        assert "debate_id" in body
        assert "contributions" in body
        assert "summary" in body

    async def test_groups_by_agent(self):
        """Groups contributions by agent."""
        manager = ProvenanceManager(debate_id="test-agent-group")
        manager.record_evidence(
            content="Agent 1 contribution",
            source_type=SourceType.AGENT_GENERATED,
            source_id="agent-1",
        )
        manager.record_evidence(
            content="Agent 2 contribution",
            source_type=SourceType.AGENT_GENERATED,
            source_id="agent-2",
        )
        register_provenance_manager("test-agent-group", manager)

        result = await handle_get_agent_contributions("test-agent-group")

        assert len(result.body["contributions"]) == 2

    async def test_filters_by_agent_id(self):
        """Filters contributions by agent_id."""
        manager = ProvenanceManager(debate_id="test-agent-filter")
        manager.record_evidence(
            content="Agent 1",
            source_type=SourceType.AGENT_GENERATED,
            source_id="agent-1",
        )
        manager.record_evidence(
            content="Agent 2",
            source_type=SourceType.AGENT_GENERATED,
            source_id="agent-2",
        )
        register_provenance_manager("test-agent-filter", manager)

        result = await handle_get_agent_contributions("test-agent-filter", agent_id="agent-1")

        assert len(result.body["contributions"]) == 1
        assert result.body["contributions"][0]["agent_id"] == "agent-1"

    async def test_summary_contains_statistics(self):
        """Summary contains contribution statistics."""
        result = await handle_get_agent_contributions("test-summary-stats")

        summary = result.body["summary"]
        assert "total_arguments" in summary
        assert "total_evidence" in summary
        assert "total_syntheses" in summary
        assert "unique_agents" in summary
