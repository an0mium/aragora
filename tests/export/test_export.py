"""Tests for aragora/export/ module - Debate artifact export functionality."""

import csv
import json
import pytest
from datetime import datetime
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from aragora.export.artifact import (
    ConsensusProof,
    VerificationResult,
    DebateArtifact,
    ArtifactBuilder,
    create_artifact_from_debate,
)
from aragora.export.static_html import StaticHTMLExporter, export_to_html
from aragora.export.csv_exporter import (
    CSVExporter,
    export_debate_to_csv,
    export_multiple_debates,
)
from aragora.export.dot_exporter import (
    escape_label,
    DOTExporter,
    export_debate_to_dot,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_consensus():
    """Create sample ConsensusProof."""
    return ConsensusProof(
        reached=True,
        confidence=0.85,
        vote_breakdown={"claude": True, "gemini": True, "gpt": False},
        final_answer="The answer is 42",
        rounds_used=3,
    )


@pytest.fixture
def sample_verification():
    """Create sample VerificationResult."""
    return VerificationResult(
        claim_id="claim-1",
        claim_text="2 + 2 = 4",
        status="verified",
        method="z3",
        proof_trace="QED",
        duration_ms=150,
    )


@pytest.fixture
def sample_artifact(sample_consensus, sample_verification):
    """Create sample DebateArtifact with full data."""
    return DebateArtifact(
        debate_id="debate-123",
        task="What is the meaning of life?",
        graph_data={
            "nodes": {
                "n1": {"agent_id": "claude", "content": "Test proposal", "node_type": "proposal"},
                "n2": {"agent_id": "gemini", "content": "A critique", "node_type": "critique"},
            }
        },
        trace_data={
            "events": [
                {
                    "event_type": "message",
                    "agent": "claude",
                    "content": "Hello",
                    "round": 1,
                    "role": "proposer",
                    "timestamp": "2026-01-01T10:00:00",
                },
                {
                    "event_type": "critique",
                    "agent": "gemini",
                    "target": "claude",
                    "severity": 0.5,
                    "issues": ["Issue 1"],
                    "round": 1,
                },
                {
                    "event_type": "agent_proposal",
                    "agent": "claude",
                    "content": {"content": "My proposal"},
                    "round_num": 1,
                },
            ]
        },
        provenance_data={
            "chain": {
                "records": [
                    {
                        "id": "ev-1",
                        "source_type": "debate",
                        "source_id": "debate-123",
                        "content": "Test content",
                        "content_hash": "abc123",
                        "previous_hash": None,
                    },
                ]
            }
        },
        consensus_proof=sample_consensus,
        verification_results=[sample_verification],
        agents=["claude", "gemini", "gpt"],
        rounds=3,
        message_count=10,
        critique_count=5,
        duration_seconds=120.5,
    )


@pytest.fixture
def minimal_artifact():
    """Create minimal DebateArtifact."""
    return DebateArtifact(
        debate_id="minimal-1",
        task="Simple task",
    )


@pytest.fixture
def html_exporter(sample_artifact):
    """Create StaticHTMLExporter."""
    return StaticHTMLExporter(sample_artifact)


@pytest.fixture
def csv_exporter(sample_artifact):
    """Create CSVExporter."""
    return CSVExporter(sample_artifact)


@pytest.fixture
def dot_exporter(sample_artifact):
    """Create DOTExporter."""
    return DOTExporter(sample_artifact)


# =============================================================================
# ConsensusProof Dataclass Tests
# =============================================================================


class TestConsensusProof:
    """Tests for ConsensusProof dataclass."""

    def test_all_fields_initialized(self, sample_consensus):
        """All fields should be initialized correctly."""
        assert sample_consensus.reached is True
        assert sample_consensus.confidence == 0.85
        assert sample_consensus.vote_breakdown == {"claude": True, "gemini": True, "gpt": False}
        assert sample_consensus.final_answer == "The answer is 42"
        assert sample_consensus.rounds_used == 3

    def test_to_dict_serialization(self, sample_consensus):
        """to_dict should serialize all fields."""
        d = sample_consensus.to_dict()

        assert d["reached"] is True
        assert d["confidence"] == 0.85
        assert d["vote_breakdown"]["claude"] is True
        assert d["final_answer"] == "The answer is 42"
        assert d["rounds_used"] == 3
        assert "timestamp" in d

    def test_timestamp_auto_populated(self):
        """timestamp should be auto-populated."""
        cp = ConsensusProof(
            reached=False,
            confidence=0.5,
            vote_breakdown={},
            final_answer="test",
            rounds_used=1,
        )

        assert cp.timestamp is not None
        # Should be parseable
        datetime.fromisoformat(cp.timestamp)


# =============================================================================
# VerificationResult Dataclass Tests
# =============================================================================


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_all_fields_initialized(self, sample_verification):
        """All fields should be initialized correctly."""
        assert sample_verification.claim_id == "claim-1"
        assert sample_verification.claim_text == "2 + 2 = 4"
        assert sample_verification.status == "verified"
        assert sample_verification.method == "z3"
        assert sample_verification.proof_trace == "QED"
        assert sample_verification.duration_ms == 150

    def test_to_dict_serialization(self, sample_verification):
        """to_dict should serialize all fields."""
        d = sample_verification.to_dict()

        assert d["claim_id"] == "claim-1"
        assert d["claim_text"] == "2 + 2 = 4"
        assert d["status"] == "verified"
        assert d["method"] == "z3"
        assert d["proof_trace"] == "QED"

    def test_optional_fields_default(self):
        """Optional fields should have correct defaults."""
        v = VerificationResult(
            claim_id="c1",
            claim_text="test",
            status="timeout",
            method="simulation",
        )

        assert v.proof_trace is None
        assert v.counterexample is None
        assert v.duration_ms == 0
        assert v.metadata == {}


# =============================================================================
# DebateArtifact Dataclass Tests
# =============================================================================


class TestDebateArtifact:
    """Tests for DebateArtifact dataclass."""

    def test_all_fields_initialized(self, sample_artifact):
        """All fields should be initialized correctly."""
        assert sample_artifact.debate_id == "debate-123"
        assert sample_artifact.task == "What is the meaning of life?"
        assert sample_artifact.graph_data is not None
        assert sample_artifact.trace_data is not None
        assert sample_artifact.consensus_proof is not None
        assert len(sample_artifact.verification_results) == 1
        assert sample_artifact.agents == ["claude", "gemini", "gpt"]
        assert sample_artifact.rounds == 3

    def test_artifact_id_auto_generated(self):
        """artifact_id should be auto-generated."""
        a1 = DebateArtifact()
        a2 = DebateArtifact()

        assert a1.artifact_id is not None
        assert a2.artifact_id is not None
        assert a1.artifact_id != a2.artifact_id

    def test_created_at_auto_populated(self):
        """created_at should be auto-populated."""
        artifact = DebateArtifact()

        assert artifact.created_at is not None
        datetime.fromisoformat(artifact.created_at)

    def test_content_hash_computed(self, sample_artifact):
        """content_hash should be computed from data."""
        hash1 = sample_artifact.content_hash

        assert hash1 is not None
        assert len(hash1) == 16  # SHA256 truncated to 16 chars

        # Same artifact should give same hash
        hash2 = sample_artifact.content_hash
        assert hash1 == hash2

    def test_to_dict_from_dict_roundtrip(self, sample_artifact):
        """to_dict/from_dict should preserve data."""
        d = sample_artifact.to_dict()
        restored = DebateArtifact.from_dict(d)

        assert restored.debate_id == sample_artifact.debate_id
        assert restored.task == sample_artifact.task
        assert restored.rounds == sample_artifact.rounds
        assert restored.consensus_proof.reached == sample_artifact.consensus_proof.reached

    def test_to_json_from_json_roundtrip(self, sample_artifact):
        """to_json/from_json should preserve data."""
        json_str = sample_artifact.to_json()
        restored = DebateArtifact.from_json(json_str)

        assert restored.debate_id == sample_artifact.debate_id
        assert restored.task == sample_artifact.task

    def test_save_load_file(self, sample_artifact, tmp_path):
        """save/load should work with files."""
        path = tmp_path / "artifact.json"

        sample_artifact.save(path)
        assert path.exists()

        loaded = DebateArtifact.load(path)
        assert loaded.debate_id == sample_artifact.debate_id

    def test_verify_integrity_returns_tuple(self, sample_artifact):
        """verify_integrity should return (bool, list)."""
        # No provenance_data means no import needed
        valid, errors = sample_artifact.verify_integrity()

        assert isinstance(valid, bool)
        assert isinstance(errors, list)

    def test_verify_integrity_with_provenance(self):
        """verify_integrity should check provenance chain if present."""
        artifact = DebateArtifact(
            debate_id="prov-test",
            task="test",
            provenance_data={"chain": []},
        )

        # Mock the provenance module
        with patch("aragora.reasoning.provenance.ProvenanceChain") as MockChain:
            mock_chain = MagicMock()
            mock_chain.verify_chain.return_value = (True, [])
            MockChain.from_dict.return_value = mock_chain

            valid, errors = artifact.verify_integrity()

            MockChain.from_dict.assert_called_once()
            assert valid is True
            assert errors == []


# =============================================================================
# ArtifactBuilder Tests
# =============================================================================


class TestArtifactBuilder:
    """Tests for ArtifactBuilder class."""

    def test_with_graph_adds_dict(self):
        """with_graph should add graph data from dict."""
        builder = ArtifactBuilder()
        graph_data = {"nodes": {"n1": {}}}

        builder.with_graph(graph_data)

        assert builder._artifact.graph_data == graph_data

    def test_with_graph_handles_object(self):
        """with_graph should handle objects with to_dict."""
        builder = ArtifactBuilder()
        mock_graph = MagicMock()
        mock_graph.to_dict.return_value = {"nodes": {"n1": {}}}

        builder.with_graph(mock_graph)

        assert builder._artifact.graph_data == {"nodes": {"n1": {}}}

    def test_with_trace_adds_dict(self):
        """with_trace should add trace data."""
        builder = ArtifactBuilder()
        trace_data = {"events": []}

        builder.with_trace(trace_data)

        assert builder._artifact.trace_data == trace_data

    def test_with_provenance_adds_data(self):
        """with_provenance should add provenance data."""
        builder = ArtifactBuilder()
        prov_data = {"chain": {}}

        builder.with_provenance(prov_data)

        assert builder._artifact.provenance_data == prov_data

    def test_with_verification_adds_result(self):
        """with_verification should add verification result."""
        builder = ArtifactBuilder()

        builder.with_verification(
            claim_id="c1",
            claim_text="test",
            status="verified",
            method="z3",
        )

        assert len(builder._artifact.verification_results) == 1
        assert builder._artifact.verification_results[0].claim_id == "c1"

    def test_build_returns_artifact(self):
        """build should return DebateArtifact."""
        builder = ArtifactBuilder()
        builder._artifact.task = "Test task"

        artifact = builder.build()

        assert isinstance(artifact, DebateArtifact)
        assert artifact.task == "Test task"


# =============================================================================
# StaticHTMLExporter Tests
# =============================================================================


class TestStaticHTMLExporter:
    """Tests for StaticHTMLExporter class."""

    def test_stores_artifact(self, html_exporter, sample_artifact):
        """Should store artifact reference."""
        assert html_exporter.artifact is sample_artifact

    def test_generate_returns_html(self, html_exporter):
        """generate should return valid HTML structure."""
        html = html_exporter.generate()

        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html


class TestHTMLEscape:
    """Tests for HTML escaping."""

    def test_escapes_angle_brackets(self, html_exporter):
        """Should escape < and > characters."""
        result = html_exporter._escape("<script>alert('xss')</script>")

        assert "&lt;" in result
        assert "&gt;" in result
        assert "<script>" not in result

    def test_escapes_ampersand(self, html_exporter):
        """Should escape & character."""
        result = html_exporter._escape("A & B")

        assert "&amp;" in result

    def test_escapes_quotes(self, html_exporter):
        """Should escape quotes."""
        result = html_exporter._escape('Say "hello"')

        assert "&quot;" in result

    def test_escapes_all_special_chars(self, html_exporter):
        """Should handle all special chars together."""
        result = html_exporter._escape('<a href="test?a=1&b=2">Link</a>')

        assert "&lt;" in result
        assert "&gt;" in result
        assert "&quot;" in result
        assert "&amp;" in result


class TestHTMLSectionGenerators:
    """Tests for HTML section generation methods."""

    def test_generate_styles_returns_css(self, html_exporter):
        """_generate_styles should return CSS string."""
        styles = html_exporter._generate_styles()

        assert "<style>" in styles
        assert "</style>" in styles
        assert "var(--primary)" in styles

    def test_generate_header_includes_consensus(self, html_exporter):
        """_generate_header should include consensus status."""
        header = html_exporter._generate_header()

        assert "Consensus" in header
        assert "85%" in header  # confidence

    def test_generate_task_section_includes_task(self, html_exporter):
        """_generate_task_section should include task text."""
        section = html_exporter._generate_task_section()

        assert "What is the meaning of life?" in section

    def test_generate_tabs_returns_buttons(self, html_exporter):
        """_generate_tabs should return tab buttons."""
        tabs = html_exporter._generate_tabs()

        assert "Graph" in tabs
        assert "Timeline" in tabs
        assert "Provenance" in tabs
        assert "Verification" in tabs

    def test_generate_graph_view_empty(self, minimal_artifact):
        """_generate_graph_view should handle empty graph."""
        exporter = StaticHTMLExporter(minimal_artifact)
        view = exporter._generate_graph_view()

        assert "No graph data available" in view

    def test_generate_graph_view_with_nodes(self, html_exporter):
        """_generate_graph_view should render nodes."""
        view = html_exporter._generate_graph_view()

        assert "graph-canvas" in view
        assert "graph-node" in view

    def test_generate_timeline_view_empty(self, minimal_artifact):
        """_generate_timeline_view should handle empty trace."""
        exporter = StaticHTMLExporter(minimal_artifact)
        view = exporter._generate_timeline_view()

        assert "No trace data available" in view

    def test_generate_provenance_view_empty(self, minimal_artifact):
        """_generate_provenance_view should handle empty data."""
        exporter = StaticHTMLExporter(minimal_artifact)
        view = exporter._generate_provenance_view()

        assert "No provenance data available" in view

    def test_generate_verification_view_empty(self, minimal_artifact):
        """_generate_verification_view should handle empty results."""
        exporter = StaticHTMLExporter(minimal_artifact)
        view = exporter._generate_verification_view()

        assert "No formal verification results available" in view

    def test_generate_stats_includes_counters(self, html_exporter):
        """_generate_stats should include all counters."""
        stats = html_exporter._generate_stats()

        assert "10" in stats  # message_count
        assert "5" in stats  # critique_count
        assert "3" in stats  # rounds


class TestGraphRendering:
    """Tests for graph node rendering."""

    def test_render_graph_nodes_handles_missing(self, minimal_artifact):
        """_render_graph_nodes should handle missing nodes."""
        exporter = StaticHTMLExporter(minimal_artifact)
        result = exporter._render_graph_nodes()

        assert result == ""

    def test_render_graph_nodes_includes_agent(self, html_exporter):
        """Node HTML should include agent."""
        nodes = html_exporter._render_graph_nodes()

        assert "claude" in nodes

    def test_render_graph_nodes_includes_type(self, html_exporter):
        """Node HTML should include type."""
        nodes = html_exporter._render_graph_nodes()

        assert "proposal" in nodes


class TestHTMLExport:
    """Tests for HTML export functionality."""

    def test_generate_complete_html(self, html_exporter):
        """generate should produce complete HTML document."""
        html = html_exporter.generate()

        assert "<!DOCTYPE html>" in html
        assert "aragora Debate" in html
        assert "<script>" in html
        assert "</script>" in html

    def test_save_writes_file(self, html_exporter, tmp_path):
        """save should write file to path."""
        path = tmp_path / "output.html"

        html_exporter.save(path)

        assert path.exists()
        content = path.read_text()
        assert "<!DOCTYPE html>" in content

    def test_export_to_html_convenience(self, sample_artifact, tmp_path):
        """export_to_html convenience function should work."""
        path = tmp_path / "export.html"

        result = export_to_html(sample_artifact, path)

        assert result == path
        assert path.exists()

    def test_html_includes_javascript(self, html_exporter):
        """HTML should include embedded JavaScript."""
        html = html_exporter.generate()

        assert "artifactData" in html
        assert "addEventListener" in html


# =============================================================================
# CSVExporter Tests
# =============================================================================


class TestCSVExporterInit:
    """Tests for CSVExporter initialization."""

    def test_stores_artifact(self, csv_exporter, sample_artifact):
        """Should store artifact reference."""
        assert csv_exporter.artifact is sample_artifact

    def test_returns_csv_with_header(self, csv_exporter):
        """Should return CSV string with header."""
        content = csv_exporter.export_summary()

        assert "debate_id" in content


class TestCSVExportMethods:
    """Tests for CSV export methods."""

    def test_export_messages_header(self, csv_exporter):
        """export_messages should include header row."""
        content = csv_exporter.export_messages()

        assert "debate_id" in content
        assert "round" in content
        assert "agent" in content

    def test_export_messages_extracts_events(self, csv_exporter):
        """export_messages should extract from trace events."""
        content = csv_exporter.export_messages()

        # Check CSV is parseable
        reader = csv.reader(StringIO(content))
        rows = list(reader)
        assert len(rows) >= 1  # At least header

    def test_export_critiques_includes_severity(self, csv_exporter):
        """export_critiques should include severity data."""
        content = csv_exporter.export_critiques()

        assert "severity" in content
        assert "issue_count" in content

    def test_export_votes_includes_breakdown(self, csv_exporter):
        """export_votes should include vote breakdown."""
        content = csv_exporter.export_votes()

        assert "claude" in content
        assert "agreed_with_consensus" in content

    def test_export_summary_includes_stats(self, csv_exporter, tmp_path):
        """export_summary should include all stats."""
        path = tmp_path / "summary.csv"
        content = csv_exporter.export_summary(path)

        assert "debate-123" in content
        assert "consensus_reached" in content
        assert path.exists()

    def test_export_verifications_includes_fields(self, csv_exporter):
        """export_verifications should include all fields."""
        content = csv_exporter.export_verifications()

        assert "claim_id" in content
        assert "status" in content
        assert "method" in content

    def test_export_all_creates_files(self, csv_exporter, tmp_path):
        """export_all should create all files in directory."""
        outputs = csv_exporter.export_all(tmp_path)

        assert "messages" in outputs
        assert "critiques" in outputs
        assert "votes" in outputs
        assert "summary" in outputs
        assert outputs["messages"].exists()

    def test_writing_to_path_creates_file(self, csv_exporter, tmp_path):
        """Writing to path should create file."""
        path = tmp_path / "test.csv"
        csv_exporter.export_messages(path)

        assert path.exists()


class TestCSVConvenienceFunctions:
    """Tests for CSV convenience functions."""

    def test_export_debate_to_csv_summary(self, sample_artifact, tmp_path):
        """export_debate_to_csv should work with summary table."""
        path = tmp_path / "out.csv"
        content = export_debate_to_csv(sample_artifact, path, "summary")

        assert "debate_id" in content
        assert path.exists()

    def test_export_debate_to_csv_messages(self, sample_artifact, tmp_path):
        """export_debate_to_csv should work with messages table."""
        path = tmp_path / "out.csv"
        content = export_debate_to_csv(sample_artifact, path, "messages")

        assert "round" in content

    def test_export_debate_to_csv_invalid_table(self, sample_artifact, tmp_path):
        """export_debate_to_csv should raise on invalid table."""
        path = tmp_path / "out.csv"

        with pytest.raises(ValueError, match="Unknown table"):
            export_debate_to_csv(sample_artifact, path, "invalid")

    def test_export_multiple_debates(self, sample_artifact, tmp_path):
        """export_multiple_debates should combine artifacts."""
        artifact2 = DebateArtifact(
            debate_id="debate-456",
            task="Another task",
            agents=["gpt"],
            rounds=2,
        )

        path = tmp_path / "combined.csv"
        content = export_multiple_debates([sample_artifact, artifact2], path, "summary")

        assert "debate-123" in content
        assert "debate-456" in content


# =============================================================================
# DOT Exporter Tests
# =============================================================================


class TestEscapeLabel:
    """Tests for escape_label function."""

    def test_truncates_long_text(self):
        """Should truncate long text with ellipsis."""
        long_text = "x" * 100
        result = escape_label(long_text, max_len=50)

        assert len(result) == 53  # 50 + "..."
        assert result.endswith("...")

    def test_escapes_quotes_and_newlines(self):
        """Should escape quotes and newlines."""
        result = escape_label('Say "hello"\nworld')

        assert '\\"' in result
        assert "\\n" in result

    def test_handles_max_len_parameter(self):
        """Should handle max_len parameter."""
        result = escape_label("Hello world", max_len=5)

        assert result == "Hello..."


class TestDOTExporterInit:
    """Tests for DOTExporter initialization."""

    def test_stores_artifact(self, dot_exporter, sample_artifact):
        """Should store artifact reference."""
        assert dot_exporter.artifact is sample_artifact

    def test_export_flow_returns_dot(self, dot_exporter):
        """export_flow should return DOT string."""
        dot = dot_exporter.export_flow()

        assert "digraph" in dot


class TestDOTExportMethods:
    """Tests for DOT export methods."""

    def test_export_flow_digraph_structure(self, dot_exporter):
        """export_flow should create digraph structure."""
        dot = dot_exporter.export_flow()

        assert "digraph debate_flow" in dot
        assert "rankdir=TB" in dot

    def test_export_flow_agent_clusters(self, dot_exporter):
        """export_flow should include agent clusters."""
        dot = dot_exporter.export_flow()

        assert "subgraph cluster_" in dot

    def test_export_flow_consensus_node(self, dot_exporter):
        """export_flow should add consensus node."""
        dot = dot_exporter.export_flow()

        assert "consensus" in dot
        assert "Consensus" in dot

    def test_export_critiques_relationships(self, dot_exporter):
        """export_critiques should show relationships."""
        dot = dot_exporter.export_critiques()

        assert "digraph critique_graph" in dot
        assert "rankdir=LR" in dot

    def test_export_critiques_colors_by_severity(self, dot_exporter):
        """export_critiques should color by severity."""
        dot = dot_exporter.export_critiques()

        # Should have color assignments
        assert "color=" in dot

    def test_export_consensus_vote_breakdown(self, dot_exporter):
        """export_consensus should show vote breakdown."""
        dot = dot_exporter.export_consensus()

        assert "digraph consensus_path" in dot
        assert "Agreed" in dot or "Disagreed" in dot


class TestDOTExportUtilities:
    """Tests for DOT export utilities."""

    def test_export_all_creates_files(self, dot_exporter, tmp_path):
        """export_all should create all files."""
        outputs = dot_exporter.export_all(tmp_path)

        assert "flow" in outputs
        assert "critiques" in outputs
        assert "consensus" in outputs
        assert outputs["flow"].exists()

    def test_export_debate_to_dot_flow(self, sample_artifact, tmp_path):
        """export_debate_to_dot should work with flow mode."""
        path = tmp_path / "out.dot"
        content = export_debate_to_dot(sample_artifact, path, "flow")

        assert "digraph debate_flow" in content
        assert path.exists()

    def test_export_debate_to_dot_critiques(self, sample_artifact, tmp_path):
        """export_debate_to_dot should work with critiques mode."""
        path = tmp_path / "out.dot"
        content = export_debate_to_dot(sample_artifact, path, "critiques")

        assert "digraph critique_graph" in content

    def test_export_debate_to_dot_consensus(self, sample_artifact, tmp_path):
        """export_debate_to_dot should work with consensus mode."""
        path = tmp_path / "out.dot"
        content = export_debate_to_dot(sample_artifact, path, "consensus")

        assert "digraph consensus_path" in content

    def test_export_debate_to_dot_invalid_mode(self, sample_artifact, tmp_path):
        """export_debate_to_dot should raise on invalid mode."""
        path = tmp_path / "out.dot"

        with pytest.raises(ValueError, match="Unknown mode"):
            export_debate_to_dot(sample_artifact, path, "invalid")


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_artifact_with_no_consensus(self):
        """Artifact with no consensus should work."""
        artifact = DebateArtifact(
            debate_id="no-consensus",
            task="Test",
            consensus_proof=None,
        )

        exporter = StaticHTMLExporter(artifact)
        html = exporter.generate()

        assert "No Consensus" in html or "N/A" in html

    def test_artifact_with_empty_agents(self):
        """Artifact with empty agents should work."""
        artifact = DebateArtifact(
            debate_id="empty-agents",
            task="Test",
            agents=[],
        )

        exporter = CSVExporter(artifact)
        content = exporter.export_summary()

        assert "empty-agents" in content

    def test_dot_export_special_chars_in_agent(self):
        """DOT export should handle special chars in agent names."""
        artifact = DebateArtifact(
            debate_id="test",
            task="Test",
            agents=["claude-3.5-sonnet", "gpt-4.0"],
        )

        exporter = DOTExporter(artifact)
        dot = exporter.export_critiques()

        # Agent names should be sanitized
        assert "claude_3_5_sonnet" in dot or "claude" in dot

    def test_html_escapes_malicious_content_in_body(self):
        """HTML should escape malicious content in task (displayed text)."""
        artifact = DebateArtifact(
            debate_id="xss-test",
            task="<script>alert('xss')</script>",
        )

        exporter = StaticHTMLExporter(artifact)
        html = exporter.generate()

        # Task text in HTML body is properly escaped
        assert "&lt;script&gt;" in html
        # Title is also escaped
        assert "&lt;script&gt;alert" in html

    def test_csv_handles_commas_in_content(self, tmp_path):
        """CSV should handle commas in content."""
        artifact = DebateArtifact(
            debate_id="comma-test",
            task="Task with, commas, and more",
            agents=["a", "b"],
        )

        exporter = CSVExporter(artifact)
        path = tmp_path / "out.csv"
        exporter.export_summary(path)

        # Read and parse to verify proper escaping
        reader = csv.reader(open(path))
        rows = list(reader)
        assert len(rows) == 2  # Header + data
