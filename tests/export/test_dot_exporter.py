"""
Tests for aragora.export.dot_exporter module.

Tests cover:
- DOTExporter initialization
- Flow graph export
- Critique graph export
- Consensus graph export
- Batch export (export_all)
- Helper functions (escape_label)
- Convenience function (export_debate_to_dot)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from aragora.export.artifact import (
    ConsensusProof,
    DebateArtifact,
)
from aragora.export.dot_exporter import (
    DOTExporter,
    escape_label,
    export_debate_to_dot,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_artifact() -> DebateArtifact:
    """Create a basic debate artifact for testing."""
    return DebateArtifact(
        artifact_id="test-artifact-001",
        debate_id="debate-001",
        task="Analyze the security of the API",
        agents=["claude", "gpt-4", "gemini"],
        rounds=3,
    )


@pytest.fixture
def artifact_with_trace() -> DebateArtifact:
    """Create an artifact with trace data for flow visualization."""
    return DebateArtifact(
        artifact_id="test-artifact-002",
        debate_id="debate-002",
        task="Review code quality",
        agents=["claude", "gpt-4"],
        rounds=2,
        trace_data={
            "events": [
                {
                    "event_type": "message",
                    "round": 1,
                    "agent": "claude",
                    "content": "I propose we focus on input validation first.",
                },
                {
                    "event_type": "message",
                    "round": 1,
                    "agent": "gpt-4",
                    "content": "Good point, but we should also consider rate limiting.",
                },
                {
                    "event_type": "critique",
                    "round": 1,
                    "agent": "gpt-4",
                    "target": "claude",
                    "severity": 0.6,
                    "issues": ["Missing edge cases"],
                },
                {
                    "event_type": "message",
                    "round": 2,
                    "agent": "claude",
                    "content": "Incorporating feedback on edge cases.",
                },
                {
                    "event_type": "critique",
                    "round": 2,
                    "agent": "claude",
                    "target": "gpt-4",
                    "severity": 0.3,
                    "issues": ["Minor style issue"],
                },
            ]
        },
    )


@pytest.fixture
def artifact_with_consensus() -> DebateArtifact:
    """Create an artifact with consensus proof for visualization."""
    return DebateArtifact(
        artifact_id="test-artifact-003",
        debate_id="debate-003",
        task="Decide on API design approach",
        agents=["claude", "gpt-4", "gemini"],
        rounds=3,
        consensus_proof=ConsensusProof(
            reached=True,
            confidence=0.85,
            vote_breakdown={
                "claude": True,
                "gpt-4": True,
                "gemini": False,
            },
            final_answer="Use REST API with GraphQL for complex queries",
            rounds_used=3,
        ),
    )


@pytest.fixture
def artifact_no_consensus() -> DebateArtifact:
    """Create an artifact without consensus."""
    return DebateArtifact(
        artifact_id="test-artifact-004",
        debate_id="debate-004",
        task="Debate architecture",
        agents=["claude", "gpt-4"],
        consensus_proof=ConsensusProof(
            reached=False,
            confidence=0.45,
            vote_breakdown={
                "claude": True,
                "gpt-4": False,
            },
            final_answer="No agreement reached",
            rounds_used=5,
        ),
    )


# =============================================================================
# TestEscapeLabel
# =============================================================================


class TestEscapeLabel:
    """Tests for escape_label helper function."""

    def test_truncates_long_text(self):
        """Should truncate text longer than max_len."""
        long_text = "A" * 100
        result = escape_label(long_text, max_len=50)

        assert len(result) == 53  # 50 chars + "..."
        assert result.endswith("...")

    def test_preserves_short_text(self):
        """Should preserve text shorter than max_len."""
        short_text = "Hello world"
        result = escape_label(short_text, max_len=50)

        assert result == short_text

    def test_escapes_quotes(self):
        """Should escape double quotes."""
        text = 'Text with "quotes"'
        result = escape_label(text)

        assert '\\"' in result
        assert '"quotes"' not in result

    def test_escapes_newlines(self):
        """Should escape newlines."""
        text = "Line 1\nLine 2"
        result = escape_label(text)

        assert "\\n" in result
        assert "\n" not in result

    def test_handles_empty_string(self):
        """Should handle empty string."""
        result = escape_label("")
        assert result == ""

    def test_custom_max_len(self):
        """Should respect custom max_len."""
        text = "A" * 20
        result = escape_label(text, max_len=10)

        assert len(result) == 13  # 10 + "..."


# =============================================================================
# TestDOTExporterInit
# =============================================================================


class TestDOTExporterInit:
    """Tests for DOTExporter initialization."""

    def test_init_with_artifact(self, basic_artifact: DebateArtifact):
        """Should initialize with a DebateArtifact."""
        exporter = DOTExporter(basic_artifact)
        assert exporter.artifact is basic_artifact

    def test_stores_artifact_reference(self, basic_artifact: DebateArtifact):
        """Should store artifact reference for later use."""
        exporter = DOTExporter(basic_artifact)
        assert exporter.artifact.artifact_id == "test-artifact-001"


# =============================================================================
# TestDOTExporterFlow
# =============================================================================


class TestDOTExporterFlow:
    """Tests for DOTExporter.export_flow()."""

    def test_returns_dot_string(self, artifact_with_trace: DebateArtifact):
        """Should return a valid DOT string."""
        exporter = DOTExporter(artifact_with_trace)
        result = exporter.export_flow()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_digraph_declaration(self, artifact_with_trace: DebateArtifact):
        """Should include digraph declaration."""
        exporter = DOTExporter(artifact_with_trace)
        result = exporter.export_flow()

        assert "digraph debate_flow {" in result
        assert result.strip().endswith("}")

    def test_includes_agent_subgraphs(self, artifact_with_trace: DebateArtifact):
        """Should include subgraphs for each agent."""
        exporter = DOTExporter(artifact_with_trace)
        result = exporter.export_flow()

        assert "subgraph cluster_claude {" in result
        assert "subgraph cluster_gpt_4 {" in result

    def test_includes_message_nodes(self, artifact_with_trace: DebateArtifact):
        """Should include nodes for messages."""
        exporter = DOTExporter(artifact_with_trace)
        result = exporter.export_flow()

        assert "msg_1 [" in result
        assert "msg_2 [" in result
        assert "msg_3 [" in result

    def test_includes_edges_between_messages(self, artifact_with_trace: DebateArtifact):
        """Should include edges between sequential messages."""
        exporter = DOTExporter(artifact_with_trace)
        result = exporter.export_flow()

        assert "msg_1 -> msg_2" in result
        assert "msg_2 -> msg_3" in result

    def test_includes_consensus_node(self, artifact_with_trace: DebateArtifact):
        """Should include consensus node when consensus proof exists."""
        artifact_with_trace.consensus_proof = ConsensusProof(
            reached=True,
            confidence=0.9,
            vote_breakdown={"claude": True, "gpt-4": True},
            final_answer="Agreed on approach",
            rounds_used=2,
        )

        exporter = DOTExporter(artifact_with_trace)
        result = exporter.export_flow()

        assert "consensus [" in result
        assert "Consensus" in result
        assert "90%" in result

    def test_writes_to_file(self, artifact_with_trace: DebateArtifact):
        """Should write DOT to file when path provided."""
        exporter = DOTExporter(artifact_with_trace)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "flow.dot"
            result = exporter.export_flow(output_path)

            assert output_path.exists()
            assert output_path.read_text() == result

    def test_handles_empty_trace(self, basic_artifact: DebateArtifact):
        """Should handle artifact without trace data."""
        exporter = DOTExporter(basic_artifact)
        result = exporter.export_flow()

        assert "digraph debate_flow {" in result
        assert "}" in result


# =============================================================================
# TestDOTExporterCritiques
# =============================================================================


class TestDOTExporterCritiques:
    """Tests for DOTExporter.export_critiques()."""

    def test_returns_dot_string(self, artifact_with_trace: DebateArtifact):
        """Should return a valid DOT string."""
        exporter = DOTExporter(artifact_with_trace)
        result = exporter.export_critiques()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_digraph_declaration(self, artifact_with_trace: DebateArtifact):
        """Should include digraph declaration."""
        exporter = DOTExporter(artifact_with_trace)
        result = exporter.export_critiques()

        assert "digraph critique_graph {" in result
        assert result.strip().endswith("}")

    def test_includes_agent_nodes(self, artifact_with_trace: DebateArtifact):
        """Should include nodes for each agent."""
        exporter = DOTExporter(artifact_with_trace)
        result = exporter.export_critiques()

        assert "claude [" in result
        assert "gpt_4 [" in result

    def test_includes_critique_edges(self, artifact_with_trace: DebateArtifact):
        """Should include edges for critique relationships."""
        exporter = DOTExporter(artifact_with_trace)
        result = exporter.export_critiques()

        assert "gpt_4 -> claude" in result
        assert "claude -> gpt_4" in result

    def test_shows_critique_count_and_severity(self, artifact_with_trace: DebateArtifact):
        """Should show critique count and average severity in edge labels."""
        exporter = DOTExporter(artifact_with_trace)
        result = exporter.export_critiques()

        # Should include count and severity in label
        assert "1x" in result  # Count
        assert "0.6" in result or "0.3" in result  # Severity

    def test_uses_severity_based_colors(self, artifact_with_trace: DebateArtifact):
        """Should use colors based on severity level."""
        exporter = DOTExporter(artifact_with_trace)
        result = exporter.export_critiques()

        # Should include color attributes
        assert "color=" in result

    def test_handles_empty_trace(self, basic_artifact: DebateArtifact):
        """Should handle artifact without trace data."""
        exporter = DOTExporter(basic_artifact)
        result = exporter.export_critiques()

        assert "digraph critique_graph {" in result


# =============================================================================
# TestDOTExporterConsensus
# =============================================================================


class TestDOTExporterConsensus:
    """Tests for DOTExporter.export_consensus()."""

    def test_returns_dot_string(self, artifact_with_consensus: DebateArtifact):
        """Should return a valid DOT string."""
        exporter = DOTExporter(artifact_with_consensus)
        result = exporter.export_consensus()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_digraph_declaration(self, artifact_with_consensus: DebateArtifact):
        """Should include digraph declaration."""
        exporter = DOTExporter(artifact_with_consensus)
        result = exporter.export_consensus()

        assert "digraph consensus_path {" in result
        assert result.strip().endswith("}")

    def test_includes_task_node(self, artifact_with_consensus: DebateArtifact):
        """Should include task node at the top."""
        exporter = DOTExporter(artifact_with_consensus)
        result = exporter.export_consensus()

        assert "task [" in result
        assert "API design" in result  # Part of task

    def test_includes_agent_vote_nodes(self, artifact_with_consensus: DebateArtifact):
        """Should include nodes showing each agent's vote."""
        exporter = DOTExporter(artifact_with_consensus)
        result = exporter.export_consensus()

        assert "claude [" in result
        assert "gpt_4 [" in result
        assert "gemini [" in result
        assert "Agreed" in result
        assert "Disagreed" in result

    def test_uses_color_for_vote_status(self, artifact_with_consensus: DebateArtifact):
        """Should use green for agreement, red for disagreement."""
        exporter = DOTExporter(artifact_with_consensus)
        result = exporter.export_consensus()

        assert "#C8E6C9" in result  # Green for agreement
        assert "#FFCDD2" in result  # Red for disagreement

    def test_includes_final_answer_node(self, artifact_with_consensus: DebateArtifact):
        """Should include final answer node."""
        exporter = DOTExporter(artifact_with_consensus)
        result = exporter.export_consensus()

        assert "final [" in result
        assert "CONSENSUS" in result
        assert "85%" in result

    def test_shows_no_consensus_status(self, artifact_no_consensus: DebateArtifact):
        """Should show NO CONSENSUS when consensus not reached."""
        exporter = DOTExporter(artifact_no_consensus)
        result = exporter.export_consensus()

        assert "NO CONSENSUS" in result

    def test_handles_missing_consensus_proof(self, basic_artifact: DebateArtifact):
        """Should handle artifact without consensus proof."""
        exporter = DOTExporter(basic_artifact)
        result = exporter.export_consensus()

        assert "digraph consensus_path {" in result
        assert "task [" in result


# =============================================================================
# TestDOTExporterExportAll
# =============================================================================


class TestDOTExporterExportAll:
    """Tests for DOTExporter.export_all()."""

    def test_creates_all_dot_files(self, artifact_with_trace: DebateArtifact):
        """Should create all DOT files in output directory."""
        artifact_with_trace.consensus_proof = ConsensusProof(
            reached=True,
            confidence=0.9,
            vote_breakdown={"claude": True, "gpt-4": True},
            final_answer="Agreed",
            rounds_used=2,
        )

        exporter = DOTExporter(artifact_with_trace)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            outputs = exporter.export_all(output_dir)

            assert "flow" in outputs
            assert "critiques" in outputs
            assert "consensus" in outputs

            for path in outputs.values():
                assert path.exists()
                assert path.suffix == ".dot"

    def test_creates_output_directory(self, basic_artifact: DebateArtifact):
        """Should create output directory if it doesn't exist."""
        exporter = DOTExporter(basic_artifact)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nested" / "output"
            exporter.export_all(output_dir)

            assert output_dir.exists()

    def test_uses_artifact_id_in_filenames(self, basic_artifact: DebateArtifact):
        """Should use artifact_id in output filenames."""
        exporter = DOTExporter(basic_artifact)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            outputs = exporter.export_all(output_dir)

            for path in outputs.values():
                assert "test-artifact-001" in path.name


# =============================================================================
# TestExportDebateToDot
# =============================================================================


class TestExportDebateToDot:
    """Tests for export_debate_to_dot convenience function."""

    def test_exports_flow_by_default(self, artifact_with_trace: DebateArtifact):
        """Should export flow mode by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "export.dot"
            result = export_debate_to_dot(artifact_with_trace, output_path)

            assert "debate_flow" in result
            assert output_path.exists()

    def test_exports_critiques_mode(self, artifact_with_trace: DebateArtifact):
        """Should export critiques when mode='critiques'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "critiques.dot"
            result = export_debate_to_dot(artifact_with_trace, output_path, mode="critiques")

            assert "critique_graph" in result

    def test_exports_consensus_mode(self, artifact_with_consensus: DebateArtifact):
        """Should export consensus when mode='consensus'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "consensus.dot"
            result = export_debate_to_dot(artifact_with_consensus, output_path, mode="consensus")

            assert "consensus_path" in result

    def test_raises_for_unknown_mode(self, basic_artifact: DebateArtifact):
        """Should raise ValueError for unknown mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "export.dot"

            with pytest.raises(ValueError, match="Unknown mode"):
                export_debate_to_dot(basic_artifact, output_path, mode="unknown")


# =============================================================================
# TestDOTExporterEdgeCases
# =============================================================================


class TestDOTExporterEdgeCases:
    """Edge case tests for DOT exporter."""

    def test_handles_agent_names_with_hyphens(self):
        """Should sanitize agent names with hyphens for DOT syntax."""
        artifact = DebateArtifact(
            artifact_id="test",
            agents=["claude-3", "gpt-4-turbo"],
            trace_data={
                "events": [
                    {
                        "event_type": "critique",
                        "agent": "gpt-4-turbo",
                        "target": "claude-3",
                        "severity": 0.5,
                        "issues": [],
                    }
                ]
            },
        )

        exporter = DOTExporter(artifact)
        result = exporter.export_critiques()

        # Should replace hyphens with underscores in node names
        assert "gpt_4_turbo" in result
        assert "claude_3" in result

    def test_handles_agent_names_with_dots(self):
        """Should sanitize agent names with dots for DOT syntax."""
        artifact = DebateArtifact(
            artifact_id="test",
            agents=["claude.v3", "gpt.4"],
        )

        exporter = DOTExporter(artifact)
        result = exporter.export_flow()

        # Should replace dots with underscores
        assert "claude_v3" in result
        assert "gpt_4" in result

    def test_handles_special_characters_in_content(self):
        """Should escape special characters in labels."""
        artifact = DebateArtifact(
            artifact_id="test",
            task='Task with "quotes" and\nnewlines',
        )

        exporter = DOTExporter(artifact)
        result = exporter.export_consensus()

        # Should be valid DOT that doesn't break on quotes
        assert "digraph" in result
        assert "}" in result

    def test_handles_many_agents(self):
        """Should handle artifact with many agents."""
        agents = [f"agent-{i}" for i in range(10)]
        artifact = DebateArtifact(
            artifact_id="test",
            agents=agents,
        )

        exporter = DOTExporter(artifact)
        result = exporter.export_flow()

        # Should include all agent subgraphs
        for i in range(10):
            assert f"agent_{i}" in result

    def test_handles_high_severity_critique(self):
        """Should use red color for high severity critiques."""
        artifact = DebateArtifact(
            artifact_id="test",
            agents=["claude", "gpt-4"],
            trace_data={
                "events": [
                    {
                        "event_type": "critique",
                        "agent": "gpt-4",
                        "target": "claude",
                        "severity": 0.9,  # High severity
                        "issues": ["Critical issue"],
                    }
                ]
            },
        )

        exporter = DOTExporter(artifact)
        result = exporter.export_critiques()

        assert "#F44336" in result  # Red for high severity

    def test_handles_low_severity_critique(self):
        """Should use green color for low severity critiques."""
        artifact = DebateArtifact(
            artifact_id="test",
            agents=["claude", "gpt-4"],
            trace_data={
                "events": [
                    {
                        "event_type": "critique",
                        "agent": "gpt-4",
                        "target": "claude",
                        "severity": 0.2,  # Low severity
                        "issues": ["Minor issue"],
                    }
                ]
            },
        )

        exporter = DOTExporter(artifact)
        result = exporter.export_critiques()

        assert "#4CAF50" in result  # Green for low severity

    def test_handles_medium_severity_critique(self):
        """Should use orange color for medium severity critiques."""
        artifact = DebateArtifact(
            artifact_id="test",
            agents=["claude", "gpt-4"],
            trace_data={
                "events": [
                    {
                        "event_type": "critique",
                        "agent": "gpt-4",
                        "target": "claude",
                        "severity": 0.5,  # Medium severity
                        "issues": ["Moderate issue"],
                    }
                ]
            },
        )

        exporter = DOTExporter(artifact)
        result = exporter.export_critiques()

        assert "#FF9800" in result  # Orange for medium severity
