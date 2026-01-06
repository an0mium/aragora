"""Tests for aragora.export.dot_exporter module.

Comprehensive tests for DOTExporter class and related functions.
"""

import pytest
from pathlib import Path

from aragora.export.artifact import (
    DebateArtifact,
    ConsensusProof,
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
def minimal_artifact():
    """Create minimal artifact for testing."""
    return DebateArtifact(
        artifact_id="art-001",
        debate_id="debate-123",
        task="Test task",
        agents=["agent1", "agent2"],
        rounds=3,
    )


@pytest.fixture
def artifact_with_trace():
    """Create artifact with trace data."""
    return DebateArtifact(
        artifact_id="art-trace",
        debate_id="debate-trace",
        task="Trace test task",
        trace_data={
            "events": [
                {
                    "event_type": "message",
                    "round": 1,
                    "agent": "agent1",
                    "content": "This is my proposal.",
                },
                {
                    "event_type": "message",
                    "round": 1,
                    "agent": "agent2",
                    "content": "I have concerns.",
                },
                {
                    "event_type": "critique",
                    "round": 1,
                    "agent": "agent2",
                    "target": "agent1",
                    "severity": 0.7,
                },
                {
                    "event_type": "message",
                    "round": 2,
                    "agent": "agent1",
                    "content": "Updated proposal.",
                },
            ]
        },
        agents=["agent1", "agent2"],
        rounds=2,
    )


@pytest.fixture
def artifact_with_consensus():
    """Create artifact with consensus proof."""
    return DebateArtifact(
        artifact_id="art-consensus",
        debate_id="debate-consensus",
        task="Consensus test task for decision making",
        consensus_proof=ConsensusProof(
            reached=True,
            confidence=0.85,
            vote_breakdown={"agent1": True, "agent2": True, "agent3": False},
            final_answer="The consensus answer is option A.",
            rounds_used=3,
        ),
        agents=["agent1", "agent2", "agent3"],
        rounds=3,
    )


@pytest.fixture
def artifact_with_critiques():
    """Create artifact with multiple critique events."""
    return DebateArtifact(
        artifact_id="art-critiques",
        debate_id="debate-critiques",
        task="Critique test",
        trace_data={
            "events": [
                {"event_type": "critique", "agent": "agent2", "target": "agent1", "severity": 0.8},
                {"event_type": "critique", "agent": "agent2", "target": "agent1", "severity": 0.6},
                {"event_type": "critique", "agent": "agent1", "target": "agent2", "severity": 0.3},
                {"event_type": "critique", "agent": "agent3", "target": "agent1", "severity": 0.9},
            ]
        },
        agents=["agent1", "agent2", "agent3"],
    )


# =============================================================================
# escape_label Tests
# =============================================================================


class TestEscapeLabel:
    """Tests for escape_label function."""

    def test_short_text_unchanged(self):
        """Test short text is not truncated."""
        result = escape_label("Hello")
        assert result == "Hello"

    def test_truncates_long_text(self):
        """Test long text is truncated."""
        long_text = "A" * 100
        result = escape_label(long_text)
        assert len(result) == 53  # 50 + "..."
        assert result.endswith("...")

    def test_custom_max_length(self):
        """Test custom max length."""
        text = "A" * 50
        result = escape_label(text, max_len=20)
        assert len(result) == 23  # 20 + "..."

    def test_exact_max_length(self):
        """Test text at exact max length."""
        text = "A" * 50
        result = escape_label(text, max_len=50)
        assert result == text  # No truncation

    def test_escapes_quotes(self):
        """Test quotes are escaped."""
        result = escape_label('Text with "quotes"')
        assert '\\"' in result
        assert '"quotes"' not in result

    def test_escapes_newlines(self):
        """Test newlines are escaped."""
        result = escape_label("Line1\nLine2")
        assert "\\n" in result
        assert "\n" not in result

    def test_multiple_special_characters(self):
        """Test multiple special characters."""
        result = escape_label('Text "with" many\nspecial\nchars')
        assert '\\"' in result
        assert "\\n" in result

    def test_empty_string(self):
        """Test empty string."""
        result = escape_label("")
        assert result == ""

    def test_truncation_before_escape(self):
        """Test truncation happens before escaping."""
        # Create text where quotes appear after the truncation point
        text = "A" * 45 + '"quote"'
        result = escape_label(text, max_len=45)
        assert result == "A" * 45 + "..."  # Quote is truncated


# =============================================================================
# DOTExporter.export_flow Tests
# =============================================================================


class TestDOTExporterFlow:
    """Tests for DOTExporter.export_flow()."""

    def test_exports_valid_digraph(self, minimal_artifact):
        """Test export produces valid DOT digraph."""
        exporter = DOTExporter(minimal_artifact)
        content = exporter.export_flow()

        assert content.startswith("digraph debate_flow {")
        assert content.strip().endswith("}")

    def test_includes_rankdir(self, minimal_artifact):
        """Test export includes rank direction."""
        exporter = DOTExporter(minimal_artifact)
        content = exporter.export_flow()

        assert "rankdir=TB" in content

    def test_creates_agent_subgraphs(self, minimal_artifact):
        """Test creates subgraph for each agent."""
        exporter = DOTExporter(minimal_artifact)
        content = exporter.export_flow()

        assert "subgraph cluster_agent1" in content
        assert "subgraph cluster_agent2" in content
        assert 'label="agent1"' in content
        assert 'label="agent2"' in content

    def test_sanitizes_agent_names(self):
        """Test sanitizes agent names with special chars."""
        artifact = DebateArtifact(
            agents=["claude-3.5", "gpt-4.0", "agent_ok"],
        )
        exporter = DOTExporter(artifact)
        content = exporter.export_flow()

        assert "cluster_claude_3_5" in content
        assert "cluster_gpt_4_0" in content
        assert "cluster_agent_ok" in content

    def test_creates_message_nodes(self, artifact_with_trace):
        """Test creates nodes for messages."""
        exporter = DOTExporter(artifact_with_trace)
        content = exporter.export_flow()

        assert "msg_1" in content
        assert "msg_2" in content
        assert "msg_3" in content

    def test_links_message_nodes(self, artifact_with_trace):
        """Test links message nodes sequentially."""
        exporter = DOTExporter(artifact_with_trace)
        content = exporter.export_flow()

        assert "msg_1 -> msg_2" in content
        assert "msg_2 -> msg_3" in content

    def test_includes_round_in_label(self, artifact_with_trace):
        """Test includes round number in label."""
        exporter = DOTExporter(artifact_with_trace)
        content = exporter.export_flow()

        assert "R1:" in content
        assert "R2:" in content

    def test_adds_consensus_node(self, artifact_with_consensus):
        """Test adds consensus result node."""
        artifact = DebateArtifact(
            trace_data={"events": [{"event_type": "message", "content": "test"}]},
            consensus_proof=artifact_with_consensus.consensus_proof,
        )
        exporter = DOTExporter(artifact)
        content = exporter.export_flow()

        assert "consensus" in content
        assert "Consensus" in content
        assert "85%" in content

    def test_consensus_no_consensus(self):
        """Test handles consensus not reached."""
        artifact = DebateArtifact(
            # Need at least one message for prev_node to be defined
            trace_data={
                "events": [
                    {"event_type": "message", "content": "test"}
                ]
            },
            consensus_proof=ConsensusProof(
                reached=False,
                confidence=0.3,
                vote_breakdown={},
                final_answer="",
                rounds_used=5,
            ),
        )
        exporter = DOTExporter(artifact)
        content = exporter.export_flow()

        assert "No Consensus" in content
        assert "30%" in content

    def test_no_trace_data(self, minimal_artifact):
        """Test handles missing trace data."""
        exporter = DOTExporter(minimal_artifact)
        content = exporter.export_flow()

        # Should still produce valid DOT
        assert "digraph debate_flow {" in content
        assert "msg_" not in content

    def test_writes_to_file(self, minimal_artifact, tmp_path):
        """Test writes DOT to file."""
        exporter = DOTExporter(minimal_artifact)
        output_path = tmp_path / "flow.dot"

        content = exporter.export_flow(output_path)

        assert output_path.exists()
        assert output_path.read_text() == content


# =============================================================================
# DOTExporter.export_critiques Tests
# =============================================================================


class TestDOTExporterCritiques:
    """Tests for DOTExporter.export_critiques()."""

    def test_exports_valid_digraph(self, minimal_artifact):
        """Test export produces valid DOT digraph."""
        exporter = DOTExporter(minimal_artifact)
        content = exporter.export_critiques()

        assert content.startswith("digraph critique_graph {")
        assert content.strip().endswith("}")

    def test_includes_rankdir_lr(self, minimal_artifact):
        """Test uses left-to-right direction."""
        exporter = DOTExporter(minimal_artifact)
        content = exporter.export_critiques()

        assert "rankdir=LR" in content

    def test_creates_agent_nodes(self, minimal_artifact):
        """Test creates nodes for each agent."""
        exporter = DOTExporter(minimal_artifact)
        content = exporter.export_critiques()

        assert "agent1" in content
        assert "agent2" in content

    def test_creates_critique_edges(self, artifact_with_critiques):
        """Test creates edges for critiques."""
        exporter = DOTExporter(artifact_with_critiques)
        content = exporter.export_critiques()

        assert "agent2 -> agent1" in content
        assert "agent1 -> agent2" in content
        assert "agent3 -> agent1" in content

    def test_aggregates_multiple_critiques(self, artifact_with_critiques):
        """Test aggregates multiple critiques between same agents."""
        exporter = DOTExporter(artifact_with_critiques)
        content = exporter.export_critiques()

        # agent2 critiqued agent1 twice
        assert "2x" in content

    def test_calculates_average_severity(self, artifact_with_critiques):
        """Test calculates average severity."""
        exporter = DOTExporter(artifact_with_critiques)
        content = exporter.export_critiques()

        # agent2 -> agent1: (0.8 + 0.6) / 2 = 0.7
        assert "0.7" in content

    def test_high_severity_color(self):
        """Test high severity gets red color."""
        artifact = DebateArtifact(
            trace_data={
                "events": [
                    {"event_type": "critique", "agent": "a", "target": "b", "severity": 0.9}
                ]
            },
            agents=["a", "b"],
        )
        exporter = DOTExporter(artifact)
        content = exporter.export_critiques()

        assert "#F44336" in content  # Red

    def test_medium_severity_color(self):
        """Test medium severity gets orange color."""
        artifact = DebateArtifact(
            trace_data={
                "events": [
                    {"event_type": "critique", "agent": "a", "target": "b", "severity": 0.5}
                ]
            },
            agents=["a", "b"],
        )
        exporter = DOTExporter(artifact)
        content = exporter.export_critiques()

        assert "#FF9800" in content  # Orange

    def test_low_severity_color(self):
        """Test low severity gets green color."""
        artifact = DebateArtifact(
            trace_data={
                "events": [
                    {"event_type": "critique", "agent": "a", "target": "b", "severity": 0.2}
                ]
            },
            agents=["a", "b"],
        )
        exporter = DOTExporter(artifact)
        content = exporter.export_critiques()

        assert "#4CAF50" in content  # Green

    def test_edge_width_proportional(self):
        """Test edge width is proportional to critique count."""
        artifact = DebateArtifact(
            trace_data={
                "events": [
                    {"event_type": "critique", "agent": "a", "target": "b", "severity": 0.5}
                    for _ in range(3)
                ]
            },
            agents=["a", "b"],
        )
        exporter = DOTExporter(artifact)
        content = exporter.export_critiques()

        assert "penwidth=3" in content

    def test_edge_width_capped(self):
        """Test edge width is capped at 5."""
        artifact = DebateArtifact(
            trace_data={
                "events": [
                    {"event_type": "critique", "agent": "a", "target": "b", "severity": 0.5}
                    for _ in range(10)
                ]
            },
            agents=["a", "b"],
        )
        exporter = DOTExporter(artifact)
        content = exporter.export_critiques()

        assert "penwidth=5" in content

    def test_no_critiques(self, minimal_artifact):
        """Test handles no critique events."""
        exporter = DOTExporter(minimal_artifact)
        content = exporter.export_critiques()

        # Should still produce valid DOT with agent nodes
        assert "digraph critique_graph {" in content
        assert "->" not in content or "task ->" in content  # No critique edges

    def test_writes_to_file(self, artifact_with_critiques, tmp_path):
        """Test writes DOT to file."""
        exporter = DOTExporter(artifact_with_critiques)
        output_path = tmp_path / "critiques.dot"

        exporter.export_critiques(output_path)

        assert output_path.exists()


# =============================================================================
# DOTExporter.export_consensus Tests
# =============================================================================


class TestDOTExporterConsensus:
    """Tests for DOTExporter.export_consensus()."""

    def test_exports_valid_digraph(self, minimal_artifact):
        """Test export produces valid DOT digraph."""
        exporter = DOTExporter(minimal_artifact)
        content = exporter.export_consensus()

        assert content.startswith("digraph consensus_path {")
        assert content.strip().endswith("}")

    def test_includes_task_node(self, minimal_artifact):
        """Test includes task node."""
        exporter = DOTExporter(minimal_artifact)
        content = exporter.export_consensus()

        assert "task [label=" in content
        assert "Test task" in content

    def test_truncates_long_task(self):
        """Test truncates long task."""
        artifact = DebateArtifact(task="A" * 100)
        exporter = DOTExporter(artifact)
        content = exporter.export_consensus()

        # escape_label uses default max_len=50, but consensus uses 80
        assert "..." in content

    def test_creates_vote_nodes(self, artifact_with_consensus):
        """Test creates node for each voter."""
        exporter = DOTExporter(artifact_with_consensus)
        content = exporter.export_consensus()

        assert "agent1" in content
        assert "agent2" in content
        assert "agent3" in content

    def test_colors_agreed_votes_green(self, artifact_with_consensus):
        """Test agreed votes are colored green."""
        exporter = DOTExporter(artifact_with_consensus)
        content = exporter.export_consensus()

        assert "#C8E6C9" in content  # Green

    def test_colors_disagreed_votes_red(self, artifact_with_consensus):
        """Test disagreed votes are colored red."""
        exporter = DOTExporter(artifact_with_consensus)
        content = exporter.export_consensus()

        assert "#FFCDD2" in content  # Red

    def test_links_task_to_votes(self, artifact_with_consensus):
        """Test task node links to vote nodes."""
        exporter = DOTExporter(artifact_with_consensus)
        content = exporter.export_consensus()

        assert "task -> agent1" in content
        assert "task -> agent2" in content
        assert "task -> agent3" in content

    def test_includes_final_answer(self, artifact_with_consensus):
        """Test includes final answer node."""
        exporter = DOTExporter(artifact_with_consensus)
        content = exporter.export_consensus()

        assert "final [label=" in content
        assert "CONSENSUS" in content
        assert "85%" in content

    def test_links_votes_to_final(self, artifact_with_consensus):
        """Test vote nodes link to final answer."""
        exporter = DOTExporter(artifact_with_consensus)
        content = exporter.export_consensus()

        assert "agent1 -> final" in content
        assert "agent2 -> final" in content
        assert "agent3 -> final" in content
        assert "style=dashed" in content

    def test_no_consensus_status(self):
        """Test handles consensus not reached."""
        artifact = DebateArtifact(
            task="Test",
            consensus_proof=ConsensusProof(
                reached=False,
                confidence=0.4,
                vote_breakdown={"a": False, "b": False},
                final_answer="No agreement",
                rounds_used=5,
            ),
        )
        exporter = DOTExporter(artifact)
        content = exporter.export_consensus()

        assert "NO CONSENSUS" in content
        assert "40%" in content

    def test_no_consensus_proof(self, minimal_artifact):
        """Test handles missing consensus proof."""
        exporter = DOTExporter(minimal_artifact)
        content = exporter.export_consensus()

        # Should still produce valid DOT with task
        assert "digraph consensus_path {" in content
        assert "task [label=" in content
        assert "final" not in content

    def test_writes_to_file(self, artifact_with_consensus, tmp_path):
        """Test writes DOT to file."""
        exporter = DOTExporter(artifact_with_consensus)
        output_path = tmp_path / "consensus.dot"

        exporter.export_consensus(output_path)

        assert output_path.exists()


# =============================================================================
# DOTExporter.export_all Tests
# =============================================================================


class TestDOTExporterAll:
    """Tests for DOTExporter.export_all()."""

    def test_creates_directory(self, minimal_artifact, tmp_path):
        """Test creates output directory."""
        exporter = DOTExporter(minimal_artifact)
        output_dir = tmp_path / "output" / "nested"

        exporter.export_all(output_dir)

        assert output_dir.exists()

    def test_returns_paths(self, minimal_artifact, tmp_path):
        """Test returns dict of paths."""
        exporter = DOTExporter(minimal_artifact)
        output_dir = tmp_path / "output"

        outputs = exporter.export_all(output_dir)

        assert "flow" in outputs
        assert "critiques" in outputs
        assert "consensus" in outputs

    def test_creates_all_files(self, minimal_artifact, tmp_path):
        """Test creates all DOT files."""
        exporter = DOTExporter(minimal_artifact)
        output_dir = tmp_path / "output"

        outputs = exporter.export_all(output_dir)

        for path in outputs.values():
            assert path.exists()

    def test_file_naming(self, minimal_artifact, tmp_path):
        """Test uses artifact_id in filenames."""
        exporter = DOTExporter(minimal_artifact)
        output_dir = tmp_path / "output"

        outputs = exporter.export_all(output_dir)

        assert "art-001_flow.dot" in str(outputs["flow"])
        assert "art-001_critiques.dot" in str(outputs["critiques"])
        assert "art-001_consensus.dot" in str(outputs["consensus"])


# =============================================================================
# export_debate_to_dot Tests
# =============================================================================


class TestExportDebateToDot:
    """Tests for export_debate_to_dot convenience function."""

    def test_export_flow(self, artifact_with_trace, tmp_path):
        """Test export with mode='flow'."""
        output_path = tmp_path / "flow.dot"

        content = export_debate_to_dot(artifact_with_trace, output_path, mode="flow")

        assert output_path.exists()
        assert "debate_flow" in content

    def test_export_critiques(self, artifact_with_critiques, tmp_path):
        """Test export with mode='critiques'."""
        output_path = tmp_path / "critiques.dot"

        content = export_debate_to_dot(artifact_with_critiques, output_path, mode="critiques")

        assert "critique_graph" in content

    def test_export_consensus(self, artifact_with_consensus, tmp_path):
        """Test export with mode='consensus'."""
        output_path = tmp_path / "consensus.dot"

        content = export_debate_to_dot(artifact_with_consensus, output_path, mode="consensus")

        assert "consensus_path" in content

    def test_export_invalid_mode(self, minimal_artifact, tmp_path):
        """Test export raises error for invalid mode."""
        output_path = tmp_path / "invalid.dot"

        with pytest.raises(ValueError) as exc_info:
            export_debate_to_dot(minimal_artifact, output_path, mode="invalid")

        assert "Unknown mode" in str(exc_info.value)


# =============================================================================
# Edge Cases
# =============================================================================


class TestDOTExporterEdgeCases:
    """Edge case tests for DOT export."""

    def test_empty_agents_list(self):
        """Test handles empty agents list."""
        artifact = DebateArtifact(agents=[])
        exporter = DOTExporter(artifact)

        content = exporter.export_flow()
        assert "digraph debate_flow {" in content

    def test_special_characters_in_content(self):
        """Test handles special characters in message content."""
        artifact = DebateArtifact(
            trace_data={
                "events": [
                    {
                        "event_type": "message",
                        "content": 'Content with "quotes" and\nnewlines',
                    }
                ]
            },
        )
        exporter = DOTExporter(artifact)
        content = exporter.export_flow()

        # Should be escaped
        assert '\\"' in content
        assert "\\n" in content

    def test_unicode_content(self):
        """Test handles unicode content."""
        artifact = DebateArtifact(
            task="日本語タスク",
            agents=["agent-日本語"],
            trace_data={
                "events": [
                    {"event_type": "message", "content": "Emoji: 👍"}
                ]
            },
        )
        exporter = DOTExporter(artifact)

        # Should not raise
        content = exporter.export_flow()
        assert "日本語" in content

    def test_agent_names_with_dots_and_dashes(self):
        """Test agent names with dots and dashes are sanitized."""
        artifact = DebateArtifact(
            agents=["claude-3.5-sonnet", "gpt-4.0-turbo"],
        )
        exporter = DOTExporter(artifact)
        content = exporter.export_critiques()

        assert "claude_3_5_sonnet" in content
        assert "gpt_4_0_turbo" in content

    def test_very_long_message_content(self):
        """Test truncates very long message content."""
        long_content = "A" * 500
        artifact = DebateArtifact(
            trace_data={
                "events": [
                    {"event_type": "message", "content": long_content}
                ]
            },
        )
        exporter = DOTExporter(artifact)
        content = exporter.export_flow()

        # Content is first truncated to 100 chars, then escape_label
        # truncates to 50 total ("R0: " = 4 chars + 46 content chars + "...")
        assert "..." in content
        # Should have some A's but not 100
        assert "AAAAA" in content

    def test_missing_event_fields(self):
        """Test handles events with missing fields."""
        artifact = DebateArtifact(
            trace_data={
                "events": [
                    {"event_type": "message"},  # Missing all other fields
                    {"event_type": "critique"},  # Missing all other fields
                ]
            },
        )
        exporter = DOTExporter(artifact)

        # Should not raise
        flow = exporter.export_flow()
        critiques = exporter.export_critiques()

        assert "digraph debate_flow {" in flow
        assert "digraph critique_graph {" in critiques

    def test_zero_severity_critique(self):
        """Test handles zero severity critique."""
        artifact = DebateArtifact(
            trace_data={
                "events": [
                    {"event_type": "critique", "agent": "a", "target": "b", "severity": 0}
                ]
            },
            agents=["a", "b"],
        )
        exporter = DOTExporter(artifact)
        content = exporter.export_critiques()

        assert "#4CAF50" in content  # Green (low severity)

    def test_consensus_proof_empty_vote_breakdown(self):
        """Test handles empty vote breakdown."""
        artifact = DebateArtifact(
            task="Test",
            consensus_proof=ConsensusProof(
                reached=False,
                confidence=0.0,
                vote_breakdown={},
                final_answer="",
                rounds_used=0,
            ),
        )
        exporter = DOTExporter(artifact)
        content = exporter.export_consensus()

        assert "final [label=" in content
        assert "NO CONSENSUS" in content
