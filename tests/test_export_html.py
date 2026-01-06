"""Tests for aragora.export.static_html module.

Comprehensive tests for StaticHTMLExporter class and related functions.
"""

import pytest
from pathlib import Path

from aragora.export.artifact import (
    DebateArtifact,
    ConsensusProof,
    VerificationResult,
)
from aragora.export.static_html import (
    StaticHTMLExporter,
    export_to_html,
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
        task="Test task for debate",
        created_at="2024-01-15T12:00:00",
        agents=["agent1", "agent2"],
        rounds=3,
        duration_seconds=60.0,
        message_count=10,
        critique_count=5,
    )


@pytest.fixture
def artifact_with_consensus():
    """Create artifact with consensus proof."""
    return DebateArtifact(
        artifact_id="art-consensus",
        debate_id="debate-consensus",
        task="Consensus test task",
        consensus_proof=ConsensusProof(
            reached=True,
            confidence=0.85,
            vote_breakdown={"agent1": True, "agent2": True},
            final_answer="The consensus answer",
            rounds_used=3,
        ),
        agents=["agent1", "agent2"],
        rounds=3,
    )


@pytest.fixture
def artifact_with_graph():
    """Create artifact with graph data."""
    return DebateArtifact(
        artifact_id="art-graph",
        debate_id="debate-graph",
        task="Graph test",
        graph_data={
            "nodes": {
                "n1": {"node_type": "root", "agent_id": "agent1", "content": "Initial proposal"},
                "n2": {"node_type": "proposal", "agent_id": "agent1", "content": "Detailed proposal"},
                "n3": {"node_type": "critique", "agent_id": "agent2", "content": "Critique of proposal"},
                "n4": {"node_type": "synthesis", "agent_id": "agent1", "content": "Synthesis"},
            }
        },
        agents=["agent1", "agent2"],
    )


@pytest.fixture
def artifact_with_trace():
    """Create artifact with trace data."""
    return DebateArtifact(
        artifact_id="art-trace",
        debate_id="debate-trace",
        task="Trace test",
        trace_data={
            "events": [
                {
                    "event_type": "agent_proposal",
                    "agent": "agent1",
                    "round_num": 1,
                    "content": {"content": "My proposal is to implement feature X."},
                },
                {
                    "event_type": "agent_critique",
                    "agent": "agent2",
                    "round_num": 1,
                    "content": {"issues": ["Missing tests", "No documentation"]},
                },
                {
                    "event_type": "agent_synthesis",
                    "agent": "agent1",
                    "round_num": 2,
                    "content": {"content": "Updated proposal addressing concerns."},
                },
            ]
        },
        agents=["agent1", "agent2"],
    )


@pytest.fixture
def artifact_with_provenance():
    """Create artifact with provenance data."""
    return DebateArtifact(
        artifact_id="art-prov",
        debate_id="debate-prov",
        task="Provenance test",
        provenance_data={
            "chain": {
                "records": [
                    {
                        "id": "ev-001",
                        "source_type": "web",
                        "source_id": "https://example.com",
                        "content": "Evidence content here",
                        "content_hash": "abc123def456",
                        "previous_hash": "000000000000",
                    },
                    {
                        "id": "ev-002",
                        "source_type": "paper",
                        "source_id": "arxiv:2024.12345",
                        "content": "More evidence",
                        "content_hash": "def456ghi789",
                        "previous_hash": "abc123def456",
                    },
                ]
            }
        },
    )


@pytest.fixture
def artifact_with_verifications():
    """Create artifact with verification results."""
    return DebateArtifact(
        artifact_id="art-verify",
        debate_id="debate-verify",
        task="Verification test",
        verification_results=[
            VerificationResult(
                claim_id="c1",
                claim_text="All inputs are valid",
                status="verified",
                method="z3",
                proof_trace="QED by induction",
                duration_ms=150,
            ),
            VerificationResult(
                claim_id="c2",
                claim_text="Output is always positive",
                status="refuted",
                method="z3",
                counterexample="x = -1",
                duration_ms=200,
            ),
            VerificationResult(
                claim_id="c3",
                claim_text="Algorithm terminates",
                status="timeout",
                method="lean",
                duration_ms=60000,
            ),
        ],
    )


@pytest.fixture
def full_artifact():
    """Create artifact with all components."""
    return DebateArtifact(
        artifact_id="art-full",
        debate_id="debate-full",
        task="Full test task with all components",
        created_at="2024-01-15T12:00:00",
        graph_data={
            "nodes": {
                "n1": {"node_type": "proposal", "agent_id": "a1", "content": "Proposal"},
            }
        },
        trace_data={
            "events": [
                {"event_type": "agent_proposal", "agent": "a1", "round_num": 1, "content": {"content": "Test"}},
            ]
        },
        provenance_data={
            "chain": {
                "records": [{"id": "e1", "source_type": "test", "content": "data"}]
            }
        },
        consensus_proof=ConsensusProof(
            reached=True,
            confidence=0.9,
            vote_breakdown={"a1": True},
            final_answer="Answer",
            rounds_used=2,
        ),
        verification_results=[
            VerificationResult("c1", "Claim", "verified", "z3"),
        ],
        agents=["a1", "a2"],
        rounds=2,
        duration_seconds=120.0,
        message_count=15,
        critique_count=5,
    )


# =============================================================================
# HTML Escaping Tests
# =============================================================================


class TestHTMLEscaping:
    """Tests for HTML escaping functionality."""

    def test_escapes_ampersand(self, minimal_artifact):
        """Test ampersand is escaped."""
        exporter = StaticHTMLExporter(minimal_artifact)
        result = exporter._escape("foo & bar")
        assert "&amp;" in result
        assert " & " not in result

    def test_escapes_less_than(self, minimal_artifact):
        """Test less than is escaped."""
        exporter = StaticHTMLExporter(minimal_artifact)
        result = exporter._escape("foo < bar")
        assert "&lt;" in result
        assert " < " not in result

    def test_escapes_greater_than(self, minimal_artifact):
        """Test greater than is escaped."""
        exporter = StaticHTMLExporter(minimal_artifact)
        result = exporter._escape("foo > bar")
        assert "&gt;" in result
        assert " > " not in result

    def test_escapes_double_quotes(self, minimal_artifact):
        """Test double quotes are escaped."""
        exporter = StaticHTMLExporter(minimal_artifact)
        result = exporter._escape('foo "bar" baz')
        assert "&quot;" in result
        assert '"bar"' not in result

    def test_escapes_single_quotes(self, minimal_artifact):
        """Test single quotes are escaped."""
        exporter = StaticHTMLExporter(minimal_artifact)
        result = exporter._escape("foo 'bar' baz")
        assert "&#39;" in result
        assert "'bar'" not in result

    def test_escapes_all_characters(self, minimal_artifact):
        """Test all special characters are escaped."""
        exporter = StaticHTMLExporter(minimal_artifact)
        result = exporter._escape("<script>alert('xss' & \"test\")</script>")

        assert "&lt;" in result
        assert "&gt;" in result
        assert "&#39;" in result
        assert "&amp;" in result
        assert "&quot;" in result
        assert "<script>" not in result

    def test_empty_string(self, minimal_artifact):
        """Test empty string escaping."""
        exporter = StaticHTMLExporter(minimal_artifact)
        result = exporter._escape("")
        assert result == ""


# =============================================================================
# HTML Generation Tests
# =============================================================================


class TestHTMLGeneration:
    """Tests for HTML document generation."""

    def test_generates_valid_html5_doctype(self, minimal_artifact):
        """Test generates valid HTML5 DOCTYPE."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert html.startswith("<!DOCTYPE html>")

    def test_includes_html_tag_with_lang(self, minimal_artifact):
        """Test includes html tag with lang attribute."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert '<html lang="en">' in html

    def test_includes_head_section(self, minimal_artifact):
        """Test includes head section."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "<head>" in html
        assert "</head>" in html

    def test_includes_body_section(self, minimal_artifact):
        """Test includes body section."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "<body>" in html
        assert "</body>" in html

    def test_includes_meta_charset(self, minimal_artifact):
        """Test includes charset meta tag."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert '<meta charset="UTF-8">' in html

    def test_includes_viewport_meta(self, minimal_artifact):
        """Test includes viewport meta tag."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert 'viewport' in html

    def test_includes_title(self, minimal_artifact):
        """Test includes title with task."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "<title>" in html
        assert "aragora Debate" in html

    def test_includes_styles(self, minimal_artifact):
        """Test includes embedded CSS."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "<style>" in html
        assert "</style>" in html

    def test_includes_scripts(self, minimal_artifact):
        """Test includes embedded JavaScript."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "<script>" in html
        assert "</script>" in html

    def test_closes_html_tag(self, minimal_artifact):
        """Test closes html tag."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert html.strip().endswith("</html>")


# =============================================================================
# Header Tests
# =============================================================================


class TestHTMLHeader:
    """Tests for header section generation."""

    def test_includes_header_tag(self, minimal_artifact):
        """Test includes header tag."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "<header>" in html
        assert "</header>" in html

    def test_includes_artifact_id(self, minimal_artifact):
        """Test includes artifact ID in header."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "art-001" in html

    def test_consensus_reached_badge(self, artifact_with_consensus):
        """Test shows consensus reached badge."""
        exporter = StaticHTMLExporter(artifact_with_consensus)
        html = exporter.generate()
        assert "Consensus Reached" in html
        assert "85%" in html
        assert "reached" in html

    def test_no_consensus_badge(self, minimal_artifact):
        """Test shows no consensus badge."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "No Consensus" in html
        assert "N/A" in html


# =============================================================================
# Task Section Tests
# =============================================================================


class TestHTMLTaskSection:
    """Tests for task section generation."""

    def test_includes_task_section(self, minimal_artifact):
        """Test includes task section."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "task-section" in html

    def test_includes_task_text(self, minimal_artifact):
        """Test includes task text."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "Test task for debate" in html

    def test_escapes_task_text(self):
        """Test escapes special characters in task section."""
        artifact = DebateArtifact(task="<script>alert('xss')</script>")
        exporter = StaticHTMLExporter(artifact)
        html = exporter.generate()
        # The task is escaped in the task-section
        assert "&lt;script&gt;" in html
        # Check that the task section contains escaped version
        assert "task-section" in html


# =============================================================================
# Tabs Tests
# =============================================================================


class TestHTMLTabs:
    """Tests for tabs navigation generation."""

    def test_includes_tabs(self, minimal_artifact):
        """Test includes tabs container."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert 'class="tabs"' in html

    def test_includes_all_tab_buttons(self, minimal_artifact):
        """Test includes all four tab buttons."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert 'data-tab="graph"' in html
        assert 'data-tab="timeline"' in html
        assert 'data-tab="provenance"' in html
        assert 'data-tab="verification"' in html


# =============================================================================
# Graph View Tests
# =============================================================================


class TestHTMLGraphView:
    """Tests for graph view generation."""

    def test_empty_state_no_graph(self, minimal_artifact):
        """Test shows empty state when no graph data."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "No graph data available" in html

    def test_renders_graph_nodes(self, artifact_with_graph):
        """Test renders graph nodes."""
        exporter = StaticHTMLExporter(artifact_with_graph)
        html = exporter.generate()
        assert "graph-node" in html

    def test_includes_node_types(self, artifact_with_graph):
        """Test includes node type classes."""
        exporter = StaticHTMLExporter(artifact_with_graph)
        html = exporter.generate()
        assert "proposal" in html
        assert "critique" in html
        assert "synthesis" in html

    def test_positions_nodes(self, artifact_with_graph):
        """Test positions nodes with CSS."""
        exporter = StaticHTMLExporter(artifact_with_graph)
        html = exporter.generate()
        assert "style=" in html
        assert "left:" in html
        assert "top:" in html


# =============================================================================
# Timeline View Tests
# =============================================================================


class TestHTMLTimelineView:
    """Tests for timeline view generation."""

    def test_empty_state_no_trace(self, minimal_artifact):
        """Test shows empty state when no trace data."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "No trace data available" in html

    def test_renders_timeline_items(self, artifact_with_trace):
        """Test renders timeline items."""
        exporter = StaticHTMLExporter(artifact_with_trace)
        html = exporter.generate()
        assert "timeline-item" in html

    def test_includes_timeline_controls(self, artifact_with_trace):
        """Test includes timeline controls."""
        exporter = StaticHTMLExporter(artifact_with_trace)
        html = exporter.generate()
        assert "btn-prev" in html
        assert "btn-next" in html
        assert "btn-play" in html
        assert "timeline-slider" in html

    def test_filters_event_types(self, artifact_with_trace):
        """Test filters for specific event types."""
        exporter = StaticHTMLExporter(artifact_with_trace)
        html = exporter.generate()
        # Should include proposal and critique content
        assert "agent1" in html
        assert "agent2" in html


# =============================================================================
# Provenance View Tests
# =============================================================================


class TestHTMLProvenanceView:
    """Tests for provenance view generation."""

    def test_empty_state_no_provenance(self, minimal_artifact):
        """Test shows empty state when no provenance data."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "No provenance data available" in html

    def test_renders_provenance_items(self, artifact_with_provenance):
        """Test renders provenance items."""
        exporter = StaticHTMLExporter(artifact_with_provenance)
        html = exporter.generate()
        assert "provenance-item" in html

    def test_shows_record_count(self, artifact_with_provenance):
        """Test shows record count."""
        exporter = StaticHTMLExporter(artifact_with_provenance)
        html = exporter.generate()
        assert "2 records" in html

    def test_includes_hash_chain(self, artifact_with_provenance):
        """Test includes hash chain visualization."""
        exporter = StaticHTMLExporter(artifact_with_provenance)
        html = exporter.generate()
        assert "chain-link" in html
        assert "abc123" in html


# =============================================================================
# Verification View Tests
# =============================================================================


class TestHTMLVerificationView:
    """Tests for verification view generation."""

    def test_empty_state_no_verifications(self, minimal_artifact):
        """Test shows empty state when no verifications."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "No formal verification results available" in html

    def test_renders_verification_items(self, artifact_with_verifications):
        """Test renders verification items."""
        exporter = StaticHTMLExporter(artifact_with_verifications)
        html = exporter.generate()
        assert "verification-item" in html

    def test_shows_verification_statuses(self, artifact_with_verifications):
        """Test shows all verification statuses."""
        exporter = StaticHTMLExporter(artifact_with_verifications)
        html = exporter.generate()
        assert "VERIFIED" in html
        assert "REFUTED" in html
        assert "TIMEOUT" in html

    def test_includes_proof_trace(self, artifact_with_verifications):
        """Test includes proof trace when available."""
        exporter = StaticHTMLExporter(artifact_with_verifications)
        html = exporter.generate()
        assert "QED by induction" in html


# =============================================================================
# Stats Section Tests
# =============================================================================


class TestHTMLStatsSection:
    """Tests for stats section generation."""

    def test_includes_stats_section(self, minimal_artifact):
        """Test includes stats section."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert 'class="stats"' in html

    def test_shows_rounds(self, minimal_artifact):
        """Test shows rounds count."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "Rounds" in html

    def test_shows_messages(self, minimal_artifact):
        """Test shows message count."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "Messages" in html

    def test_shows_critiques(self, minimal_artifact):
        """Test shows critique count."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "Critiques" in html

    def test_shows_duration(self, minimal_artifact):
        """Test shows duration."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "Duration" in html


# =============================================================================
# Footer Tests
# =============================================================================


class TestHTMLFooter:
    """Tests for footer section generation."""

    def test_includes_footer(self, minimal_artifact):
        """Test includes footer tag."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "<footer>" in html
        assert "</footer>" in html

    def test_includes_artifact_id(self, minimal_artifact):
        """Test includes artifact ID in footer."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "art-001" in html

    def test_includes_content_hash(self, minimal_artifact):
        """Test includes content hash in footer."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "Hash:" in html


# =============================================================================
# JavaScript Tests
# =============================================================================


class TestHTMLScripts:
    """Tests for embedded JavaScript generation."""

    def test_includes_artifact_data(self, minimal_artifact):
        """Test includes embedded artifact data."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "artifactData" in html

    def test_includes_tab_switching_code(self, minimal_artifact):
        """Test includes tab switching JavaScript."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "querySelectorAll" in html
        assert "classList" in html

    def test_includes_timeline_controls_code(self, minimal_artifact):
        """Test includes timeline controls JavaScript."""
        exporter = StaticHTMLExporter(minimal_artifact)
        html = exporter.generate()
        assert "updateTimeline" in html
        assert "currentIndex" in html


# =============================================================================
# File I/O Tests
# =============================================================================


class TestHTMLSave:
    """Tests for file saving functionality."""

    def test_save_creates_file(self, minimal_artifact, tmp_path):
        """Test save creates file."""
        exporter = StaticHTMLExporter(minimal_artifact)
        output_path = tmp_path / "debate.html"

        result = exporter.save(output_path)

        assert output_path.exists()
        assert result == output_path

    def test_save_writes_valid_html(self, minimal_artifact, tmp_path):
        """Test save writes valid HTML."""
        exporter = StaticHTMLExporter(minimal_artifact)
        output_path = tmp_path / "debate.html"

        exporter.save(output_path)

        content = output_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "</html>" in content


# =============================================================================
# export_to_html Tests
# =============================================================================


class TestExportToHTML:
    """Tests for export_to_html convenience function."""

    def test_creates_file(self, minimal_artifact, tmp_path):
        """Test creates HTML file."""
        output_path = tmp_path / "export.html"

        result = export_to_html(minimal_artifact, output_path)

        assert output_path.exists()
        assert result == output_path

    def test_returns_path(self, minimal_artifact, tmp_path):
        """Test returns path."""
        output_path = tmp_path / "export.html"

        result = export_to_html(minimal_artifact, output_path)

        assert isinstance(result, Path)


# =============================================================================
# Security Tests
# =============================================================================


class TestHTMLSecurity:
    """Security tests for HTML export."""

    def test_xss_in_task_is_escaped(self):
        """Test XSS prevention in task field - task is escaped in HTML content."""
        artifact = DebateArtifact(task="<script>alert('xss')</script>")
        exporter = StaticHTMLExporter(artifact)
        html = exporter.generate()

        # The task section should have escaped content
        assert "&lt;script&gt;" in html

    def test_xss_in_graph_agent_is_escaped(self):
        """Test XSS prevention in graph agent names."""
        artifact = DebateArtifact(
            graph_data={
                "nodes": {
                    "n1": {"agent_id": "<b>bold</b>", "content": "test", "node_type": "proposal"}
                }
            },
        )
        exporter = StaticHTMLExporter(artifact)
        html = exporter.generate()

        # Agent IDs in graph nodes should be escaped
        assert "&lt;b&gt;" in html

    def test_escape_function_works(self):
        """Test the _escape function properly escapes all dangerous chars."""
        artifact = DebateArtifact()
        exporter = StaticHTMLExporter(artifact)

        dangerous = "<script>alert('xss' & \"test\")</script>"
        escaped = exporter._escape(dangerous)

        assert "<script>" not in escaped
        assert "&lt;script&gt;" in escaped
        assert "&#39;" in escaped  # single quote
        assert "&quot;" in escaped  # double quote
        assert "&amp;" in escaped  # ampersand

    def test_verification_claim_is_escaped(self):
        """Test verification claim text is escaped."""
        artifact = DebateArtifact(
            verification_results=[
                VerificationResult(
                    claim_id="c1",
                    claim_text="<b>bold</b> & 'quoted'",
                    status="verified",
                    method="z3",
                ),
            ],
        )
        exporter = StaticHTMLExporter(artifact)
        html = exporter.generate()

        # The verification view calls _escape on claim_text
        assert "&lt;b&gt;" in html


# =============================================================================
# Edge Cases
# =============================================================================


class TestHTMLEdgeCases:
    """Edge case tests for HTML export."""

    def test_empty_artifact(self):
        """Test with minimal empty artifact."""
        artifact = DebateArtifact()
        exporter = StaticHTMLExporter(artifact)

        # Should not raise
        html = exporter.generate()
        assert "<!DOCTYPE html>" in html

    def test_very_long_task(self):
        """Test with very long task."""
        artifact = DebateArtifact(task="A" * 10000)
        exporter = StaticHTMLExporter(artifact)

        html = exporter.generate()
        # Title should be truncated
        assert "..." in html

    def test_unicode_content(self):
        """Test with unicode content."""
        artifact = DebateArtifact(
            task="日本語タスク 🎯",
            agents=["エージェント1", "エージェント2"],
        )
        exporter = StaticHTMLExporter(artifact)

        html = exporter.generate()
        assert "日本語" in html

    def test_special_characters_in_all_fields(self):
        """Test special characters don't break HTML."""
        artifact = DebateArtifact(
            task="Task with <tags> & 'quotes' \"double\"",
            agents=["agent<1>", "agent&2"],
            graph_data={
                "nodes": {
                    "n1": {"content": "Content with <html> & entities"}
                }
            },
        )
        exporter = StaticHTMLExporter(artifact)

        # Should not raise
        html = exporter.generate()
        assert "<!DOCTYPE html>" in html

    def test_many_graph_nodes(self):
        """Test with many graph nodes."""
        nodes = {f"n{i}": {"node_type": "proposal", "content": f"Node {i}"} for i in range(50)}
        artifact = DebateArtifact(graph_data={"nodes": nodes})
        exporter = StaticHTMLExporter(artifact)

        html = exporter.generate()
        assert "graph-node" in html

    def test_many_verification_results(self):
        """Test with many verification results."""
        verifications = [
            VerificationResult(f"c{i}", f"Claim {i}", "verified", "z3")
            for i in range(20)
        ]
        artifact = DebateArtifact(verification_results=verifications)
        exporter = StaticHTMLExporter(artifact)

        html = exporter.generate()
        assert "verification-item" in html

    def test_nested_provenance_records(self):
        """Test with many provenance records (shows last 10)."""
        records = [
            {"id": f"ev-{i}", "source_type": "test", "content": f"Evidence {i}"}
            for i in range(20)
        ]
        artifact = DebateArtifact(
            provenance_data={"chain": {"records": records}}
        )
        exporter = StaticHTMLExporter(artifact)

        html = exporter.generate()
        assert "20 records" in html
        # Should show last 10
        assert "ev-19" in html
        assert "ev-10" in html

    def test_consensus_confidence_zero(self):
        """Test with zero confidence."""
        artifact = DebateArtifact(
            consensus_proof=ConsensusProof(
                reached=False,
                confidence=0.0,
                vote_breakdown={},
                final_answer="",
                rounds_used=5,
            ),
        )
        exporter = StaticHTMLExporter(artifact)

        html = exporter.generate()
        assert "0%" in html

    def test_consensus_confidence_one(self):
        """Test with 100% confidence."""
        artifact = DebateArtifact(
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=1.0,
                vote_breakdown={"a": True},
                final_answer="Answer",
                rounds_used=1,
            ),
        )
        exporter = StaticHTMLExporter(artifact)

        html = exporter.generate()
        assert "100%" in html
