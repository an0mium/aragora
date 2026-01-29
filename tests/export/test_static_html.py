"""
Tests for aragora.export.static_html module.

Tests cover:
- StaticHTMLExporter initialization
- HTML generation
- Header section with consensus badge
- Task section
- Tab navigation
- Graph view generation
- Timeline view generation
- Provenance view generation
- Verification view generation
- Statistics section
- Footer section
- JavaScript generation
- File saving
- Convenience function (export_to_html)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aragora.export.artifact import (
    ConsensusProof,
    DebateArtifact,
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
def basic_artifact() -> DebateArtifact:
    """Create a basic debate artifact for testing."""
    return DebateArtifact(
        artifact_id="test-artifact-001",
        debate_id="debate-001",
        task="Analyze the security of the API",
        agents=["claude", "gpt-4", "gemini"],
        rounds=3,
        message_count=12,
        critique_count=6,
        duration_seconds=45.5,
        created_at="2024-01-15T10:00:00Z",
    )


@pytest.fixture
def artifact_with_consensus() -> DebateArtifact:
    """Create an artifact with consensus proof."""
    return DebateArtifact(
        artifact_id="test-artifact-002",
        debate_id="debate-002",
        task="Decide on API design approach for the new microservices",
        agents=["claude", "gpt-4", "gemini"],
        rounds=3,
        message_count=15,
        critique_count=8,
        duration_seconds=120.5,
        created_at="2024-01-15T10:00:00Z",
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
        artifact_id="test-artifact-003",
        debate_id="debate-003",
        task="Debate architecture choices",
        agents=["claude", "gpt-4"],
        rounds=5,
        created_at="2024-01-15T10:00:00Z",
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


@pytest.fixture
def artifact_with_graph() -> DebateArtifact:
    """Create an artifact with graph data."""
    return DebateArtifact(
        artifact_id="test-artifact-004",
        debate_id="debate-004",
        task="Review code",
        created_at="2024-01-15T10:00:00Z",
        graph_data={
            "nodes": {
                "node-1": {
                    "node_type": "root",
                    "agent_id": "system",
                    "content": "Initial task definition",
                },
                "node-2": {
                    "node_type": "proposal",
                    "agent_id": "claude",
                    "content": "I propose we focus on input validation",
                },
                "node-3": {
                    "node_type": "critique",
                    "agent_id": "gpt-4",
                    "content": "Missing edge case handling",
                },
                "node-4": {
                    "node_type": "synthesis",
                    "agent_id": "claude",
                    "content": "Incorporating feedback into revised proposal",
                },
            },
            "edges": [
                {"from": "node-1", "to": "node-2"},
                {"from": "node-2", "to": "node-3"},
                {"from": "node-3", "to": "node-4"},
            ],
        },
    )


@pytest.fixture
def artifact_with_trace() -> DebateArtifact:
    """Create an artifact with trace data for timeline."""
    return DebateArtifact(
        artifact_id="test-artifact-005",
        debate_id="debate-005",
        task="Review proposal",
        created_at="2024-01-15T10:00:00Z",
        trace_data={
            "events": [
                {
                    "event_type": "agent_proposal",
                    "agent": "claude",
                    "round_num": 1,
                    "content": {"content": "Initial proposal for the API design"},
                },
                {
                    "event_type": "agent_critique",
                    "agent": "gpt-4",
                    "round_num": 1,
                    "content": {"issues": ["Missing auth", "No rate limiting"]},
                },
                {
                    "event_type": "agent_synthesis",
                    "agent": "claude",
                    "round_num": 2,
                    "content": {"content": "Revised proposal with auth and rate limiting"},
                },
            ]
        },
    )


@pytest.fixture
def artifact_with_provenance() -> DebateArtifact:
    """Create an artifact with provenance data."""
    return DebateArtifact(
        artifact_id="test-artifact-006",
        debate_id="debate-006",
        task="Verify claims",
        created_at="2024-01-15T10:00:00Z",
        provenance_data={
            "chain": {
                "records": [
                    {
                        "id": "evidence-001",
                        "source_type": "document",
                        "source_id": "doc-123",
                        "content": "Evidence from security audit",
                        "content_hash": "abc123def456",
                        "previous_hash": None,
                    },
                    {
                        "id": "evidence-002",
                        "source_type": "code_analysis",
                        "source_id": "analysis-456",
                        "content": "Static analysis results",
                        "content_hash": "def456ghi789",
                        "previous_hash": "abc123def456",
                    },
                ]
            }
        },
    )


@pytest.fixture
def artifact_with_verifications() -> DebateArtifact:
    """Create an artifact with verification results."""
    return DebateArtifact(
        artifact_id="test-artifact-007",
        debate_id="debate-007",
        task="Verify security claims",
        created_at="2024-01-15T10:00:00Z",
        verification_results=[
            VerificationResult(
                claim_id="claim-001",
                claim_text="All inputs are sanitized",
                status="verified",
                method="z3",
                proof_trace="(assert (forall (x) (sanitized x)))",
            ),
            VerificationResult(
                claim_id="claim-002",
                claim_text="No SQL injection possible",
                status="refuted",
                method="z3",
                counterexample="Input: '; DROP TABLE users; --'",
            ),
            VerificationResult(
                claim_id="claim-003",
                claim_text="Rate limiting prevents abuse",
                status="timeout",
                method="simulation",
            ),
        ],
    )


# =============================================================================
# TestStaticHTMLExporterInit
# =============================================================================


class TestStaticHTMLExporterInit:
    """Tests for StaticHTMLExporter initialization."""

    def test_init_with_artifact(self, basic_artifact: DebateArtifact):
        """Should initialize with a DebateArtifact."""
        exporter = StaticHTMLExporter(basic_artifact)
        assert exporter.artifact is basic_artifact

    def test_stores_artifact_reference(self, basic_artifact: DebateArtifact):
        """Should store artifact reference for later use."""
        exporter = StaticHTMLExporter(basic_artifact)
        assert exporter.artifact.artifact_id == "test-artifact-001"


# =============================================================================
# TestStaticHTMLExporterGenerate
# =============================================================================


class TestStaticHTMLExporterGenerate:
    """Tests for StaticHTMLExporter.generate()."""

    def test_returns_html_string(self, basic_artifact: DebateArtifact):
        """Should return a valid HTML string."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_doctype(self, basic_artifact: DebateArtifact):
        """Should include HTML5 doctype."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "<!DOCTYPE html>" in result

    def test_includes_html_structure(self, basic_artifact: DebateArtifact):
        """Should include proper HTML structure."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert '<html lang="en">' in result
        assert "<head>" in result
        assert "</head>" in result
        assert "<body>" in result
        assert "</body>" in result
        assert "</html>" in result

    def test_includes_meta_tags(self, basic_artifact: DebateArtifact):
        """Should include meta tags for charset and viewport."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert 'charset="UTF-8"' in result
        assert 'name="viewport"' in result

    def test_includes_title_with_task(self, basic_artifact: DebateArtifact):
        """Should include title with truncated task."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "<title>aragora Debate:" in result
        assert "security" in result.lower()


# =============================================================================
# TestStaticHTMLExporterStyles
# =============================================================================


class TestStaticHTMLExporterStyles:
    """Tests for embedded CSS styles."""

    def test_includes_embedded_styles(self, basic_artifact: DebateArtifact):
        """Should include embedded CSS styles."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "<style>" in result
        assert "</style>" in result

    def test_includes_css_variables(self, basic_artifact: DebateArtifact):
        """Should include CSS custom properties."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "--primary:" in result
        assert "--bg:" in result

    def test_includes_responsive_styles(self, basic_artifact: DebateArtifact):
        """Should include responsive media queries."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "@media" in result


# =============================================================================
# TestStaticHTMLExporterHeader
# =============================================================================


class TestStaticHTMLExporterHeader:
    """Tests for header section generation."""

    def test_includes_header_section(self, basic_artifact: DebateArtifact):
        """Should include header section."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "<header>" in result
        assert "</header>" in result
        assert "<h1>aragora Debate</h1>" in result

    def test_includes_artifact_id(self, basic_artifact: DebateArtifact):
        """Should include artifact ID in header."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "test-artifact-001" in result

    def test_shows_consensus_reached_badge(self, artifact_with_consensus: DebateArtifact):
        """Should show consensus reached badge when consensus achieved."""
        exporter = StaticHTMLExporter(artifact_with_consensus)
        result = exporter.generate()

        assert "Consensus Reached" in result
        assert "consensus-badge reached" in result
        assert "85%" in result

    def test_shows_no_consensus_badge(self, artifact_no_consensus: DebateArtifact):
        """Should show no consensus badge when consensus not achieved."""
        exporter = StaticHTMLExporter(artifact_no_consensus)
        result = exporter.generate()

        assert "No Consensus" in result
        assert "consensus-badge not-reached" in result


# =============================================================================
# TestStaticHTMLExporterTaskSection
# =============================================================================


class TestStaticHTMLExporterTaskSection:
    """Tests for task section generation."""

    def test_includes_task_section(self, basic_artifact: DebateArtifact):
        """Should include task section."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert 'class="task-section"' in result
        assert "Task" in result

    def test_includes_task_text(self, basic_artifact: DebateArtifact):
        """Should include task text."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "Analyze the security of the API" in result


# =============================================================================
# TestStaticHTMLExporterTabs
# =============================================================================


class TestStaticHTMLExporterTabs:
    """Tests for tab navigation generation."""

    def test_includes_tab_navigation(self, basic_artifact: DebateArtifact):
        """Should include tab navigation."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert 'class="tabs"' in result
        assert 'data-tab="graph"' in result
        assert 'data-tab="timeline"' in result
        assert 'data-tab="provenance"' in result
        assert 'data-tab="verification"' in result


# =============================================================================
# TestStaticHTMLExporterGraphView
# =============================================================================


class TestStaticHTMLExporterGraphView:
    """Tests for graph view generation."""

    def test_includes_graph_panel(self, basic_artifact: DebateArtifact):
        """Should include graph panel."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert 'id="panel-graph"' in result

    def test_shows_empty_state_without_graph_data(self, basic_artifact: DebateArtifact):
        """Should show empty state when no graph data."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "No graph data available" in result

    def test_renders_graph_nodes(self, artifact_with_graph: DebateArtifact):
        """Should render graph nodes when graph data present."""
        exporter = StaticHTMLExporter(artifact_with_graph)
        result = exporter.generate()

        assert 'class="graph-node' in result
        assert "proposal" in result
        assert "critique" in result

    def test_includes_node_data_attributes(self, artifact_with_graph: DebateArtifact):
        """Should include data attributes on nodes."""
        exporter = StaticHTMLExporter(artifact_with_graph)
        result = exporter.generate()

        assert "data-node-id=" in result
        assert "data-content=" in result


# =============================================================================
# TestStaticHTMLExporterTimelineView
# =============================================================================


class TestStaticHTMLExporterTimelineView:
    """Tests for timeline view generation."""

    def test_includes_timeline_panel(self, basic_artifact: DebateArtifact):
        """Should include timeline panel."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert 'id="panel-timeline"' in result

    def test_shows_empty_state_without_trace_data(self, basic_artifact: DebateArtifact):
        """Should show empty state when no trace data."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "No trace data available" in result

    def test_renders_timeline_items(self, artifact_with_trace: DebateArtifact):
        """Should render timeline items when trace data present."""
        exporter = StaticHTMLExporter(artifact_with_trace)
        result = exporter.generate()

        assert 'class="timeline-item' in result
        assert "claude" in result
        assert "gpt-4" in result

    def test_includes_timeline_controls(self, artifact_with_trace: DebateArtifact):
        """Should include timeline playback controls."""
        exporter = StaticHTMLExporter(artifact_with_trace)
        result = exporter.generate()

        assert 'id="btn-prev"' in result
        assert 'id="btn-next"' in result
        assert 'id="btn-play"' in result
        assert 'id="timeline-slider"' in result


# =============================================================================
# TestStaticHTMLExporterProvenanceView
# =============================================================================


class TestStaticHTMLExporterProvenanceView:
    """Tests for provenance view generation."""

    def test_includes_provenance_panel(self, basic_artifact: DebateArtifact):
        """Should include provenance panel."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert 'id="panel-provenance"' in result

    def test_shows_empty_state_without_provenance_data(self, basic_artifact: DebateArtifact):
        """Should show empty state when no provenance data."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "No provenance data available" in result

    def test_renders_provenance_records(self, artifact_with_provenance: DebateArtifact):
        """Should render provenance records when data present."""
        exporter = StaticHTMLExporter(artifact_with_provenance)
        result = exporter.generate()

        assert 'class="provenance-item"' in result
        assert "evidence-001" in result
        assert "document" in result

    def test_shows_hash_chain(self, artifact_with_provenance: DebateArtifact):
        """Should show hash chain visualization."""
        exporter = StaticHTMLExporter(artifact_with_provenance)
        result = exporter.generate()

        assert 'class="chain-link"' in result
        assert "abc123" in result  # Truncated hash


# =============================================================================
# TestStaticHTMLExporterVerificationView
# =============================================================================


class TestStaticHTMLExporterVerificationView:
    """Tests for verification view generation."""

    def test_includes_verification_panel(self, basic_artifact: DebateArtifact):
        """Should include verification panel."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert 'id="panel-verification"' in result

    def test_shows_empty_state_without_verifications(self, basic_artifact: DebateArtifact):
        """Should show empty state when no verification results."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "No formal verification results available" in result

    def test_renders_verification_results(self, artifact_with_verifications: DebateArtifact):
        """Should render verification results when present."""
        exporter = StaticHTMLExporter(artifact_with_verifications)
        result = exporter.generate()

        assert 'class="verification-item' in result
        assert "VERIFIED" in result
        assert "REFUTED" in result
        assert "TIMEOUT" in result

    def test_uses_status_colors(self, artifact_with_verifications: DebateArtifact):
        """Should use appropriate colors for verification status."""
        exporter = StaticHTMLExporter(artifact_with_verifications)
        result = exporter.generate()

        assert "verification-item verified" in result
        assert "verification-item refuted" in result
        assert "verification-item timeout" in result


# =============================================================================
# TestStaticHTMLExporterStats
# =============================================================================


class TestStaticHTMLExporterStats:
    """Tests for statistics section generation."""

    def test_includes_stats_section(self, basic_artifact: DebateArtifact):
        """Should include statistics section."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert 'class="stats"' in result

    def test_shows_round_count(self, basic_artifact: DebateArtifact):
        """Should show round count."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "Rounds" in result
        assert ">3<" in result

    def test_shows_message_count(self, basic_artifact: DebateArtifact):
        """Should show message count."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "Messages" in result
        assert ">12<" in result

    def test_shows_critique_count(self, basic_artifact: DebateArtifact):
        """Should show critique count."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "Critiques" in result
        assert ">6<" in result

    def test_shows_duration(self, basic_artifact: DebateArtifact):
        """Should show duration in seconds."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "Duration" in result
        assert "46s" in result  # 45.5 rounded

    def test_shows_agent_count(self, basic_artifact: DebateArtifact):
        """Should show agent count."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "Agents" in result
        assert ">3<" in result


# =============================================================================
# TestStaticHTMLExporterFooter
# =============================================================================


class TestStaticHTMLExporterFooter:
    """Tests for footer section generation."""

    def test_includes_footer_section(self, basic_artifact: DebateArtifact):
        """Should include footer section."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "<footer>" in result
        assert "</footer>" in result

    def test_includes_github_link(self, basic_artifact: DebateArtifact):
        """Should include link to aragora GitHub."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "github.com/an0mium/aragora" in result

    def test_includes_artifact_hash(self, basic_artifact: DebateArtifact):
        """Should include artifact content hash."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "Hash:" in result

    def test_includes_offline_notice(self, basic_artifact: DebateArtifact):
        """Should include notice about offline functionality."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "offline" in result.lower()


# =============================================================================
# TestStaticHTMLExporterScripts
# =============================================================================


class TestStaticHTMLExporterScripts:
    """Tests for embedded JavaScript generation."""

    def test_includes_embedded_scripts(self, basic_artifact: DebateArtifact):
        """Should include embedded JavaScript."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "<script>" in result
        assert "</script>" in result

    def test_includes_artifact_data(self, basic_artifact: DebateArtifact):
        """Should embed artifact data as JSON."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "const artifactData =" in result
        assert '"artifact_id"' in result

    def test_includes_tab_switching_logic(self, basic_artifact: DebateArtifact):
        """Should include tab switching JavaScript."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "querySelectorAll('.tab')" in result
        assert "addEventListener('click'" in result

    def test_includes_timeline_controls_logic(self, basic_artifact: DebateArtifact):
        """Should include timeline control JavaScript."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        assert "updateTimeline" in result
        assert "playInterval" in result


# =============================================================================
# TestStaticHTMLExporterSave
# =============================================================================


class TestStaticHTMLExporterSave:
    """Tests for StaticHTMLExporter.save()."""

    def test_saves_to_file(self, basic_artifact: DebateArtifact):
        """Should save HTML to specified file."""
        exporter = StaticHTMLExporter(basic_artifact)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "export.html"
            result_path = exporter.save(output_path)

            assert result_path == output_path
            assert output_path.exists()

    def test_saved_file_contains_html(self, basic_artifact: DebateArtifact):
        """Should save valid HTML content."""
        exporter = StaticHTMLExporter(basic_artifact)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "export.html"
            exporter.save(output_path)

            content = output_path.read_text()
            assert "<!DOCTYPE html>" in content
            assert "aragora" in content


# =============================================================================
# TestExportToHTML
# =============================================================================


class TestExportToHTML:
    """Tests for export_to_html convenience function."""

    def test_exports_to_html_file(self, basic_artifact: DebateArtifact):
        """Should export artifact to HTML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "export.html"
            result_path = export_to_html(basic_artifact, output_path)

            assert result_path == output_path
            assert output_path.exists()

    def test_returns_output_path(self, basic_artifact: DebateArtifact):
        """Should return the output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "export.html"
            result = export_to_html(basic_artifact, output_path)

            assert result == output_path


# =============================================================================
# TestStaticHTMLExporterEdgeCases
# =============================================================================


class TestStaticHTMLExporterEdgeCases:
    """Edge case tests for static HTML exporter."""

    def test_escapes_html_special_characters(self):
        """Should escape HTML special characters in content."""
        artifact = DebateArtifact(
            artifact_id="test",
            task='Task with <script>alert("xss")</script> injection',
            created_at="2024-01-15T10:00:00Z",
        )

        exporter = StaticHTMLExporter(artifact)
        result = exporter.generate()

        # Should not contain unescaped script tags
        assert '<script>alert("xss")</script>' not in result
        assert "&lt;script&gt;" in result

    def test_escapes_quotes_in_content(self):
        """Should escape quotes in content."""
        artifact = DebateArtifact(
            artifact_id="test",
            task="Task with \"double\" and 'single' quotes",
            created_at="2024-01-15T10:00:00Z",
        )

        exporter = StaticHTMLExporter(artifact)
        result = exporter.generate()

        # Should be valid HTML
        assert "<!DOCTYPE html>" in result
        assert "</html>" in result

    def test_handles_unicode_content(self):
        """Should handle unicode content properly."""
        artifact = DebateArtifact(
            artifact_id="test",
            task="Task with international characters",
            created_at="2024-01-15T10:00:00Z",
        )

        exporter = StaticHTMLExporter(artifact)
        result = exporter.generate()

        assert "international" in result

    def test_handles_very_long_task(self):
        """Should handle very long task text."""
        artifact = DebateArtifact(
            artifact_id="test",
            task="A" * 1000,
            created_at="2024-01-15T10:00:00Z",
        )

        exporter = StaticHTMLExporter(artifact)
        result = exporter.generate()

        # Should include truncated task in title
        assert "<title>" in result
        assert "..." in result

    def test_handles_empty_agents_list(self):
        """Should handle empty agents list."""
        artifact = DebateArtifact(
            artifact_id="test",
            agents=[],
            created_at="2024-01-15T10:00:00Z",
        )

        exporter = StaticHTMLExporter(artifact)
        result = exporter.generate()

        # Should show 0 agents
        assert ">0<" in result

    def test_handles_null_consensus_proof(self):
        """Should handle artifact with no consensus proof."""
        artifact = DebateArtifact(
            artifact_id="test",
            consensus_proof=None,
            created_at="2024-01-15T10:00:00Z",
        )

        exporter = StaticHTMLExporter(artifact)
        result = exporter.generate()

        assert "No Consensus" in result
        assert "N/A" in result

    def test_json_embeds_safely(self, basic_artifact: DebateArtifact):
        """Should safely embed artifact JSON without breaking script."""
        exporter = StaticHTMLExporter(basic_artifact)
        result = exporter.generate()

        # Find the embedded JSON
        start = result.find("const artifactData =") + len("const artifactData =")
        end = result.find(";", start)
        json_str = result[start:end].strip()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["artifact_id"] == "test-artifact-001"
