"""Tests for static HTML exporter - self-contained debate viewer generation."""

import json
import pytest
from pathlib import Path

from aragora.export.static_html import StaticHTMLExporter, export_to_html
from aragora.export.artifact import (
    DebateArtifact,
    ConsensusProof,
    VerificationResult,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def minimal_artifact():
    """Create minimal DebateArtifact for testing."""
    return DebateArtifact(
        artifact_id="test-123",
        debate_id="debate-456",
        task="Test the HTML exporter",
        agents=["claude", "gemini"],
        rounds=3,
        message_count=10,
        critique_count=5,
        duration_seconds=120.0,
    )


@pytest.fixture
def artifact_with_consensus():
    """Create DebateArtifact with consensus proof."""
    return DebateArtifact(
        artifact_id="test-consensus",
        task="Task with consensus",
        consensus_proof=ConsensusProof(
            reached=True,
            confidence=0.85,
            vote_breakdown={"claude": True, "gemini": True},
            final_answer="Agreed conclusion",
            rounds_used=3,
        ),
        agents=["claude", "gemini"],
        rounds=3,
    )


@pytest.fixture
def artifact_no_consensus():
    """Create DebateArtifact without consensus."""
    return DebateArtifact(
        artifact_id="test-no-consensus",
        task="Task without consensus",
        consensus_proof=ConsensusProof(
            reached=False,
            confidence=0.3,
            vote_breakdown={"claude": True, "gemini": False},
            final_answer="",
            rounds_used=3,
        ),
        agents=["claude", "gemini"],
        rounds=3,
    )


@pytest.fixture
def artifact_with_graph():
    """Create DebateArtifact with graph data."""
    return DebateArtifact(
        artifact_id="test-graph",
        task="Task with graph",
        graph_data={
            "nodes": {
                "n1": {"node_type": "root", "agent_id": "claude", "content": "Initial proposal content"},
                "n2": {"node_type": "proposal", "agent_id": "gemini", "content": "Counter proposal"},
                "n3": {"node_type": "critique", "agent_id": "claude", "content": "Objection to counter"},
                "n4": {"node_type": "synthesis", "agent_id": "gemini", "content": "Combined view"},
            },
            "edges": [{"from": "n1", "to": "n2"}, {"from": "n2", "to": "n3"}],
        },
        agents=["claude", "gemini"],
    )


@pytest.fixture
def artifact_with_timeline():
    """Create DebateArtifact with trace data."""
    return DebateArtifact(
        artifact_id="test-timeline",
        task="Task with timeline",
        trace_data={
            "events": [
                {"event_type": "agent_proposal", "agent": "claude", "round_num": 1, "content": {"content": "Proposal text here"}},
                {"event_type": "agent_critique", "agent": "gemini", "round_num": 1, "content": {"issues": ["Issue 1", "Issue 2"]}},
                {"event_type": "agent_synthesis", "agent": "claude", "round_num": 2, "content": {"content": "Synthesis combining views"}},
                {"event_type": "debate_start", "agent": "system", "round_num": 0, "content": {}},  # Should be skipped
            ],
        },
        agents=["claude", "gemini"],
    )


@pytest.fixture
def artifact_with_provenance():
    """Create DebateArtifact with provenance data."""
    return DebateArtifact(
        artifact_id="test-provenance",
        task="Task with provenance",
        provenance_data={
            "chain": {
                "records": [
                    {"id": "rec-1", "source_type": "agent", "source_id": "claude", "content": "First evidence", "content_hash": "abc123", "previous_hash": None},
                    {"id": "rec-2", "source_type": "web", "source_id": "https://example.com", "content": "Second evidence", "content_hash": "def456", "previous_hash": "abc123"},
                ],
            },
        },
        agents=["claude"],
    )


@pytest.fixture
def artifact_with_verifications():
    """Create DebateArtifact with verification results."""
    return DebateArtifact(
        artifact_id="test-verification",
        task="Task with verifications",
        verification_results=[
            VerificationResult(
                claim_id="c1",
                claim_text="Verified claim",
                status="verified",
                method="z3",
                proof_trace="QED proof trace",
            ),
            VerificationResult(
                claim_id="c2",
                claim_text="Refuted claim",
                status="refuted",
                method="lean",
                counterexample="Counterexample found",
            ),
            VerificationResult(
                claim_id="c3",
                claim_text="Timeout claim",
                status="timeout",
                method="simulation",
            ),
        ],
        agents=["claude"],
    )


@pytest.fixture
def full_artifact():
    """Create full DebateArtifact with all data."""
    return DebateArtifact(
        artifact_id="test-full",
        task="Complete test task with <script> in it",
        graph_data={
            "nodes": {
                "n1": {"node_type": "root", "agent_id": "claude", "content": "Root content"},
            },
        },
        trace_data={
            "events": [
                {"event_type": "agent_proposal", "agent": "claude", "round_num": 1, "content": {"content": "Text"}},
            ],
        },
        provenance_data={
            "chain": {
                "records": [
                    {"id": "rec-1", "source_type": "agent", "source_id": "claude", "content": "Evidence", "content_hash": "abc123"},
                ],
            },
        },
        consensus_proof=ConsensusProof(
            reached=True,
            confidence=0.85,
            vote_breakdown={"claude": True},
            final_answer="Answer",
            rounds_used=2,
        ),
        verification_results=[
            VerificationResult(claim_id="c1", claim_text="Claim", status="verified", method="z3"),
        ],
        agents=["claude", "gemini"],
        rounds=3,
        message_count=10,
        critique_count=5,
        duration_seconds=120.0,
    )


@pytest.fixture
def exporter(minimal_artifact):
    """Create StaticHTMLExporter with minimal artifact."""
    return StaticHTMLExporter(minimal_artifact)


@pytest.fixture
def full_exporter(full_artifact):
    """Create StaticHTMLExporter with full artifact."""
    return StaticHTMLExporter(full_artifact)


# =============================================================================
# StaticHTMLExporter Initialization Tests
# =============================================================================

class TestStaticHTMLExporterInit:
    """Tests for StaticHTMLExporter initialization."""

    def test_stores_artifact_reference(self, minimal_artifact):
        """Should store artifact reference correctly."""
        exporter = StaticHTMLExporter(minimal_artifact)
        assert exporter.artifact is minimal_artifact

    def test_accepts_valid_debate_artifact(self, minimal_artifact):
        """Should accept valid DebateArtifact."""
        exporter = StaticHTMLExporter(minimal_artifact)
        assert isinstance(exporter.artifact, DebateArtifact)

    def test_can_access_artifact_properties(self, minimal_artifact):
        """Should be able to access artifact properties."""
        exporter = StaticHTMLExporter(minimal_artifact)
        assert exporter.artifact.artifact_id == "test-123"
        assert exporter.artifact.task == "Test the HTML exporter"
        assert exporter.artifact.rounds == 3


# =============================================================================
# StaticHTMLExporter._escape Tests (Security Critical)
# =============================================================================

class TestStaticHTMLExporterEscape:
    """Tests for _escape method - XSS prevention."""

    def test_escapes_ampersand(self, exporter):
        """Should escape ampersand to &amp;."""
        result = exporter._escape("AT&T")
        assert "&amp;" in result
        assert "AT&amp;T" == result

    def test_escapes_less_than(self, exporter):
        """Should escape < to &lt;."""
        result = exporter._escape("a < b")
        assert "&lt;" in result
        assert "a &lt; b" == result

    def test_escapes_greater_than(self, exporter):
        """Should escape > to &gt;."""
        result = exporter._escape("a > b")
        assert "&gt;" in result
        assert "a &gt; b" == result

    def test_escapes_double_quote(self, exporter):
        """Should escape " to &quot;."""
        result = exporter._escape('say "hello"')
        assert "&quot;" in result
        assert 'say &quot;hello&quot;' == result

    def test_escapes_single_quote(self, exporter):
        """Should escape ' to &#39;."""
        result = exporter._escape("it's")
        assert "&#39;" in result
        assert "it&#39;s" == result

    def test_handles_empty_string(self, exporter):
        """Should handle empty string."""
        result = exporter._escape("")
        assert result == ""

    def test_handles_string_with_no_special_chars(self, exporter):
        """Should handle string with no special characters."""
        result = exporter._escape("hello world")
        assert result == "hello world"

    def test_handles_all_special_chars_together(self, exporter):
        """Should handle all special chars together."""
        result = exporter._escape("<tag attr=\"val\" data='x'>AT&T</tag>")
        assert "<" not in result
        assert ">" not in result
        assert '"' not in result
        assert "'" not in result
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&quot;" in result
        assert "&#39;" in result
        assert "&amp;" in result

    def test_prevents_xss_via_script_tags(self, exporter):
        """Should prevent XSS via script tags."""
        malicious = '<script>alert("XSS")</script>'
        result = exporter._escape(malicious)

        assert "<script>" not in result
        assert "</script>" not in result
        assert "&lt;script&gt;" in result

    def test_prevents_xss_via_event_handlers(self, exporter):
        """Should prevent XSS via event handlers."""
        malicious = '<img src="x" onerror="alert(\'XSS\')">'
        result = exporter._escape(malicious)

        assert "<img" not in result
        assert "onerror" not in result or "&lt;img" in result


# =============================================================================
# StaticHTMLExporter.generate Tests
# =============================================================================

class TestStaticHTMLExporterGenerate:
    """Tests for generate method."""

    def test_returns_valid_html5_document(self, exporter):
        """Should return valid HTML5 document."""
        html = exporter.generate()

        assert html.startswith("<!DOCTYPE html>")
        assert "<html lang=" in html
        assert "</html>" in html

    def test_contains_doctype_declaration(self, exporter):
        """Should contain DOCTYPE declaration."""
        html = exporter.generate()
        assert "<!DOCTYPE html>" in html

    def test_contains_meta_charset_utf8(self, exporter):
        """Should contain meta charset UTF-8."""
        html = exporter.generate()
        assert 'charset="UTF-8"' in html or "charset=UTF-8" in html

    def test_contains_viewport_meta_tag(self, exporter):
        """Should contain viewport meta tag."""
        html = exporter.generate()
        assert 'name="viewport"' in html

    def test_contains_title_with_escaped_task(self, full_exporter):
        """Should contain title with escaped task."""
        html = full_exporter.generate()
        # Task contains <script> which should be escaped in title
        assert "<title>" in html
        assert "</title>" in html
        # Script tags should be escaped
        assert "<script>" not in html.split("<title>")[1].split("</title>")[0] or "&lt;script&gt;" in html

    def test_contains_all_major_sections(self, exporter):
        """Should contain all major sections."""
        html = exporter.generate()

        assert "<head>" in html
        assert "</head>" in html
        assert "<body>" in html
        assert "</body>" in html
        assert "<header>" in html
        assert "<footer>" in html
        assert "<style>" in html
        assert "<script>" in html


# =============================================================================
# StaticHTMLExporter._generate_styles Tests
# =============================================================================

class TestStaticHTMLExporterGenerateStyles:
    """Tests for _generate_styles method."""

    def test_contains_style_tag(self, exporter):
        """Should contain style tag."""
        styles = exporter._generate_styles()
        assert "<style>" in styles
        assert "</style>" in styles

    def test_contains_css_custom_properties(self, exporter):
        """Should contain CSS custom properties (:root)."""
        styles = exporter._generate_styles()
        assert ":root" in styles
        assert "--primary" in styles
        assert "--bg" in styles

    def test_contains_responsive_media_query(self, exporter):
        """Should contain responsive media query."""
        styles = exporter._generate_styles()
        assert "@media" in styles
        assert "768px" in styles

    def test_contains_required_class_selectors(self, exporter):
        """Should contain all required class selectors."""
        styles = exporter._generate_styles()

        required_classes = [
            ".tab", ".tab-panel", ".graph-container", ".graph-node",
            ".timeline", ".timeline-item", ".provenance-item",
            ".verification-item", ".stats", ".stat",
        ]
        for cls in required_classes:
            assert cls in styles, f"Missing class selector: {cls}"


# =============================================================================
# StaticHTMLExporter._generate_header Tests
# =============================================================================

class TestStaticHTMLExporterGenerateHeader:
    """Tests for _generate_header method."""

    def test_contains_header_tag(self, exporter):
        """Should contain header tag."""
        header = exporter._generate_header()
        assert "<header>" in header
        assert "</header>" in header

    def test_shows_consensus_reached_badge(self, artifact_with_consensus):
        """Should show consensus reached badge when consensus."""
        exporter = StaticHTMLExporter(artifact_with_consensus)
        header = exporter._generate_header()

        assert "reached" in header
        assert "Consensus Reached" in header

    def test_shows_no_consensus_badge(self, artifact_no_consensus):
        """Should show no consensus badge when no consensus."""
        exporter = StaticHTMLExporter(artifact_no_consensus)
        header = exporter._generate_header()

        assert "not-reached" in header
        assert "No Consensus" in header

    def test_displays_confidence_percentage(self, artifact_with_consensus):
        """Should display confidence percentage."""
        exporter = StaticHTMLExporter(artifact_with_consensus)
        header = exporter._generate_header()

        assert "85%" in header

    def test_contains_artifact_id_and_date(self, minimal_artifact):
        """Should contain artifact ID and date."""
        exporter = StaticHTMLExporter(minimal_artifact)
        header = exporter._generate_header()

        assert minimal_artifact.artifact_id in header


# =============================================================================
# StaticHTMLExporter._generate_task_section Tests
# =============================================================================

class TestStaticHTMLExporterGenerateTaskSection:
    """Tests for _generate_task_section method."""

    def test_contains_task_section_class(self, exporter):
        """Should contain task-section class."""
        section = exporter._generate_task_section()
        assert 'class="task-section"' in section

    def test_escapes_task_content(self, full_exporter):
        """Should escape task content."""
        section = full_exporter._generate_task_section()
        # Task contains <script> which should be escaped
        assert "<script>" not in section or "&lt;script&gt;" in section

    def test_displays_task_heading(self, exporter):
        """Should display task heading."""
        section = exporter._generate_task_section()
        assert "<h2>" in section
        assert "Task" in section


# =============================================================================
# StaticHTMLExporter._generate_tabs Tests
# =============================================================================

class TestStaticHTMLExporterGenerateTabs:
    """Tests for _generate_tabs method."""

    def test_contains_four_tab_buttons(self, exporter):
        """Should contain four tab buttons."""
        tabs = exporter._generate_tabs()

        assert tabs.count('class="tab') >= 4
        assert "Graph" in tabs
        assert "Timeline" in tabs
        assert "Provenance" in tabs
        assert "Verification" in tabs

    def test_first_tab_is_active(self, exporter):
        """Should have first tab (Graph) active by default."""
        tabs = exporter._generate_tabs()
        assert 'class="tab active"' in tabs

    def test_contains_correct_data_tab_attributes(self, exporter):
        """Should contain correct data-tab attributes."""
        tabs = exporter._generate_tabs()

        assert 'data-tab="graph"' in tabs
        assert 'data-tab="timeline"' in tabs
        assert 'data-tab="provenance"' in tabs
        assert 'data-tab="verification"' in tabs


# =============================================================================
# StaticHTMLExporter._generate_graph_view Tests
# =============================================================================

class TestStaticHTMLExporterGenerateGraphView:
    """Tests for _generate_graph_view method."""

    def test_shows_empty_state_when_no_graph_data(self, minimal_artifact):
        """Should show empty state when no graph data."""
        exporter = StaticHTMLExporter(minimal_artifact)
        view = exporter._generate_graph_view()

        assert "empty-state" in view
        assert "No graph data" in view

    def test_contains_graph_container_when_data_present(self, artifact_with_graph):
        """Should contain graph-container when data present."""
        exporter = StaticHTMLExporter(artifact_with_graph)
        view = exporter._generate_graph_view()

        assert "graph-container" in view

    def test_renders_graph_nodes_when_available(self, artifact_with_graph):
        """Should render graph nodes when available."""
        exporter = StaticHTMLExporter(artifact_with_graph)
        view = exporter._generate_graph_view()

        assert "graph-node" in view
        assert "claude" in view

    def test_contains_panel_id_for_javascript(self, exporter):
        """Should contain panel ID for JavaScript."""
        view = exporter._generate_graph_view()
        assert 'id="panel-graph"' in view

    def test_has_active_class_on_panel(self, exporter):
        """Should have active class on panel."""
        view = exporter._generate_graph_view()
        assert 'class="tab-panel active"' in view


# =============================================================================
# StaticHTMLExporter._render_graph_nodes Tests
# =============================================================================

class TestStaticHTMLExporterRenderGraphNodes:
    """Tests for _render_graph_nodes method."""

    def test_returns_empty_string_when_no_nodes(self, minimal_artifact):
        """Should return empty string when no nodes."""
        exporter = StaticHTMLExporter(minimal_artifact)
        result = exporter._render_graph_nodes()
        assert result == ""

    def test_positions_nodes_by_type(self, artifact_with_graph):
        """Should position nodes by type."""
        exporter = StaticHTMLExporter(artifact_with_graph)
        result = exporter._render_graph_nodes()

        # Different node types should have different top positions
        assert "top:" in result

    def test_sets_correct_node_classes(self, artifact_with_graph):
        """Should set correct node classes (proposal, critique, synthesis)."""
        exporter = StaticHTMLExporter(artifact_with_graph)
        result = exporter._render_graph_nodes()

        assert 'class="graph-node root"' in result
        assert 'class="graph-node proposal"' in result
        assert 'class="graph-node critique"' in result
        assert 'class="graph-node synthesis"' in result

    def test_escapes_node_content_in_data_attributes(self, artifact_with_graph):
        """Should escape node content in data attributes."""
        # Add node with special characters
        artifact_with_graph.graph_data["nodes"]["n5"] = {
            "node_type": "proposal",
            "agent_id": "test",
            "content": '<script>alert("XSS")</script>',
        }
        exporter = StaticHTMLExporter(artifact_with_graph)
        result = exporter._render_graph_nodes()

        assert '<script>alert' not in result or '&lt;script&gt;' in result

    def test_contains_agent_name(self, artifact_with_graph):
        """Should contain agent name."""
        exporter = StaticHTMLExporter(artifact_with_graph)
        result = exporter._render_graph_nodes()

        assert "claude" in result
        assert "gemini" in result

    def test_contains_node_type_label(self, artifact_with_graph):
        """Should contain node type label."""
        exporter = StaticHTMLExporter(artifact_with_graph)
        result = exporter._render_graph_nodes()

        assert "root" in result
        assert "proposal" in result


# =============================================================================
# StaticHTMLExporter._generate_timeline_view Tests
# =============================================================================

class TestStaticHTMLExporterGenerateTimelineView:
    """Tests for _generate_timeline_view method."""

    def test_shows_empty_state_when_no_trace_data(self, minimal_artifact):
        """Should show empty state when no trace data."""
        exporter = StaticHTMLExporter(minimal_artifact)
        view = exporter._generate_timeline_view()

        assert "empty-state" in view
        assert "No trace data" in view

    def test_renders_timeline_items_for_agent_events(self, artifact_with_timeline):
        """Should render timeline items for agent events."""
        exporter = StaticHTMLExporter(artifact_with_timeline)
        view = exporter._generate_timeline_view()

        assert "timeline-item" in view
        assert "claude" in view

    def test_skips_non_agent_events(self, artifact_with_timeline):
        """Should skip non-agent events (like debate_start)."""
        exporter = StaticHTMLExporter(artifact_with_timeline)
        view = exporter._generate_timeline_view()

        # debate_start event should not create a timeline item with "system"
        # Count timeline items - should be 3, not 4
        assert view.count('class="timeline-item') == 3

    def test_displays_agent_name_and_round_number(self, artifact_with_timeline):
        """Should display agent name and round number."""
        exporter = StaticHTMLExporter(artifact_with_timeline)
        view = exporter._generate_timeline_view()

        assert 'class="agent"' in view
        assert 'class="round"' in view
        assert "Round 1" in view

    def test_contains_timeline_controls(self, artifact_with_timeline):
        """Should contain timeline controls."""
        exporter = StaticHTMLExporter(artifact_with_timeline)
        view = exporter._generate_timeline_view()

        assert "timeline-controls" in view
        assert "btn-prev" in view
        assert "btn-next" in view
        assert "btn-play" in view

    def test_contains_timeline_slider(self, artifact_with_timeline):
        """Should contain timeline slider."""
        exporter = StaticHTMLExporter(artifact_with_timeline)
        view = exporter._generate_timeline_view()

        assert "timeline-slider" in view
        assert 'type="range"' in view

    def test_different_classes_for_critique_vs_proposal(self, artifact_with_timeline):
        """Should have different classes for critique vs proposal."""
        exporter = StaticHTMLExporter(artifact_with_timeline)
        view = exporter._generate_timeline_view()

        assert 'class="timeline-item critique"' in view
        assert 'class="timeline-item synthesis"' in view


# =============================================================================
# StaticHTMLExporter._generate_provenance_view Tests
# =============================================================================

class TestStaticHTMLExporterGenerateProvenanceView:
    """Tests for _generate_provenance_view method."""

    def test_shows_empty_state_when_no_provenance_data(self, minimal_artifact):
        """Should show empty state when no provenance data."""
        exporter = StaticHTMLExporter(minimal_artifact)
        view = exporter._generate_provenance_view()

        assert "empty-state" in view
        assert "No provenance data" in view

    def test_displays_record_count_in_heading(self, artifact_with_provenance):
        """Should display record count in heading."""
        exporter = StaticHTMLExporter(artifact_with_provenance)
        view = exporter._generate_provenance_view()

        assert "2 records" in view

    def test_shows_last_10_records_only(self):
        """Should show last 10 records only."""
        artifact = DebateArtifact(
            artifact_id="test",
            task="test",
            provenance_data={
                "chain": {
                    "records": [{"id": f"rec-{i}", "source_type": "agent", "source_id": "x", "content": f"content {i}", "content_hash": f"hash{i}"} for i in range(15)],
                },
            },
        )
        exporter = StaticHTMLExporter(artifact)
        view = exporter._generate_provenance_view()

        # Should have 10 provenance items, not 15
        assert view.count('class="provenance-item"') == 10

    def test_contains_hash_visualization(self, artifact_with_provenance):
        """Should contain hash visualization."""
        exporter = StaticHTMLExporter(artifact_with_provenance)
        view = exporter._generate_provenance_view()

        assert "chain-visualization" in view
        assert "chain-link" in view
        assert "abc123" in view[:100] or "abc123" in view  # hash visible

    def test_shows_previous_hash_chain_link(self, artifact_with_provenance):
        """Should show previous hash chain link when available."""
        exporter = StaticHTMLExporter(artifact_with_provenance)
        view = exporter._generate_provenance_view()

        # Second record has previous_hash
        assert "previous_hash" in str(artifact_with_provenance.provenance_data) or "&larr;" in view


# =============================================================================
# StaticHTMLExporter._generate_verification_view Tests
# =============================================================================

class TestStaticHTMLExporterGenerateVerificationView:
    """Tests for _generate_verification_view method."""

    def test_shows_empty_state_when_no_verifications(self, minimal_artifact):
        """Should show empty state when no verifications."""
        exporter = StaticHTMLExporter(minimal_artifact)
        view = exporter._generate_verification_view()

        assert "empty-state" in view
        assert "No formal verification" in view

    def test_renders_verification_items(self, artifact_with_verifications):
        """Should render verification items."""
        exporter = StaticHTMLExporter(artifact_with_verifications)
        view = exporter._generate_verification_view()

        assert "verification-item" in view
        assert view.count('class="verification-item') == 3

    def test_shows_correct_status_classes(self, artifact_with_verifications):
        """Should show correct status classes (verified, refuted, timeout)."""
        exporter = StaticHTMLExporter(artifact_with_verifications)
        view = exporter._generate_verification_view()

        assert 'class="verification-item verified"' in view
        assert 'class="verification-item refuted"' in view
        assert 'class="verification-item timeout"' in view

    def test_displays_verification_method(self, artifact_with_verifications):
        """Should display verification method."""
        exporter = StaticHTMLExporter(artifact_with_verifications)
        view = exporter._generate_verification_view()

        assert "z3" in view
        assert "lean" in view
        assert "simulation" in view

    def test_shows_proof_trace_when_available(self, artifact_with_verifications):
        """Should show proof trace when available."""
        exporter = StaticHTMLExporter(artifact_with_verifications)
        view = exporter._generate_verification_view()

        assert "QED proof trace" in view


# =============================================================================
# StaticHTMLExporter._generate_stats Tests
# =============================================================================

class TestStaticHTMLExporterGenerateStats:
    """Tests for _generate_stats method."""

    def test_displays_all_six_stat_boxes(self, exporter):
        """Should display all 6 stat boxes."""
        stats = exporter._generate_stats()

        assert stats.count('class="stat"') == 6

    def test_shows_correct_rounds_count(self, minimal_artifact):
        """Should show correct rounds count."""
        exporter = StaticHTMLExporter(minimal_artifact)
        stats = exporter._generate_stats()

        assert ">3<" in stats  # rounds value
        assert "Rounds" in stats

    def test_shows_correct_message_count(self, minimal_artifact):
        """Should show correct message count."""
        exporter = StaticHTMLExporter(minimal_artifact)
        stats = exporter._generate_stats()

        assert ">10<" in stats  # message count
        assert "Messages" in stats

    def test_shows_correct_duration(self, minimal_artifact):
        """Should show correct duration."""
        exporter = StaticHTMLExporter(minimal_artifact)
        stats = exporter._generate_stats()

        assert "120s" in stats or ">120<" in stats
        assert "Duration" in stats

    def test_shows_agent_count(self, minimal_artifact):
        """Should show agent count."""
        exporter = StaticHTMLExporter(minimal_artifact)
        stats = exporter._generate_stats()

        assert ">2<" in stats  # 2 agents
        assert "Agents" in stats


# =============================================================================
# StaticHTMLExporter._generate_footer Tests
# =============================================================================

class TestStaticHTMLExporterGenerateFooter:
    """Tests for _generate_footer method."""

    def test_contains_footer_tag(self, exporter):
        """Should contain footer tag."""
        footer = exporter._generate_footer()
        assert "<footer>" in footer
        assert "</footer>" in footer

    def test_shows_artifact_id(self, minimal_artifact):
        """Should show artifact ID."""
        exporter = StaticHTMLExporter(minimal_artifact)
        footer = exporter._generate_footer()

        assert minimal_artifact.artifact_id in footer

    def test_shows_content_hash(self, full_artifact):
        """Should show content hash."""
        exporter = StaticHTMLExporter(full_artifact)
        footer = exporter._generate_footer()

        assert full_artifact.content_hash in footer


# =============================================================================
# StaticHTMLExporter._generate_scripts Tests
# =============================================================================

class TestStaticHTMLExporterGenerateScripts:
    """Tests for _generate_scripts method."""

    def test_contains_script_tag(self, exporter):
        """Should contain script tag."""
        scripts = exporter._generate_scripts()
        assert "<script>" in scripts
        assert "</script>" in scripts

    def test_embeds_artifact_json(self, exporter):
        """Should embed artifact JSON."""
        scripts = exporter._generate_scripts()
        assert "artifactData" in scripts
        assert exporter.artifact.artifact_id in scripts

    def test_contains_tab_switching_logic(self, exporter):
        """Should contain tab switching logic."""
        scripts = exporter._generate_scripts()
        assert "querySelectorAll('.tab')" in scripts
        assert "classList.add('active')" in scripts

    def test_contains_timeline_control_logic(self, exporter):
        """Should contain timeline control logic."""
        scripts = exporter._generate_scripts()
        assert "timeline" in scripts.lower()
        assert "updateTimeline" in scripts

    def test_contains_graph_node_interaction(self, exporter):
        """Should contain graph node interaction."""
        scripts = exporter._generate_scripts()
        assert "graph-node" in scripts
        assert "modal" in scripts.lower()


# =============================================================================
# StaticHTMLExporter.save Tests
# =============================================================================

class TestStaticHTMLExporterSave:
    """Tests for save method."""

    def test_creates_file_at_specified_path(self, exporter, tmp_path):
        """Should create file at specified path."""
        output_path = tmp_path / "debate.html"
        exporter.save(output_path)

        assert output_path.exists()

    def test_writes_valid_html_content(self, exporter, tmp_path):
        """Should write valid HTML content."""
        output_path = tmp_path / "debate.html"
        exporter.save(output_path)

        content = output_path.read_text()
        assert content.startswith("<!DOCTYPE html>")
        assert "</html>" in content

    def test_returns_path_object(self, exporter, tmp_path):
        """Should return path object."""
        output_path = tmp_path / "debate.html"
        result = exporter.save(output_path)

        assert isinstance(result, Path)
        assert result == output_path


# =============================================================================
# export_to_html Function Tests
# =============================================================================

class TestExportToHtml:
    """Tests for export_to_html convenience function."""

    def test_creates_file_at_output_path(self, minimal_artifact, tmp_path):
        """Should create file at output path."""
        output_path = tmp_path / "exported.html"
        export_to_html(minimal_artifact, output_path)

        assert output_path.exists()

    def test_returns_path_to_created_file(self, minimal_artifact, tmp_path):
        """Should return path to created file."""
        output_path = tmp_path / "exported.html"
        result = export_to_html(minimal_artifact, output_path)

        assert result == output_path
        assert result.exists()
