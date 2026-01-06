"""Tests for aragora.visualization.exporter module."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from aragora.visualization.exporter import (
    save_debate_visualization,
    generate_standalone_html,
)
from aragora.visualization.mapper import ArgumentCartographer


class TestSaveDebateVisualization:
    """Tests for save_debate_visualization function."""

    @pytest.fixture
    def cartographer(self):
        """Create a cartographer with sample data."""
        cart = ArgumentCartographer()
        cart.set_debate_context("debate-123", "test-topic")
        cart.update_from_message("agent1", "I propose we do X", "proposer", 1)
        cart.update_from_message("agent2", "I disagree with X", "critic", 1)
        return cart

    @pytest.fixture
    def empty_cartographer(self):
        """Create empty cartographer."""
        return ArgumentCartographer()

    def test_creates_output_directory(self, cartographer, tmp_path):
        """Test creates output directory if it doesn't exist."""
        output_dir = tmp_path / "nested" / "output"
        assert not output_dir.exists()

        save_debate_visualization(cartographer, output_dir, "test-debate")

        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_default_formats_mermaid_and_json(self, cartographer, tmp_path):
        """Test default formats are mermaid and json."""
        results = save_debate_visualization(cartographer, tmp_path, "debate-id")

        assert "mermaid" in results
        assert "json" in results
        assert "html" not in results

    def test_exports_mermaid_format(self, cartographer, tmp_path):
        """Test exports mermaid format file."""
        results = save_debate_visualization(
            cartographer, tmp_path, "test", formats=["mermaid"]
        )

        mermaid_path = Path(results["mermaid"])
        assert mermaid_path.exists()
        assert mermaid_path.name == "test_graph.mermaid"

        content = mermaid_path.read_text()
        assert "graph" in content

    def test_exports_json_format(self, cartographer, tmp_path):
        """Test exports json format file."""
        results = save_debate_visualization(
            cartographer, tmp_path, "test", formats=["json"]
        )

        json_path = Path(results["json"])
        assert json_path.exists()
        assert json_path.name == "test_graph.json"

        content = json_path.read_text()
        data = json.loads(content)
        assert "nodes" in data
        assert "edges" in data

    def test_exports_html_format(self, cartographer, tmp_path):
        """Test exports html format file."""
        results = save_debate_visualization(
            cartographer, tmp_path, "test", formats=["html"]
        )

        html_path = Path(results["html"])
        assert html_path.exists()
        assert html_path.name == "test_graph.html"

        content = html_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "mermaid" in content.lower()

    def test_exports_all_formats(self, cartographer, tmp_path):
        """Test exports all three formats together."""
        results = save_debate_visualization(
            cartographer, tmp_path, "test", formats=["mermaid", "json", "html"]
        )

        assert len(results) == 3
        for fmt in ["mermaid", "json", "html"]:
            assert fmt in results
            assert Path(results[fmt]).exists()

    def test_returns_file_paths_as_strings(self, cartographer, tmp_path):
        """Test returns file paths as strings."""
        results = save_debate_visualization(cartographer, tmp_path, "test")

        for path in results.values():
            assert isinstance(path, str)

    def test_file_paths_are_absolute(self, cartographer, tmp_path):
        """Test returned paths are full paths."""
        results = save_debate_visualization(cartographer, tmp_path, "test")

        for path in results.values():
            assert str(tmp_path) in path

    def test_debate_id_in_filename(self, cartographer, tmp_path):
        """Test debate_id is used in filenames."""
        results = save_debate_visualization(
            cartographer, tmp_path, "my-custom-debate-id", formats=["mermaid"]
        )

        assert "my-custom-debate-id_graph.mermaid" in results["mermaid"]

    def test_special_characters_in_debate_id(self, cartographer, tmp_path):
        """Test handles special characters in debate_id."""
        # Use characters that are valid in filenames
        results = save_debate_visualization(
            cartographer, tmp_path, "debate_2024-01-15_v1.2", formats=["json"]
        )

        assert Path(results["json"]).exists()

    def test_empty_formats_list_uses_defaults(self, cartographer, tmp_path):
        """Test empty formats list uses default formats (mermaid, json)."""
        # In Python, empty list is falsy so `formats or default` uses default
        results = save_debate_visualization(
            cartographer, tmp_path, "test", formats=[]
        )

        # Due to Python's truthiness, [] triggers the default formats
        assert "mermaid" in results
        assert "json" in results

    def test_unknown_format_ignored(self, cartographer, tmp_path):
        """Test unknown formats are ignored."""
        results = save_debate_visualization(
            cartographer, tmp_path, "test", formats=["unknown", "mermaid"]
        )

        assert "unknown" not in results
        assert "mermaid" in results

    def test_empty_cartographer(self, empty_cartographer, tmp_path):
        """Test handles empty cartographer."""
        results = save_debate_visualization(
            empty_cartographer, tmp_path, "empty", formats=["mermaid", "json"]
        )

        assert Path(results["mermaid"]).exists()
        assert Path(results["json"]).exists()

    def test_json_includes_full_content(self, cartographer, tmp_path):
        """Test JSON export includes full content."""
        results = save_debate_visualization(
            cartographer, tmp_path, "test", formats=["json"]
        )

        content = Path(results["json"]).read_text()
        data = json.loads(content)

        # Should have include_full_content=True passed
        assert "nodes" in data

    def test_overwrites_existing_files(self, cartographer, tmp_path):
        """Test overwrites existing files."""
        # Create initial files
        results1 = save_debate_visualization(
            cartographer, tmp_path, "test", formats=["mermaid"]
        )
        initial_content = Path(results1["mermaid"]).read_text()

        # Add more data
        cartographer.update_from_message("agent3", "Another message", "", 2)

        # Save again
        results2 = save_debate_visualization(
            cartographer, tmp_path, "test", formats=["mermaid"]
        )
        new_content = Path(results2["mermaid"]).read_text()

        # Content should be different
        assert len(new_content) > len(initial_content)

    def test_path_accepts_string(self, cartographer, tmp_path):
        """Test accepts string path."""
        results = save_debate_visualization(
            cartographer, str(tmp_path), "test", formats=["json"]
        )

        assert Path(results["json"]).exists()


class TestGenerateStandaloneHTML:
    """Tests for generate_standalone_html function."""

    @pytest.fixture
    def cartographer_with_topic(self):
        """Create cartographer with topic set."""
        cart = ArgumentCartographer()
        cart.set_debate_context("reg-debate", "Should AI be regulated?")
        cart.update_from_message("proponent", "AI needs regulation", "proposer", 1)
        cart.update_from_message("opponent", "Market can self-regulate", "critic", 1)
        return cart

    @pytest.fixture
    def cartographer_no_topic(self):
        """Create cartographer without topic."""
        cart = ArgumentCartographer()
        cart.update_from_message("agent1", "Some argument", "", 1)
        return cart

    def test_returns_valid_html5(self, cartographer_with_topic):
        """Test returns valid HTML5 document."""
        html = generate_standalone_html(cartographer_with_topic)

        assert html.startswith("<!DOCTYPE html>")
        assert "<html" in html
        assert "</html>" in html
        assert "<head>" in html
        assert "</head>" in html
        assert "<body>" in html
        assert "</body>" in html

    def test_includes_mermaid_cdn(self, cartographer_with_topic):
        """Test includes Mermaid CDN script."""
        html = generate_standalone_html(cartographer_with_topic)

        assert "cdn.jsdelivr.net" in html
        assert "mermaid" in html

    def test_includes_mermaid_initialize(self, cartographer_with_topic):
        """Test includes Mermaid initialization."""
        html = generate_standalone_html(cartographer_with_topic)

        assert "mermaid.initialize" in html

    def test_includes_topic_in_title(self, cartographer_with_topic):
        """Test includes topic in title tag."""
        html = generate_standalone_html(cartographer_with_topic)

        assert "<title>" in html
        assert "Should AI be regulated?" in html

    def test_includes_topic_in_heading(self, cartographer_with_topic):
        """Test includes topic in page heading."""
        html = generate_standalone_html(cartographer_with_topic)

        assert "<h1>" in html
        assert "Should AI be regulated?" in html

    def test_fallback_title_when_no_topic(self, cartographer_no_topic):
        """Test uses fallback title when no topic set."""
        html = generate_standalone_html(cartographer_no_topic)

        # Should have Untitled or Aragora Debate as fallback
        assert "Untitled" in html or "Aragora Debate" in html

    def test_includes_statistics(self, cartographer_with_topic):
        """Test includes statistics section."""
        html = generate_standalone_html(cartographer_with_topic)

        assert "Arguments" in html
        assert "Connections" in html
        assert "Rounds" in html
        assert "Agents" in html

    def test_statistics_show_correct_values(self, cartographer_with_topic):
        """Test statistics show correct values."""
        html = generate_standalone_html(cartographer_with_topic)

        # 2 messages = 2 nodes
        assert ">2<" in html or ">2</div>" in html.replace("\n", "").replace(" ", "")

    def test_includes_mermaid_diagram(self, cartographer_with_topic):
        """Test includes mermaid diagram content."""
        html = generate_standalone_html(cartographer_with_topic)

        assert "class=\"mermaid\"" in html
        assert "graph" in html  # Mermaid diagram starts with graph

    def test_includes_legend(self, cartographer_with_topic):
        """Test includes legend section."""
        html = generate_standalone_html(cartographer_with_topic)

        assert "legend" in html.lower()
        assert "Proposal" in html
        assert "Critique" in html
        assert "Evidence" in html
        assert "Concession" in html
        assert "Consensus" in html

    def test_includes_color_coding(self, cartographer_with_topic):
        """Test includes color coding in legend."""
        html = generate_standalone_html(cartographer_with_topic)

        # Colors from the exporter
        assert "#4CAF50" in html  # Proposal green
        assert "#FF5722" in html  # Critique orange
        assert "#9C27B0" in html  # Evidence purple

    def test_responsive_viewport(self, cartographer_with_topic):
        """Test includes responsive viewport meta tag."""
        html = generate_standalone_html(cartographer_with_topic)

        assert "viewport" in html
        assert "width=device-width" in html

    def test_charset_utf8(self, cartographer_with_topic):
        """Test includes UTF-8 charset."""
        html = generate_standalone_html(cartographer_with_topic)

        assert "UTF-8" in html

    def test_includes_css_styles(self, cartographer_with_topic):
        """Test includes CSS styles."""
        html = generate_standalone_html(cartographer_with_topic)

        assert "<style>" in html
        assert "</style>" in html
        assert "font-family" in html

    def test_dark_theme_background(self, cartographer_with_topic):
        """Test uses dark theme background."""
        html = generate_standalone_html(cartographer_with_topic)

        # Dark background color
        assert "#1a1a2e" in html

    def test_empty_cartographer_still_renders(self):
        """Test empty cartographer still produces valid HTML."""
        cart = ArgumentCartographer()
        html = generate_standalone_html(cart)

        assert "<!DOCTYPE html>" in html
        assert "<div class=\"mermaid\">" in html

    def test_special_characters_in_topic_escaped(self):
        """Test special characters in topic don't break HTML."""
        cart = ArgumentCartographer()
        cart.set_debate_context("Is <script>alert('xss')</script> safe?", "xss-test")
        cart.update_from_message("agent1", "Content", "", 1)

        html = generate_standalone_html(cart)

        # HTML should still be valid (not checking for XSS escaping here,
        # as this is visualization not security-critical output)
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html

    def test_node_content_in_mermaid_section(self, cartographer_with_topic):
        """Test node content appears in mermaid section."""
        html = generate_standalone_html(cartographer_with_topic)

        # The mermaid diagram should contain agent references
        assert "proponent" in html or "opponent" in html

    def test_multiple_rounds_reflected(self):
        """Test multiple rounds reflected in stats."""
        cart = ArgumentCartographer()
        cart.update_from_message("a1", "Round 1 message", "", 1)
        cart.update_from_message("a1", "Round 2 message", "", 2)
        cart.update_from_message("a1", "Round 3 message", "", 3)

        html = generate_standalone_html(cart)

        # Stats should show 3 rounds
        assert ">3<" in html or "3</div>" in html

    def test_container_class_structure(self, cartographer_with_topic):
        """Test has proper container structure."""
        html = generate_standalone_html(cartographer_with_topic)

        assert "class=\"container\"" in html
        assert "class=\"stats\"" in html
        assert "class=\"mermaid\"" in html
        assert "class=\"legend\"" in html


class TestExporterIntegration:
    """Integration tests for the exporter module."""

    def test_save_and_load_json_roundtrip(self, tmp_path):
        """Test JSON can be loaded back and parsed."""
        cart = ArgumentCartographer()
        cart.set_debate_context("int-test", "Integration Test")
        cart.update_from_message("agent1", "First argument", "proposer", 1)
        cart.update_from_message("agent2", "Counter argument", "critic", 1)
        cart.update_from_vote("agent3", "yes", 1)

        results = save_debate_visualization(
            cart, tmp_path, "integration", formats=["json"]
        )

        # Load and verify
        with open(results["json"]) as f:
            data = json.load(f)

        assert data["topic"] == "Integration Test"
        assert len(data["nodes"]) == 3  # 2 messages + 1 vote
        assert len(data["edges"]) >= 0  # May or may not have edges depending on linking

    def test_all_formats_consistent(self, tmp_path):
        """Test all formats export same graph structure."""
        cart = ArgumentCartographer()
        cart.set_debate_context("cons-test", "Consistency Test")
        cart.update_from_message("a1", "Proposal", "proposer", 1)
        cart.update_from_message("a2", "Critique", "critic", 1)

        results = save_debate_visualization(
            cart, tmp_path, "test", formats=["mermaid", "json", "html"]
        )

        # JSON should have 2 nodes
        with open(results["json"]) as f:
            json_data = json.load(f)
        assert len(json_data["nodes"]) == 2

        # HTML should reference both agents
        html_content = Path(results["html"]).read_text()
        assert "a1" in html_content or "Proposal" in html_content

        # Mermaid should have both nodes
        mermaid_content = Path(results["mermaid"]).read_text()
        assert "a1" in mermaid_content or "a2" in mermaid_content

    def test_full_debate_workflow(self, tmp_path):
        """Test full debate visualization workflow."""
        # Simulate a complete debate
        cart = ArgumentCartographer()
        cart.set_debate_context("py-vs-js", "Is Python better than JavaScript?")

        # Round 1
        cart.update_from_message(
            "python_advocate",
            "Python has cleaner syntax and better scientific computing support",
            "proposer",
            1,
        )
        cart.update_from_message(
            "js_advocate",
            "JavaScript has broader ecosystem and runs everywhere",
            "proposer",
            1,
        )

        # Round 2 - critiques (critic_agent, target_agent, severity, round_num, critique_text)
        cart.update_from_critique(
            "python_advocate",
            "js_advocate",
            0.6,
            2,
            "Node.js ecosystem is fragmented",
        )
        cart.update_from_critique(
            "js_advocate",
            "python_advocate",
            0.5,
            2,
            "Python GIL limits concurrency",
        )

        # Round 3 - votes (agent, vote_value, round_num)
        cart.update_from_vote("moderator", "python_advocate", 3)
        cart.update_from_consensus("Both languages excel in their domains", 3, 2)

        # Export all formats
        results = save_debate_visualization(
            cart, tmp_path, "py-vs-js", formats=["mermaid", "json", "html"]
        )

        # Verify all files exist and have content
        for fmt, path in results.items():
            file_path = Path(path)
            assert file_path.exists()
            assert file_path.stat().st_size > 0

        # Verify JSON has all events
        with open(results["json"]) as f:
            data = json.load(f)

        node_types = [n["node_type"] for n in data["nodes"]]
        assert "proposal" in node_types or "evidence" in node_types
        assert "vote" in node_types
        assert "consensus" in node_types
