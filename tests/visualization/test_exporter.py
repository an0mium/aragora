"""
Tests for visualization exporter.

Tests cover:
- Export cache functionality
- save_debate_visualization function
- generate_standalone_html function
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.visualization.exporter import (
    _cache_export,
    _get_cached_export,
    _get_graph_hash,
    clear_export_cache,
    generate_standalone_html,
    save_debate_visualization,
)
from aragora.visualization.mapper import ArgumentCartographer


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def cartographer():
    """Create a cartographer with sample data."""
    cart = ArgumentCartographer()
    cart.set_debate_context("debate-123", "Test Topic")
    cart.update_from_message("claude", "Test proposal content", "proposal", 1)
    cart.update_from_message("gpt", "Test critique content", "critique", 1)
    return cart


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear export cache before each test."""
    clear_export_cache()
    yield
    clear_export_cache()


# ============================================================================
# Cache Tests
# ============================================================================


class TestExportCache:
    """Tests for export caching functionality."""

    def test_get_graph_hash(self, cartographer):
        """Test graph hash generation."""
        hash1 = _get_graph_hash(cartographer)

        assert isinstance(hash1, str)
        assert len(hash1) == 16  # Truncated SHA256

    def test_graph_hash_changes_with_content(self, cartographer):
        """Test graph hash changes when content changes."""
        hash1 = _get_graph_hash(cartographer)

        # Add more content
        cartographer.update_from_message("gemini", "New content", "rebuttal", 2)
        hash2 = _get_graph_hash(cartographer)

        assert hash1 != hash2

    def test_cache_export_and_retrieve(self):
        """Test caching and retrieving exports."""
        _cache_export("debate-1", "json", "hash123", '{"test": true}')

        cached = _get_cached_export("debate-1", "json", "hash123")
        assert cached == '{"test": true}'

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cached = _get_cached_export("nonexistent", "json", "hash")
        assert cached is None

    def test_cache_different_formats(self):
        """Test different formats are cached separately."""
        _cache_export("d1", "json", "hash", "json content")
        _cache_export("d1", "mermaid", "hash", "mermaid content")

        assert _get_cached_export("d1", "json", "hash") == "json content"
        assert _get_cached_export("d1", "mermaid", "hash") == "mermaid content"

    def test_cache_different_hashes(self):
        """Test different hashes are cached separately."""
        _cache_export("d1", "json", "hash1", "content v1")
        _cache_export("d1", "json", "hash2", "content v2")

        assert _get_cached_export("d1", "json", "hash1") == "content v1"
        assert _get_cached_export("d1", "json", "hash2") == "content v2"

    def test_clear_export_cache(self):
        """Test clearing the cache."""
        _cache_export("d1", "json", "hash", "content")
        _cache_export("d2", "mermaid", "hash", "content")

        count = clear_export_cache()

        assert count == 2
        assert _get_cached_export("d1", "json", "hash") is None

    def test_cache_ttl_expiry(self):
        """Test cache entries expire after TTL."""
        # This test uses the internal TTL check
        _cache_export("d1", "json", "hash", "content")

        # Mock time to simulate TTL expiry
        with patch("aragora.visualization.exporter.time") as mock_time:
            mock_time.time.return_value = time.time() + 400  # 400s > 300s TTL

            cached = _get_cached_export("d1", "json", "hash")
            assert cached is None


# ============================================================================
# save_debate_visualization Tests
# ============================================================================


class TestSaveDebateVisualization:
    """Tests for save_debate_visualization function."""

    def test_saves_mermaid_format(self, cartographer, temp_dir):
        """Test saving Mermaid format."""
        results = save_debate_visualization(
            cartographer, temp_dir, "test-debate", formats=["mermaid"]
        )

        assert "mermaid" in results
        mermaid_path = Path(results["mermaid"])
        assert mermaid_path.exists()
        assert mermaid_path.suffix == ".mermaid"

    def test_saves_json_format(self, cartographer, temp_dir):
        """Test saving JSON format."""
        results = save_debate_visualization(cartographer, temp_dir, "test-debate", formats=["json"])

        assert "json" in results
        json_path = Path(results["json"])
        assert json_path.exists()
        assert json_path.suffix == ".json"

        # Verify valid JSON
        import json

        content = json_path.read_text()
        data = json.loads(content)
        assert "nodes" in data

    def test_saves_html_format(self, cartographer, temp_dir):
        """Test saving HTML format."""
        results = save_debate_visualization(cartographer, temp_dir, "test-debate", formats=["html"])

        assert "html" in results
        html_path = Path(results["html"])
        assert html_path.exists()
        content = html_path.read_text()
        assert "<html" in content

    def test_saves_multiple_formats(self, cartographer, temp_dir):
        """Test saving multiple formats."""
        results = save_debate_visualization(
            cartographer, temp_dir, "test-debate", formats=["mermaid", "json", "html"]
        )

        assert len(results) == 3
        assert all(Path(p).exists() for p in results.values())

    def test_creates_output_directory(self, cartographer, temp_dir):
        """Test creates output directory if needed."""
        nested_dir = temp_dir / "nested" / "output"

        results = save_debate_visualization(cartographer, nested_dir, "test", formats=["json"])

        assert nested_dir.exists()
        assert Path(results["json"]).exists()

    def test_default_formats(self, cartographer, temp_dir):
        """Test default formats are mermaid and json."""
        results = save_debate_visualization(cartographer, temp_dir, "test")

        assert "mermaid" in results
        assert "json" in results
        assert "html" not in results

    def test_uses_cache(self, cartographer, temp_dir):
        """Test caching is used for performance."""
        # First call - populates cache
        save_debate_visualization(cartographer, temp_dir, "test", formats=["json"], use_cache=True)

        # Second call - should use cache
        with patch("aragora.visualization.exporter._get_cached_export") as mock_cache:
            mock_cache.return_value = '{"cached": true}'
            save_debate_visualization(
                cartographer, temp_dir, "test", formats=["json"], use_cache=True
            )

            mock_cache.assert_called()

    def test_skips_cache_when_disabled(self, cartographer, temp_dir):
        """Test cache is skipped when disabled."""
        with patch("aragora.visualization.exporter._get_cached_export") as mock_cache:
            save_debate_visualization(
                cartographer, temp_dir, "test", formats=["json"], use_cache=False
            )

            mock_cache.assert_not_called()


# ============================================================================
# generate_standalone_html Tests
# ============================================================================


class TestGenerateStandaloneHtml:
    """Tests for generate_standalone_html function."""

    def test_generates_valid_html(self, cartographer):
        """Test generates valid HTML structure."""
        html = generate_standalone_html(cartographer)

        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<head>" in html
        assert "<body>" in html

    def test_includes_topic(self, cartographer):
        """Test includes debate topic in title."""
        html = generate_standalone_html(cartographer)

        assert "Test Topic" in html

    def test_includes_mermaid_code(self, cartographer):
        """Test includes Mermaid diagram code."""
        html = generate_standalone_html(cartographer)

        assert "mermaid" in html.lower()
        assert '<div class="mermaid">' in html

    def test_includes_statistics(self, cartographer):
        """Test includes statistics."""
        html = generate_standalone_html(cartographer)

        # Should include stat cards
        assert "Arguments" in html
        assert "Connections" in html
        assert "Rounds" in html
        assert "Agents" in html

    def test_includes_legend(self, cartographer):
        """Test includes legend."""
        html = generate_standalone_html(cartographer)

        assert "legend" in html.lower()
        assert "Proposal" in html
        assert "Critique" in html

    def test_includes_mermaid_script(self, cartographer):
        """Test includes Mermaid.js script."""
        html = generate_standalone_html(cartographer)

        assert "mermaid.min.js" in html or "mermaid/dist" in html

    def test_html_includes_topic(self):
        """Test topic is included in HTML output."""
        cart = ArgumentCartographer()
        cart.set_debate_context("d1", "Test Debate Topic")
        cart.update_from_message("a", "test", "proposal", 1)

        html = generate_standalone_html(cart)

        assert "Test Debate Topic" in html


# ============================================================================
# Integration Tests
# ============================================================================


class TestExporterIntegration:
    """Integration tests for exporter functionality."""

    def test_full_export_workflow(self, temp_dir):
        """Test complete export workflow."""
        # Create cartographer with realistic data
        cart = ArgumentCartographer()
        cart.set_debate_context("debate-integration", "Should we use TypeScript?")

        cart.update_from_message(
            "claude",
            "TypeScript provides type safety and catches errors at compile time.",
            "proposal",
            1,
        )
        cart.update_from_message(
            "gpt", "JavaScript is more flexible and has wider adoption.", "proposal", 1
        )
        cart.update_from_message(
            "claude", "The flexibility comes at the cost of runtime errors.", "critique", 1
        )
        cart.update_from_critique("claude", "gpt", 0.6, 1)
        cart.update_from_vote("claude", "typescript", 2)
        cart.update_from_vote("gpt", "partial_agree", 2)

        # Export all formats
        results = save_debate_visualization(
            cart, temp_dir, "integration-test", formats=["mermaid", "json", "html"]
        )

        # Verify all files exist and are valid
        assert len(results) == 3

        # Check JSON is valid
        import json

        json_content = Path(results["json"]).read_text()
        data = json.loads(json_content)
        assert len(data["nodes"]) >= 4

        # Check HTML has expected content
        html_content = Path(results["html"]).read_text()
        assert "TypeScript" in html_content
        assert "mermaid" in html_content.lower()

        # Check Mermaid is valid syntax
        mermaid_content = Path(results["mermaid"]).read_text()
        assert "graph" in mermaid_content.lower() or "flowchart" in mermaid_content.lower()
