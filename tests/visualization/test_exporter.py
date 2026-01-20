"""Tests for visualization export utilities."""

import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch

from aragora.visualization.exporter import (
    _get_graph_hash,
    _get_cached_export,
    _cache_export,
    clear_export_cache,
    save_debate_visualization,
    generate_standalone_html,
    _EXPORT_CACHE_TTL,
)
from aragora.visualization.mapper import ArgumentCartographer


class TestGetGraphHash:
    """Tests for graph hashing."""

    def test_returns_string(self):
        """Should return a hash string."""
        cart = ArgumentCartographer()
        hash_val = _get_graph_hash(cart)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 16

    def test_same_graph_same_hash(self):
        """Same graph should produce same hash."""
        cart1 = ArgumentCartographer()
        cart1.update_from_message("agent", "test", "agent", 1)
        hash1 = _get_graph_hash(cart1)

        cart2 = ArgumentCartographer()
        cart2.update_from_message("agent", "test", "agent", 1)
        hash2 = _get_graph_hash(cart2)

        assert hash1 == hash2

    def test_different_graphs_different_hash(self):
        """Different graphs should produce different hashes."""
        cart1 = ArgumentCartographer()
        cart1.update_from_message("agent1", "test", "agent", 1)
        hash1 = _get_graph_hash(cart1)

        cart2 = ArgumentCartographer()
        cart2.update_from_message("agent2", "different", "agent", 1)
        hash2 = _get_graph_hash(cart2)

        assert hash1 != hash2


class TestExportCache:
    """Tests for export caching."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_export_cache()

    def test_cache_miss_returns_none(self):
        """Cache miss should return None."""
        result = _get_cached_export("unknown", "json", "hash")
        assert result is None

    def test_cache_hit_returns_content(self):
        """Cache hit should return content."""
        _cache_export("debate-1", "json", "hash123", '{"data": "test"}')
        result = _get_cached_export("debate-1", "json", "hash123")
        assert result == '{"data": "test"}'

    def test_cache_key_includes_all_params(self):
        """Cache key should include debate_id, format, and hash."""
        _cache_export("debate-1", "json", "hash1", "content1")
        _cache_export("debate-1", "mermaid", "hash1", "content2")
        _cache_export("debate-1", "json", "hash2", "content3")

        assert _get_cached_export("debate-1", "json", "hash1") == "content1"
        assert _get_cached_export("debate-1", "mermaid", "hash1") == "content2"
        assert _get_cached_export("debate-1", "json", "hash2") == "content3"

    def test_clear_cache(self):
        """Should clear all cache entries."""
        _cache_export("d1", "json", "h1", "c1")
        _cache_export("d2", "json", "h2", "c2")

        count = clear_export_cache()

        assert count == 2
        assert _get_cached_export("d1", "json", "h1") is None
        assert _get_cached_export("d2", "json", "h2") is None

    def test_expired_cache_returns_none(self):
        """Expired cache entries should return None."""
        _cache_export("debate", "json", "hash", "content")

        # Mock time to be past TTL
        with patch("aragora.visualization.exporter.time.time", return_value=time.time() + _EXPORT_CACHE_TTL + 1):
            result = _get_cached_export("debate", "json", "hash")

        assert result is None


class TestSaveDebateVisualization:
    """Tests for saving debate visualizations."""

    @pytest.fixture
    def cartographer(self):
        """Create a populated cartographer."""
        cart = ArgumentCartographer()
        cart.set_debate_context("test-debate", "Test topic")
        cart.update_from_message("claude", "Test proposal", "proposer", 1)
        cart.update_from_message("gemini", "Test critique", "critic", 1)
        return cart

    def setup_method(self):
        """Clear cache before each test."""
        clear_export_cache()

    def test_creates_output_directory(self, tmp_path, cartographer):
        """Should create output directory if it doesn't exist."""
        output_dir = tmp_path / "new_dir" / "nested"
        
        save_debate_visualization(cartographer, output_dir, "test")
        
        assert output_dir.exists()

    def test_saves_mermaid_by_default(self, tmp_path, cartographer):
        """Should save Mermaid format by default."""
        results = save_debate_visualization(cartographer, tmp_path, "test")
        
        assert "mermaid" in results
        mermaid_path = Path(results["mermaid"])
        assert mermaid_path.exists()
        assert "graph" in mermaid_path.read_text()

    def test_saves_json_by_default(self, tmp_path, cartographer):
        """Should save JSON format by default."""
        results = save_debate_visualization(cartographer, tmp_path, "test")
        
        assert "json" in results
        json_path = Path(results["json"])
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "nodes" in data

    def test_saves_html_when_requested(self, tmp_path, cartographer):
        """Should save HTML when specified."""
        results = save_debate_visualization(
            cartographer, tmp_path, "test", formats=["html"]
        )
        
        assert "html" in results
        html_path = Path(results["html"])
        assert html_path.exists()
        assert "<!DOCTYPE html>" in html_path.read_text()

    def test_saves_all_formats(self, tmp_path, cartographer):
        """Should save all requested formats."""
        results = save_debate_visualization(
            cartographer, tmp_path, "test", formats=["mermaid", "json", "html"]
        )
        
        assert len(results) == 3
        assert all(Path(p).exists() for p in results.values())

    def test_uses_debate_id_in_filename(self, tmp_path, cartographer):
        """Should use debate_id in filenames."""
        results = save_debate_visualization(cartographer, tmp_path, "my-debate-123")
        
        assert "my-debate-123" in results["mermaid"]
        assert "my-debate-123" in results["json"]

    def test_uses_cache_by_default(self, tmp_path, cartographer):
        """Should use cache by default."""
        # First save populates cache
        save_debate_visualization(cartographer, tmp_path, "test")
        
        # Second save should use cache
        results2 = save_debate_visualization(cartographer, tmp_path / "second", "test")
        
        # Both should have same content (from cache)
        assert Path(results2["json"]).exists()

    def test_can_disable_cache(self, tmp_path, cartographer):
        """Should respect use_cache=False."""
        # This should work even with empty cache
        results = save_debate_visualization(
            cartographer, tmp_path, "test", use_cache=False
        )
        
        assert len(results) >= 2


class TestGenerateStandaloneHtml:
    """Tests for standalone HTML generation."""

    @pytest.fixture
    def cartographer(self):
        """Create a populated cartographer."""
        cart = ArgumentCartographer()
        cart.set_debate_context("test-debate", "Test topic")
        cart.update_from_message("claude", "Test proposal", "proposer", 1)
        return cart

    def test_generates_valid_html(self, cartographer):
        """Should generate valid HTML document."""
        html = generate_standalone_html(cartographer)
        
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html
        assert "<head>" in html
        assert "<body>" in html

    def test_includes_topic(self, cartographer):
        """Should include debate topic."""
        html = generate_standalone_html(cartographer)
        assert "Test topic" in html

    def test_includes_mermaid_code(self, cartographer):
        """Should include Mermaid diagram code."""
        html = generate_standalone_html(cartographer)
        
        assert "mermaid" in html.lower()
        assert "graph" in html

    def test_includes_statistics(self, cartographer):
        """Should include statistics display."""
        html = generate_standalone_html(cartographer)
        
        assert "Arguments" in html
        assert "Connections" in html
        assert "Rounds" in html

    def test_includes_legend(self, cartographer):
        """Should include node type legend."""
        html = generate_standalone_html(cartographer)
        
        assert "Proposal" in html
        assert "Critique" in html
        assert "legend" in html.lower()

    def test_includes_mermaid_script(self, cartographer):
        """Should include Mermaid.js script."""
        html = generate_standalone_html(cartographer)
        
        assert "mermaid" in html
        assert "<script" in html


class TestExportCacheEviction:
    """Tests for cache eviction behavior."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_export_cache()

    def test_expired_entries_not_returned(self):
        """Should not return expired entries when accessed."""
        # Add entry
        _cache_export("old", "json", "hash", "old content")

        # Verify entry exists normally
        assert _get_cached_export("old", "json", "hash") == "old content"

        # Compute future time before patching
        future_time = time.time() + _EXPORT_CACHE_TTL + 100

        # Make it expired by mocking time when reading
        with patch("aragora.visualization.exporter.time.time") as mock_time:
            mock_time.return_value = future_time
            # Expired entry should return None
            result = _get_cached_export("old", "json", "hash")
            assert result is None

    def test_handles_many_entries(self):
        """Should handle many cache entries without error."""
        for i in range(50):
            _cache_export(f"debate-{i}", "json", f"hash-{i}", f"content-{i}")
        
        # Should still work
        result = _get_cached_export("debate-25", "json", "hash-25")
        assert result == "content-25"
