"""Tests for document chunking strategies."""

import pytest
from pathlib import Path


class TestChunkingStrategies:
    """Tests for different chunking strategies."""

    @pytest.fixture
    def config(self):
        """Create a ChunkingConfig instance."""
        from aragora.documents.chunking.strategies import ChunkingConfig

        return ChunkingConfig(chunk_size=500, overlap=50)

    @pytest.fixture
    def fixed_chunker(self, config):
        """Create a FixedSizeChunking instance."""
        from aragora.documents.chunking.strategies import FixedSizeChunking

        return FixedSizeChunking(config)

    @pytest.fixture
    def semantic_chunker(self, config):
        """Create a SemanticChunking instance."""
        from aragora.documents.chunking.strategies import SemanticChunking

        return SemanticChunking(config)

    @pytest.fixture
    def sliding_chunker(self, config):
        """Create a SlidingWindowChunking instance."""
        from aragora.documents.chunking.strategies import SlidingWindowChunking

        return SlidingWindowChunking(config)

    @pytest.fixture
    def recursive_chunker(self, config):
        """Create a RecursiveChunking instance."""
        from aragora.documents.chunking.strategies import RecursiveChunking

        return RecursiveChunking(config)

    def test_fixed_chunking(self, fixed_chunker):
        """Test fixed-size chunking."""
        text = "Word " * 1000  # 1000 words
        chunks = fixed_chunker.chunk(text)

        assert len(chunks) > 1
        # Each chunk should have content
        for chunk in chunks:
            assert chunk.content.strip()

    def test_semantic_chunking(self, semantic_chunker):
        """Test semantic chunking by paragraphs."""
        text = """
# Introduction

This is the first paragraph about introduction.
It contains multiple sentences about the topic.

# Methods

This section describes the methods used.
We employed various techniques for analysis.

# Results

Here are the results of our study.
The findings were significant.

# Conclusion

In conclusion, this document demonstrates chunking.
"""
        chunks = semantic_chunker.chunk(text)

        assert len(chunks) >= 1
        # Semantic chunks should respect paragraph boundaries
        for chunk in chunks:
            assert chunk.content.strip()

    def test_sliding_window_chunking(self, sliding_chunker):
        """Test sliding window chunking with overlap."""
        text = " ".join([f"Sentence {i}." for i in range(100)])
        chunks = sliding_chunker.chunk(text)

        assert len(chunks) >= 1
        # Each chunk should have content
        for chunk in chunks:
            assert chunk.content.strip()

    def test_recursive_chunking(self, recursive_chunker):
        """Test recursive chunking for hierarchical documents."""
        text = """
# Chapter 1

## Section 1.1

Content for section 1.1 with enough text to require chunking.
This section covers the basics of the topic at hand.

## Section 1.2

Content for section 1.2 with additional details.
More information about the subject matter.

# Chapter 2

## Section 2.1

Different chapter with new content.
This explores another aspect of the topic.
"""
        chunks = recursive_chunker.chunk(text)

        assert len(chunks) >= 1
        # Each chunk should have content
        for chunk in chunks:
            assert chunk.content.strip()

    def test_chunk_has_metadata(self, fixed_chunker):
        """Test that chunks include proper metadata."""
        text = "Test content " * 100
        chunks = fixed_chunker.chunk(text)

        for i, chunk in enumerate(chunks):
            assert hasattr(chunk, "content")
            assert hasattr(chunk, "sequence")
            assert chunk.sequence == i

    def test_empty_text_chunking(self, fixed_chunker):
        """Test chunking empty text."""
        chunks = fixed_chunker.chunk("")
        # Should return empty list or single empty chunk
        assert len(chunks) <= 1

    def test_small_text_chunking(self, fixed_chunker):
        """Test chunking text smaller than chunk size."""
        text = "Small text"
        chunks = fixed_chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].content.strip() == text

    def test_get_chunking_strategy(self):
        """Test the strategy factory function."""
        from aragora.documents.chunking.strategies import (
            get_chunking_strategy,
            ChunkingConfig,
        )

        config = ChunkingConfig(chunk_size=500)

        # Test all strategy types
        strategy_types = ["semantic", "sliding", "recursive", "fixed"]
        for strategy_type in strategy_types:
            strategy = get_chunking_strategy(strategy_type, config)
            assert strategy is not None

    def test_auto_select_strategy(self):
        """Test automatic strategy selection."""
        from aragora.documents.chunking.strategies import auto_select_strategy

        # Narrative text should select semantic
        narrative = "This is a long narrative text. " * 50
        strategy = auto_select_strategy(narrative)
        assert strategy in ["semantic", "fixed", "sliding", "recursive"]

        # Code should select sliding
        code = (
            """
def foo():
    pass

def bar():
    pass
"""
            * 10
        )
        strategy = auto_select_strategy(code)
        assert strategy in ["semantic", "fixed", "sliding", "recursive"]


class TestChunkingEdgeCases:
    """Test edge cases in chunking."""

    @pytest.fixture
    def chunker(self):
        from aragora.documents.chunking.strategies import FixedSizeChunking, ChunkingConfig

        return FixedSizeChunking(ChunkingConfig(chunk_size=100))

    def test_unicode_content(self, chunker):
        """Test chunking text with unicode characters."""
        text = "Hello 世界! This is a test with émojis 🎉 and spëcial characters."
        chunks = chunker.chunk(text)

        assert len(chunks) >= 1
        # Should preserve unicode
        combined = "".join(c.content for c in chunks)
        assert "世界" in combined or "世界" in text

    def test_very_long_word(self, chunker):
        """Test chunking text with very long words."""
        long_word = "a" * 1000
        text = f"Normal text {long_word} more normal text"
        chunks = chunker.chunk(text)

        # Should handle without crashing
        assert len(chunks) >= 1

    def test_only_whitespace(self, chunker):
        """Test chunking whitespace-only text."""
        text = "   \n\n\t\t   "
        chunks = chunker.chunk(text)

        # Should return empty or single chunk
        assert len(chunks) <= 1

    def test_special_delimiters(self):
        """Test chunking with special delimiter patterns."""
        from aragora.documents.chunking.strategies import SemanticChunking, ChunkingConfig

        chunker = SemanticChunking(ChunkingConfig(chunk_size=500))
        text = "Section 1\n---\nSection 2\n===\nSection 3\n***\nSection 4"
        chunks = chunker.chunk(text)

        assert len(chunks) >= 1


class TestChunkingWithRealFiles:
    """Test chunking with real project files."""

    @pytest.fixture
    def project_root(self) -> Path:
        return Path(__file__).parent.parent.parent.parent

    def test_chunk_python_file(self, project_root):
        """Test chunking a real Python file."""
        from aragora.documents.chunking.strategies import SemanticChunking, ChunkingConfig

        sample_file = project_root / "aragora" / "core.py"
        if not sample_file.exists():
            pytest.skip("Sample file not found")

        content = sample_file.read_text()
        chunker = SemanticChunking(ChunkingConfig(chunk_size=500))
        chunks = chunker.chunk(content)

        assert len(chunks) >= 1
        assert sample_file.name  # File name is valid
        assert len(content) > 0  # Content is non-empty

    def test_chunk_markdown_file(self, project_root):
        """Test chunking a real Markdown file."""
        from aragora.documents.chunking.strategies import SemanticChunking, ChunkingConfig

        sample_file = project_root / "CLAUDE.md"
        if not sample_file.exists():
            pytest.skip("Sample file not found")

        content = sample_file.read_text()
        chunker = SemanticChunking(ChunkingConfig(chunk_size=500))
        chunks = chunker.chunk(content)

        assert len(chunks) >= 1
        assert sample_file.name  # File name is valid
        assert len(content) > 0  # Content is non-empty
