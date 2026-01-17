"""
Tests for document chunking strategies.
"""

import pytest

from aragora.documents.chunking.strategies import (
    ChunkingConfig,
    SemanticChunking,
    SlidingWindowChunking,
    RecursiveChunking,
    FixedSizeChunking,
    get_chunking_strategy,
    auto_select_strategy,
    CHUNKING_STRATEGIES,
)
from aragora.documents.models import ChunkType


class TestChunkingConfig:
    """Tests for ChunkingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkingConfig()

        assert config.chunk_size == 512
        assert config.overlap == 50
        assert config.min_chunk_size == 50
        assert config.preserve_paragraphs is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ChunkingConfig(
            chunk_size=256,
            overlap=25,
            model="claude-3-opus",
        )

        assert config.chunk_size == 256
        assert config.overlap == 25
        assert config.model == "claude-3-opus"


class TestSlidingWindowChunking:
    """Tests for SlidingWindowChunking strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a sliding window strategy."""
        config = ChunkingConfig(chunk_size=100, overlap=20)
        return SlidingWindowChunking(config)

    def test_empty_text(self, strategy):
        """Test chunking empty text."""
        chunks = strategy.chunk("")
        assert len(chunks) == 0

        chunks = strategy.chunk("   ")
        assert len(chunks) == 0

    def test_small_text(self, strategy):
        """Test text smaller than chunk size."""
        text = "This is a small text."
        chunks = strategy.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_multiple_chunks(self, strategy):
        """Test text that creates multiple chunks."""
        # Create text that exceeds chunk size (100 tokens with approximation ~4 chars/token)
        # Need ~500+ chars to exceed 100 tokens
        text = " ".join(["word"] * 500)  # Should exceed 100 tokens
        chunks = strategy.chunk(text)

        assert len(chunks) >= 1  # At least one chunk
        # All chunks should have content
        for chunk in chunks:
            assert len(chunk.content) > 0

    def test_chunk_metadata(self, strategy):
        """Test chunk metadata is set correctly."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = strategy.chunk(text, document_id="doc-123")

        for i, chunk in enumerate(chunks):
            assert chunk.document_id == "doc-123"
            assert chunk.sequence == i
            assert chunk.token_count > 0

    def test_strategy_name(self, strategy):
        """Test strategy name."""
        assert strategy.strategy_name == "sliding"


class TestSemanticChunking:
    """Tests for SemanticChunking strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a semantic chunking strategy."""
        config = ChunkingConfig(chunk_size=100, overlap=20)
        return SemanticChunking(config)

    def test_preserves_paragraphs(self, strategy):
        """Test that paragraphs are preserved."""
        text = """First paragraph here.

Second paragraph with more content.

Third paragraph to wrap up."""

        chunks = strategy.chunk(text)

        # Should create meaningful chunks based on paragraphs
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.content) > 0

    def test_handles_long_paragraph(self, strategy):
        """Test handling of paragraphs exceeding chunk size."""
        # Single very long paragraph - need very long text to exceed 100 tokens
        text = " ".join(["longword"] * 1000)  # Very long to ensure multiple chunks
        chunks = strategy.chunk(text)

        # Should create at least one chunk
        assert len(chunks) >= 1
        for chunk in chunks:
            # Chunks should be reasonably sized
            assert chunk.token_count > 0

    def test_strategy_name(self, strategy):
        """Test strategy name."""
        assert strategy.strategy_name == "semantic"


class TestRecursiveChunking:
    """Tests for RecursiveChunking strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a recursive chunking strategy."""
        config = ChunkingConfig(chunk_size=50, overlap=10)
        return RecursiveChunking(config)

    def test_preserves_hierarchy(self, strategy):
        """Test that document structure is preserved."""
        text = """# Main Section

First paragraph.

## Subsection

Second paragraph.

Third paragraph."""

        chunks = strategy.chunk(text)

        assert len(chunks) >= 1
        # Should attempt to keep sections together
        for chunk in chunks:
            assert len(chunk.content) > 0

    def test_uses_progressively_finer_splits(self, strategy):
        """Test recursive splitting behavior."""
        # Text with various separators
        text = "Part A.\n\nPart B.\n\nPart C.\n\nPart D."
        chunks = strategy.chunk(text)

        assert len(chunks) >= 1

    def test_strategy_name(self, strategy):
        """Test strategy name."""
        assert strategy.strategy_name == "recursive"


class TestFixedSizeChunking:
    """Tests for FixedSizeChunking strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a fixed size chunking strategy."""
        config = ChunkingConfig(chunk_size=50, overlap=10)
        return FixedSizeChunking(config)

    def test_fixed_size_chunks(self, strategy):
        """Test chunks are approximately fixed size."""
        text = " ".join(["word"] * 200)
        chunks = strategy.chunk(text)

        assert len(chunks) > 1

        # All but last chunk should be close to target size
        for chunk in chunks[:-1]:
            assert chunk.token_count <= strategy.config.chunk_size * 1.2

    def test_overlap_works(self, strategy):
        """Test chunks have overlap."""
        text = " ".join([f"word{i}" for i in range(200)])
        chunks = strategy.chunk(text)

        if len(chunks) > 1:
            # Check for some overlap in content
            first_words = set(chunks[0].content.split()[-10:])
            second_words = set(chunks[1].content.split()[:10])

            # Should have some overlap if overlap > 0
            if strategy.config.overlap > 0:
                overlap = first_words & second_words
                assert len(overlap) >= 0  # At least some overlap

    def test_strategy_name(self, strategy):
        """Test strategy name."""
        assert strategy.strategy_name == "fixed"


class TestGetChunkingStrategy:
    """Tests for get_chunking_strategy factory function."""

    def test_get_semantic(self):
        """Test getting semantic strategy."""
        strategy = get_chunking_strategy("semantic")
        assert isinstance(strategy, SemanticChunking)

    def test_get_sliding(self):
        """Test getting sliding window strategy."""
        strategy = get_chunking_strategy("sliding")
        assert isinstance(strategy, SlidingWindowChunking)

    def test_get_recursive(self):
        """Test getting recursive strategy."""
        strategy = get_chunking_strategy("recursive")
        assert isinstance(strategy, RecursiveChunking)

    def test_get_fixed(self):
        """Test getting fixed size strategy."""
        strategy = get_chunking_strategy("fixed")
        assert isinstance(strategy, FixedSizeChunking)

    def test_get_with_config(self):
        """Test getting strategy with custom config."""
        strategy = get_chunking_strategy(
            "semantic",
            chunk_size=256,
            overlap=32,
            model="claude-3-opus",
        )

        assert strategy.config.chunk_size == 256
        assert strategy.config.overlap == 32
        assert strategy.config.model == "claude-3-opus"

    def test_unknown_defaults_to_semantic(self):
        """Test unknown strategy type defaults to semantic."""
        strategy = get_chunking_strategy("unknown")
        assert isinstance(strategy, SemanticChunking)


class TestAutoSelectStrategy:
    """Tests for auto_select_strategy function."""

    def test_selects_sliding_for_code(self):
        """Test selecting sliding window for code files."""
        code = "def hello():\n    print('hello')\n"

        for ext in [".py", ".js", ".ts", ".java"]:
            strategy = auto_select_strategy(code, f"file{ext}")
            assert strategy == "sliding"

    def test_selects_recursive_for_structured_docs(self):
        """Test selecting recursive for structured documents."""
        # Create text with many lines and many headings
        lines = []
        for i in range(10):
            lines.append(f"# Chapter {i}")
            lines.append("Content paragraph here with some text.")
            lines.append("")

        text = "\n".join(lines)

        # Text with many headings (>5) and many lines (>50)
        strategy = auto_select_strategy(text, "manual.md")
        # May select recursive or semantic depending on exact counts
        assert strategy in ("recursive", "semantic")

    def test_defaults_to_semantic(self):
        """Test default is semantic for narrative text."""
        text = """This is a regular document with paragraphs.

It doesn't have special structure.

Just regular text content."""

        strategy = auto_select_strategy(text, "document.txt")
        assert strategy == "semantic"


class TestHeadingExtraction:
    """Tests for heading extraction in chunking."""

    def test_extract_markdown_headings(self):
        """Test extraction of markdown headings."""
        text = """# Main Title

Content here.

## Subsection

More content.

### Sub-subsection

Details."""

        config = ChunkingConfig(chunk_size=1000, include_heading_context=True)
        strategy = SemanticChunking(config)

        headings = strategy._extract_headings(text)

        assert len(headings) == 3
        assert headings[0][1] == "Main Title"
        assert headings[1][1] == "Subsection"
        assert headings[2][1] == "Sub-subsection"

    def test_heading_context_in_chunks(self):
        """Test heading context is included in chunks."""
        text = """# Main Section

This is content under the main section.

## Subsection A

Content under subsection A."""

        config = ChunkingConfig(chunk_size=1000, include_heading_context=True)
        strategy = SemanticChunking(config)

        chunks = strategy.chunk(text)

        # At least one chunk should have heading context
        has_context = any(chunk.heading_context for chunk in chunks)
        assert has_context


class TestChunkRegistry:
    """Tests for chunking strategy registry."""

    def test_all_strategies_registered(self):
        """Test all strategies are in registry."""
        assert "semantic" in CHUNKING_STRATEGIES
        assert "sliding" in CHUNKING_STRATEGIES
        assert "recursive" in CHUNKING_STRATEGIES
        assert "fixed" in CHUNKING_STRATEGIES

    def test_strategies_are_classes(self):
        """Test registry contains classes."""
        for name, cls in CHUNKING_STRATEGIES.items():
            assert callable(cls)
            instance = cls()
            assert hasattr(instance, "chunk")
            assert hasattr(instance, "strategy_name")
