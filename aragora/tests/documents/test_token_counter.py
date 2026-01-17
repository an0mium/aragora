"""Tests for token counting service."""

import pytest
from pathlib import Path


class TestTokenCounter:
    """Tests for the TokenCounter class."""

    @pytest.fixture
    def token_counter(self):
        """Create a TokenCounter instance."""
        from aragora.documents.chunking.token_counter import TokenCounter

        return TokenCounter()

    def test_count_tokens_simple(self, token_counter):
        """Test counting tokens in simple text."""
        text = "Hello, world! This is a test."
        count = token_counter.count(text)
        assert count > 0
        assert count < 20  # Simple text should be under 20 tokens

    def test_count_tokens_empty(self, token_counter):
        """Test counting tokens in empty string."""
        count = token_counter.count("")
        assert count == 0

    def test_count_tokens_code(self, token_counter):
        """Test counting tokens in code."""
        code = '''
def hello_world():
    """Print hello world."""
    print("Hello, world!")
    return True
'''
        count = token_counter.count(code)
        assert count > 10
        assert count < 100

    def test_count_tokens_different_models(self, token_counter):
        """Test token counting with different model names."""
        text = "This is a test sentence for token counting."

        # Should work with different model formats
        count_gpt4 = token_counter.count(text, model="gpt-4")
        count_claude = token_counter.count(text, model="claude-3")
        count_gemini = token_counter.count(text, model="gemini-1.5-flash")

        # All should return reasonable counts
        assert count_gpt4 > 0
        assert count_claude > 0
        assert count_gemini > 0

    def test_count_tokens_long_text(self, token_counter):
        """Test counting tokens in longer text."""
        sample_text = """
        # Sample Document

        This is a sample document for testing the document auditing system.

        ## Section 1: Introduction

        The effective date is 01/15/2024. This document expires on 12/31/2025.
        """
        count = token_counter.count(sample_text)
        assert count > 20
        assert count < 500

    def test_fits_in_context(self, token_counter):
        """Test checking if text fits in context window."""
        short_text = "Hello world"
        long_text = "This is a much longer sentence with many words. " * 5000  # ~250K chars

        # fits_context reserves 1000 tokens by default, so use max_tokens=2000 for short text
        assert token_counter.fits_context(short_text, model="gpt-4", max_tokens=2000)
        # Long text should not fit with small max_tokens
        assert not token_counter.fits_context(long_text, model="gpt-4", max_tokens=1500)

    def test_truncate_to_tokens(self, token_counter):
        """Test truncating text to token limit."""
        text = "This is a sentence. " * 100
        truncated = token_counter.truncate_to_tokens(text, max_tokens=50)

        truncated_count = token_counter.count(truncated)
        assert truncated_count <= 60  # Allow some buffer for approximation

    def test_count_real_python_file(self, token_counter):
        """Test counting tokens from a real Python file."""
        # Use conftest.py as a sample file
        sample_file = Path(__file__).parent.parent / "conftest.py"
        if not sample_file.exists():
            pytest.skip("Sample file not found")

        content = sample_file.read_text()
        count = token_counter.count(content)
        assert count > 0

    def test_batch_count_tokens(self, token_counter):
        """Test counting tokens in multiple texts."""
        texts = ["First text", "Second longer text with more words", "Third text"]
        counts = [token_counter.count(text) for text in texts]

        assert len(counts) == 3
        assert all(c > 0 for c in counts)
        assert counts[1] > counts[0]  # Second text should have more tokens
