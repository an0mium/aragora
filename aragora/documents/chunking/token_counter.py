"""
Token counting service for multi-model document processing.

Provides accurate token counting for different LLM providers using
tiktoken (OpenAI) and model-specific approximations.

Usage:
    from aragora.documents.chunking.token_counter import TokenCounter

    counter = TokenCounter()

    # Count tokens for a specific model
    tokens = counter.count("Hello world", model="gpt-4")

    # Count tokens for Claude (approximation)
    tokens = counter.count("Hello world", model="claude-3-opus")

    # Get appropriate encoder for model
    encoder = counter.get_encoder("gpt-4")
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# Try to import tiktoken
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False
    logger.info("tiktoken not available, using approximation for token counting")


# Model family to encoding mapping for tiktoken
MODEL_ENCODINGS = {
    # OpenAI models
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    # Default
    "default": "cl100k_base",
}

# Characters per token approximations for non-OpenAI models
# Based on empirical testing and model documentation
CHARS_PER_TOKEN = {
    # Anthropic (Claude uses similar tokenization to GPT-4)
    "claude": 4.0,
    # Google (Gemini uses SentencePiece)
    "gemini": 4.2,
    # Mistral (uses SentencePiece)
    "mistral": 4.0,
    # xAI (similar to GPT)
    "grok": 4.0,
    # Default approximation
    "default": 4.0,
}


ModelFamily = Literal["openai", "anthropic", "google", "mistral", "xai", "default"]


def _get_model_family(model: str) -> ModelFamily:
    """Determine the model family from model name."""
    model_lower = model.lower()

    if any(x in model_lower for x in ["gpt", "openai", "text-embedding", "davinci", "curie"]):
        return "openai"
    elif any(x in model_lower for x in ["claude", "anthropic"]):
        return "anthropic"
    elif any(x in model_lower for x in ["gemini", "palm", "bard", "google"]):
        return "google"
    elif any(x in model_lower for x in ["mistral", "mixtral", "codestral"]):
        return "mistral"
    elif any(x in model_lower for x in ["grok", "xai"]):
        return "xai"

    return "default"


@lru_cache(maxsize=16)
def _get_tiktoken_encoding(encoding_name: str):
    """Get tiktoken encoding (cached)."""
    if not TIKTOKEN_AVAILABLE:
        return None
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(f"Failed to load tiktoken encoding {encoding_name}: {e}")
        return None


class TokenCounter:
    """
    Multi-model token counter.

    Uses tiktoken for accurate OpenAI token counts and character-based
    approximations for other providers.
    """

    def __init__(self, default_model: str = "gpt-4"):
        """
        Initialize token counter.

        Args:
            default_model: Default model to use for counting
        """
        self.default_model = default_model
        self._cache: dict[tuple[str, str], int] = {}

    def count(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in text for a specific model.

        Args:
            text: The text to count tokens for
            model: Model name (defaults to self.default_model)

        Returns:
            Token count
        """
        if not text:
            return 0

        model = model or self.default_model
        family = _get_model_family(model)

        # Use tiktoken for OpenAI models
        if family == "openai" and TIKTOKEN_AVAILABLE:
            return self._count_tiktoken(text, model)

        # Use approximation for other models
        return self._count_approximate(text, family)

    def _count_tiktoken(self, text: str, model: str) -> int:
        """Count tokens using tiktoken."""
        # Determine encoding
        model_lower = model.lower()
        encoding_name = MODEL_ENCODINGS.get(model_lower, MODEL_ENCODINGS["default"])

        # Handle GPT-4o's newer encoding
        if "gpt-4o" in model_lower:
            encoding_name = "o200k_base"

        encoding = _get_tiktoken_encoding(encoding_name)
        if encoding is None:
            return self._count_approximate(text, "openai")

        try:
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"tiktoken encoding failed: {e}")
            return self._count_approximate(text, "openai")

    def _count_approximate(self, text: str, family: ModelFamily) -> int:
        """
        Approximate token count based on character ratio.

        Uses empirically-derived characters-per-token ratios.
        """
        chars_per_token = CHARS_PER_TOKEN.get(family, CHARS_PER_TOKEN["default"])

        # Basic approximation
        base_count = len(text) / chars_per_token

        # Adjust for whitespace (tokenizers often merge whitespace)
        whitespace_count = len(re.findall(r"\s+", text))
        adjustment = whitespace_count * 0.5

        # Adjust for special characters (often become separate tokens)
        special_count = len(re.findall(r"[^\w\s]", text))
        adjustment -= special_count * 0.3

        return max(1, int(base_count - adjustment))

    def count_messages(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
    ) -> int:
        """
        Count tokens in a list of chat messages.

        Accounts for message formatting overhead.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name

        Returns:
            Total token count including formatting
        """
        model = model or self.default_model
        total = 0

        # Message formatting overhead (varies by model)
        family = _get_model_family(model)

        # OpenAI uses ~4 tokens per message for formatting
        # Claude uses ~3 tokens per message
        overhead_per_message = 4 if family == "openai" else 3

        for message in messages:
            content = message.get("content", "")
            total += self.count(content, model)
            total += overhead_per_message

        # Final overhead for the prompt/response structure
        total += 3

        return total

    def estimate_chunks(
        self,
        text: str,
        chunk_size: int,
        overlap: int = 0,
        model: Optional[str] = None,
    ) -> int:
        """
        Estimate number of chunks needed for text.

        Args:
            text: Text to chunk
            chunk_size: Target tokens per chunk
            overlap: Token overlap between chunks
            model: Model for token counting

        Returns:
            Estimated number of chunks
        """
        total_tokens = self.count(text, model)

        if total_tokens <= chunk_size:
            return 1

        effective_size = chunk_size - overlap
        if effective_size <= 0:
            effective_size = chunk_size // 2

        return max(1, (total_tokens - overlap) // effective_size + 1)

    def fits_context(
        self,
        text: str,
        model: str,
        max_tokens: Optional[int] = None,
        reserve_tokens: int = 1000,
    ) -> bool:
        """
        Check if text fits within model context window.

        Args:
            text: Text to check
            model: Model name
            max_tokens: Override max tokens (otherwise uses model default)
            reserve_tokens: Tokens to reserve for response

        Returns:
            True if text fits in context
        """
        from aragora.documents.models import get_model_token_limit

        if max_tokens is None:
            max_tokens = get_model_token_limit(model)

        available = max_tokens - reserve_tokens
        token_count = self.count(text, model)

        return token_count <= available

    def truncate_to_tokens(
        self,
        text: str,
        max_tokens: int,
        model: Optional[str] = None,
        suffix: str = "...",
    ) -> str:
        """
        Truncate text to fit within token limit.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens
            model: Model for counting
            suffix: Suffix to add if truncated

        Returns:
            Truncated text
        """
        if self.count(text, model) <= max_tokens:
            return text

        # Binary search for the right cutoff point
        low, high = 0, len(text)
        suffix_tokens = self.count(suffix, model)
        target = max_tokens - suffix_tokens

        while low < high:
            mid = (low + high + 1) // 2
            if self.count(text[:mid], model) <= target:
                low = mid
            else:
                high = mid - 1

        # Try to cut at word boundary
        truncated = text[:low]
        last_space = truncated.rfind(" ")
        if last_space > low * 0.8:  # Don't cut too much
            truncated = truncated[:last_space]

        return truncated.rstrip() + suffix


# Global token counter instance
_token_counter: Optional[TokenCounter] = None


def get_token_counter() -> TokenCounter:
    """Get the global token counter instance."""
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter()
    return _token_counter


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Convenience function to count tokens."""
    return get_token_counter().count(text, model)


__all__ = [
    "TokenCounter",
    "get_token_counter",
    "count_tokens",
    "TIKTOKEN_AVAILABLE",
    "MODEL_ENCODINGS",
    "CHARS_PER_TOKEN",
]
