"""
Unified Embedding Services for Aragora.

This module provides a single entry point for all embedding operations,
consolidating multiple implementations across the codebase.

Primary Entry Points:
    - get_embedding_provider(): Get the best available embedding provider
    - EmbeddingProvider: Base class for all providers
    - embed_text(): Simple function for one-off embeddings
    - embed_batch(): Batch embedding for efficiency

Supported Providers:
    - OpenAI (text-embedding-3-small, 1536 dimensions)
    - Google Gemini (text-embedding-004, 768 dimensions)
    - Ollama (local, nomic-embed-text, 768 dimensions)
    - Hash-based fallback (always available)

Vector Stores (for persistence):
    - WeaviateVectorStore
    - QdrantVectorStore
    - ChromaVectorStore
    - InMemoryVectorStore

Usage:
    from aragora.embeddings import get_embedding_provider, embed_text

    # Get the best available provider
    provider = get_embedding_provider()
    embedding = await provider.embed("Hello world")

    # Or use the simple function
    embedding = await embed_text("Hello world")

    # Batch embeddings (more efficient)
    embeddings = await provider.embed_batch(["text1", "text2", "text3"])

For vector stores, use:
    from aragora.knowledge.mound.vector_abstraction import (
        WeaviateVectorStore,
        QdrantVectorStore,
        ChromaVectorStore,
        InMemoryVectorStore,
    )
"""

from typing import List, Optional, Protocol, runtime_checkable

# Re-export from canonical implementations
from aragora.memory.embeddings import (
    EmbeddingProvider,
    OpenAIEmbedding,
    GeminiEmbedding,
    OllamaEmbedding,
    EmbeddingCache,
    SemanticRetriever,
)

# Re-export core protocol for type checking
from aragora.core_protocols import EmbeddingBackend


def get_embedding_provider() -> EmbeddingProvider:
    """
    Get the best available embedding provider.

    Tries providers in order of preference:
    1. OpenAI (if OPENAI_API_KEY is set)
    2. Gemini (if GEMINI_API_KEY is set)
    3. Ollama (if available locally)
    4. Hash-based fallback (always available)

    Returns:
        Best available EmbeddingProvider instance
    """
    import os

    # Try OpenAI
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbedding()

    # Try Gemini
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return GeminiEmbedding()

    # Try Ollama (local)
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=1)
        if response.status_code == 200:
            return OllamaEmbedding()
    except Exception:
        pass

    # Fallback to hash-based
    return EmbeddingProvider()


@runtime_checkable
class EmbeddingProviderProtocol(Protocol):
    """Protocol for embedding providers (for type checking)."""

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        ...

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        ...


# Simple convenience functions

_default_provider: Optional[EmbeddingProvider] = None


async def embed_text(text: str) -> List[float]:
    """
    Embed a single text using the best available provider.

    This is a convenience function for one-off embeddings.
    For batch operations, use get_embedding_provider() directly.

    Args:
        text: Text to embed

    Returns:
        Embedding vector as list of floats
    """
    global _default_provider
    if _default_provider is None:
        _default_provider = get_embedding_provider()
    return await _default_provider.embed(text)


async def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Embed multiple texts using the best available provider.

    More efficient than calling embed_text() multiple times.

    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors
    """
    global _default_provider
    if _default_provider is None:
        _default_provider = get_embedding_provider()
    return await _default_provider.embed_batch(texts)


def reset_default_provider() -> None:
    """Reset the default provider (useful for testing)."""
    global _default_provider
    _default_provider = None


__all__ = [
    # Core providers
    "EmbeddingProvider",
    "OpenAIEmbedding",
    "GeminiEmbedding",
    "OllamaEmbedding",
    # Factory
    "get_embedding_provider",
    # Cache
    "EmbeddingCache",
    # Retrieval
    "SemanticRetriever",
    # Protocols
    "EmbeddingBackend",
    "EmbeddingProviderProtocol",
    # Convenience functions
    "embed_text",
    "embed_batch",
    "reset_default_provider",
]
