"""Type definitions for the unified embedding service."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EmbeddingProvider(str, Enum):
    """Available embedding providers."""

    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    HASH = "hash"  # Fallback hash-based embeddings


@dataclass
class EmbeddingConfig:
    """Configuration for embedding service.

    Attributes:
        provider: Embedding provider to use (auto-detected if not specified)
        model: Model name (provider-specific)
        dimension: Embedding dimension (auto-set based on model)
        api_key: API key (from environment if not specified)
        cache_enabled: Whether to cache embeddings
        cache_ttl: Cache TTL in seconds
        cache_max_size: Maximum cache entries
    """

    provider: Optional[str] = None  # Auto-detect if None
    model: Optional[str] = None  # Provider default if None
    dimension: int = 1536  # text-embedding-3-small default
    api_key: Optional[str] = None

    # Cache settings
    cache_enabled: bool = True
    cache_ttl: float = 3600.0  # 1 hour default
    cache_max_size: int = 1000

    # Retry settings
    max_retries: int = 3
    base_delay: float = 1.0
    timeout: float = 30.0

    # Provider-specific
    ollama_host: Optional[str] = None  # Default: http://localhost:11434

    def __post_init__(self):
        """Set dimension based on model if known."""
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "text-embedding-004": 768,  # Gemini
            "nomic-embed-text": 768,  # Ollama
        }
        if self.model and self.model in model_dimensions:
            self.dimension = model_dimensions[self.model]


@dataclass
class EmbeddingResult:
    """Result of an embedding operation.

    Attributes:
        embedding: The embedding vector
        text: Original text (truncated)
        provider: Provider that generated the embedding
        model: Model used
        cached: Whether result was from cache
        dimension: Embedding dimension
    """

    embedding: list[float]
    text: str = ""
    provider: str = "unknown"
    model: str = "unknown"
    cached: bool = False
    dimension: int = 0

    def __post_init__(self):
        if self.dimension == 0 and self.embedding:
            self.dimension = len(self.embedding)


@dataclass
class CacheStats:
    """Statistics for embedding cache.

    Attributes:
        size: Current number of cached entries
        valid: Number of non-expired entries
        hits: Total cache hits
        misses: Total cache misses
        hit_rate: Cache hit rate (0-1)
        ttl_seconds: Cache TTL
    """

    size: int = 0
    valid: int = 0
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    ttl_seconds: float = 3600.0
