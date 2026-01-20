"""Embedding provider backends.

Each backend implements the EmbeddingBackend protocol for generating
text embeddings using different services (OpenAI, Gemini, Ollama, etc.).
"""

from abc import ABC, abstractmethod
from typing import Optional

from aragora.core.embeddings.types import EmbeddingConfig


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends.

    All embedding providers must implement this interface.
    """

    def __init__(self, config: EmbeddingConfig):
        """Initialize backend with configuration.

        Args:
            config: Embedding configuration
        """
        self.config = config
        self.dimension = config.dimension

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'openai', 'gemini')."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name being used."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Default implementation processes sequentially. Backends with
        native batch APIs should override for better performance.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        import asyncio
        import logging

        logger = logging.getLogger(__name__)

        # Use return_exceptions to prevent first failure from canceling others
        results = await asyncio.gather(
            *[self.embed(t) for t in texts],
            return_exceptions=True
        )

        # Process results, replacing exceptions with zero vectors
        embeddings = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.warning(f"embed_batch: failed to embed text {i}: {result}")
                embeddings.append([0.0] * self.dimension)
            else:
                embeddings.append(result)

        return embeddings

    def is_available(self) -> bool:
        """Check if this backend is available (has required credentials/connectivity)."""
        return True


# Import concrete backends
from aragora.core.embeddings.backends.openai import OpenAIBackend
from aragora.core.embeddings.backends.gemini import GeminiBackend
from aragora.core.embeddings.backends.ollama import OllamaBackend
from aragora.core.embeddings.backends.hash import HashBackend

__all__ = [
    "EmbeddingBackend",
    "OpenAIBackend",
    "GeminiBackend",
    "OllamaBackend",
    "HashBackend",
]
