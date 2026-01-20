"""Hash-based fallback embedding backend.

Provides deterministic pseudo-embeddings when no API keys are available.
Useful for testing and graceful degradation.
"""

import hashlib
import struct
from typing import Optional

from aragora.core.embeddings.backends import EmbeddingBackend
from aragora.core.embeddings.types import EmbeddingConfig


class HashBackend(EmbeddingBackend):
    """Hash-based pseudo-embedding backend.

    Generates deterministic embeddings based on text hash.
    Does not provide semantic similarity but ensures the system
    continues to function without API access.

    Used as fallback when:
    - No API keys are configured
    - Ollama is not running
    - Testing without external dependencies
    """

    def __init__(self, config: EmbeddingConfig):
        """Initialize hash backend.

        Args:
            config: Embedding configuration
        """
        super().__init__(config)
        self.dimension = config.dimension or 256

    @property
    def provider_name(self) -> str:
        return "hash"

    @property
    def model_name(self) -> str:
        return "hash-pseudo-embedding"

    def is_available(self) -> bool:
        """Hash backend is always available."""
        return True

    async def embed(self, text: str) -> list[float]:
        """Generate deterministic pseudo-embedding from text hash.

        Uses multiple hash seeds to create a fixed-dimension vector.
        The same text will always produce the same embedding.

        Args:
            text: Text to embed

        Returns:
            Pseudo-embedding vector (deterministic but not semantic)
        """
        if not text:
            return [0.0] * self.dimension

        embedding = []
        for seed in range(self.dimension):
            h = hashlib.sha256(f"{seed}:{text}".encode()).digest()
            # Convert first 4 bytes to float in [-1, 1]
            val = struct.unpack("<i", h[:4])[0] / (2**31)
            embedding.append(val)

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Hash backend is fast enough to process sequentially.
        """
        return [await self.embed(text) for text in texts]
