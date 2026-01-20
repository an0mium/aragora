"""OpenAI embedding backend."""

import asyncio
import logging
import os
from typing import Optional

import aiohttp

from aragora.core.embeddings.backends import EmbeddingBackend
from aragora.core.embeddings.types import EmbeddingConfig

logger = logging.getLogger(__name__)


class OpenAIBackend(EmbeddingBackend):
    """OpenAI text-embedding-3-small backend.

    Uses the OpenAI embeddings API with automatic retry and rate limiting.
    """

    DEFAULT_MODEL = "text-embedding-3-small"
    API_URL = "https://api.openai.com/v1/embeddings"

    def __init__(self, config: EmbeddingConfig):
        """Initialize OpenAI backend.

        Args:
            config: Embedding configuration
        """
        super().__init__(config)
        self._api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        self._model = config.model or self.DEFAULT_MODEL
        self._timeout = aiohttp.ClientTimeout(total=config.timeout)

        # Set dimension based on model
        model_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self.dimension = model_dims.get(self._model, 1536)

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        """Check if OpenAI API key is available."""
        return bool(self._api_key)

    async def embed(self, text: str) -> list[float]:
        """Generate embedding using OpenAI API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return await self._call_api([text])[0] if text else [0.0] * self.dimension

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts using batch API.

        OpenAI supports native batching which is more efficient.
        """
        if not texts:
            return []
        return await self._call_api(texts)

    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Call OpenAI embeddings API with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=self._timeout) as session:
                    async with session.post(
                        self.API_URL,
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json={"model": self._model, "input": texts},
                    ) as response:
                        if response.status == 429:
                            # Rate limited - retry with backoff
                            delay = self.config.base_delay * (2 ** attempt)
                            logger.warning(f"OpenAI rate limited, retrying in {delay}s")
                            await asyncio.sleep(delay)
                            continue

                        if response.status != 200:
                            error_text = await response.text()
                            raise RuntimeError(
                                f"OpenAI API error ({response.status}): {error_text}"
                            )

                        data = await response.json()
                        # Sort by index to maintain order
                        return [
                            d["embedding"]
                            for d in sorted(data["data"], key=lambda x: x["index"])
                        ]

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                if attempt == self.config.max_retries - 1:
                    raise RuntimeError(f"OpenAI API call failed after retries: {e}")
                delay = self.config.base_delay * (2 ** attempt)
                logger.warning(f"OpenAI API call failed, retrying in {delay}s: {e}")
                await asyncio.sleep(delay)

        raise RuntimeError("OpenAI API call failed after max retries")
