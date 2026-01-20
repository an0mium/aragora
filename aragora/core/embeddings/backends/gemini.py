"""Google Gemini embedding backend."""

import asyncio
import logging
import os
from typing import Optional

import aiohttp

from aragora.core.embeddings.backends import EmbeddingBackend
from aragora.core.embeddings.types import EmbeddingConfig

logger = logging.getLogger(__name__)


class GeminiBackend(EmbeddingBackend):
    """Google Gemini text-embedding-004 backend.

    Uses the Gemini embeddings API with automatic retry.
    """

    DEFAULT_MODEL = "text-embedding-004"
    API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, config: EmbeddingConfig):
        """Initialize Gemini backend.

        Args:
            config: Embedding configuration
        """
        super().__init__(config)
        self._api_key = config.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._model = config.model or self.DEFAULT_MODEL
        self._timeout = aiohttp.ClientTimeout(total=config.timeout)
        self.dimension = 768  # Gemini text-embedding-004

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        """Check if Gemini API key is available."""
        return bool(self._api_key)

    async def embed(self, text: str) -> list[float]:
        """Generate embedding using Gemini API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if not text:
            return [0.0] * self.dimension

        url = f"{self.API_BASE}/{self._model}:embedContent"

        for attempt in range(self.config.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=self._timeout) as session:
                    async with session.post(
                        url,
                        headers={
                            "x-goog-api-key": self._api_key,
                            "Content-Type": "application/json",
                        },
                        json={"content": {"parts": [{"text": text}]}},
                    ) as response:
                        if response.status == 429:
                            delay = self.config.base_delay * (2 ** attempt)
                            logger.warning(f"Gemini rate limited, retrying in {delay}s")
                            await asyncio.sleep(delay)
                            continue

                        if response.status != 200:
                            error_text = await response.text()
                            raise RuntimeError(
                                f"Gemini API error ({response.status}): {error_text}"
                            )

                        data = await response.json()
                        return data["embedding"]["values"]

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                if attempt == self.config.max_retries - 1:
                    raise RuntimeError(f"Gemini API call failed after retries: {e}")
                delay = self.config.base_delay * (2 ** attempt)
                logger.warning(f"Gemini API call failed, retrying in {delay}s: {e}")
                await asyncio.sleep(delay)

        raise RuntimeError("Gemini API call failed after max retries")
