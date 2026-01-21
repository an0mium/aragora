"""OpenAI embedding backend."""

import asyncio
import logging
import os

import aiohttp

from aragora.core.embeddings.backends import EmbeddingBackend
from aragora.core.embeddings.types import (
    EmbeddingConfig,
    EmbeddingRateLimitError,
    EmbeddingAuthError,
    EmbeddingTimeoutError,
    EmbeddingConnectionError,
    EmbeddingQuotaError,
    EmbeddingError,
)

logger = logging.getLogger(__name__)


class OpenAIBackend(EmbeddingBackend):
    """OpenAI text-embedding-3-small backend.

    Uses the OpenAI embeddings API with automatic retry and rate limiting.
    Integrates with circuit breaker for resilience.
    """

    DEFAULT_MODEL = "text-embedding-3-small"
    API_URL = "https://api.openai.com/v1/embeddings"

    def __init__(self, config: EmbeddingConfig, use_circuit_breaker: bool = True):
        """Initialize OpenAI backend.

        Args:
            config: Embedding configuration
            use_circuit_breaker: Whether to use circuit breaker
        """
        super().__init__(config, use_circuit_breaker)
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

        Raises:
            EmbeddingError: On API errors
        """
        if not text:
            return [0.0] * self.dimension

        results = await self._call_api([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts using batch API.

        OpenAI supports native batching which is more efficient.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: On API errors
        """
        if not texts:
            return []
        return await self._call_api(texts)

    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Call OpenAI embeddings API with retry logic.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingRateLimitError: On rate limit (429)
            EmbeddingAuthError: On authentication error (401)
            EmbeddingQuotaError: On quota exceeded (402)
            EmbeddingTimeoutError: On timeout
            EmbeddingConnectionError: On connection failure
            EmbeddingError: On other API errors
        """
        # Check circuit breaker
        self._check_circuit_breaker()

        last_error: EmbeddingError | None = None

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
                            retry_after = response.headers.get("Retry-After")
                            retry_seconds = float(retry_after) if retry_after else None
                            delay = retry_seconds or self.config.base_delay * (2**attempt)
                            logger.warning(f"OpenAI rate limited, retrying in {delay}s")
                            last_error = EmbeddingRateLimitError(
                                "OpenAI rate limit exceeded",
                                provider=self.provider_name,
                                retry_after=delay,
                            )
                            await asyncio.sleep(delay)
                            continue

                        if response.status == 401:
                            self._record_failure()
                            raise EmbeddingAuthError(
                                "Invalid OpenAI API key",
                                provider=self.provider_name,
                            )

                        if response.status == 402:
                            self._record_failure()
                            raise EmbeddingQuotaError(
                                "OpenAI quota exceeded",
                                provider=self.provider_name,
                            )

                        if response.status != 200:
                            error_text = await response.text()
                            self._record_failure()
                            raise EmbeddingError(
                                f"OpenAI API error: {error_text}",
                                provider=self.provider_name,
                                status_code=response.status,
                            )

                        data = await response.json()
                        self._record_success()
                        # Sort by index to maintain order
                        return [
                            d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])
                        ]

            except asyncio.TimeoutError as e:
                self._record_failure()
                last_error = EmbeddingTimeoutError(
                    f"OpenAI API timeout after {self.config.timeout}s",
                    provider=self.provider_name,
                    timeout=self.config.timeout,
                    original_error=e,
                )
                if attempt < self.config.max_retries - 1:
                    delay = self.config.base_delay * (2**attempt)
                    logger.warning(f"OpenAI timeout, retrying in {delay}s")
                    await asyncio.sleep(delay)
                    continue
                raise last_error

            except aiohttp.ClientConnectorError as e:
                self._record_failure()
                last_error = EmbeddingConnectionError(
                    f"Cannot connect to OpenAI API: {e}",
                    provider=self.provider_name,
                    host=self.API_URL,
                    original_error=e,
                )
                if attempt < self.config.max_retries - 1:
                    delay = self.config.base_delay * (2**attempt)
                    logger.warning(f"OpenAI connection error, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                    continue
                raise last_error

            except aiohttp.ClientError as e:
                self._record_failure()
                if attempt < self.config.max_retries - 1:
                    delay = self.config.base_delay * (2**attempt)
                    logger.warning(f"OpenAI API error, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                    continue
                raise EmbeddingError(
                    f"OpenAI API call failed: {e}",
                    provider=self.provider_name,
                    original_error=e,
                )

            except EmbeddingError:
                # Re-raise our typed exceptions
                raise

        # Should not reach here, but handle if we do
        if last_error:
            raise last_error
        raise EmbeddingError(
            "OpenAI API call failed after max retries", provider=self.provider_name
        )
