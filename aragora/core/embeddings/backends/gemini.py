"""Google Gemini embedding backend."""

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


class GeminiBackend(EmbeddingBackend):
    """Google Gemini text-embedding-004 backend.

    Uses the Gemini embeddings API with automatic retry.
    Integrates with circuit breaker for resilience.
    """

    DEFAULT_MODEL = "text-embedding-004"
    API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, config: EmbeddingConfig, use_circuit_breaker: bool = True):
        """Initialize Gemini backend.

        Args:
            config: Embedding configuration
            use_circuit_breaker: Whether to use circuit breaker
        """
        super().__init__(config, use_circuit_breaker)
        self._api_key = (
            config.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
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

        Raises:
            EmbeddingError: On API errors
        """
        if not text:
            return [0.0] * self.dimension

        # Check circuit breaker
        self._check_circuit_breaker()

        url = f"{self.API_BASE}/{self._model}:embedContent"
        last_error: EmbeddingError | None = None

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
                            delay = self.config.base_delay * (2**attempt)
                            logger.warning(f"Gemini rate limited, retrying in {delay}s")
                            last_error = EmbeddingRateLimitError(
                                "Gemini rate limit exceeded",
                                provider=self.provider_name,
                                retry_after=delay,
                            )
                            await asyncio.sleep(delay)
                            continue

                        if response.status == 401:
                            self._record_failure()
                            raise EmbeddingAuthError(
                                "Invalid Gemini API key",
                                provider=self.provider_name,
                            )

                        if response.status == 403:
                            self._record_failure()
                            raise EmbeddingAuthError(
                                "Gemini API access forbidden - check API key permissions",
                                provider=self.provider_name,
                            )

                        if response.status == 402:
                            self._record_failure()
                            raise EmbeddingQuotaError(
                                "Gemini quota exceeded",
                                provider=self.provider_name,
                            )

                        if response.status != 200:
                            error_text = await response.text()
                            self._record_failure()
                            raise EmbeddingError(
                                f"Gemini API error: {error_text}",
                                provider=self.provider_name,
                                status_code=response.status,
                            )

                        data = await response.json()
                        self._record_success()
                        return data["embedding"]["values"]

            except asyncio.TimeoutError as e:
                self._record_failure()
                last_error = EmbeddingTimeoutError(
                    f"Gemini API timeout after {self.config.timeout}s",
                    provider=self.provider_name,
                    timeout=self.config.timeout,
                    original_error=e,
                )
                if attempt < self.config.max_retries - 1:
                    delay = self.config.base_delay * (2**attempt)
                    logger.warning(f"Gemini timeout, retrying in {delay}s")
                    await asyncio.sleep(delay)
                    continue
                raise last_error

            except aiohttp.ClientConnectorError as e:
                self._record_failure()
                last_error = EmbeddingConnectionError(
                    f"Cannot connect to Gemini API: {e}",
                    provider=self.provider_name,
                    host=self.API_BASE,
                    original_error=e,
                )
                if attempt < self.config.max_retries - 1:
                    delay = self.config.base_delay * (2**attempt)
                    logger.warning(f"Gemini connection error, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                    continue
                raise last_error

            except aiohttp.ClientError as e:
                self._record_failure()
                if attempt < self.config.max_retries - 1:
                    delay = self.config.base_delay * (2**attempt)
                    logger.warning(f"Gemini API error, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                    continue
                raise EmbeddingError(
                    f"Gemini API call failed: {e}",
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
            "Gemini API call failed after max retries", provider=self.provider_name
        )
