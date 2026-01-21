"""Ollama local embedding backend."""

import asyncio
import logging
import os
import socket

import aiohttp

from aragora.core.embeddings.backends import EmbeddingBackend
from aragora.core.embeddings.types import (
    EmbeddingConfig,
    EmbeddingTimeoutError,
    EmbeddingConnectionError,
    EmbeddingModelError,
    EmbeddingError,
)

logger = logging.getLogger(__name__)


class OllamaBackend(EmbeddingBackend):
    """Local Ollama embeddings backend.

    Uses a local Ollama instance for embeddings, requiring no API key.
    Falls back gracefully if Ollama is not running.
    Integrates with circuit breaker for resilience.
    """

    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_HOST = "http://localhost:11434"

    def __init__(self, config: EmbeddingConfig, use_circuit_breaker: bool = True):
        """Initialize Ollama backend.

        Args:
            config: Embedding configuration
            use_circuit_breaker: Whether to use circuit breaker
        """
        super().__init__(config, use_circuit_breaker)
        self._host = config.ollama_host or os.environ.get("OLLAMA_HOST", self.DEFAULT_HOST)
        self._model = config.model or self.DEFAULT_MODEL
        self._timeout = aiohttp.ClientTimeout(total=config.timeout)
        self.dimension = 768  # nomic-embed-text default

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            host = self._host.replace("http://", "").replace("https://", "")
            port = 11434

            if ":" in host:
                parts = host.rsplit(":", 1)
                if len(parts) == 2:
                    host = parts[0]
                    try:
                        port = int(parts[1])
                    except ValueError:
                        pass

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.5)
                result = sock.connect_ex((host, port))
                return result == 0

        except (OSError, socket.error, socket.timeout):
            return False

    async def embed(self, text: str) -> list[float]:
        """Generate embedding using local Ollama.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            EmbeddingConnectionError: If Ollama is not running
            EmbeddingModelError: If model is not available
            EmbeddingTimeoutError: On timeout
            EmbeddingError: On other errors
        """
        if not text:
            return [0.0] * self.dimension

        # Check circuit breaker
        self._check_circuit_breaker()

        url = f"{self._host}/api/embeddings"
        last_error: EmbeddingError | None = None

        for attempt in range(self.config.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=self._timeout) as session:
                    async with session.post(
                        url,
                        json={"model": self._model, "prompt": text},
                    ) as response:
                        if response.status == 404:
                            self._record_failure()
                            raise EmbeddingModelError(
                                f"Model '{self._model}' not found. Pull it with: ollama pull {self._model}",
                                provider=self.provider_name,
                                model=self._model,
                            )

                        if response.status != 200:
                            error_text = await response.text()
                            self._record_failure()
                            raise EmbeddingError(
                                f"Ollama API error: {error_text}",
                                provider=self.provider_name,
                                status_code=response.status,
                            )

                        data = await response.json()
                        self._record_success()
                        return data["embedding"]

            except aiohttp.ClientConnectorError as e:
                self._record_failure()
                raise EmbeddingConnectionError(
                    f"Cannot connect to Ollama at {self._host}. "
                    "Is Ollama running? Start with: ollama serve",
                    provider=self.provider_name,
                    host=self._host,
                    original_error=e,
                )

            except asyncio.TimeoutError as e:
                self._record_failure()
                last_error = EmbeddingTimeoutError(
                    f"Ollama API timeout after {self.config.timeout}s",
                    provider=self.provider_name,
                    timeout=self.config.timeout,
                    original_error=e,
                )
                if attempt < self.config.max_retries - 1:
                    delay = self.config.base_delay * (2**attempt)
                    logger.warning(f"Ollama timeout, retrying in {delay}s")
                    await asyncio.sleep(delay)
                    continue
                raise last_error

            except aiohttp.ClientError as e:
                self._record_failure()
                if attempt < self.config.max_retries - 1:
                    delay = self.config.base_delay * (2**attempt)
                    logger.warning(f"Ollama API error, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                    continue
                raise EmbeddingError(
                    f"Ollama API call failed: {e}",
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
            "Ollama API call failed after max retries", provider=self.provider_name
        )
