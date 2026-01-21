"""Embedding provider backends.

Each backend implements the EmbeddingBackend protocol for generating
text embeddings using different services (OpenAI, Gemini, Ollama, etc.).
"""

from abc import ABC, abstractmethod

from aragora.core.embeddings.types import EmbeddingCircuitOpenError, EmbeddingConfig

# Circuit breaker configuration
CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_COOLDOWN = 60.0


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends.

    All embedding providers must implement this interface.

    Supports optional circuit breaker integration for resilience.
    """

    def __init__(self, config: EmbeddingConfig, use_circuit_breaker: bool = True):
        """Initialize backend with configuration.

        Args:
            config: Embedding configuration
            use_circuit_breaker: Whether to use circuit breaker for resilience
        """
        self.config = config
        self.dimension = config.dimension
        self._use_circuit_breaker = use_circuit_breaker
        self._circuit_breaker = None

        if use_circuit_breaker:
            try:
                from aragora.resilience import get_circuit_breaker

                self._circuit_breaker = get_circuit_breaker(
                    f"embedding_{self.__class__.__name__.lower()}",
                    failure_threshold=CIRCUIT_BREAKER_THRESHOLD,
                    cooldown_seconds=CIRCUIT_BREAKER_COOLDOWN,
                )
            except ImportError:
                pass  # Resilience module not available

    def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker allows requests.

        Raises:
            EmbeddingCircuitOpenError: If circuit breaker is open
        """
        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            raise EmbeddingCircuitOpenError(
                f"Circuit breaker open for {self.provider_name}",
                provider=self.provider_name,
            )

    def _record_success(self) -> None:
        """Record successful request to circuit breaker."""
        if self._circuit_breaker:
            self._circuit_breaker.record_success()

    def _record_failure(self) -> None:
        """Record failed request to circuit breaker."""
        if self._circuit_breaker:
            self._circuit_breaker.record_failure()

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
        results = await asyncio.gather(*[self.embed(t) for t in texts], return_exceptions=True)

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
