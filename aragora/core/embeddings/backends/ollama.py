"""Ollama local embedding backend."""

import asyncio
import logging
import os
import socket
from typing import Optional

import aiohttp

from aragora.core.embeddings.backends import EmbeddingBackend
from aragora.core.embeddings.types import EmbeddingConfig

logger = logging.getLogger(__name__)


class OllamaBackend(EmbeddingBackend):
    """Local Ollama embeddings backend.

    Uses a local Ollama instance for embeddings, requiring no API key.
    Falls back gracefully if Ollama is not running.
    """

    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_HOST = "http://localhost:11434"

    def __init__(self, config: EmbeddingConfig):
        """Initialize Ollama backend.

        Args:
            config: Embedding configuration
        """
        super().__init__(config)
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
        """
        if not text:
            return [0.0] * self.dimension

        url = f"{self._host}/api/embeddings"

        for attempt in range(self.config.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=self._timeout) as session:
                    async with session.post(
                        url,
                        json={"model": self._model, "prompt": text},
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise RuntimeError(
                                f"Ollama API error ({response.status}): {error_text}"
                            )

                        data = await response.json()
                        return data["embedding"]

            except aiohttp.ClientConnectorError as e:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self._host}. "
                    "Is Ollama running? Start with: ollama serve"
                ) from e

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                if attempt == self.config.max_retries - 1:
                    raise RuntimeError(f"Ollama API call failed after retries: {e}")
                delay = self.config.base_delay * (2 ** attempt)
                logger.warning(f"Ollama API call failed, retrying in {delay}s: {e}")
                await asyncio.sleep(delay)

        raise RuntimeError("Ollama API call failed after max retries")
