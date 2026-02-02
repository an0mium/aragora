"""
Trending Topics Strategy (Pulse).

Gathers trending context from social platforms using the PulseManager.

Platforms:
- Google Trends
- Hacker News
- Reddit
- GitHub Trending
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .base import ContextStrategy

logger = logging.getLogger(__name__)

# Cache size limit
MAX_TRENDING_CACHE_SIZE = int(os.getenv("ARAGORA_MAX_TRENDING_CACHE", "50"))
TRENDING_TIMEOUT = float(os.getenv("ARAGORA_TRENDING_TIMEOUT", "5.0"))


class TrendingStrategy(ContextStrategy):
    """
    Gather trending topics from social platforms.

    Uses PulseManager with available ingestors to fetch
    current trending topics that may be relevant to debates.
    """

    name = "trending"
    default_timeout = TRENDING_TIMEOUT

    def __init__(self, prompt_builder: Any = None, enabled: bool = True) -> None:
        self._prompt_builder = prompt_builder
        self._enabled = enabled
        self._topics_cache: list[Any] = []

    def set_prompt_builder(self, prompt_builder: Any) -> None:
        """Set the prompt builder for trending injection."""
        self._prompt_builder = prompt_builder

    def get_topics(self) -> list[Any]:
        """Get cached trending topics."""
        return self._topics_cache

    def is_available(self) -> bool:
        """Check if pulse module is available."""
        if not self._enabled:
            return False
        try:
            from aragora.pulse.ingestor import PulseManager  # noqa: F401

            return True
        except ImportError:
            return False

    async def gather(self, task: str = "", **kwargs: Any) -> str | None:
        """
        Gather pulse/trending context from social platforms.

        Args:
            task: Not used for trending (task-agnostic).

        Returns:
            Formatted trending topics context, or None if unavailable.
        """
        if not self._enabled:
            logger.debug("[pulse] Trending context disabled")
            return None

        try:
            from aragora.pulse.ingestor import (
                GitHubTrendingIngestor,
                GoogleTrendsIngestor,
                HackerNewsIngestor,
                PulseManager,
                RedditIngestor,
            )

            manager = PulseManager()
            # Free, no-auth sources for real trending data:
            manager.add_ingestor("google", GoogleTrendsIngestor())
            manager.add_ingestor("hackernews", HackerNewsIngestor())
            manager.add_ingestor("reddit", RedditIngestor())
            manager.add_ingestor("github", GitHubTrendingIngestor())

            topics = await manager.get_trending_topics(limit_per_platform=3)

            if topics:
                # Cache topics
                self._topics_cache = list(topics)[:MAX_TRENDING_CACHE_SIZE]

                # Pass to prompt builder if available
                if self._prompt_builder:
                    self._prompt_builder.set_trending_topics(self._topics_cache)
                    logger.debug(
                        "[pulse] Injected %d trending topics into PromptBuilder",
                        len(self._topics_cache),
                    )

                # Format context
                trending_context = (
                    "## TRENDING CONTEXT\nCurrent trending topics that may be relevant:\n"
                )
                for t in topics[:5]:
                    trending_context += (
                        f"- {t.topic} ({t.platform}, {t.volume:,} engagement, {t.category})\n"
                    )
                return trending_context

        except ImportError as e:
            logger.debug("Pulse module not available: %s", e)
        except (ConnectionError, OSError) as e:
            logger.debug("Pulse context network error: %s", e)
        except (ValueError, RuntimeError) as e:
            logger.debug("Pulse context unavailable: %s", e)
        except Exception as e:  # noqa: BLE001
            logger.warning("Unexpected error getting pulse context: %s", e)

        return None
