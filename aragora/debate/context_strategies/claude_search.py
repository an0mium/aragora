"""
Claude Web Search Strategy.

Uses Claude's built-in web_search tool to gather current, high-quality
research information for debates.

This is typically the primary research method - uses Claude Opus 4.5
with web search capability.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .base import ContextStrategy

logger = logging.getLogger(__name__)

# Configurable timeout (default 240s to allow thorough web search)
CLAUDE_SEARCH_TIMEOUT = float(os.getenv("ARAGORA_CLAUDE_SEARCH_TIMEOUT", "240.0"))


class ClaudeSearchStrategy(ContextStrategy):
    """
    Gather research context using Claude's web search capability.

    This strategy calls into the research_phase module to perform
    web searches and synthesize results into debate context.
    """

    name = "claude_search"
    default_timeout = CLAUDE_SEARCH_TIMEOUT

    def is_available(self) -> bool:
        """Check if research_phase module is available."""
        try:
            from aragora.server.research_phase import research_for_debate  # noqa: F401

            return True
        except ImportError:
            return False

    async def gather(self, task: str, **kwargs: Any) -> str | None:
        """
        Perform web search using Claude's built-in web_search tool.

        Args:
            task: The debate topic/task description.

        Returns:
            Formatted research context, or None if search fails.
        """
        try:
            from aragora.server.research_phase import research_for_debate

            logger.info("[research] Starting Claude web search for debate context...")

            result = await research_for_debate(task)

            if result:
                trimmed = result.strip()
                # Reject low-signal summaries
                if "Key Sources" not in trimmed and len(trimmed) < 200:
                    logger.info(
                        "[research] Claude web search returned low-signal summary; ignoring"
                    )
                    return None
                logger.info("[research] Claude web search complete: %d chars", len(result))
                return result
            else:
                logger.info("[research] Claude web search returned no results")
                return None

        except ImportError:
            logger.debug("[research] research_phase module not available")
            return None
        except (ConnectionError, OSError) as e:
            # Expected: network or API issues
            logger.warning("[research] Claude web search network error: %s", e)
            return None
        except (ValueError, RuntimeError) as e:
            # Expected: API or response processing issues
            logger.warning("[research] Claude web search failed: %s", e)
            return None
        except Exception as e:  # noqa: BLE001
            # Unexpected error
            logger.warning("[research] Unexpected error in Claude web search: %s", e)
            return None
