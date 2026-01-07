"""
Quota detection and fallback utilities for API agents.

Provides shared logic for detecting quota/rate limit errors and falling back
to OpenRouter when the primary provider is unavailable.
"""

import logging
import os
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .api_agents import OpenRouterAgent

logger = logging.getLogger(__name__)

# Common keywords indicating quota/rate limit errors across providers
QUOTA_ERROR_KEYWORDS = frozenset([
    # Rate limiting
    "rate limit",
    "rate_limit",
    "ratelimit",
    "too many requests",
    # Quota exceeded
    "quota",
    "exceeded",
    "limit exceeded",
    "resource exhausted",
    "resource_exhausted",
    # Billing/credits
    "billing",
    "credit balance",
    "insufficient",
    "insufficient_quota",
    "purchase credits",
])


class QuotaFallbackMixin:
    """Mixin providing shared quota detection and OpenRouter fallback logic.

    This mixin extracts the common quota error detection and fallback pattern
    used by Gemini, Anthropic, OpenAI, and Grok agents.

    The mixin expects the following attributes on the class:
        - name: str - Agent name for logging
        - enable_fallback: bool - Whether fallback is enabled
        - fallback_model: str - Model to use with OpenRouter fallback
        - model: str - Current model name
        - max_tokens: int - Max tokens setting
        - timeout: int - Timeout setting

    Usage:
        class MyAgent(APIAgent, QuotaFallbackMixin):
            async def generate(self, prompt, context):
                # ... make API call ...
                if self.is_quota_error(status, error_text):
                    result = await self.fallback_generate(prompt, context)
                    if result is not None:
                        return result
                    # No fallback available, raise error
                    raise RuntimeError(...)
    """

    def is_quota_error(self, status_code: int, error_text: str) -> bool:
        """Check if an error indicates quota/rate limit issues.

        This is a unified check that works across providers:
        - 429: Rate limit (all providers)
        - 403: Can indicate quota exceeded (Gemini)

        Args:
            status_code: HTTP status code from response
            error_text: Error message text from response body

        Returns:
            True if this appears to be a quota/rate limit error
        """
        # 429 is universally rate limit
        if status_code == 429:
            return True

        # 403 can indicate quota exceeded (especially for Gemini)
        if status_code == 403:
            # Only treat as quota if error text contains quota keywords
            error_lower = error_text.lower()
            if any(kw in error_lower for kw in ["quota", "exceeded", "billing"]):
                return True

        # Check for quota-related keywords in any error
        error_lower = error_text.lower()
        return any(kw in error_lower for kw in QUOTA_ERROR_KEYWORDS)

    def _get_openrouter_fallback(self) -> Optional["OpenRouterAgent"]:
        """Get an OpenRouter fallback agent if available.

        Returns:
            OpenRouterAgent instance if OPENROUTER_API_KEY is set, None otherwise
        """
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_key:
            return None

        # Lazy import to avoid circular dependency
        from .api_agents import OpenRouterAgent

        # Use the configured fallback model or derive from current model
        fallback_model = getattr(self, 'fallback_model', None)
        if not fallback_model:
            # Try to map current model to an OpenRouter equivalent
            model = getattr(self, 'model', 'gpt-4')
            fallback_model = f"openai/{model}"

        return OpenRouterAgent(
            model=fallback_model,
            max_tokens=getattr(self, 'max_tokens', 4096),
            timeout=getattr(self, 'timeout', 120),
        )

    async def fallback_generate(
        self,
        prompt: str,
        context: Optional[list] = None,
    ) -> Optional[str]:
        """Attempt to generate using OpenRouter fallback.

        Args:
            prompt: The prompt to send
            context: Optional conversation context

        Returns:
            Generated response string if fallback succeeded, None otherwise
        """
        if not getattr(self, 'enable_fallback', True):
            return None

        fallback = self._get_openrouter_fallback()
        if not fallback:
            name = getattr(self, 'name', 'unknown')
            logger.warning(
                f"{name} quota exceeded but OPENROUTER_API_KEY not set - cannot fallback"
            )
            return None

        name = getattr(self, 'name', 'unknown')
        logger.warning(
            f"API quota/rate limit error for {name}, falling back to OpenRouter"
        )
        return await fallback.generate(prompt, context)

    async def fallback_generate_stream(
        self,
        prompt: str,
        context: Optional[list] = None,
    ):
        """Attempt to stream using OpenRouter fallback.

        Args:
            prompt: The prompt to send
            context: Optional conversation context

        Yields:
            Content tokens from fallback stream, or nothing if fallback unavailable
        """
        if not getattr(self, 'enable_fallback', True):
            return

        fallback = self._get_openrouter_fallback()
        if not fallback:
            name = getattr(self, 'name', 'unknown')
            logger.warning(
                f"{name} quota exceeded but OPENROUTER_API_KEY not set - cannot fallback"
            )
            return

        name = getattr(self, 'name', 'unknown')
        logger.warning(
            f"API quota/rate limit error for {name}, falling back to OpenRouter streaming"
        )
        async for token in fallback.generate_stream(prompt, context):
            yield token
