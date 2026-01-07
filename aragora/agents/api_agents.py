"""
API-based agent implementations.

These agents call AI APIs directly (HTTP), enabling use without CLI tools.
Supports Gemini, Ollama (local), and direct OpenAI/Anthropic API calls.
"""

import asyncio
import aiohttp
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

from aragora.agents.base import CritiqueMixin
from aragora.config import DB_TIMEOUT_SECONDS, get_api_key
from aragora.core import Agent, Critique, Message
from aragora.server.error_utils import sanitize_error_text as _sanitize_error_message

logger = logging.getLogger(__name__)

# Maximum buffer size for streaming responses (prevents DoS via memory exhaustion)
MAX_STREAM_BUFFER_SIZE = 10 * 1024 * 1024  # 10MB


# ============================================================================
# OpenRouter Rate Limiting
# ============================================================================

@dataclass
class OpenRouterTier:
    """Rate limit configuration for an OpenRouter pricing tier."""
    name: str
    requests_per_minute: int
    tokens_per_minute: int = 0  # 0 = unlimited
    burst_size: int = 10  # Allow short bursts


# OpenRouter tier configurations (based on their pricing)
OPENROUTER_TIERS = {
    "free": OpenRouterTier(name="free", requests_per_minute=20, burst_size=5),
    "basic": OpenRouterTier(name="basic", requests_per_minute=60, burst_size=15),
    "standard": OpenRouterTier(name="standard", requests_per_minute=200, burst_size=30),
    "premium": OpenRouterTier(name="premium", requests_per_minute=500, burst_size=50),
    "unlimited": OpenRouterTier(name="unlimited", requests_per_minute=1000, burst_size=100),
}


class OpenRouterRateLimiter:
    """Rate limiter for OpenRouter API calls.

    Uses token bucket algorithm with configurable tiers.
    Thread-safe for use across multiple agent instances.
    """

    def __init__(self, tier: str = "standard"):
        """
        Initialize rate limiter with specified tier.

        Tier can be set via OPENROUTER_TIER environment variable.
        """
        tier_name = os.environ.get("OPENROUTER_TIER", tier).lower()
        self.tier = OPENROUTER_TIERS.get(tier_name, OPENROUTER_TIERS["standard"])

        self._tokens = float(self.tier.burst_size)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

        # Track rate limit headers from API
        self._api_limit: Optional[int] = None
        self._api_remaining: Optional[int] = None
        self._api_reset: Optional[float] = None

        logger.debug(f"OpenRouter rate limiter initialized: tier={self.tier.name}, rpm={self.tier.requests_per_minute}")

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed_minutes = (now - self._last_refill) / 60.0
        refill_amount = elapsed_minutes * self.tier.requests_per_minute
        self._tokens = min(self.tier.burst_size, self._tokens + refill_amount)
        self._last_refill = now

    async def acquire(self, timeout: float = 30.0) -> bool:
        """
        Acquire permission to make an API request.

        Blocks until a token is available or timeout is reached.
        Returns True if acquired, False if timed out.
        """
        deadline = time.monotonic() + timeout

        while True:
            with self._lock:
                self._refill()

                # Check API-reported limits if available
                if self._api_remaining is not None and self._api_remaining <= 0:
                    wait_time = (self._api_reset or 60) - time.time()
                    if wait_time > 0 and wait_time < timeout:
                        logger.debug(f"OpenRouter API limit reached, waiting {wait_time:.1f}s")
                        await asyncio.sleep(min(wait_time, 1.0))
                        continue

                if self._tokens >= 1:
                    self._tokens -= 1
                    return True

            # Wait and retry
            if time.monotonic() >= deadline:
                logger.warning("OpenRouter rate limit timeout")
                return False

            wait_time = 60.0 / self.tier.requests_per_minute  # Time for 1 token
            await asyncio.sleep(min(wait_time, 1.0))

    def update_from_headers(self, headers: dict) -> None:
        """Update rate limit state from API response headers.

        OpenRouter returns standard rate limit headers:
        - X-RateLimit-Limit: Total requests allowed
        - X-RateLimit-Remaining: Requests remaining
        - X-RateLimit-Reset: Unix timestamp when limit resets
        """
        with self._lock:
            if "X-RateLimit-Limit" in headers:
                try:
                    self._api_limit = int(headers["X-RateLimit-Limit"])
                except ValueError as e:
                    logger.warning(f"Failed to parse X-RateLimit-Limit header: {headers.get('X-RateLimit-Limit')!r} - {e}")

            if "X-RateLimit-Remaining" in headers:
                try:
                    self._api_remaining = int(headers["X-RateLimit-Remaining"])
                except ValueError as e:
                    logger.warning(f"Failed to parse X-RateLimit-Remaining header: {headers.get('X-RateLimit-Remaining')!r} - {e}")

            if "X-RateLimit-Reset" in headers:
                try:
                    self._api_reset = float(headers["X-RateLimit-Reset"])
                except ValueError as e:
                    logger.warning(f"Failed to parse X-RateLimit-Reset header: {headers.get('X-RateLimit-Reset')!r} - {e}")

    def release_on_error(self) -> None:
        """Release a token back on request error (optional, for retries)."""
        with self._lock:
            self._tokens = min(self.tier.burst_size, self._tokens + 1.0)

    @property
    def stats(self) -> dict:
        """Get current rate limiter statistics."""
        with self._lock:
            return {
                "tier": self.tier.name,
                "rpm_limit": self.tier.requests_per_minute,
                "tokens_available": int(self._tokens),
                "burst_size": self.tier.burst_size,
                "api_limit": self._api_limit,
                "api_remaining": self._api_remaining,
            }


# Global rate limiter instance (shared across all OpenRouterAgent instances)
_openrouter_limiter: Optional[OpenRouterRateLimiter] = None
_openrouter_limiter_lock = threading.Lock()


def get_openrouter_limiter() -> OpenRouterRateLimiter:
    """Get or create the global OpenRouter rate limiter."""
    global _openrouter_limiter
    with _openrouter_limiter_lock:
        if _openrouter_limiter is None:
            _openrouter_limiter = OpenRouterRateLimiter()
        return _openrouter_limiter


def set_openrouter_tier(tier: str) -> None:
    """Set the OpenRouter rate limit tier.

    Valid tiers: free, basic, standard, premium, unlimited
    """
    global _openrouter_limiter
    with _openrouter_limiter_lock:
        _openrouter_limiter = OpenRouterRateLimiter(tier=tier)


class APIAgent(CritiqueMixin, Agent):
    """Base class for API-based agents."""

    def __init__(
        self,
        name: str,
        model: str,
        role: str = "proposer",
        timeout: int = 120,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        super().__init__(name, model, role)
        self.timeout = timeout
        self.api_key = api_key
        self.base_url = base_url
        self.agent_type = "api"  # Default for API agents

    def _build_context_prompt(self, context: list[Message] | None = None) -> str:
        """Build context from previous messages.

        Delegates to CritiqueMixin (no truncation for API agents).
        """
        return CritiqueMixin._build_context_prompt(self, context, truncate=False)

    # _parse_critique is inherited from CritiqueMixin


class GeminiAgent(APIAgent):
    """Agent that uses Google Gemini API directly (not CLI).

    Note: The gemini CLI sends massive folder context by default and
    can exhaust quota quickly. This API agent is much more efficient.

    Supports automatic fallback to OpenRouter when Google API returns
    rate limit/quota errors.
    """

    # Model mapping from Gemini to OpenRouter format
    OPENROUTER_MODEL_MAP = {
        "gemini-3-pro-preview": "google/gemini-2.0-flash-001",
        "gemini-3-pro": "google/gemini-2.0-flash-001",
        "gemini-2.5-pro": "google/gemini-2.0-flash-001",
        "gemini-2.0-flash": "google/gemini-2.0-flash-001",
        "gemini-2.0-flash-001": "google/gemini-2.0-flash-001",
        "gemini-1.5-pro": "google/gemini-pro-1.5",
        "gemini-1.5-flash": "google/gemini-flash-1.5",
        "gemini-pro": "google/gemini-pro",
    }

    def __init__(
        self,
        name: str = "gemini",
        model: str = "gemini-3-pro-preview",  # Gemini 3 Pro Preview - advanced reasoning
        role: str = "proposer",
        timeout: int = 120,
        api_key: str | None = None,
        enable_fallback: bool = True,
    ):
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            api_key=api_key or get_api_key("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta",
        )
        self.agent_type = "gemini"
        self.enable_fallback = enable_fallback
        self._fallback_agent = None  # Lazy-loaded OpenRouter fallback

    def _get_fallback_agent(self):
        """Get or create the OpenRouter fallback agent for Gemini models."""
        if self._fallback_agent is None:
            # Map the model to OpenRouter format
            openrouter_model = self.OPENROUTER_MODEL_MAP.get(
                self.model, "google/gemini-2.0-flash-001"
            )

            # OpenRouterAgent is defined in this module
            self._fallback_agent = OpenRouterAgent(
                name=f"{self.name}_fallback",
                model=openrouter_model,
                role=self.role,
                system_prompt=self.system_prompt,
                timeout=self.timeout,
            )
            logger.info(f"Created OpenRouter fallback agent with model {openrouter_model}")
        return self._fallback_agent

    def _is_gemini_quota_error(self, status_code: int, error_text: str) -> bool:
        """Check if the error is a rate limit/quota error from Gemini."""
        # 429 is rate limit, 403 can be quota exceeded
        if status_code in (429, 403):
            return True
        # Check for quota-related messages in any error code
        quota_keywords = [
            "quota",
            "rate limit",
            "rate_limit",
            "resource exhausted",
            "resource_exhausted",
            "too many requests",
            "billing",
            "exceeded",
            "limit exceeded",
        ]
        error_lower = error_text.lower()
        return any(kw in error_lower for kw in quota_keywords)

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using Gemini API."""

        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        if self.system_prompt:
            full_prompt = f"System context: {self.system_prompt}\n\n{full_prompt}"

        url = f"{self.base_url}/models/{self.model}:generateContent"

        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 65536,  # Gemini 2.5 supports up to 65k output tokens
            },
        }

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    sanitized = _sanitize_error_message(error_text)

                    # Check if this is a quota/rate limit error and fallback is enabled
                    if self.enable_fallback and self._is_gemini_quota_error(
                        response.status, error_text
                    ):
                        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
                        if openrouter_key:
                            logger.warning(
                                f"Gemini API quota/rate limit error (status {response.status}), "
                                f"falling back to OpenRouter for {self.name}"
                            )
                            fallback = self._get_fallback_agent()
                            return await fallback.generate(prompt, context)
                        else:
                            logger.warning(
                                "Gemini quota exceeded but OPENROUTER_API_KEY not set - cannot fallback"
                            )

                    raise RuntimeError(f"Gemini API error {response.status}: {sanitized}")

                data = await response.json()

                # Extract text from response with robust error handling
                try:
                    candidate = data["candidates"][0]
                    finish_reason = candidate.get("finishReason", "UNKNOWN")

                    # Handle empty content (MAX_TOKENS, SAFETY, etc.)
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])
                    text = parts[0].get("text", "") if parts else ""

                    # Handle truncation: if we have partial text, use it with a warning
                    if finish_reason == "MAX_TOKENS" and text.strip():
                        # Got partial content - use it but log warning
                        logger.warning(f"Gemini response truncated at {len(text)} chars, using partial content")
                        return text

                    if not text.strip():
                        if finish_reason == "MAX_TOKENS":
                            raise RuntimeError(
                                f"Gemini response truncated (MAX_TOKENS): output limit reached with no content. "
                                f"Consider reducing prompt length or increasing maxOutputTokens."
                            )
                        elif finish_reason == "SAFETY":
                            raise RuntimeError(f"Gemini blocked response (SAFETY filter)")
                        else:
                            raise RuntimeError(
                                f"Gemini returned empty content (finishReason: {finish_reason})"
                            )

                    return text
                except (KeyError, IndexError) as e:
                    raise RuntimeError(f"Unexpected Gemini response format: {data}")

    async def generate_stream(self, prompt: str, context: list[Message] | None = None):
        """Stream tokens from Gemini API.

        Yields chunks of text as they arrive from the API.
        """
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        if self.system_prompt:
            full_prompt = f"System context: {self.system_prompt}\n\n{full_prompt}"

        # Use streamGenerateContent for streaming
        url = f"{self.base_url}/models/{self.model}:streamGenerateContent"

        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 65536,
            },
        }

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    sanitized = _sanitize_error_message(error_text)
                    raise RuntimeError(f"Gemini streaming API error {response.status}: {sanitized}")

                # Gemini streams as JSON array chunks
                buffer = b""
                try:
                    async for chunk in response.content.iter_any():
                        buffer += chunk
                        # Prevent unbounded buffer growth (DoS protection)
                        if len(buffer) > MAX_STREAM_BUFFER_SIZE:
                            raise RuntimeError("Streaming buffer exceeded maximum size")

                        # Try to parse complete JSON objects from buffer
                        # Gemini streams as a JSON array: [{...}, {...}, ...]
                        text = buffer.decode('utf-8', errors='ignore')

                        # Find complete candidate objects
                        # Max iterations guard to prevent infinite loop on malformed data
                        max_parse_iterations = 100
                        parse_iterations = 0
                        while parse_iterations < max_parse_iterations:
                            parse_iterations += 1
                            # Look for text content in the buffer
                            try:
                                # Parse as JSON array (Gemini format)
                                if text.strip().startswith('['):
                                    # Remove trailing incomplete parts
                                    bracket_count = 0
                                    last_complete = -1
                                    for i, c in enumerate(text):
                                        if c == '[':
                                            bracket_count += 1
                                        elif c == ']':
                                            bracket_count -= 1
                                            if bracket_count == 0:
                                                last_complete = i

                                    if last_complete > 0:
                                        complete_json = text[:last_complete + 1]
                                        data = json.loads(complete_json)

                                        # Extract text from all candidates
                                        for item in data:
                                            if 'candidates' in item:
                                                for candidate in item['candidates']:
                                                    content = candidate.get('content', {})
                                                    for part in content.get('parts', []):
                                                        if 'text' in part:
                                                            yield part['text']

                                        # Clear processed data from buffer
                                        buffer = text[last_complete + 1:].encode('utf-8')
                                        text = buffer.decode('utf-8', errors='ignore')
                                    else:
                                        break
                                else:
                                    break
                            except json.JSONDecodeError:
                                break
                except asyncio.TimeoutError:
                    logger.warning(f"[{self.name}] Streaming timeout")
                    raise
                except aiohttp.ClientError as e:
                    logger.warning(f"[{self.name}] Streaming connection error: {e}")
                    raise RuntimeError(f"Streaming connection error: {e}")

    async def critique(self, proposal: str, task: str, context: list[Message] | None = None) -> Critique:
        """Critique a proposal using Gemini."""
        critique_prompt = f"""You are a critical reviewer. Analyze this proposal for the given task.

Task: {task}

Proposal to critique:
{proposal}

Provide a structured critique with:
1. ISSUES: List specific problems, errors, or weaknesses (use bullet points)
2. SUGGESTIONS: List concrete improvements (use bullet points)
3. SEVERITY: Rate 0.0 (minor) to 1.0 (critical)
4. REASONING: Brief explanation of your assessment

Be constructive but thorough."""

        response = await self.generate(critique_prompt, context)
        return self._parse_critique(response, "proposal", proposal)


class OllamaAgent(APIAgent):
    """Agent that uses local Ollama API."""

    def __init__(
        self,
        name: str = "ollama",
        model: str = "llama3.2",
        role: str = "proposer",
        timeout: int = 180,
        base_url: str | None = None,
    ):
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            base_url=base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        )
        self.agent_type = "ollama"

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using Ollama API."""
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        if self.system_prompt:
            full_prompt = f"System context: {self.system_prompt}\n\n{full_prompt}"

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        sanitized = _sanitize_error_message(error_text)
                        raise RuntimeError(f"Ollama API error {response.status}: {sanitized}")

                    try:
                        data = await response.json()
                    except (json.JSONDecodeError, aiohttp.ContentTypeError) as e:
                        raise RuntimeError(f"Ollama returned invalid JSON: {e}")
                    return data.get("response", "")

            except aiohttp.ClientConnectorError:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Is Ollama running? Start with: ollama serve"
                )

    async def critique(self, proposal: str, task: str, context: list[Message] | None = None) -> Critique:
        """Critique a proposal using Ollama."""
        critique_prompt = f"""You are a critical reviewer. Analyze this proposal:

Task: {task}

Proposal:
{proposal}

Provide structured feedback:
ISSUES:
- issue 1
- issue 2

SUGGESTIONS:
- suggestion 1
- suggestion 2

SEVERITY: X.X (0.0 minor to 1.0 critical)
REASONING: explanation"""

        response = await self.generate(critique_prompt, context)
        return self._parse_critique(response, "proposal", proposal)


class AnthropicAPIAgent(APIAgent):
    """Agent that uses Anthropic API directly (without CLI).

    Supports automatic fallback to OpenRouter when Anthropic API returns
    billing/quota errors (e.g., "credit balance is too low").
    """

    # Model mapping from Anthropic to OpenRouter format
    OPENROUTER_MODEL_MAP = {
        "claude-opus-4-5-20251101": "anthropic/claude-sonnet-4",
        "claude-sonnet-4-20250514": "anthropic/claude-sonnet-4",
        "claude-3-5-sonnet-20241022": "anthropic/claude-3.5-sonnet",
        "claude-3-opus-20240229": "anthropic/claude-3-opus",
        "claude-3-sonnet-20240229": "anthropic/claude-3-sonnet",
        "claude-3-haiku-20240307": "anthropic/claude-3-haiku",
    }

    def __init__(
        self,
        name: str = "claude-api",
        model: str = "claude-opus-4-5-20251101",
        role: str = "proposer",
        timeout: int = 120,
        api_key: str | None = None,
        enable_fallback: bool = True,
    ):
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            api_key=api_key or get_api_key("ANTHROPIC_API_KEY"),
            base_url="https://api.anthropic.com/v1",
        )
        self.agent_type = "anthropic"
        self.enable_fallback = enable_fallback
        self._fallback_agent = None  # Lazy-loaded OpenRouter fallback

    def _get_fallback_agent(self):
        """Get or create the OpenRouter fallback agent for Claude models."""
        if self._fallback_agent is None:
            # Map the model to OpenRouter format
            openrouter_model = self.OPENROUTER_MODEL_MAP.get(
                self.model, "anthropic/claude-sonnet-4"
            )

            # OpenRouterAgent is defined in this module
            self._fallback_agent = OpenRouterAgent(
                name=f"{self.name}_fallback",
                model=openrouter_model,
                role=self.role,
                system_prompt=self.system_prompt,
                timeout=self.timeout,
            )
            logger.info(f"Created OpenRouter fallback agent with model {openrouter_model}")
        return self._fallback_agent

    def _is_anthropic_quota_error(self, status_code: int, error_text: str) -> bool:
        """Check if the error is a billing/quota/rate limit error from Anthropic."""
        # 429 is rate limit
        if status_code == 429:
            return True
        # Check for billing/credit-related messages in any error code
        quota_keywords = [
            "credit balance",
            "insufficient",
            "quota",
            "rate_limit",
            "billing",
            "exceeded",
            "purchase credits",
        ]
        error_lower = error_text.lower()
        return any(kw in error_lower for kw in quota_keywords)

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using Anthropic API.

        Falls back to OpenRouter if billing/quota errors are encountered
        and OPENROUTER_API_KEY is set.
        """
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        url = f"{self.base_url}/messages"

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": full_prompt}],
        }

        if self.system_prompt:
            payload["system"] = self.system_prompt

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    sanitized = _sanitize_error_message(error_text)

                    # Check if this is a quota/billing error and fallback is enabled
                    if self.enable_fallback and self._is_anthropic_quota_error(
                        response.status, error_text
                    ):
                        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
                        if openrouter_key:
                            logger.warning(
                                f"Anthropic API billing/quota error (status {response.status}), "
                                f"falling back to OpenRouter for {self.name}"
                            )
                            fallback = self._get_fallback_agent()
                            return await fallback.generate(prompt, context)
                        else:
                            logger.warning(
                                "Anthropic quota exceeded but OPENROUTER_API_KEY not set - cannot fallback"
                            )

                    raise RuntimeError(f"Anthropic API error {response.status}: {sanitized}")

                data = await response.json()

                try:
                    return data["content"][0]["text"]
                except (KeyError, IndexError):
                    raise RuntimeError(f"Unexpected Anthropic response format: {data}")

    async def generate_stream(self, prompt: str, context: list[Message] | None = None):
        """Stream tokens from Anthropic API.

        Yields chunks of text as they arrive from the API using SSE.
        """
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        url = f"{self.base_url}/messages"

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": full_prompt}],
            "stream": True,
        }

        if self.system_prompt:
            payload["system"] = self.system_prompt

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    sanitized = _sanitize_error_message(error_text)

                    # Check for quota/billing errors and fallback to OpenRouter
                    if self.enable_fallback and self._is_anthropic_quota_error(
                        response.status, error_text
                    ):
                        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
                        if openrouter_key:
                            logger.warning(
                                f"Anthropic API billing/quota error (status {response.status}), "
                                f"falling back to OpenRouter streaming for {self.name}"
                            )
                            fallback = self._get_fallback_agent()
                            async for chunk in fallback.generate_stream(prompt, context):
                                yield chunk
                            return
                        else:
                            logger.warning(
                                "Anthropic quota exceeded but OPENROUTER_API_KEY not set - cannot fallback"
                            )

                    raise RuntimeError(f"Anthropic streaming API error {response.status}: {sanitized}")

                # Anthropic uses SSE format: data: {...}\n\n
                buffer = ""
                try:
                    async for chunk in response.content.iter_any():
                        buffer += chunk.decode('utf-8', errors='ignore')
                        # Prevent unbounded buffer growth (DoS protection)
                        if len(buffer) > MAX_STREAM_BUFFER_SIZE:
                            raise RuntimeError("Streaming buffer exceeded maximum size")

                        # Process complete SSE lines
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()

                            if not line or not line.startswith('data: '):
                                continue

                            data_str = line[6:]  # Remove 'data: ' prefix

                            if data_str == '[DONE]':
                                return

                            try:
                                event = json.loads(data_str)
                                event_type = event.get('type', '')

                                # Handle content_block_delta events
                                if event_type == 'content_block_delta':
                                    delta = event.get('delta', {})
                                    if delta.get('type') == 'text_delta':
                                        text = delta.get('text', '')
                                        if text:
                                            yield text

                            except json.JSONDecodeError:
                                continue
                except asyncio.TimeoutError:
                    logger.warning(f"[{self.name}] Streaming timeout")
                    raise
                except aiohttp.ClientError as e:
                    logger.warning(f"[{self.name}] Streaming connection error: {e}")
                    raise RuntimeError(f"Streaming connection error: {e}")

    async def critique(self, proposal: str, task: str, context: list[Message] | None = None) -> Critique:
        """Critique a proposal using Anthropic API."""
        critique_prompt = f"""Analyze this proposal critically:

Task: {task}

Proposal:
{proposal}

Provide structured feedback:
- ISSUES: Specific problems (bullet points)
- SUGGESTIONS: Improvements (bullet points)
- SEVERITY: 0.0-1.0 rating
- REASONING: Brief explanation"""

        response = await self.generate(critique_prompt, context)
        return self._parse_critique(response, "proposal", proposal)


class OpenAIAPIAgent(APIAgent):
    """Agent that uses OpenAI API directly (without CLI).

    Includes automatic fallback to OpenRouter when OpenAI quota is exceeded (429 error).
    The fallback uses the same GPT model via OpenRouter's API.
    """

    # Model mapping from OpenAI to OpenRouter format
    OPENROUTER_MODEL_MAP = {
        "gpt-4o": "openai/gpt-4o",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "gpt-4-turbo": "openai/gpt-4-turbo",
        "gpt-4": "openai/gpt-4",
        "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
        "gpt-5.2": "openai/gpt-4o",  # Fallback to gpt-4o if gpt-5.2 not available
    }

    def __init__(
        self,
        name: str = "openai-api",
        model: str = "gpt-5.2",
        role: str = "proposer",
        timeout: int = 120,
        api_key: str | None = None,
        enable_fallback: bool = True,
    ):
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            api_key=api_key or get_api_key("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1",
        )
        self.agent_type = "openai"
        self.enable_fallback = enable_fallback
        self._fallback_agent = None  # Lazy-loaded OpenRouter fallback

    def _get_fallback_agent(self):
        """Get or create the OpenRouter fallback agent."""
        if self._fallback_agent is None:
            # Map the model to OpenRouter format
            openrouter_model = self.OPENROUTER_MODEL_MAP.get(self.model, "openai/gpt-4o")

            # OpenRouterAgent is defined in this module
            self._fallback_agent = OpenRouterAgent(
                name=f"{self.name}_fallback",
                model=openrouter_model,
                role=self.role,
                system_prompt=self.system_prompt,
                timeout=self.timeout,
            )
            logger.info(f"Created OpenRouter fallback agent with model {openrouter_model}")
        return self._fallback_agent

    def _is_quota_error(self, status_code: int, error_text: str) -> bool:
        """Check if the error is a quota/rate limit error."""
        if status_code == 429:
            return True
        # Also check for quota-related messages in other error codes
        quota_keywords = ["quota", "rate_limit", "insufficient_quota", "exceeded"]
        return any(kw in error_text.lower() for kw in quota_keywords)

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using OpenAI API."""
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = [{"role": "user", "content": full_prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    sanitized = _sanitize_error_message(error_text)

                    # Check if this is a quota error and fallback is enabled
                    if self.enable_fallback and self._is_quota_error(response.status, error_text):
                        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
                        if openrouter_key:
                            logger.warning(
                                f"OpenAI quota exceeded (status {response.status}), "
                                f"falling back to OpenRouter for {self.name}"
                            )
                            fallback = self._get_fallback_agent()
                            return await fallback.generate(prompt, context)
                        else:
                            logger.warning(
                                "OpenAI quota exceeded but OPENROUTER_API_KEY not set - cannot fallback"
                            )

                    raise RuntimeError(f"OpenAI API error {response.status}: {sanitized}")

                data = await response.json()

                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    raise RuntimeError(f"Unexpected OpenAI response format: {data}")

    async def generate_stream(self, prompt: str, context: list[Message] | None = None):
        """Stream tokens from OpenAI API.

        Yields chunks of text as they arrive from the API using SSE.
        """
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = [{"role": "user", "content": full_prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
            "stream": True,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    sanitized = _sanitize_error_message(error_text)

                    # Check if this is a quota error and fallback is enabled
                    if self.enable_fallback and self._is_quota_error(response.status, error_text):
                        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
                        if openrouter_key:
                            logger.warning(
                                f"OpenAI quota exceeded (status {response.status}), "
                                f"falling back to OpenRouter streaming for {self.name}"
                            )
                            fallback = self._get_fallback_agent()
                            # Yield from fallback's stream
                            async for token in fallback.generate_stream(prompt, context):
                                yield token
                            return
                        else:
                            logger.warning(
                                "OpenAI quota exceeded but OPENROUTER_API_KEY not set - cannot fallback"
                            )

                    raise RuntimeError(f"OpenAI streaming API error {response.status}: {sanitized}")

                # OpenAI uses SSE format: data: {...}\n\n
                buffer = ""
                try:
                    async for chunk in response.content.iter_any():
                        buffer += chunk.decode('utf-8', errors='ignore')
                        # Prevent unbounded buffer growth (DoS protection)
                        if len(buffer) > MAX_STREAM_BUFFER_SIZE:
                            raise RuntimeError("Streaming buffer exceeded maximum size")

                        # Process complete SSE lines
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()

                            if not line or not line.startswith('data: '):
                                continue

                            data_str = line[6:]  # Remove 'data: ' prefix

                            if data_str == '[DONE]':
                                return

                            try:
                                event = json.loads(data_str)
                                choices = event.get('choices', [])
                                if choices:
                                    delta = choices[0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield content

                            except json.JSONDecodeError:
                                continue
                except asyncio.TimeoutError:
                    logger.warning(f"[{self.name}] Streaming timeout")
                    raise
                except aiohttp.ClientError as e:
                    logger.warning(f"[{self.name}] Streaming connection error: {e}")
                    raise RuntimeError(f"Streaming connection error: {e}")

    async def critique(self, proposal: str, task: str, context: list[Message] | None = None) -> Critique:
        """Critique a proposal using OpenAI API."""
        critique_prompt = f"""Critically analyze this proposal:

Task: {task}
Proposal: {proposal}

Format your response as:
ISSUES:
- issue 1
- issue 2

SUGGESTIONS:
- suggestion 1
- suggestion 2

SEVERITY: X.X
REASONING: explanation"""

        response = await self.generate(critique_prompt, context)
        return self._parse_critique(response, "proposal", proposal)


class GrokAgent(APIAgent):
    """Agent that uses xAI's Grok API (OpenAI-compatible).

    Uses the xAI API at https://api.x.ai/v1 with models like grok-3.
    """

    def __init__(
        self,
        name: str = "grok",
        model: str = "grok-4",
        role: str = "proposer",
        timeout: int = 120,
        api_key: str | None = None,
    ):
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            api_key=api_key or get_api_key("XAI_API_KEY", "GROK_API_KEY"),
            base_url="https://api.x.ai/v1",
        )
        self.agent_type = "grok"

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using Grok API."""

        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = [{"role": "user", "content": full_prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    sanitized = _sanitize_error_message(error_text)
                    raise RuntimeError(f"Grok API error {response.status}: {sanitized}")

                data = await response.json()

                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    raise RuntimeError(f"Unexpected Grok response format: {data}")

    async def generate_stream(self, prompt: str, context: list[Message] | None = None):
        """Stream tokens from Grok API."""
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = [{"role": "user", "content": full_prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
            "stream": True,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    sanitized = _sanitize_error_message(error_text)
                    raise RuntimeError(f"Grok streaming API error {response.status}: {sanitized}")

                try:
                    buffer = ""
                    async for chunk in response.content.iter_any():
                        buffer += chunk.decode('utf-8', errors='ignore')

                        # Prevent unbounded buffer growth (DoS protection)
                        if len(buffer) > MAX_STREAM_BUFFER_SIZE:
                            raise RuntimeError("Streaming buffer exceeded maximum size")

                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()

                            if not line or not line.startswith('data: '):
                                continue

                            data_str = line[6:]

                            if data_str == '[DONE]':
                                return

                            try:
                                event = json.loads(data_str)
                                choices = event.get('choices', [])
                                if choices:
                                    delta = choices[0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield content

                            except json.JSONDecodeError:
                                continue
                except asyncio.TimeoutError:
                    logger.warning(f"[{self.name}] Streaming timeout")
                    raise
                except aiohttp.ClientError as e:
                    logger.warning(f"[{self.name}] Streaming connection error: {e}")
                    raise RuntimeError(f"Streaming connection error: {e}")

    async def critique(self, proposal: str, task: str, context: list[Message] | None = None) -> Critique:
        """Critique a proposal using Grok API."""
        critique_prompt = f"""Critically analyze this proposal:

Task: {task}
Proposal: {proposal}

Format your response as:
ISSUES:
- issue 1
- issue 2

SUGGESTIONS:
- suggestion 1
- suggestion 2

SEVERITY: X.X
REASONING: explanation"""

        response = await self.generate(critique_prompt, context)
        return self._parse_critique(response, "proposal", proposal)


class OpenRouterAgent(APIAgent):
    """Agent that uses OpenRouter API for access to many models.

    OpenRouter provides unified access to models like DeepSeek, Llama, Mistral,
    and others through an OpenAI-compatible API.

    Supported models (via model parameter):
    - deepseek/deepseek-chat (DeepSeek V3)
    - deepseek/deepseek-reasoner (DeepSeek R1)
    - meta-llama/llama-3.3-70b-instruct
    - mistralai/mistral-large-2411
    - google/gemini-2.0-flash-exp:free
    - anthropic/claude-3.5-sonnet
    - openai/gpt-4o
    """

    def __init__(
        self,
        name: str = "openrouter",
        role: str = "analyst",
        model: str = "deepseek/deepseek-chat",
        system_prompt: str | None = None,
        timeout: int = 300,
    ):
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            api_key=get_api_key("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        self.agent_type = "openrouter"
        if system_prompt:
            self.system_prompt = system_prompt

    def _build_context_prompt(self, context: list[Message]) -> str:
        """Build context prompt from message history."""
        if not context:
            return ""
        prompt = "Previous discussion:\n"
        for msg in context[-5:]:
            prompt += f"- {msg.agent} ({msg.role}): {msg.content[:500]}...\n"
        return prompt + "\n"

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using OpenRouter API with rate limiting and retry."""
        max_retries = 3
        base_delay = 30  # Start with 30s backoff for rate limits

        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://aragora.ai",
            "X-Title": "Aragora Multi-Agent Debate",
        }

        messages = [{"role": "user", "content": full_prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
        }

        last_error = None
        for attempt in range(max_retries):
            # Acquire rate limit token for each attempt
            limiter = get_openrouter_limiter()
            if not await limiter.acquire(timeout=DB_TIMEOUT_SECONDS):
                raise RuntimeError("OpenRouter rate limit exceeded, request timed out")

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as response:
                        # Update rate limit state from headers
                        limiter.update_from_headers(dict(response.headers))

                        if response.status == 429:
                            # Rate limited by API - release token and calculate wait time
                            limiter.release_on_error()
                            retry_after_header = response.headers.get("Retry-After")
                            if retry_after_header:
                                try:
                                    wait_time = float(retry_after_header)
                                except ValueError:
                                    wait_time = base_delay * (2 ** attempt)
                            else:
                                wait_time = base_delay * (2 ** attempt)
                            wait_time = min(wait_time, 300)  # Cap at 5 minutes

                            if attempt < max_retries - 1:
                                logger.warning(
                                    f"OpenRouter rate limited (429), waiting {wait_time:.0f}s before retry {attempt + 2}/{max_retries}"
                                )
                                await asyncio.sleep(wait_time)
                                last_error = f"Rate limited (429)"
                                continue
                            else:
                                raise RuntimeError(f"OpenRouter rate limited (429) after {max_retries} retries")

                        if response.status != 200:
                            error_text = await response.text()
                            sanitized = _sanitize_error_message(error_text)
                            raise RuntimeError(f"OpenRouter API error {response.status}: {sanitized}")

                        data = await response.json()
                        try:
                            return data["choices"][0]["message"]["content"]
                        except (KeyError, IndexError):
                            raise RuntimeError(f"Unexpected OpenRouter response format: {data}")

            except aiohttp.ClientError as e:
                limiter.release_on_error()
                last_error = str(e)
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    logger.warning(f"OpenRouter connection error, waiting {wait_time:.0f}s before retry: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                raise RuntimeError(f"OpenRouter connection failed after {max_retries} retries: {last_error}")

    async def generate_stream(self, prompt: str, context: list[Message] | None = None):
        """Stream tokens from OpenRouter API with rate limiting and retry.

        Yields chunks of text as they arrive from the API using SSE.
        Implements retry logic with exponential backoff for 429 rate limit errors.
        """
        max_retries = 3
        base_delay = 2.0

        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://aragora.ai",
            "X-Title": "Aragora Multi-Agent Debate",
        }

        messages = [{"role": "user", "content": full_prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
            "stream": True,
        }

        last_error = None
        for attempt in range(max_retries):
            # Acquire rate limit token for each attempt
            limiter = get_openrouter_limiter()
            if not await limiter.acquire(timeout=DB_TIMEOUT_SECONDS):
                raise RuntimeError("OpenRouter rate limit exceeded, request timed out")

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as response:
                        # Update rate limit state from headers
                        limiter.update_from_headers(dict(response.headers))

                        if response.status == 429:
                            # Rate limited by API - release token and calculate wait time
                            limiter.release_on_error()
                            retry_after_header = response.headers.get("Retry-After")
                            if retry_after_header:
                                try:
                                    wait_time = float(retry_after_header)
                                except ValueError:
                                    wait_time = base_delay * (2 ** attempt)
                            else:
                                wait_time = base_delay * (2 ** attempt)
                            wait_time = min(wait_time, 300)  # Cap at 5 minutes

                            if attempt < max_retries - 1:
                                logger.warning(
                                    f"OpenRouter streaming rate limited (429), waiting {wait_time:.0f}s before retry {attempt + 2}/{max_retries}"
                                )
                                await asyncio.sleep(wait_time)
                                last_error = f"Rate limited (429)"
                                continue
                            else:
                                raise RuntimeError(f"OpenRouter streaming rate limited (429) after {max_retries} retries")

                        if response.status != 200:
                            error_text = await response.text()
                            sanitized = _sanitize_error_message(error_text)
                            raise RuntimeError(f"OpenRouter streaming API error {response.status}: {sanitized}")

                        # OpenRouter uses SSE format (OpenAI-compatible)
                        buffer = ""
                        async for chunk in response.content.iter_any():
                            buffer += chunk.decode('utf-8', errors='ignore')

                            while '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
                                line = line.strip()

                                if not line or not line.startswith('data: '):
                                    continue

                                data_str = line[6:]

                                if data_str == '[DONE]':
                                    return

                                try:
                                    event = json.loads(data_str)
                                    choices = event.get('choices', [])
                                    if choices:
                                        delta = choices[0].get('delta', {})
                                        content = delta.get('content', '')
                                        if content:
                                            yield content

                                except json.JSONDecodeError:
                                    continue

                        # Successfully streamed - exit retry loop
                        return

            except aiohttp.ClientError as e:
                limiter.release_on_error()
                last_error = str(e)
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    logger.warning(f"OpenRouter streaming connection error, waiting {wait_time:.0f}s before retry: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                raise RuntimeError(f"OpenRouter streaming failed after {max_retries} retries: {last_error}")

    async def critique(self, proposal: str, task: str, context: list[Message] | None = None) -> Critique:
        """Critique a proposal using OpenRouter API."""
        critique_prompt = f"""Critically analyze this proposal:

Task: {task}
Proposal: {proposal}

Format your response as:
ISSUES:
- issue 1
- issue 2

SUGGESTIONS:
- suggestion 1
- suggestion 2

SEVERITY: X.X
REASONING: explanation"""

        response = await self.generate(critique_prompt, context)
        return self._parse_critique(response, "proposal", proposal)


# Convenience aliases for specific OpenRouter models
class DeepSeekAgent(OpenRouterAgent):
    """DeepSeek V3.2 via OpenRouter - latest model with integrated thinking + tool-use."""

    def __init__(self, name: str = "deepseek", role: str = "analyst", system_prompt: str | None = None):
        super().__init__(
            name=name,
            role=role,
            model="deepseek/deepseek-v3.2",  # V3.2 latest
            system_prompt=system_prompt,
        )
        self.agent_type = "deepseek"


class DeepSeekReasonerAgent(OpenRouterAgent):
    """DeepSeek R1 via OpenRouter - reasoning model with chain-of-thought."""

    def __init__(self, name: str = "deepseek-r1", role: str = "analyst", system_prompt: str | None = None):
        super().__init__(
            name=name,
            role=role,
            model="deepseek/deepseek-reasoner",  # R1 reasoning model
            system_prompt=system_prompt,
        )
        self.agent_type = "deepseek-r1"


class DeepSeekV3Agent(OpenRouterAgent):
    """DeepSeek V3.2 via OpenRouter - integrated thinking + tool-use, GPT-5 class reasoning."""

    def __init__(self, name: str = "deepseek-v3", role: str = "analyst", system_prompt: str | None = None):
        super().__init__(
            name=name,
            role=role,
            model="deepseek/deepseek-v3.2",  # V3.2 with integrated thinking + tool-use
            system_prompt=system_prompt,
        )
        self.agent_type = "deepseek-v3"


class LlamaAgent(OpenRouterAgent):
    """Llama 3.3 70B via OpenRouter."""

    def __init__(self, name: str = "llama", role: str = "analyst", system_prompt: str | None = None):
        super().__init__(
            name=name,
            role=role,
            model="meta-llama/llama-3.3-70b-instruct",
            system_prompt=system_prompt,
        )
        self.agent_type = "llama"


class MistralAgent(OpenRouterAgent):
    """Mistral Large via OpenRouter."""

    def __init__(self, name: str = "mistral", role: str = "analyst", system_prompt: str | None = None):
        super().__init__(
            name=name,
            role=role,
            model="mistralai/mistral-large-2411",
            system_prompt=system_prompt,
        )
        self.agent_type = "mistral"
