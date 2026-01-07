"""
Gemini agent for Google Generative AI API.
"""

import aiohttp
import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING, AsyncGenerator

from aragora.agents.api_agents.base import APIAgent

if TYPE_CHECKING:
    from aragora.agents.api_agents.openrouter import OpenRouterAgent
from aragora.agents.api_agents.common import (
    Message,
    Critique,
    handle_agent_errors,
    AgentRateLimitError,
    AgentConnectionError,
    AgentTimeoutError,
    get_api_key,
    _sanitize_error_message,
    MAX_STREAM_BUFFER_SIZE,
)
from aragora.agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


@AgentRegistry.register(
    "gemini",
    default_model="gemini-3-pro-preview",
    agent_type="API",
    env_vars="GEMINI_API_KEY or GOOGLE_API_KEY",
    accepts_api_key=True,
)
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

    def _get_fallback_agent(self) -> "OpenRouterAgent":
        """Get or create the OpenRouter fallback agent for Gemini models."""
        if self._fallback_agent is None:
            # Lazy import to avoid circular dependency
            from aragora.agents.api_agents.openrouter import OpenRouterAgent

            # Map the model to OpenRouter format
            openrouter_model = self.OPENROUTER_MODEL_MAP.get(
                self.model, "google/gemini-2.0-flash-001"
            )

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

    @handle_agent_errors(
        max_retries=3,
        retry_delay=1.0,
        retry_backoff=2.0,
        retryable_exceptions=(AgentRateLimitError, AgentConnectionError, AgentTimeoutError),
    )
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

    async def generate_stream(self, prompt: str, context: list[Message] | None = None) -> AsyncGenerator[str, None]:
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


__all__ = ["GeminiAgent"]
