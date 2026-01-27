"""
Anthropic API agent with OpenRouter fallback support.

Supports web search tool for web-capable responses when URLs
or web-related keywords are detected in the prompt.
"""

import logging
import re
from typing import AsyncGenerator

from aragora.agents.api_agents.base import APIAgent
from aragora.core_types import AgentRole
from aragora.agents.api_agents.common import (
    AgentAPIError,
    AgentCircuitOpenError,
    AgentConnectionError,
    AgentRateLimitError,
    AgentStreamError,
    AgentTimeoutError,
    Critique,
    Message,
    _sanitize_error_message,
    create_anthropic_sse_parser,
    create_client_session,
    get_primary_api_key,
    get_trace_headers,
    handle_agent_errors,
)
from aragora.agents.fallback import QuotaFallbackMixin
from aragora.agents.registry import AgentRegistry

logger = logging.getLogger(__name__)

# Patterns that indicate web search would be helpful
WEB_SEARCH_INDICATORS = [
    r"https?://",  # URLs
    r"github\.com",  # GitHub repos
    r"\brepo\b",  # Repository mentions
    r"\bwebsite\b",  # Website mentions
    r"\bweb\s*page\b",  # Web page mentions
    r"\bonline\b",  # Online content
    r"\blatest\b",  # Latest information (might need fresh data)
    r"\bcurrent\b",  # Current information
    r"\brecent\b",  # Recent information
    r"\bnews\b",  # News
    r"\barticle\b",  # Articles
]


@AgentRegistry.register(
    "anthropic-api",
    default_model="claude-opus-4-5-20251101",
    default_name="claude-api",
    agent_type="API",
    env_vars="ANTHROPIC_API_KEY",
    accepts_api_key=True,
)
class AnthropicAPIAgent(QuotaFallbackMixin, APIAgent):
    """Agent that uses Anthropic API directly (without CLI).

    Supports automatic fallback to OpenRouter when Anthropic API returns
    billing/quota errors (e.g., "credit balance is too low").

    Uses QuotaFallbackMixin for shared quota detection and fallback logic.
    """

    # Model mapping from Anthropic to OpenRouter format (used by QuotaFallbackMixin)
    OPENROUTER_MODEL_MAP = {
        "claude-opus-4-5-20251101": "anthropic/claude-sonnet-4",
        "claude-sonnet-4-20250514": "anthropic/claude-sonnet-4",
        "claude-3-5-sonnet-20241022": "anthropic/claude-3.5-sonnet",
        "claude-3-opus-20240229": "anthropic/claude-3-opus",
        "claude-3-sonnet-20240229": "anthropic/claude-3-sonnet",
        "claude-3-haiku-20240307": "anthropic/claude-3-haiku",
    }
    DEFAULT_FALLBACK_MODEL = "anthropic/claude-sonnet-4"

    def __init__(
        self,
        name: str = "claude-api",
        model: str = "claude-opus-4-5-20251101",
        role: AgentRole = "proposer",
        timeout: int = 120,
        api_key: str | None = None,
        enable_fallback: bool | None = None,  # None = use config setting
    ) -> None:
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            api_key=api_key
            or get_primary_api_key("ANTHROPIC_API_KEY", allow_openrouter_fallback=True),
            base_url="https://api.anthropic.com/v1",
        )
        self.agent_type = "anthropic"
        # Use config setting if not explicitly provided
        if enable_fallback is None:
            from aragora.agents.fallback import get_default_fallback_enabled

            self.enable_fallback = get_default_fallback_enabled()
        else:
            self.enable_fallback = enable_fallback
        self._fallback_agent = None  # Cached by QuotaFallbackMixin
        self.enable_web_search = True  # Enable web search tool by default

    def _needs_web_search(self, prompt: str) -> bool:
        """Detect if the prompt would benefit from web search.

        Returns True if the prompt contains URLs, GitHub references,
        or keywords indicating need for current/web information.
        """
        if not self.enable_web_search:
            return False

        for pattern in WEB_SEARCH_INDICATORS:
            if re.search(pattern, prompt, re.IGNORECASE):
                return True
        return False

    @handle_agent_errors(
        max_retries=3,
        retry_delay=1.0,
        retry_backoff=2.0,
        retryable_exceptions=(AgentRateLimitError, AgentConnectionError, AgentTimeoutError),
    )
    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using Anthropic API.

        Falls back to OpenRouter if billing/quota errors are encountered
        and OPENROUTER_API_KEY is set.

        Includes circuit breaker protection to prevent cascading failures.
        """
        # Check circuit breaker before attempting API call
        if self._circuit_breaker is not None and not self._circuit_breaker.can_proceed():
            raise AgentCircuitOpenError(
                f"Circuit breaker open for {self.name} - too many recent failures",
                agent_name=self.name,
            )

        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        url = f"{self.base_url}/messages"

        # Check if web search is needed
        use_web_search = self._needs_web_search(full_prompt)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            **get_trace_headers(),  # Distributed tracing
        }

        # Add beta header for web search if enabled
        if use_web_search:
            logger.info(f"[{self.name}] Enabling web search tool for web content")
            headers["anthropic-beta"] = "web-search-2025-03-05"

        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": full_prompt}],
        }

        # Add web search tool if enabled
        if use_web_search:
            payload["tools"] = [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                }
            ]

        # Apply generation parameters from persona if set
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p

        if self.system_prompt:
            payload["system"] = self.system_prompt

        try:
            async with create_client_session(timeout=self.timeout) as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        sanitized = _sanitize_error_message(error_text)

                        # Record failure for circuit breaker (non-quota errors)
                        if self._circuit_breaker is not None and not self.is_quota_error(
                            response.status, error_text
                        ):
                            self._circuit_breaker.record_failure()

                        # Check if this is a quota/billing error and fallback is enabled
                        if self.is_quota_error(response.status, error_text):
                            result = await self.fallback_generate(prompt, context, response.status)
                            if result is not None:
                                return result

                        raise AgentAPIError(
                            f"Anthropic API error {response.status}: {sanitized}",
                            agent_name=self.name,
                            status_code=response.status,
                        )

                    data = await response.json()

                    # Record token usage for billing
                    usage = data.get("usage", {})
                    self._record_token_usage(
                        tokens_in=usage.get("input_tokens", 0),
                        tokens_out=usage.get("output_tokens", 0),
                    )

                    try:
                        # Extract text from response content blocks
                        # May include multiple blocks: text, web_search_tool_result, etc.
                        content_blocks = data.get("content", [])
                        text_parts = []
                        for block in content_blocks:
                            if block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif block.get("type") == "web_search_tool_result":
                                # Include web search citations in response
                                search_results = block.get("content", [])
                                for result in search_results:
                                    if result.get("type") == "web_search_result":
                                        # Format as a citation
                                        title = result.get("title", "")
                                        url = result.get("url", "")
                                        if title and url:
                                            text_parts.append(f"\n[Source: {title}]({url})")

                        if text_parts:
                            output = "\n".join(text_parts)
                        else:
                            # Fallback to old format
                            output = data["content"][0]["text"]

                        if not output or not output.strip():
                            if self._circuit_breaker is not None:
                                self._circuit_breaker.record_failure()
                            raise AgentAPIError(
                                "Anthropic returned empty content",
                                agent_name=self.name,
                            )

                        # Record success for circuit breaker
                        if self._circuit_breaker is not None:
                            self._circuit_breaker.record_success()

                        return output
                    except (KeyError, IndexError):
                        if self._circuit_breaker is not None:
                            self._circuit_breaker.record_failure()
                        raise AgentAPIError(
                            f"Unexpected Anthropic response format: {data}",
                            agent_name=self.name,
                        )
        except (AgentAPIError, AgentCircuitOpenError):
            raise  # Re-raise without double-recording
        except Exception:
            # Record failure for unexpected errors
            if self._circuit_breaker is not None:
                self._circuit_breaker.record_failure()
            raise

    async def generate_stream(
        self, prompt: str, context: list[Message] | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from Anthropic API.

        Yields chunks of text as they arrive from the API using SSE.
        """
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        url = f"{self.base_url}/messages"

        # Check if web search is needed
        use_web_search = self._needs_web_search(full_prompt)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            **get_trace_headers(),  # Distributed tracing
        }

        # Add beta header for web search if enabled
        if use_web_search:
            logger.info(f"[{self.name}] Enabling web search tool for streaming")
            headers["anthropic-beta"] = "web-search-2025-03-05"

        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": full_prompt}],
            "stream": True,
        }

        # Add web search tool if enabled
        if use_web_search:
            payload["tools"] = [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                }
            ]

        # Apply generation parameters from persona if set
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p

        if self.system_prompt:
            payload["system"] = self.system_prompt

        async with create_client_session(timeout=self.timeout) as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    sanitized = _sanitize_error_message(error_text)

                    # Check for quota/billing errors and fallback to OpenRouter
                    if self.is_quota_error(response.status, error_text):
                        async for chunk in self.fallback_generate_stream(
                            prompt, context, response.status
                        ):
                            yield chunk
                        return

                    raise AgentStreamError(
                        f"Anthropic streaming API error {response.status}: {sanitized}",
                        agent_name=self.name,
                    )

                # Use SSEStreamParser for consistent SSE parsing
                try:
                    parser = create_anthropic_sse_parser()
                    async for content in parser.parse_stream(response.content, self.name):
                        yield content
                except RuntimeError as e:
                    raise AgentStreamError(str(e), agent_name=self.name)

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list[Message] | None = None,
        target_agent: str | None = None,
    ) -> Critique:
        """Critique a proposal using Anthropic API."""
        target_desc = f"from {target_agent}" if target_agent else ""
        critique_prompt = f"""Analyze this proposal {target_desc} critically:

Task: {task}

Proposal:
{proposal}

Provide structured feedback:
- ISSUES: Specific problems (bullet points)
- SUGGESTIONS: Improvements (bullet points)
- SEVERITY: 0-10 rating (0=trivial, 10=critical)
- REASONING: Brief explanation"""

        response = await self.generate(critique_prompt, context)
        return self._parse_critique(response, target_agent or "proposal", proposal)


__all__ = ["AnthropicAPIAgent"]
