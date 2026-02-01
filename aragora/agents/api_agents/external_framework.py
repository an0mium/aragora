"""
External Framework Agent for proxying to external agent frameworks.

Supports proxying requests to external agent frameworks like OpenClaw,
LangChain servers, AutoGPT endpoints, and other OpenAI-compatible
agent framework servers.

Features:
- HTTP proxy with configurable timeout and retries
- Circuit breaker for failure handling
- Response sanitization
- Flexible endpoint configuration
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.gateway.credential_proxy import CredentialProxy

import aiohttp

from aragora.agents.api_agents.base import APIAgent
from aragora.agents.api_agents.common import (
    AgentAPIError,
    AgentConnectionError,
    AgentRateLimitError,
    AgentTimeoutError,
    Critique,
    Message,
    _sanitize_error_message,
    handle_agent_errors,
)
from aragora.agents.registry import AgentRegistry
from aragora.core_types import AgentRole
from aragora.security.ssrf_protection import validate_url, SSRFValidationError

logger = logging.getLogger(__name__)


@dataclass
class ExternalFrameworkConfig:
    """Configuration for external framework agent.

    Attributes:
        base_url: Base URL of the external framework server.
        generate_endpoint: Endpoint for generation requests (default: /generate).
        critique_endpoint: Endpoint for critique requests (default: /critique).
        vote_endpoint: Endpoint for vote requests (default: /vote).
        health_endpoint: Endpoint for health checks (default: /health).
        api_key_header: Header name for API key (default: Authorization).
        api_key_prefix: Prefix for API key value (default: Bearer).
        extra_headers: Additional headers to include in requests.
        timeout: Request timeout in seconds (default: 120).
        max_retries: Maximum number of retries on failure (default: 3).
        retry_delay: Base delay between retries in seconds (default: 1.0).
        retry_backoff: Backoff multiplier for retries (default: 2.0).
        enable_response_sanitization: Sanitize responses (default: True).
        max_response_length: Maximum response length in characters (default: 100000).
    """

    base_url: str
    generate_endpoint: str = "/generate"
    critique_endpoint: str = "/critique"
    vote_endpoint: str = "/vote"
    health_endpoint: str = "/health"
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer"
    extra_headers: dict[str, str] = field(default_factory=dict)
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    enable_response_sanitization: bool = True
    max_response_length: int = 100000


@AgentRegistry.register(
    "external-framework",
    default_model="external",
    default_name="external-framework",
    agent_type="API",
    requires="External agent framework server (e.g., OpenClaw, LangChain serve)",
    env_vars="EXTERNAL_FRAMEWORK_URL, EXTERNAL_FRAMEWORK_API_KEY (optional)",
    description="Proxy to external agent frameworks",
    accepts_api_key=True,
)
class ExternalFrameworkAgent(APIAgent):
    """Agent that proxies requests to external agent frameworks.

    Supports any OpenAI-compatible or custom agent framework server.
    Provides circuit breaker protection, retry logic, and response sanitization.

    Example:
        >>> config = ExternalFrameworkConfig(
        ...     base_url="http://localhost:8000",
        ...     generate_endpoint="/v1/chat/completions",
        ... )
        >>> agent = ExternalFrameworkAgent(config=config, api_key="your-key")
        >>> response = await agent.generate("Hello, world!")
    """

    def __init__(
        self,
        name: str = "external-framework",
        model: str = "external",
        role: AgentRole = "proposer",
        timeout: int = 120,
        api_key: str | None = None,
        base_url: str | None = None,
        config: ExternalFrameworkConfig | None = None,
        # Circuit breaker configuration
        enable_circuit_breaker: bool = True,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_cooldown: float = 60.0,
        # Credential proxy integration
        credential_proxy: "CredentialProxy | None" = None,
        credential_id: str | None = None,
    ) -> None:
        """Initialize external framework agent.

        Args:
            name: Agent instance name.
            model: Model identifier (passed to external framework).
            role: Agent role (proposer, critic, synthesizer).
            timeout: Request timeout in seconds.
            api_key: API key for authentication.
            base_url: Base URL (overrides config.base_url if provided).
            config: Full configuration object.
            enable_circuit_breaker: Enable circuit breaker protection.
            circuit_breaker_threshold: Failures before circuit opens.
            circuit_breaker_cooldown: Seconds before circuit resets.
            credential_proxy: Optional CredentialProxy for secure credential mediation.
            credential_id: Credential ID to resolve via the proxy.
        """
        # Build config from parameters or use provided config
        if config is None:
            import os

            config = ExternalFrameworkConfig(
                base_url=base_url
                or os.environ.get("EXTERNAL_FRAMEWORK_URL", "http://localhost:8000"),
                timeout=timeout,
            )
        elif base_url is not None:
            # Override base_url in config
            config = ExternalFrameworkConfig(
                base_url=base_url,
                generate_endpoint=config.generate_endpoint,
                critique_endpoint=config.critique_endpoint,
                vote_endpoint=config.vote_endpoint,
                health_endpoint=config.health_endpoint,
                api_key_header=config.api_key_header,
                api_key_prefix=config.api_key_prefix,
                extra_headers=config.extra_headers,
                timeout=config.timeout,
                max_retries=config.max_retries,
                retry_delay=config.retry_delay,
                retry_backoff=config.retry_backoff,
                enable_response_sanitization=config.enable_response_sanitization,
                max_response_length=config.max_response_length,
            )

        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=config.timeout,
            api_key=api_key or self._get_api_key_from_env(),
            base_url=config.base_url,
            enable_circuit_breaker=enable_circuit_breaker,
            circuit_breaker_threshold=circuit_breaker_threshold,
            circuit_breaker_cooldown=circuit_breaker_cooldown,
        )

        self.config = config
        self.agent_type = "external-framework"
        self._credential_proxy = credential_proxy
        self._credential_id = credential_id
        self._session: aiohttp.ClientSession | None = None

        # Validate base URL against SSRF attacks
        self._validate_endpoint_url(config.base_url)

    @staticmethod
    def _get_api_key_from_env() -> str | None:
        """Get API key from environment variable."""
        import os

        return os.environ.get("EXTERNAL_FRAMEWORK_API_KEY")

    def _validate_endpoint_url(self, url: str) -> None:
        """Validate URL is safe from SSRF attacks."""
        import os

        allowed_domains_str = os.environ.get("ARAGORA_GATEWAY_ALLOWED_DOMAINS", "")
        allowed_domains = (
            set(d.strip() for d in allowed_domains_str.split(",") if d.strip()) or None
        )

        result = validate_url(url, allowed_domains=allowed_domains)
        if not result.is_safe:
            raise SSRFValidationError(
                f"Unsafe external framework URL blocked: {result.error}",
                url=url,
            )

    def _resolve_api_key(self) -> str | None:
        """Resolve API key from credential proxy or direct configuration.

        Priority:
        1. Credential proxy (if configured)
        2. Direct api_key parameter
        3. Environment variable
        """
        if self._credential_proxy and self._credential_id:
            try:
                cred = self._credential_proxy.get_credential(self._credential_id)
                if cred and not cred.is_expired:
                    return cred.api_key
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to resolve credential via proxy: {e}")
        return self.api_key

    def _build_headers(self) -> dict[str, str]:
        """Build request headers including authentication."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Add API key if configured (resolved via proxy or direct)
        api_key = self._resolve_api_key()
        if api_key:
            prefix = self.config.api_key_prefix
            if prefix:
                headers[self.config.api_key_header] = f"{prefix} {api_key}"
            else:
                headers[self.config.api_key_header] = api_key

        # Add extra headers
        headers.update(self.config.extra_headers)

        return headers

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a reusable HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=float(self.timeout)),
                headers=self._build_headers(),
            )
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _sanitize_response(self, text: str) -> str:
        """Sanitize response text.

        Removes potentially harmful content and enforces length limits.

        Args:
            text: Raw response text.

        Returns:
            Sanitized response text.
        """
        if not self.config.enable_response_sanitization:
            return text

        # Enforce length limit
        if len(text) > self.config.max_response_length:
            text = text[: self.config.max_response_length] + "... [truncated]"

        # Remove null bytes
        text = text.replace("\x00", "")

        # Remove ANSI escape sequences
        ansi_pattern = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
        text = ansi_pattern.sub("", text)

        return text

    async def is_available(self) -> bool:
        """Check if external framework server is accessible.

        Returns:
            True if server responds to health check, False otherwise.
        """
        url = f"{self.base_url}{self.config.health_endpoint}"

        try:
            self._validate_endpoint_url(url)
        except SSRFValidationError:
            return False

        try:
            session = await self._get_session()
            async with session.get(url, headers=self._build_headers()) as response:
                return response.status in (200, 204)
        except (aiohttp.ClientError, OSError, TimeoutError):
            return False

    @handle_agent_errors(
        max_retries=3,
        retry_delay=1.0,
        retry_backoff=2.0,
        retryable_exceptions=(AgentRateLimitError, AgentConnectionError, AgentTimeoutError),
    )
    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response from the external framework.

        Args:
            prompt: The prompt to generate from.
            context: Optional conversation context.

        Returns:
            Generated response text.

        Raises:
            AgentAPIError: On API errors.
            AgentConnectionError: On connection failures.
            AgentTimeoutError: On timeout.
        """
        # Check circuit breaker
        if self.is_circuit_open():
            raise AgentConnectionError(
                f"Circuit breaker open for {self.name}",
                agent_name=self.name,
            )

        # Build full prompt with context
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        if self.system_prompt:
            full_prompt = f"System: {self.system_prompt}\n\n{full_prompt}"

        url = f"{self.base_url}{self.config.generate_endpoint}"

        # Validate URL against SSRF attacks
        self._validate_endpoint_url(url)

        # Build request payload (OpenAI-compatible format)
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "messages": [{"role": "user", "content": full_prompt}],
        }

        # Add generation parameters if set
        gen_params = self.get_generation_params()
        if gen_params:
            payload.update(gen_params)

        session = await self._get_session()
        try:
            async with session.post(url, json=payload, headers=self._build_headers()) as response:
                if response.status == 429:
                    error_text = await response.text()
                    raise AgentRateLimitError(
                        f"Rate limited by external framework: {_sanitize_error_message(error_text)}",
                        agent_name=self.name,
                        retry_after=self._parse_retry_after(response),
                    )

                if response.status != 200:
                    error_text = await response.text()
                    sanitized = _sanitize_error_message(error_text)
                    if self._circuit_breaker:
                        self._circuit_breaker.record_failure()
                    raise AgentAPIError(
                        f"External framework API error {response.status}: {sanitized}",
                        agent_name=self.name,
                        status_code=response.status,
                    )

                try:
                    data = await response.json()
                except (json.JSONDecodeError, aiohttp.ContentTypeError) as e:
                    raise AgentAPIError(
                        f"External framework returned invalid JSON: {e}",
                        agent_name=self.name,
                    )

                # Record success for circuit breaker
                if self._circuit_breaker:
                    self._circuit_breaker.record_success()

                # Extract response (support multiple formats)
                text = self._extract_response_text(data)
                return self._sanitize_response(text)

        except aiohttp.ClientConnectorError as e:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()
            raise AgentConnectionError(
                f"Cannot connect to external framework at {self.base_url}: {e}",
                agent_name=self.name,
                cause=e,
            )
        except TimeoutError as e:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()
            raise AgentTimeoutError(
                f"External framework request timed out after {self.timeout}s",
                agent_name=self.name,
                cause=e,
            )

    def _extract_response_text(self, data: dict[str, Any]) -> str:
        """Extract response text from various response formats.

        Supports:
        - OpenAI format: {"choices": [{"message": {"content": "..."}}]}
        - Simple format: {"response": "..."}
        - Text format: {"text": "..."}
        - Content format: {"content": "..."}
        - Output format: {"output": "..."}

        Args:
            data: Response JSON data.

        Returns:
            Extracted text content.
        """
        # OpenAI chat format
        if "choices" in data:
            choices = data["choices"]
            if choices and isinstance(choices, list):
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    # Chat completion format
                    message = first_choice.get("message", {})
                    if isinstance(message, dict) and "content" in message:
                        return message.get("content", "")
                    # Completion format
                    if "text" in first_choice:
                        return first_choice.get("text", "")

        # Simple response formats
        for key in ("response", "text", "content", "output", "result", "answer"):
            if key in data:
                value = data[key]
                if isinstance(value, str):
                    return value

        # Fallback: stringify the entire response
        logger.warning(
            f"[{self.name}] Unknown response format, returning raw JSON: {list(data.keys())}"
        )
        return json.dumps(data)

    def _parse_retry_after(self, response: aiohttp.ClientResponse) -> float | None:
        """Parse Retry-After header from response.

        Args:
            response: HTTP response object.

        Returns:
            Retry delay in seconds, or None if not present.
        """
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return None

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list[Message] | None = None,
        target_agent: str | None = None,
    ) -> Critique:
        """Critique a proposal using the external framework.

        Args:
            proposal: The proposal to critique.
            task: The original task description.
            context: Optional conversation context.
            target_agent: Name of the agent who made the proposal.

        Returns:
            Structured critique with issues and suggestions.
        """
        target_desc = f" from {target_agent}" if target_agent else ""
        critique_prompt = f"""You are a critical reviewer. Analyze this proposal{target_desc}:

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

SEVERITY: X (0=trivial, 10=critical)
REASONING: explanation"""

        # Try to use dedicated critique endpoint if available
        critique_url = f"{self.base_url}{self.config.critique_endpoint}"

        # Validate URL against SSRF attacks
        self._validate_endpoint_url(critique_url)

        try:
            session = await self._get_session()
            payload = {
                "model": self.model,
                "proposal": proposal,
                "task": task,
                "target_agent": target_agent,
                "prompt": critique_prompt,
            }

            async with session.post(
                critique_url, json=payload, headers=self._build_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    text = self._extract_response_text(data)
                    return self._parse_critique(text, target_agent or "proposal", proposal)
        except (aiohttp.ClientError, OSError):
            # Fall back to generate endpoint
            pass

        # Fallback: use generate endpoint with critique prompt
        response = await self.generate(critique_prompt, context)
        return self._parse_critique(response, target_agent or "proposal", proposal)

    async def vote(
        self,
        proposals: dict[str, str],
        task: str,
        context: list[Message] | None = None,
    ) -> Any:
        """Vote on proposals using the external framework.

        Args:
            proposals: Dict mapping agent names to their proposals.
            task: The original task description.
            context: Optional conversation context.

        Returns:
            Vote object with choice, confidence, and reasoning.
        """
        # Build vote prompt
        proposals_text = "\n\n".join(
            f"**{agent}**: {proposal}" for agent, proposal in proposals.items()
        )

        vote_prompt = f"""You are a judge evaluating proposals for a task.

Task: {task}

Proposals:
{proposals_text}

Select the best proposal and explain your reasoning.

CHOICE: [agent name]
CONFIDENCE: [0.0-1.0]
CONTINUE: [yes/no - should debate continue?]
REASONING: [your explanation]"""

        # Try dedicated vote endpoint first
        vote_url = f"{self.base_url}{self.config.vote_endpoint}"

        # Validate URL against SSRF attacks
        self._validate_endpoint_url(vote_url)

        try:
            session = await self._get_session()
            payload = {
                "model": self.model,
                "proposals": proposals,
                "task": task,
                "prompt": vote_prompt,
            }

            async with session.post(
                vote_url, json=payload, headers=self._build_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    text = self._extract_response_text(data)
                    return self._parse_vote(text, proposals)
        except (aiohttp.ClientError, OSError):
            # Fall back to generate endpoint
            pass

        # Fallback: use generate endpoint with vote prompt
        response = await self.generate(vote_prompt, context)
        return self._parse_vote(response, proposals)

    def _parse_vote(self, response: str, proposals: dict[str, str]) -> Any:
        """Parse vote response into Vote object.

        Args:
            response: Raw response text.
            proposals: Available proposals for validation.

        Returns:
            Vote object.
        """
        from aragora.core import Vote

        # Parse choice
        choice = None
        choice_match = re.search(r"CHOICE:\s*(\S+)", response, re.IGNORECASE)
        if choice_match:
            choice = choice_match.group(1).strip()
            # Validate choice is a known agent
            if choice not in proposals:
                # Try to find closest match
                for agent in proposals:
                    if agent.lower() in choice.lower() or choice.lower() in agent.lower():
                        choice = agent
                        break
                else:
                    # Default to first agent
                    choice = next(iter(proposals.keys()))

        # Parse confidence
        confidence = 0.5
        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", response, re.IGNORECASE)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                pass

        # Parse continue flag
        continue_debate = False
        continue_match = re.search(r"CONTINUE:\s*(yes|no)", response, re.IGNORECASE)
        if continue_match:
            continue_debate = continue_match.group(1).lower() == "yes"

        # Parse reasoning
        reasoning = ""
        reasoning_match = re.search(
            r"REASONING:\s*(.+?)(?=\n[A-Z]+:|$)", response, re.IGNORECASE | re.DOTALL
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        return Vote(
            agent=self.name,
            choice=choice or next(iter(proposals.keys())),
            confidence=confidence,
            continue_debate=continue_debate,
            reasoning=reasoning,
        )


__all__ = ["ExternalFrameworkAgent", "ExternalFrameworkConfig"]
