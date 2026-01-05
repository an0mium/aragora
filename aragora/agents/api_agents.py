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
from typing import Optional

from aragora.core import Agent, Critique, Message

logger = logging.getLogger(__name__)


# Patterns that might contain sensitive data in error messages
_SENSITIVE_PATTERNS = [
    (r'sk-[a-zA-Z0-9]{20,}', '<REDACTED_KEY>'),  # OpenAI key pattern
    (r'AIza[a-zA-Z0-9_-]{35}', '<REDACTED_KEY>'),  # Google API key pattern
    (r'["\']?api[_-]?key["\']?\s*[:=]\s*["\']?[\w-]+["\']?', 'api_key=<REDACTED>'),
    (r'["\']?authorization["\']?\s*[:=]\s*["\']?Bearer\s+[\w.-]+["\']?', 'authorization=<REDACTED>'),
    (r'["\']?token["\']?\s*[:=]\s*["\']?[\w.-]+["\']?', 'token=<REDACTED>'),
    (r'["\']?secret["\']?\s*[:=]\s*["\']?[\w-]+["\']?', 'secret=<REDACTED>'),
    (r'x-api-key:\s*[\w-]+', 'x-api-key: <REDACTED>'),
]


def _sanitize_error_message(error_text: str, max_length: int = 500) -> str:
    """Sanitize error message to remove potential secrets.

    - Redacts patterns that look like API keys or tokens
    - Truncates to prevent log flooding
    - Preserves useful diagnostic info (status codes, error types)
    """
    sanitized = error_text

    # Apply all redaction patterns
    for pattern, replacement in _SENSITIVE_PATTERNS:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

    # Truncate long messages
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "... [truncated]"

    return sanitized


class APIAgent(Agent):
    """Base class for API-based agents."""

    def __init__(
        self,
        name: str,
        model: str,
        role: str = "proposer",
        timeout: int = 120,
        api_key: str = None,
        base_url: str = None,
    ):
        super().__init__(name, model, role)
        self.timeout = timeout
        self.api_key = api_key
        self.base_url = base_url
        self.agent_type = "api"  # Default for API agents

    def _build_context_prompt(self, context: list[Message] = None) -> str:
        """Build context from previous messages."""
        if not context:
            return ""

        context_str = "\n\n".join([
            f"[Round {m.round}] {m.role} ({m.agent}):\n{m.content}"
            for m in context[-10:]
        ])
        return f"\n\nPrevious discussion:\n{context_str}\n\n"

    def _parse_critique(self, response: str, target_agent: str, target_content: str) -> Critique:
        """Parse a critique response into structured format."""
        issues = []
        suggestions = []
        severity = 0.5
        reasoning = ""

        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            lower = line.lower()
            if 'issue' in lower or 'problem' in lower or 'concern' in lower:
                current_section = 'issues'
            elif 'suggest' in lower or 'recommend' in lower or 'improvement' in lower:
                current_section = 'suggestions'
            elif 'severity' in lower:
                match = re.search(r'(\d+\.?\d*)', line)
                if match:
                    try:
                        severity = min(1.0, max(0.0, float(match.group(1))))
                        if severity > 1:
                            severity = severity / 10
                    except (ValueError, TypeError):
                        pass
            elif line.startswith(('-', '*', '•')):
                item = line.lstrip('-*• ').strip()
                if current_section == 'issues':
                    issues.append(item)
                elif current_section == 'suggestions':
                    suggestions.append(item)
                else:
                    issues.append(item)

        if not issues and not suggestions:
            sentences = [s.strip() for s in response.replace('\n', ' ').split('.') if s.strip()]
            mid = len(sentences) // 2
            issues = sentences[:mid] if sentences else ["See full response"]
            suggestions = sentences[mid:] if len(sentences) > mid else []
            reasoning = response[:500]
        else:
            reasoning = response[:500]

        return Critique(
            agent=self.name,
            target_agent=target_agent,
            target_content=target_content[:200],
            issues=issues[:5],
            suggestions=suggestions[:5],
            severity=severity,
            reasoning=reasoning,
        )


class GeminiAgent(APIAgent):
    """Agent that uses Google Gemini API directly (not CLI).

    Note: The gemini CLI sends massive folder context by default and
    can exhaust quota quickly. This API agent is much more efficient.
    """

    def __init__(
        self,
        name: str = "gemini",
        model: str = "gemini-3-pro-preview",  # Gemini 3 Pro Preview - advanced reasoning
        role: str = "proposer",
        timeout: int = 120,
        api_key: str = None,
    ):
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            api_key=api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta",
        )
        self.agent_type = "gemini"

    async def generate(self, prompt: str, context: list[Message] = None) -> str:
        """Generate a response using Gemini API."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable required")

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
                        print(f"  [gemini] Warning: Response truncated at {len(text)} chars, using partial content")
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

    async def generate_stream(self, prompt: str, context: list[Message] = None):
        """Stream tokens from Gemini API.

        Yields chunks of text as they arrive from the API.
        """
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable required")

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
                async for chunk in response.content.iter_any():
                    buffer += chunk
                    # Try to parse complete JSON objects from buffer
                    # Gemini streams as a JSON array: [{...}, {...}, ...]
                    text = buffer.decode('utf-8', errors='ignore')

                    # Find complete candidate objects
                    while True:
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

    async def critique(self, proposal: str, task: str, context: list[Message] = None) -> Critique:
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
        base_url: str = None,
    ):
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            base_url=base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        )
        self.agent_type = "ollama"

    async def generate(self, prompt: str, context: list[Message] = None) -> str:
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

                    data = await response.json()
                    return data.get("response", "")

            except aiohttp.ClientConnectorError:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Is Ollama running? Start with: ollama serve"
                )

    async def critique(self, proposal: str, task: str, context: list[Message] = None) -> Critique:
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
    """Agent that uses Anthropic API directly (without CLI)."""

    def __init__(
        self,
        name: str = "claude-api",
        model: str = "claude-opus-4-5-20251101",
        role: str = "proposer",
        timeout: int = 120,
        api_key: str = None,
    ):
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            base_url="https://api.anthropic.com/v1",
        )
        self.agent_type = "anthropic"

    async def generate(self, prompt: str, context: list[Message] = None) -> str:
        """Generate a response using Anthropic API."""
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable required")

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
                    raise RuntimeError(f"Anthropic API error {response.status}: {sanitized}")

                data = await response.json()

                try:
                    return data["content"][0]["text"]
                except (KeyError, IndexError):
                    raise RuntimeError(f"Unexpected Anthropic response format: {data}")

    async def generate_stream(self, prompt: str, context: list[Message] = None):
        """Stream tokens from Anthropic API.

        Yields chunks of text as they arrive from the API using SSE.
        """
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable required")

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
                    raise RuntimeError(f"Anthropic streaming API error {response.status}: {sanitized}")

                # Anthropic uses SSE format: data: {...}\n\n
                buffer = ""
                async for chunk in response.content.iter_any():
                    buffer += chunk.decode('utf-8', errors='ignore')

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

    async def critique(self, proposal: str, task: str, context: list[Message] = None) -> Critique:
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
        api_key: str = None,
        enable_fallback: bool = True,
    ):
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
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

            # Import here to avoid circular imports
            from aragora.agents.api_agents import OpenRouterAgent

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

    async def generate(self, prompt: str, context: list[Message] = None) -> str:
        """Generate a response using OpenAI API."""
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")

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

    async def generate_stream(self, prompt: str, context: list[Message] = None):
        """Stream tokens from OpenAI API.

        Yields chunks of text as they arrive from the API using SSE.
        """
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")

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
                async for chunk in response.content.iter_any():
                    buffer += chunk.decode('utf-8', errors='ignore')

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

    async def critique(self, proposal: str, task: str, context: list[Message] = None) -> Critique:
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
        api_key: str = None,
    ):
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            api_key=api_key or os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY"),
            base_url="https://api.x.ai/v1",
        )
        self.agent_type = "grok"

    async def generate(self, prompt: str, context: list[Message] = None) -> str:
        """Generate a response using Grok API."""
        if not self.api_key:
            raise ValueError("XAI_API_KEY or GROK_API_KEY environment variable required")

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

    async def generate_stream(self, prompt: str, context: list[Message] = None):
        """Stream tokens from Grok API."""
        if not self.api_key:
            raise ValueError("XAI_API_KEY or GROK_API_KEY environment variable required")

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

    async def critique(self, proposal: str, task: str, context: list[Message] = None) -> Critique:
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
        system_prompt: str = None,
        timeout: int = 300,
    ):
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
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

    async def generate(self, prompt: str, context: list[Message] = None) -> str:
        """Generate a response using OpenRouter API."""
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable required")

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
                    raise RuntimeError(f"OpenRouter API error {response.status}: {sanitized}")

                data = await response.json()
                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    raise RuntimeError(f"Unexpected OpenRouter response format: {data}")

    async def generate_stream(self, prompt: str, context: list[Message] = None):
        """Stream tokens from OpenRouter API.

        Yields chunks of text as they arrive from the API using SSE.
        """
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable required")

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

    async def critique(self, proposal: str, task: str, context: list[Message] = None) -> Critique:
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

    def __init__(self, name: str = "deepseek", role: str = "analyst", system_prompt: str = None):
        super().__init__(
            name=name,
            role=role,
            model="deepseek/deepseek-v3.2",  # V3.2 latest
            system_prompt=system_prompt,
        )
        self.agent_type = "deepseek"


class DeepSeekReasonerAgent(OpenRouterAgent):
    """DeepSeek R1 via OpenRouter - reasoning model with chain-of-thought."""

    def __init__(self, name: str = "deepseek-r1", role: str = "analyst", system_prompt: str = None):
        super().__init__(
            name=name,
            role=role,
            model="deepseek/deepseek-reasoner",  # R1 reasoning model
            system_prompt=system_prompt,
        )
        self.agent_type = "deepseek-r1"


class DeepSeekV3Agent(OpenRouterAgent):
    """DeepSeek V3.2 via OpenRouter - integrated thinking + tool-use, GPT-5 class reasoning."""

    def __init__(self, name: str = "deepseek-v3", role: str = "analyst", system_prompt: str = None):
        super().__init__(
            name=name,
            role=role,
            model="deepseek/deepseek-v3.2",  # V3.2 with integrated thinking + tool-use
            system_prompt=system_prompt,
        )
        self.agent_type = "deepseek-v3"


class LlamaAgent(OpenRouterAgent):
    """Llama 3.3 70B via OpenRouter."""

    def __init__(self, name: str = "llama", role: str = "analyst", system_prompt: str = None):
        super().__init__(
            name=name,
            role=role,
            model="meta-llama/llama-3.3-70b-instruct",
            system_prompt=system_prompt,
        )
        self.agent_type = "llama"


class MistralAgent(OpenRouterAgent):
    """Mistral Large via OpenRouter."""

    def __init__(self, name: str = "mistral", role: str = "analyst", system_prompt: str = None):
        super().__init__(
            name=name,
            role=role,
            model="mistralai/mistral-large-2411",
            system_prompt=system_prompt,
        )
        self.agent_type = "mistral"
