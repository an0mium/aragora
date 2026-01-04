"""
API-based agent implementations.

These agents call AI APIs directly (HTTP), enabling use without CLI tools.
Supports Gemini, Ollama (local), and direct OpenAI/Anthropic API calls.
"""

import asyncio
import aiohttp
import json
import os
import re
from typing import Optional

from aragora.core import Agent, Critique, Message


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
                    except:
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
        model: str = "gemini-2.5-flash",  # Fast, capable, good for debates
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

    async def generate(self, prompt: str, context: list[Message] = None) -> str:
        """Generate a response using Gemini API."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable required")

        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        if self.system_prompt:
            full_prompt = f"System context: {self.system_prompt}\n\n{full_prompt}"

        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 65536,  # Gemini 2.5 supports up to 65k output tokens
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Gemini API error {response.status}: {error_text}")

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
                        raise RuntimeError(f"Ollama API error {response.status}: {error_text}")

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
        model: str = "claude-sonnet-4-20250514",
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
                    raise RuntimeError(f"Anthropic API error {response.status}: {error_text}")

                data = await response.json()

                try:
                    return data["content"][0]["text"]
                except (KeyError, IndexError):
                    raise RuntimeError(f"Unexpected Anthropic response format: {data}")

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
    """Agent that uses OpenAI API directly (without CLI)."""

    def __init__(
        self,
        name: str = "openai-api",
        model: str = "gpt-4o",
        role: str = "proposer",
        timeout: int = 120,
        api_key: str = None,
    ):
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1",
        )

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
                    raise RuntimeError(f"OpenAI API error {response.status}: {error_text}")

                data = await response.json()

                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    raise RuntimeError(f"Unexpected OpenAI response format: {data}")

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
