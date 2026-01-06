"""
CLI-based agent implementations that wrap external AI tools.

These agents invoke CLI tools (codex, claude, openai) as subprocesses,
enabling heterogeneous multi-model debates.

Supports automatic fallback to OpenRouter API when CLI commands fail due to
rate limits, timeouts, or other errors. Enable fallback by setting
enable_fallback=True (default) and providing OPENROUTER_API_KEY.
"""

import asyncio
import logging
import os
import subprocess
import json
import re
from typing import Optional, TYPE_CHECKING

from aragora.agents.base import CritiqueMixin, MAX_CONTEXT_CHARS, MAX_MESSAGE_CHARS
from aragora.core import Agent, Critique, Message

if TYPE_CHECKING:
    from aragora.agents.api_agents import OpenRouterAgent

# Re-export constants for backward compatibility
__all__ = [
    "CLIAgent", "CodexAgent", "ClaudeAgent", "OpenAIAgent",
    "GeminiCLIAgent", "GrokCLIAgent", "QwenCLIAgent", "DeepseekCLIAgent", "KiloCodeAgent",
    "MAX_CONTEXT_CHARS", "MAX_MESSAGE_CHARS",
]

logger = logging.getLogger(__name__)

# Patterns that indicate rate limiting or quota errors in CLI output
RATE_LIMIT_PATTERNS = [
    "rate limit", "rate_limit", "ratelimit",
    "429", "too many requests",
    "quota exceeded", "quota_exceeded",
    "resource exhausted", "resource_exhausted",
    "billing", "credit balance",
    "insufficient_quota",
]


class CLIAgent(CritiqueMixin, Agent):
    """Base class for CLI-based agents.

    Supports automatic fallback to OpenRouter API when CLI commands fail.
    Enable with enable_fallback=True (default) and OPENROUTER_API_KEY env var.
    """

    # Map CLI agent models to OpenRouter model identifiers
    OPENROUTER_MODEL_MAP: dict[str, str] = {
        # Claude models
        "claude-opus-4-5-20251101": "anthropic/claude-opus-4",
        "claude-sonnet-4-20250514": "anthropic/claude-sonnet-4",
        "claude-3-opus-20240229": "anthropic/claude-3-opus",
        "claude-3-sonnet-20240229": "anthropic/claude-3-sonnet",
        # OpenAI/Codex models
        "gpt-5.2-codex": "openai/gpt-4o",
        "gpt-4o": "openai/gpt-4o",
        "gpt-4-turbo": "openai/gpt-4-turbo",
        "gpt-4": "openai/gpt-4",
        # Gemini models
        "gemini-3-pro": "google/gemini-2.0-flash-001",
        "gemini-2.0-flash": "google/gemini-2.0-flash-001",
        "gemini-1.5-pro": "google/gemini-pro-1.5",
        # Grok models
        "grok-3": "x-ai/grok-2-1212",
        "grok-2": "x-ai/grok-2-1212",
        # Deepseek models
        "deepseek-coder": "deepseek/deepseek-chat",
        "deepseek-v3": "deepseek/deepseek-chat",
        # Qwen models
        "qwen-2.5-coder": "qwen/qwen-2.5-coder-32b-instruct",
    }

    def __init__(
        self,
        name: str,
        model: str,
        role: str = "proposer",
        timeout: int = 120,
        enable_fallback: bool = True,
    ):
        super().__init__(name, model, role)
        self.timeout = timeout
        self.enable_fallback = enable_fallback
        self._fallback_agent: Optional["OpenRouterAgent"] = None
        self._fallback_used = False  # Track if fallback was triggered this session

    def _get_fallback_agent(self) -> Optional["OpenRouterAgent"]:
        """Get or create the OpenRouter fallback agent.

        Returns None if fallback is disabled or OPENROUTER_API_KEY is not set.
        """
        if not self.enable_fallback:
            return None

        if self._fallback_agent is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                logger.debug(f"[{self.name}] No OPENROUTER_API_KEY set, fallback disabled")
                return None

            # Import here to avoid circular dependency
            from aragora.agents.api_agents import OpenRouterAgent

            # Map the model to OpenRouter format
            openrouter_model = self.OPENROUTER_MODEL_MAP.get(
                self.model, "anthropic/claude-sonnet-4"  # Default fallback model
            )

            self._fallback_agent = OpenRouterAgent(
                name=f"{self.name}_fallback",
                model=openrouter_model,
                role=self.role,
                api_key=api_key,
                timeout=self.timeout,
            )
            # Copy system prompt if set
            if self.system_prompt:
                self._fallback_agent.system_prompt = self.system_prompt
            logger.info(f"[{self.name}] Created OpenRouter fallback agent with model {openrouter_model}")

        return self._fallback_agent

    def _is_fallback_error(self, error: Exception) -> bool:
        """Check if the error should trigger a fallback to OpenRouter.

        Detects rate limits, timeouts, and CLI-specific errors.
        """
        error_str = str(error).lower()

        # Check for rate limit patterns
        for pattern in RATE_LIMIT_PATTERNS:
            if pattern in error_str:
                return True

        # Timeout errors should trigger fallback
        if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            return True

        # CLI command failures (non-zero exit, process errors)
        if isinstance(error, RuntimeError) and "cli command failed" in error_str:
            return True

        return False

    def _sanitize_cli_arg(self, arg: str) -> str:
        """Sanitize a string for use as a CLI argument.

        Removes null bytes and other control characters that can cause
        'embedded null byte' ValueError from subprocess calls.
        Command-line arguments are null-terminated in C, so null bytes
        in arguments are not allowed by the OS.
        """
        if not isinstance(arg, str):
            return str(arg)
        # Remove null bytes (cause 'embedded null byte' error)
        sanitized = arg.replace('\x00', '')
        # Remove other problematic control characters (except newlines/tabs)
        sanitized = re.sub(r'[\x01-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', sanitized)
        return sanitized

    async def _run_cli(self, command: list[str], input_text: str | None = None) -> str:
        """Run a CLI command and return output."""
        # Sanitize all command arguments to prevent 'embedded null byte' errors
        sanitized_command = [self._sanitize_cli_arg(arg) for arg in command]
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *sanitized_command,
                stdin=asyncio.subprocess.PIPE if input_text else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Also sanitize stdin input (used by ClaudeAgent)
            sanitized_input = self._sanitize_cli_arg(input_text) if input_text else None
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=sanitized_input.encode() if sanitized_input else None),
                timeout=self.timeout,
            )

            if proc.returncode != 0:
                raise RuntimeError(f"CLI command failed: {stderr.decode('utf-8', errors='replace')}")

            return stdout.decode('utf-8', errors='replace').strip()

        except asyncio.TimeoutError:
            if proc:
                proc.kill()
                await proc.wait()  # Ensure process is fully cleaned up
            raise TimeoutError(f"CLI command timed out after {self.timeout}s")
        except Exception as e:
            if proc and proc.returncode is None:
                logger.debug(f"[cleanup] Killing subprocess after error: {e}")
                proc.kill()
                await proc.wait()  # Cleanup zombie processes
            raise

    def _build_context_prompt(self, context: list[Message] | None = None) -> str:
        """Build context from previous messages with truncation for large contexts.

        Delegates to CritiqueMixin with CLI-specific settings.
        """
        # Use mixin method with truncation and CLI sanitization
        return CritiqueMixin._build_context_prompt(
            self, context, truncate=True, sanitize_fn=self._sanitize_cli_arg
        )

    # _parse_critique is inherited from CritiqueMixin


class CodexAgent(CLIAgent):
    """Agent that uses OpenAI Codex CLI.

    Falls back to OpenRouter (OpenAI GPT-4o) on CLI failures if enabled.
    """

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using codex exec.

        Automatically falls back to OpenRouter API on rate limits or CLI errors.
        """
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        if self.system_prompt:
            full_prompt = f"System context: {self.system_prompt}\n\n{full_prompt}"

        try:
            # Use codex exec for non-interactive execution
            result = await self._run_cli([
                "codex", "exec", "--skip-git-repo-check", full_prompt
            ])

            # Extract the actual response (skip the header)
            lines = result.split('\n')
            # Find where the actual response starts (after "codex" line)
            response_lines = []
            in_response = False
            for line in lines:
                if line.strip() == 'codex':
                    in_response = True
                    continue
                if in_response:
                    # Skip token count lines
                    if line.startswith('tokens used'):
                        continue
                    response_lines.append(line)

            return '\n'.join(response_lines).strip() if response_lines else result

        except Exception as e:
            # Check if we should fallback
            if self._is_fallback_error(e):
                fallback = self._get_fallback_agent()
                if fallback:
                    logger.warning(
                        f"[{self.name}] CLI failed ({type(e).__name__}: {str(e)[:100]}), "
                        f"falling back to OpenRouter"
                    )
                    self._fallback_used = True
                    return await fallback.generate(prompt, context)
            raise

    async def critique(self, proposal: str, task: str, context: list[Message] | None = None) -> Critique:
        """Critique a proposal using codex."""
        critique_prompt = f"""You are a critical reviewer. Analyze this proposal for the given task.

Task: {task}

Proposal to critique:
{proposal}

Provide a structured critique with:
1. ISSUES: List specific problems, errors, or weaknesses (use bullet points)
2. SUGGESTIONS: List concrete improvements (use bullet points)
3. SEVERITY: Rate 0.0 (minor) to 1.0 (critical)
4. REASONING: Brief explanation of your assessment

Be constructive but thorough. Identify both technical and conceptual issues."""

        response = await self.generate(critique_prompt, context)
        return self._parse_critique(response, "proposal", proposal)


class ClaudeAgent(CLIAgent):
    """Agent that uses Claude CLI (claude-code).

    Falls back to OpenRouter (Anthropic Claude) on CLI failures if enabled.
    """

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using claude CLI.

        Automatically falls back to OpenRouter API on rate limits or CLI errors.
        """
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        if self.system_prompt:
            full_prompt = f"System context: {self.system_prompt}\n\n{full_prompt}"

        try:
            # Use claude with --print flag for non-interactive output
            # Pass prompt via stdin to avoid shell argument length limits on large prompts
            result = await self._run_cli(
                ["claude", "--print", "-p", "-"],
                input_text=full_prompt
            )
            return result

        except Exception as e:
            # Check if we should fallback
            if self._is_fallback_error(e):
                fallback = self._get_fallback_agent()
                if fallback:
                    logger.warning(
                        f"[{self.name}] CLI failed ({type(e).__name__}: {str(e)[:100]}), "
                        f"falling back to OpenRouter"
                    )
                    self._fallback_used = True
                    return await fallback.generate(prompt, context)
            raise

    async def critique(self, proposal: str, task: str, context: list[Message] | None = None) -> Critique:
        """Critique a proposal using claude."""
        critique_prompt = f"""Analyze this proposal critically for the given task.

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


class GeminiCLIAgent(CLIAgent):
    """Agent that uses Google Gemini CLI (v0.22+).

    Falls back to OpenRouter (Google Gemini) on CLI failures if enabled.
    """

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using gemini CLI.

        Automatically falls back to OpenRouter API on rate limits or CLI errors.
        """
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        if self.system_prompt:
            full_prompt = f"System context: {self.system_prompt}\n\n{full_prompt}"

        try:
            # Use gemini with positional prompt and --yolo for auto-approval
            # Output format json for easier parsing, text for human-readable
            result = await self._run_cli([
                "gemini", "--yolo", "-o", "text", full_prompt
            ])

            # Filter out YOLO mode message if present
            lines = result.split('\n')
            filtered = [l for l in lines if not l.startswith('YOLO mode is enabled')]
            return '\n'.join(filtered).strip()

        except Exception as e:
            if self._is_fallback_error(e):
                fallback = self._get_fallback_agent()
                if fallback:
                    logger.warning(
                        f"[{self.name}] CLI failed ({type(e).__name__}: {str(e)[:100]}), "
                        f"falling back to OpenRouter"
                    )
                    self._fallback_used = True
                    return await fallback.generate(prompt, context)
            raise

    async def critique(self, proposal: str, task: str, context: list[Message] | None = None) -> Critique:
        """Critique a proposal using gemini."""
        critique_prompt = f"""Analyze this proposal critically for the given task.

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


class KiloCodeAgent(CLIAgent):
    """
    Agent that uses Kilo Code CLI for codebase exploration.

    Kilo Code is an agentic coding assistant that can explore codebases
    autonomously. It supports multiple AI providers including Gemini and Grok
    via direct API or OpenRouter.

    This agent is particularly useful for context gathering phases where
    the AI needs to read and understand the codebase structure.

    Provider IDs (configured in ~/.kilocode/cli/config.json):
    - gemini-explorer: Gemini 3 Pro via direct API
    - grok-explorer: Grok via xAI API
    - openrouter-gemini: Gemini via OpenRouter
    - openrouter-grok: Grok via OpenRouter
    """

    def __init__(
        self,
        name: str,
        provider_id: str = "gemini-explorer",
        model: str | None = None,
        role: str = "proposer",
        timeout: int = 600,
        mode: str = "architect",
    ):
        # Model name is informational - actual model is set by provider_id
        super().__init__(name, model or provider_id, role, timeout)
        self.provider_id = provider_id
        self.mode = mode  # architect, code, ask, debug

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using kilocode CLI with codebase access."""
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        if self.system_prompt:
            full_prompt = f"System context: {self.system_prompt}\n\n{full_prompt}"

        # Use kilocode with:
        # --auto: autonomous mode (non-interactive)
        # --yolo: auto-approve tool permissions
        # --json: output as JSON for parsing
        # --provider: select the configured provider
        # --mode: architect mode for exploration
        # --timeout: prevent hanging
        cmd = [
            "kilocode",
            "--auto",
            "--yolo",
            "--json",
            "-pv", self.provider_id,
            "-m", self.mode,
            "-t", str(self.timeout),
            full_prompt,
        ]

        result = await self._run_cli(cmd)
        return self._extract_kilocode_response(result)

    def _extract_kilocode_response(self, output: str) -> str:
        """Extract the assistant response from Kilo Code JSON output."""
        # Kilocode with --json outputs JSON lines
        lines = output.strip().split('\n')
        responses = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                # Look for assistant messages
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if content:
                        responses.append(content)
                # Also check for 'text' field in some message formats
                elif msg.get("type") == "text":
                    text = msg.get("text", "")
                    if text:
                        responses.append(text)
            except json.JSONDecodeError:
                # If not JSON, might be plain text
                continue

        # Return combined responses or raw output if nothing extracted
        if responses:
            return "\n\n".join(responses)
        return output

    async def critique(self, proposal: str, task: str, context: list[Message] | None = None) -> Critique:
        """Critique a proposal using kilocode."""
        critique_prompt = f"""Analyze this proposal critically for the given task.

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


class GrokCLIAgent(CLIAgent):
    """Agent that uses xAI Grok CLI.

    Falls back to OpenRouter (xAI Grok) on CLI failures if enabled.
    """

    def _extract_grok_response(self, output: str) -> str:
        """Extract the final assistant response from Grok CLI JSON output.

        Grok CLI returns JSON lines with full conversation history.
        We need to extract only the final assistant content.
        """
        # Try to parse as JSON lines (each line is a JSON object)
        lines = output.strip().split('\n')
        final_content = None

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                # Look for assistant messages with actual content (not just tool calls)
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    # Skip messages that are just "Using tools..." placeholders
                    if content and not content.startswith("Using tools"):
                        final_content = content
            except json.JSONDecodeError:
                # If it's not JSON, might be plain text response
                # This could be the actual response if Grok outputs plain text
                if not output.startswith('{"role":'):
                    return output
                continue

        # Return the final content we found, or the raw output if nothing was extracted
        return final_content if final_content else output

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using grok CLI.

        Automatically falls back to OpenRouter API on rate limits or CLI errors.
        """
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        if self.system_prompt:
            full_prompt = f"System context: {self.system_prompt}\n\n{full_prompt}"

        try:
            # Use grok with -p flag for prompt mode (non-interactive)
            result = await self._run_cli([
                "grok", "-p", full_prompt
            ])

            # Extract actual response from JSON conversation format
            return self._extract_grok_response(result)

        except Exception as e:
            if self._is_fallback_error(e):
                fallback = self._get_fallback_agent()
                if fallback:
                    logger.warning(
                        f"[{self.name}] CLI failed ({type(e).__name__}: {str(e)[:100]}), "
                        f"falling back to OpenRouter"
                    )
                    self._fallback_used = True
                    return await fallback.generate(prompt, context)
            raise

    async def critique(self, proposal: str, task: str, context: list[Message] | None = None) -> Critique:
        """Critique a proposal using grok."""
        critique_prompt = f"""Analyze this proposal critically for the given task.

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


class QwenCLIAgent(CLIAgent):
    """Agent that uses Alibaba Qwen Code CLI.

    Falls back to OpenRouter (Qwen) on CLI failures if enabled.
    """

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using qwen CLI.

        Automatically falls back to OpenRouter API on rate limits or CLI errors.
        """
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        if self.system_prompt:
            full_prompt = f"System context: {self.system_prompt}\n\n{full_prompt}"

        try:
            # Use qwen with -p flag for prompt mode (non-interactive)
            result = await self._run_cli([
                "qwen", "-p", full_prompt
            ])
            return result

        except Exception as e:
            if self._is_fallback_error(e):
                fallback = self._get_fallback_agent()
                if fallback:
                    logger.warning(
                        f"[{self.name}] CLI failed ({type(e).__name__}: {str(e)[:100]}), "
                        f"falling back to OpenRouter"
                    )
                    self._fallback_used = True
                    return await fallback.generate(prompt, context)
            raise

    async def critique(self, proposal: str, task: str, context: list[Message] | None = None) -> Critique:
        """Critique a proposal using qwen."""
        critique_prompt = f"""Analyze this proposal critically for the given task.

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


class DeepseekCLIAgent(CLIAgent):
    """Agent that uses Deepseek CLI.

    Falls back to OpenRouter (Deepseek) on CLI failures if enabled.
    """

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using deepseek CLI.

        Automatically falls back to OpenRouter API on rate limits or CLI errors.
        """
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        if self.system_prompt:
            full_prompt = f"System context: {self.system_prompt}\n\n{full_prompt}"

        try:
            # Use deepseek CLI
            result = await self._run_cli([
                "deepseek", "-p", full_prompt
            ])
            return result

        except Exception as e:
            if self._is_fallback_error(e):
                fallback = self._get_fallback_agent()
                if fallback:
                    logger.warning(
                        f"[{self.name}] CLI failed ({type(e).__name__}: {str(e)[:100]}), "
                        f"falling back to OpenRouter"
                    )
                    self._fallback_used = True
                    return await fallback.generate(prompt, context)
            raise

    async def critique(self, proposal: str, task: str, context: list[Message] | None = None) -> Critique:
        """Critique a proposal using deepseek."""
        critique_prompt = f"""Analyze this proposal critically for the given task.

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


class OpenAIAgent(CLIAgent):
    """Agent that uses OpenAI CLI."""

    def __init__(self, name: str, model: str = "gpt-4o", role: str = "proposer", timeout: int = 120):
        super().__init__(name, model, role, timeout)

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using openai CLI."""
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        if self.system_prompt:
            full_prompt = f"System context: {self.system_prompt}\n\n{full_prompt}"

        # Use openai api chat.completions.create
        messages = json.dumps([{"role": "user", "content": full_prompt}])

        result = await self._run_cli([
            "openai", "api", "chat.completions.create",
            "-m", self.model,
            "-g", "user", full_prompt,
        ])

        # Parse JSON response
        try:
            data = json.loads(result)
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", result)
            return result
        except json.JSONDecodeError:
            return result

    async def critique(self, proposal: str, task: str, context: list[Message] | None = None) -> Critique:
        """Critique a proposal using openai."""
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


# Synchronous wrappers for convenience
def run_sync(coro):
    """Run an async function synchronously.

    Uses asyncio.run() which properly creates and closes the event loop,
    avoiding resource leaks and deprecation warnings from get_event_loop().
    """
    return asyncio.run(coro)
