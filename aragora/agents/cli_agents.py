"""
CLI-based agent implementations that wrap external AI tools.

These agents invoke CLI tools (codex, claude, openai) as subprocesses,
enabling heterogeneous multi-model debates.
"""

import asyncio
import subprocess
import json
import re
from typing import Optional

from aragora.core import Agent, Critique, Message

# Context window limits (in characters, ~4 chars per token)
# Use 60% of available window to leave room for response
MAX_CONTEXT_CHARS = 120_000  # ~30k tokens, safe for most models
MAX_MESSAGE_CHARS = 20_000   # Individual message truncation limit


class CLIAgent(Agent):
    """Base class for CLI-based agents."""

    def __init__(self, name: str, model: str, role: str = "proposer", timeout: int = 120):
        super().__init__(name, model, role)
        self.timeout = timeout

    async def _run_cli(self, command: list[str], input_text: str = None) -> str:
        """Run a CLI command and return output."""
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE if input_text else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=input_text.encode() if input_text else None),
                timeout=self.timeout,
            )

            if proc.returncode != 0:
                raise RuntimeError(f"CLI command failed: {stderr.decode()}")

            return stdout.decode().strip()

        except asyncio.TimeoutError:
            if proc:
                proc.kill()
                await proc.wait()  # Ensure process is fully cleaned up
            raise TimeoutError(f"CLI command timed out after {self.timeout}s")
        except Exception:
            if proc and proc.returncode is None:
                proc.kill()
                await proc.wait()  # Cleanup zombie processes
            raise

    def _build_context_prompt(self, context: list[Message] = None) -> str:
        """Build context from previous messages with truncation for large contexts."""
        if not context:
            return ""

        # Build messages with individual truncation
        messages = []
        total_chars = 0

        for m in context[-10:]:  # Last 10 messages
            content = m.content
            # Truncate individual messages that are too long
            if len(content) > MAX_MESSAGE_CHARS:
                # Keep start and end, truncate middle
                half = MAX_MESSAGE_CHARS // 2 - 50
                content = (
                    content[:half] +
                    f"\n\n[... {len(m.content) - MAX_MESSAGE_CHARS} chars truncated ...]\n\n" +
                    content[-half:]
                )

            msg_str = f"[Round {m.round}] {m.role} ({m.agent}):\n{content}"

            # Check if adding this message would exceed total limit
            if total_chars + len(msg_str) > MAX_CONTEXT_CHARS:
                # Truncate and stop adding more
                remaining = MAX_CONTEXT_CHARS - total_chars - 100
                if remaining > 500:
                    msg_str = msg_str[:remaining] + "\n[... truncated ...]"
                    messages.append(msg_str)
                break

            messages.append(msg_str)
            total_chars += len(msg_str) + 4  # +4 for separator

        context_str = "\n\n".join(messages)
        return f"\n\nPrevious discussion:\n{context_str}\n\n"

    def _parse_critique(self, response: str, target_agent: str, target_content: str) -> Critique:
        """Parse a critique response into structured format."""
        # Extract issues (lines starting with - or *)
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
                # Try to extract severity
                match = re.search(r'(\d+\.?\d*)', line)
                if match:
                    try:
                        severity = min(1.0, max(0.0, float(match.group(1))))
                        if severity > 1:
                            severity = severity / 10  # Handle 0-10 scale
                    except (ValueError, TypeError):
                        pass
            elif line.startswith(('-', '*', '•')):
                item = line.lstrip('-*• ').strip()
                if current_section == 'issues':
                    issues.append(item)
                elif current_section == 'suggestions':
                    suggestions.append(item)
                else:
                    # Default to issues
                    issues.append(item)

        # If no structured extraction, use the whole response
        if not issues and not suggestions:
            # Split response into issues (first half) and suggestions (second half)
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
            issues=issues[:5],  # Limit to 5 issues
            suggestions=suggestions[:5],  # Limit to 5 suggestions
            severity=severity,
            reasoning=reasoning,
        )


class CodexAgent(CLIAgent):
    """Agent that uses OpenAI Codex CLI."""

    async def generate(self, prompt: str, context: list[Message] = None) -> str:
        """Generate a response using codex exec."""
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        if self.system_prompt:
            full_prompt = f"System context: {self.system_prompt}\n\n{full_prompt}"

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

    async def critique(self, proposal: str, task: str, context: list[Message] = None) -> Critique:
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
    """Agent that uses Claude CLI (claude-code)."""

    async def generate(self, prompt: str, context: list[Message] = None) -> str:
        """Generate a response using claude CLI."""
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        if self.system_prompt:
            full_prompt = f"System context: {self.system_prompt}\n\n{full_prompt}"

        # Use claude with --print flag for non-interactive output
        # Pass prompt via stdin to avoid shell argument length limits on large prompts
        result = await self._run_cli(
            ["claude", "--print", "-p", "-"],
            input_text=full_prompt
        )

        return result

    async def critique(self, proposal: str, task: str, context: list[Message] = None) -> Critique:
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
    """Agent that uses Google Gemini CLI (v0.22+)."""

    async def generate(self, prompt: str, context: list[Message] = None) -> str:
        """Generate a response using gemini CLI."""
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        if self.system_prompt:
            full_prompt = f"System context: {self.system_prompt}\n\n{full_prompt}"

        # Use gemini with positional prompt and --yolo for auto-approval
        # Output format json for easier parsing, text for human-readable
        result = await self._run_cli([
            "gemini", "--yolo", "-o", "text", full_prompt
        ])

        # Filter out YOLO mode message if present
        lines = result.split('\n')
        filtered = [l for l in lines if not l.startswith('YOLO mode is enabled')]
        return '\n'.join(filtered).strip()

    async def critique(self, proposal: str, task: str, context: list[Message] = None) -> Critique:
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
        model: str = None,
        role: str = "proposer",
        timeout: int = 600,
        mode: str = "architect",
    ):
        # Model name is informational - actual model is set by provider_id
        super().__init__(name, model or provider_id, role, timeout)
        self.provider_id = provider_id
        self.mode = mode  # architect, code, ask, debug

    async def generate(self, prompt: str, context: list[Message] = None) -> str:
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

    async def critique(self, proposal: str, task: str, context: list[Message] = None) -> Critique:
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
    """Agent that uses xAI Grok CLI."""

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

    async def generate(self, prompt: str, context: list[Message] = None) -> str:
        """Generate a response using grok CLI."""
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        if self.system_prompt:
            full_prompt = f"System context: {self.system_prompt}\n\n{full_prompt}"

        # Use grok with -p flag for prompt mode (non-interactive)
        result = await self._run_cli([
            "grok", "-p", full_prompt
        ])

        # Extract actual response from JSON conversation format
        return self._extract_grok_response(result)

    async def critique(self, proposal: str, task: str, context: list[Message] = None) -> Critique:
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
    """Agent that uses Alibaba Qwen Code CLI."""

    async def generate(self, prompt: str, context: list[Message] = None) -> str:
        """Generate a response using qwen CLI."""
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        if self.system_prompt:
            full_prompt = f"System context: {self.system_prompt}\n\n{full_prompt}"

        # Use qwen with -p flag for prompt mode (non-interactive)
        result = await self._run_cli([
            "qwen", "-p", full_prompt
        ])

        return result

    async def critique(self, proposal: str, task: str, context: list[Message] = None) -> Critique:
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
    """Agent that uses Deepseek CLI."""

    async def generate(self, prompt: str, context: list[Message] = None) -> str:
        """Generate a response using deepseek CLI."""
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        if self.system_prompt:
            full_prompt = f"System context: {self.system_prompt}\n\n{full_prompt}"

        # Use deepseek CLI
        result = await self._run_cli([
            "deepseek", "-p", full_prompt
        ])

        return result

    async def critique(self, proposal: str, task: str, context: list[Message] = None) -> Critique:
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

    async def generate(self, prompt: str, context: list[Message] = None) -> str:
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
            return data.get("choices", [{}])[0].get("message", {}).get("content", result)
        except json.JSONDecodeError:
            return result

    async def critique(self, proposal: str, task: str, context: list[Message] = None) -> Critique:
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
    """Run an async function synchronously."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)
