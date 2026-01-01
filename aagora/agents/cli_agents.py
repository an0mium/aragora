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

from aagora.core import Agent, Critique, Message


class CLIAgent(Agent):
    """Base class for CLI-based agents."""

    def __init__(self, name: str, model: str, role: str = "proposer", timeout: int = 120):
        super().__init__(name, model, role)
        self.timeout = timeout

    async def _run_cli(self, command: list[str], input_text: str = None) -> str:
        """Run a CLI command and return output."""
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
            proc.kill()
            raise TimeoutError(f"CLI command timed out after {self.timeout}s")

    def _build_context_prompt(self, context: list[Message] = None) -> str:
        """Build context from previous messages."""
        if not context:
            return ""

        context_str = "\n\n".join([
            f"[Round {m.round}] {m.role} ({m.agent}):\n{m.content}"
            for m in context[-10:]  # Last 10 messages to avoid context overflow
        ])
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
                    except:
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
        result = await self._run_cli([
            "claude", "--print", "-p", full_prompt
        ])

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
    """Agent that uses Google Gemini CLI."""

    async def generate(self, prompt: str, context: list[Message] = None) -> str:
        """Generate a response using gemini CLI."""
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt

        if self.system_prompt:
            full_prompt = f"System context: {self.system_prompt}\n\n{full_prompt}"

        # Use gemini with -p flag for prompt mode (non-interactive)
        result = await self._run_cli([
            "gemini", "-p", full_prompt
        ])

        return result

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


class GrokCLIAgent(CLIAgent):
    """Agent that uses xAI Grok CLI."""

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

        return result

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
