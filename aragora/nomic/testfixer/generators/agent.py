"""
Agent-backed CodeGenerator for TestFixer.

Uses Aragora agents (Codex, Claude, etc.) to propose, critique,
and synthesize test fixes. Designed to work with the TestFixer
loop while keeping output structured and easy to apply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from aragora.agents.base import create_agent
from aragora.nomic.testfixer.analyzer import FailureAnalysis
from aragora.nomic.testfixer.proposer import CodeGenerator

_FILE_RE = re.compile(r"<file>(.*?)</file>", re.DOTALL | re.IGNORECASE)
_RATIONALE_RE = re.compile(r"<rationale>(.*?)</rationale>", re.DOTALL | re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"<confidence>(.*?)</confidence>", re.DOTALL | re.IGNORECASE)
_VERDICT_RE = re.compile(r"<verdict>(.*?)</verdict>", re.DOTALL | re.IGNORECASE)
_CRITIQUE_RE = re.compile(r"<critique>(.*?)</critique>", re.DOTALL | re.IGNORECASE)


@dataclass
class AgentGeneratorConfig:
    """Configuration for agent-backed code generation."""

    agent_type: str
    model: str | None = None
    role: str = "proposer"
    name: str | None = None
    api_key: str | None = None
    timeout_seconds: float | None = None
    max_context_chars: int = 80_000
    max_output_chars: int = 200_000


class AgentCodeGenerator(CodeGenerator):
    """CodeGenerator backed by an Aragora agent."""

    def __init__(self, config: AgentGeneratorConfig):
        self.config = config
        self.agent = create_agent(
            model_type=config.agent_type,
            name=config.name or f"testfixer_{config.agent_type}",
            role=config.role,
            model=config.model,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

    def _truncate(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        head = limit // 2 - 200
        tail = limit // 2 - 200
        return f"{text[:head]}\n\n[... {len(text) - limit} chars truncated ...]\n\n{text[-tail:]}"

    def _extract_tag(self, pattern: re.Pattern, text: str) -> str | None:
        match = pattern.search(text)
        if not match:
            return None
        return match.group(1).strip()

    def _clean_file_output(self, text: str) -> str:
        file_content = self._extract_tag(_FILE_RE, text)
        if file_content is not None:
            return file_content

        # Strip code fences if present
        fence_match = re.search(r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()

        return text.strip()

    def _parse_confidence(self, text: str, default: float = 0.5) -> float:
        raw = self._extract_tag(_CONFIDENCE_RE, text)
        if not raw:
            return default
        try:
            return max(0.0, min(1.0, float(raw)))
        except (TypeError, ValueError):
            return default

    async def generate_fix(
        self,
        analysis: FailureAnalysis,
        file_content: str,
        file_path: str,
    ) -> tuple[str, str, float]:
        """Generate fix using the agent."""

        analysis_prompt = analysis.to_fix_prompt()
        prompt = (
            "You are fixing a failing test or implementation. "
            "Return ONLY the updated full file contents wrapped in <file>...</file>.\n"
            "Also include <rationale>...</rationale> and <confidence>0-1</confidence>.\n"
            "Do NOT include extra commentary or markdown outside these tags.\n\n"
            f"Target file: {file_path}\n\n"
            f"Failure analysis:\n{analysis_prompt}\n\n"
            "Current file contents:\n"
            f"{self._truncate(file_content, self.config.max_context_chars)}"
        )

        response = await self.agent.generate(prompt)
        fixed_content = self._clean_file_output(response)
        rationale = self._extract_tag(_RATIONALE_RE, response) or ""
        confidence = self._parse_confidence(response, default=0.55)

        if not fixed_content:
            fixed_content = file_content
            rationale = rationale or "No changes returned"
            confidence = min(confidence, 0.3)

        if len(fixed_content) > self.config.max_output_chars:
            fixed_content = fixed_content[: self.config.max_output_chars]

        return fixed_content, rationale, confidence

    async def critique_fix(
        self,
        analysis: FailureAnalysis,
        original_content: str,
        proposed_fix: str,
        rationale: str,
    ) -> tuple[str, bool]:
        """Critique a proposed fix."""

        prompt = (
            "Review the proposed fix. Reply with <verdict>pass|fail</verdict> and "
            "<critique>...</critique>. Be strict about correctness and side effects.\n\n"
            f"Failure analysis:\n{analysis.to_fix_prompt()}\n\n"
            "Original file:\n"
            f"{self._truncate(original_content, self.config.max_context_chars)}\n\n"
            "Proposed fix:\n"
            f"{self._truncate(proposed_fix, self.config.max_context_chars)}\n\n"
            f"Proposer rationale: {rationale}\n"
        )

        response = await self.agent.generate(prompt)
        verdict = (self._extract_tag(_VERDICT_RE, response) or "fail").lower()
        critique = self._extract_tag(_CRITIQUE_RE, response) or response.strip()
        is_ok = verdict.startswith("pass")
        return critique, is_ok

    async def synthesize_fixes(
        self,
        analysis: FailureAnalysis,
        proposals: list[tuple[str, str, float]],
        critiques: list[str],
    ) -> tuple[str, str, float]:
        """Synthesize multiple proposals into the best fix."""

        proposal_text = []
        for idx, (content, rationale, confidence) in enumerate(proposals, start=1):
            proposal_text.append(
                f"Proposal {idx} (confidence {confidence:.2f}):\n"
                f"Rationale: {rationale}\n"
                f"Content:\n{self._truncate(content, self.config.max_context_chars)}\n"
            )

        critique_text = "\n".join(f"- {c}" for c in critiques)

        prompt = (
            "Synthesize the best fix. Return ONLY the updated full file contents wrapped in "
            "<file>...</file>, plus <rationale>...</rationale> and <confidence>0-1</confidence>.\n\n"
            f"Failure analysis:\n{analysis.to_fix_prompt()}\n\n"
            "Proposals:\n" + "\n\n".join(proposal_text) + "\n\n"
            "Critiques:\n" + critique_text
        )

        response = await self.agent.generate(prompt)
        fixed_content = self._clean_file_output(response)
        rationale = self._extract_tag(_RATIONALE_RE, response) or ""
        confidence = self._parse_confidence(response, default=0.6)

        if not fixed_content:
            fixed_content = proposals[0][0] if proposals else ""
            rationale = rationale or "No synthesized content returned"
            confidence = min(confidence, 0.4)

        return fixed_content, rationale, confidence
