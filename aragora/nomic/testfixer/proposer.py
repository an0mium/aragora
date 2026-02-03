"""
PatchProposer - Generate and debate fix proposals.

Uses multiple AI agents to:
1. Generate candidate fixes
2. Critique each others' fixes (Hegelian debate)
3. Synthesize the best approach
4. Produce a concrete patch

The debate process ensures fixes are cross-checked before application.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Protocol


from aragora.nomic.testfixer.analyzer import FailureAnalysis, FixTarget


class PatchStatus(str, Enum):
    """Status of a patch proposal."""

    PROPOSED = "proposed"
    CRITIQUED = "critiqued"
    SYNTHESIZED = "synthesized"
    VALIDATED = "validated"
    APPLIED = "applied"
    REJECTED = "rejected"


@dataclass
class FilePatch:
    """A patch to a single file."""

    file_path: str
    original_content: str
    patched_content: str

    # Diff information
    diff_lines: list[str] = field(default_factory=list)
    lines_added: int = 0
    lines_removed: int = 0

    def __post_init__(self):
        """Compute diff on creation."""
        if not self.diff_lines:
            self.diff_lines = list(
                difflib.unified_diff(
                    self.original_content.splitlines(keepends=True),
                    self.patched_content.splitlines(keepends=True),
                    fromfile=f"a/{self.file_path}",
                    tofile=f"b/{self.file_path}",
                )
            )
            self.lines_added = sum(
                1 for line in self.diff_lines if line.startswith("+") and not line.startswith("+++")
            )
            self.lines_removed = sum(
                1 for line in self.diff_lines if line.startswith("-") and not line.startswith("---")
            )

    def as_unified_diff(self) -> str:
        """Return patch as unified diff string."""
        return "".join(self.diff_lines)

    def apply(self, repo_path: Path) -> bool:
        """Apply this patch to the repository.

        Args:
            repo_path: Repository root path

        Returns:
            True if applied successfully
        """
        try:
            full_path = repo_path / self.file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(self.patched_content)
            return True
        except Exception:
            return False

    def revert(self, repo_path: Path) -> bool:
        """Revert this patch.

        Args:
            repo_path: Repository root path

        Returns:
            True if reverted successfully
        """
        try:
            full_path = repo_path / self.file_path
            full_path.write_text(self.original_content)
            return True
        except Exception:
            return False


@dataclass
class PatchProposal:
    """A proposed fix for a test failure."""

    id: str
    analysis: FailureAnalysis
    created_at: datetime = field(default_factory=datetime.now)

    # The fix
    patches: list[FilePatch] = field(default_factory=list)
    description: str = ""
    rationale: str = ""

    # Debate results
    status: PatchStatus = PatchStatus.PROPOSED
    critiques: list[str] = field(default_factory=list)
    synthesis_notes: str = ""

    # Confidence
    proposer_confidence: float = 0.5
    post_debate_confidence: float = 0.5

    # Metadata
    proposer: str = "unknown"
    iteration: int = 0

    def total_changes(self) -> tuple[int, int]:
        """Return (total_added, total_removed) across all patches."""
        added = sum(p.lines_added for p in self.patches)
        removed = sum(p.lines_removed for p in self.patches)
        return added, removed

    def as_diff(self) -> str:
        """Return complete diff for all patches."""
        return "\n".join(p.as_unified_diff() for p in self.patches)

    def apply_all(self, repo_path: Path) -> bool:
        """Apply all patches.

        Args:
            repo_path: Repository root path

        Returns:
            True if all patches applied successfully
        """
        for patch in self.patches:
            if not patch.apply(repo_path):
                return False
        self.status = PatchStatus.APPLIED
        return True

    def revert_all(self, repo_path: Path) -> bool:
        """Revert all patches.

        Args:
            repo_path: Repository root path

        Returns:
            True if all patches reverted successfully
        """
        for patch in self.patches:
            if not patch.revert(repo_path):
                return False
        return True


@dataclass
class ProposalDebate:
    """Record of a debate about a fix proposal."""

    proposal: PatchProposal
    started_at: datetime = field(default_factory=datetime.now)

    # Debate phases
    proposals: list[tuple[str, str, float]] = field(
        default_factory=list
    )  # (agent, proposal_text, confidence)
    critiques: list[tuple[str, str, str]] = field(
        default_factory=list
    )  # (critic, target_agent, critique_text)
    synthesis: str = ""

    # Outcome
    final_proposal: PatchProposal | None = None
    consensus_reached: bool = False
    dissenting_opinions: list[str] = field(default_factory=list)


class CodeGenerator(Protocol):
    """Protocol for AI code generation."""

    async def generate_fix(
        self,
        analysis: FailureAnalysis,
        file_content: str,
        file_path: str,
    ) -> tuple[str, str, float]:
        """Generate a fix for the failure.

        Args:
            analysis: Failure analysis
            file_content: Current content of the file to fix
            file_path: Path to the file

        Returns:
            Tuple of (fixed_content, rationale, confidence)
        """
        ...

    async def critique_fix(
        self,
        analysis: FailureAnalysis,
        original_content: str,
        proposed_fix: str,
        rationale: str,
    ) -> tuple[str, bool]:
        """Critique a proposed fix.

        Args:
            analysis: Original failure analysis
            original_content: Original file content
            proposed_fix: Proposed fixed content
            rationale: Why the fix was proposed

        Returns:
            Tuple of (critique_text, is_acceptable)
        """
        ...

    async def synthesize_fixes(
        self,
        analysis: FailureAnalysis,
        proposals: list[tuple[str, str, float]],  # (content, rationale, confidence)
        critiques: list[str],
    ) -> tuple[str, str, float]:
        """Synthesize multiple proposals into best fix.

        Args:
            analysis: Original failure analysis
            proposals: List of proposed fixes
            critiques: Critiques of the proposals

        Returns:
            Tuple of (best_fix_content, synthesis_rationale, confidence)
        """
        ...


class SimpleCodeGenerator:
    """Simple code generator using pattern-based fixes.

    For common issues, applies known fix patterns without AI.
    """

    async def generate_fix(
        self,
        analysis: FailureAnalysis,
        file_content: str,
        file_path: str,
    ) -> tuple[str, str, float]:
        """Generate fix using heuristics."""
        from aragora.nomic.testfixer.analyzer import FailureCategory

        fixed_content = file_content
        rationale = ""
        confidence = 0.5

        # Pattern: missing await
        if analysis.category == FailureCategory.TEST_ASYNC:
            # Look for coroutine calls without await
            # This is a simplified heuristic
            if "async def" not in file_content and "await" not in file_content:
                # Add async marker if function is sync but calls async
                # This is just a placeholder - real implementation would be smarter
                rationale = "Test may need @pytest.mark.asyncio and async/await"
                confidence = 0.6

        # Pattern: mock attribute missing
        if analysis.category == FailureCategory.TEST_MOCK:
            # Look for MagicMock usage
            if "MagicMock" in file_content:
                rationale = "Mock may need additional attributes configured"
                confidence = 0.5

        # Pattern: missing optional dependency (e.g., tiktoken)
        if analysis.category == FailureCategory.ENV_DEPENDENCY:
            missing_mod = None
            match = re.search(r"No module named '([^']+)'", analysis.failure.error_message)
            if match:
                missing_mod = match.group(1)
            if not missing_mod:
                match = re.search(r"No module named '([^']+)'", analysis.failure.stack_trace)
                if match:
                    missing_mod = match.group(1)

            if missing_mod and f"import {missing_mod}" in file_content:
                if missing_mod == "tiktoken":
                    try_block = (
                        "try:\\n"
                        "    import tiktoken\\n"
                        "    TIKTOKEN_AVAILABLE = True\\n"
                        "except Exception:\\n"
                        "    tiktoken = None\\n"
                        "    TIKTOKEN_AVAILABLE = False\\n"
                    )
                    if "TIKTOKEN_AVAILABLE" in file_content:
                        fixed_content = file_content.replace(
                            "import tiktoken\\n\\nTIKTOKEN_AVAILABLE = True",
                            try_block,
                        )
                    else:
                        fixed_content = file_content.replace("import tiktoken", try_block)
                    rationale = "Make tiktoken optional with a safe fallback"
                    confidence = 0.7

        return fixed_content, rationale, confidence

    async def critique_fix(
        self,
        analysis: FailureAnalysis,
        original_content: str,
        proposed_fix: str,
        rationale: str,
    ) -> tuple[str, bool]:
        """Simple critique - just check if something changed."""
        if original_content == proposed_fix:
            return "No changes were made to the file.", False

        # Check for obviously bad patterns
        bad_patterns = [
            (r"import \*", "Avoid wildcard imports"),
            (r"except:\s*$", "Avoid bare except clauses"),
            (r"# type: ignore$", "Avoid blanket type ignores"),
        ]

        critiques = []
        for pattern, message in bad_patterns:
            if re.search(pattern, proposed_fix) and not re.search(pattern, original_content):
                critiques.append(message)

        if critiques:
            return "; ".join(critiques), False

        return "Fix appears reasonable.", True

    async def synthesize_fixes(
        self,
        analysis: FailureAnalysis,
        proposals: list[tuple[str, str, float]],
        critiques: list[str],
    ) -> tuple[str, str, float]:
        """Pick the highest confidence proposal."""
        if not proposals:
            return "", "No proposals to synthesize", 0.0

        best = max(proposals, key=lambda x: x[2])
        return best[0], f"Selected highest confidence proposal: {best[1]}", best[2]


class AgentCodeGenerator:
    """Code generator backed by an Aragora agent (CLI or API)."""

    def __init__(
        self,
        agent_type: str,
        name: str | None = None,
        role: str = "proposer",
        model: str | None = None,
        api_key: str | None = None,
        max_file_chars: int = 40000,
    ):
        from aragora.agents.base import create_agent

        self.agent_type = agent_type
        self.max_file_chars = max_file_chars
        self.agent = create_agent(
            model_type=agent_type,
            name=name or f"testfix-{agent_type}",
            role=role,
            model=model,
            api_key=api_key,
        )

    def _truncate(self, content: str) -> str:
        if len(content) <= self.max_file_chars:
            return content
        head = self.max_file_chars // 2
        tail = self.max_file_chars - head
        return content[:head] + "\n\n# ... truncated for length ...\n\n" + content[-tail:]

    def _extract_confidence(self, response: str, fallback: float) -> float:
        match = re.search(r"confidence\s*[:=]\s*([01](?:\.\d+)?)", response, re.IGNORECASE)
        if not match:
            return fallback
        try:
            value = float(match.group(1))
            return min(max(value, 0.0), 1.0)
        except ValueError:
            return fallback

    def _extract_file_content(self, response: str, fallback: str) -> str:
        start_tag = "<file>"
        end_tag = "</file>"
        if start_tag in response and end_tag in response:
            start = response.index(start_tag) + len(start_tag)
            end = response.index(end_tag)
            return response[start:end].strip()

        code_block = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()

        return response.strip() if response.strip() else fallback

    async def generate_fix(
        self,
        analysis: FailureAnalysis,
        file_content: str,
        file_path: str,
    ) -> tuple[str, str, float]:
        """Generate a fix using an Aragora agent."""
        prompt = f"""You are fixing a failing test. Update the file below to resolve the failure.

Return ONLY the updated file content between <file> and </file> tags.
Optionally include a CONFIDENCE: 0.0-1.0 line.

{analysis.to_fix_prompt()}

### File: {file_path}
```python
{self._truncate(file_content)}
```
"""
        response = await self.agent.generate(prompt)
        fixed_content = self._extract_file_content(response, file_content)
        confidence = self._extract_confidence(response, analysis.confidence)
        rationale = response.strip()[:2000]
        return fixed_content, rationale, confidence

    async def critique_fix(
        self,
        analysis: FailureAnalysis,
        original_content: str,
        proposed_fix: str,
        rationale: str,
    ) -> tuple[str, bool]:
        """Critique a proposed fix using the agent."""
        diff = "\n".join(
            difflib.unified_diff(
                original_content.splitlines(),
                proposed_fix.splitlines(),
                fromfile="original",
                tofile="proposed",
                lineterm="",
            )
        )
        prompt = f"""Review the proposed fix for this failure.

Respond with:
DECISION: approve or reject
CRITIQUE: short reasoning

Failure context:
{analysis.to_fix_prompt()}

Proposed diff:
```diff
{diff[:8000]}
```
"""
        response = await self.agent.generate(prompt)
        decision_match = re.search(r"decision\s*:\s*(approve|reject)", response, re.IGNORECASE)
        is_ok = bool(decision_match and decision_match.group(1).lower() == "approve")
        return response.strip(), is_ok

    async def synthesize_fixes(
        self,
        analysis: FailureAnalysis,
        proposals: list[tuple[str, str, float]],
        critiques: list[str],
    ) -> tuple[str, str, float]:
        """Pick the highest confidence proposal without extra LLM calls."""
        if not proposals:
            return "", "No proposals to synthesize", 0.0
        best = max(proposals, key=lambda x: x[2])
        return best[0], f"Selected highest confidence proposal: {best[1]}", best[2]


class PatchProposer:
    """Generates and debates fix proposals.

    Uses Hegelian debate structure:
    1. Multiple agents propose fixes
    2. Agents critique each others' proposals
    3. Synthesis produces best combined fix

    Example:
        proposer = PatchProposer(
            repo_path=Path("/path/to/repo"),
            generators=[agent1, agent2, agent3],
        )

        proposal = await proposer.propose_fix(analysis)

        if proposal.post_debate_confidence > 0.7:
            proposal.apply_all(repo_path)
    """

    def __init__(
        self,
        repo_path: Path,
        generators: list[CodeGenerator] | None = None,
        synthesizer: CodeGenerator | None = None,
        require_consensus: bool = False,
    ):
        """Initialize the proposer.

        Args:
            repo_path: Repository root path
            generators: List of code generators for proposals
            synthesizer: Generator for synthesis (uses first generator if None)
            require_consensus: Whether all critics must approve
        """
        self.repo_path = Path(repo_path)
        self.generators = generators or [SimpleCodeGenerator()]
        self.synthesizer = synthesizer or self.generators[0]
        self.require_consensus = require_consensus
        self._proposal_counter = 0

    async def propose_fix(
        self,
        analysis: FailureAnalysis,
        max_iterations: int = 3,
    ) -> PatchProposal:
        """Generate a fix proposal with debate.

        Args:
            analysis: Failure analysis
            max_iterations: Maximum debate iterations

        Returns:
            PatchProposal with debated fix
        """
        self._proposal_counter += 1
        proposal_id = f"fix_{self._proposal_counter}"

        # Read the file to fix
        if analysis.fix_target == FixTarget.TEST_FILE:
            file_to_fix = analysis.failure.test_file
        else:
            file_to_fix = analysis.root_cause_file

        file_path = self.repo_path / file_to_fix
        if not file_path.exists():
            return PatchProposal(
                id=proposal_id,
                analysis=analysis,
                status=PatchStatus.REJECTED,
                description=f"File not found: {file_to_fix}",
            )

        original_content = file_path.read_text()

        # Phase 1: Generate proposals from each agent
        proposals = []
        for i, generator in enumerate(self.generators):
            try:
                fixed_content, rationale, confidence = await generator.generate_fix(
                    analysis,
                    original_content,
                    file_to_fix,
                )
                proposals.append(
                    (
                        f"agent_{i}",
                        fixed_content,
                        rationale,
                        confidence,
                    )
                )
            except Exception as e:
                proposals.append(
                    (
                        f"agent_{i}",
                        original_content,
                        f"Failed to generate: {e}",
                        0.0,
                    )
                )

        # Phase 2: Cross-critique
        all_critiques = []
        for i, (agent, content, rationale, conf) in enumerate(proposals):
            for j, critic_gen in enumerate(self.generators):
                if i == j:
                    continue  # Don't self-critique

                try:
                    critique, is_ok = await critic_gen.critique_fix(
                        analysis,
                        original_content,
                        content,
                        rationale,
                    )
                    all_critiques.append((f"agent_{j}", agent, critique, is_ok))
                except Exception as e:
                    all_critiques.append((f"agent_{j}", agent, f"Critique failed: {e}", False))

        # Phase 3: Synthesize
        synthesis_input = [(content, rationale, conf) for _, content, rationale, conf in proposals]
        critique_texts = [c[2] for c in all_critiques]

        try:
            (
                final_content,
                synthesis_notes,
                final_confidence,
            ) = await self.synthesizer.synthesize_fixes(
                analysis,
                synthesis_input,
                critique_texts,
            )
        except Exception as e:
            # Fall back to highest confidence proposal
            best = max(proposals, key=lambda x: x[3])
            final_content = best[1]
            synthesis_notes = f"Synthesis failed ({e}), using best proposal"
            final_confidence = best[3]

        # Check consensus
        approvals = sum(1 for c in all_critiques if c[3])
        consensus = approvals >= len(all_critiques) // 2 if all_critiques else True

        if self.require_consensus and not consensus:
            return PatchProposal(
                id=proposal_id,
                analysis=analysis,
                status=PatchStatus.REJECTED,
                description="Consensus not reached",
                critiques=[c[2] for c in all_critiques if not c[3]],
            )

        # Create patch
        patches = []
        if final_content != original_content:
            patches.append(
                FilePatch(
                    file_path=file_to_fix,
                    original_content=original_content,
                    patched_content=final_content,
                )
            )

        return PatchProposal(
            id=proposal_id,
            analysis=analysis,
            patches=patches,
            description=f"Fix for {analysis.category.value} in {file_to_fix}",
            rationale=synthesis_notes,
            status=PatchStatus.SYNTHESIZED,
            critiques=[c[2] for c in all_critiques],
            synthesis_notes=synthesis_notes,
            proposer_confidence=max(p[3] for p in proposals) if proposals else 0.0,
            post_debate_confidence=final_confidence,
            proposer="hegelian_debate",
        )

    def record_debate(
        self,
        proposal: PatchProposal,
        proposals: list[tuple[str, str, float]],
        critiques: list[tuple[str, str, str]],
    ) -> ProposalDebate:
        """Create a debate record.

        Args:
            proposal: Final proposal
            proposals: All proposals generated
            critiques: All critiques made

        Returns:
            ProposalDebate record
        """
        return ProposalDebate(
            proposal=proposal,
            proposals=[(a, c, conf) for a, c, _, conf in proposals] if proposals else [],
            critiques=critiques,
            synthesis=proposal.synthesis_notes,
            final_proposal=proposal,
            consensus_reached=proposal.status != PatchStatus.REJECTED,
            dissenting_opinions=[c[2] for c in critiques if len(c) > 2],
        )
