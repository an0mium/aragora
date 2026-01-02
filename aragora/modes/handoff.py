"""
Mode Handoff System.

Provides stateless context transfer between modes, enabling
smooth transitions while preserving essential information.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class HandoffContext:
    """
    Context transferred between modes during a handoff.

    Contains a summary of work done and important state to preserve.
    """

    from_mode: str
    to_mode: str
    task_summary: str
    key_findings: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_prompt(self) -> str:
        """Convert handoff context to a prompt for the next mode."""
        lines = [
            f"## Handoff from {self.from_mode} to {self.to_mode}",
            "",
            "### Task Summary",
            self.task_summary,
            "",
        ]

        if self.key_findings:
            lines.append("### Key Findings")
            for finding in self.key_findings:
                lines.append(f"- {finding}")
            lines.append("")

        if self.files_touched:
            lines.append("### Files Touched")
            for file in self.files_touched:
                lines.append(f"- `{file}`")
            lines.append("")

        if self.open_questions:
            lines.append("### Open Questions")
            for question in self.open_questions:
                lines.append(f"- {question}")
            lines.append("")

        return "\n".join(lines)


class ModeHandoff:
    """
    Manages transitions between operational modes.

    Provides utilities for creating handoff summaries and generating
    transition prompts.
    """

    def __init__(self):
        self.history: list[HandoffContext] = []

    def create_context(
        self,
        from_mode: str,
        to_mode: str,
        task_summary: str,
        key_findings: list[str] | None = None,
        files_touched: list[str] | None = None,
        open_questions: list[str] | None = None,
        artifacts: dict[str, Any] | None = None,
    ) -> HandoffContext:
        """Create a handoff context for mode transition."""
        context = HandoffContext(
            from_mode=from_mode,
            to_mode=to_mode,
            task_summary=task_summary,
            key_findings=key_findings or [],
            files_touched=files_touched or [],
            open_questions=open_questions or [],
            artifacts=artifacts or {},
        )
        self.history.append(context)
        return context

    def generate_transition_prompt(
        self,
        context: HandoffContext,
        target_mode_prompt: str,
    ) -> str:
        """
        Generate a full transition prompt combining handoff context
        with the target mode's system prompt.
        """
        return f"""{target_mode_prompt}

---

{context.to_prompt()}

Continue from where the previous mode left off. Address any open questions
and build on the key findings.
"""

    def get_history(self) -> list[HandoffContext]:
        """Get the full handoff history for this session."""
        return self.history.copy()

    def summarize_session(self) -> str:
        """Generate a summary of all mode transitions in this session."""
        if not self.history:
            return "No mode transitions recorded."

        lines = ["## Session Mode History", ""]

        for i, ctx in enumerate(self.history, 1):
            timestamp = ctx.timestamp.strftime("%H:%M:%S")
            lines.append(f"{i}. [{timestamp}] {ctx.from_mode} -> {ctx.to_mode}")
            lines.append(f"   {ctx.task_summary[:80]}...")
            if ctx.files_touched:
                lines.append(f"   Files: {len(ctx.files_touched)}")
            lines.append("")

        return "\n".join(lines)
