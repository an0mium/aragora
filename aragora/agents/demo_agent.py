"""
Built-in demo agent for offline demos and CI smoke tests.
"""

from __future__ import annotations

from aragora.agents.registry import AgentRegistry
from aragora.core import Agent, Critique, Message, Vote


_TASK_LABELS = ("Task:", "Original Task:")
_MAX_TASK_LEN = 140

_PROPOSAL_VARIANTS = (
    {
        "title": "Baseline plan",
        "points": (
            "Define the scope and success metrics up front.",
            "Start with a minimal design and iterate with feedback.",
            "Add monitoring and rollback from day one.",
            "Document assumptions and open questions.",
        ),
    },
    {
        "title": "Risk-first plan",
        "points": (
            "List failure modes and mitigation strategies.",
            "Add guardrails and safe defaults early.",
            "Roll out in stages with clear exit criteria.",
            "Instrument for latency, error rate, and cost.",
        ),
    },
    {
        "title": "Speed-first plan",
        "points": (
            "Ship a small slice to validate the approach quickly.",
            "Automate tests and observability before scaling.",
            "Prefer simple components over complex dependencies.",
            "Schedule a post-launch hardening pass.",
        ),
    },
)

_CRITIQUE_ISSUES = (
    "Success metrics are underspecified.",
    "Failure modes and rollback paths are missing.",
    "Operational ownership and runbooks are unclear.",
)

_CRITIQUE_SUGGESTIONS = (
    "Define measurable SLOs and acceptance criteria.",
    "Add a staged rollout with automated rollback checks.",
    "Clarify on-call ownership and escalation paths.",
)

_SYNTHESIS_POINTS = (
    "Combine a minimal baseline with explicit risk guardrails.",
    "Make metrics and rollback criteria first-class requirements.",
    "Ship in phases and revisit assumptions after initial data.",
)


def _extract_task(prompt: str) -> str:
    for label in _TASK_LABELS:
        idx = prompt.find(label)
        if idx == -1:
            continue
        line = prompt[idx + len(label):].strip()
        line = line.splitlines()[0].strip() if line else ""
        if line:
            return _truncate(line, _MAX_TASK_LEN)
    return "the task"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _seed_from_name(name: str) -> int:
    return sum(ord(ch) for ch in name) or 1


@AgentRegistry.register(
    "demo",
    default_model="demo",
    agent_type="Built-in",
    description="Built-in demo agent (no external dependencies)",
)
class DemoAgent(Agent):
    """Deterministic agent for demos and smoke tests."""

    def __init__(self, name: str, role: str = "proposer", model: str = "demo"):
        super().__init__(name=name, model=model, role=role)
        self.agent_type = "demo"
        self._variant = _seed_from_name(name) % len(_PROPOSAL_VARIANTS)

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        task = _extract_task(prompt)
        lower = prompt.lower()

        if "choice:" in lower:
            return (
                f"CHOICE: {self.name}\n"
                "CONFIDENCE: 0.6\n"
                "CONTINUE: no\n"
                "REASONING: Demo vote to keep the flow moving."
            )

        if "synthes" in lower or self.role == "synthesizer":
            return self._synthesis_response(task)

        if "critique" in lower or "issues" in lower or self.role == "critic":
            return self._critic_response(task)

        return self._proposal_response(task)

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list[Message] | None = None,
    ) -> Critique:
        issues = list(_CRITIQUE_ISSUES)
        suggestions = list(_CRITIQUE_SUGGESTIONS)
        return Critique(
            agent=self.name,
            target_agent="proposal",
            target_content=_truncate(proposal, 200),
            issues=issues,
            suggestions=suggestions,
            severity=0.4,
            reasoning=f"Demo critique for: {_truncate(task, 80)}",
        )

    async def vote(self, proposals: dict[str, str], task: str) -> Vote:
        if proposals:
            choice = sorted(proposals.keys())[0]
        else:
            choice = self.name
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning="Demo vote (deterministic ordering).",
            confidence=0.6,
            continue_debate=False,
        )

    def _proposal_response(self, task: str) -> str:
        variant = _PROPOSAL_VARIANTS[self._variant]
        points = "\n".join(f"- {point}" for point in variant["points"])
        return (
            f"Demo proposal ({variant['title']}) for: {task}\n\n"
            f"{points}\n\n"
            "Risks:\n"
            "- Scope creep\n"
            "- Missing observability\n"
            "- Unclear ownership"
        )

    def _critic_response(self, task: str) -> str:
        issues = "\n".join(f"- {issue}" for issue in _CRITIQUE_ISSUES)
        suggestions = "\n".join(f"- {suggestion}" for suggestion in _CRITIQUE_SUGGESTIONS)
        return (
            f"Demo critique for: {task}\n\n"
            "Issues:\n"
            f"{issues}\n\n"
            "Suggestions:\n"
            f"{suggestions}\n"
            "Severity: 0.4"
        )

    def _synthesis_response(self, task: str) -> str:
        points = "\n".join(f"- {point}" for point in _SYNTHESIS_POINTS)
        return (
            f"Demo synthesis for: {task}\n\n"
            f"{points}\n\n"
            "Decision: Proceed with a phased rollout and explicit success metrics."
        )


__all__ = ["DemoAgent"]
