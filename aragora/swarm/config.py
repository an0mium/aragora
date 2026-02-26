"""Configuration for the Swarm Commander system."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InterrogatorConfig:
    """Configuration for the SwarmInterrogator."""

    max_turns: int = 8
    model: str = "claude-sonnet-4-20250514"
    system_prompt: str = ""
    fallback_to_fixed_questions: bool = True

    def __post_init__(self) -> None:
        if not self.system_prompt:
            self.system_prompt = (
                "You are a CTO having a conversation with your CEO. They're telling you "
                "what they want, and your job is to understand their vision well enough "
                "to make it happen.\n\n"
                "Rules:\n"
                "1. Ask ONE question at a time\n"
                "2. After each answer, briefly paraphrase what you heard to confirm "
                "understanding before asking the next question\n"
                "3. Explain any technical concepts in plain language. Example: "
                "\"We'd need to change the login page -- that's the screen you see "
                'when you first open the app."\n'
                "4. Be proactive: suggest what COULD be done, not just ask. Example: "
                '"Based on what you described, we could also add a password reset '
                'button -- would that be useful?"\n'
                "5. Focus on: WHAT (outcome), WHY (problem), SCOPE (which parts), "
                "ACCEPTANCE (how to know it worked), CONSTRAINTS (budget, don't-touch)\n"
                "6. When you have enough info (usually 3-5 questions), give a plain-language "
                "summary of everything you plan to do, then respond with: SPEC_READY\n"
                "7. Never use jargon without immediately explaining it\n"
                "8. Never ask about implementation details -- your engineering team "
                "handles those\n\n"
                "The project is Aragora, a multi-agent decision platform."
            )


@dataclass
class SwarmCommanderConfig:
    """Configuration for the SwarmCommander."""

    interrogator: InterrogatorConfig = field(default_factory=InterrogatorConfig)
    budget_limit_usd: float | None = 50.0
    require_approval: bool = False
    use_worktree_isolation: bool = True
    enable_meta_planning: bool = True
    enable_gauntlet_validation: bool = True
    enable_mode_enforcement: bool = True
    generate_receipts: bool = True
    spectate_stream: bool = True
    max_parallel_tasks: int = 20
    max_cycles: int = 5
    max_subtasks: int = 15
    max_parallel_branches: int = 16
    iterative_mode: bool = True
