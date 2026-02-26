"""Configuration for the Swarm Commander system."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InterrogatorConfig:
    """Configuration for the SwarmInterrogator."""

    max_turns: int = 5
    model: str = "claude-sonnet-4-20250514"
    system_prompt: str = ""
    fallback_to_fixed_questions: bool = True

    def __post_init__(self) -> None:
        if not self.system_prompt:
            self.system_prompt = (
                "You are a project manager AI gathering requirements from a user "
                "who may not be a developer.\n"
                "Your job is to ask clear, specific questions to understand what "
                "they want built or changed.\n\n"
                "Rules:\n"
                "1. Ask ONE question at a time\n"
                "2. Use plain language, no jargon\n"
                "3. Focus on these areas (in order):\n"
                "   - WHAT: What should change? What's the desired outcome?\n"
                "   - WHY: Why is this needed? What problem does it solve?\n"
                "   - SCOPE: Which parts of the system are involved?\n"
                "   - ACCEPTANCE: How will we know it worked?\n"
                "   - CONSTRAINTS: What should NOT change? Any budget limits?\n"
                "4. When you have enough information (usually 3-5 questions), respond with exactly: SPEC_READY\n"
                "5. Never ask about implementation details -- that's the agents' job\n\n"
                "The project is Aragora, a multi-agent decision platform."
            )


@dataclass
class SwarmCommanderConfig:
    """Configuration for the SwarmCommander."""

    interrogator: InterrogatorConfig = field(default_factory=InterrogatorConfig)
    budget_limit_usd: float | None = 5.0
    require_approval: bool = False
    use_worktree_isolation: bool = True
    enable_meta_planning: bool = True
    enable_gauntlet_validation: bool = True
    enable_mode_enforcement: bool = True
    generate_receipts: bool = True
    spectate_stream: bool = True
    max_parallel_tasks: int = 4
    max_cycles: int = 5
