"""
Centralized event type constants for Spectator Mode.
Prevents typos and ensures consistency across the codebase.
"""


class SpectatorEvents:
    """String constants for all spectator event types."""

    # Debate lifecycle
    DEBATE_START = "debate_start"
    DEBATE_END = "debate_end"

    # Round lifecycle
    ROUND_START = "round_start"
    ROUND_END = "round_end"

    # Agent actions
    PROPOSAL = "proposal"
    CRITIQUE = "critique"
    REFINE = "refine"
    VOTE = "vote"
    JUDGE = "judge"

    # Consensus/convergence
    CONSENSUS = "consensus"
    CONVERGENCE = "convergence"
    CONVERGED = "converged"
    EARLY_STOP = "early_stop"

    # Memory/Learning
    MEMORY_RECALL = "memory_recall"

    # Human-in-the-loop breakpoints
    BREAKPOINT = "breakpoint"
    BREAKPOINT_RESOLVED = "breakpoint_resolved"

    # Pipeline lifecycle
    PIPELINE_STARTED = "pipeline.started"
    PIPELINE_STAGE_STARTED = "pipeline.stage_started"
    PIPELINE_STAGE_COMPLETED = "pipeline.stage_completed"
    PIPELINE_COMPLETED = "pipeline.completed"
    PIPELINE_FAILED = "pipeline.failed"

    # System
    SYSTEM = "system"
    ERROR = "error"


# Icon and color mappings for visual styling
# Format: (emoji_icon, ansi_color_code)
EVENT_STYLES: dict[str, tuple[str, str]] = {
    SpectatorEvents.DEBATE_START: ("🎬", "\033[95m"),  # Magenta
    SpectatorEvents.DEBATE_END: ("🏁", "\033[95m"),
    SpectatorEvents.ROUND_START: ("⏱️", "\033[96m"),  # Cyan
    SpectatorEvents.ROUND_END: ("✓", "\033[96m"),
    SpectatorEvents.PROPOSAL: ("💡", "\033[94m"),  # Blue
    SpectatorEvents.CRITIQUE: ("🔍", "\033[91m"),  # Red
    SpectatorEvents.REFINE: ("✨", "\033[94m"),
    SpectatorEvents.VOTE: ("🗳️", "\033[93m"),  # Yellow
    SpectatorEvents.JUDGE: ("⚖️", "\033[93m"),
    SpectatorEvents.CONSENSUS: ("🤝", "\033[92m"),  # Green
    SpectatorEvents.CONVERGENCE: ("📊", "\033[92m"),
    SpectatorEvents.CONVERGED: ("🎉", "\033[92m"),
    SpectatorEvents.EARLY_STOP: ("⏹️", "\033[93m"),  # Yellow - early termination
    SpectatorEvents.MEMORY_RECALL: ("🧠", "\033[94m"),  # Blue - memory retrieval
    SpectatorEvents.BREAKPOINT: ("⚠️", "\033[33m"),  # Yellow/orange - needs attention
    SpectatorEvents.BREAKPOINT_RESOLVED: ("✅", "\033[32m"),  # Green - resolved
    SpectatorEvents.PIPELINE_STARTED: ("🚀", "\033[95m"),  # Magenta - pipeline kickoff
    SpectatorEvents.PIPELINE_STAGE_STARTED: ("▶️", "\033[96m"),  # Cyan - stage begin
    SpectatorEvents.PIPELINE_STAGE_COMPLETED: ("✅", "\033[92m"),  # Green - stage done
    SpectatorEvents.PIPELINE_COMPLETED: ("🏁", "\033[92m"),  # Green - pipeline done
    SpectatorEvents.PIPELINE_FAILED: ("❌", "\033[91m"),  # Red - pipeline failure
    SpectatorEvents.SYSTEM: ("⚙️", "\033[0m"),
    SpectatorEvents.ERROR: ("❌", "\033[91m"),
}

# ASCII fallbacks for non-UTF8 environments
EVENT_ASCII: dict[str, str] = {
    SpectatorEvents.DEBATE_START: "[START]",
    SpectatorEvents.DEBATE_END: "[END]",
    SpectatorEvents.ROUND_START: "[ROUND]",
    SpectatorEvents.ROUND_END: "[/ROUND]",
    SpectatorEvents.PROPOSAL: "[PROPOSE]",
    SpectatorEvents.CRITIQUE: "[CRITIQUE]",
    SpectatorEvents.REFINE: "[REFINE]",
    SpectatorEvents.VOTE: "[VOTE]",
    SpectatorEvents.JUDGE: "[JUDGE]",
    SpectatorEvents.CONSENSUS: "[CONSENSUS]",
    SpectatorEvents.CONVERGENCE: "[CONVERGE]",
    SpectatorEvents.CONVERGED: "[DONE]",
    SpectatorEvents.EARLY_STOP: "[EARLY_STOP]",
    SpectatorEvents.MEMORY_RECALL: "[MEMORY]",
    SpectatorEvents.BREAKPOINT: "[BREAK]",
    SpectatorEvents.BREAKPOINT_RESOLVED: "[RESOLVED]",
    SpectatorEvents.PIPELINE_STARTED: "[PIPE_START]",
    SpectatorEvents.PIPELINE_STAGE_STARTED: "[STAGE_START]",
    SpectatorEvents.PIPELINE_STAGE_COMPLETED: "[STAGE_DONE]",
    SpectatorEvents.PIPELINE_COMPLETED: "[PIPE_DONE]",
    SpectatorEvents.PIPELINE_FAILED: "[PIPE_FAIL]",
    SpectatorEvents.SYSTEM: "[SYS]",
    SpectatorEvents.ERROR: "[ERR]",
}
