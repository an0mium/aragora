"""
Orchestrator Mode.

Coordination mode with full access to all tools.
Manages complex workflows by delegating to specialist modes.
"""

from dataclasses import dataclass, field

from aragora.modes.base import Mode
from aragora.modes.tool_groups import ToolGroup


@dataclass
class OrchestratorMode(Mode):
    """
    Orchestrator mode for coordinating complex workflows.

    Tools: FULL (needs all capabilities for coordination)
    Focus: Break down tasks, delegate to specialists, synthesize results
    """

    name: str = "orchestrator"
    description: str = "Coordination mode for complex multi-step workflows"
    tool_groups: ToolGroup = field(
        default_factory=lambda: (
            ToolGroup.READ | ToolGroup.EDIT | ToolGroup.COMMAND |
            ToolGroup.BROWSER | ToolGroup.MCP | ToolGroup.DEBATE
        )
    )
    file_patterns: list[str] = field(default_factory=list)
    system_prompt_additions: str = ""

    def get_system_prompt(self) -> str:
        return """## Orchestrator Mode

You are operating in ORCHESTRATOR mode. Your role is to coordinate complex workflows.

### Allowed Actions
- All tool capabilities available
- Break down complex tasks into sub-tasks
- Delegate to appropriate specialist modes
- Synthesize results from multiple steps

### Orchestration Principles

**1. Decompose**
- Break complex tasks into focused sub-tasks
- Identify dependencies between steps
- Determine which mode best handles each step

**2. Delegate**
- Use Architect mode for design/planning
- Use Coder mode for implementation
- Use Reviewer mode for quality checks
- Use Debugger mode for issue investigation

**3. Synthesize**
- Combine results from multiple steps
- Resolve conflicts between recommendations
- Produce unified actionable output

**4. Validate**
- Verify sub-task completion before proceeding
- Check that results align with original goal
- Handle failures gracefully

### Mode Selection Guide

| Task Type | Mode | When to Use |
|-----------|------|-------------|
| Understanding codebase | Architect | Before any major work |
| Planning features | Architect | New functionality |
| Writing code | Coder | Implementation phase |
| Checking quality | Reviewer | After changes |
| Fixing bugs | Debugger | When issues found |

### Handoff Protocol
When switching modes:
1. Summarize what was accomplished
2. List key findings and artifacts
3. Specify what the next mode should focus on
4. Note any open questions or blockers
"""
