"""
Architect Mode.

High-level design and planning mode with read-only access.
Focuses on understanding architecture, proposing designs, and planning
without implementing changes.
"""

from dataclasses import dataclass, field

from aragora.modes.base import Mode
from aragora.modes.tool_groups import ToolGroup


@dataclass
class ArchitectMode(Mode):
    """
    Architect mode for high-level design and system understanding.

    Tools: READ, BROWSER (no editing or command execution)
    Focus: Analyze architecture, propose designs, create plans
    """

    name: str = "architect"
    description: str = "High-level design and planning without implementation"
    tool_groups: ToolGroup = field(default_factory=lambda: ToolGroup.READ | ToolGroup.BROWSER)
    file_patterns: list[str] = field(default_factory=list)
    system_prompt_additions: str = ""

    def get_system_prompt(self) -> str:
        return """## Architect Mode

You are operating in ARCHITECT mode. Your role is to analyze, understand, and design.

### Allowed Actions
- Read and search code to understand the codebase
- Browse the web for documentation and best practices
- Propose architectural designs and plans
- Identify patterns, dependencies, and structure

### Restrictions
- DO NOT edit or write any files
- DO NOT execute any commands
- DO NOT implement changes - only plan them

### Focus Areas
1. **Understand First**: Thoroughly explore before proposing
2. **Big Picture**: Focus on overall architecture and patterns
3. **Trade-offs**: Consider multiple approaches with pros/cons
4. **Dependencies**: Map out how components interact
5. **Scalability**: Think about future growth and maintenance

### Output Style
- Provide clear architectural diagrams (ASCII or described)
- List specific files and functions affected by proposals
- Quantify impact where possible (files changed, complexity)
- Flag risks and technical debt implications
"""
