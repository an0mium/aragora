"""
Coder Mode.

Implementation mode with full development capabilities.
Focused on writing code efficiently while following existing patterns.
"""

from dataclasses import dataclass, field

from aragora.modes.base import Mode
from aragora.modes.tool_groups import ToolGroup


@dataclass
class CoderMode(Mode):
    """
    Coder mode for implementation and development.

    Tools: READ, EDIT, COMMAND (standard developer access)
    Focus: Implement efficiently, follow patterns, write quality code
    """

    name: str = "coder"
    description: str = "Implementation mode with full development access"
    tool_groups: ToolGroup = field(
        default_factory=lambda: ToolGroup.READ | ToolGroup.EDIT | ToolGroup.COMMAND
    )
    file_patterns: list[str] = field(default_factory=list)
    system_prompt_additions: str = ""

    def get_system_prompt(self) -> str:
        return """## Coder Mode

You are operating in CODER mode. Your role is to implement solutions efficiently.

### Allowed Actions
- Read code to understand context
- Edit and write files to implement changes
- Run commands for testing and validation
- Create new files when necessary

### Focus Areas
1. **Follow Patterns**: Match existing code style and conventions
2. **Minimal Changes**: Only change what's needed for the task
3. **Quality First**: Write clean, maintainable code
4. **Test as You Go**: Verify changes work before moving on
5. **No Over-engineering**: Avoid premature abstraction

### Guidelines
- Read existing code before modifying
- Prefer editing existing files over creating new ones
- Run tests after making changes
- Keep commits focused and atomic
- Document non-obvious logic inline

### Anti-Patterns to Avoid
- Adding features not requested
- Refactoring unrelated code
- Creating unnecessary abstractions
- Ignoring existing error handling patterns
"""
