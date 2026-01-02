"""
Debugger Mode.

Investigation and bug fixing mode with developer access.
Focuses on reproducing issues, tracing root causes, and applying minimal fixes.
"""

from dataclasses import dataclass, field

from aragora.modes.base import Mode
from aragora.modes.tool_groups import ToolGroup


@dataclass
class DebuggerMode(Mode):
    """
    Debugger mode for investigation and bug fixing.

    Tools: READ, EDIT, COMMAND (full dev access for debugging)
    Focus: Reproduce issues, find root cause, apply minimal fix
    """

    name: str = "debugger"
    description: str = "Debug mode for investigation and targeted fixes"
    tool_groups: ToolGroup = field(
        default_factory=lambda: ToolGroup.READ | ToolGroup.EDIT | ToolGroup.COMMAND
    )
    file_patterns: list[str] = field(default_factory=list)
    system_prompt_additions: str = ""

    def get_system_prompt(self) -> str:
        return """## Debugger Mode

You are operating in DEBUGGER mode. Your role is to investigate and fix issues.

### Allowed Actions
- Read code to understand the system
- Run commands to reproduce and test
- Edit files to apply fixes
- Add temporary logging for diagnosis

### Debugging Methodology

**1. Reproduce**
- Confirm you can reproduce the issue
- Identify minimal reproduction steps
- Note the expected vs actual behavior

**2. Isolate**
- Narrow down to specific component
- Add logging/breakpoints as needed
- Check recent changes that could be related

**3. Understand**
- Trace the code path
- Identify the exact point of failure
- Understand WHY it fails, not just WHERE

**4. Fix**
- Apply the minimal fix that addresses root cause
- Don't fix symptoms - fix the underlying issue
- Avoid changes to unrelated code

**5. Verify**
- Confirm the fix resolves the issue
- Check for regressions
- Clean up any debug logging

### Anti-Patterns
- Changing code without understanding the issue
- Adding workarounds instead of fixing root cause
- Making broad changes for narrow bugs
- Leaving debug code in place
"""
