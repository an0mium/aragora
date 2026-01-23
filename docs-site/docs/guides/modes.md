---
title: Aragora Modes Guide
description: Aragora Modes Guide
---

# Aragora Modes Guide

> **Last Updated:** 2026-01-18

Operational modes for focused AI agent behavior. Modes control what tools an agent can use and provide specialized system prompts for different tasks.

## Related Documentation

| Document | Purpose |
|----------|---------|
| **MODES_GUIDE.md** (this) | Operational modes (Architect, Coder, etc.) |
| [MODES_REFERENCE.md](./modes-reference) | Debate modes (RedTeam, Prober, Audit) |
| [GAUNTLET.md](./gauntlet) | Comprehensive stress-testing |
| [PROBE_STRATEGIES.md](./probe-strategies) | Probing strategies reference |

## Overview

Modes are inspired by Kilocode's multi-mode architecture. Each mode defines:
- **Tool Permissions**: Which tools the agent can access
- **File Patterns**: Optional file access restrictions
- **System Prompt**: Behavioral guidelines for the mode

## Available Modes

| Mode | Tools | Use Case |
|------|-------|----------|
| **Architect** | Read, Browser | Design and planning |
| **Coder** | Read, Edit, Command | Implementation |
| **Debugger** | Read, Edit, Command | Bug investigation |
| **Reviewer** | Read, Browser | Code review |
| **Orchestrator** | Full access | Complex workflows |

---

## Architect Mode

**Purpose**: High-level design and planning without implementation.

**Tools**: `READ`, `BROWSER` (read-only)

**Best For**:
- Understanding codebase architecture
- Designing new features or systems
- Planning refactoring efforts
- Analyzing dependencies and patterns

### Behavior

In Architect mode, the agent:
- Reads and searches code to understand the codebase
- Browses the web for documentation and best practices
- Proposes architectural designs and plans
- Identifies patterns, dependencies, and structure

**Restrictions**:
- Cannot edit or write files
- Cannot execute commands
- Can only plan changes, not implement them

### Example Usage

```python
from aragora.modes.builtin import ArchitectMode
from aragora.modes.base import ModeRegistry

# Activate architect mode
mode = ArchitectMode()
agent.set_mode(mode)

# Now agent will only read/browse, not edit
response = agent.run("Analyze the authentication system architecture")
```

### Output Style

Architect mode produces:
- ASCII architectural diagrams
- Lists of specific files and functions affected
- Quantified impact assessments (files changed, complexity)
- Risk and technical debt flags

---

## Coder Mode

**Purpose**: Implementation with full development capabilities.

**Tools**: `READ`, `EDIT`, `COMMAND`

**Best For**:
- Writing new features
- Implementing bug fixes
- Refactoring code
- Creating tests

### Behavior

In Coder mode, the agent:
- Reads code to understand context
- Edits and writes files
- Runs commands for testing and validation
- Creates new files when necessary

### Guidelines

1. **Follow Patterns**: Match existing code style
2. **Minimal Changes**: Only change what's needed
3. **Quality First**: Write clean, maintainable code
4. **Test as You Go**: Verify changes work
5. **No Over-engineering**: Avoid premature abstraction

### Anti-Patterns to Avoid

- Adding unrequested features
- Refactoring unrelated code
- Creating unnecessary abstractions
- Ignoring existing error handling patterns

### Example Usage

```python
from aragora.modes.builtin import CoderMode

mode = CoderMode()
agent.set_mode(mode)

# Agent can now read, edit, and run commands
response = agent.run("Implement the user registration endpoint")
```

---

## Debugger Mode

**Purpose**: Investigation and bug fixing.

**Tools**: `READ`, `EDIT`, `COMMAND`

**Best For**:
- Reproducing issues
- Tracing root causes
- Applying minimal fixes
- Understanding unexpected behavior

### Debugging Methodology

**1. Reproduce**
- Confirm the issue can be reproduced
- Identify minimal reproduction steps
- Note expected vs actual behavior

**2. Isolate**
- Narrow down to specific component
- Add logging/breakpoints as needed
- Check recent related changes

**3. Understand**
- Trace the code path
- Identify the exact point of failure
- Understand WHY it fails, not just WHERE

**4. Fix**
- Apply minimal fix for root cause
- Don't fix symptoms - fix underlying issue
- Avoid changes to unrelated code

**5. Verify**
- Confirm fix resolves the issue
- Check for regressions
- Clean up debug logging

### Anti-Patterns

- Changing code without understanding
- Adding workarounds instead of fixes
- Making broad changes for narrow bugs
- Leaving debug code in place

### Example Usage

```python
from aragora.modes.builtin import DebuggerMode

mode = DebuggerMode()
agent.set_mode(mode)

response = agent.run("""
Bug: Users can't log in after password reset.
Error: 'Invalid token' on first login attempt.
Steps: 1. Reset password 2. Log in with new password
""")
```

---

## Reviewer Mode

**Purpose**: Code review and quality analysis.

**Tools**: `READ`, `BROWSER` (read-only)

**Best For**:
- Code review
- Security audits
- Performance analysis
- Quality assessments

### Review Checklist

**Correctness**
- Logic errors and edge cases
- Off-by-one errors
- Null/undefined handling
- Type safety issues

**Security**
- Input validation
- Injection vulnerabilities (SQL, XSS, command)
- Authentication/authorization gaps
- Secrets in code

**Performance**
- Unnecessary loops or allocations
- Missing caching opportunities
- Database query efficiency
- Memory leaks

**Maintainability**
- Code clarity and naming
- Appropriate abstraction level
- Missing or misleading comments
- Test coverage gaps

### Output Format

For each issue, Reviewer mode reports:
1. **Location**: File and line number
2. **Severity**: Critical / High / Medium / Low / Suggestion
3. **Issue**: Clear description
4. **Why**: Impact explanation
5. **Fix**: Suggested solution (conceptual)

### Example Usage

```python
from aragora.modes.builtin import ReviewerMode

mode = ReviewerMode()
agent.set_mode(mode)

response = agent.run("Review the authentication handler at auth.py")
```

---

## Orchestrator Mode

**Purpose**: Coordinate complex multi-step workflows.

**Tools**: `FULL` (Read, Edit, Command, Browser, MCP, Debate)

**Best For**:
- Complex multi-step tasks
- Workflows requiring multiple modes
- Tasks with dependencies
- Synthesis of multiple results

### Orchestration Principles

**1. Decompose**
- Break complex tasks into focused sub-tasks
- Identify dependencies between steps
- Determine which mode best handles each step

**2. Delegate**
| Task Type | Mode |
|-----------|------|
| Understanding codebase | Architect |
| Planning features | Architect |
| Writing code | Coder |
| Checking quality | Reviewer |
| Fixing bugs | Debugger |

**3. Synthesize**
- Combine results from multiple steps
- Resolve conflicts between recommendations
- Produce unified actionable output

**4. Validate**
- Verify sub-task completion before proceeding
- Check that results align with original goal
- Handle failures gracefully

### Handoff Protocol

When switching modes:
1. Summarize what was accomplished
2. List key findings and artifacts
3. Specify what the next mode should focus on
4. Note any open questions or blockers

### Example Usage

```python
from aragora.modes.builtin import OrchestratorMode

mode = OrchestratorMode()
agent.set_mode(mode)

response = agent.run("""
Build a user notification system:
1. Design the architecture
2. Implement the backend
3. Review for security issues
4. Fix any problems found
""")
```

---

## Tool Groups

Modes use tool groups to define permissions:

| Group | Capabilities |
|-------|--------------|
| `READ` | Read files, glob, grep |
| `EDIT` | Edit, write files |
| `COMMAND` | Execute shell commands |
| `BROWSER` | Web fetch, web search |
| `MCP` | MCP server tools |
| `DEBATE` | Debate participation |

### Composite Groups

```python
from aragora.modes.tool_groups import ToolGroup

# Read-only access with web browsing
READONLY = ToolGroup.READ | ToolGroup.BROWSER

# Standard development access
DEVELOPER = ToolGroup.READ | ToolGroup.EDIT | ToolGroup.COMMAND

# Full access to all tools
FULL = ToolGroup.READ | ToolGroup.EDIT | ToolGroup.COMMAND | ToolGroup.BROWSER | ToolGroup.MCP | ToolGroup.DEBATE
```

---

## Creating Custom Modes

You can create custom modes by extending the `Mode` base class:

```python
from dataclasses import dataclass, field
from aragora.modes.base import Mode
from aragora.modes.tool_groups import ToolGroup

@dataclass
class SecurityAuditorMode(Mode):
    """Custom mode for security-focused analysis."""

    name: str = "security_auditor"
    description: str = "Security audit mode with read-only access"
    tool_groups: ToolGroup = field(
        default_factory=lambda: ToolGroup.READ | ToolGroup.BROWSER
    )
    file_patterns: list[str] = field(default_factory=list)

    def get_system_prompt(self) -> str:
        return """## Security Auditor Mode

You are operating in SECURITY AUDITOR mode. Focus on:
- Finding security vulnerabilities
- Checking for OWASP Top 10 issues
- Identifying authentication/authorization gaps
- Reviewing input validation

DO NOT edit files - provide a detailed security report.
"""
```

### File Pattern Restrictions

Restrict mode to specific files:

```python
@dataclass
class TestOnlyMode(Mode):
    name: str = "test_only"
    description: str = "Only access test files"
    tool_groups: ToolGroup = field(
        default_factory=lambda: ToolGroup.READ | ToolGroup.EDIT
    )
    file_patterns: list[str] = field(
        default_factory=lambda: ["**/test_*.py", "**/tests/**"]
    )
```

---

## Mode Registry

Modes auto-register when instantiated:

```python
from aragora.modes.base import ModeRegistry

# List all available modes
modes = ModeRegistry.list_all()
# ['architect', 'coder', 'debugger', 'reviewer', 'orchestrator']

# Get a specific mode
mode = ModeRegistry.get("architect")

# Get with error if not found
mode = ModeRegistry.get_or_raise("invalid")  # Raises KeyError
```

---

## Best Practices

### 1. Start with the Right Mode

| Starting Task | Recommended Mode |
|---------------|------------------|
| "Add feature X" | Architect → Coder |
| "Fix bug in Y" | Debugger |
| "Review PR #123" | Reviewer |
| "Build entire system" | Orchestrator |
| "Understand codebase" | Architect |

### 2. Mode Transitions

For complex tasks, transition through modes:

```
Architect (understand/plan)
    ↓
Coder (implement)
    ↓
Reviewer (verify quality)
    ↓
Debugger (if issues found)
    ↓
Coder (apply fixes)
```

### 3. Use Orchestrator for Complex Workflows

When a task requires multiple modes, use Orchestrator to coordinate:

```python
mode = OrchestratorMode()
agent.set_mode(mode)

# Orchestrator will internally delegate to appropriate modes
agent.run("Add authentication to the API")
```

### 4. Read-Only First

When uncertain, start with read-only modes (Architect, Reviewer) to understand before modifying.

---

## See Also

- [PROBE_STRATEGIES.md](./probe-strategies) - Red-teaming and capability testing
- [AGENT_SELECTION.md](../core-concepts/agent-selection) - Choosing agents for debates
- [ARCHITECTURE.md](../core-concepts/architecture) - System architecture overview
