---
title: Debate Modes Reference
description: Debate Modes Reference
---

# Debate Modes Reference

Aragora provides two complementary mode systems for controlling agent behavior and debate protocols.

## Overview

| System | Purpose | Examples |
|--------|---------|----------|
| **Operational Modes** | Tool access control | Architect, Coder, Reviewer |
| **Debate Modes** | Adversarial protocols | RedTeam, DeepAudit, Prober |

---

## Operational Modes

Kilocode-inspired modes that define what tools an agent can use.

### Built-in Modes

| Mode | Tool Groups | Purpose |
|------|-------------|---------|
| `ArchitectMode` | Read, Navigate | High-level design and planning |
| `CoderMode` | Read, Write, Navigate, Command | Implementation |
| `ReviewerMode` | Read, Navigate | Code review and analysis |
| `DebuggerMode` | Read, Navigate, Command | Debugging and troubleshooting |
| `OrchestratorMode` | All | Full system orchestration |

### Tool Groups

```python
from aragora.modes import ToolGroup

class ToolGroup(Flag):
    NONE = 0
    READ = auto()      # Read files
    WRITE = auto()     # Write/edit files
    NAVIGATE = auto()  # Search, glob, grep
    COMMAND = auto()   # Execute commands
    AGENT = auto()     # Spawn sub-agents
    MCP = auto()       # MCP server tools
    ALL = READ | WRITE | NAVIGATE | COMMAND | AGENT | MCP
```

### Usage

```python
from aragora.modes import ArchitectMode, CoderMode, ModeRegistry

# Get a mode
architect = ModeRegistry.get("architect")
coder = ModeRegistry.get("coder")

# Check tool access
architect.can_access_tool("read")    # True
architect.can_access_tool("write")   # False
coder.can_access_tool("write")       # True

# Get system prompt
prompt = architect.get_system_prompt()
```

### Custom Modes

Create custom modes via YAML:

```yaml
# custom_mode.yaml
name: security-reviewer
description: Security-focused code review
tool_groups:
  - read
  - navigate
file_patterns:
  - "**/*.py"
  - "**/*.js"
system_prompt_additions: |
  Focus on security vulnerabilities:
  - SQL injection
  - XSS
  - Authentication flaws
```

```python
from aragora.modes import CustomModeLoader

loader = CustomModeLoader()
mode = loader.load_from_file("custom_mode.yaml")
ModeRegistry.register(mode)
```

---

## Debate Modes

Specialized protocols for different types of agent debates.

### RedTeam Mode

Adversarial security testing with attack/defense rounds.

```python
from aragora.modes import RedTeamMode, AttackType

# Create red team debate
mode = RedTeamMode(
    attack_types=[AttackType.LOGICAL, AttackType.TECHNICAL],
    max_rounds=5,
)

# Run red team session
result = await mode.run_debate(
    target="Our authentication system uses JWT tokens...",
    attacker_agent=attacker,
    defender_agent=defender,
)

print(result.vulnerabilities)  # List of found issues
print(result.overall_score)    # 0.0 - 1.0 security score
```

**Attack Types:**
- `LOGICAL` - Reasoning flaws, contradictions
- `TECHNICAL` - Implementation issues
- `ETHICAL` - Policy violations
- `BOUNDARY` - Edge case handling
- `PROMPT_INJECTION` - Prompt security

### Prober Mode

Capability probing to test agent weaknesses.

```python
from aragora.modes import (
    CapabilityProber,
    ContradictionTrap,
    HallucinationBait,
    SycophancyTest,
)

prober = CapabilityProber(
    strategies=[
        ContradictionTrap(),      # Test for self-contradiction
        HallucinationBait(),      # Test for hallucination
        SycophancyTest(),         # Test for agreement bias
    ]
)

# Probe an agent
report = await prober.probe_agent(agent, max_probes=10)

print(report.vulnerability_count)
print(report.severity_distribution)
```

**Probe Strategies:**
- `ContradictionTrap` - Tests if agent contradicts itself
- `HallucinationBait` - Tests for fabricated information
- `SycophancyTest` - Tests for excessive agreement
- `PersistenceChallenge` - Tests reasoning under pressure

### Deep Audit Mode

Intensive multi-round analysis (Heavy3-inspired).

```python
from aragora.modes import (
    DeepAuditOrchestrator,
    DeepAuditConfig,
    CODE_ARCHITECTURE_AUDIT,
)

# Use pre-configured audit
orchestrator = DeepAuditOrchestrator(config=CODE_ARCHITECTURE_AUDIT)

# Or create custom config
config = DeepAuditConfig(
    name="Custom Audit",
    rounds=6,
    enable_cross_examination=True,
    severity_threshold=0.7,
)

# Run audit
verdict = await orchestrator.run_audit(
    target="Your code or strategy here...",
    auditors=[agent1, agent2, agent3],
)

print(verdict.findings)        # List of AuditFinding
print(verdict.final_score)     # 0.0 - 1.0
print(verdict.recommendations)
```

**Pre-configured Audits:**
- `STRATEGY_AUDIT` - Business strategy analysis
- `CONTRACT_AUDIT` - Legal/contract review
- `CODE_ARCHITECTURE_AUDIT` - Code design review

### Gauntlet Mode

Comprehensive stress-testing combining multiple modes.

```python
from aragora.gauntlet import GauntletRunner

runner = GauntletRunner(
    phases=[
        "redteam",      # Adversarial attacks
        "probe",        # Capability probing
        "audit",        # Deep analysis
        "verification", # Formal verification
    ],
    timeout_minutes=30,
)

result = await runner.run(target=agent_under_test)
print(result.overall_verdict)
```

---

## Mode Handoff

Switch modes during a session:

```python
from aragora.modes import HandoffContext, ModeHandoff

# Create handoff
handoff = ModeHandoff(
    from_mode="architect",
    to_mode="coder",
    context=HandoffContext(
        summary="Design complete, ready for implementation",
        files_to_modify=["src/api.py", "src/models.py"],
        constraints=["Must maintain backward compatibility"],
    ),
)

# Execute handoff
new_mode = await handoff.execute()
```

---

## Mode Selection Guide

| Task | Recommended Mode |
|------|------------------|
| Planning features | `ArchitectMode` |
| Writing code | `CoderMode` |
| Code review | `ReviewerMode` |
| Fixing bugs | `DebuggerMode` |
| Security testing | `RedTeamMode` |
| Agent evaluation | `CapabilityProber` |
| Due diligence | `DeepAuditOrchestrator` |
| Comprehensive testing | `GauntletRunner` |

---

## See Also

- [Debate Orchestration](../core-concepts/architecture) - Core debate system
- [Agent Configuration](../core-concepts/agent-development) - Agent setup
- [Gauntlet Testing](./gauntlet) - Stress testing
