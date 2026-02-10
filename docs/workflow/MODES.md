# Aragora Mode System

Aragora provides two complementary mode systems for controlling agent behavior and debate protocols.

## Overview

| System | Purpose | Examples |
|--------|---------|----------|
| **Operational Modes** | Tool access control | Architect, Coder, Reviewer, Debugger, Orchestrator |
| **Debate Modes** | Adversarial protocols | RedTeam, Prober, DeepAudit, Gauntlet |

## Operational Modes

Kilocode-inspired modes that define what tools an agent can access during execution.

### Tool Groups

Operational modes control access through composable tool groups:

```python
from aragora.modes import ToolGroup

class ToolGroup(Flag):
    NONE = 0
    READ = auto()      # Read files
    EDIT = auto()      # Write/edit files
    COMMAND = auto()   # Execute shell commands
    BROWSER = auto()   # Web fetch and search
    MCP = auto()       # MCP server tools
    DEBATE = auto()    # Debate participation

    # Composite groups
    READONLY = READ | BROWSER
    DEVELOPER = READ | EDIT | COMMAND
    FULL = READ | EDIT | COMMAND | BROWSER | MCP | DEBATE
```

### Built-in Modes

#### Architect Mode

**Tool Access:** READ | BROWSER

**Purpose:** High-level design, planning, and codebase analysis without modification.

**Best For:**
- Analyzing codebase patterns and architecture
- Proposing designs and feature plans
- Dependency mapping and impact analysis
- Creating architectural diagrams

**Restrictions:**
- Cannot modify files (read-only)
- Cannot execute commands
- Focus on analysis and planning only

```python
from aragora.modes import ArchitectMode, ModeRegistry

architect = ModeRegistry.get("architect")
prompt = architect.get_system_prompt()
```

#### Coder Mode

**Tool Access:** READ | EDIT | COMMAND

**Purpose:** Implementation with full development capabilities.

**Best For:**
- Writing and editing code
- Running tests and builds
- Making commits
- Iterative development

**Guidelines:**
- Follow existing patterns
- Make minimal, focused changes
- Test as you go
- Create atomic commits

```python
from aragora.modes import CoderMode

coder = ModeRegistry.get("coder")
can_write = coder.can_access_tool("edit")  # True
```

#### Reviewer Mode

**Tool Access:** READ | BROWSER

**Purpose:** Code review and quality analysis.

**Best For:**
- Code review with structured feedback
- Security analysis
- Performance review
- Maintainability assessment

**Output Format:**
- Location (file:line)
- Severity (critical/high/medium/low)
- Issue description
- Why it matters
- Conceptual fix suggestion

**Restrictions:**
- Read-only access
- Suggestions, not implementations

#### Debugger Mode

**Tool Access:** READ | EDIT | COMMAND

**Purpose:** Bug investigation and targeted fixes.

**Best For:**
- Reproducing bugs
- Isolating root causes
- Applying minimal fixes
- Verifying solutions

**Methodology:**
1. Reproduce the issue
2. Isolate to minimal case
3. Understand root cause
4. Fix with minimal changes
5. Verify with tests

#### Orchestrator Mode

**Tool Access:** FULL (all tools)

**Purpose:** Coordinate complex multi-step workflows and delegate to appropriate modes.

**Best For:**
- Complex multi-phase tasks
- Mode selection and handoff coordination
- Cross-cutting concerns

**Principles:**
1. Decompose tasks into sub-tasks
2. Delegate to appropriate modes
3. Synthesize results
4. Validate outcomes

### Custom Modes

Create custom modes via YAML configuration:

```yaml
# ~/.aragora/modes/security-reviewer.yaml
name: security-reviewer
description: Security-focused code review
base_mode: reviewer  # Inherit from reviewer
tool_groups:
  - read
  - browser
file_patterns:
  - "**/*.py"
  - "**/*.js"
  - "**/auth/**"
system_prompt_additions: |
  Focus on security vulnerabilities:
  - SQL injection
  - XSS
  - CSRF
  - Authentication flaws
  - Authorization bypasses
```

Load and register custom modes:

```python
from aragora.modes import CustomModeLoader, ModeRegistry

loader = CustomModeLoader()
mode = loader.load_from_file("~/.aragora/modes/security-reviewer.yaml")
ModeRegistry.register(mode)

# Or load all from search paths
loader.load_and_register_all()
```

**Search Paths:**
- `.aragora/modes/` (project-local)
- `~/.config/aragora/modes/` (user-global)

---

## Debate Modes

Specialized protocols for adversarial agent interactions.

### RedTeam Mode

Adversarial security testing with attack/defense rounds.

```python
from aragora.modes import RedTeamMode, AttackType

mode = RedTeamMode(
    attack_types=[AttackType.LOGICAL, AttackType.SECURITY],
    max_rounds=5,
)

result = await mode.run_debate(
    target="Our authentication system uses JWT tokens...",
    attacker_agent=attacker,
    defender_agent=defender,
)

print(result.vulnerabilities)  # List of found issues
print(result.overall_score)    # 0.0 - 1.0 security score
```

**Attack Types:**

| Type | Description |
|------|-------------|
| `LOGICAL_FALLACY` | Find reasoning errors and contradictions |
| `EDGE_CASE` | Break at boundaries |
| `UNSTATED_ASSUMPTION` | Expose hidden premises |
| `COUNTEREXAMPLE` | Disprove with specific cases |
| `SCALABILITY` | Stress at scale |
| `SECURITY` | Find security vulnerabilities |
| `ADVERSARIAL_INPUT` | Malicious inputs |
| `RESOURCE_EXHAUSTION` | DoS conditions |
| `RACE_CONDITION` | Concurrency issues |
| `DEPENDENCY_FAILURE` | What if X fails? |

**Key Functions:**
- `run_redteam_code_review()` - Code-specific attacks
- `run_redteam_policy()` - Policy-focused attacks

### Prober Mode

Capability probing to test agent weaknesses and calibration.

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
        HallucinationBait(),      # Test for fabricated information
        SycophancyTest(),         # Test for agreement bias
    ]
)

report = await prober.probe_agent(agent, max_probes=10)

print(report.vulnerability_count)
print(report.severity_distribution)
```

**Probe Strategies:**

| Strategy | Purpose |
|----------|---------|
| `ContradictionTrap` | Tests if agent contradicts itself |
| `HallucinationBait` | Tests for fabricated information |
| `SycophancyTest` | Tests for excessive agreement bias |
| `PersistenceChallenge` | Tests reasoning under pressure |
| `ReasoningDepthProbe` | Shallow vs deep reasoning detection |
| `EdgeCaseProbe` | Boundary condition handling |
| `InstructionInjectionProbe` | Prompt security testing |
| `CapabilityExaggerationProbe` | Overconfidence detection |
| `ConfidenceCalibrationProbe` | Uncertainty quantification |

**ProbeBeforePromote Pattern:**

Test agent capability before using in debates. Applies ELO penalty for unreliable behavior, creating evolutionary pressure toward robustness.

### Deep Audit Mode

Intensive multi-round analysis inspired by Heavy3 methodology.

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
    enable_research=True,
    enable_cross_examination=True,
    require_citations=True,
    severity_threshold=0.7,
)

verdict = await orchestrator.run_audit(
    target="Your code or strategy here...",
    auditors=[agent1, agent2, agent3],
)

print(verdict.findings)         # List of AuditFinding
print(verdict.final_score)      # 0.0 - 1.0
print(verdict.recommendations)
print(verdict.unanimous_issues)
print(verdict.split_opinions)
```

**Cognitive Roles (rotated each round):**
- **Analyst** - Systematic breakdown and analysis
- **Skeptic** - Challenge assumptions, find weaknesses
- **Lateral Thinker** - Alternative perspectives and approaches
- **Advocate** - Defend strengths, balance critique
- **Synthesizer** - Integrate findings (final round)

**Pre-configured Audits:**
- `STRATEGY_AUDIT` - Business strategy analysis
- `CONTRACT_AUDIT` - Legal/contract review
- `CODE_ARCHITECTURE_AUDIT` - Code design review

### Gauntlet Mode

Comprehensive stress-testing combining multiple modes.

```python
from aragora.gauntlet import GauntletRunner, GauntletConfig

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

**Verdicts:**
- `APPROVED` - Passed all checks
- `APPROVED_WITH_CONDITIONS` - Minor issues noted
- `NEEDS_REVIEW` - Significant issues requiring attention
- `REJECTED` - Critical issues found

**Pre-configured Profiles:**

| Profile | Duration | Focus |
|---------|----------|-------|
| `QUICK_GAUNTLET` | ~2 min | Minimal verification, fast feedback |
| `THOROUGH_GAUNTLET` | ~15 min | Full verification, comprehensive |
| `CODE_REVIEW_GAUNTLET` | Variable | Security, edge cases, concurrency |
| `POLICY_GAUNTLET` | Variable | Logic, assumptions, counterexamples |
| `GDPR_GAUNTLET` | Variable | GDPR compliance focus |
| `HIPAA_GAUNTLET` | Variable | HIPAA compliance focus |

---

## Mode Handoff

Smoothly transition between modes during a session:

```python
from aragora.modes import HandoffContext, ModeHandoff

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

# Generate transition prompt
prompt = handoff.generate_transition_prompt()

# Get session summary
timeline = handoff.summarize_session()
```

**HandoffContext Fields:**
- `task_summary` - What was accomplished
- `key_findings` - Important discoveries
- `files_touched` - Modified files list
- `open_questions` - Unresolved issues
- `artifacts` - Arbitrary state dictionary

---

## Mode Selection Guide

| Task | Recommended Mode |
|------|------------------|
| Planning features | `ArchitectMode` |
| Writing code | `CoderMode` |
| Code review | `ReviewerMode` |
| Fixing bugs | `DebuggerMode` |
| Complex workflows | `OrchestratorMode` |
| Security testing | `RedTeamMode` |
| Agent evaluation | `CapabilityProber` |
| Due diligence | `DeepAuditOrchestrator` |
| Comprehensive testing | `GauntletRunner` |

---

## API Reference

### Core Classes

```python
from aragora.modes import (
    # Registry
    Mode, ModeRegistry,

    # Tool groups
    ToolGroup, can_use_tool, get_required_group,

    # Built-in operational modes
    ArchitectMode, CoderMode, ReviewerMode, DebuggerMode, OrchestratorMode,

    # Custom modes
    CustomMode, CustomModeLoader,

    # Debate modes
    RedTeamMode, RedTeamResult, Attack, Defense, AttackType,
    CapabilityProber, ProbeType, VulnerabilityReport,
    DeepAuditOrchestrator, DeepAuditConfig, DeepAuditVerdict,

    # Handoff
    HandoffContext, ModeHandoff,
)

from aragora.gauntlet import (
    GauntletOrchestrator, GauntletConfig, GauntletResult,
    QUICK_GAUNTLET, THOROUGH_GAUNTLET,
)
```

### Common Operations

```python
# Get a mode
mode = ModeRegistry.get("architect")

# Check tool access
mode.can_access_tool("read")     # True
mode.can_access_tool("edit")     # False

# List all modes
ModeRegistry.list_all()

# Get mode or raise error
mode = ModeRegistry.get_or_raise("unknown")  # KeyError

# Clear registry (testing only)
ModeRegistry.clear()
```

---

## Best Practices

### Mode Selection

1. **Start with read-only** - Use Architect/Reviewer first for safety
2. **Match mode to task phase** - Design (Architect) → Implement (Coder) → Review (Reviewer)
3. **Use Orchestrator for complexity** - Let it coordinate multi-phase work

### File Pattern Safety

1. Restrict file access when possible
2. Document intended scope in custom modes
3. Use glob patterns consistently

### Debate Mode Strategy

1. Calibrate severity thresholds per domain
2. Use appropriate attack types for context
3. Run probes before promoting agents to debates
4. Review findings critically (not all are actionable)

### Gauntlet Testing

1. Start with `QUICK_GAUNTLET` for fast feedback
2. Use pre-configured profiles for common scenarios
3. Customize thresholds based on risk tolerance
4. Review high-confidence verdicts first

---

## Troubleshooting

### "Mode not found"

Check available modes with `ModeRegistry.list_all()` and verify the mode name.

### "Permission denied" for custom mode

Verify the custom mode YAML file is in an allowed directory:
- `.aragora/modes/` (project)
- `~/.config/aragora/modes/` (user)

### Debate results seem extreme

1. Validate severity thresholds for your domain
2. Check agent calibration with probes
3. Review attack type selection

### Gauntlet timeouts

1. Reduce phases or parallelism
2. Use `QUICK_GAUNTLET` profile
3. Increase timeout configuration

---

## See Also

- [ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture overview
- [GAUNTLET.md](./GAUNTLET.md) - Detailed gauntlet documentation
- [AGENT_DEVELOPMENT.md](./AGENT_DEVELOPMENT.md) - Creating custom agents
- [PROBE_STRATEGIES.md](./PROBE_STRATEGIES.md) - Probe strategy details
