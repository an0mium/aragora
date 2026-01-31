# Aragora Mode System

The mode system provides granular control over agent behavior and capabilities through two complementary subsystems:

1. **Operational Modes** - Control tool access and behavioral guidelines
2. **Capability Probing** - Adversarial testing for agent reliability

## Quick Start

```python
from aragora.modes import (
    ModeRegistry,
    CoderMode,
    ArchitectMode,
    CapabilityProber,
    ProbeType,
)

# Get a registered mode
coder = ModeRegistry.get("coder")
print(coder.can_access_tool("edit"))  # True
print(coder.get_system_prompt())  # Behavioral guidelines

# Run capability probing
prober = CapabilityProber()
report = await prober.probe_agent(
    agent,
    run_agent_fn,
    probe_types=[ProbeType.HALLUCINATION, ProbeType.SYCOPHANCY],
)
print(f"Vulnerability rate: {report.vulnerability_rate:.1%}")
```

## Operational Modes

### Built-in Modes

| Mode | Tools | Purpose |
|------|-------|---------|
| **Architect** | READ, BROWSER | High-level design and planning without implementation |
| **Coder** | READ, EDIT, COMMAND | Implementation with full development access |
| **Reviewer** | READ, BROWSER | Code review and quality analysis (read-only) |
| **Debugger** | READ, EDIT, COMMAND | Investigation and targeted bug fixes |
| **Orchestrator** | FULL | Coordination of complex multi-step workflows |

### Tool Groups

Tool groups control which operations a mode can perform:

```python
from aragora.modes import ToolGroup, can_use_tool

# Individual groups
ToolGroup.READ      # read, glob, grep
ToolGroup.EDIT      # edit, write, notebook_edit
ToolGroup.COMMAND   # bash, kill_shell
ToolGroup.BROWSER   # web_fetch, web_search
ToolGroup.MCP       # MCP server tools
ToolGroup.DEBATE    # debate, arena

# Composite groups
ToolGroup.READONLY()   # READ | BROWSER
ToolGroup.DEVELOPER()  # READ | EDIT | COMMAND
ToolGroup.FULL()       # All tools

# Check permissions
perms = ToolGroup.READ | ToolGroup.EDIT
can_use_tool(perms, "bash")  # False
can_use_tool(perms, "edit")  # True
```

### Custom Modes

Create custom modes via YAML configuration:

```yaml
# .aragora/modes/security-auditor.yaml
name: security-auditor
description: Security-focused code auditor
base_mode: reviewer  # Inherit from reviewer
tool_groups:
  - read
  - browser
file_patterns:
  - "**/*.py"
  - "**/*.js"
system_prompt_additions: |
  Focus on OWASP Top 10 vulnerabilities.
  Check for injection, XSS, and auth issues.
```

Load custom modes:

```python
from aragora.modes import CustomModeLoader, ModeRegistry

loader = CustomModeLoader()
count = loader.load_and_register_all()  # Load from default paths
print(f"Registered {count} custom modes")

# Access the custom mode
auditor = ModeRegistry.get("security-auditor")
```

### Mode Handoffs

Transfer context between modes during complex workflows:

```python
from aragora.modes import ModeHandoff

handoff = ModeHandoff()

# Create handoff context when transitioning
context = handoff.create_context(
    from_mode="architect",
    to_mode="coder",
    task_summary="Designed new caching layer with Redis backend",
    key_findings=[
        "Use Redis for distributed caching",
        "TTL of 5 minutes for session data",
        "Implement cache-aside pattern",
    ],
    files_touched=["cache/redis.py", "config/cache.yaml"],
    open_questions=["What cache invalidation strategy?"],
)

# Generate transition prompt for the target mode
prompt = handoff.generate_transition_prompt(
    context,
    CoderMode().get_system_prompt(),
)

# Review session history
print(handoff.summarize_session())
```

## Capability Probing

The probing system tests agents for reliability issues and vulnerabilities.

### Probe Types

| Type | Description | Severity Range |
|------|-------------|----------------|
| **CONTRADICTION** | Tests for self-contradictions under pressure | HIGH |
| **HALLUCINATION** | Detects fabricated information about fake entities | CRITICAL |
| **SYCOPHANCY** | Catches inappropriate agreement with false claims | CRITICAL |
| **PERSISTENCE** | Tests if agents abandon valid positions too easily | MEDIUM |
| **CONFIDENCE_CALIBRATION** | Measures accuracy of stated confidence levels | MEDIUM |
| **REASONING_DEPTH** | Tests multi-step logical reasoning | MEDIUM |
| **EDGE_CASE** | Identifies failures on boundary conditions | MEDIUM |
| **INSTRUCTION_INJECTION** | Tests for prompt injection vulnerabilities | CRITICAL |
| **CAPABILITY_EXAGGERATION** | Detects overclaiming of abilities | HIGH |

### Running Probes

```python
from aragora.modes import (
    CapabilityProber,
    ProbeType,
    generate_probe_report_markdown,
)

# Initialize prober (optionally with ELO system for penalties)
prober = CapabilityProber(
    elo_system=elo,
    elo_penalty_multiplier=5.0,
)

# Run comprehensive probing
report = await prober.probe_agent(
    target_agent,
    run_agent_fn,
    probe_types=[
        ProbeType.HALLUCINATION,
        ProbeType.SYCOPHANCY,
        ProbeType.INSTRUCTION_INJECTION,
    ],
    probes_per_type=3,
)

# Generate Markdown report
markdown = generate_probe_report_markdown(report)
print(markdown)
```

### Probe Strategies

Each probe type has a corresponding strategy class:

```python
from aragora.modes.probes import (
    ContradictionTrap,
    HallucinationBait,
    SycophancyTest,
    PersistenceChallenge,
    ConfidenceCalibrationProbe,
    ReasoningDepthProbe,
    EdgeCaseProbe,
    InstructionInjectionProbe,
    CapabilityExaggerationProbe,
)

# Strategies can be used directly
strategy = HallucinationBait()
probe_prompt = strategy.generate_probe(context=[], previous_probes=[])
vulnerable, description, severity = strategy.analyze_response(
    probe_prompt,
    agent_response,
    context=[],
)
```

### ELO Integration

Probing results can automatically affect agent ELO ratings:

```python
from aragora.modes import ProbeBeforePromote

# Gate ELO promotions on passing probes
middleware = ProbeBeforePromote(
    elo_system=elo,
    prober=prober,
    max_vulnerability_rate=0.2,  # 20% max
    max_critical=0,  # No critical vulnerabilities allowed
)

# Check before applying ELO gain
approved, report = await middleware.check_promotion(
    agent,
    run_agent_fn,
    pending_elo_gain=15.0,
)

if not approved:
    print(f"Promotion denied: {report.vulnerability_rate:.1%} vulnerability rate")
    # Agent can retry after improvement
    approved, report = await middleware.retry_promotion(agent, run_agent_fn)
```

## Module Structure

```
aragora/modes/
├── __init__.py          # Main exports
├── base.py              # Mode and ModeRegistry base classes
├── tool_groups.py       # ToolGroup flags and utilities
├── handoff.py           # Mode handoff system
├── custom.py            # Custom mode YAML loader
├── prober.py            # CapabilityProber main class
├── builtin/             # Built-in operational modes
│   ├── architect.py
│   ├── coder.py
│   ├── debugger.py
│   ├── reviewer.py
│   └── orchestrator.py
├── probes/              # Probe strategies
│   ├── models.py        # ProbeResult, VulnerabilityReport
│   └── strategies.py    # All probe strategy implementations
├── redteam.py           # Red team debate mode
├── deep_audit.py        # Deep audit debate protocol
└── gauntlet.py          # Gauntlet testing mode
```

## API Reference

### Core Classes

- `Mode` - Abstract base class for operational modes
- `ModeRegistry` - Global registry of available modes
- `ToolGroup` - Flag enum for tool permissions
- `HandoffContext` - Context data for mode transitions
- `ModeHandoff` - Manager for mode transitions
- `CustomMode` - YAML-defined custom modes
- `CustomModeLoader` - Loads custom modes from files

### Probing Classes

- `CapabilityProber` - Main probing orchestrator
- `ProbeBeforePromote` - ELO gating middleware
- `ProbeStrategy` - Abstract base for probe strategies
- `ProbeResult` - Single probe result
- `VulnerabilityReport` - Comprehensive probe session report

### Enums

- `ProbeType` - Types of capability probes
- `VulnerabilitySeverity` - LOW, MEDIUM, HIGH, CRITICAL

## Best Practices

1. **Mode Selection**: Use the most restrictive mode that allows completing the task
2. **Handoff Context**: Always include key findings and open questions when transitioning
3. **Probing Frequency**: Run probes periodically, not just during development
4. **Custom Modes**: Inherit from built-in modes to extend rather than replace
5. **ELO Integration**: Use ProbeBeforePromote for production ranking systems
