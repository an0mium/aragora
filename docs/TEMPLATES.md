# Debate Templates

The templates module (`aragora/templates/`) provides pre-built configurations for structured multi-agent debates. Templates define roles, phases, evaluation criteria, and output formats for specific domains.

## Overview

| Component | Purpose |
|-----------|---------|
| `TemplateType` | Enum of template categories |
| `DebateRole` | Participant role definition |
| `DebatePhase` | Phase structure with rounds |
| `DebateTemplate` | Complete debate specification |
| `get_template()` | Retrieve template by type |
| `list_templates()` | List all available templates |
| `template_to_protocol()` | Convert template to DebateProtocol |

## Available Templates

| Template | Domain | Agents | Phases | Consensus | Difficulty |
|----------|--------|--------|--------|-----------|------------|
| Code Review | Engineering | 4 | 4 | 0.70 | 0.6 |
| Design Doc | Architecture | 4 | 3 | 0.60 | 0.7 |
| Incident Response | Operations | 4 | 3 | 0.70 | 0.8 |
| Research Synthesis | Research | 4 | 3 | 0.60 | 0.7 |
| Security Audit | Security | 5 | 4 | 0.80 | 0.9 |
| Architecture Review | Architecture | 5 | 4 | 0.70 | 0.8 |
| Healthcare Compliance | Healthcare | 5 | 4 | 0.80 | 0.9 |
| Financial Risk | Finance | 5 | 4 | 0.75 | 0.9 |

## Core Types

### TemplateType

Categories of debate templates:

```python
class TemplateType(str, Enum):
    CODE_REVIEW = "code_review"
    DESIGN_DOC = "design_doc"
    INCIDENT_RESPONSE = "incident_response"
    RESEARCH_SYNTHESIS = "research_synthesis"
    POLICY_REVIEW = "policy_review"        # Planned
    SECURITY_AUDIT = "security_audit"
    ARCHITECTURE_REVIEW = "architecture_review"
    PRODUCT_STRATEGY = "product_strategy"  # Planned
    HEALTHCARE_COMPLIANCE = "healthcare_compliance"
    FINANCIAL_RISK = "financial_risk"
```

### DebateRole

Defines a participant's role in a debate:

```python
@dataclass
class DebateRole:
    name: str                     # Role identifier
    description: str              # Role purpose
    objectives: list[str]         # Goals for the role
    evaluation_criteria: list[str]  # Performance assessment
    example_prompts: list[str] = field(default_factory=list)
```

### DebatePhase

Defines a structured phase within the debate:

```python
@dataclass
class DebatePhase:
    name: str                # Phase identifier
    description: str         # Phase purpose
    duration_rounds: int     # Number of rounds
    roles_active: list[str]  # Participating roles
    objectives: list[str]    # Phase goals
    outputs: list[str]       # Expected outputs
```

### DebateTemplate

Complete debate specification:

```python
@dataclass
class DebateTemplate:
    template_id: str              # Unique identifier
    template_type: TemplateType   # Category
    name: str                     # Display name
    description: str              # Full description
    roles: list[DebateRole]       # Participant roles
    phases: list[DebatePhase]     # Structured phases
    recommended_agents: int       # Suggested team size
    max_rounds: int               # Maximum rounds
    consensus_threshold: float    # Agreement requirement
    rubric: dict[str, float]      # Weighted evaluation criteria
    output_format: str            # Markdown template
    domain: str                   # Domain area
    difficulty: float = 0.5       # Difficulty score
    tags: list[str] = field(default_factory=list)
```

## Template Details

### Code Review Template

Multi-perspective code analysis for security, performance, and maintainability.

**Roles:**
- `author` - Explains design decisions and context
- `security_critic` - Identifies vulnerabilities and attack vectors
- `performance_critic` - Analyzes efficiency and bottlenecks
- `maintainability_critic` - Evaluates readability and patterns
- `synthesizer` - Combines findings into actionable summary

**Phases:**
1. Initial Review (1 round) - Critics examine code
2. Author Response (1 round) - Author addresses concerns
3. Debate (2 rounds) - Discussion and refinement
4. Synthesis (1 round) - Final recommendations

**Rubric:**
| Criterion | Weight |
|-----------|--------|
| Security Coverage | 0.30 |
| Performance Impact | 0.20 |
| Maintainability | 0.20 |
| Actionability | 0.20 |
| Consensus | 0.10 |

### Security Audit Template

Comprehensive security assessment with threat modeling and red/blue team simulation.

**Roles:**
- `threat_modeler` - Maps attack surface and threat actors
- `vulnerability_analyst` - Identifies specific weaknesses
- `red_team` - Simulates attacker techniques
- `blue_team` - Proposes defensive controls
- `compliance_officer` - Ensures regulatory alignment

**Phases:**
1. Reconnaissance (1 round) - Gather attack surface info
2. Vulnerability Assessment (2 rounds) - Identify weaknesses
3. Attack Simulation (2 rounds) - Red/blue team exercise
4. Remediation Planning (1 round) - Prioritized fixes

**Rubric:**
| Criterion | Weight |
|-----------|--------|
| Vulnerability Accuracy | 0.25 |
| Threat Coverage | 0.20 |
| Attack Realism | 0.20 |
| Remediation Quality | 0.20 |
| Defense Assessment | 0.15 |

### Healthcare Compliance Template

HIPAA/HITECH compliance audit for healthcare systems.

**Roles:**
- `privacy_officer` - PHI handling and consent
- `security_analyst` - Technical safeguards
- `compliance_auditor` - Regulatory alignment
- `clinical_operations` - Workflow impact
- `breach_analyst` - Incident response readiness

**Phases:**
1. Inventory (1 round) - Map PHI flows
2. Control Assessment (2 rounds) - Evaluate safeguards
3. Risk Analysis (2 rounds) - Identify gaps
4. Remediation (1 round) - Compliance roadmap

**Rubric:**
| Criterion | Weight |
|-----------|--------|
| Privacy Rule Coverage | 0.25 |
| Security Rule Coverage | 0.25 |
| Risk Analysis Quality | 0.20 |
| Breach Readiness | 0.15 |
| Remediation Practicality | 0.15 |

### Financial Risk Template

Trading strategy stress-testing and risk assessment.

**Roles:**
- `strategist` - Presents trading logic
- `quant_analyst` - Mathematical validation
- `risk_manager` - Risk quantification
- `market_skeptic` - Challenge assumptions
- `compliance_reviewer` - Regulatory check

**Phases:**
1. Strategy Presentation (1 round) - Present thesis
2. Quantitative Review (2 rounds) - Validate models
3. Stress Testing (2 rounds) - Extreme scenarios
4. Final Assessment (1 round) - Risk summary

**Rubric:**
| Criterion | Weight |
|-----------|--------|
| Quantitative Rigor | 0.25 |
| Risk Assessment | 0.25 |
| Strategy Validity | 0.20 |
| Stress Test Coverage | 0.20 |
| Compliance Check | 0.10 |

## Usage

### Basic Template Usage

```python
from aragora import Arena, Environment
from aragora.templates import get_template, TemplateType

# Get template
template = get_template(TemplateType.CODE_REVIEW)

# Create environment
env = Environment(task="Review authentication module changes")

# Build arena with template
arena = (Arena.builder(env, agents)
    .with_template(template)
    .build()
)

# Run structured debate
result = await arena.run()
```

### With Protocol Overrides

```python
from aragora.templates import template_to_protocol

template = get_template(TemplateType.SECURITY_AUDIT)

# Override default settings
protocol = template_to_protocol(template, overrides={
    "rounds": 8,                    # More rounds
    "consensus": "unanimous",       # Stricter consensus
    "convergence_threshold": 0.95,  # Higher agreement
})

arena = Arena(env, agents, protocol)
```

### Listing Available Templates

```python
from aragora.templates import list_templates

for tmpl in list_templates():
    print(f"{tmpl['name']}: {tmpl['description']}")
    print(f"  Domain: {tmpl['domain']}, Agents: {tmpl['agents']}")
```

### Custom Template Creation

```python
from aragora.templates import (
    DebateTemplate,
    DebateRole,
    DebatePhase,
    TemplateType,
)

# Define roles
reviewer = DebateRole(
    name="reviewer",
    description="Primary document reviewer",
    objectives=["Identify issues", "Suggest improvements"],
    evaluation_criteria=["Thoroughness", "Actionability"],
)

author = DebateRole(
    name="author",
    description="Document author",
    objectives=["Clarify intent", "Address feedback"],
    evaluation_criteria=["Responsiveness", "Quality of fixes"],
)

# Define phases
review_phase = DebatePhase(
    name="review",
    description="Initial review",
    duration_rounds=2,
    roles_active=["reviewer"],
    objectives=["Identify all issues"],
    outputs=["Issue list"],
)

discussion_phase = DebatePhase(
    name="discussion",
    description="Author-reviewer discussion",
    duration_rounds=3,
    roles_active=["reviewer", "author"],
    objectives=["Resolve issues"],
    outputs=["Resolution plan"],
)

# Create template
custom_template = DebateTemplate(
    template_id="doc-review-v1",
    template_type=TemplateType.DESIGN_DOC,
    name="Document Review",
    description="Structured document review process",
    roles=[reviewer, author],
    phases=[review_phase, discussion_phase],
    recommended_agents=2,
    max_rounds=5,
    consensus_threshold=0.7,
    rubric={
        "issue_coverage": 0.4,
        "resolution_quality": 0.4,
        "consensus": 0.2,
    },
    output_format="# Review Summary\n{findings}\n{resolutions}",
    domain="documentation",
    difficulty=0.5,
)
```

## Protocol Conversion

The `template_to_protocol()` function converts templates to debate protocols with intelligent defaults:

### Topology Selection

| Template Type | Topology | Reason |
|---------------|----------|--------|
| Research Synthesis | all-to-all | Benefits from full discussion |
| Templates with 4+ roles | round-robin | Structured turn-based |
| Others | all-to-all | General discussion |

### Default Protocol Settings

```python
DebateProtocol(
    rounds=max(total_phase_rounds, template.max_rounds),
    consensus="majority",
    consensus_threshold=template.consensus_threshold,
    topology=<selected based on template>,
    role_rotation=True,        # Enable role cycling
    require_reasoning=True,    # Force detailed explanations
    early_stopping=True,       # Allow vote to stop
    convergence_detection=True,  # Detect semantic agreement
    enable_calibration=True,   # Track confidence
)
```

## Output Formats

Each template defines a Markdown output format with placeholders:

```markdown
# Code Review Summary

## Risk Score: {risk_score}/10

## Critical Issues
{critical_issues}

## Security ({security_score})
{security_findings}

## Performance ({performance_score})
{performance_findings}

## Maintainability ({maintainability_score})
{maintainability_findings}

## Action Items
{action_items}

## Consensus Notes
{consensus_notes}
```

## Integration with Arena

### ArenaBuilder Pattern

```python
arena = (ArenaBuilder(environment, agents)
    .with_template(template, overrides={"rounds": 6})
    .with_memory(critique_store)
    .with_elo_system(elo)
    .build()
)
```

The builder:
1. Stores the template reference
2. Calls `template_to_protocol()` with overrides
3. Configures the Arena with the resulting protocol
4. Enables role rotation based on template phases

### Phase Progression

During debate execution:
1. Arena tracks current phase via round count
2. Role rotation activates appropriate roles per phase
3. Phase objectives guide agent responses
4. Output format structures final results

## Validation

Templates are validated at load time:

**Rubric Weights:**
- Must sum to approximately 1.0 (±0.01)
- All weights must be positive

**Phase Consistency:**
- Total phase rounds ≤ 2 × max_rounds
- Each phase has ≥1 round
- All active roles defined in template

**Role Consistency:**
- No duplicate role names
- All phase references valid

## Gauntlet Templates

Note: Gauntlet templates (`aragora/gauntlet/templates.py`) are separate from debate templates. They use `GauntletConfig` and are designed for adversarial validation rather than structured debate.

| System | Purpose | Configuration |
|--------|---------|---------------|
| Debate Templates | Structured multi-agent discussion | DebateTemplate |
| Gauntlet Templates | Adversarial stress testing | GauntletConfig |

## See Also

- [ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture
- [MODES.md](./MODES.md) - Operational modes
- [GAUNTLET.md](./GAUNTLET.md) - Adversarial testing
