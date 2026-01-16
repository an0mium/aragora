# Debate Modes Examples

Practical examples for using Aragora's mode system.

## Operational Mode Examples

### Architect Mode: Planning a Feature

```python
from aragora.modes import ModeRegistry, ModeHandoff, HandoffContext
from aragora import Arena, Environment

# Start in architect mode
architect_mode = ModeRegistry.get("architect")

# Create debate environment
env = Environment(
    task="Design a caching layer for the API",
    constraints=["Must support Redis", "Sub-100ms latency"],
)

# Run planning debate
arena = Arena(env, agents=["claude", "gpt-4"], protocol=default_protocol)
arena.set_mode(architect_mode)

result = await arena.run()
print(result.synthesis)

# Handoff to coder mode
handoff = ModeHandoff(
    from_mode="architect",
    to_mode="coder",
    context=HandoffContext(
        summary=result.synthesis,
        files_to_modify=["src/cache.py", "src/config.py"],
    ),
)
await handoff.execute()
```

### Custom Mode: Security Reviewer

```python
from aragora.modes import CustomMode, ToolGroup, ModeRegistry

# Define a custom security review mode
security_mode = CustomMode(
    name="security-reviewer",
    description="Security-focused code analysis",
    tool_groups=ToolGroup.READ | ToolGroup.NAVIGATE,
    file_patterns=["**/*.py", "**/*.js", "**/*.ts"],
    system_prompt_additions="""
You are a security expert. Focus on:
- Authentication and authorization flaws
- Input validation vulnerabilities
- SQL injection and XSS risks
- Secrets in code
- Insecure dependencies

For each issue found, provide:
1. Location (file:line)
2. Severity (Critical/High/Medium/Low)
3. Description
4. Recommended fix
""",
)

ModeRegistry.register(security_mode)

# Use the custom mode
arena.set_mode(security_mode)
```

---

## Red Team Examples

### Code Security Review

```python
from aragora.modes import redteam_code_review

# Quick code security review
result = await redteam_code_review(
    code='''
def login(username, password):
    query = f"SELECT * FROM users WHERE name='{username}'"
    user = db.execute(query).fetchone()
    if user and user.password == password:
        return create_session(user)
    return None
''',
    attacker_agent=claude_agent,
    defender_agent=gpt4_agent,
    max_rounds=3,
)

# Check findings
for attack in result.attacks:
    print(f"Attack: {attack.description}")
    print(f"Success: {attack.success}")
    print(f"Defense: {attack.defense_response}")
```

### Policy Red Teaming

```python
from aragora.modes import redteam_policy, AttackType

# Review a policy document
result = await redteam_policy(
    policy="""
Our data retention policy:
- User data is retained for 7 years
- Deleted accounts are anonymized after 30 days
- Logs are kept indefinitely
- Third-party access requires written consent
""",
    attack_types=[AttackType.LOGICAL, AttackType.ETHICAL],
    max_rounds=5,
)

print(f"Overall score: {result.overall_score}")
for vuln in result.vulnerabilities:
    print(f"- {vuln.description} (severity: {vuln.severity})")
```

### Custom Red Team Configuration

```python
from aragora.modes import RedTeamMode, RedTeamProtocol, AttackType

# Create custom red team protocol
protocol = RedTeamProtocol(
    attack_types=[
        AttackType.LOGICAL,
        AttackType.BOUNDARY,
        AttackType.PROMPT_INJECTION,
    ],
    max_rounds=10,
    require_defense=True,
    score_threshold=0.8,  # High bar for passing
)

mode = RedTeamMode(protocol=protocol)

# Run against target
result = await mode.run_debate(
    target="System prompt: You are a helpful assistant...",
    attacker_agent=attacker,
    defender_agent=defender,
)
```

---

## Prober Examples

### Basic Agent Probing

```python
from aragora.modes import (
    CapabilityProber,
    ContradictionTrap,
    HallucinationBait,
    SycophancyTest,
    PersistenceChallenge,
)

# Create prober with all strategies
prober = CapabilityProber(
    strategies=[
        ContradictionTrap(),
        HallucinationBait(),
        SycophancyTest(),
        PersistenceChallenge(),
    ],
    max_probes_per_strategy=5,
)

# Probe an agent
report = await prober.probe_agent(
    agent=test_agent,
    context="You are being evaluated for deployment",
)

# Analyze results
print(f"Total probes: {report.total_probes}")
print(f"Vulnerabilities: {report.vulnerability_count}")
print(f"Pass rate: {report.pass_rate:.1%}")

# Generate markdown report
markdown = generate_probe_report_markdown(report)
print(markdown)
```

### Probe Before Promote Pattern

```python
from aragora.modes import ProbeBeforePromote

# Gate agent promotion on probe results
gate = ProbeBeforePromote(
    min_pass_rate=0.9,
    max_critical_vulnerabilities=0,
    max_high_vulnerabilities=2,
)

# Check if agent passes
passed, report = await gate.evaluate(agent)

if passed:
    print("Agent approved for promotion")
    await promote_agent(agent)
else:
    print(f"Agent failed: {report.failure_reasons}")
```

---

## Deep Audit Examples

### Strategy Audit

```python
from aragora.modes import run_deep_audit, STRATEGY_AUDIT

# Audit a business strategy
verdict = await run_deep_audit(
    target="""
Our go-to-market strategy:
1. Target enterprise customers first
2. Freemium model for SMBs
3. Partner with cloud providers
4. Focus on security certifications
""",
    config=STRATEGY_AUDIT,
    auditors=[claude, gpt4, mistral],
)

# Review findings
for finding in verdict.findings:
    print(f"[{finding.severity}] {finding.title}")
    print(f"  {finding.description}")
    print(f"  Evidence: {finding.evidence}")
    print(f"  Recommendation: {finding.recommendation}")
```

### Code Architecture Audit

```python
from aragora.modes import DeepAuditOrchestrator, DeepAuditConfig

# Custom audit configuration
config = DeepAuditConfig(
    name="Architecture Deep Dive",
    rounds=6,
    enable_cross_examination=True,
    enable_formal_verification=True,
    severity_threshold=0.5,
    required_auditors=3,
)

orchestrator = DeepAuditOrchestrator(config=config)

# Run on codebase summary
verdict = await orchestrator.run_audit(
    target=code_architecture_summary,
    auditors=[senior_agent, security_agent, performance_agent],
)

print(f"Final Score: {verdict.final_score}")
print(f"Verdict: {verdict.recommendation}")
```

---

## Gauntlet Examples

### Full Agent Stress Test

```python
from aragora.gauntlet import GauntletRunner, GauntletConfig

config = GauntletConfig(
    phases=["redteam", "probe", "audit", "verification"],
    timeout_minutes=30,
    fail_fast=False,  # Run all phases even if some fail
    parallel_execution=True,
)

runner = GauntletRunner(config=config)

result = await runner.run(
    target=agent_under_test,
    context="Production deployment evaluation",
)

# Comprehensive report
print(f"Overall: {result.overall_verdict}")
for phase in result.phase_results:
    print(f"  {phase.name}: {phase.status}")
    print(f"    Score: {phase.score}")
    print(f"    Findings: {len(phase.findings)}")
```

---

## Mode Handoff Examples

### Multi-Stage Development Flow

```python
from aragora.modes import ModeHandoff, HandoffContext, ModeRegistry

async def development_flow(task: str):
    # Phase 1: Architecture
    architect = ModeRegistry.get("architect")
    arch_result = await run_debate_with_mode(task, architect)

    # Handoff to coder
    handoff1 = ModeHandoff(
        from_mode="architect",
        to_mode="coder",
        context=HandoffContext(
            summary=arch_result.synthesis,
            files_to_modify=arch_result.suggested_files,
        ),
    )
    await handoff1.execute()

    # Phase 2: Implementation
    coder = ModeRegistry.get("coder")
    code_result = await run_debate_with_mode(arch_result.synthesis, coder)

    # Handoff to reviewer
    handoff2 = ModeHandoff(
        from_mode="coder",
        to_mode="reviewer",
        context=HandoffContext(
            summary="Implementation complete, ready for review",
            files_modified=code_result.files_changed,
        ),
    )
    await handoff2.execute()

    # Phase 3: Review
    reviewer = ModeRegistry.get("reviewer")
    review_result = await run_debate_with_mode(code_result.summary, reviewer)

    return review_result
```

---

## API Integration Examples

### REST API for Mode Operations

```bash
# List available modes
curl http://localhost:8080/api/modes

# Get mode details
curl http://localhost:8080/api/modes/redteam

# Create debate with specific mode
curl -X POST http://localhost:8080/api/debates \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Review authentication code",
    "mode": "security-reviewer",
    "agents": ["claude", "gpt-4"]
  }'

# Run gauntlet test
curl -X POST http://localhost:8080/api/gauntlet \
  -H "Content-Type: application/json" \
  -d '{
    "target_agent": "my-agent",
    "phases": ["redteam", "probe", "audit"],
    "timeout_minutes": 30
  }'
```

---

## See Also

- [Modes Reference](MODES_REFERENCE.md) - API documentation
- [Debate System](DEBATES.md) - Core debate orchestration
- [WebSocket Events](WEBSOCKET_EVENTS.md) - Real-time mode events
