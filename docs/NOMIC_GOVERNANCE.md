# Nomic Loop Governance Guide

The Nomic Loop is Aragora's autonomous self-improvement cycle where AI agents debate, design, implement, and verify code changes. This guide covers the governance mechanisms that ensure safety and auditability.

## Overview

The Nomic Loop is a 6-phase cycle:

```
IDLE -> CONTEXT -> DEBATE -> DESIGN -> IMPLEMENT -> VERIFY -> COMMIT -> IDLE
          |          |         |          |           |         |
          +----------+-------- +----------+-----------+---------+
                               |
                               v
                           RECOVERY -> IDLE
```

| Phase | Description | Timeout |
|-------|-------------|---------|
| Context | Gather codebase understanding | 20 min |
| Debate | Multi-agent debate on improvements | 60 min |
| Design | Design implementation approach | 30 min |
| Implement | Write code changes | 40 min |
| Verify | Run tests and validation | 30 min |
| Commit | Commit approved changes | 5 min |

## Approval Gates

Approval gates are safety checkpoints that require explicit approval before critical phases proceed. Gates create audit trails and prevent unauthorized changes.

### Gate Types

| Gate | When | Purpose |
|------|------|---------|
| `DesignGate` | Before implementation | Approve design before code is written |
| `TestQualityGate` | After verification | Validate test quality thresholds |
| `CommitGate` | Before commit | Final approval before committing changes |

### DesignGate

Validates design documents before implementation begins.

**Checks:**
- Design complexity score (default max: 0.8)
- Affected files list (required by default)

**Configuration:**
```python
from aragora.nomic.gates import DesignGate

gate = DesignGate(
    enabled=True,
    auto_approve_dev=True,      # Auto-approve in dev mode
    max_complexity_score=0.8,   # Maximum design complexity
    require_files_list=True,    # Design must list affected files
)

# Usage
decision = await gate.require_approval(
    artifact=design_document,
    context={
        "complexity_score": 0.5,
        "files_affected": ["src/module.py"]
    }
)
```

### TestQualityGate

Ensures test quality meets thresholds before allowing commit.

**Checks:**
- All tests pass (configurable)
- Code coverage meets minimum (optional)
- New warnings within limit

**Configuration:**
```python
from aragora.nomic.gates import TestQualityGate

gate = TestQualityGate(
    enabled=True,
    require_all_tests_pass=True,
    min_coverage=80.0,      # Minimum coverage percentage
    max_new_warnings=0,     # Maximum new warnings allowed
)

decision = await gate.require_approval(
    artifact=test_output,
    context={
        "tests_passed": True,
        "coverage": 85.0,
        "warnings_count": 0
    }
)
```

### CommitGate

Final approval before committing changes.

**Features:**
- Structured diff view
- Rollback information
- Web UI option (via callback)
- CLI fallback

**Configuration:**
```python
from aragora.nomic.gates import CommitGate
from pathlib import Path

gate = CommitGate(
    enabled=True,
    aragora_path=Path("/path/to/aragora"),
    web_ui_callback=my_approval_callback,  # Optional
)

decision = await gate.require_approval(
    artifact=commit_message,
    context={
        "files_changed": ["src/a.py", "src/b.py"],
        "improvement_summary": "Add rate limiting"
    }
)
```

### Approval Decisions

Each gate produces an `ApprovalDecision`:

```python
@dataclass
class ApprovalDecision:
    gate_type: GateType          # design, test_quality, commit
    status: ApprovalStatus       # pending, approved, rejected, skipped
    timestamp: datetime
    approver: str                # human, auto, system, web_ui
    artifact_hash: str           # SHA-256 hash of approved content
    reason: str
    metadata: Dict[str, Any]
```

### Standard Gate Configuration

Use `create_standard_gates()` for common configurations:

```python
from aragora.nomic.gates import create_standard_gates, GateType

gates = create_standard_gates(
    aragora_path=Path.cwd(),
    dev_mode=True,  # Enable dev mode auto-approval
)

design_gate = gates[GateType.DESIGN]
quality_gate = gates[GateType.TEST_QUALITY]
commit_gate = gates[GateType.COMMIT]
```

## Audit Logging

The audit system provides a comprehensive event trail for all Nomic loop operations, stored in SQLite for queryability and durability.

### Event Types

| Category | Events |
|----------|--------|
| Lifecycle | `cycle_start`, `cycle_end`, `cycle_abort` |
| Phase | `phase_start`, `phase_end`, `phase_error` |
| Gate | `gate_check`, `gate_approved`, `gate_rejected`, `gate_skipped` |
| State | `state_transition`, `checkpoint_saved`, `checkpoint_loaded` |
| Safety | `constitution_check`, `protected_file_access`, `rollback` |

### AuditLogger

```python
from aragora.nomic.audit import AuditLogger, AuditEvent, AuditEventType

# Initialize logger
logger = AuditLogger(
    db_path=Path(".nomic/audit.db"),  # Default location
    enabled=True,
    max_events=100000,  # Older events auto-pruned
)

# Log events
logger.log_cycle_start("cycle_001", config={"agents": 3})
logger.log_phase_start("cycle_001", "debate")
logger.log_phase_end("cycle_001", "debate", success=True, duration_seconds=120.0)
logger.log_gate_decision(
    cycle_id="cycle_001",
    gate_type="design",
    status="approved",
    approver="human",
    artifact_hash="abc123...",
    reason="Design looks good"
)
logger.log_cycle_end("cycle_001", success=True, duration_seconds=300.0)
```

### Querying Events

```python
# Get events by cycle
events = logger.get_events(cycle_id="cycle_001", limit=100)

# Get events by type
gate_events = logger.get_events(event_type=AuditEventType.GATE_APPROVED)

# Get cycle summary
summary = logger.get_cycle_summary("cycle_001")
# {
#   "cycle_id": "cycle_001",
#   "found": True,
#   "success": True,
#   "duration_seconds": 300.0,
#   "phases_executed": ["context", "debate", "design", "implement", "verify", "commit"],
#   "gate_decisions": [{"type": "design", "status": "approved", "approver": "human"}],
#   "errors": [],
#   "total_events": 15
# }
```

### Global Logger

```python
from aragora.nomic.audit import get_audit_logger

# Get or create global logger
logger = get_audit_logger(enabled=True)
```

## State Machine

The Nomic loop uses an event-driven state machine for robust execution.

### States

| State | Description | Checkpointed |
|-------|-------------|--------------|
| `IDLE` | Waiting for trigger | No |
| `CONTEXT` | Gathering codebase context | Yes |
| `DEBATE` | Multi-agent debate | Yes |
| `DESIGN` | Designing implementation | Yes |
| `IMPLEMENT` | Writing code | Yes |
| `VERIFY` | Running tests | Yes |
| `COMMIT` | Committing changes | Yes |
| `RECOVERY` | Handling errors | Yes |
| `COMPLETED` | Cycle completed | Yes |
| `FAILED` | Cycle failed | Yes |
| `PAUSED` | Manually paused | Yes |

### State Context

State context is passed between phases and checkpointed:

```python
from aragora.nomic.states import StateContext, NomicState

ctx = StateContext(
    cycle_id="cycle_001",
    current_state=NomicState.DEBATE,
    previous_state=NomicState.CONTEXT,
)

# Phase results accumulated
ctx.context_result = {"files_analyzed": 50}
ctx.debate_result = {"consensus": "implement_caching"}

# Serialize for checkpointing
data = ctx.to_dict()

# Restore from checkpoint
ctx = StateContext.from_dict(data)
```

### Recovery System

The recovery system handles errors with circuit breakers and intelligent strategies.

**Recovery Strategies:**
| Strategy | Description |
|----------|-------------|
| `RETRY` | Retry the same state |
| `SKIP` | Skip to next state |
| `ROLLBACK` | Rollback to previous state |
| `RESTART` | Restart from beginning |
| `PAUSE` | Pause for human intervention |
| `FAIL` | Mark as failed, stop |

**Circuit Breakers:**
```python
from aragora.nomic.recovery import CircuitBreaker, CircuitBreakerRegistry

# Create circuit breaker
breaker = CircuitBreaker(
    name="claude_agent",
    failure_threshold=3,
    reset_timeout_seconds=300,
)

# Check if open
if breaker.is_open:
    print("Agent unavailable")
else:
    try:
        result = await agent.run()
        breaker.record_success()
    except Exception:
        breaker.record_failure()

# Registry for multiple agents
registry = CircuitBreakerRegistry()
claude_breaker = registry.get_or_create("claude", failure_threshold=3)
gpt_breaker = registry.get_or_create("gpt4", failure_threshold=5)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_DEV_MODE` | Enable dev mode (auto-approve gates) | `0` |
| `ARAGORA_SKIP_GATES` | Skip all gates (dangerous) | `0` |
| `NOMIC_AUTO_APPROVE` | Auto-approve in non-interactive mode | `0` |
| `NOMIC_AUTO_COMMIT` | Auto-commit without approval | `0` |

**Safety Note:** Production deployments should never set `ARAGORA_SKIP_GATES=1` or `NOMIC_AUTO_COMMIT=1`.

## Running the Nomic Loop

### CLI

```bash
# Run with streaming output
python scripts/run_nomic_with_stream.py run --cycles 3

# Run single cycle
python scripts/nomic_loop.py
```

### Programmatic

```python
from aragora.nomic import (
    NomicStateMachine,
    create_standard_gates,
    get_audit_logger,
)

# Create state machine
sm = NomicStateMachine()

# Configure gates
gates = create_standard_gates(dev_mode=False)

# Run cycle
result = await sm.run_cycle(
    gates=gates,
    audit_logger=get_audit_logger(),
)
```

## Best Practices

1. **Never disable gates in production** - Gates are your safety net
2. **Review audit logs regularly** - Check for anomalies
3. **Use circuit breakers** - Protect against agent failures
4. **Checkpoint frequently** - Enable recovery from failures
5. **Human approval for critical changes** - Don't auto-commit in production
6. **Test in dev mode first** - Use `ARAGORA_DEV_MODE=1` for testing

## Troubleshooting

### Gate Approval Stuck

If gates are blocking in non-interactive mode:
```bash
# Enable auto-approval (dev only!)
export NOMIC_AUTO_APPROVE=1
```

### Circuit Breaker Open

If an agent's circuit breaker is open:
```python
registry = CircuitBreakerRegistry()
breaker = registry.get_or_create("agent_name")
breaker.reset()  # Manual reset
```

### Recovery from Checkpoint

If a cycle fails, resume from checkpoint:
```python
from aragora.nomic.checkpoints import load_latest_checkpoint

checkpoint = load_latest_checkpoint()
if checkpoint:
    # Resume from checkpoint
    sm = NomicStateMachine()
    result = await sm.resume_from_checkpoint(checkpoint)
```

## See Also

- [ADMIN.md](ADMIN.md) - Admin console guide
- [RUNBOOK.md](RUNBOOK.md) - Operational procedures
- [API_REFERENCE.md](API_REFERENCE.md) - Full API documentation
