# Aragora Governance

**Policy enforcement, RBAC, and audit logging for the multi-agent deliberation control plane**

How rule changes are processed, who signs off, and how to recover from bad updates. This document covers the governance framework for Aragora's control plane, ensuring compliance, security, and accountability for enterprise deployments.

## Governance Philosophy

Aragora operates on a principle borrowed from the game of Nomic: **the rules of the system can be changed by the system itself, under constraints**. This enables continuous improvement while preventing runaway self-modification.

The key tension is:
- **Flexibility**: The system should improve itself over time
- **Safety**: Changes must not break the system or bypass safety controls
- **Auditability**: All changes must be traceable and reversible

We resolve this tension through **layered governance**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  IMMUTABLE (Constitution)                                                   │
│  Cryptographically signed rules that cannot be changed by agents            │
├─────────────────────────────────────────────────────────────────────────────┤
│  AMENDABLE (Policies)                                                       │
│  Rules that can be changed with sufficient consensus (75%+)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  ADVISORY (Guidelines)                                                      │
│  Suggestions that guide behavior but don't block actions                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## The Constitution

The Constitution (`.nomic/constitution.json`) defines rules that the nomic loop **cannot** modify, enforced via cryptographic signatures.

### Immutable Rules

These rules cannot be changed by any agent process:

| Rule ID | Rule | Rationale |
|---------|------|-----------|
| CORE-001 | Never delete backup/restore mechanism | Rollback is the last line of defense |
| CORE-002 | Never modify Constitution verification code | Prevents self-disabling of safety |
| CORE-003 | All changes must pass existing tests | Prevents breaking changes |
| EVAL-001 | Judging must include objective metrics | Prevents wireheading via subjective-only evaluation |
| SAFETY-001 | Changes to protected files require human approval | Maintains human oversight |

### Protected Files

These files cannot be modified by agents without human approval:
- `scripts/nomic/safety/constitution.py`
- `scripts/nomic/safety/checksums.py`
- `scripts/nomic/safety/backups.py`
- `.nomic/constitution.json`

### Constitution Verification

Before each nomic cycle:
1. The Constitution's Ed25519 signature is verified
2. If invalid, the cycle aborts
3. The signature is created offline and cannot be forged by agents

```bash
# Verify Constitution integrity
python scripts/sign_constitution.py verify

# Output: Constitution signature is VALID
```

### Amending the Constitution

Amendable rules (like "minimum 3 agents for consensus") can be changed through a supermajority process:

1. **Proposal**: An agent proposes a rule change
2. **Debate**: All agents debate the change for 3+ rounds
3. **Vote**: Requires 75%+ consensus (weighted by ELO)
4. **Human Review**: Human must approve amendable rule changes
5. **Re-sign**: Constitution is re-signed offline with the new rule

**Immutable rules cannot be amended** - they require forking the project.

## Policy Engine

The Policy Engine (`aragora/policy/`) enforces per-tool and per-task policies:

### Risk Levels

| Level | Description | Example |
|-------|-------------|---------|
| NONE | No side effects | Read file |
| LOW | Easily reversible | Create file |
| MEDIUM | Reversible with effort | Write existing file |
| HIGH | May need human review | Delete file |
| CRITICAL | Requires human approval | Deploy to production |

### Blast Radius

| Level | Description | Example |
|-------|-------------|---------|
| READ_ONLY | No mutations | Git diff |
| DRAFT | Changes discardable | Local buffer |
| LOCAL | Reversible via git | File changes |
| SHARED | Affects team/staging | Push to branch |
| PRODUCTION | Affects live users | Deploy |

### Risk Budgets

Each nomic cycle gets a risk budget (default: 100). Actions consume budget:

```
cost = risk_level × (blast_radius + 1) × multiplier
```

When budget exceeds threshold (80), human approval is required.

### Example Policy

```python
Policy(
    name="protect_core_files",
    description="Prevent modification of core Aragora files",
    tools=["file_writer"],
    capabilities=["write_file", "delete_file"],
    conditions=["'aragora/core.py' in file_path"],
    allow=False,  # Denied
    priority=100,
)
```

## Rule Change Process

### Step 1: Proposal

Agent proposes a change through the nomic loop:

```
Topic: "Add new evaluation metric: evidence coverage"
Impact: MEDIUM
Files: aragora/debate/trickster.py, tests/test_trickster.py
```

### Step 2: Debate

Agents debate the change:
- **Proposer**: Explains rationale
- **Skeptic**: Challenges assumptions
- **Judge**: Evaluates arguments

### Step 3: Design

If debate reaches consensus, agents design the implementation:
- Architecture decisions
- Test plan
- Rollback plan

### Step 4: Implementation

Winning agent implements the change:
- Protected files trigger human approval
- Risk budget is consumed
- All changes are backed up

### Step 5: Verification

Before commit:
1. Syntax check (`python -m py_compile`)
2. Import check (`python -c "import aragora"`)
3. Test suite (`pytest`)
4. Constitution verification

### Step 6: Commit or Rollback

- **If all checks pass**: Changes are committed with detailed message
- **If any check fails**: Changes are rolled back from backup

## Recovery Procedures

### Automatic Rollback

If verification fails, the nomic loop automatically rolls back:

```python
# Automatic rollback triggered by:
# - Test failures
# - Import errors
# - Constitution violations
# - Risk budget exceeded without approval
```

### Manual Recovery

If a bad change slips through:

```bash
# View recent changes
git log --oneline -20

# Revert specific commit
git revert <commit-hash>

# Restore from backup
python scripts/nomic/safety/backups.py restore --timestamp <time>
```

### Emergency Stop

To halt the nomic loop immediately:

```bash
# Kill running process
pkill -f "nomic_loop"

# Or set circuit breaker
echo '{"state": "open"}' > .nomic/circuit_breaker.json
```

### Investigating Failures

```bash
# View nomic loop logs
cat .nomic/replays/nomic-cycle-*/meta.json

# Check outcome history
sqlite3 .nomic/outcomes.db "SELECT * FROM outcomes ORDER BY timestamp DESC LIMIT 10"

# Review audit trail
cat .nomic/audit.log
```

## Human Oversight

### When Human Approval is Required

1. **Protected file modification**: Any change to files in `protected_files`
2. **Risk budget threshold**: When cumulative risk exceeds 80%
3. **Critical actions**: Production deployments, git push, database deletions
4. **Constitution amendments**: Changing amendable rules

### Approval Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Agent       │────►│ Policy       │────►│ Human       │
│ requests    │     │ Engine       │     │ approves    │
│ action      │     │ escalates    │     │ or denies   │
└─────────────┘     └──────────────┘     └─────────────┘
```

### Monitoring Dashboard

The live dashboard (`https://aragora.ai`) shows:
- Current nomic cycle status
- Pending approvals
- Risk budget utilization
- Recent changes and rollbacks

## Audit Trail

All nomic loop actions are logged:

```json
{
  "timestamp": "2026-01-09T21:45:00Z",
  "cycle_id": "nomic-cycle-42",
  "phase": "implement",
  "agent": "anthropic-api",
  "action": "write_file",
  "file": "aragora/debate/trickster.py",
  "risk_cost": 8.0,
  "budget_remaining": 72.0,
  "outcome": "success"
}
```

### Audit Storage

| Location | Contents |
|----------|----------|
| `.nomic/replays/` | Full debate transcripts |
| `.nomic/outcomes.db` | Implementation outcomes |
| `.nomic/audit.log` | Policy engine decisions |
| `git log` | All committed changes |

## Best Practices

### For Operators

1. **Review before auto-commit**: Start with `--human-approval` flag
2. **Monitor risk budgets**: Watch for sessions approaching threshold
3. **Test rollback**: Periodically verify backup/restore works
4. **Update Constitution**: Re-sign after any legitimate changes

### For Developers

1. **Add tests first**: New features need tests before implementation
2. **Small changes**: Prefer many small changes over few large ones
3. **Document rationale**: Explain why, not just what
4. **Respect protected files**: Request human review when needed

### For Security

1. **Keep private key secure**: `ARAGORA_CONSTITUTION_KEY` must not be exposed
2. **Monitor Constitution**: Alert on signature verification failures
3. **Review audit logs**: Check for unusual patterns
4. **Test adversarially**: Use capability probes and red team mode

## FAQ

**Q: Can agents modify the Constitution verification code?**
A: No. It's protected by CORE-002 and the file is in `protected_files`.

**Q: What happens if all agents conspire to bypass safety?**
A: The Constitution signature prevents this - agents can't forge signatures without the offline private key.

**Q: How do I add a new immutable rule?**
A: Immutable rules can only be added by forking the project. Amendable rules can be added through the normal process.

**Q: What if I need to make an emergency fix to a protected file?**
A: Use the `--human-approval` flag or manually edit and re-sign the Constitution.

**Q: How do I know if the system is trustworthy?**
A: Check Constitution signature, review audit logs, monitor outcome tracker calibration curves.
