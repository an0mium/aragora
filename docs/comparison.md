# Aragora Competitive Comparison

How Aragora's adversarial validation approach compares to alternatives for reviewing technical decisions, specifications, and architectures.

## Quick Comparison

| Capability | Aragora | Multi-Agent Frameworks | Manual Code Review | Traditional Pentesting |
|------------|---------|----------------------|-------------------|----------------------|
| **Primary approach** | Adversarial debate | Cooperative task execution | Human expertise | Manual security testing |
| **Time to results** | 15-45 minutes | Hours | Days to weeks | Weeks to months |
| **Cost per review** | ~$5-50 | ~$3-30 | $10,000-100,000 | $25,000-150,000 |
| **Audit-ready artifacts** | Yes (DecisionReceipt) | Limited | Varies | Report-based |
| **Regulatory personas** | 8 built-in | DIY | N/A | Some |
| **Heterogeneous models** | Yes (required) | Optional | N/A | N/A |
| **Dissent tracking** | Yes | No | Informal | No |
| **Real-time streaming** | Yes | Limited | N/A | No |
| **Transaction rollback** | Yes (cross-system) | No | N/A | N/A |
| **ELO agent ranking** | Built-in | No | N/A | N/A |
| **Self-improvement** | Nomic Loop | No | N/A | N/A |

## Detailed Analysis

### Aragora vs. Multi-Agent Frameworks (CrewAI, AutoGen, LangGraph)

**Key Difference:** Aragora is adversarial-first; most frameworks are cooperative-first.

| Aspect | Aragora | Typical Multi-Agent Framework |
|--------|---------|-------------------------------|
| **Agent relationship** | Adversarial - agents try to find flaws | Cooperative - agents work toward shared goal |
| **Consensus mechanism** | Formal consensus with dissent tracking | Task completion (pass/fail) |
| **Output artifact** | DecisionReceipt with evidence chain | Task result or conversation log |
| **Model diversity** | Required (heterogeneous debate) | Optional (often single provider) |
| **Regulatory compliance** | 8 pre-built personas (GDPR, HIPAA, SOX, etc.) | Build your own |
| **Formal verification** | Z3 SMT solver integration | None |

**When to use Aragora:**
- Validating decisions before commitment
- Regulatory compliance review
- Finding flaws in specifications
- Need audit-ready evidence

**When to use cooperative frameworks:**
- Task automation (data pipelines, workflows)
- Content generation
- Customer support agents
- Orchestrating tool calls

### Aragora vs. Manual Expert Review

**Key Difference:** Aragora provides speed and consistency; experts provide depth and context.

| Aspect | Aragora | Manual Expert Review |
|--------|---------|---------------------|
| **Time to results** | 15-45 minutes | Days to weeks |
| **Cost** | ~$5-50 per run | $10,000-100,000 |
| **Scalability** | Unlimited parallel runs | Constrained by availability |
| **Consistency** | Same rigor every time | Varies by reviewer |
| **Regulatory currency** | Updated with model knowledge | Depends on expert |
| **Novel insights** | Limited to training data | Can identify unprecedented issues |
| **Context understanding** | Spec-limited | Deep organizational context |
| **Accountability** | Cryptographic provenance | Professional liability |

**Recommended approach:** Use Aragora for first-pass validation to identify obvious issues quickly, then engage experts for complex judgment calls and organizational context.

### Aragora vs. Traditional Penetration Testing

**Key Difference:** Aragora reviews specifications; pentesting tests running systems.

| Aspect | Aragora | Traditional Pentesting |
|--------|---------|----------------------|
| **What's tested** | Specs, designs, architecture docs | Running systems and code |
| **When in lifecycle** | Design phase (shift-left) | After implementation |
| **Time to results** | 15-45 minutes | 2-6 weeks |
| **Cost** | ~$5-50 | $25,000-150,000 |
| **Exploitability proof** | Theoretical (finding-based) | Practical (exploit-based) |
| **Compliance coverage** | 8 regulatory frameworks | Security-focused |
| **Retesting** | Instant re-run after changes | Re-engagement required |

**Recommended approach:** Use Aragora during design to catch issues before they're built. Use pentesting before launch to validate implementation and find runtime vulnerabilities.

## Feature Deep Dive

### DecisionReceipt (Unique to Aragora)

Every Gauntlet run produces a cryptographically signed DecisionReceipt containing:

- **Verdict:** APPROVED / APPROVED_WITH_CONDITIONS / REJECTED
- **Confidence score:** 0.0-1.0 based on model agreement
- **Evidence chain:** Every finding with supporting quotes from each model
- **Dissent record:** Minority opinions when models disagree
- **Provenance hash:** Tamper-evident record of inputs and outputs

This artifact is designed for audit trails, compliance evidence, and decision documentation.

### Regulatory Personas

Pre-built adversarial personas for compliance stress-testing:

| Persona | Regulation | Attack Categories |
|---------|------------|-------------------|
| GDPR | EU 2016/679 | Legal basis, data minimization, rights, transfers, security, privacy by design |
| HIPAA | US Health Insurance | PHI protection, access controls, audit logging, encryption, BAAs |
| SOX | Sarbanes-Oxley | ITGC, access management, change management, audit trails, material weakness |
| AI Act | EU AI Regulation | Risk classification, transparency, human oversight, data governance |
| SOC 2 | Trust Services | Security, availability, processing integrity, confidentiality, privacy |
| PCI-DSS | Payment Card Industry | Cardholder data, access control, network security, vulnerability management |
| NIST CSF | Cybersecurity Framework | Identify, protect, detect, respond, recover |
| Security | Red Team | Injection, authentication, access control, data exposure, API security |

### Heterogeneous Model Debate

Aragora requires multiple different AI models to participate in debates:

- **Why it matters:** Single-model reviews have correlated blind spots. Different models catch different issues.
- **Supported models:** Claude (Anthropic), GPT-4 (OpenAI), Gemini (Google), Mistral, and others via OpenRouter
- **Consensus detection:** Formal agreement tracking across models with configurable thresholds

### Transaction-Safe Memory Coordination (Unique)

Aragora provides atomic writes across multiple memory systems with automatic rollback:

| System | Write Method | Rollback Method |
|--------|-------------|-----------------|
| Continuum | `store_pattern()` | `delete(archive=True)` |
| Consensus | `store_consensus()` | `delete_consensus(cascade=True)` |
| Critique | `store_result()` | `delete_debate(cascade=True)` |
| Knowledge Mound | `ingest_outcome()` | `delete_item()` |

```python
from aragora.memory.coordinator import MemoryCoordinator, CoordinatorOptions

coordinator = MemoryCoordinator(
    continuum_memory=continuum,
    consensus_memory=consensus,
    critique_store=critique,
    knowledge_mound=mound,
)

# If any write fails, all previous writes are rolled back
tx = await coordinator.commit_debate_outcome(
    context,
    options=CoordinatorOptions(rollback_on_failure=True),
)

if tx.partial_failure:
    print(f"Rolled back: {tx.rolled_back}")
```

Monitor via API:
```bash
curl http://localhost:8080/api/memory/coordinator/metrics
# Returns: success_rate, rollbacks_performed, memory_systems status
```

### ELO-Based Agent Selection (Unique)

Track agent performance across debates with skill-based ranking:

```python
from aragora.ranking.elo import EloRankingSystem

elo = EloRankingSystem()
elo.record_outcome(debate_id="d-123", winner="claude", loser="gpt-4")

# Rankings influence team selection for future debates
rankings = elo.get_rankings()  # {"claude": 1520, "gpt-4": 1480, ...}
```

### Nomic Loop Self-Improvement (Unique)

Autonomous cycle where agents debate, design, implement, and verify improvements:

```bash
python scripts/nomic_loop.py --cycles 3
```

| Phase | Description |
|-------|-------------|
| Context | Gather codebase understanding |
| Debate | Agents propose improvements |
| Design | Architecture planning |
| Implement | Code generation |
| Verify | Tests and verification |

No other framework includes autonomous self-improvement capabilities.

## Pricing Comparison

Estimated costs for reviewing a medium-complexity system specification:

| Approach | One-Time Cost | Ongoing (Monthly) |
|----------|--------------|-------------------|
| Aragora Gauntlet | $15-50 | $15-50 per re-run |
| Multi-agent framework | $10-30 (DIY setup required) | $10-30 |
| Manual architecture review | $15,000-40,000 | N/A (re-engagement) |
| SOC 2 Type II audit | $30,000-100,000 | Annual re-audit |
| Penetration test | $25,000-150,000 | Annual re-test |

## When Aragora Is Not the Right Choice

- **Runtime vulnerability testing:** Aragora reviews specs, not running systems
- **Code-level analysis:** Use static analysis tools (SAST) for actual code
- **Compliance certification:** Aragora provides evidence but isn't an auditor
- **Novel attack research:** Human security researchers find 0-days better
- **Highly contextual decisions:** Domain experts understand organizational nuance

## Integration Options

Aragora fits into existing workflows:

- **CLI:** `aragora gauntlet spec.md --persona security --profile thorough`
- **API:** REST + WebSocket for integration with CI/CD, dashboards
- **Python SDK:** Programmatic access for custom workflows

```python
from aragora import Gauntlet

result = await Gauntlet.run(
    input_path="api-spec.md",
    persona="gdpr",
    profile="thorough"
)

if result.verdict == "REJECTED":
    print(f"Found {result.critical_findings} critical issues")
```

## Summary

| Use Case | Best Choice |
|----------|-------------|
| Quick validation of specs/designs | Aragora |
| Regulatory compliance pre-check | Aragora |
| Automated task workflows | Multi-agent framework |
| Deep security testing of running systems | Penetration testing |
| Complex judgment requiring context | Human experts |
| Compliance certification | Certified auditors (use Aragora for prep) |
