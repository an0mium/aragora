# Aragora Landing Page - Features Section

## Core Features

### 1. Gauntlet Mode
**Adversarial validation pipeline with configurable attack categories**

Subject any decision artifact to systematic stress-testing:
- Product specifications
- System architectures
- Security policies
- Legal contracts
- Business strategies

**Attack Categories:**
- Security (injection, privilege escalation)
- Compliance (GDPR, HIPAA, AI Act)
- Logic (edge cases, assumptions, stakeholder conflicts)
- Architecture (scalability, performance, dependencies)

**Output:**
- Risk heatmap showing category x severity
- Ranked findings with mitigation recommendations
- Confidence-scored verdict (Pass / Conditional / Fail)

---

### 2. Regulatory Stress-Testing
**Pre-configured personas for compliance validation**

Built-in regulatory personas simulate how auditors will challenge your systems:

| Persona | Regulation | Focus Areas |
|---|---|---|
| **GDPR Auditor** | EU GDPR | Data minimization, consent, right to erasure |
| **HIPAA Inspector** | US HIPAA | PHI protection, access controls, audit logs |
| **AI Act Assessor** | EU AI Act | High-risk classification, transparency, human oversight |
| **Security Researcher** | OWASP/CWE | Injection, auth bypass, data exposure |

**Value:** Identify compliance gaps before regulators do. Avoid EUR10M+ fines.

---

### 3. Audit-Ready Decision Receipts
**Tamper-evident documentation for every validation**

Each Gauntlet run generates a Decision Receipt:

```
DECISION RECEIPT
================
Receipt ID: receipt-abc123def456
Input Hash: SHA256:a1b2c3d4...
Timestamp: 2026-01-11T12:00:00Z (signed)

Verdict: APPROVED_WITH_CONDITIONS
Confidence: 87%

Findings Summary:
- Critical: 0
- High: 2
- Medium: 5
- Low: 8

Attacks Attempted: 47
Attacks Successful: 7
Probes Run: 24
Vulnerabilities Found: 15

Integrity Check: sha256:...
```

**Value:** Prove due diligence in audits, litigation, and board reviews.

---

### 4. Multi-Agent Debate Engine
**Heterogeneous AI perspectives prevent blind spots**

Unlike single-LLM tools, Aragora orchestrates debates between:
- Claude (Anthropic)
- GPT-4 (OpenAI)
- Gemini (Google)
- Grok (xAI)
- Mistral (Mistral AI)
- DeepSeek
- Qwen (Alibaba)
- And more...

Each agent attacks from a different angle. Consensus requires surviving all perspectives.

---

### 5. Formal Verification
**Z3 SMT solver for provable claims**

For claims that must be mathematically certain, Aragora attempts formal proofs:
- Logical consistency
- Invariant preservation
- Policy completeness

**Output:** Proof status (verified, falsified, undecidable) with proof hash.

---

### 6. Real-Time Streaming
**Watch debates unfold live**

WebSocket-based streaming shows:
- Agent reasoning in real-time
- Attack attempts and defenses
- Consensus formation
- Confidence evolution

Integrate with Slack, Discord, or custom dashboards.

---

## Integration Points

### CLI
```bash
pip install aragora
aragora gauntlet run spec.md --persona gdpr
```

### API
```python
from aragora import GauntletRunner

runner = GauntletRunner(config)
result = await runner.run(spec_content)
print(result.verdict)
```

### Webhooks
```json
{
  "event": "gauntlet.completed",
  "gauntlet_id": "gauntlet-abc123",
  "verdict": "approved_with_conditions",
  "findings_count": 15
}
```

---

## Deployment Options

| Option | Description | Best For |
|---|---|---|
| **SaaS** | Hosted by Aragora | Quick start, no ops |
| **Self-Hosted** | Docker/Kubernetes | Data residency requirements |
| **Hybrid** | Agents hosted, your data local | Balance of convenience + control |
