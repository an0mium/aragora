# Aragora Landing Page - Features Section

## Core Features

### 1. Control Plane for Deliberation
**Orchestrate agent fleets, queues, and governance** across high-stakes decisions.

- Agent registry + health monitoring
- Priority task queues
- Policy enforcement and audit logging
- Deliberations as first-class tasks

---

### 2. Multi-Agent Deliberation Engine
**Heterogeneous AI perspectives prevent blind spots** by design.

Agents debate, critique, and synthesize outputs with evidence chains and
confidence scoring.

---

### 3. Omnivorous Ingestion + Multi-Channel Delivery
**Ingest anything, deliver anywhere.**

- 25+ document formats + databases + APIs
- Email, chat, voice, and web inputs
- Results routed back to Slack, Teams, email, or voice

---

### 4. Decision Receipts
**Audit-ready documentation for every decision.**

Each robust decisionmaking session produces a tamper-evident receipt with evidence trails,
dissent tracking, and provenance metadata.

---

### 5. Gauntlet Mode (Decision Stress-Test)
**Adversarial validation pipeline with configurable attack categories.**

Subject any decision artifact to systematic stress-testing:
- Product specs, architectures, policies, contracts
- Security, compliance, logic, performance
- Risk heatmaps + remediation recommendations

---

### 6. Formal Verification
**Z3-backed proofs** for claims that must be mathematically certain.

---

### 7. Real-Time Streaming
**Watch robust decisionmaking sessions unfold live** with WebSocket updates and dashboards.

---

## Integration Points

### CLI
```bash
pip install aragora
aragora gauntlet spec.md --persona gdpr
```

### API
```python
from aragora.control_plane import ControlPlaneCoordinator

coordinator = ControlPlaneCoordinator()
# submit tasks or robust decisionmaking sessions via API
```

### Webhooks
```json
{
  "event": "deliberation.completed",
  "deliberation_id": "delib-abc123",
  "confidence": 0.87
}
```

---

## Deployment Options

| Option | Description | Best For |
|---|---|---|
| **SaaS** | Hosted by Aragora | Quick start, no ops |
| **Self-Hosted** | Docker/Kubernetes | Data residency requirements |
| **Hybrid** | Agents hosted, your data local | Balance of convenience + control |
