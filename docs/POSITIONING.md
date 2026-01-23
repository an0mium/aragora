# Aragora: Market Positioning & Go-to-Market Strategy

*Strategic positioning for B2B market entry*

**Last Updated:** January 2026

---

## Executive Summary

**Aragora is the control plane for multi-agent robust decisionmaking across organizational knowledge and channels.** It orchestrates 15+ AI models to debate your organization's knowledge (documents, databases, APIs) and deliver defensible decisions to any channel (Slack, Teams, Discord, voice).

> "Aragora: Control plane for multi-agent robust decisionmaking across org knowledge and channels."

Deliberation is the internal engine. The product is a defensible decision record with full audit trails.

**For high-stakes teams** in legal, finance, compliance, and security—where "the AI said so" isn't good enough—Aragora provides decision assurance with receipts.

---

## What Aragora Is NOT

- **Not a consumer chatbot** - An enterprise control plane, not a casual assistant
- **Not a wrapper** - Not another LLM API aggregator
- **Not a general debate product** - Debate is internal, outputs are outcomes
- **Not autonomous code change** - The Nomic loop is experimental and review-gated

---

## Key Differentiators vs Competition

| Capability | Aragora | Multi-Agent Frameworks | Human Review | Traditional Pentesting |
|------------|---------|------------------------|--------------|------------------------|
| Adversarial Validation | Yes | Limited | Yes | Runtime only |
| Audit Artifact | DecisionReceipt | No | Reports | Reports |
| Multi-Model Dissent | Required | Optional | N/A | N/A |
| Formal Verification | Z3/Lean | No | Rare | No |
| Regulatory Personas | 8 built-in | DIY | Expert-dependent | N/A |
| Learning Memory | 4-tier Continuum | Limited | Tribal knowledge | None |

---

## Ideal Customer Profile (ICP)

### Tier 1: Primary ICP (Best Fit)

**Titles:** VP Engineering, CTO, Head of Security, Chief Compliance Officer

**Company Profile:**
- Series B+ startups or mid-market ($10M-$500M ARR)
- Industries: FinTech, HealthTech, SaaS, Enterprise Software
- Teams of 50-500 engineers
- SOC 2, HIPAA, or GDPR compliance requirements
- Ships weekly or faster

**Pain Points:**
1. Architecture review bottleneck - senior engineers are the constraint
2. Pre-commit compliance anxiety - "Will this pass audit?"
3. Post-mortem fatigue - learning from incidents but not preventing them
4. Design doc purgatory - specs sit in review for weeks

**Buying Triggers:**
- Recent compliance audit finding
- Senior engineer attrition
- Failed security review or penetration test
- Acquisition due diligence preparation
- New regulatory requirement (AI Act, state privacy laws)

**Budget Authority:** $50K-$200K annual software spend

### Tier 2: Secondary ICP (Good Fit)

**Titles:** Director of Engineering, Security Architect, DevSecOps Lead

**Company Profile:**
- Enterprise ($500M+ ARR) with autonomous product teams
- Heavy API exposure (B2B2C platforms)
- Regulated industries: Banking, Insurance, Healthcare
- Multiple compliance frameworks simultaneously

**Pain Points:**
1. Inconsistent review quality across teams
2. Tribal knowledge loss when reviewers leave
3. Compliance evidence generation is manual
4. Different teams make the same mistakes

**Budget Authority:** $200K-$1M annual software spend

### Anti-ICP (Poor Fit)

**Avoid:**
- Pre-seed/seed startups (no compliance pressure yet)
- Agencies doing one-off projects (no recurring need)
- Non-technical buyers expecting magic
- Companies with "no AI" policies
- Heavily outsourced engineering teams

---

## Workflow Wedge Strategy

### Entry Point: Design Review Gate

The highest-leverage insertion point is between "design approved" and "implementation started":

1. Changes are still cheap (no code written yet)
2. Stakes are highest (architectural decisions compound)
3. Pain is acute (review bottleneck is visible)
4. Audit trail is valuable (compliance requires documentation)

### Implementation Path

```
BEFORE ARAGORA                          AFTER ARAGORA
─────────────                           ─────────────

[Design Doc] ──► [Manual Review] ──► [Approval]
                   (days-weeks)     Single perspective

════════════════════════════════════════════════════════

[Design Doc] ──► [Aragora Gauntlet] ──► [Human Review] ──► [Approval]
                   (30 minutes)       Risk-annotated    Decision
                Multi-model           pre-screened      Receipt
                stress-test           document          generated
```

### Expansion Path

| Phase | Timeframe | Capability | Integration |
|-------|-----------|------------|-------------|
| **Land** | Month 1-3 | Design Gate | CLI or file upload |
| **Expand** | Month 4-6 | PR Review | GitHub/GitLab CI |
| **Deepen** | Month 7-12 | Compliance Automation | Scheduled scans |
| **Consolidate** | Year 2+ | Platform | Custom personas, dashboards |

---

## Value Propositions by Persona

### For Engineering Leaders
> "Stop being the review bottleneck. Aragora gives every design the 5-perspective analysis you'd do yourself—before it hits your desk."

- 80% reduction in design review cycle time
- 10x more designs reviewed per senior engineer
- Zero increase in post-deploy incidents

### For Security Teams
> "Red-team every spec before a single line of code is written. Aragora finds the vulnerabilities that make it to production."

- 60% of vulnerabilities caught at design (vs. 20% industry average)
- $150K average cost avoidance per prevented incident
- 100% of high-risk designs stress-tested

### For Compliance Officers
> "From 'trust us, we reviewed it' to 'here's the cryptographic proof.' Aragora makes compliance evidence automatic."

- 4x faster audit preparation
- 100% of decisions documented with evidence chain
- Zero manual evidence compilation

---

## Messaging Framework

### Tagline Options
1. "Control plane for multi-agent robust decisionmaking across org knowledge and channels."
2. "The multi-agent control plane for defensible decisions."
3. "Decision assurance from any source, delivered anywhere."

### One-Liner
"Aragora orchestrates multi-agent robust decisionmaking across your org’s knowledge to deliver defensible, audit-ready decisions to every channel."

### Elevator Pitch (30 seconds)
"When you make a high-stakes technical decision—a new architecture, API design, or policy change—how do you know you've considered all the angles? Aragora is the control plane for multi-agent robust decisionmaking: it runs structured debates across heterogeneous AI models, stress-testing decisions from security, compliance, scalability, and maintainability perspectives. You get a cryptographic Decision Receipt documenting the analysis—perfect for audits, post-mortems, and decision records."

---

## Competitive Positioning

### vs. Manual Review
| Dimension | Manual Review | Aragora |
|-----------|--------------|---------|
| Speed | Days-weeks | 30 minutes |
| Consistency | Varies by reviewer | Same rigor every time |
| Scalability | Limited by headcount | Unlimited parallel |
| Documentation | Manual notes | Cryptographic receipt |

**Positioning:** "Aragora doesn't replace your senior engineers—it multiplies them."

### vs. LLM Chat (ChatGPT/Claude)
| Dimension | Single LLM | Aragora |
|-----------|-----------|---------|
| Perspective | One model | 15+ heterogeneous models |
| Rigor | Prompt-dependent | Structured adversarial protocol |
| Audit trail | Copy/paste | Cryptographic receipt |
| Regulatory | DIY | 8 pre-built personas |

**Positioning:** "Aragora turns 'I asked ChatGPT' into auditable validation."

### vs. CrewAI/AutoGen/LangGraph
| Dimension | Cooperative Frameworks | Aragora |
|-----------|----------------------|---------|
| Paradigm | Cooperative | Adversarial |
| Output | Task completion | Decision validation |
| Audit | Limited | Cryptographic provenance |
| Compliance | DIY | Built-in personas |

**Positioning:** "They automate tasks. We validate decisions."

---

## Sales Motion

### Discovery Questions

1. "Walk me through your last design review that took longer than expected."
2. "When was the last time a production incident traced back to a design review decision?"
3. "How do you document why you chose one architecture over another?"
4. "What happens when your most experienced reviewer is on vacation?"
5. "How do you prepare for compliance audits today?"

### Objection Handling

**"We already do design reviews."**
> "Aragora is the first pass that catches obvious issues, so your senior engineers can focus on the subtle stuff only humans notice."

**"AI can't understand our context."**
> "Aragora learns from your past decisions. After a few months, it knows your patterns better than a new hire would."

**"We can just use ChatGPT."**
> "ChatGPT is one perspective that agrees with itself. Aragora runs 15 models that argue against each other. And when SOC 2 asks 'how did you validate this?' you can show a DecisionReceipt."

**"It's too expensive."**
> "An Aragora run costs $5-50. A senior engineer review costs $500-1000 in loaded salary. A production incident costs $50K-500K."

---

## Pricing Strategy

| Tier | Monthly | Per-Run | Included Runs | Annual |
|------|---------|---------|---------------|--------|
| **Starter** | $499 | $25 | 20 | $5,988 |
| **Team** | $1,999 | $15 | 150 | $23,988 |
| **Enterprise** | Custom | $8 | Unlimited | $100K+ |

### Enterprise Add-Ons
- SSO (OIDC/SAML)
- Multi-tenant isolation
- Custom personas
- Dedicated support
- SLA guarantees
- On-premise deployment

---

## Go-To-Market Phases

### Phase 1: Product-Led Growth (Month 1-6)
- Free tier: 5 runs/month
- Self-serve signup
- GitHub integration focus
- Content: "Design Review Best Practices"
- Community: Discord for power users

### Phase 2: Sales-Assisted (Month 7-12)
- Outbound to ICP accounts
- Case study development
- Partner ecosystem (consulting firms)
- SOC 2 certification
- Conference presence (QCon, StrangeLoop)

### Phase 3: Enterprise (Year 2)
- Dedicated enterprise sales
- Channel partnerships
- Federal compliance path
- International expansion

---

## ICP Validation Checklist

Before pursuing a prospect, confirm:

- [ ] Has compliance requirements (SOC 2, HIPAA, GDPR)
- [ ] Ships software at least monthly
- [ ] Has experienced design review bottlenecks
- [ ] Engineering team > 20 people
- [ ] Has budget authority in engineering/security
- [ ] Not anti-AI as organizational policy
- [ ] Has experienced "we should have caught this" incidents

**Score 5+/7 = Pursue aggressively**
**Score 3-4/7 = Pursue with caution**
**Score <3/7 = Deprioritize**

---

## Deployment Options

- **Self-hosted**: Full control, data stays on-premise
- **API-first**: Integrate with existing workflows
- **Provider-agnostic**: Works with any LLM backend
- **Stable entry points**: `aragora gauntlet`, `aragora ask`, `aragora serve`

---

## Contact

For pilot discussions:
- GitHub: [aragora-ai/aragora](https://github.com/an0mium/aragora)
- Technical documentation: `/docs/`

---

*Prepared as part of strategic market positioning, January 2026*
