# SME Starter Pack - General Availability Guide

Welcome to Aragora's SME Starter Pack! This guide helps small and medium enterprises get maximum value from multi-agent AI debates for better business decisions.

## Quick Start (5 Minutes)

### 1. Set Up Your Workspace

```bash
# Install the Python SDK
pip install aragora-client

# Or use the TypeScript SDK
npm install @aragora/sdk
```

### 2. Configure API Keys

Create a `.env` file with at least one AI provider:

```env
# Required: At least one provider
ANTHROPIC_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here

# Optional: Additional providers for more diverse debates
MISTRAL_API_KEY=your-key-here
GEMINI_API_KEY=your-key-here
```

### 3. Run Your First Debate

```python
from aragora_client import AragoraClient

client = AragoraClient()

# Quick hiring decision debate
result = client.debates.quick_start(
    template="sme_hiring_decision",
    inputs={
        "candidate_name": "Jane Smith",
        "role": "Product Manager",
        "qualifications": "5 years PM experience, MBA, led 3 product launches"
    }
)

print(result.recommendation)
print(result.consensus_summary)
```

---

## Template Gallery

### Team Decisions

| Template | Use Case | Agents | Rounds | Time |
|----------|----------|--------|--------|------|
| **Hiring Decision** | Evaluate job candidates | Claude, GPT-4 | 3 | ~5 min |
| **Performance Review** | Fair employee evaluations | Claude, Gemini | 2 | ~3 min |

### Project Management

| Template | Use Case | Agents | Rounds | Time |
|----------|----------|--------|--------|------|
| **Feature Prioritization** | Rank product features | Claude, GPT-4, Mistral | 3 | ~5 min |
| **Sprint Planning** | Scope agile sprints | Claude, GPT-4 | 2 | ~3 min |

### Vendor & Procurement

| Template | Use Case | Agents | Rounds | Time |
|----------|----------|--------|--------|------|
| **Tool Selection** | Evaluate software tools | Claude, GPT-4, Gemini | 4 | ~7 min |
| **Contract Review** | Review business contracts | Claude, GPT-4 | 3 | ~5 min |

### Policy Development

| Template | Use Case | Agents | Rounds | Time |
|----------|----------|--------|--------|------|
| **Remote Work Policy** | Develop work policies | Claude, GPT-4, Gemini | 3 | ~5 min |
| **Budget Allocation** | Strategic budget planning | Claude, GPT-4 | 2 | ~3 min |

### Operations (Automation)

| Template | Use Case | Description |
|----------|----------|-------------|
| **Invoice Generation** | Automated invoicing | Generate and send invoices |
| **Customer Follow-up** | CRM automation | Automated customer outreach |
| **Inventory Alerts** | Stock management | Low inventory notifications |
| **Report Scheduling** | Business intelligence | Automated report generation |

---

## Integration Guides

### Slack Integration

Connect Aragora to Slack for debate notifications and quick decisions:

```python
from aragora_client import AragoraClient

client = AragoraClient()

# Enable Slack notifications
client.integrations.slack.connect(
    webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
    channel="#decisions"
)

# Debates will now post results to Slack
result = client.debates.run(
    template="sme_hiring_decision",
    inputs={...},
    notify_slack=True
)
```

### Microsoft Teams Integration

```python
# Enable Teams notifications
client.integrations.teams.connect(
    webhook_url="https://outlook.office.com/webhook/YOUR/WEBHOOK",
    channel="Decisions"
)
```

### API Integration

For custom integrations, use the REST API:

```bash
# Start a debate
curl -X POST https://api.aragora.com/api/debates \
  -H "Authorization: Bearer $ARAGORA_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "template": "sme_hiring_decision",
    "inputs": {
      "candidate_name": "Jane Smith",
      "role": "Product Manager"
    }
  }'
```

---

## ROI Tracking

Track the business value of your AI-assisted decisions:

### View ROI Dashboard

```python
# Get ROI metrics
roi = client.analytics.roi.summary(period="last_30_days")

print(f"Debates completed: {roi.debates_count}")
print(f"Time saved: {roi.time_saved_hours} hours")
print(f"Estimated value: ${roi.estimated_value}")
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| **Time Saved** | Hours saved vs. manual decision processes |
| **Decision Quality** | Consensus confidence scores over time |
| **Cost per Decision** | AI API costs per debate |
| **Coverage** | % of decisions using Aragora |

### Industry Benchmarks

SME decision-making benchmarks:

| Decision Type | Traditional Time | With Aragora | Savings |
|---------------|------------------|--------------|---------|
| Hiring Decision | 4-8 hours | 15 minutes | 90%+ |
| Vendor Selection | 2-4 weeks | 2-4 hours | 95%+ |
| Policy Draft | 1-2 weeks | 1-2 hours | 95%+ |
| Sprint Planning | 2-4 hours | 30 minutes | 80%+ |

---

## Best Practices

### 1. Provide Rich Context

Better inputs = better debates:

```python
# Good: Rich context
result = client.debates.run(
    template="sme_hiring_decision",
    inputs={
        "candidate_name": "Jane Smith",
        "role": "Senior Product Manager",
        "qualifications": """
            - 5 years PM experience at SaaS companies
            - MBA from Stanford
            - Led 3 successful product launches
            - Strong technical background (former engineer)
        """,
        "interview_notes": """
            - Strong communicator
            - Asked insightful questions about our roadmap
            - Salary expectation: $150k (within budget)
        """,
        "team_context": "Team of 4 PMs, need someone senior to mentor juniors"
    }
)

# Less effective: Minimal context
result = client.debates.run(
    template="sme_hiring_decision",
    inputs={
        "candidate_name": "Jane",
        "role": "PM",
        "qualifications": "Experienced"
    }
)
```

### 2. Choose the Right Template

- **Quick decisions**: Use 2-round templates (Performance Review, Sprint Planning)
- **High-stakes decisions**: Use 3-4 round templates (Hiring, Tool Selection)
- **Complex topics**: Use 3-agent templates for more diverse perspectives

### 3. Review and Learn

After each debate:

1. Review the consensus summary
2. Check minority opinions for blind spots
3. Store decisions for future reference
4. Track outcomes to improve future debates

### 4. Customize Templates

Create custom templates for your specific needs:

```python
# Clone and customize a template
custom_template = client.templates.clone(
    source="sme_hiring_decision",
    name="my_hiring_decision",
    modifications={
        "rounds": 4,
        "agents": ["claude", "gpt-4", "gemini"],
        "evaluation_criteria": [
            "technical_skills",
            "culture_fit",
            "leadership_potential",
            "domain_expertise"
        ]
    }
)
```

---

## Troubleshooting

### Common Issues

**Debate takes too long**
- Reduce the number of rounds
- Use fewer agents
- Check API rate limits

**Inconsistent results**
- Provide more context in inputs
- Use more agents for diverse perspectives
- Increase rounds for better consensus

**API errors**
- Verify API keys are set correctly
- Check provider status pages
- Review rate limits

### Support

- Documentation: https://docs.aragora.com
- GitHub Issues: https://github.com/aragora/aragora/issues
- Email: support@aragora.com

---

## Self-Hosted Deployment

For enterprises requiring on-premise deployment:

### Docker Deployment

```bash
# Clone the repository
git clone https://github.com/aragora/aragora.git
cd aragora

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start services
docker compose up -d

# Verify deployment
curl http://localhost:8080/health
```

### Kubernetes Deployment

See `docs/KUBERNETES.md` for Helm charts and production configurations.

### Requirements

- Docker 20.10+ or Kubernetes 1.24+
- 4GB RAM minimum (8GB recommended)
- PostgreSQL 14+ (optional, for persistence)
- Redis 7+ (optional, for caching)

---

## What's Next?

1. **Explore Templates**: Try different templates to find what works for your team
2. **Set Up Integrations**: Connect Slack/Teams for seamless workflows
3. **Track ROI**: Monitor the dashboard to quantify value
4. **Customize**: Create templates specific to your business processes
5. **Scale**: Add team members and expand usage across departments

Welcome to better decision-making with Aragora!
