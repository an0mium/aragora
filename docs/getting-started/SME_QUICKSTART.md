# SME Quickstart Guide

Get Aragora running and make your first AI-vetted decision in under 10 minutes.

## 1. Five-Minute Setup

**Docker (recommended)** -- no API keys needed for a test drive:

```bash
git clone https://github.com/aragora-ai/aragora.git && cd aragora
docker compose -f docker-compose.quickstart.yml up -d
```

Open [http://localhost:3000](http://localhost:3000) (dashboard) and `:8080` (API).
The quickstart stack uses mock agents and SQLite so you can explore immediately.

When you are ready for real AI agents, add a `.env`:

```bash
cp .env.example .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env   # at minimum
docker compose -f docker-compose.quickstart.yml up -d --force-recreate
```

**pip install** (alternative):

```bash
pip install aragora && cp .env.example .env   # set ANTHROPIC_API_KEY in .env
aragora serve --demo                          # API :8080, WebSocket :8765
```

**Production SME stack** (Postgres, Redis, backups, Grafana):

```bash
cp .env.example .env   # set ANTHROPIC_API_KEY, optionally OPENAI_API_KEY
docker compose -f docker-compose.sme.yml up -d
# API at :8080, Grafana at :3001 (admin/admin)
```

Verify: `curl http://localhost:8080/healthz` should return `{"status":"ok"}`.

---

## 2. Your First Debate

**CLI:**

```bash
aragora ask "Should we use React or Vue for our new customer dashboard?" --rounds 3
```

Output:

```
[Round 1] anthropic-api (proposer): React is the stronger choice for...
[Round 1] openai-api (critic): Vue offers a gentler learning curve...
[Round 2] mistral (synthesizer): Weighing team expertise and ecosystem...
Consensus reached (round 3): React recommended. Confidence: 0.82
Receipt: receipt_a1b2c3.json
```

Every debate produces a **decision receipt** -- a SHA-256 hash-chained audit trail.

**Web UI:** Open [localhost:3000/debates](http://localhost:3000/debates), click
**New Debate**, enter your question, and watch agents stream in real time.

**API:**

```bash
curl -X POST http://localhost:8080/api/v1/debates \
  -H "Content-Type: application/json" \
  -d '{"question": "Migrate to Kubernetes or stay on ECS?", "rounds": 3}'
```

---

## 3. Decision Playbooks for SMEs

Playbooks combine debate, compliance artifacts, and approval gates into a single
workflow. List them with `aragora playbook list`.

```bash
# Vendor selection (with HIPAA compliance artifacts)
aragora playbook run hipaa_vendor_assessment \
  --input "Evaluate Acme Corp as our new cloud storage provider"

# Hiring decisions
aragora playbook run hiring_committee \
  --input "Candidate A (senior React) vs candidate B (senior Go)?"

# Architecture review (security + scalability + maintainability)
aragora playbook run architecture_review \
  --input "Move from monolith to microservices for billing"

# Incident postmortem
aragora playbook run incident_postmortem \
  --input "Feb 3 outage caused by Redis connection pool exhaustion"

# Financial / compliance
aragora playbook run sox_financial_decision \
  --input "Approve Q1 infra spend increase from 50k to 80k"

aragora playbook run pricing_change \
  --input "Raise Pro tier from 49 to 59 per month"
```

Add `--dry-run` to any playbook to preview steps without executing.

---

## 4. Integrate with Your Tools

**Slack** -- add three lines to `.env` and restart:

```bash
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T.../B.../xxx
```

Your team can then run `/aragora debate "Should we renew the Datadog contract?"` directly from Slack. Results post back to the originating channel.

**Email** -- add SMTP config to `.env`:

```bash
SMTP_HOST=smtp.yourcompany.com
SMTP_PORT=587
SMTP_USER=aragora@yourcompany.com
SMTP_PASSWORD=your-password
```

**GitHub** -- multi-agent code review on any PR:

```bash
aragora review --github owner/repo#42          # standard review
aragora review --github owner/repo#42 --sarif  # SARIF 2.1.0 export
aragora review --github owner/repo#42 --gauntlet  # adversarial stress-test
```

---

## 5. Team Setup

Invite members from **Settings > Team** in the dashboard, or via API:

```bash
curl -X POST http://localhost:8080/api/v1/auth/invite \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ARAGORA_API_TOKEN" \
  -d '{"email": "teammate@company.com", "role": "analyst"}'
```

| Role | Permissions |
|------|-------------|
| `admin` | Full access, user management, billing |
| `analyst` | Run debates, view receipts, use playbooks |
| `viewer` | Read-only access to debates and receipts |
| `tech_lead` | Analyst + approval gate sign-off |

Configure notifications in **Settings > Notifications** (per-user Slack DM, email
digest, or webhook). For team-wide webhooks:

```bash
ARAGORA_WEBHOOKS='[{"name":"team","url":"https://hooks.slack.com/...","event_types":["debate_end"]}]'
```

---

## 6. Troubleshooting

| Problem | Fix |
|---------|-----|
| "No providers available" | Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in `.env`, restart |
| "Debate timed out" | Increase `ARAGORA_DEBATE_TIMEOUT=900` or use `--rounds 3` |
| CORS errors in browser | Set `ARAGORA_ALLOWED_ORIGINS=http://localhost:3000` |
| WebSocket won't connect | Check that port 8765 is not blocked by firewall |
| Rate limit errors (429) | Add `OPENROUTER_API_KEY` for automatic fallback routing |

**Health checks:**

```bash
aragora doctor                                    # CLI diagnostics
curl http://localhost:8080/healthz                 # liveness probe
curl http://localhost:8080/api/v1/readiness        # full readiness (DB, Redis, providers)
```

---

## Next Steps

- **Receipts:** [localhost:3000/receipts](http://localhost:3000/receipts) -- browse audit trails
- **Trending topics:** [localhost:3000/pulse](http://localhost:3000/pulse) -- HackerNews/Reddit/Twitter context for debates
- **Self-improvement:** `aragora self-improve "Improve test coverage" --dry-run`
- **Full env reference:** [`docs/reference/ENVIRONMENT.md`](../reference/ENVIRONMENT.md)
- **API docs:** [`docs/api/API_REFERENCE.md`](../api/API_REFERENCE.md)
- **Enterprise (SSO, RBAC, compliance):** [`docs/enterprise/ENTERPRISE_FEATURES.md`](../enterprise/ENTERPRISE_FEATURES.md)
