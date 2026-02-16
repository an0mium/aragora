# Getting Started with Aragora

Pick the path that matches what you want to do:

---

## 1. Try it now (30 seconds)

No API keys, no config. Runs a full adversarial debate with mock agents:

```bash
pip install aragora
aragora demo
```

You'll see 4 agents debate, critique each other, vote, and produce a decision receipt.

---

## 2. Run the server (5 minutes)

Start the HTTP API and connect the dashboard:

```bash
# Terminal 1: Start the server (offline mode, no API keys needed)
aragora serve --offline

# Terminal 2: Start the dashboard
cd aragora/live
npm run setup   # creates .env.local from template (first time only)
npm run dev
# Open http://localhost:3000
```

To use real AI agents, set API keys in your environment:

```bash
export ANTHROPIC_API_KEY=sk-...
export OPENAI_API_KEY=sk-...
aragora serve
```

---

## 3. Use as a library (SDK)

**Python:**

```bash
pip install aragora-sdk
```

```python
from aragora_sdk import AragoraClient

client = AragoraClient(base_url="http://localhost:8080")
result = client.debates.create(question="Should we migrate to Kubernetes?")
print(result.receipt)
```

**TypeScript:**

```bash
npm install @aragora/sdk
```

```typescript
import { AragoraClient } from '@aragora/sdk';

const client = new AragoraClient({ baseUrl: 'http://localhost:8080' });
const result = await client.debates.create({ question: 'Should we migrate to Kubernetes?' });
console.log(result.receipt);
```

---

## 4. Deploy to production

**Docker Compose (recommended for most teams):**

```bash
docker compose -f docker-compose.quickstart.yml up
# Dashboard at http://localhost:3000, API at http://localhost:8080
```

**Kubernetes:**

```bash
kubectl apply -f deploy/kubernetes/
```

See [Enterprise Features](enterprise/ENTERPRISE_FEATURES.md) for SSO, RBAC, multi-tenancy, and compliance.

---

## What's next?

| Goal | Command / Doc |
|------|---------------|
| Check system health | `aragora doctor` |
| Run a real debate | `aragora ask "Your question" --agents anthropic-api,openai-api` |
| Full decision pipeline | `aragora decide "Your question"` |
| API reference | [docs/api/API_REFERENCE.md](api/API_REFERENCE.md) |
| SDK guide | [docs/SDK_GUIDE.md](SDK_GUIDE.md) |
| All features | [docs/FEATURE_DISCOVERY.md](FEATURE_DISCOVERY.md) |
