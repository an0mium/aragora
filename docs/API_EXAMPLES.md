# API Examples

Practical examples for using the Aragora HTTP and WebSocket APIs.

## Quick Start

Start the server:

```bash
aragora serve
```

Verify it's running:

```bash
curl http://localhost:8080/api/health
```

---

## HTTP API Examples

### Debates

#### List Recent Debates

```bash
# Get last 10 debates
curl http://localhost:8080/api/debates?limit=10
```

Response:
```json
{
  "debates": [
    {
      "slug": "ai-safety-2024-01",
      "topic": "Best practices for AI safety",
      "rounds_used": 3,
      "consensus_reached": true,
      "created_at": "2024-01-15T10:00:00Z"
    }
  ],
  "count": 1
}
```

#### Get Specific Debate

```bash
# By slug
curl http://localhost:8080/api/debates/slug/ai-safety-2024-01

# Or directly
curl http://localhost:8080/api/debates/ai-safety-2024-01
```

#### Export Debate

```bash
# As JSON
curl http://localhost:8080/api/debates/debate-001/export/json -o debate.json

# As CSV (messages table)
curl "http://localhost:8080/api/debates/debate-001/export/csv?table=messages" -o messages.csv

# As HTML (standalone page)
curl http://localhost:8080/api/debates/debate-001/export/html -o debate.html
```

#### Check Debate Convergence

```bash
curl http://localhost:8080/api/debates/debate-001/convergence
```

Response:
```json
{
  "debate_id": "debate-001",
  "convergence_status": "converged",
  "convergence_similarity": 0.92,
  "consensus_reached": true,
  "rounds_used": 3
}
```

#### Detect Impasse

```bash
curl http://localhost:8080/api/debates/debate-001/impasse
```

Response:
```json
{
  "debate_id": "debate-001",
  "is_impasse": false,
  "indicators": {
    "repeated_critiques": false,
    "no_convergence": false,
    "high_severity_critiques": false
  }
}
```

---

### Agents

#### Get Leaderboard

```bash
# Top 20 agents
curl http://localhost:8080/api/leaderboard?limit=20

# Filter by domain
curl "http://localhost:8080/api/leaderboard?limit=20&domain=science"
```

Response:
```json
{
  "rankings": [
    {
      "name": "anthropic-api",
      "elo": 1650,
      "wins": 15,
      "losses": 3,
      "win_rate": 0.83,
      "consistency": 0.95,
      "consistency_class": "high"
    }
  ]
}
```

#### Get Agent Profile

```bash
curl http://localhost:8080/api/agent/anthropic-api/profile
```

Response:
```json
{
  "name": "anthropic-api",
  "rating": 1650,
  "rank": 1,
  "wins": 15,
  "losses": 3,
  "win_rate": 0.83
}
```

#### Compare Two Agents

```bash
curl "http://localhost:8080/api/agent/compare?agents=anthropic-api&agents=openai-api"
```

Response:
```json
{
  "agents": [
    {"name": "anthropic-api", "rating": 1650, "wins": 15},
    {"name": "openai-api", "rating": 1580, "wins": 12}
  ],
  "head_to_head": {
    "matches": 5,
    "agent1_wins": 3,
    "agent2_wins": 2
  }
}
```

#### Get Agent Match History

```bash
curl http://localhost:8080/api/agent/anthropic-api/history?limit=10
```

#### Head-to-Head Statistics

```bash
curl http://localhost:8080/api/agent/anthropic-api/head-to-head/openai-api
```

Response:
```json
{
  "agent1": "anthropic-api",
  "agent2": "openai-api",
  "matches": 5,
  "agent1_wins": 3,
  "agent2_wins": 2
}
```

#### Get Agent Network (Rivals & Allies)

```bash
curl http://localhost:8080/api/agent/anthropic-api/network
```

Response:
```json
{
  "agent": "anthropic-api",
  "rivals": [{"name": "openai-api", "matches": 5}],
  "allies": [{"name": "gemini", "matches": 3}]
}
```

---

### Consensus

#### Get Similar Consensus Records

```bash
curl "http://localhost:8080/api/consensus/similar?topic=AI%20safety&limit=5"
```

#### Get Settled Positions

```bash
curl http://localhost:8080/api/consensus/settled?limit=10
```

#### Get Dissenting Views

```bash
curl http://localhost:8080/api/consensus/dissents?limit=10
```

#### Get Contrarian Views

```bash
curl http://localhost:8080/api/consensus/contrarian-views?limit=5
```

---

### Flips (Position Changes)

#### Recent Flips Across All Agents

```bash
curl http://localhost:8080/api/flips/recent?limit=20
```

#### Flip Summary for Dashboard

```bash
curl http://localhost:8080/api/flips/summary
```

Response:
```json
{
  "total_flips": 42,
  "by_type": {"contradiction": 15, "refinement": 27},
  "by_agent": {"anthropic-api": 5, "openai-api": 8},
  "recent_24h": 3
}
```

#### Agent-Specific Flips

```bash
curl http://localhost:8080/api/agent/anthropic-api/flips?limit=10
```

---

## Python Client Examples

### Basic HTTP Client

```python
import requests

BASE_URL = "http://localhost:8080"

def get_leaderboard(limit=10):
    """Get top agents by ELO rating."""
    resp = requests.get(f"{BASE_URL}/api/leaderboard", params={"limit": limit})
    resp.raise_for_status()
    return resp.json()["rankings"]

def get_debate(slug):
    """Get a specific debate."""
    resp = requests.get(f"{BASE_URL}/api/debates/slug/{slug}")
    resp.raise_for_status()
    return resp.json()

def compare_agents(agent1, agent2):
    """Compare two agents head-to-head."""
    resp = requests.get(
        f"{BASE_URL}/api/agent/compare",
        params={"agents": [agent1, agent2]}
    )
    resp.raise_for_status()
    return resp.json()

# Usage
if __name__ == "__main__":
    # Get leaderboard
    print("Top 5 Agents:")
    for agent in get_leaderboard(5):
        print(f"  {agent['name']}: {agent['elo']} ELO")

    # Compare agents
    result = compare_agents("anthropic-api", "openai-api")
    print(f"\nHead-to-head: {result['head_to_head']}")
```

### WebSocket Client

```python
import asyncio
import json
import websockets

async def stream_debate(debate_id):
    """Stream live debate events via WebSocket."""
    uri = "ws://localhost:8765/ws"

    async with websockets.connect(uri) as ws:
        print(f"Connected to debate: {debate_id}")
        await ws.send(json.dumps({"type": "subscribe", "debate_id": debate_id}))

        async for message in ws:
            event = json.loads(message)
            event_type = event.get("type")

            if event_type == "agent_message":
                agent = event["data"]["agent"]
                content = event["data"]["content"][:100]
                print(f"[{agent}] {content}...")

            elif event_type == "critique":
                critic = event["data"]["critic"]
                target = event["data"]["target"]
                print(f"[CRITIQUE] {critic} -> {target}")

            elif event_type == "consensus":
                print(f"[CONSENSUS] {event['data']['position']}")

            elif event_type == "debate_end":
                print("[END] Debate concluded")
                break

# Run
asyncio.run(stream_debate("live-debate-001"))
```

### Full Workflow Example

```python
import requests
import json

BASE_URL = "http://localhost:8080"

def run_debate_workflow():
    """Example workflow: find debate, analyze, export."""

    # 1. List recent debates
    print("=== Recent Debates ===")
    resp = requests.get(f"{BASE_URL}/api/debates", params={"limit": 5})
    debates = resp.json()["debates"]

    for d in debates:
        status = "Consensus" if d.get("consensus_reached") else "No consensus"
        print(f"  [{status}] {d['topic'][:50]}")

    if not debates:
        print("No debates found")
        return

    # 2. Get first debate details
    debate = debates[0]
    slug = debate["slug"]
    print(f"\n=== Debate: {slug} ===")

    resp = requests.get(f"{BASE_URL}/api/debates/slug/{slug}")
    full_debate = resp.json()

    print(f"Topic: {full_debate.get('topic')}")
    print(f"Rounds: {full_debate.get('rounds_used')}")
    print(f"Messages: {len(full_debate.get('messages', []))}")

    # 3. Check convergence
    print("\n=== Convergence ===")
    resp = requests.get(f"{BASE_URL}/api/debates/{slug}/convergence")
    conv = resp.json()

    print(f"Status: {conv.get('convergence_status')}")
    print(f"Similarity: {conv.get('convergence_similarity')}")

    # 4. Get agent stats from the debate
    print("\n=== Agent Stats ===")
    resp = requests.get(f"{BASE_URL}/api/leaderboard", params={"limit": 5})
    for agent in resp.json()["rankings"]:
        print(f"  {agent['name']}: {agent['elo']} ELO ({agent.get('win_rate', 0):.0%} win rate)")

    # 5. Export
    print("\n=== Export ===")
    resp = requests.get(f"{BASE_URL}/api/debates/{slug}/export/json")
    with open(f"{slug}.json", "w") as f:
        json.dump(resp.json(), f, indent=2)
    print(f"Saved to {slug}.json")

if __name__ == "__main__":
    run_debate_workflow()
```

---

## cURL Tips

### Pretty-Print JSON

```bash
# Using jq
curl http://localhost:8080/api/leaderboard | jq .

# Using Python
curl http://localhost:8080/api/leaderboard | python -m json.tool
```

### Save Response to File

```bash
curl http://localhost:8080/api/debates -o debates.json
```

### Include Headers

```bash
# With auth token (if configured)
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8080/api/debates

# JSON content type
curl -H "Content-Type: application/json" http://localhost:8080/api/debates
```

### Verbose Mode (Debug)

```bash
curl -v http://localhost:8080/api/health
```

---

## Error Handling

### Common HTTP Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Parse response body |
| 400 | Bad Request | Check request parameters |
| 404 | Not Found | Verify resource ID/slug |
| 500 | Server Error | Check server logs |
| 503 | Service Unavailable | Backend not ready |

### Python Error Handling

```python
import requests

def safe_api_call(url, params=None):
    """Make API call with error handling."""
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        print("Request timed out")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e.response.status_code}")
        try:
            error = e.response.json()
            print(f"  Message: {error.get('error')}")
        except:
            pass
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
```

---

## API Reference

For complete endpoint documentation, see [API_REFERENCE.md](API_REFERENCE.md).
