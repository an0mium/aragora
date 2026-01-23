# API Examples

> **Last Updated:** 2026-01-22

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

## Python SDK Examples

The following examples show how to interact with the Aragora API using Python. These patterns work with any HTTP client library.

### Setup

```python
import httpx
import asyncio
from typing import Any

# Base configuration
BASE_URL = "http://localhost:8080"
API_TOKEN = "your-api-token"  # Set via ARAGORA_API_TOKEN env var

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
}
```

### Create and Stream a Debate

```python
import httpx
import json

async def create_debate_with_streaming():
    """Create a debate and stream results in real-time."""
    async with httpx.AsyncClient(base_url=BASE_URL, headers=headers) as client:
        # Create the debate
        response = await client.post("/api/debates", json={
            "topic": "What are the risks of autonomous AI agents?",
            "rounds": 3,
            "agents": ["anthropic-api", "openai-api"],
            "consensus_mode": "supermajority",
        })
        debate = response.json()
        debate_id = debate["debate_id"]

        print(f"Debate created: {debate_id}")

        # Connect to WebSocket for real-time updates
        async with httpx.AsyncClient() as ws_client:
            async with ws_client.stream(
                "GET",
                f"{BASE_URL}/ws?debate_id={debate_id}"
            ) as stream:
                async for line in stream.aiter_lines():
                    if line.startswith("data: "):
                        event = json.loads(line[6:])
                        handle_event(event)

def handle_event(event: dict):
    """Process debate stream events."""
    event_type = event.get("type")

    if event_type == "agent_message":
        agent = event["agent"]
        content = event["content"][:100]
        print(f"[{agent}] {content}...")

    elif event_type == "consensus":
        claim = event["claim"]
        confidence = event["confidence"]
        print(f"Consensus reached: {claim} (confidence: {confidence:.0%})")

    elif event_type == "debate_end":
        print("Debate completed!")

# Run the example
asyncio.run(create_debate_with_streaming())
```

### Control Plane Task Management

```python
async def control_plane_example():
    """Submit and monitor tasks via the control plane."""
    async with httpx.AsyncClient(base_url=BASE_URL, headers=headers) as client:
        # Submit a document processing task
        response = await client.post("/api/control-plane/tasks", json={
            "task_type": "document_processing",
            "payload": {
                "document_url": "https://example.com/report.pdf",
                "extract_facts": True,
            },
            "required_capabilities": ["document", "analysis"],
            "priority": "high",
        })
        task = response.json()
        task_id = task["task_id"]

        print(f"Task submitted: {task_id}")

        # Poll for completion
        while True:
            status_response = await client.get(f"/api/control-plane/tasks/{task_id}")
            status = status_response.json()

            if status["status"] == "completed":
                print(f"Task completed: {status['result']}")
                break
            elif status["status"] == "failed":
                print(f"Task failed: {status['error']}")
                break

            await asyncio.sleep(1)  # Poll every second

asyncio.run(control_plane_example())
```

### Control Plane Deliberation

```python
async def control_plane_deliberation():
    """Run a deliberation via the control plane (sync or async)."""
    async with httpx.AsyncClient(base_url=BASE_URL, headers=headers) as client:
        # Submit a deliberation (async)
        response = await client.post("/api/control-plane/deliberations", json={
            "content": "Evaluate the rollout risk for this migration plan",
            "decision_type": "debate",
            "async": True,
            "priority": "high",
            "required_capabilities": ["deliberation"],
        })
        payload = response.json()
        request_id = payload["request_id"]

        # Poll for completion
        while True:
            status_response = await client.get(
                f"/api/control-plane/deliberations/{request_id}/status"
            )
            status = status_response.json()
            if status["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(1)

asyncio.run(control_plane_deliberation())
```

### Decision Router (Unified API)

```python
async def decision_router_example():
    """Submit a decision request via the unified router."""
    async with httpx.AsyncClient(base_url=BASE_URL, headers=headers) as client:
        response = await client.post("/api/v1/decisions", json={
            "content": "Should we adopt a service mesh this quarter?",
            "decision_type": "debate",
            "response_channels": [{"platform": "http_api"}],
        })
        result = response.json()
        request_id = result["request_id"]

        status = await client.get(f"/api/v1/decisions/{request_id}/status")
        print(status.json())

asyncio.run(decision_router_example())
```

### Codebase Security Scan

```python
async def codebase_security_scan():
    """Trigger a dependency vulnerability scan and fetch results."""
    async with httpx.AsyncClient(base_url=BASE_URL, headers=headers) as client:
        scan = await client.post("/api/v1/codebase/aragora/scan", json={
            "repo_path": "/Users/armand/Development/aragora",
            "branch": "main",
        })
        scan_data = scan.json()
        scan_id = scan_data["scan_id"]

        result = await client.get(f"/api/v1/codebase/aragora/scan/{scan_id}")
        print(result.json())

asyncio.run(codebase_security_scan())
```

### Codebase Metrics Analysis

```python
async def codebase_metrics():
    """Run code metrics analysis and fetch hotspots."""
    async with httpx.AsyncClient(base_url=BASE_URL, headers=headers) as client:
        analysis = await client.post("/api/v1/codebase/aragora/metrics/analyze", json={
            "repo_path": "/Users/armand/Development/aragora",
            "complexity_warning": 10,
            "complexity_error": 20,
        })
        analysis_id = analysis.json()["analysis_id"]

        hotspots = await client.get("/api/v1/codebase/aragora/hotspots")
        print(hotspots.json())

asyncio.run(codebase_metrics())
```

### Gauntlet Compliance Audit

```python
async def run_gauntlet_audit():
    """Run a compliance gauntlet against a proposal."""
    async with httpx.AsyncClient(base_url=BASE_URL, headers=headers) as client:
        # Start a GDPR compliance gauntlet
        response = await client.post("/api/gauntlet", json={
            "proposal": """
            Our new data collection system will:
            1. Store user location data for 5 years
            2. Share anonymized data with third-party advertisers
            3. Use AI to infer user preferences
            """,
            "personas": ["gdpr", "ai_act"],
            "severity_threshold": 0.5,
        })
        result = response.json()

        print(f"Gauntlet ID: {result['gauntlet_id']}")
        print(f"Verdict: {result['verdict']}")
        print(f"Confidence: {result['confidence']:.0%}")

        # Print findings
        for finding in result.get("findings", []):
            severity = finding["severity"]
            title = finding["title"]
            print(f"  [{severity}] {title}")

asyncio.run(run_gauntlet_audit())
```

### Evidence Collection

```python
async def collect_evidence():
    """Collect evidence from multiple sources."""
    async with httpx.AsyncClient(base_url=BASE_URL, headers=headers) as client:
        # Configure evidence sources
        await client.post("/api/evidence/sources", json={
            "source_type": "web",
            "config": {
                "domains": ["arxiv.org", "github.com"],
                "max_results": 10,
            }
        })

        # Collect evidence for a claim
        response = await client.post("/api/evidence/collect", json={
            "claim": "Large language models can exhibit emergent capabilities",
            "source_types": ["web", "academic"],
            "max_snippets": 20,
        })
        evidence = response.json()

        for snippet in evidence["snippets"]:
            print(f"[{snippet['source']}] {snippet['content'][:100]}...")
            print(f"  Relevance: {snippet['relevance']:.0%}")

asyncio.run(collect_evidence())
```

### WebSocket Real-Time Events

```python
import websockets
import json

async def listen_to_events():
    """Listen to real-time debate events via WebSocket."""
    ws_url = f"ws://localhost:8080/ws"

    async with websockets.connect(ws_url) as ws:
        # Subscribe to a specific debate
        await ws.send(json.dumps({
            "action": "subscribe",
            "debate_id": "debate-123",
        }))

        # Listen for events
        async for message in ws:
            event = json.loads(message)
            event_type = event.get("type")

            if event_type == "round_start":
                print(f"Round {event['round']} starting")

            elif event_type == "agent_message":
                print(f"[{event['agent']}] {event['content'][:50]}...")

            elif event_type == "critique":
                print(f"Critique from {event['agent']}: {event['issues']}")

            elif event_type == "vote":
                print(f"{event['agent']} votes: {event['vote_type']}")

            elif event_type == "consensus":
                print(f"Consensus: {event['claim']}")
                break  # Exit after consensus

asyncio.run(listen_to_events())
```

### Batch Operations

```python
async def batch_debate_analysis():
    """Analyze multiple debates in batch."""
    async with httpx.AsyncClient(base_url=BASE_URL, headers=headers) as client:
        # Get recent debates
        debates_response = await client.get("/api/debates", params={"limit": 10})
        debates = debates_response.json()["debates"]

        # Analyze each debate's consensus
        results = []
        for debate in debates:
            slug = debate["slug"]

            # Get convergence data
            convergence = await client.get(f"/api/debates/{slug}/convergence")

            # Get consensus proof
            proof = await client.get(f"/api/debates/{slug}/proof")

            results.append({
                "slug": slug,
                "topic": debate["topic"],
                "consensus_reached": debate.get("consensus_reached", False),
                "convergence_score": convergence.json().get("score", 0),
                "checksum": proof.json().get("checksum"),
            })

        # Print summary
        for r in results:
            status = "✓" if r["consensus_reached"] else "✗"
            print(f"{status} {r['slug']}: convergence={r['convergence_score']:.2f}")

asyncio.run(batch_debate_analysis())
```

---

## HTTP API Examples

### Debates

#### Create a New Debate

```bash
curl -X POST http://localhost:8080/api/debates \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Should AI systems be required to explain their decisions?",
    "rounds": 3,
    "agents": ["anthropic-api", "openai-api", "gemini"]
  }'
```

Response (201 Created):
```json
{
  "debate_id": "debate-20240115-abc123",
  "topic": "Should AI systems be required to explain their decisions?",
  "status": "in_progress",
  "agents": ["anthropic-api", "openai-api", "gemini"],
  "rounds": 3,
  "created_at": "2024-01-15T10:00:00Z",
  "stream_url": "ws://localhost:8080/ws?debate_id=debate-20240115-abc123"
}
```

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

#### Vote on a Debate

Submit a user vote for a debate:

```bash
curl -X POST http://localhost:8080/api/debates/debate-001/vote \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "anthropic-api",
    "reason": "Provided the most comprehensive and well-reasoned argument"
  }'
```

Response:
```json
{
  "success": true,
  "vote_id": "vote-xyz789",
  "debate_id": "debate-001",
  "voted_for": "anthropic-api",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Submit User Suggestion

Add a suggestion during a live debate:

```bash
curl -X POST http://localhost:8080/api/debates/debate-001/suggest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Consider the impact on small businesses specifically",
    "target_agent": null
  }'
```

Response:
```json
{
  "success": true,
  "suggestion_id": "sug-abc123",
  "status": "queued",
  "position": 3
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
        print(f"Connected to stream (filtering for loop_id={debate_id})")

        async for message in ws:
            event = json.loads(message)
            event_type = event.get("type")
            event_loop_id = (
                event.get("loop_id")
                or event.get("data", {}).get("debate_id")
                or event.get("data", {}).get("loop_id")
            )

            # Ignore control messages + unrelated loops
            if event_type in ("connection_info", "loop_list", "sync"):
                continue
            if event_loop_id and event_loop_id != debate_id:
                continue

            if event_type == "agent_message":
                agent = event.get("agent") or event["data"].get("agent", "unknown")
                content = event["data"].get("content", "")[:100]
                print(f"[{agent}] {content}...")

            elif event_type == "critique":
                critic = event.get("agent") or "unknown"
                target = event["data"].get("target", "unknown")
                issues = event["data"].get("issues", [])
                summary = "; ".join(issues) if issues else event["data"].get("content", "")
                print(f"[CRITIQUE] {critic} -> {target}: {summary}")

            elif event_type == "consensus":
                reached = event["data"].get("reached")
                answer = event["data"].get("answer", "")
                print(f"[CONSENSUS] reached={reached} answer={answer[:120]}")

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
