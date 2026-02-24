"""
Demo transport for the Aragora SDK.

Provides mock responses so developers can try the SDK without a running server
or API keys. Activated via ``AragoraClient(demo=True)`` or
``AragoraClient.from_env()`` with ``ARAGORA_DEMO=1``.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any


def _uid() -> str:
    return uuid.uuid4().hex[:12]


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# Route table: (method, path_prefix) -> handler
# ---------------------------------------------------------------------------

_DEMO_AGENTS = [
    {"name": "claude", "model": "claude-sonnet-4-6", "elo": 1520, "wins": 34, "losses": 12},
    {"name": "gpt-4", "model": "gpt-4.1", "elo": 1495, "wins": 30, "losses": 16},
    {"name": "gemini", "model": "gemini-3.1-pro-preview", "elo": 1480, "wins": 28, "losses": 18},
    {"name": "grok", "model": "grok-4-latest", "elo": 1460, "wins": 25, "losses": 21},
]


def _make_debate(task: str, agents: list[str] | None = None, rounds: int = 3) -> dict[str, Any]:
    debate_id = f"demo-{_uid()}"
    agent_list = agents or ["claude", "gpt-4", "gemini"]
    return {
        "debate_id": debate_id,
        "task": task,
        "status": "completed",
        "agents": agent_list,
        "rounds_completed": rounds,
        "created_at": _ts(),
        "consensus": {
            "reached": True,
            "confidence": 0.87,
            "conclusion": f"After {rounds} rounds of adversarial debate among {', '.join(agent_list)}, "
            f"the agents reached consensus on: {task}. "
            "The key factors identified were feasibility, risk mitigation, and stakeholder alignment.",
            "final_answer": f"Regarding '{task}': The panel recommends a balanced approach. "
            "Claude emphasized risk analysis, GPT-4 focused on implementation feasibility, "
            "and Gemini provided cost-benefit quantification. All agents converged on an "
            "incremental adoption strategy with defined rollback criteria.",
            "dissenting_views": [],
        },
        "messages": [
            {
                "agent": agent_list[0],
                "round": 1,
                "phase": "propose",
                "content": f"I propose we analyze '{task}' through three lenses: technical feasibility, "
                "organizational readiness, and risk tolerance.",
            },
            {
                "agent": agent_list[1] if len(agent_list) > 1 else agent_list[0],
                "round": 1,
                "phase": "critique",
                "content": "While the framework is sound, we should add a cost-benefit dimension "
                "and consider time-to-value as a key metric.",
            },
            {
                "agent": agent_list[2] if len(agent_list) > 2 else agent_list[0],
                "round": 1,
                "phase": "critique",
                "content": "I agree with the multi-lens approach. Let me quantify the expected ROI "
                "and break down implementation into phases with clear milestones.",
            },
        ],
    }


def _make_receipt(debate_id: str, task: str) -> dict[str, Any]:
    content = f"{debate_id}:{task}:{_ts()}"
    return {
        "receipt_id": f"rcpt-{_uid()}",
        "debate_id": debate_id,
        "task": task,
        "timestamp": _ts(),
        "hash": hashlib.sha256(content.encode()).hexdigest(),
        "algorithm": "sha256",
        "consensus_reached": True,
        "confidence": 0.87,
        "agents_participated": ["claude", "gpt-4", "gemini"],
        "rounds_completed": 3,
    }


def _handle_health(**_: Any) -> dict[str, Any]:
    return {
        "status": "healthy",
        "version": "2.6.3-demo",
        "uptime_seconds": 86400,
        "mode": "demo",
        "components": {
            "database": "healthy",
            "agents": "healthy",
            "memory": "healthy",
            "websocket": "healthy",
        },
    }


def _handle_debates_create(json: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    body = json or {}
    return _make_debate(
        task=body.get("task", "Demo debate topic"),
        agents=body.get("agents"),
        rounds=body.get("rounds", 3),
    )


def _handle_debates_list(**_: Any) -> dict[str, Any]:
    return {
        "debates": [
            _make_debate("Should we adopt microservices?"),
            _make_debate("Is Kubernetes worth the complexity?"),
            _make_debate("Monorepo vs polyrepo?"),
        ],
        "total": 3,
        "page": 1,
        "per_page": 20,
    }


def _handle_debates_get(**_: Any) -> dict[str, Any]:
    return _make_debate("Demo debate topic")


def _handle_agents_list(**_: Any) -> dict[str, Any]:
    return {"agents": _DEMO_AGENTS, "total": len(_DEMO_AGENTS)}


def _handle_rankings(**_: Any) -> dict[str, Any]:
    return {
        "rankings": [
            {
                "rank": i + 1,
                "agent": a["name"],
                "elo": a["elo"],
                "wins": a["wins"],
                "losses": a["losses"],
            }
            for i, a in enumerate(_DEMO_AGENTS)
        ],
        "total": len(_DEMO_AGENTS),
    }


def _handle_gauntlet_run(json: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    body = json or {}
    debate_id = f"demo-{_uid()}"
    return {
        "gauntlet_id": f"gauntlet-{_uid()}",
        "debate_id": debate_id,
        "status": "completed",
        "findings": [
            {
                "severity": "medium",
                "category": "reasoning",
                "description": "Potential confirmation bias detected in round 2",
                "agent": "gpt-4",
            },
            {
                "severity": "low",
                "category": "completeness",
                "description": "Cost analysis could be more granular",
                "agent": "claude",
            },
        ],
        "receipt": _make_receipt(debate_id, body.get("task", "Gauntlet test")),
        "score": 0.82,
    }


def _handle_receipts_list(**_: Any) -> dict[str, Any]:
    return {
        "receipts": [
            _make_receipt(f"demo-{_uid()}", "Should we adopt microservices?"),
            _make_receipt(f"demo-{_uid()}", "Evaluate CI/CD pipeline options"),
        ],
        "total": 2,
    }


def _handle_knowledge_query(json: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    body = json or {}
    return {
        "results": [
            {
                "content": "Microservices increase deployment flexibility but add operational complexity.",
                "source": "debate-demo-001",
                "confidence": 0.92,
                "type": "debate_consensus",
            },
            {
                "content": "Event-driven architectures reduce coupling but require careful error handling.",
                "source": "debate-demo-002",
                "confidence": 0.88,
                "type": "debate_consensus",
            },
        ],
        "query": body.get("query", ""),
        "total": 2,
    }


def _handle_not_found(path: str = "", **_: Any) -> dict[str, Any]:
    return {"message": f"Demo mode: endpoint '{path}' returns empty response", "data": {}}


# ---------------------------------------------------------------------------
# Route matching
# ---------------------------------------------------------------------------

_ROUTES: list[tuple[str, str, Any]] = [
    ("GET", "/api/v1/health", _handle_health),
    ("GET", "/health", _handle_health),
    ("POST", "/api/v1/debates", _handle_debates_create),
    ("GET", "/api/v1/debates", _handle_debates_list),
    ("GET", "/api/v1/agents", _handle_agents_list),
    ("GET", "/api/v1/ranking", _handle_rankings),
    ("GET", "/api/v1/rankings", _handle_rankings),
    ("POST", "/api/v1/gauntlet", _handle_gauntlet_run),
    ("GET", "/api/v1/receipts", _handle_receipts_list),
    ("POST", "/api/v1/knowledge/query", _handle_knowledge_query),
]


def demo_request(
    method: str,
    path: str,
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Resolve a request against the demo route table.

    Returns mock data that matches the real API shape.
    """
    method_upper = method.upper()
    path_clean = path.rstrip("/")

    # Exact match first
    for route_method, route_path, handler in _ROUTES:
        if method_upper == route_method and path_clean == route_path:
            return handler(json=json, params=params)

    # Prefix match for parameterized routes (e.g., /api/v1/debates/{id})
    for route_method, route_path, handler in _ROUTES:
        if method_upper == route_method and path_clean.startswith(route_path + "/"):
            return handler(json=json, params=params)

    # Fallback
    return _handle_not_found(path=path)
