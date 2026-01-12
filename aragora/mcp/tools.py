"""
MCP Tool Definitions for Aragora.

These functions can be used standalone or registered with an MCP server.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def run_debate_tool(
    question: str,
    agents: str = "anthropic-api,openai-api",
    rounds: int = 3,
    consensus: str = "majority",
) -> Dict[str, Any]:
    """
    Run a decision stress-test (debate engine) on a topic.

    Args:
        question: The question or topic to stress-test
        agents: Comma-separated agent IDs
        rounds: Number of debate rounds (1-10)
        consensus: Consensus mechanism (majority, unanimous, none)

    Returns:
        Dict with debate results including final_answer, consensus status, confidence
    """
    from aragora.agents.base import create_agent
    from aragora.debate.orchestrator import Arena, DebateProtocol
    from aragora.core import Environment

    if not question:
        return {"error": "Question is required"}

    # Validate rounds
    rounds = min(max(rounds, 1), 10)

    # Parse and create agents
    agent_names = [a.strip() for a in agents.split(",")]
    agent_list = []
    roles = ["proposer", "critic", "synthesizer"]

    for i, agent_name in enumerate(agent_names):
        role = roles[i] if i < len(roles) else "critic"
        try:
            agent = create_agent(
                model_type=agent_name,
                name=f"{agent_name}_{role}",
                role=role,
            )
            agent_list.append(agent)
        except Exception as e:
            logger.warning(f"Could not create agent {agent_name}: {e}")

    if not agent_list:
        return {"error": "No valid agents could be created. Check API keys."}

    # Create environment and protocol
    env = Environment(
        task=question,
        max_rounds=rounds,
    )

    protocol = DebateProtocol(
        rounds=rounds,
        consensus=consensus,
    )

    # Run debate
    arena = Arena(env, agent_list, protocol)
    result = await arena.run()

    # Generate debate ID
    debate_id = f"mcp_{uuid.uuid4().hex[:8]}"

    return {
        "debate_id": debate_id,
        "task": question,
        "final_answer": result.final_answer,
        "consensus_reached": result.consensus_reached,
        "confidence": result.confidence,
        "rounds_used": result.rounds_used,
        "agents": [a.name for a in agent_list],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


async def run_gauntlet_tool(
    content: str,
    content_type: str = "spec",
    profile: str = "quick",
) -> Dict[str, Any]:
    """
    Run gauntlet stress-test on content.

    Args:
        content: The content to stress-test
        content_type: Type of content (spec, code, policy, architecture)
        profile: Test profile (quick, thorough, code, security, gdpr, hipaa)

    Returns:
        Dict with verdict, risk score, and vulnerabilities found
    """
    from aragora.gauntlet import (
        GauntletRunner,
        GauntletConfig,
        QUICK_GAUNTLET,
        THOROUGH_GAUNTLET,
        CODE_REVIEW_GAUNTLET,
        SECURITY_GAUNTLET,
        GDPR_GAUNTLET,
        HIPAA_GAUNTLET,
    )

    if not content:
        return {"error": "Content is required"}

    # Select profile
    profiles = {
        "quick": QUICK_GAUNTLET,
        "thorough": THOROUGH_GAUNTLET,
        "code": CODE_REVIEW_GAUNTLET,
        "security": SECURITY_GAUNTLET,
        "gdpr": GDPR_GAUNTLET,
        "hipaa": HIPAA_GAUNTLET,
    }

    base_config = profiles.get(profile, QUICK_GAUNTLET)

    config = GauntletConfig(
        attack_categories=base_config.attack_categories,
        agents=base_config.agents,
        rounds_per_attack=base_config.rounds_per_attack,
    )

    runner = GauntletRunner(config)
    result = await runner.run(content)

    vulnerabilities = getattr(result, 'vulnerabilities', [])

    return {
        "verdict": result.verdict.value if hasattr(result, 'verdict') else "unknown",
        "risk_score": getattr(result, 'risk_score', 0),
        "vulnerabilities_count": len(vulnerabilities),
        "vulnerabilities": [
            {
                "category": v.category,
                "severity": v.severity,
                "description": v.description,
            }
            for v in vulnerabilities[:10]  # Limit to 10
        ],
        "content_type": content_type,
        "profile": profile,
    }


async def list_agents_tool() -> Dict[str, Any]:
    """
    List available AI agents.

    Returns:
        Dict with list of available agent IDs and count
    """
    try:
        from aragora.agents.registry import list_available_agents

        agents = list_available_agents()
        return {
            "agents": agents,
            "count": len(agents),
        }
    except Exception as e:
        logger.warning(f"Could not list agents: {e}")
        # Fallback list
        return {
            "agents": [
                "anthropic-api",
                "openai-api",
                "gemini",
                "grok",
                "deepseek",
            ],
            "count": 5,
            "note": "Fallback list - some agents may not be available",
        }


async def get_debate_tool(debate_id: str) -> Dict[str, Any]:
    """
    Get results of a previous debate.

    Args:
        debate_id: The debate ID to retrieve

    Returns:
        Dict with debate results or error
    """
    if not debate_id:
        return {"error": "debate_id is required"}

    # Try to load from storage
    try:
        from aragora.server.storage import get_debates_db
        db = get_debates_db()
        if db:
            debate = db.get(debate_id)
            if debate:
                return debate
    except Exception as e:
        logger.warning(f"Could not fetch debate from storage: {e}")

    return {"error": f"Debate {debate_id} not found"}


async def search_debates_tool(
    query: str = "",
    agent: str = "",
    start_date: str = "",
    end_date: str = "",
    consensus_only: bool = False,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Search debates by topic, date range, or participating agents.

    Args:
        query: Search query for topic text
        agent: Filter by agent name
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        consensus_only: Only return debates that reached consensus
        limit: Max results (1-100)

    Returns:
        Dict with matching debates and count
    """
    limit = min(max(limit, 1), 100)
    results: List[Dict[str, Any]] = []

    try:
        from aragora.server.storage import get_debates_db
        db = get_debates_db()
        if db and hasattr(db, "search"):
            results = db.search(
                query=query,
                agent=agent,
                consensus_only=consensus_only,
                limit=limit,
            )
        elif db and hasattr(db, "list"):
            # Fallback to list + filter
            all_debates = db.list(limit=limit * 2)
            for debate in all_debates:
                if query and query.lower() not in debate.get("task", "").lower():
                    continue
                if agent:
                    agents = debate.get("agents", [])
                    if not any(agent.lower() in a.lower() for a in agents):
                        continue
                if consensus_only and not debate.get("consensus_reached", False):
                    continue
                results.append(debate)
    except Exception as e:
        logger.warning(f"Could not search debates: {e}")

    results = results[:limit]

    return {
        "debates": results,
        "count": len(results),
        "query": query or "(all)",
        "filters": {
            "agent": agent or None,
            "consensus_only": consensus_only,
            "date_range": f"{start_date or '*'} to {end_date or '*'}",
        },
    }


async def get_agent_history_tool(
    agent_name: str,
    include_debates: bool = True,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Get an agent's debate history, ELO rating, and performance stats.

    Args:
        agent_name: The agent name (e.g., 'anthropic-api', 'openai-api')
        include_debates: Include recent debate summaries
        limit: Max debates to include

    Returns:
        Dict with agent stats, ELO rating, and optionally recent debates
    """
    if not agent_name:
        return {"error": "agent_name is required"}

    result: Dict[str, Any] = {
        "agent_name": agent_name,
        "elo_rating": 1500,
        "total_debates": 0,
        "consensus_rate": 0.0,
        "win_rate": 0.0,
    }

    # Get ELO rating
    try:
        from aragora.ranking.elo import ELOSystem
        elo = ELOSystem()
        rating = elo.get_rating(agent_name)
        if rating:
            result["elo_rating"] = rating.rating
            result["elo_deviation"] = rating.deviation
    except Exception as e:
        logger.debug(f"Could not get ELO: {e}")

    # Get performance stats from storage
    try:
        from aragora.server.storage import get_debates_db
        db = get_debates_db()
        if db and hasattr(db, "get_agent_stats"):
            stats = db.get_agent_stats(agent_name)
            if stats:
                result.update({
                    "total_debates": stats.get("total_debates", 0),
                    "consensus_rate": stats.get("consensus_rate", 0.0),
                    "win_rate": stats.get("win_rate", 0.0),
                    "avg_confidence": stats.get("avg_confidence", 0.0),
                })

        # Get recent debates if requested
        if include_debates and db and hasattr(db, "search"):
            debates = db.search(agent=agent_name, limit=limit)
            result["recent_debates"] = [
                {
                    "debate_id": d.get("debate_id"),
                    "task": d.get("task", "")[:80],
                    "consensus_reached": d.get("consensus_reached", False),
                    "timestamp": d.get("timestamp", ""),
                }
                for d in debates
            ]
    except Exception as e:
        logger.debug(f"Could not get agent history: {e}")

    return result


async def get_consensus_proofs_tool(
    debate_id: str = "",
    proof_type: str = "all",
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Retrieve formal verification proofs from debates.

    Args:
        debate_id: Specific debate ID to get proofs for (optional)
        proof_type: Type of proofs ('z3', 'lean', 'all')
        limit: Max proofs to return

    Returns:
        Dict with proofs list and count
    """
    proofs: List[Dict[str, Any]] = []

    try:
        from aragora.server.storage import get_proofs_db
        proofs_db = get_proofs_db()
        if proofs_db:
            proofs = proofs_db.list(
                debate_id=debate_id or None,
                proof_type=proof_type if proof_type != "all" else None,
                limit=limit,
            )
    except Exception as e:
        logger.debug(f"Proofs storage unavailable: {e}")

    # If no storage, try to get from debate data
    if not proofs and debate_id:
        try:
            from aragora.server.storage import get_debates_db
            db = get_debates_db()
            if db:
                debate = db.get(debate_id)
                if debate and "proofs" in debate:
                    for proof in debate["proofs"]:
                        if proof_type == "all" or proof.get("type") == proof_type:
                            proofs.append(proof)
        except Exception as e:
            logger.debug(f"Failed to fetch proofs for debate {debate_id}: {e}")

    proofs = proofs[:limit]

    return {
        "proofs": proofs,
        "count": len(proofs),
        "debate_id": debate_id or "(all debates)",
        "proof_type": proof_type,
    }


async def list_trending_topics_tool(
    platform: str = "all",
    category: str = "",
    min_score: float = 0.5,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Get trending topics from Pulse that could make good debates.

    Args:
        platform: Source platform ('hackernews', 'reddit', 'arxiv', 'all')
        category: Topic category filter
        min_score: Minimum topic score (0-1)
        limit: Max topics to return

    Returns:
        Dict with scored topics and count
    """
    topics: List[Dict[str, Any]] = []

    try:
        from aragora.pulse import get_trending_topics
        from aragora.pulse.scheduler import TopicSelector

        # Get raw topics
        raw_topics = await get_trending_topics(
            platforms=[platform] if platform != "all" else None,
            limit=limit * 2,
        )

        # Score topics
        selector = TopicSelector()

        for topic in raw_topics:
            if platform != "all" and topic.platform != platform:
                continue
            if category and topic.category.lower() != category.lower():
                continue

            score = selector.score_topic(topic)

            if score >= min_score:
                topics.append({
                    "topic": topic.topic,
                    "platform": topic.platform,
                    "category": topic.category,
                    "score": round(score, 3),
                    "volume": topic.volume,
                    "debate_potential": "high" if score > 0.7 else "medium",
                })

        topics.sort(key=lambda x: x["score"], reverse=True)
        topics = topics[:limit]

    except ImportError:
        logger.warning("Pulse module not available")
    except Exception as e:
        logger.warning(f"Could not fetch trending topics: {e}")

    return {
        "topics": topics,
        "count": len(topics),
        "platform": platform,
        "category": category or "(all)",
        "min_score": min_score,
    }


# Tool metadata for registration
TOOLS_METADATA = [
    {
        "name": "run_debate",
        "description": "Run a multi-agent AI debate on a topic",
        "function": run_debate_tool,
        "parameters": {
            "question": {"type": "string", "required": True},
            "agents": {"type": "string", "default": "anthropic-api,openai-api"},
            "rounds": {"type": "integer", "default": 3},
            "consensus": {"type": "string", "default": "majority"},
        },
    },
    {
        "name": "run_gauntlet",
        "description": "Stress-test content through adversarial analysis",
        "function": run_gauntlet_tool,
        "parameters": {
            "content": {"type": "string", "required": True},
            "content_type": {"type": "string", "default": "spec"},
            "profile": {"type": "string", "default": "quick"},
        },
    },
    {
        "name": "list_agents",
        "description": "List available AI agents",
        "function": list_agents_tool,
        "parameters": {},
    },
    {
        "name": "get_debate",
        "description": "Get results of a previous debate",
        "function": get_debate_tool,
        "parameters": {
            "debate_id": {"type": "string", "required": True},
        },
    },
    {
        "name": "search_debates",
        "description": "Search debates by topic, date, or agents",
        "function": search_debates_tool,
        "parameters": {
            "query": {"type": "string"},
            "agent": {"type": "string"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
            "consensus_only": {"type": "boolean", "default": False},
            "limit": {"type": "integer", "default": 20},
        },
    },
    {
        "name": "get_agent_history",
        "description": "Get agent debate history and performance stats",
        "function": get_agent_history_tool,
        "parameters": {
            "agent_name": {"type": "string", "required": True},
            "include_debates": {"type": "boolean", "default": True},
            "limit": {"type": "integer", "default": 10},
        },
    },
    {
        "name": "get_consensus_proofs",
        "description": "Retrieve formal verification proofs from debates",
        "function": get_consensus_proofs_tool,
        "parameters": {
            "debate_id": {"type": "string"},
            "proof_type": {"type": "string", "default": "all"},
            "limit": {"type": "integer", "default": 10},
        },
    },
    {
        "name": "list_trending_topics",
        "description": "Get trending topics from Pulse for debates",
        "function": list_trending_topics_tool,
        "parameters": {
            "platform": {"type": "string", "default": "all"},
            "category": {"type": "string"},
            "min_score": {"type": "number", "default": 0.5},
            "limit": {"type": "integer", "default": 10},
        },
    },
]


__all__ = [
    "run_debate_tool",
    "run_gauntlet_tool",
    "list_agents_tool",
    "get_debate_tool",
    "search_debates_tool",
    "get_agent_history_tool",
    "get_consensus_proofs_tool",
    "list_trending_topics_tool",
    "TOOLS_METADATA",
]
