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
    Run a multi-agent debate on a topic.

    Args:
        question: The question or topic to debate
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
    from aragora.agents.registry import list_available_agents

    try:
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
]


__all__ = [
    "run_debate_tool",
    "run_gauntlet_tool",
    "list_agents_tool",
    "get_debate_tool",
    "TOOLS_METADATA",
]
