"""
MCP Debate Tools.

Core debate operations: run, get, search, fork.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List

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
    from aragora.core import Environment
    from aragora.debate.orchestrator import Arena, DebateProtocol

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
                model_type=agent_name,  # type: ignore[arg-type]
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
        consensus=consensus,  # type: ignore[arg-type]
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
            search_result = db.search(
                query=query,
                limit=limit,
            )
            # search returns tuple (list[DebateMetadata], count) - extract list
            all_debates = search_result[0] if isinstance(search_result, tuple) else search_result
            # Apply additional filters (agent, consensus_only) in memory
            for debate_meta in all_debates:
                # Convert DebateMetadata to dict (dataclass has typed fields)
                debate_dict: Dict[str, Any] = {
                    "debate_id": getattr(debate_meta, "debate_id", ""),
                    "task": getattr(debate_meta, "task", ""),
                    "agents": getattr(debate_meta, "agents", []),
                    "consensus_reached": getattr(debate_meta, "consensus_reached", False),
                    "confidence": getattr(debate_meta, "confidence", 0.0),
                    "created_at": str(getattr(debate_meta, "created_at", "")),
                }
                if agent:
                    agents_list = debate_dict.get("agents", [])
                    if not any(agent.lower() in str(a).lower() for a in agents_list):
                        continue
                if consensus_only and not debate_dict.get("consensus_reached", False):
                    continue
                results.append(debate_dict)
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


async def fork_debate_tool(
    debate_id: str,
    branch_point: int = -1,
    modified_context: str = "",
) -> Dict[str, Any]:
    """
    Fork a debate to explore counterfactual scenarios.

    Args:
        debate_id: ID of the debate to fork
        branch_point: Message index to branch from (-1 for last message)
        modified_context: Optional modified context for the fork

    Returns:
        Dict with fork ID and inherited message count
    """
    if not debate_id:
        return {"error": "debate_id is required"}

    try:
        from aragora.server.storage import get_debates_db

        db = get_debates_db()
        if not db:
            return {"error": "Storage not available"}

        # Fetch the parent debate
        debate = db.get(debate_id)
        if not debate:
            return {"error": f"Debate {debate_id} not found"}

        # Get messages from the debate
        messages = debate.get("messages", [])
        if not messages:
            return {"error": "Debate has no messages to fork from"}

        # Determine branch point
        if branch_point < 0:
            branch_point = len(messages) + branch_point
        branch_point = max(0, min(branch_point, len(messages) - 1))

        # Create a fork entry in storage
        fork_id = f"fork-{uuid.uuid4().hex[:8]}"

        fork_data = {
            "debate_id": fork_id,
            "parent_debate_id": debate_id,
            "branch_point": branch_point,
            "task": modified_context or f"Fork of: {debate.get('task', 'Unknown task')}",
            "messages": messages[: branch_point + 1],  # Inherit messages up to branch point
            "consensus_reached": False,
            "confidence": 0.0,
            "status": "forked",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save the fork to storage using save_dict
        if hasattr(db, "save_dict"):
            db.save_dict(fork_data)

        return {
            "success": True,
            "fork_id": fork_id,
            "parent_debate_id": debate_id,
            "branch_point": branch_point,
            "inherited_messages": branch_point + 1,
            "modified_context": modified_context or "(none)",
        }

    except ImportError as e:
        return {"error": f"Required module not available: {e}"}
    except Exception as e:
        return {"error": f"Fork creation failed: {e}"}


async def get_forks_tool(
    debate_id: str,
    include_nested: bool = False,
) -> Dict[str, Any]:
    """
    Get all forks of a debate.

    Args:
        debate_id: ID of the parent debate
        include_nested: Include forks of forks

    Returns:
        Dict with list of forks
    """
    if not debate_id:
        return {"error": "debate_id is required"}

    try:
        from aragora.server.storage import get_debates_db

        db = get_debates_db()
        if not db:
            return {"error": "Storage not available"}

        # Get forks from storage
        forks: List[Dict[str, Any]] = []
        if hasattr(db, "get_forks"):
            forks = db.get_forks(debate_id, include_nested=include_nested)
        else:
            # Fallback: search for debates with parent_id via search
            search_result = db.search(query="", limit=100)
            all_debates = search_result[0] if isinstance(search_result, tuple) else search_result
            for debate_meta in all_debates:
                parent_id = getattr(debate_meta, "parent_debate_id", None)
                if parent_id == debate_id:
                    forks.append(
                        {
                            "branch_id": getattr(debate_meta, "debate_id", ""),
                            "task": getattr(debate_meta, "task", ""),
                            "branch_point": getattr(debate_meta, "branch_point", 0),
                            "created_at": str(getattr(debate_meta, "created_at", "")),
                        }
                    )

        return {
            "parent_debate_id": debate_id,
            "forks": forks,
            "count": len(forks),
        }

    except Exception as e:
        return {"error": f"Failed to get forks: {e}"}


__all__ = [
    "run_debate_tool",
    "get_debate_tool",
    "search_debates_tool",
    "fork_debate_tool",
    "get_forks_tool",
]
