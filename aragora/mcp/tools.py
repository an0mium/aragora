"""
MCP Tool Definitions for Aragora.

These functions can be used standalone or registered with an MCP server.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Union

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
    arena = Arena(env, agent_list, protocol)  # type: ignore[arg-type]
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
    from aragora.gauntlet import GauntletConfig, GauntletRunner
    from aragora.gauntlet.config import AttackCategory

    if not content:
        return {"error": "Content is required"}

    # Configure based on profile
    if profile == "security":
        attack_categories = [AttackCategory.SECURITY, AttackCategory.ADVERSARIAL_INPUT]
    elif profile == "code":
        attack_categories = [AttackCategory.LOGIC, AttackCategory.EDGE_CASE]
    else:  # quick/thorough/gdpr/hipaa
        attack_categories = [AttackCategory.SECURITY, AttackCategory.LOGIC, AttackCategory.ARCHITECTURE]

    config = GauntletConfig(
        name=f"{profile}_gauntlet",
        input_type=content_type,
        attack_categories=attack_categories,
        attack_rounds=2 if profile == "quick" else 3,
    )

    runner = GauntletRunner(config)
    result = await runner.run(content)

    vulnerabilities = getattr(result, "vulnerabilities", [])

    return {
        "verdict": result.verdict.value if hasattr(result, "verdict") else "unknown",
        "risk_score": getattr(result, "risk_score", 0),
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
        from aragora.agents.base import list_available_agents

        agents_dict = list_available_agents()
        agents = list(agents_dict.keys())
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
        from aragora.ranking.elo import EloSystem

        elo_system = EloSystem()
        agent_rating = elo_system.get_rating(agent_name)
        if agent_rating:
            result["elo_rating"] = agent_rating.elo
            result["wins"] = agent_rating.wins
            result["losses"] = agent_rating.losses
            result["total_debates"] = agent_rating.debates_count
    except Exception as e:
        logger.debug(f"Could not get ELO: {e}")

    # Get performance stats from storage
    try:
        from aragora.server.storage import get_debates_db

        db = get_debates_db()
        if db and hasattr(db, "get_agent_stats"):
            stats = db.get_agent_stats(agent_name)
            if stats:
                result.update(
                    {
                        "total_debates": stats.get("total_debates", 0),
                        "consensus_rate": stats.get("consensus_rate", 0.0),
                        "win_rate": stats.get("win_rate", 0.0),
                        "avg_confidence": stats.get("avg_confidence", 0.0),
                    }
                )

        # Get recent debates if requested
        if include_debates and db and hasattr(db, "search"):
            search_result = db.search(query="", limit=limit)
            # search returns tuple (list, count) - extract list and filter by agent
            all_debates = search_result[0] if isinstance(search_result, tuple) else search_result
            filtered_debates: List[Dict[str, Any]] = []
            for debate in all_debates:
                debate_dict = debate if isinstance(debate, dict) else vars(debate)
                agents_list = debate_dict.get("agents", [])
                if any(agent_name.lower() in str(a).lower() for a in agents_list):
                    filtered_debates.append(debate_dict)
            result["recent_debates"] = [
                {
                    "debate_id": d.get("debate_id"),
                    "task": d.get("task", "")[:80],
                    "consensus_reached": d.get("consensus_reached", False),
                    "timestamp": d.get("timestamp", ""),
                }
                for d in filtered_debates[:limit]
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

    # Try to get proofs from debate data
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
        from aragora.pulse import PulseManager, SchedulerConfig, TopicSelector

        # Create pulse manager and fetch topics
        pulse_manager = PulseManager()
        platforms_list = [platform] if platform != "all" else None
        raw_topics = await pulse_manager.get_trending_topics(
            platforms=platforms_list,
            limit_per_platform=limit * 2,
        )

        # Score topics
        config = SchedulerConfig()
        selector = TopicSelector(config)

        for topic in raw_topics:
            if platform != "all" and topic.platform != platform:
                continue
            if category and getattr(topic, "category", "").lower() != category.lower():
                continue

            topic_score = selector.score_topic(topic)

            if topic_score.score >= min_score:
                topics.append(
                    {
                        "topic": topic.topic,
                        "platform": topic.platform,
                        "category": getattr(topic, "category", ""),
                        "score": round(topic_score.score, 3),
                        "volume": getattr(topic, "volume", 0),
                        "debate_potential": "high" if topic_score.score > 0.7 else "medium",
                    }
                )

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


# === Memory Tools ===


async def query_memory_tool(
    query: str,
    tier: str = "all",
    limit: int = 10,
    min_importance: float = 0.0,
) -> Dict[str, Any]:
    """
    Query memories from the continuum memory system.

    Args:
        query: Search query for memory content
        tier: Memory tier (fast, medium, slow, glacial, all)
        limit: Max memories to return (1-100)
        min_importance: Minimum importance score (0-1)

    Returns:
        Dict with matching memories and count
    """
    limit = min(max(limit, 1), 100)
    memories: List[Dict[str, Any]] = []

    try:
        from aragora.memory.continuum import ContinuumMemory, MemoryTier

        continuum = ContinuumMemory()

        # Parse tier
        tiers = None
        if tier != "all":
            try:
                tiers = [MemoryTier[tier.upper()]]
            except KeyError:
                pass

        results = continuum.retrieve(
            query=query,
            tiers=tiers or list(MemoryTier),
            limit=limit,
            min_importance=min_importance,
        )

        for m in results:
            memories.append(
                {
                    "id": m.id,
                    "tier": m.tier.name.lower(),
                    "content": m.content[:500] + "..." if len(m.content) > 500 else m.content,
                    "importance": round(m.importance, 3),
                    "created_at": m.created_at if m.created_at else None,
                }
            )

    except ImportError:
        logger.warning("Continuum memory not available")
    except Exception as e:
        logger.warning(f"Memory query failed: {e}")

    return {
        "memories": memories,
        "count": len(memories),
        "query": query,
        "tier": tier,
    }


async def store_memory_tool(
    content: str,
    tier: str = "medium",
    importance: float = 0.5,
    tags: str = "",
) -> Dict[str, Any]:
    """
    Store a memory in the continuum memory system.

    Args:
        content: Memory content to store
        tier: Memory tier (fast, medium, slow, glacial)
        importance: Importance score (0-1)
        tags: Comma-separated tags

    Returns:
        Dict with stored memory ID and status
    """
    if not content:
        return {"error": "content is required"}

    try:
        import uuid

        from aragora.memory.continuum import ContinuumMemory, MemoryTier

        continuum = ContinuumMemory()

        # Parse tier
        try:
            memory_tier = MemoryTier[tier.upper()]
        except KeyError:
            memory_tier = MemoryTier.MEDIUM

        # Generate ID and store memory using the add method
        memory_id = f"mcp_{uuid.uuid4().hex[:12]}"
        continuum.add(
            id=memory_id,
            content=content,
            tier=memory_tier,
            importance=min(max(importance, 0.0), 1.0),
        )

        return {
            "success": True,
            "memory_id": memory_id,
            "tier": memory_tier.name.lower(),
            "importance": importance,
        }

    except ImportError:
        return {"error": "Continuum memory not available"}
    except Exception as e:
        return {"error": f"Failed to store memory: {e}"}


async def get_memory_pressure_tool() -> Dict[str, Any]:
    """
    Get current memory pressure and utilization.

    Returns:
        Dict with pressure score, status, and tier utilization
    """
    try:
        from aragora.memory.continuum import ContinuumMemory

        continuum = ContinuumMemory()
        pressure = continuum.get_memory_pressure()
        stats = continuum.get_stats()

        # Determine status
        if pressure < 0.5:
            status = "normal"
        elif pressure < 0.8:
            status = "elevated"
        elif pressure < 0.9:
            status = "high"
        else:
            status = "critical"

        return {
            "pressure": round(pressure, 3),
            "status": status,
            "total_memories": stats.get("total_memories", 0),
            "tier_stats": stats.get("by_tier", {}),
            "cleanup_recommended": pressure > 0.9,
        }

    except ImportError:
        return {"error": "Continuum memory not available"}
    except Exception as e:
        return {"error": f"Failed to get memory pressure: {e}"}


# === Fork Tools ===


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
        import uuid

        from aragora.debate.counterfactual import CounterfactualOrchestrator, PivotClaim
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
            "messages": messages[:branch_point + 1],  # Inherit messages up to branch point
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


# === Genesis Tools ===


async def get_agent_lineage_tool(
    agent_name: str,
    depth: int = 5,
) -> Dict[str, Any]:
    """
    Get the evolutionary lineage of an agent.

    Args:
        agent_name: Name of the agent
        depth: How many generations back to trace

    Returns:
        Dict with lineage tree
    """
    if not agent_name:
        return {"error": "agent_name is required"}

    depth = min(max(depth, 1), 20)

    try:
        from aragora.genesis.genome import GenomeStore

        # Try to get genome from the genome store
        store = GenomeStore()
        genome = store.get_by_name(agent_name)

        if not genome:
            # Try looking up by genome_id directly
            genome = store.get(agent_name)

        if not genome:
            return {
                "agent_name": agent_name,
                "lineage": [],
                "generation": 0,
                "note": "Agent not found in genesis database. May be a base agent without evolutionary history.",
            }

        # Build lineage tree
        lineage = []
        current = genome
        visited = set()

        for _ in range(depth):
            if not current or current.genome_id in visited:
                break

            visited.add(current.genome_id)
            lineage.append({
                "genome_id": current.genome_id,
                "name": current.name,
                "generation": current.generation,
                "fitness_score": current.fitness_score,
                "parent_genomes": current.parent_genomes,
                "model_preference": current.model_preference,
                "birth_debate_id": current.birth_debate_id,
            })

            # Get first parent for next iteration
            if current.parent_genomes:
                parent_id = current.parent_genomes[0]
                current = store.get(parent_id)
            else:
                break

        return {
            "agent_name": agent_name,
            "genome_id": genome.genome_id,
            "generation": genome.generation,
            "lineage": lineage,
            "depth_traced": len(lineage),
        }

    except ImportError:
        return {"error": "Genesis module not available"}
    except Exception as e:
        return {"error": f"Failed to get lineage: {e}"}


async def breed_agents_tool(
    parent_a: str,
    parent_b: str,
    mutation_rate: float = 0.1,
) -> Dict[str, Any]:
    """
    Breed two agents to create a new offspring agent.

    Args:
        parent_a: First parent agent name or genome_id
        parent_b: Second parent agent name or genome_id
        mutation_rate: Mutation rate (0-1)

    Returns:
        Dict with offspring agent info
    """
    if not parent_a or not parent_b:
        return {"error": "Both parent_a and parent_b are required"}

    mutation_rate = min(max(mutation_rate, 0.0), 1.0)

    try:
        from aragora.genesis.breeding import GenomeBreeder
        from aragora.genesis.genome import GenomeStore

        store = GenomeStore()

        # Look up parent genomes (by name or genome_id)
        genome_a = store.get_by_name(parent_a) or store.get(parent_a)
        genome_b = store.get_by_name(parent_b) or store.get(parent_b)

        if not genome_a:
            return {"error": f"Parent agent '{parent_a}' not found in genesis database"}
        if not genome_b:
            return {"error": f"Parent agent '{parent_b}' not found in genesis database"}

        # Create breeder with specified mutation rate
        breeder = GenomeBreeder(mutation_rate=mutation_rate)

        # Crossover to create offspring
        offspring = breeder.crossover(
            parent_a=genome_a,
            parent_b=genome_b,
            debate_id=f"mcp_breed_{uuid.uuid4().hex[:8]}",
        )

        # Apply mutation
        if mutation_rate > 0:
            offspring = breeder.mutate(offspring, rate=mutation_rate)

        # Save to store
        store.save(offspring)

        return {
            "success": True,
            "offspring": {
                "genome_id": offspring.genome_id,
                "name": offspring.name,
                "generation": offspring.generation,
                "parent_genomes": offspring.parent_genomes,
                "model_preference": offspring.model_preference,
                "fitness_score": offspring.fitness_score,
                "traits_count": len(offspring.traits),
                "expertise_count": len(offspring.expertise),
            },
            "parents": {
                "parent_a": {
                    "genome_id": genome_a.genome_id,
                    "name": genome_a.name,
                    "generation": genome_a.generation,
                },
                "parent_b": {
                    "genome_id": genome_b.genome_id,
                    "name": genome_b.name,
                    "generation": genome_b.generation,
                },
            },
            "mutation_rate": mutation_rate,
        }

    except ImportError:
        return {"error": "Genesis/breeding module not available"}
    except Exception as e:
        return {"error": f"Breeding failed: {e}"}


# === Checkpoint Tools ===


async def create_checkpoint_tool(
    debate_id: str,
    label: str = "",
    storage_backend: str = "file",
) -> Dict[str, Any]:
    """
    Create a checkpoint for a debate to enable resume later.

    Args:
        debate_id: ID of the debate to checkpoint
        label: Optional label for the checkpoint
        storage_backend: Storage backend (file, s3, git, database)

    Returns:
        Dict with checkpoint ID and status
    """
    if not debate_id:
        return {"error": "debate_id is required"}

    try:
        from aragora.core import Critique, Message, Vote
        from aragora.debate.checkpoint import (
            AgentState,
            CheckpointConfig,
            CheckpointManager,
            CheckpointStore,
            DatabaseCheckpointStore,
            FileCheckpointStore,
        )
        from aragora.server.storage import get_debates_db

        db = get_debates_db()
        if not db:
            return {"error": "Storage not available"}

        # Fetch debate data
        debate = db.get(debate_id)
        if not debate:
            return {"error": f"Debate {debate_id} not found"}

        # Choose storage backend
        checkpoint_store: CheckpointStore
        if storage_backend == "database":
            checkpoint_store = DatabaseCheckpointStore()
        else:
            checkpoint_store = FileCheckpointStore()

        manager = CheckpointManager(store=checkpoint_store)

        # Convert stored messages to Message objects
        messages = []
        for m in debate.get("messages", []):
            messages.append(Message(
                role=m.get("role", "assistant"),
                agent=m.get("agent", "unknown"),
                content=m.get("content", ""),
                round=m.get("round", 0),
            ))

        # Convert stored votes to Vote objects
        votes = []
        for v in debate.get("votes", []):
            votes.append(Vote(
                agent=v.get("agent", "unknown"),
                choice=v.get("choice", ""),
                confidence=v.get("confidence", 0.5),
                reasoning=v.get("reasoning", ""),
            ))

        # Convert stored critiques to Critique objects
        critiques = []
        for c in debate.get("critiques", []):
            critiques.append(Critique(
                agent=c.get("agent", "unknown"),
                target_agent=c.get("target_agent", ""),
                target_content=c.get("target_content", ""),
                issues=c.get("issues", []),
                suggestions=c.get("suggestions", []),
                severity=c.get("severity", "low"),
                reasoning=c.get("reasoning", ""),
            ))

        # Create simple agent states from debate metadata
        # Use a simple class to hold agent info for checkpoint creation
        class _SimpleAgentHolder:
            def __init__(self, name: str, model: str = "unknown", role: str = "participant"):
                self.name = name
                self.model = model
                self.role = role

        agents_info = debate.get("agents", [])
        agents = []
        for a in agents_info:
            if isinstance(a, str):
                # Just agent name string
                agents.append(_SimpleAgentHolder(a))
            elif isinstance(a, dict):
                agents.append(_SimpleAgentHolder(
                    name=a.get("name", "unknown"),
                    model=a.get("model", "unknown"),
                    role=a.get("role", "participant"),
                ))

        # Create checkpoint
        checkpoint = await manager.create_checkpoint(
            debate_id=debate_id,
            task=debate.get("task", "Unknown task"),
            current_round=debate.get("rounds_used", len(messages) // 3),
            total_rounds=debate.get("total_rounds", 3),
            phase=debate.get("phase", "completed"),
            messages=messages,
            critiques=critiques,
            votes=votes,
            agents=agents,
            current_consensus=debate.get("final_answer"),
        )

        return {
            "success": True,
            "checkpoint_id": checkpoint.checkpoint_id,
            "debate_id": debate_id,
            "label": label or "(none)",
            "storage_backend": storage_backend,
            "current_round": checkpoint.current_round,
            "message_count": len(checkpoint.messages),
            "created_at": checkpoint.created_at,
        }

    except ImportError as e:
        return {"error": f"Checkpoint module not available: {e}"}
    except Exception as e:
        return {"error": f"Failed to create checkpoint: {e}"}


async def list_checkpoints_tool(
    debate_id: str = "",
    include_expired: bool = False,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    List checkpoints for a debate or all debates.

    Args:
        debate_id: Optional debate ID to filter by
        include_expired: Include expired checkpoints
        limit: Max checkpoints to return

    Returns:
        Dict with list of checkpoints
    """
    limit = min(max(limit, 1), 100)

    try:
        from aragora.debate.checkpoint import CheckpointManager

        manager = CheckpointManager()
        # Use store.list_checkpoints which is the actual API
        checkpoints = await manager.store.list_checkpoints(
            debate_id=debate_id or None,
            limit=limit,
        )

        return {
            "checkpoints": [
                {
                    "checkpoint_id": c.get("checkpoint_id", ""),
                    "debate_id": c.get("debate_id", ""),
                    "task": c.get("task", ""),
                    "current_round": c.get("current_round", 0),
                    "message_count": c.get("message_count", 0),
                }
                for c in checkpoints
            ],
            "count": len(checkpoints),
            "debate_id": debate_id or "(all)",
        }

    except ImportError:
        return {"error": "Checkpoint module not available"}
    except Exception as e:
        return {"error": f"Failed to list checkpoints: {e}"}


async def resume_checkpoint_tool(
    checkpoint_id: str,
) -> Dict[str, Any]:
    """
    Resume a debate from a checkpoint.

    Args:
        checkpoint_id: ID of the checkpoint to resume

    Returns:
        Dict with resumed debate info
    """
    if not checkpoint_id:
        return {"error": "checkpoint_id is required"}

    try:
        from aragora.debate.checkpoint import CheckpointManager

        manager = CheckpointManager()
        # Load checkpoint via store
        checkpoint = await manager.store.load(checkpoint_id)

        if not checkpoint:
            return {"error": f"Checkpoint {checkpoint_id} not found"}

        return {
            "success": True,
            "checkpoint_id": checkpoint_id,
            "debate_id": checkpoint.debate_id,
            "messages_count": len(checkpoint.messages),
            "round": checkpoint.current_round,
            "phase": checkpoint.phase,
            "task": checkpoint.task,
        }

    except ImportError:
        return {"error": "Checkpoint module not available"}
    except Exception as e:
        return {"error": f"Failed to resume checkpoint: {e}"}


async def delete_checkpoint_tool(
    checkpoint_id: str,
) -> Dict[str, Any]:
    """
    Delete a checkpoint.

    Args:
        checkpoint_id: ID of the checkpoint to delete

    Returns:
        Dict with deletion status
    """
    if not checkpoint_id:
        return {"error": "checkpoint_id is required"}

    try:
        from aragora.debate.checkpoint import CheckpointManager

        manager = CheckpointManager()
        # Delete via store
        success = await manager.store.delete(checkpoint_id)

        return {
            "success": success,
            "checkpoint_id": checkpoint_id,
            "message": "Checkpoint deleted" if success else "Checkpoint not found",
        }

    except ImportError:
        return {"error": "Checkpoint module not available"}
    except Exception as e:
        return {"error": f"Failed to delete checkpoint: {e}"}


# === Verification Tools ===


async def verify_consensus_tool(
    debate_id: str,
    backend: str = "z3",
) -> Dict[str, Any]:
    """
    Verify the consensus of a completed debate using formal methods.

    Args:
        debate_id: ID of the debate to verify
        backend: Verification backend (z3, lean4)

    Returns:
        Dict with verification result and proof
    """
    if not debate_id:
        return {"error": "debate_id is required"}

    try:
        from aragora.verification.formal import FormalVerificationManager

        manager = FormalVerificationManager()

        # Get debate data
        from aragora.server.storage import get_debates_db

        db = get_debates_db()
        if not db:
            return {"error": "Storage not available"}

        debate = db.get(debate_id)
        if not debate:
            return {"error": f"Debate {debate_id} not found"}

        # Extract consensus claim
        consensus = debate.get("final_answer", "")
        if not consensus:
            return {"error": "Debate has no consensus to verify"}

        # Verify
        result = await manager.attempt_formal_verification(
            claim=consensus,
            context=f"Debate consensus from {debate_id}",
        )

        return {
            "debate_id": debate_id,
            "status": (
                result.status.value if hasattr(result.status, "value") else str(result.status)
            ),
            "is_verified": result.is_verified,
            "language": result.language.value if hasattr(result.language, "value") else backend,
            "formal_statement": result.formal_statement,
            "proof_hash": result.proof_hash,
            "translation_time_ms": result.translation_time_ms,
            "proof_search_time_ms": result.proof_search_time_ms,
        }

    except ImportError:
        return {"error": "Verification module not available"}
    except Exception as e:
        return {"error": f"Verification failed: {e}"}


async def generate_proof_tool(
    claim: str,
    output_format: str = "lean4",
    context: str = "",
) -> Dict[str, Any]:
    """
    Generate a formal proof for a claim without verification.

    Args:
        claim: The claim to translate to formal language
        output_format: Output format (lean4, z3_smt)
        context: Additional context for translation

    Returns:
        Dict with formal statement and confidence
    """
    if not claim:
        return {"error": "claim is required"}

    try:
        from aragora.verification.formal import LeanBackend, Z3Backend

        backend: Union[LeanBackend, Z3Backend]
        if output_format == "lean4":
            backend = LeanBackend()
        else:
            backend = Z3Backend()

        formal_statement = await backend.translate(claim, context)

        return {
            "success": formal_statement is not None,
            "claim": claim,
            "formal_statement": formal_statement,
            "format": output_format,
            "confidence": 0.7 if formal_statement else 0.0,
        }

    except ImportError:
        return {"error": "Verification module not available"}
    except Exception as e:
        return {"error": f"Proof generation failed: {e}"}


# === Evidence Tools ===


async def search_evidence_tool(
    query: str,
    sources: str = "all",
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Search for evidence across configured sources.

    Args:
        query: Search query
        sources: Comma-separated sources (arxiv, hackernews, reddit, all)
        limit: Max results per source

    Returns:
        Dict with evidence results
    """
    if not query:
        return {"error": "query is required"}

    limit = min(max(limit, 1), 50)
    results: List[Dict[str, Any]] = []

    try:
        from aragora.evidence.collector import EvidenceCollector

        collector = EvidenceCollector()
        source_list = None if sources == "all" else [s.strip() for s in sources.split(",")]

        # Use collect_evidence which is the actual API
        evidence_pack = await collector.collect_evidence(
            task=query,
            enabled_connectors=source_list,
        )

        for e in evidence_pack.snippets[:limit]:
            results.append(
                {
                    "id": e.id,
                    "title": e.title,
                    "source": e.source,
                    "url": e.url,
                    "snippet": e.snippet[:300] if e.snippet else "",
                    "score": e.reliability_score,
                    "published": str(e.fetched_at) if e.fetched_at else None,
                }
            )

    except ImportError:
        logger.warning("Evidence collector not available")
    except Exception as e:
        logger.warning(f"Evidence search failed: {e}")

    return {
        "query": query,
        "sources": sources,
        "results": results,
        "count": len(results),
    }


async def cite_evidence_tool(
    debate_id: str,
    evidence_id: str,
    message_index: int,
    citation_text: str = "",
) -> Dict[str, Any]:
    """
    Add a citation to evidence in a debate message.

    Args:
        debate_id: ID of the debate
        evidence_id: ID of the evidence to cite
        message_index: Index of the message to add citation to
        citation_text: Optional citation text

    Returns:
        Dict with citation status
    """
    if not debate_id or not evidence_id:
        return {"error": "debate_id and evidence_id are required"}

    try:
        from aragora.server.storage import get_debates_db

        db = get_debates_db()
        if not db:
            return {"error": "Storage not available"}

        debate = db.get(debate_id)
        if not debate:
            return {"error": f"Debate {debate_id} not found"}

        # Add citation to debate metadata
        citations = debate.get("citations", [])
        citation = {
            "evidence_id": evidence_id,
            "message_index": message_index,
            "text": citation_text,
            "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        citations.append(citation)

        # Update debate
        if hasattr(db, "update"):
            db.update(debate_id, {"citations": citations})

        return {
            "success": True,
            "debate_id": debate_id,
            "evidence_id": evidence_id,
            "message_index": message_index,
            "citation_count": len(citations),
        }

    except Exception as e:
        return {"error": f"Failed to add citation: {e}"}


async def verify_citation_tool(
    url: str,
) -> Dict[str, Any]:
    """
    Verify that a citation URL is valid and accessible.

    Args:
        url: URL to verify

    Returns:
        Dict with verification status and metadata
    """
    if not url:
        return {"error": "url is required"}

    import aiohttp

    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                return {
                    "url": url,
                    "valid": response.status == 200,
                    "status_code": response.status,
                    "content_type": response.headers.get("Content-Type", "unknown"),
                    "accessible": response.status < 400,
                }
    except asyncio.TimeoutError:
        return {"url": url, "valid": False, "error": "Timeout"}
    except Exception as e:
        return {"url": url, "valid": False, "error": str(e)}


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
    # Memory tools
    {
        "name": "query_memory",
        "description": "Query memories from the continuum memory system",
        "function": query_memory_tool,
        "parameters": {
            "query": {"type": "string", "required": True},
            "tier": {"type": "string", "default": "all"},
            "limit": {"type": "integer", "default": 10},
            "min_importance": {"type": "number", "default": 0.0},
        },
    },
    {
        "name": "store_memory",
        "description": "Store a memory in the continuum memory system",
        "function": store_memory_tool,
        "parameters": {
            "content": {"type": "string", "required": True},
            "tier": {"type": "string", "default": "medium"},
            "importance": {"type": "number", "default": 0.5},
            "tags": {"type": "string", "default": ""},
        },
    },
    {
        "name": "get_memory_pressure",
        "description": "Get current memory pressure and utilization",
        "function": get_memory_pressure_tool,
        "parameters": {},
    },
    # Fork tools
    {
        "name": "fork_debate",
        "description": "Fork a debate to explore counterfactual scenarios",
        "function": fork_debate_tool,
        "parameters": {
            "debate_id": {"type": "string", "required": True},
            "branch_point": {"type": "integer", "default": -1},
            "modified_context": {"type": "string", "default": ""},
        },
    },
    {
        "name": "get_forks",
        "description": "Get all forks of a debate",
        "function": get_forks_tool,
        "parameters": {
            "debate_id": {"type": "string", "required": True},
            "include_nested": {"type": "boolean", "default": False},
        },
    },
    # Genesis tools
    {
        "name": "get_agent_lineage",
        "description": "Get the evolutionary lineage of an agent",
        "function": get_agent_lineage_tool,
        "parameters": {
            "agent_name": {"type": "string", "required": True},
            "depth": {"type": "integer", "default": 5},
        },
    },
    {
        "name": "breed_agents",
        "description": "Breed two agents to create a new offspring agent",
        "function": breed_agents_tool,
        "parameters": {
            "parent_a": {"type": "string", "required": True},
            "parent_b": {"type": "string", "required": True},
            "mutation_rate": {"type": "number", "default": 0.1},
        },
    },
    # Checkpoint tools
    {
        "name": "create_checkpoint",
        "description": "Create a checkpoint for a debate to enable resume later",
        "function": create_checkpoint_tool,
        "parameters": {
            "debate_id": {"type": "string", "required": True},
            "label": {"type": "string", "default": ""},
            "storage_backend": {"type": "string", "default": "file"},
        },
    },
    {
        "name": "list_checkpoints",
        "description": "List checkpoints for a debate or all debates",
        "function": list_checkpoints_tool,
        "parameters": {
            "debate_id": {"type": "string", "default": ""},
            "include_expired": {"type": "boolean", "default": False},
            "limit": {"type": "integer", "default": 20},
        },
    },
    {
        "name": "resume_checkpoint",
        "description": "Resume a debate from a checkpoint",
        "function": resume_checkpoint_tool,
        "parameters": {
            "checkpoint_id": {"type": "string", "required": True},
        },
    },
    {
        "name": "delete_checkpoint",
        "description": "Delete a checkpoint",
        "function": delete_checkpoint_tool,
        "parameters": {
            "checkpoint_id": {"type": "string", "required": True},
        },
    },
    # Verification tools
    {
        "name": "verify_consensus",
        "description": "Verify the consensus of a completed debate using formal methods",
        "function": verify_consensus_tool,
        "parameters": {
            "debate_id": {"type": "string", "required": True},
            "backend": {"type": "string", "default": "z3"},
        },
    },
    {
        "name": "generate_proof",
        "description": "Generate a formal proof for a claim without verification",
        "function": generate_proof_tool,
        "parameters": {
            "claim": {"type": "string", "required": True},
            "output_format": {"type": "string", "default": "lean4"},
            "context": {"type": "string", "default": ""},
        },
    },
    # Evidence tools
    {
        "name": "search_evidence",
        "description": "Search for evidence across configured sources",
        "function": search_evidence_tool,
        "parameters": {
            "query": {"type": "string", "required": True},
            "sources": {"type": "string", "default": "all"},
            "limit": {"type": "integer", "default": 10},
        },
    },
    {
        "name": "cite_evidence",
        "description": "Add a citation to evidence in a debate message",
        "function": cite_evidence_tool,
        "parameters": {
            "debate_id": {"type": "string", "required": True},
            "evidence_id": {"type": "string", "required": True},
            "message_index": {"type": "integer", "required": True},
            "citation_text": {"type": "string", "default": ""},
        },
    },
    {
        "name": "verify_citation",
        "description": "Verify that a citation URL is valid and accessible",
        "function": verify_citation_tool,
        "parameters": {
            "url": {"type": "string", "required": True},
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
    # Memory tools
    "query_memory_tool",
    "store_memory_tool",
    "get_memory_pressure_tool",
    # Fork tools
    "fork_debate_tool",
    "get_forks_tool",
    # Genesis tools
    "get_agent_lineage_tool",
    "breed_agents_tool",
    # Checkpoint tools
    "create_checkpoint_tool",
    "list_checkpoints_tool",
    "resume_checkpoint_tool",
    "delete_checkpoint_tool",
    # Verification tools
    "verify_consensus_tool",
    "generate_proof_tool",
    # Evidence tools
    "search_evidence_tool",
    "cite_evidence_tool",
    "verify_citation_tool",
    "TOOLS_METADATA",
]
