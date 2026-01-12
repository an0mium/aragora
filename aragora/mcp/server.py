"""
Aragora MCP Server Implementation.

Implements the MCP 1.0 protocol to expose Aragora capabilities.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check if mcp package is available
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        Resource,
        ResourceTemplate,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None  # type: ignore
    stdio_server = None  # type: ignore


class AragoraMCPServer:
    """
    MCP Server for Aragora.

    Exposes the following tools:
    - run_debate: Run a decision stress-test (debate engine) on a topic
    - run_gauntlet: Stress-test a document/spec
    - list_agents: List available agents
    - get_debate: Get results of a past debate
    - search_debates: Search debates by topic, date, agents
    - get_agent_history: Get agent debate history and stats
    - get_consensus_proofs: Retrieve formal proofs from debates
    - list_trending_topics: Get trending topics from Pulse

    And the following resources:
    - debate://{id}: Access debate results
    - agent://{name}/stats: Access agent statistics
    - trending://topics: Access trending topics
    """

    def __init__(self):
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP package not installed. Install with: pip install mcp"
            )

        self.server = Server("aragora")
        self._setup_handlers()
        self._debates_cache: Dict[str, Dict[str, Any]] = {}
        self._agents_cache: Dict[str, Dict[str, Any]] = {}

    def _setup_handlers(self) -> None:
        """Set up MCP request handlers."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="run_debate",
                    description=(
                        "Run a decision stress-test (debate engine). "
                        "Multiple AI agents will critique and converge on a hardened answer."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question or topic to stress-test",
                            },
                            "agents": {
                                "type": "string",
                                "description": "Comma-separated agent IDs (e.g., 'anthropic-api,openai-api,gemini'). Default: anthropic-api,openai-api",
                                "default": "anthropic-api,openai-api",
                            },
                            "rounds": {
                                "type": "integer",
                                "description": "Number of debate rounds (1-10). Default: 3",
                                "default": 3,
                                "minimum": 1,
                                "maximum": 10,
                            },
                            "consensus": {
                                "type": "string",
                                "enum": ["majority", "unanimous", "none"],
                                "description": "Consensus mechanism. Default: majority",
                                "default": "majority",
                            },
                        },
                        "required": ["question"],
                    },
                ),
                Tool(
                    name="run_gauntlet",
                    description="Stress-test a document, specification, or code through adversarial multi-agent analysis. Identifies vulnerabilities and risks.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The content to stress-test (spec, code, policy, etc.)",
                            },
                            "content_type": {
                                "type": "string",
                                "enum": ["spec", "code", "policy", "architecture"],
                                "description": "Type of content being tested. Default: spec",
                                "default": "spec",
                            },
                            "profile": {
                                "type": "string",
                                "enum": ["quick", "thorough", "code", "security", "gdpr", "hipaa"],
                                "description": "Test profile. Default: quick",
                                "default": "quick",
                            },
                        },
                        "required": ["content"],
                    },
                ),
                Tool(
                    name="list_agents",
                    description="List available AI agents that can participate in debates.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    },
                ),
                Tool(
                    name="get_debate",
                    description="Get the results of a previous debate by ID.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "debate_id": {
                                "type": "string",
                                "description": "The debate ID to retrieve",
                            },
                        },
                        "required": ["debate_id"],
                    },
                ),
                Tool(
                    name="search_debates",
                    description="Search debates by topic, date range, or participating agents.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for topic text",
                            },
                            "agent": {
                                "type": "string",
                                "description": "Filter by agent name",
                            },
                            "start_date": {
                                "type": "string",
                                "description": "Start date (YYYY-MM-DD)",
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date (YYYY-MM-DD)",
                            },
                            "consensus_only": {
                                "type": "boolean",
                                "description": "Only return debates that reached consensus",
                                "default": False,
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max results (1-100). Default: 20",
                                "default": 20,
                                "minimum": 1,
                                "maximum": 100,
                            },
                        },
                    },
                ),
                Tool(
                    name="get_agent_history",
                    description="Get an agent's debate history, ELO rating, and performance stats.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "agent_name": {
                                "type": "string",
                                "description": "The agent name (e.g., 'anthropic-api', 'openai-api')",
                            },
                            "include_debates": {
                                "type": "boolean",
                                "description": "Include recent debate summaries",
                                "default": True,
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max debates to include. Default: 10",
                                "default": 10,
                            },
                        },
                        "required": ["agent_name"],
                    },
                ),
                Tool(
                    name="get_consensus_proofs",
                    description="Retrieve formal verification proofs from debates.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "debate_id": {
                                "type": "string",
                                "description": "Specific debate ID to get proofs for",
                            },
                            "proof_type": {
                                "type": "string",
                                "enum": ["z3", "lean", "all"],
                                "description": "Type of proofs to retrieve. Default: all",
                                "default": "all",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max proofs to return. Default: 10",
                                "default": 10,
                            },
                        },
                    },
                ),
                Tool(
                    name="list_trending_topics",
                    description="Get trending topics from Pulse that could make good debates.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "platform": {
                                "type": "string",
                                "enum": ["hackernews", "reddit", "arxiv", "all"],
                                "description": "Source platform. Default: all",
                                "default": "all",
                            },
                            "category": {
                                "type": "string",
                                "description": "Topic category filter (e.g., 'tech', 'science', 'ai')",
                            },
                            "min_score": {
                                "type": "number",
                                "description": "Minimum topic score (0-1). Default: 0.5",
                                "default": 0.5,
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max topics to return. Default: 10",
                                "default": 10,
                            },
                        },
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            try:
                if name == "run_debate":
                    result = await self._run_debate(arguments)
                elif name == "run_gauntlet":
                    result = await self._run_gauntlet(arguments)
                elif name == "list_agents":
                    result = await self._list_agents()
                elif name == "get_debate":
                    result = await self._get_debate(arguments)
                elif name == "search_debates":
                    result = await self._search_debates(arguments)
                elif name == "get_agent_history":
                    result = await self._get_agent_history(arguments)
                elif name == "get_consensus_proofs":
                    result = await self._get_consensus_proofs(arguments)
                elif name == "list_trending_topics":
                    result = await self._list_trending_topics(arguments)
                else:
                    result = {"error": f"Unknown tool: {name}"}

                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2),
                )]
            except Exception as e:
                logger.error(f"Tool {name} failed: {e}", exc_info=True)
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)}),
                )]

        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available resources (cached debates)."""
            resources = []
            for debate_id, debate_data in self._debates_cache.items():
                resources.append(Resource(
                    uri=f"debate://{debate_id}",
                    name=f"Debate: {debate_data.get('task', 'Unknown')[:50]}",
                    description=f"Debate result from {debate_data.get('timestamp', 'unknown time')}",
                    mimeType="application/json",
                ))
            return resources

        @self.server.list_resource_templates()
        async def list_resource_templates() -> List[ResourceTemplate]:
            """List resource templates."""
            return [
                ResourceTemplate(
                    uriTemplate="debate://{debate_id}",
                    name="Debate Result",
                    description="Access a debate result by ID",
                    mimeType="application/json",
                ),
                ResourceTemplate(
                    uriTemplate="agent://{agent_name}/stats",
                    name="Agent Statistics",
                    description="Access agent ELO rating and performance statistics",
                    mimeType="application/json",
                ),
                ResourceTemplate(
                    uriTemplate="trending://topics",
                    name="Trending Topics",
                    description="Access current trending topics from Pulse",
                    mimeType="application/json",
                ),
            ]

        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read a resource by URI."""
            if uri.startswith("debate://"):
                debate_id = uri.replace("debate://", "")
                debate_data = self._debates_cache.get(debate_id)
                if debate_data:
                    return json.dumps(debate_data, indent=2)
                # Try storage
                result = await self._get_debate({"debate_id": debate_id})
                return json.dumps(result, indent=2)

            elif uri.startswith("agent://") and uri.endswith("/stats"):
                # Extract agent name from agent://{name}/stats
                agent_name = uri.replace("agent://", "").replace("/stats", "")
                result = await self._get_agent_history({"agent_name": agent_name})
                return json.dumps(result, indent=2)

            elif uri == "trending://topics":
                result = await self._list_trending_topics({})
                return json.dumps(result, indent=2)

            return json.dumps({"error": f"Unknown resource: {uri}"})

    async def _run_debate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run a debate and return results."""
        from aragora.agents.base import create_agent
        from aragora.debate.orchestrator import Arena, DebateProtocol
        from aragora.core import Environment
        import time
        import uuid

        question = args.get("question", "")
        agents_str = args.get("agents", "anthropic-api,openai-api")
        rounds = min(max(args.get("rounds", 3), 1), 10)
        consensus = args.get("consensus", "majority")

        if not question:
            return {"error": "Question is required"}

        # Parse and create agents
        agent_names = [a.strip() for a in agents_str.split(",")]
        agents = []
        roles = ["proposer", "critic", "synthesizer"]

        for i, agent_name in enumerate(agent_names):
            role = roles[i] if i < len(roles) else "critic"
            try:
                agent = create_agent(
                    model_type=agent_name,
                    name=f"{agent_name}_{role}",
                    role=role,
                )
                agents.append(agent)
            except Exception as e:
                logger.warning(f"Could not create agent {agent_name}: {e}")

        if not agents:
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
        arena = Arena(env, agents, protocol)
        result = await arena.run()

        # Generate debate ID
        debate_id = f"mcp_{uuid.uuid4().hex[:8]}"

        # Cache result
        debate_data = {
            "debate_id": debate_id,
            "task": question,
            "final_answer": result.final_answer,
            "consensus_reached": result.consensus_reached,
            "confidence": result.confidence,
            "rounds_used": result.rounds_used,
            "agents": [a.name for a in agents],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._debates_cache[debate_id] = debate_data

        return debate_data

    async def _run_gauntlet(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run gauntlet stress-test."""
        from aragora.gauntlet import (
            GauntletRunner,
            GauntletConfig,
            AttackCategory,
            QUICK_GAUNTLET,
            THOROUGH_GAUNTLET,
            CODE_REVIEW_GAUNTLET,
            SECURITY_GAUNTLET,
            GDPR_GAUNTLET,
            HIPAA_GAUNTLET,
        )

        content = args.get("content", "")
        content_type = args.get("content_type", "spec")
        profile = args.get("profile", "quick")

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

        config = profiles.get(profile, QUICK_GAUNTLET)

        # Update config with content
        config = GauntletConfig(
            attack_categories=config.attack_categories,
            agents=config.agents,
            rounds_per_attack=config.rounds_per_attack,
        )

        runner = GauntletRunner(config)
        result = await runner.run(content)

        return {
            "verdict": result.verdict.value if hasattr(result, 'verdict') else "unknown",
            "risk_score": getattr(result, 'risk_score', 0),
            "vulnerabilities_count": len(getattr(result, 'vulnerabilities', [])),
            "vulnerabilities": [
                {
                    "category": v.category,
                    "severity": v.severity,
                    "description": v.description,
                }
                for v in getattr(result, 'vulnerabilities', [])[:5]  # Limit to 5
            ],
            "summary": result.summary() if hasattr(result, 'summary') else str(result),
        }

    async def _list_agents(self) -> Dict[str, Any]:
        """List available agents."""
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

    async def _get_debate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get a previous debate by ID."""
        debate_id = args.get("debate_id", "")

        if not debate_id:
            return {"error": "debate_id is required"}

        # Check cache
        if debate_id in self._debates_cache:
            return self._debates_cache[debate_id]

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

    async def _search_debates(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search debates by topic, date, or agents."""
        from datetime import datetime

        query = args.get("query", "")
        agent_filter = args.get("agent", "")
        start_date = args.get("start_date", "")
        end_date = args.get("end_date", "")
        consensus_only = args.get("consensus_only", False)
        limit = min(max(args.get("limit", 20), 1), 100)

        results = []

        # Search cache first
        for debate_id, debate_data in self._debates_cache.items():
            # Apply filters
            if query and query.lower() not in debate_data.get("task", "").lower():
                continue
            if agent_filter:
                agents = debate_data.get("agents", [])
                if not any(agent_filter.lower() in a.lower() for a in agents):
                    continue
            if consensus_only and not debate_data.get("consensus_reached", False):
                continue
            if start_date:
                debate_ts = debate_data.get("timestamp", "")
                if debate_ts < start_date:
                    continue
            if end_date:
                debate_ts = debate_data.get("timestamp", "")
                if debate_ts > end_date:
                    continue

            results.append({
                "debate_id": debate_id,
                "task": debate_data.get("task", "")[:100],
                "consensus_reached": debate_data.get("consensus_reached", False),
                "confidence": debate_data.get("confidence", 0),
                "timestamp": debate_data.get("timestamp", ""),
            })

        # Try storage for more results
        try:
            from aragora.server.storage import get_debates_db
            db = get_debates_db()
            if db and hasattr(db, "search"):
                stored = db.search(
                    query=query,
                    agent=agent_filter,
                    consensus_only=consensus_only,
                    limit=limit,
                )
                for debate in stored:
                    if debate.get("debate_id") not in [r["debate_id"] for r in results]:
                        results.append(debate)
        except Exception as e:
            logger.debug(f"Storage search unavailable: {e}")

        # Sort by timestamp (newest first) and limit
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        results = results[:limit]

        return {
            "debates": results,
            "count": len(results),
            "query": query or "(all)",
            "filters": {
                "agent": agent_filter or None,
                "consensus_only": consensus_only,
                "date_range": f"{start_date or '*'} to {end_date or '*'}",
            },
        }

    async def _get_agent_history(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get agent debate history and stats."""
        agent_name = args.get("agent_name", "")
        include_debates = args.get("include_debates", True)
        limit = args.get("limit", 10)

        if not agent_name:
            return {"error": "agent_name is required"}

        result: Dict[str, Any] = {
            "agent_name": agent_name,
            "elo_rating": 1500,  # Default
            "total_debates": 0,
            "consensus_rate": 0.0,
            "win_rate": 0.0,
        }

        # Try to get ELO rating
        try:
            from aragora.ranking.elo import ELOSystem
            elo = ELOSystem()
            rating = elo.get_rating(agent_name)
            if rating:
                result["elo_rating"] = rating.rating
                result["elo_deviation"] = rating.deviation
        except Exception as e:
            logger.debug(f"Could not get ELO: {e}")

        # Try to get performance stats
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
        except Exception as e:
            logger.debug(f"Could not get agent stats: {e}")

        # Get recent debates from cache
        if include_debates:
            agent_debates = []
            for debate_id, debate_data in self._debates_cache.items():
                agents = debate_data.get("agents", [])
                if any(agent_name.lower() in a.lower() for a in agents):
                    agent_debates.append({
                        "debate_id": debate_id,
                        "task": debate_data.get("task", "")[:80],
                        "consensus_reached": debate_data.get("consensus_reached", False),
                        "timestamp": debate_data.get("timestamp", ""),
                    })

            agent_debates.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            result["recent_debates"] = agent_debates[:limit]
            result["total_debates"] = max(result["total_debates"], len(agent_debates))

        return result

    async def _get_consensus_proofs(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get formal verification proofs from debates."""
        debate_id = args.get("debate_id", "")
        proof_type = args.get("proof_type", "all")
        limit = args.get("limit", 10)

        proofs: List[Dict[str, Any]] = []

        # If specific debate requested, get proofs from that debate
        if debate_id:
            debate_data = self._debates_cache.get(debate_id)
            if not debate_data:
                # Try storage
                try:
                    from aragora.server.storage import get_debates_db
                    db = get_debates_db()
                    if db:
                        debate_data = db.get(debate_id)
                except Exception as e:
                    logger.debug(f"Failed to fetch debate {debate_id} from storage: {e}")

            if debate_data and "proofs" in debate_data:
                for proof in debate_data["proofs"]:
                    if proof_type == "all" or proof.get("type") == proof_type:
                        proofs.append(proof)
        else:
            # Get proofs from all cached debates
            for did, ddata in self._debates_cache.items():
                if "proofs" in ddata:
                    for proof in ddata["proofs"]:
                        if proof_type == "all" or proof.get("type") == proof_type:
                            proof_entry = {**proof, "debate_id": did}
                            proofs.append(proof_entry)

        # Try to get from verification storage
        try:
            from aragora.server.storage import get_proofs_db
            proofs_db = get_proofs_db()
            if proofs_db:
                stored_proofs = proofs_db.list(
                    debate_id=debate_id or None,
                    proof_type=proof_type if proof_type != "all" else None,
                    limit=limit,
                )
                for p in stored_proofs:
                    if p not in proofs:
                        proofs.append(p)
        except Exception as e:
            logger.debug(f"Proofs storage unavailable: {e}")

        # Limit results
        proofs = proofs[:limit]

        return {
            "proofs": proofs,
            "count": len(proofs),
            "debate_id": debate_id or "(all debates)",
            "proof_type": proof_type,
        }

    async def _list_trending_topics(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get trending topics from Pulse."""
        platform = args.get("platform", "all")
        category = args.get("category", "")
        min_score = args.get("min_score", 0.5)
        limit = args.get("limit", 10)

        topics: List[Dict[str, Any]] = []

        try:
            from aragora.pulse import get_trending_topics, TrendingTopic
            from aragora.pulse.scheduler import TopicSelector

            # Get trending topics
            raw_topics = await get_trending_topics(
                platforms=[platform] if platform != "all" else None,
                limit=limit * 2,  # Get more, then filter
            )

            # Score and filter topics
            selector = TopicSelector()

            for topic in raw_topics:
                # Apply platform filter
                if platform != "all" and topic.platform != platform:
                    continue

                # Apply category filter
                if category and topic.category.lower() != category.lower():
                    continue

                # Score the topic
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

            # Sort by score and limit
            topics.sort(key=lambda x: x["score"], reverse=True)
            topics = topics[:limit]

        except ImportError:
            logger.warning("Pulse module not available")
            topics = []
        except Exception as e:
            logger.warning(f"Could not fetch trending topics: {e}")
            topics = []

        return {
            "topics": topics,
            "count": len(topics),
            "platform": platform,
            "category": category or "(all)",
            "min_score": min_score,
        }

    async def run(self) -> None:
        """Run the MCP server."""
        logger.info("Starting Aragora MCP server...")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


async def run_server() -> None:
    """Run the Aragora MCP server."""
    if not MCP_AVAILABLE:
        print("Error: MCP package not installed.", file=sys.stderr)
        print("Install with: pip install mcp", file=sys.stderr)
        sys.exit(1)

    server = AragoraMCPServer()
    await server.run()


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Server stopped")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
