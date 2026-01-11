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
    - run_debate: Run a multi-agent debate on a topic
    - run_gauntlet: Stress-test a document/spec
    - list_agents: List available agents
    - get_debate: Get results of a past debate

    And the following resources:
    - debate://{id}: Access debate results
    """

    def __init__(self):
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP package not installed. Install with: pip install mcp"
            )

        self.server = Server("aragora")
        self._setup_handlers()
        self._debates_cache: Dict[str, Dict[str, Any]] = {}

    def _setup_handlers(self) -> None:
        """Set up MCP request handlers."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="run_debate",
                    description="Run a multi-agent AI debate on a topic. Multiple AI agents will discuss, critique, and converge on an answer.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question or topic to debate",
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
            ]

        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read a resource by URI."""
            if uri.startswith("debate://"):
                debate_id = uri.replace("debate://", "")
                debate_data = self._debates_cache.get(debate_id)
                if debate_data:
                    return json.dumps(debate_data, indent=2)
                return json.dumps({"error": f"Debate {debate_id} not found"})
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
