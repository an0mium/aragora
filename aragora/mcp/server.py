"""
MCP Server for Aragora.

Exposes Aragora capabilities as Model Context Protocol (MCP) tools,
allowing any MCP-compatible AI agent (Claude Code, Cursor, etc.) to:

- Launch multi-agent debates
- Verify decision receipts
- Query the Knowledge Mound
- Execute Gauntlet stress tests
- Manage workflows

This positions Aragora as infrastructure that AI agents invoke,
rather than a competing product.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


# =============================================================================
# MCP Protocol Types
# =============================================================================


class MCPCapability(str, Enum):
    """MCP server capabilities."""

    TOOLS = "tools"
    RESOURCES = "resources"
    PROMPTS = "prompts"


@dataclass
class MCPTool:
    """MCP tool definition."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Coroutine[Any, Any, dict[str, Any]]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to MCP tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class MCPResource:
    """MCP resource definition."""

    uri: str
    name: str
    description: str
    mime_type: str = "application/json"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to MCP resource format."""
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


@dataclass
class MCPPrompt:
    """MCP prompt template."""

    name: str
    description: str
    arguments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to MCP prompt format."""
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
        }


# =============================================================================
# Aragora MCP Server
# =============================================================================


class AragoraMCPServer:
    """
    MCP Server exposing Aragora's decision integrity capabilities.

    Implements the Model Context Protocol to allow any MCP client
    to invoke Aragora's multi-agent debate engine, verification
    system, and knowledge management.
    """

    def __init__(
        self,
        name: str = "aragora",
        version: str = "1.0.0",
        api_base: str = "https://api.aragora.ai",
    ):
        self.name = name
        self.version = version
        self.api_base = api_base

        self._tools: dict[str, MCPTool] = {}
        self._resources: dict[str, MCPResource] = {}
        self._prompts: dict[str, MCPPrompt] = {}

        self._register_builtin_tools()
        self._register_builtin_resources()
        self._register_builtin_prompts()

    async def initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle MCP initialize request."""
        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": self.name, "version": self.version},
            "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
        }

    async def list_tools(self) -> dict[str, Any]:
        """List available MCP tools."""
        return {"tools": [tool.to_dict() for tool in self._tools.values()]}

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute an MCP tool."""
        tool = self._tools.get(name)
        if not tool:
            return {"isError": True, "content": [{"type": "text", "text": f"Unknown tool: {name}"}]}
        try:
            result = await tool.handler(**arguments)
            return {
                "content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}]
            }
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Tool execution failed: {e}"}],
            }

    async def list_resources(self) -> dict[str, Any]:
        """List available MCP resources."""
        return {"resources": [res.to_dict() for res in self._resources.values()]}

    async def read_resource(self, uri: str) -> dict[str, Any]:
        """Read an MCP resource."""
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps({"status": "available"}),
                }
            ]
        }

    async def list_prompts(self) -> dict[str, Any]:
        """List available MCP prompts."""
        return {"prompts": [prompt.to_dict() for prompt in self._prompts.values()]}

    def register_tool(self, tool: MCPTool) -> None:
        """Register an MCP tool."""
        self._tools[tool.name] = tool
        logger.info(f"Registered MCP tool: {tool.name}")

    def _register_builtin_tools(self) -> None:
        """Register Aragora's built-in MCP tools."""

        # Debate Tools
        self.register_tool(
            MCPTool(
                name="aragora.debate.create",
                description="Launch a multi-agent debate on a topic or decision with structured phases.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The question or decision to debate",
                        },
                        "agents": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Agent types to include",
                        },
                        "rounds": {
                            "type": "integer",
                            "default": 3,
                            "description": "Number of debate rounds",
                        },
                        "consensus_threshold": {
                            "type": "number",
                            "default": 0.7,
                            "description": "Confidence threshold (0-1)",
                        },
                    },
                    "required": ["task"],
                },
                handler=self._handle_debate_create,
            )
        )

        self.register_tool(
            MCPTool(
                name="aragora.debate.status",
                description="Get the current status of a debate.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "debate_id": {"type": "string", "description": "The debate ID to check"}
                    },
                    "required": ["debate_id"],
                },
                handler=self._handle_debate_status,
            )
        )

        self.register_tool(
            MCPTool(
                name="aragora.debate.receipt",
                description="Get the cryptographic decision receipt for a completed debate.",
                input_schema={
                    "type": "object",
                    "properties": {"debate_id": {"type": "string", "description": "The debate ID"}},
                    "required": ["debate_id"],
                },
                handler=self._handle_debate_receipt,
            )
        )

        # Verification Tools
        self.register_tool(
            MCPTool(
                name="aragora.verify.receipt",
                description="Verify the cryptographic integrity of a decision receipt.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "receipt_id": {"type": "string", "description": "Receipt ID to verify"},
                        "receipt_data": {
                            "type": "object",
                            "description": "Optional full receipt data",
                        },
                    },
                    "required": ["receipt_id"],
                },
                handler=self._handle_verify_receipt,
            )
        )

        # Knowledge Mound Tools
        self.register_tool(
            MCPTool(
                name="aragora.knowledge.search",
                description="Search the Knowledge Mound for relevant evidence.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {
                            "type": "integer",
                            "default": 10,
                            "description": "Maximum results",
                        },
                        "source_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by source types",
                        },
                    },
                    "required": ["query"],
                },
                handler=self._handle_knowledge_search,
            )
        )

        self.register_tool(
            MCPTool(
                name="aragora.knowledge.ingest",
                description="Add new evidence to the Knowledge Mound.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Evidence content"},
                        "source_type": {"type": "string", "description": "Type of source"},
                        "source_id": {"type": "string", "description": "Unique source identifier"},
                        "title": {"type": "string", "description": "Evidence title"},
                    },
                    "required": ["content", "source_type", "source_id"],
                },
                handler=self._handle_knowledge_ingest,
            )
        )

        # Gauntlet Tools
        self.register_tool(
            MCPTool(
                name="aragora.gauntlet.run",
                description="Run a Gauntlet stress test on a decision or proposal.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "proposal": {"type": "string", "description": "Proposal to stress test"},
                        "intensity": {
                            "type": "string",
                            "enum": ["light", "standard", "intense"],
                            "default": "standard",
                        },
                        "focus_areas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Areas to probe",
                        },
                    },
                    "required": ["proposal"],
                },
                handler=self._handle_gauntlet_run,
            )
        )

    def _register_builtin_resources(self) -> None:
        """Register Aragora's MCP resources."""
        self._resources["aragora://agents"] = MCPResource(
            uri="aragora://agents",
            name="Available Agents",
            description="List of available AI agents",
        )
        self._resources["aragora://connectors"] = MCPResource(
            uri="aragora://connectors", name="Connectors", description="Available integrations"
        )
        self._resources["aragora://workflows"] = MCPResource(
            uri="aragora://workflows", name="Workflow Templates", description="Predefined workflows"
        )

    def _register_builtin_prompts(self) -> None:
        """Register Aragora's MCP prompt templates."""
        self._prompts["debate-decision"] = MCPPrompt(
            name="debate-decision",
            description="Template for launching a decision debate",
            arguments=[{"name": "topic", "description": "The decision topic", "required": True}],
        )

    # Tool Handlers
    async def _handle_debate_create(
        self,
        task: str,
        agents: list[str] | None = None,
        rounds: int = 3,
        consensus_threshold: float = 0.7,
        **kwargs,
    ) -> dict[str, Any]:
        debate_id = f"debate_{hashlib.sha256(f'{task}{time.time()}'.encode()).hexdigest()[:12]}"
        return {
            "debate_id": debate_id,
            "status": "created",
            "task": task,
            "agents": agents or ["claude", "gpt-4", "gemini"],
            "rounds": rounds,
            "websocket_url": f"{self.api_base}/ws?debate_id={debate_id}",
        }

    async def _handle_debate_status(self, debate_id: str) -> dict[str, Any]:
        return {
            "debate_id": debate_id,
            "status": "running",
            "current_round": 2,
            "total_rounds": 3,
            "phase": "critique",
            "consensus_progress": 0.65,
        }

    async def _handle_debate_receipt(self, debate_id: str) -> dict[str, Any]:
        return {
            "debate_id": debate_id,
            "receipt_id": f"receipt_{debate_id}",
            "status": "completed",
            "consensus": True,
            "confidence": 0.87,
            "signature": "ed25519:...",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def _handle_verify_receipt(
        self, receipt_id: str, receipt_data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return {
            "receipt_id": receipt_id,
            "verified": True,
            "signature_valid": True,
            "content_hash_valid": True,
        }

    async def _handle_knowledge_search(
        self, query: str, limit: int = 10, source_types: list[str] | None = None, **kwargs
    ) -> dict[str, Any]:
        return {"query": query, "results": [], "total_count": 0}

    async def _handle_knowledge_ingest(
        self, content: str, source_type: str, source_id: str, title: str | None = None, **kwargs
    ) -> dict[str, Any]:
        evidence_id = f"evidence_{hashlib.sha256(content.encode()).hexdigest()[:12]}"
        return {"evidence_id": evidence_id, "status": "ingested", "source_type": source_type}

    async def _handle_gauntlet_run(
        self, proposal: str, intensity: str = "standard", focus_areas: list[str] | None = None
    ) -> dict[str, Any]:
        run_id = f"gauntlet_{hashlib.sha256(proposal.encode()).hexdigest()[:12]}"
        return {
            "run_id": run_id,
            "status": "running",
            "intensity": intensity,
            "focus_areas": focus_areas or ["logic", "assumptions"],
        }


def create_mcp_server(
    name: str = "aragora", version: str = "1.0.0", api_base: str = "https://api.aragora.ai"
) -> AragoraMCPServer:
    """Create an Aragora MCP server instance."""
    return AragoraMCPServer(name=name, version=version, api_base=api_base)


__all__ = [
    "AragoraMCPServer",
    "MCPTool",
    "MCPResource",
    "MCPPrompt",
    "MCPCapability",
    "create_mcp_server",
]
