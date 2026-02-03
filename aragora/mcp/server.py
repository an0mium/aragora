"""
MCP Server for Aragora.

Exposes Aragora capabilities as Model Context Protocol (MCP) tools for
integration with MCP-compatible clients (Claude Desktop, Cursor, etc.).

This module provides two layers:
- AragoraMCPServer: lightweight, testable tool registry + request handlers
- run_server/main: runtime server using FastMCP when MCP is installed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Coroutine

from aragora.config import MAX_CONTENT_LENGTH, MAX_QUESTION_LENGTH
from aragora.mcp.tools import TOOLS_METADATA

logger = logging.getLogger(__name__)

# MCP optional dependency
try:
    from mcp.types import (
        Tool,
        TextContent,
        Resource,
        ResourceTemplate,
        ListToolsRequest,
        ListToolsResult,
        CallToolRequest,
        CallToolResult,
        ListResourcesRequest,
        ListResourcesResult,
        ListResourceTemplatesRequest,
        ListResourceTemplatesResult,
        ReadResourceRequest,
        ReadResourceResult,
    )

    MCP_AVAILABLE = True
except Exception:  # pragma: no cover - handled by MCP_AVAILABLE flag
    MCP_AVAILABLE = False
    Tool = TextContent = Resource = ResourceTemplate = None  # type: ignore[assignment]
    ListToolsRequest = ListToolsResult = None  # type: ignore[assignment]
    CallToolRequest = CallToolResult = None  # type: ignore[assignment]
    ListResourcesRequest = ListResourcesResult = None  # type: ignore[assignment]
    ListResourceTemplatesRequest = ListResourceTemplatesResult = None  # type: ignore[assignment]
    ReadResourceRequest = ReadResourceResult = None  # type: ignore[assignment]


MAX_QUERY_LENGTH = 1000


# =============================================================================
# Rate Limiting
# =============================================================================


class RateLimiter:
    """Simple per-tool rate limiter with a rolling window."""

    def __init__(self, limits: dict[str, int] | None = None, window_seconds: int = 60):
        self._limits = limits or {}
        self._window_seconds = window_seconds
        self._usage: dict[str, dict[str, float | int]] = {}

    def _get_bucket(self, tool: str) -> dict[str, float | int]:
        bucket = self._usage.get(tool)
        if bucket is None:
            bucket = {"count": 0, "window_start": time.time()}
            self._usage[tool] = bucket
        return bucket

    def check(self, tool: str) -> tuple[bool, str | None]:
        limit = self._limits.get(tool)
        if not limit:
            return True, None

        bucket = self._get_bucket(tool)
        now = time.time()
        window_start = float(bucket["window_start"])

        if now - window_start >= self._window_seconds:
            bucket["window_start"] = now
            bucket["count"] = 0

        if int(bucket["count"]) >= limit:
            retry_in = int(self._window_seconds - (now - float(bucket["window_start"])))
            return False, f"Rate limit exceeded for {tool}. Try again in {retry_in}s"

        bucket["count"] = int(bucket["count"]) + 1
        return True, None

    def get_remaining(self, tool: str) -> int | None:
        limit = self._limits.get(tool)
        if limit is None:
            return None
        bucket = self._get_bucket(tool)
        now = time.time()
        if now - float(bucket["window_start"]) >= self._window_seconds:
            bucket["window_start"] = now
            bucket["count"] = 0
        return max(0, limit - int(bucket["count"]))


# =============================================================================
# MCP Protocol Types
# =============================================================================


@dataclass
class MCPTool:
    """MCP tool definition."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Coroutine[Any, Any, dict[str, Any]]]

    def to_dict(self) -> dict[str, Any]:
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
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
        }


# =============================================================================
# Aragora MCP Server
# =============================================================================


class AragoraMCPServer:
    """Lightweight MCP server registry with optional MCP request handlers."""

    def __init__(
        self,
        name: str = "aragora",
        version: str = "1.0.0",
        tools_metadata: list[dict[str, Any]] | None = None,
        require_mcp: bool = True,
        rate_limits: dict[str, int] | None = None,
    ) -> None:
        if require_mcp and not MCP_AVAILABLE:
            raise ImportError("MCP package not installed")

        self.name = name
        self.version = version
        self._tools_metadata = tools_metadata or TOOLS_METADATA
        self._tools: dict[str, MCPTool] = {}
        self._resources: dict[str, MCPResource] = {}
        self._prompts: dict[str, MCPPrompt] = {}
        self._rate_limiter = RateLimiter(rate_limits)
        self._debates_cache: dict[str, dict[str, Any]] = {}

        self._register_builtin_tools()
        self._register_builtin_resources()
        self._register_builtin_prompts()

        # MCP request handlers shim (used in tests when MCP is installed)
        self.server = self._build_mcp_server_shim() if MCP_AVAILABLE else None

    # ---------------------------------------------------------------------
    # Registry helpers
    # ---------------------------------------------------------------------

    def register_tool(self, tool: MCPTool) -> None:
        self._tools[tool.name] = tool
        logger.debug("Registered MCP tool: %s", tool.name)

    def register_resource(self, resource: MCPResource) -> None:
        self._resources[resource.uri] = resource

    def register_prompt(self, prompt: MCPPrompt) -> None:
        self._prompts[prompt.name] = prompt

    def _register_builtin_tools(self) -> None:
        overrides = {
            "run_debate": self._run_debate,
            "get_debate": self._get_debate,
            "search_debates": self._search_debates,
            "run_gauntlet": self._run_gauntlet,
            "list_agents": self._list_agents,
        }

        for meta in self._tools_metadata:
            name = meta.get("name")
            if not name:
                continue
            description = meta.get("description", "")
            input_schema = _build_input_schema(meta.get("parameters", {}))
            handler = overrides.get(name) or meta.get("function")
            if handler is None:
                continue
            self.register_tool(
                MCPTool(
                    name=name,
                    description=description,
                    input_schema=input_schema,
                    handler=handler,
                )
            )

    def _register_builtin_resources(self) -> None:
        # Resources are generated dynamically from cached debates.
        pass

    def _register_builtin_prompts(self) -> None:
        self.register_prompt(
            MCPPrompt(
                name="debate-decision",
                description="Template for launching a decision debate",
                arguments=[
                    {"name": "topic", "description": "The decision topic", "required": True}
                ],
            )
        )

    # ---------------------------------------------------------------------
    # Input validation & sanitization
    # ---------------------------------------------------------------------

    def _sanitize_arguments(self, args: dict[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str):
                sanitized[key] = value.strip()
            else:
                sanitized[key] = value
        return sanitized

    def _validate_input(self, tool_name: str, args: dict[str, Any]) -> str | None:
        if tool_name == "run_debate":
            question = args.get("question")
            if not question:
                return "Question is required"
            if len(question) > MAX_QUESTION_LENGTH:
                return f"Question exceeds maximum length ({MAX_QUESTION_LENGTH})"
            rounds = args.get("rounds")
            if rounds is not None and not isinstance(rounds, int):
                return "Rounds must be an integer"
        elif tool_name == "run_gauntlet":
            content = args.get("content")
            if not content:
                return "Content is required"
            if len(content) > MAX_CONTENT_LENGTH:
                return f"Content exceeds maximum length ({MAX_CONTENT_LENGTH})"
        elif tool_name == "search_debates":
            query = args.get("query", "")
            if query and len(query) > MAX_QUERY_LENGTH:
                return f"Query exceeds maximum length ({MAX_QUERY_LENGTH})"
        return None

    # ---------------------------------------------------------------------
    # Tool execution entry points
    # ---------------------------------------------------------------------

    async def invoke_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}"}

        args = self._sanitize_arguments(arguments or {})
        error = self._validate_input(name, args)
        if error:
            return {"error": error}

        allowed, rate_error = self._rate_limiter.check(name)
        if not allowed:
            return {"error": rate_error or "Rate limit exceeded"}

        try:
            result = tool.handler(args) if _expects_dict(tool.handler) else tool.handler(**args)
            if isinstance(result, Awaitable):
                result = await result
            return result if isinstance(result, dict) else {"result": result}
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return {"error": f"Tool execution failed: {e}"}

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self.invoke_tool(name, arguments)
        is_error = bool(result.get("error"))
        return {
            "isError": is_error,
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, indent=2, default=str),
                }
            ],
        }

    async def initialize(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        _ = params or {}
        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": self.name, "version": self.version},
            "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
        }

    async def list_tools(self) -> dict[str, Any]:
        return {"tools": [tool.to_dict() for tool in self._tools.values()]}

    async def list_resources(self) -> dict[str, Any]:
        resources = []
        for debate_id, debate in self._debates_cache.items():
            task = debate.get("task", "")
            name = f"Debate {debate_id}: {task[:60]}" if task else f"Debate {debate_id}"
            resources.append(
                MCPResource(
                    uri=f"debate://{debate_id}",
                    name=name,
                    description="Cached debate result",
                ).to_dict()
            )
        return {"resources": resources}

    async def read_resource(self, uri: str) -> dict[str, Any]:
        if uri.startswith("debate://"):
            debate_id = uri.replace("debate://", "")
            debate = self._debates_cache.get(debate_id)
            if debate:
                return {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(debate, indent=2, default=str),
                        }
                    ]
                }
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps({"error": f"Debate {debate_id} not found"}),
                    }
                ]
            }

        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps({"error": f"Unknown resource: {uri}"}),
                }
            ]
        }

    async def list_prompts(self) -> dict[str, Any]:
        return {"prompts": [prompt.to_dict() for prompt in self._prompts.values()]}

    # ---------------------------------------------------------------------
    # Internal tool handlers (used for caching/validation)
    # ---------------------------------------------------------------------

    async def _run_debate(
        self, args: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        from aragora.mcp.tools_module.debate import run_debate_tool

        payload = args or kwargs
        if not payload.get("question"):
            return {"error": "Question is required"}

        result = await run_debate_tool(**payload)
        if "error" not in result and result.get("debate_id"):
            self._debates_cache[result["debate_id"]] = result
        return result

    async def _get_debate(
        self, args: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        from aragora.mcp.tools_module.debate import get_debate_tool

        payload = args or kwargs
        debate_id = payload.get("debate_id") if isinstance(payload, dict) else None
        if not debate_id:
            return {"error": "debate_id is required"}

        if debate_id in self._debates_cache:
            return self._debates_cache[debate_id]

        result = await get_debate_tool(debate_id=debate_id)
        return result

    async def _search_debates(
        self, args: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        from aragora.mcp.tools_module.debate import search_debates_tool

        payload = args or kwargs
        return await search_debates_tool(**payload)

    async def _run_gauntlet(
        self, args: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        from aragora.mcp.tools_module.gauntlet import run_gauntlet_tool

        payload = args or kwargs
        if not payload.get("content"):
            return {"error": "Content is required"}
        return await run_gauntlet_tool(**payload)

    async def _list_agents(
        self, args: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        from aragora.mcp.tools_module.agent import list_agents_tool

        _ = args or kwargs
        return await list_agents_tool()

    # ---------------------------------------------------------------------
    # MCP request handler shim for tests
    # ---------------------------------------------------------------------

    def _build_mcp_server_shim(self) -> Any:
        if not MCP_AVAILABLE:
            return None

        async def handle_list_tools(_request: Any) -> Any:
            tools = [
                Tool(name=t.name, description=t.description, inputSchema=t.input_schema)
                for t in self._tools.values()
            ]
            return ListToolsResult(tools=tools)

        async def handle_call_tool(request: Any) -> Any:
            name = request.params.name  # type: ignore[attr-defined]
            arguments = request.params.arguments or {}  # type: ignore[attr-defined]
            result = await self.invoke_tool(name, arguments)
            is_error = bool(result.get("error"))
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, default=str))],
                isError=is_error,
            )

        async def handle_list_resources(_request: Any) -> Any:
            resources = []
            for debate_id, debate in self._debates_cache.items():
                task = debate.get("task", "")
                name = f"Debate {debate_id}: {task[:60]}" if task else f"Debate {debate_id}"
                resources.append(
                    Resource(
                        uri=f"debate://{debate_id}",
                        name=name,
                        description="Cached debate result",
                        mimeType="application/json",
                    )
                )
            return ListResourcesResult(resources=resources)

        async def handle_list_resource_templates(_request: Any) -> Any:
            templates = [
                ResourceTemplate(
                    name="debate",
                    uriTemplate="debate://{debate_id}",
                    description="Cached debate result",
                    mimeType="application/json",
                )
            ]
            return ListResourceTemplatesResult(resourceTemplates=templates)

        async def handle_read_resource(request: Any) -> Any:
            uri = request.params.uri  # type: ignore[attr-defined]
            content = await self.read_resource(uri)
            text = content["contents"][0]["text"] if content.get("contents") else ""
            return ReadResourceResult(
                contents=[
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": text,
                    }
                ]
            )

        request_handlers = {
            ListToolsRequest: handle_list_tools,
            CallToolRequest: handle_call_tool,
            ListResourcesRequest: handle_list_resources,
            ListResourceTemplatesRequest: handle_list_resource_templates,
            ReadResourceRequest: handle_read_resource,
        }

        return SimpleNamespace(name=self.name, request_handlers=request_handlers)


# =============================================================================
# FastMCP runtime integration
# =============================================================================


def build_fastmcp_app(server: AragoraMCPServer) -> Any:
    """Create a FastMCP app from an AragoraMCPServer registry."""
    if not MCP_AVAILABLE:
        raise ImportError("MCP package not installed")

    from mcp.server.fastmcp import FastMCP

    app = FastMCP(name=server.name)

    for tool in server._tools.values():

        async def _handler(*, _tool_name: str = tool.name, **kwargs: Any) -> dict[str, Any]:
            return await server.invoke_tool(_tool_name, kwargs)

        app.add_tool(_handler, name=tool.name, description=tool.description)

    return app


async def run_server(transport: str = "stdio") -> None:
    """Run the MCP server using FastMCP."""
    if not MCP_AVAILABLE:
        raise SystemExit(1)

    server = AragoraMCPServer(require_mcp=True)
    app = build_fastmcp_app(server)
    app.run(transport=transport)


def main() -> None:
    """CLI entrypoint for running the MCP server."""
    parser = argparse.ArgumentParser(description="Run Aragora MCP server")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    args = parser.parse_args()

    try:
        asyncio.run(run_server(transport=args.transport))
    except SystemExit:
        raise
    except Exception as e:
        logger.error("MCP server failed: %s", e)
        raise SystemExit(1) from e


def create_mcp_server(
    name: str = "aragora",
    version: str = "1.0.0",
    tools_metadata: list[dict[str, Any]] | None = None,
    require_mcp: bool = True,
    rate_limits: dict[str, int] | None = None,
) -> AragoraMCPServer:
    return AragoraMCPServer(
        name=name,
        version=version,
        tools_metadata=tools_metadata,
        require_mcp=require_mcp,
        rate_limits=rate_limits,
    )


def _build_input_schema(parameters: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, spec in parameters.items():
        if not isinstance(spec, dict):
            continue
        schema = {k: v for k, v in spec.items() if k not in {"required"}}
        if "type" in spec:
            schema["type"] = spec["type"]
        if spec.get("required"):
            required.append(name)
        properties[name] = schema

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _expects_dict(handler: Callable[..., Any]) -> bool:
    name = getattr(handler, "__name__", "")
    return bool(name) and name.startswith("_")


__all__ = [
    "AragoraMCPServer",
    "MCPTool",
    "MCPResource",
    "MCPPrompt",
    "create_mcp_server",
    "build_fastmcp_app",
    "run_server",
    "MCP_AVAILABLE",
    "RateLimiter",
]


if __name__ == "__main__":
    main()
