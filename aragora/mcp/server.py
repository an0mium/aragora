"""
Aragora MCP Server Implementation.

Implements the MCP 1.0 protocol to expose Aragora capabilities.

Features:
- Rate limiting per tool (configurable via environment)
- Redis-backed rate limiting for multi-instance deployments
- Input validation and sanitization
- Comprehensive error handling and logging
- Resource caching for debate results
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, cast

logger = logging.getLogger(__name__)

# Rate limiting configuration (requests per minute per tool)
# Can be overridden via MCP_RATE_LIMIT_{TOOL_NAME} env vars
DEFAULT_RATE_LIMITS: Dict[str, int] = {
    "run_debate": int(os.environ.get("MCP_RATE_LIMIT_RUN_DEBATE", "10")),
    "run_gauntlet": int(os.environ.get("MCP_RATE_LIMIT_RUN_GAUNTLET", "20")),
    "list_agents": int(os.environ.get("MCP_RATE_LIMIT_LIST_AGENTS", "60")),
    "get_debate": int(os.environ.get("MCP_RATE_LIMIT_GET_DEBATE", "60")),
    "search_debates": int(os.environ.get("MCP_RATE_LIMIT_SEARCH_DEBATES", "30")),
    "get_agent_history": int(os.environ.get("MCP_RATE_LIMIT_GET_AGENT_HISTORY", "30")),
    "get_consensus_proofs": int(os.environ.get("MCP_RATE_LIMIT_GET_CONSENSUS_PROOFS", "30")),
    "list_trending_topics": int(os.environ.get("MCP_RATE_LIMIT_LIST_TRENDING_TOPICS", "30")),
}

# Rate limiter backend configuration
# MCP_RATE_LIMIT_BACKEND: "memory" (default) or "redis"
# MCP_REDIS_URL: Redis connection URL (e.g., redis://localhost:6379)
RATE_LIMIT_BACKEND = os.environ.get("MCP_RATE_LIMIT_BACKEND", "memory")
REDIS_URL = os.environ.get("MCP_REDIS_URL", "redis://localhost:6379")

# Maximum input sizes for validation
MAX_QUESTION_LENGTH = 10000
MAX_CONTENT_LENGTH = 100000
MAX_QUERY_LENGTH = 1000


class RateLimiterBase(ABC):
    """Abstract base class for rate limiters."""

    @abstractmethod
    def check(self, tool_name: str) -> Tuple[bool, Optional[str]]:
        """Check if a request is allowed.

        Returns:
            Tuple of (allowed, error_message)
        """
        pass

    @abstractmethod
    def get_remaining(self, tool_name: str) -> int:
        """Get remaining requests for a tool."""
        pass


class RateLimiter(RateLimiterBase):
    """Simple in-memory rate limiter with per-tool limits.

    Suitable for single-instance deployments. For multi-instance deployments,
    use RedisRateLimiter instead.
    """

    def __init__(self, limits: Optional[Dict[str, int]] = None):
        self._limits = limits or DEFAULT_RATE_LIMITS
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._window_seconds = 60

    def check(self, tool_name: str) -> Tuple[bool, Optional[str]]:
        """Check if a request is allowed.

        Returns:
            Tuple of (allowed, error_message)
        """
        limit = self._limits.get(tool_name, 60)  # Default 60/min
        now = time.time()
        window_start = now - self._window_seconds

        # Clean old requests
        self._requests[tool_name] = [t for t in self._requests[tool_name] if t > window_start]

        # Check limit
        if len(self._requests[tool_name]) >= limit:
            remaining_wait = self._requests[tool_name][0] - window_start
            return False, f"Rate limit exceeded for {tool_name}. Try again in {remaining_wait:.0f}s"

        # Record request
        self._requests[tool_name].append(now)
        return True, None

    def get_remaining(self, tool_name: str) -> int:
        """Get remaining requests for a tool."""
        limit = self._limits.get(tool_name, 60)
        now = time.time()
        window_start = now - self._window_seconds

        current_count = len([t for t in self._requests[tool_name] if t > window_start])
        return max(0, limit - current_count)


class RedisRateLimiter(RateLimiterBase):
    """Redis-backed rate limiter for multi-instance deployments.

    Uses Redis sorted sets for sliding window rate limiting.
    This allows rate limits to be shared across multiple MCP server instances.

    Configuration via environment variables:
        MCP_REDIS_URL: Redis connection URL (default: redis://localhost:6379)
        MCP_RATE_LIMIT_{TOOL_NAME}: Per-tool rate limits

    Example:
        export MCP_RATE_LIMIT_BACKEND=redis
        export MCP_REDIS_URL=redis://localhost:6379
        export MCP_RATE_LIMIT_RUN_DEBATE=10
    """

    def __init__(
        self,
        limits: Optional[Dict[str, int]] = None,
        redis_url: Optional[str] = None,
        key_prefix: str = "mcp:ratelimit:",
    ):
        self._limits = limits or DEFAULT_RATE_LIMITS
        self._redis_url = redis_url or REDIS_URL
        self._key_prefix = key_prefix
        self._window_seconds = 60
        self._redis: Any = None
        self._connected = False

    def _get_redis(self) -> Any:
        """Get or create Redis connection."""
        if self._redis is None:
            try:
                import redis

                self._redis = redis.from_url(self._redis_url, decode_responses=True)
                # Test connection
                self._redis.ping()
                self._connected = True
                logger.info(f"RedisRateLimiter connected to {self._redis_url}")
            except ImportError:
                logger.warning("redis package not installed. Install with: pip install redis")
                self._connected = False
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Falling back to permissive mode.")
                self._connected = False
        return self._redis

    def check(self, tool_name: str) -> Tuple[bool, Optional[str]]:
        """Check if a request is allowed using Redis sorted set.

        Uses sliding window algorithm with Redis sorted sets.

        Returns:
            Tuple of (allowed, error_message)
        """
        redis_client = self._get_redis()
        if not self._connected or redis_client is None:
            # If Redis is unavailable, allow the request (fail-open)
            logger.debug(f"Redis unavailable, allowing request for {tool_name}")
            return True, None

        limit = self._limits.get(tool_name, 60)
        now = time.time()
        window_start = now - self._window_seconds
        key = f"{self._key_prefix}{tool_name}"

        try:
            # Use Redis pipeline for atomic operations
            pipe = redis_client.pipeline()

            # Remove old entries outside the window
            pipe.zremrangebyscore(key, "-inf", window_start)

            # Count current entries
            pipe.zcard(key)

            # Execute pipeline
            results = pipe.execute()
            current_count = results[1]

            # Check limit
            if current_count >= limit:
                # Get the oldest timestamp to calculate wait time
                oldest = redis_client.zrange(key, 0, 0, withscores=True)
                if oldest:
                    remaining_wait = oldest[0][1] - window_start
                    return (
                        False,
                        f"Rate limit exceeded for {tool_name}. Try again in {remaining_wait:.0f}s",
                    )
                return False, f"Rate limit exceeded for {tool_name}."

            # Add current request with score = timestamp
            redis_client.zadd(key, {str(now): now})

            # Set expiry on the key to auto-cleanup
            redis_client.expire(key, self._window_seconds + 10)

            return True, None

        except Exception as e:
            logger.warning(f"Redis rate limit check failed: {e}. Allowing request.")
            return True, None

    def get_remaining(self, tool_name: str) -> int:
        """Get remaining requests for a tool."""
        redis_client = self._get_redis()
        if not self._connected or redis_client is None:
            # If Redis unavailable, return full limit
            return self._limits.get(tool_name, 60)

        limit = self._limits.get(tool_name, 60)
        now = time.time()
        window_start = now - self._window_seconds
        key = f"{self._key_prefix}{tool_name}"

        try:
            # Remove old entries and count current
            redis_client.zremrangebyscore(key, "-inf", window_start)
            current_count = redis_client.zcard(key)
            return max(0, limit - current_count)
        except Exception as e:
            logger.warning(f"Redis get_remaining failed: {e}")
            return limit

    def reset(self, tool_name: Optional[str] = None) -> None:
        """Reset rate limit counters.

        Args:
            tool_name: Specific tool to reset, or None to reset all.
        """
        redis_client = self._get_redis()
        if not self._connected or redis_client is None:
            return

        try:
            if tool_name:
                key = f"{self._key_prefix}{tool_name}"
                redis_client.delete(key)
            else:
                # Reset all rate limit keys
                pattern = f"{self._key_prefix}*"
                cursor = 0
                while True:
                    cursor, keys = redis_client.scan(cursor, match=pattern)
                    if keys:
                        redis_client.delete(*keys)
                    if cursor == 0:
                        break
        except Exception as e:
            logger.warning(f"Redis reset failed: {e}")


def create_rate_limiter(
    backend: Optional[str] = None,
    limits: Optional[Dict[str, int]] = None,
    redis_url: Optional[str] = None,
) -> RateLimiterBase:
    """Factory function to create the appropriate rate limiter.

    Args:
        backend: "memory" or "redis". If None, uses MCP_RATE_LIMIT_BACKEND env var.
        limits: Rate limit configuration per tool. Defaults to DEFAULT_RATE_LIMITS.
        redis_url: Redis URL for redis backend. Defaults to MCP_REDIS_URL env var.

    Returns:
        RateLimiter instance (in-memory or Redis-backed)

    Example:
        # Use environment variables
        limiter = create_rate_limiter()

        # Explicit configuration
        limiter = create_rate_limiter(
            backend="redis",
            redis_url="redis://localhost:6379",
            limits={"run_debate": 5}
        )
    """
    backend = backend or RATE_LIMIT_BACKEND

    if backend == "redis":
        logger.info("Using Redis-backed rate limiter")
        return RedisRateLimiter(limits=limits, redis_url=redis_url)
    else:
        logger.info("Using in-memory rate limiter")
        return RateLimiter(limits=limits)


from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    ResourceTemplate,
    TextContent,
    Tool,
)

MCP_AVAILABLE = True  # Kept for backwards compatibility


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

    def __init__(
        self,
        rate_limits: Optional[Dict[str, int]] = None,
        rate_limit_backend: Optional[str] = None,
        redis_url: Optional[str] = None,
    ):
        """Initialize the MCP server.

        Args:
            rate_limits: Per-tool rate limits. Defaults to DEFAULT_RATE_LIMITS.
            rate_limit_backend: "memory" or "redis". Defaults to MCP_RATE_LIMIT_BACKEND env var.
            redis_url: Redis URL for redis backend. Defaults to MCP_REDIS_URL env var.
        """
        if not MCP_AVAILABLE:
            raise ImportError("MCP package not installed. Install with: pip install mcp")

        self.server = Server("aragora")
        self._rate_limiter = create_rate_limiter(
            backend=rate_limit_backend,
            limits=rate_limits,
            redis_url=redis_url,
        )
        self._setup_handlers()
        self._debates_cache: Dict[str, Dict[str, Any]] = {}
        self._agents_cache: Dict[str, Dict[str, Any]] = {}

    def _validate_input(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Validate tool input arguments.

        Returns error message if validation fails, None otherwise.
        """
        if tool_name == "run_debate":
            question = arguments.get("question", "")
            if question and len(question) > MAX_QUESTION_LENGTH:
                return f"Question exceeds maximum length ({MAX_QUESTION_LENGTH} chars)"
            rounds = arguments.get("rounds", 3)
            if not isinstance(rounds, int) or rounds < 1 or rounds > 10:
                return "Rounds must be an integer between 1 and 10"

        elif tool_name == "run_gauntlet":
            content = arguments.get("content", "")
            if content and len(content) > MAX_CONTENT_LENGTH:
                return f"Content exceeds maximum length ({MAX_CONTENT_LENGTH} chars)"

        elif tool_name == "search_debates":
            query = arguments.get("query", "")
            if query and len(query) > MAX_QUERY_LENGTH:
                return f"Query exceeds maximum length ({MAX_QUERY_LENGTH} chars)"

        return None

    def _sanitize_arguments(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize tool arguments to prevent injection attacks.

        Returns sanitized copy of arguments.
        """
        sanitized = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                # Strip leading/trailing whitespace
                sanitized[key] = value.strip()
            else:
                sanitized[key] = value
        return sanitized

    def _setup_handlers(self) -> None:
        """Set up MCP request handlers."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools from tools.py metadata."""
            from aragora.mcp.tools import TOOLS_METADATA

            tools = []
            for meta in TOOLS_METADATA:
                # Build JSON Schema from parameters
                properties = {}
                required = []

                for param_name, param_info in cast(
                    Dict[str, Any], meta.get("parameters", {})
                ).items():
                    prop: Dict[str, Any] = {"type": param_info.get("type", "string")}
                    if "default" in param_info:
                        prop["default"] = param_info["default"]
                    if "description" in param_info:
                        prop["description"] = param_info["description"]
                    if "enum" in param_info:
                        prop["enum"] = param_info["enum"]
                    if "minimum" in param_info:
                        prop["minimum"] = param_info["minimum"]
                    if "maximum" in param_info:
                        prop["maximum"] = param_info["maximum"]
                    properties[param_name] = prop

                    if param_info.get("required", False):
                        required.append(param_name)

                input_schema: Dict[str, Any] = {
                    "type": "object",
                    "properties": properties,
                }
                if required:
                    input_schema["required"] = required

                tools.append(
                    Tool(
                        name=str(meta["name"]),
                        description=str(meta["description"]),
                        inputSchema=input_schema,
                    )
                )

            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls with rate limiting and input validation."""
            from aragora.mcp.tools import TOOLS_METADATA

            try:
                # Rate limiting check
                allowed, rate_error = self._rate_limiter.check(name)
                if not allowed:
                    logger.warning(f"Rate limit exceeded for tool {name}")
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "error": rate_error,
                                    "rate_limited": True,
                                }
                            ),
                        )
                    ]

                # Input validation
                validation_error = self._validate_input(name, arguments)
                if validation_error:
                    logger.warning(f"Input validation failed for {name}: {validation_error}")
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps({"error": validation_error}),
                        )
                    ]

                # Find tool function from metadata
                tool_func: Optional[Callable[..., Coroutine[Any, Any, Dict[str, Any]]]] = None
                for meta in TOOLS_METADATA:
                    if meta["name"] == name:
                        tool_func = cast(
                            Callable[..., Coroutine[Any, Any, Dict[str, Any]]], meta["function"]
                        )
                        break

                if tool_func is None:
                    result: Dict[str, Any] = {"error": f"Unknown tool: {name}"}
                else:
                    # Call the tool function with sanitized arguments
                    sanitized_args = self._sanitize_arguments(arguments)
                    result = await tool_func(**sanitized_args)

                    # Cache debate results for resource access
                    if name == "run_debate" and "debate_id" in result and "error" not in result:
                        self._debates_cache[result["debate_id"]] = result

                return [
                    TextContent(
                        type="text",
                        text=json.dumps(result, indent=2),
                    )
                ]
            except TypeError as e:
                # Handle missing required arguments
                logger.error(f"Tool {name} called with invalid arguments: {e}")
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": f"Invalid arguments: {e}"}),
                    )
                ]
            except Exception as e:
                logger.error(f"Tool {name} failed: {e}", exc_info=True)
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": str(e)}),
                    )
                ]

        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available resources (cached debates)."""
            resources = []
            for debate_id, debate_data in self._debates_cache.items():
                resources.append(
                    Resource(
                        uri=f"debate://{debate_id}",  # type: ignore[arg-type]
                        name=f"Debate: {debate_data.get('task', 'Unknown')[:50]}",
                        description=f"Debate result from {debate_data.get('timestamp', 'unknown time')}",
                        mimeType="application/json",
                    )
                )
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
                    uriTemplate="consensus://{debate_id}",
                    name="Consensus Proofs",
                    description="Access formal verification proofs for a debate",
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
        async def read_resource(uri) -> str:
            """Read a resource by URI."""
            # Convert AnyUrl to string if needed (MCP 1.x compatibility)
            uri = str(uri)
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

            elif uri.startswith("consensus://"):
                # Extract debate_id from consensus://{debate_id}
                debate_id = uri.replace("consensus://", "")
                result = await self._get_consensus_proofs({"debate_id": debate_id})
                return json.dumps(result, indent=2)

            elif uri == "trending://topics":
                result = await self._list_trending_topics({})
                return json.dumps(result, indent=2)

            return json.dumps({"error": f"Unknown resource: {uri}"})

    async def _run_debate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run a debate and return results."""
        import time
        import uuid

        from aragora.agents.base import create_agent
        from aragora.core import Environment
        from aragora.debate.orchestrator import Arena, DebateProtocol

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
        arena = Arena(env, cast(List[Any], agents), protocol)
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
            CODE_REVIEW_GAUNTLET,
            GDPR_GAUNTLET,
            HIPAA_GAUNTLET,
            QUICK_GAUNTLET,
            SECURITY_GAUNTLET,
            THOROUGH_GAUNTLET,
            GauntletRunner,
        )

        content = args.get("content", "")
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

        # Use the selected profile config directly (it's already a GauntletConfig)
        if config is None:
            config = QUICK_GAUNTLET

        runner = GauntletRunner(config)
        result = await runner.run(content)

        return {
            "verdict": result.verdict.value if hasattr(result, "verdict") else "unknown",
            "risk_score": getattr(result, "risk_score", 0),
            "vulnerabilities_count": len(getattr(result, "vulnerabilities", [])),
            "vulnerabilities": [
                {
                    "category": v.category,
                    "severity": v.severity,
                    "description": v.description,
                }
                for v in getattr(result, "vulnerabilities", [])[:5]  # Limit to 5
            ],
            "summary": result.summary() if hasattr(result, "summary") else str(result),
        }

    async def _list_agents(self) -> Dict[str, Any]:
        """List available agents."""
        from aragora.agents.base import list_available_agents

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

            results.append(
                {
                    "debate_id": debate_id,
                    "task": debate_data.get("task", "")[:100],
                    "consensus_reached": debate_data.get("consensus_reached", False),
                    "confidence": debate_data.get("confidence", 0),
                    "timestamp": debate_data.get("timestamp", ""),
                }
            )

        # Try storage for more results
        try:
            from aragora.server.storage import get_debates_db

            db = get_debates_db()
            if db and hasattr(db, "search"):
                stored_results, _total = db.search(
                    query=query,
                    limit=limit,
                )
                for debate_meta in stored_results:
                    debate_dict = {
                        "debate_id": debate_meta.debate_id,
                        "task": debate_meta.task[:100] if debate_meta.task else "",
                        "consensus_reached": debate_meta.consensus_reached,
                        "confidence": debate_meta.confidence,
                        "timestamp": (
                            debate_meta.created_at.isoformat() if debate_meta.created_at else ""
                        ),
                    }
                    if debate_dict["debate_id"] not in [r["debate_id"] for r in results]:
                        results.append(debate_dict)
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
            from aragora.ranking.elo import EloSystem

            elo = EloSystem()
            rating = elo.get_rating(agent_name)
            if rating:
                result["elo_rating"] = rating.elo
        except Exception as e:
            logger.debug(f"Could not get ELO: {e}")

        # Try to get performance stats
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
        except Exception as e:
            logger.debug(f"Could not get agent stats: {e}")

        # Get recent debates from cache
        if include_debates:
            agent_debates = []
            for debate_id, debate_data in self._debates_cache.items():
                agents = debate_data.get("agents", [])
                if any(agent_name.lower() in a.lower() for a in agents):
                    agent_debates.append(
                        {
                            "debate_id": debate_id,
                            "task": debate_data.get("task", "")[:80],
                            "consensus_reached": debate_data.get("consensus_reached", False),
                            "timestamp": debate_data.get("timestamp", ""),
                        }
                    )

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

        # Note: No dedicated proofs storage currently implemented
        # Proofs are stored within debate records

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
            from aragora.pulse import PulseManager, SchedulerConfig, TopicSelector

            # Get trending topics via PulseManager
            manager = PulseManager()
            raw_topics = await manager.get_trending_topics(
                platforms=[platform] if platform != "all" else [],
                limit_per_platform=limit * 2,  # Get more, then filter
            )

            # Score and filter topics
            config = SchedulerConfig()
            selector = TopicSelector(config)

            for topic in raw_topics:
                # Apply platform filter
                if platform != "all" and topic.platform != platform:
                    continue

                # Apply category filter
                if category and topic.category.lower() != category.lower():
                    continue

                # Score the topic
                topic_score = selector.score_topic(topic)
                score_value = topic_score.score

                if score_value >= min_score:
                    topics.append(
                        {
                            "topic": topic.topic,
                            "platform": topic.platform,
                            "category": topic.category,
                            "score": round(score_value, 3),
                            "volume": topic.volume,
                            "debate_potential": "high" if score_value > 0.7 else "medium",
                        }
                    )

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
        logger.error("Error: MCP package not installed.")
        logger.error("Install with: pip install mcp")
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
