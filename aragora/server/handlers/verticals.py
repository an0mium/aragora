"""
Vertical specialist endpoint handlers.

Exposes the vertical specialist system for domain-specific AI agents.

Endpoints:
- GET /api/verticals - List available verticals
- GET /api/verticals/:id - Get vertical config
- GET /api/verticals/:id/tools - Get vertical tools
- GET /api/verticals/:id/compliance - Get compliance frameworks
- POST /api/verticals/:id/debate - Create vertical-specific debate
- POST /api/verticals/:id/agent - Create specialist agent instance
- GET /api/verticals/suggest - Suggest vertical for a task
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from aragora.server.validation import validate_path_segment, SAFE_ID_PATTERN

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_string_param,
    json_response,
)
from .utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)


class VerticalsHandler(BaseHandler):
    """Handler for vertical specialist endpoints."""

    ROUTES = [
        "/api/verticals",
        "/api/verticals/suggest",
        "/api/verticals/*",
        "/api/verticals/*/tools",
        "/api/verticals/*/compliance",
        "/api/verticals/*/debate",
        "/api/verticals/*/agent",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can handle the request."""
        if path == "/api/verticals":
            return True
        if path == "/api/verticals/suggest":
            return True
        if path.startswith("/api/verticals/"):
            return True
        return False

    @rate_limit(rpm=60)
    async def handle(  # type: ignore[override]
        self, path: str, method: str, handler: Any = None
    ) -> Optional[HandlerResult]:
        """Route request to appropriate handler method."""
        query_params: Dict[str, Any] = {}
        if handler:
            query_str = handler.path.split("?", 1)[1] if "?" in handler.path else ""
            from urllib.parse import parse_qs
            query_params = parse_qs(query_str)

        # GET /api/verticals - List all verticals
        if path == "/api/verticals" and method == "GET":
            return self._list_verticals(query_params)

        # GET /api/verticals/suggest - Suggest vertical for task
        if path == "/api/verticals/suggest" and method == "GET":
            return self._suggest_vertical(query_params)

        # Handle specific vertical endpoints
        if path.startswith("/api/verticals/"):
            parts = path.split("/")

            # GET /api/verticals/:id/tools
            if len(parts) == 5 and parts[4] == "tools" and method == "GET":
                vertical_id = parts[3]
                is_valid, err = validate_path_segment(vertical_id, "vertical_id", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
                return self._get_tools(vertical_id)

            # GET /api/verticals/:id/compliance
            if len(parts) == 5 and parts[4] == "compliance" and method == "GET":
                vertical_id = parts[3]
                is_valid, err = validate_path_segment(vertical_id, "vertical_id", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
                return self._get_compliance(vertical_id, query_params)

            # POST /api/verticals/:id/debate
            if len(parts) == 5 and parts[4] == "debate" and method == "POST":
                vertical_id = parts[3]
                is_valid, err = validate_path_segment(vertical_id, "vertical_id", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
                return await self._create_debate(vertical_id, handler)

            # POST /api/verticals/:id/agent
            if len(parts) == 5 and parts[4] == "agent" and method == "POST":
                vertical_id = parts[3]
                is_valid, err = validate_path_segment(vertical_id, "vertical_id", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
                return self._create_agent(vertical_id, handler)

            # GET /api/verticals/:id
            if len(parts) == 4 and method == "GET":
                vertical_id = parts[3]
                is_valid, err = validate_path_segment(vertical_id, "vertical_id", SAFE_ID_PATTERN)
                if not is_valid:
                    return error_response(err, 400)
                return self._get_vertical(vertical_id)

        return None

    def _get_registry(self) -> Optional[Any]:
        """Get the VerticalRegistry, handling import errors."""
        try:
            from aragora.verticals import VerticalRegistry
            return VerticalRegistry
        except ImportError:
            logger.warning("Verticals module not available")
            return None

    def _list_verticals(self, query_params: Dict[str, Any]) -> HandlerResult:
        """List all available verticals."""
        registry = self._get_registry()
        if registry is None:
            return json_response({
                "verticals": [],
                "total": 0,
                "message": "Verticals module not available",
            }, status=503)

        try:
            # Get optional keyword filter
            keyword = get_string_param(query_params, "keyword", None)

            if keyword:
                # Filter by keyword
                matching_ids = registry.get_by_keyword(keyword)
                all_verticals = registry.list_all()
                verticals = {
                    vid: all_verticals[vid]
                    for vid in matching_ids
                    if vid in all_verticals
                }
            else:
                verticals = registry.list_all()

            return json_response({
                "verticals": [
                    {
                        "vertical_id": vid,
                        **data,
                    }
                    for vid, data in verticals.items()
                ],
                "total": len(verticals),
            })

        except Exception as e:
            logger.error(f"Failed to list verticals: {e}")
            return error_response(f"Failed to list verticals: {e}", 500)

    def _get_vertical(self, vertical_id: str) -> HandlerResult:
        """Get a specific vertical's configuration."""
        registry = self._get_registry()
        if registry is None:
            return error_response("Verticals module not available", 503)

        try:
            spec = registry.get(vertical_id)
            if spec is None:
                available = registry.get_registered_ids()
                return error_response(
                    f"Vertical not found: {vertical_id}. "
                    f"Available: {', '.join(available)}",
                    404
                )

            config = spec.config

            return json_response({
                "vertical_id": vertical_id,
                "display_name": config.display_name,
                "description": spec.description,
                "domain_keywords": config.domain_keywords,
                "expertise_areas": config.expertise_areas,
                "tools": [t.to_dict() for t in config.tools],
                "compliance_frameworks": [c.to_dict() for c in config.compliance_frameworks],
                "model_config": config.model_config.to_dict(),
                "version": config.version,
                "author": config.author,
                "tags": config.tags,
            })

        except Exception as e:
            logger.error(f"Failed to get vertical {vertical_id}: {e}")
            return error_response(f"Failed to get vertical: {e}", 500)

    def _get_tools(self, vertical_id: str) -> HandlerResult:
        """Get tools available for a vertical."""
        registry = self._get_registry()
        if registry is None:
            return error_response("Verticals module not available", 503)

        try:
            config = registry.get_config(vertical_id)
            if config is None:
                return error_response(f"Vertical not found: {vertical_id}", 404)

            tools = config.tools
            enabled_tools = config.get_enabled_tools()

            return json_response({
                "vertical_id": vertical_id,
                "tools": [t.to_dict() for t in tools],
                "enabled_count": len(enabled_tools),
                "total_count": len(tools),
            })

        except Exception as e:
            logger.error(f"Failed to get tools for {vertical_id}: {e}")
            return error_response(f"Failed to get tools: {e}", 500)

    def _get_compliance(self, vertical_id: str, query_params: Dict[str, Any]) -> HandlerResult:
        """Get compliance frameworks for a vertical."""
        registry = self._get_registry()
        if registry is None:
            return error_response("Verticals module not available", 503)

        try:
            from aragora.verticals.config import ComplianceLevel

            config = registry.get_config(vertical_id)
            if config is None:
                return error_response(f"Vertical not found: {vertical_id}", 404)

            # Optional filter by level
            level_filter = get_string_param(query_params, "level", None)
            level_enum = None
            if level_filter:
                try:
                    level_enum = ComplianceLevel(level_filter)
                except ValueError:
                    return error_response(
                        f"Invalid level: {level_filter}. "
                        f"Valid values: {[l.value for l in ComplianceLevel]}",
                        400
                    )

            frameworks = config.get_compliance_frameworks(level=level_enum)

            return json_response({
                "vertical_id": vertical_id,
                "compliance_frameworks": [f.to_dict() for f in frameworks],
                "total": len(frameworks),
            })

        except ImportError:
            return error_response("Compliance module not available", 503)
        except Exception as e:
            logger.error(f"Failed to get compliance for {vertical_id}: {e}")
            return error_response(f"Failed to get compliance: {e}", 500)

    def _suggest_vertical(self, query_params: Dict[str, Any]) -> HandlerResult:
        """Suggest the best vertical for a task description."""
        registry = self._get_registry()
        if registry is None:
            return error_response("Verticals module not available", 503)

        task = get_string_param(query_params, "task", None)
        if not task:
            return error_response("Missing required parameter: task", 400)

        try:
            suggested = registry.get_for_task(task)

            if suggested is None:
                return json_response({
                    "suggestion": None,
                    "message": "No vertical matches the given task",
                    "available_verticals": registry.get_registered_ids(),
                })

            spec = registry.get(suggested)
            config = spec.config if spec else None

            return json_response({
                "suggestion": {
                    "vertical_id": suggested,
                    "display_name": config.display_name if config else suggested,
                    "description": spec.description if spec else "",
                    "expertise_areas": config.expertise_areas if config else [],
                },
                "task": task,
            })

        except Exception as e:
            logger.error(f"Failed to suggest vertical: {e}")
            return error_response(f"Failed to suggest vertical: {e}", 500)

    async def _create_debate(self, vertical_id: str, handler: Any) -> HandlerResult:
        """Create a debate using a vertical specialist."""
        registry = self._get_registry()
        if registry is None:
            return error_response("Verticals module not available", 503)

        # Validate vertical exists
        if not registry.is_registered(vertical_id):
            return error_response(f"Vertical not found: {vertical_id}", 404)

        # Parse request body
        data = self.read_json_body(handler)
        if data is None:
            return error_response("Invalid or too large request body", 400)

        topic = data.get("topic")
        if not topic:
            return error_response("Missing required field: topic", 400)

        try:
            # Import debate infrastructure
            from aragora.core import DebateProtocol, Environment
            from aragora.debate.orchestrator import Arena

            # Create specialist agent
            agent_name = data.get("agent_name", f"{vertical_id}-specialist")
            model = data.get("model")
            role = data.get("role", "specialist")

            specialist = registry.create_specialist(
                vertical_id=vertical_id,
                name=agent_name,
                model=model,
                role=role,
            )

            # Get additional agents if specified
            agents = [specialist]
            additional_agents = data.get("additional_agents", [])

            if additional_agents:
                from aragora.agents.base import create_agent

                for agent_spec in additional_agents:
                    if isinstance(agent_spec, str):
                        agents.append(create_agent(agent_spec))
                    elif isinstance(agent_spec, dict):
                        agents.append(create_agent(
                            agent_spec.get("type", "anthropic-api"),
                            name=agent_spec.get("name"),
                            role=agent_spec.get("role"),
                        ))

            # Create environment and protocol
            env = Environment(task=topic)
            protocol = DebateProtocol(
                rounds=data.get("rounds", 3),
                consensus=data.get("consensus", "weighted"),
            )

            # Run debate
            arena = Arena(env, agents=agents, protocol=protocol)
            result = await arena.run()

            return json_response({
                "debate_id": result.debate_id if hasattr(result, "debate_id") else None,
                "vertical_id": vertical_id,
                "topic": topic,
                "consensus_reached": (
                    result.consensus_reached if hasattr(result, "consensus_reached") else False
                ),
                "final_answer": result.final_answer if hasattr(result, "final_answer") else None,
                "confidence": result.confidence if hasattr(result, "confidence") else 0.0,
                "participants": [a.name for a in agents],
            })

        except ImportError as e:
            logger.error(f"Debate infrastructure not available: {e}")
            return error_response("Debate infrastructure not available", 503)
        except Exception as e:
            logger.error(f"Failed to create debate for {vertical_id}: {e}")
            return error_response(f"Failed to create debate: {e}", 500)

    def _create_agent(self, vertical_id: str, handler: Any) -> HandlerResult:
        """Create a specialist agent instance."""
        registry = self._get_registry()
        if registry is None:
            return error_response("Verticals module not available", 503)

        # Validate vertical exists
        if not registry.is_registered(vertical_id):
            return error_response(f"Vertical not found: {vertical_id}", 404)

        # Parse request body
        data = self.read_json_body(handler)
        if data is None:
            return error_response("Invalid or too large request body", 400)

        try:
            # Create specialist with provided config
            name = data.get("name", f"{vertical_id}-agent")
            model = data.get("model")
            role = data.get("role", "specialist")

            specialist = registry.create_specialist(
                vertical_id=vertical_id,
                name=name,
                model=model,
                role=role,
            )

            return json_response({
                "agent": specialist.to_dict(),
                "vertical_id": vertical_id,
                "name": specialist.name,
                "model": specialist.model,
                "role": specialist.role,
                "expertise_areas": specialist.expertise_areas,
                "tools": [t.name for t in specialist.get_enabled_tools()],
                "message": "Specialist agent created successfully",
            })

        except ValueError as e:
            return error_response(str(e), 400)
        except Exception as e:
            logger.error(f"Failed to create agent for {vertical_id}: {e}")
            return error_response(f"Failed to create agent: {e}", 500)
