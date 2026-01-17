"""
Agent configuration endpoint handlers.

Endpoints:
- GET /api/agents/configs - List available YAML agent configurations
- GET /api/agents/configs/{name} - Get specific agent configuration
- POST /api/agents/configs/{name}/create - Create agent from configuration
- POST /api/agents/configs/reload - Reload all configurations from disk
- GET /api/agents/configs/search - Search configs by expertise/capability/tag
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from ..base import (
    SAFE_AGENT_PATTERN,
    BaseHandler,
    HandlerResult,
    error_response,
    get_string_param,
    handle_errors,
    json_response,
    validate_path_segment,
)
from ..utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# Global config loader instance
_config_loader: Any = None


def get_config_loader() -> Any:
    """Get or create the global AgentConfigLoader instance."""
    global _config_loader
    if _config_loader is None:
        try:
            from aragora.agents.config_loader import AgentConfigLoader

            _config_loader = AgentConfigLoader()
            # Load default configs
            default_dir = Path(__file__).parent.parent.parent.parent / "agents" / "configs"
            if default_dir.exists():
                _config_loader.load_directory(default_dir)
                logger.info(f"Loaded {len(_config_loader.list_configs())} agent configs")
        except ImportError as e:
            logger.warning(f"AgentConfigLoader not available: {e}")
    return _config_loader


class AgentConfigHandler(BaseHandler):
    """Handler for agent configuration endpoints."""

    ROUTES = [
        "/api/agents/configs",
        "/api/agents/configs/reload",
        "/api/agents/configs/search",
        "/api/agents/configs/*",
        "/api/agents/configs/*/create",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path.startswith("/api/agents/configs")

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route config requests to appropriate methods."""
        # List all configs
        if path == "/api/agents/configs":
            return self._list_configs(query_params)

        # Reload configs
        if path == "/api/agents/configs/reload":
            return self._reload_configs()

        # Search configs
        if path == "/api/agents/configs/search":
            return self._search_configs(query_params)

        # Handle specific config endpoints
        if path.startswith("/api/agents/configs/"):
            return self._handle_config_endpoint(path, query_params)

        return None

    def _handle_config_endpoint(self, path: str, query_params: dict) -> Optional[HandlerResult]:
        """Handle /api/agents/configs/{name}/* endpoints."""
        parts = path.split("/")
        if len(parts) < 5:
            return error_response("Invalid config path", 400)

        # Extract and validate config name
        config_name = parts[4]
        is_valid, err = validate_path_segment(config_name, "config_name", SAFE_AGENT_PATTERN)
        if not is_valid:
            return error_response(err, 400)

        # Create agent from config: /api/agents/configs/{name}/create
        if len(parts) >= 6 and parts[5] == "create":
            return self._create_agent_from_config(config_name)

        # Get specific config: /api/agents/configs/{name}
        return self._get_config(config_name)

    @rate_limit(rpm=30, limiter_name="config_list")
    @handle_errors("list configs")
    def _list_configs(self, query_params: dict) -> HandlerResult:
        """List all available YAML agent configurations.

        Query params:
            priority: Filter by priority level (low, normal, high, critical)
            role: Filter by role (proposer, critic, synthesizer, judge)

        Returns:
            List of configuration summaries
        """
        loader = get_config_loader()
        if not loader:
            return error_response("Config loader not available", 503)

        priority_filter = get_string_param(query_params, "priority")
        role_filter = get_string_param(query_params, "role")

        configs = []
        for name in loader.list_configs():
            config = loader.get_config(name)
            if config:
                # Apply filters
                if priority_filter and config.priority != priority_filter:
                    continue
                if role_filter and config.role != role_filter:
                    continue

                configs.append(
                    {
                        "name": config.name,
                        "model_type": config.model_type,
                        "role": config.role,
                        "priority": config.priority,
                        "description": config.description,
                        "expertise_domains": config.expertise_domains,
                        "capabilities": config.capabilities,
                        "tags": config.tags,
                    }
                )

        return json_response(
            {
                "configs": configs,
                "total": len(configs),
            }
        )

    @rate_limit(rpm=30, limiter_name="config_get")
    @handle_errors("get config")
    def _get_config(self, name: str) -> HandlerResult:
        """Get a specific agent configuration by name.

        Returns:
            Full configuration details
        """
        loader = get_config_loader()
        if not loader:
            return error_response("Config loader not available", 503)

        config = loader.get_config(name)
        if not config:
            return error_response(f"Config not found: {name}", 404)

        return json_response(
            {
                "config": config.to_dict(),
            }
        )

    @rate_limit(rpm=10, limiter_name="config_create")
    @handle_errors("create agent from config")
    def _create_agent_from_config(self, name: str) -> HandlerResult:
        """Create an agent from a YAML configuration.

        Returns:
            Created agent info
        """
        loader = get_config_loader()
        if not loader:
            return error_response("Config loader not available", 503)

        config = loader.get_config(name)
        if not config:
            return error_response(f"Config not found: {name}", 404)

        try:
            agent = loader.create_agent(config)
            return json_response(
                {
                    "success": True,
                    "agent": {
                        "name": agent.name,
                        "role": getattr(agent, "role", config.role),
                        "model_type": config.model_type,
                    },
                    "config_used": name,
                }
            )
        except Exception as e:
            logger.error(f"Failed to create agent from config {name}: {e}")
            return error_response(f"Failed to create agent: {e}", 500)

    @rate_limit(rpm=5, limiter_name="config_reload")
    @handle_errors("reload configs")
    def _reload_configs(self) -> HandlerResult:
        """Reload all configurations from disk.

        Returns:
            Reload status and count
        """
        loader = get_config_loader()
        if not loader:
            return error_response("Config loader not available", 503)

        try:
            reloaded = loader.reload_all()
            return json_response(
                {
                    "success": True,
                    "reloaded": len(reloaded),
                    "configs": list(reloaded.keys()),
                }
            )
        except Exception as e:
            logger.error(f"Failed to reload configs: {e}")
            return error_response(f"Reload failed: {e}", 500)

    @rate_limit(rpm=30, limiter_name="config_search")
    @handle_errors("search configs")
    def _search_configs(self, query_params: dict) -> HandlerResult:
        """Search configurations by expertise, capability, or tag.

        Query params:
            expertise: Domain expertise to search for
            capability: Capability to search for
            tag: Tag to search for

        Returns:
            Matching configurations
        """
        loader = get_config_loader()
        if not loader:
            return error_response("Config loader not available", 503)

        expertise = get_string_param(query_params, "expertise")
        capability = get_string_param(query_params, "capability")
        tag = get_string_param(query_params, "tag")

        if not any([expertise, capability, tag]):
            return error_response(
                "At least one search parameter required: expertise, capability, or tag", 400
            )

        results = []
        seen = set()

        if expertise:
            for config in loader.get_by_expertise(expertise):
                if config.name not in seen:
                    results.append(config)
                    seen.add(config.name)

        if capability:
            for config in loader.get_by_capability(capability):
                if config.name not in seen:
                    results.append(config)
                    seen.add(config.name)

        if tag:
            for config in loader.get_by_tag(tag):
                if config.name not in seen:
                    results.append(config)
                    seen.add(config.name)

        return json_response(
            {
                "results": [
                    {
                        "name": c.name,
                        "model_type": c.model_type,
                        "role": c.role,
                        "description": c.description,
                        "expertise_domains": c.expertise_domains,
                        "capabilities": c.capabilities,
                        "tags": c.tags,
                    }
                    for c in results
                ],
                "total": len(results),
                "search_params": {
                    "expertise": expertise,
                    "capability": capability,
                    "tag": tag,
                },
            }
        )
