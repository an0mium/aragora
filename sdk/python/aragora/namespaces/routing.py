"""
Routing Namespace API.

Provides intelligent team selection and request routing:
- Optimal agent team selection
- Auto-routing based on task domain
- Domain detection
- Routing rule management
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

RoutingStrategy = Literal["elo", "calibration", "hybrid", "round_robin", "expertise"]
Domain = Literal[
    "legal",
    "finance",
    "healthcare",
    "technology",
    "marketing",
    "hr",
    "operations",
    "security",
    "compliance",
    "general",
]


class RoutingAPI:
    """
    Synchronous Routing API.

    Provides methods for intelligent request routing:
    - Select optimal agent teams
    - Auto-route requests to appropriate agents
    - Detect task domains
    - Manage routing rules
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def select_team(
        self,
        task: str,
        team_size: int = 3,
        strategy: RoutingStrategy = "hybrid",
        required_skills: list[str] | None = None,
        excluded_agents: list[str] | None = None,
        domain: Domain | None = None,
    ) -> dict[str, Any]:
        """
        Select an optimal agent team for a task.

        Args:
            task: Task description.
            team_size: Number of agents to select.
            strategy: Selection strategy.
            required_skills: Skills that must be present.
            excluded_agents: Agents to exclude.
            domain: Specific domain hint.

        Returns:
            Selected team recommendations with agent details and rationale.
        """
        data: dict[str, Any] = {
            "task": task,
            "limit": team_size,
            "strategy": strategy,
        }
        if required_skills:
            data["required_traits"] = required_skills
        if excluded_agents:
            data["exclude"] = excluded_agents
        if domain:
            data["primary_domain"] = domain

        return self._client.request("POST", "/api/v1/routing/recommendations", json=data)

    def auto_route(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        user_preferences: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Automatically route a task to the best handling strategy.

        Args:
            task: Task description.
            context: Additional context.
            user_preferences: User routing preferences.

        Returns:
            Routing decision with recommended agents and strategy.
        """
        data: dict[str, Any] = {"task": task}
        if context:
            data["context"] = context
        if user_preferences:
            data["user_preferences"] = user_preferences

        return self._client.request("POST", "/api/v1/routing/auto-route", json=data)

    def detect_domain(self, task: str) -> dict[str, Any]:
        """
        Detect the domain of a task.

        Args:
            task: Task description.

        Returns:
            Domain detection result with confidence scores.
        """
        data: dict[str, Any] = {"task": task}
        return self._client.request("POST", "/api/v1/routing/detect-domain", json=data)

    def list_rules(
        self,
        domain: Domain | None = None,
        active_only: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List routing rules.

        Args:
            domain: Optional tag filter (mapped to routing-rule tags).
            active_only: Only return enabled rules.
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            Paginated list of routing rules.
        """
        params: dict[str, Any] = {
            "enabled_only": active_only,
            "limit": limit,
            "offset": offset,
        }
        if domain:
            params["tags"] = domain

        return self._client.request("GET", "/api/v1/routing-rules", params=params)

    def create_rule(
        self,
        name: str,
        conditions: dict[str, Any],
        actions: dict[str, Any],
        domain: Domain | None = None,
        priority: int = 0,
        description: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a routing rule.

        Args:
            name: Rule name.
            conditions: Matching conditions.
            actions: Actions to take when matched.
            domain: Optional tag value for the rule.
            priority: Rule priority (higher = more priority).
            description: Rule description.

        Returns:
            Created rule.
        """
        data: dict[str, Any] = {
            "name": name,
            "conditions": conditions,
            "actions": actions,
            "priority": priority,
        }
        if domain:
            data["tags"] = [domain]
        if description:
            data["description"] = description

        return self._client.request("POST", "/api/v1/routing-rules", json=data)

    def get_stats(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """
        Get routing statistics.

        Args:
            start_date: Start date filter (ISO format).
            end_date: End date filter (ISO format).

        Returns:
            Routing statistics.
        """
        params: dict[str, Any] = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        raise NotImplementedError("Routing stats endpoint is not exposed via the public API")


class AsyncRoutingAPI:
    """Asynchronous Routing API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def select_team(
        self,
        task: str,
        team_size: int = 3,
        strategy: RoutingStrategy = "hybrid",
        required_skills: list[str] | None = None,
        excluded_agents: list[str] | None = None,
        domain: Domain | None = None,
    ) -> dict[str, Any]:
        """Select an optimal agent team for a task."""
        data: dict[str, Any] = {
            "task": task,
            "team_size": team_size,
            "strategy": strategy,
        }
        if required_skills:
            data["required_skills"] = required_skills
        if excluded_agents:
            data["excluded_agents"] = excluded_agents
        if domain:
            data["domain"] = domain

        return await self._client.request("POST", "/api/v1/routing/recommendations", json=data)

    async def auto_route(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        user_preferences: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Automatically route a task to the best handling strategy."""
        data: dict[str, Any] = {"task": task}
        if context:
            data["context"] = context
        if user_preferences:
            data["user_preferences"] = user_preferences

        return await self._client.request("POST", "/api/v1/routing/auto-route", json=data)

    async def detect_domain(self, task: str) -> dict[str, Any]:
        """Detect the domain of a task."""
        data: dict[str, Any] = {"task": task}
        return await self._client.request("POST", "/api/v1/routing/detect-domain", json=data)

    async def list_rules(
        self,
        domain: Domain | None = None,
        active_only: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List routing rules."""
        params: dict[str, Any] = {
            "enabled_only": active_only,
            "limit": limit,
            "offset": offset,
        }
        if domain:
            params["tags"] = domain

        return await self._client.request("GET", "/api/v1/routing-rules", params=params)

    async def create_rule(
        self,
        name: str,
        conditions: dict[str, Any],
        actions: dict[str, Any],
        domain: Domain | None = None,
        priority: int = 0,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Create a routing rule."""
        data: dict[str, Any] = {
            "name": name,
            "conditions": conditions,
            "actions": actions,
            "priority": priority,
        }
        if domain:
            data["tags"] = [domain]
        if description:
            data["description"] = description

        return await self._client.request("POST", "/api/v1/routing-rules", json=data)

    async def get_stats(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Get routing statistics."""
        params: dict[str, Any] = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        raise NotImplementedError("Routing stats endpoint is not exposed via the public API")
