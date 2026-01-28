"""
History Namespace API

Provides access to historical data including debate history,
nomic cycles, and system events.

Features:
- List historical debates
- View nomic improvement cycles
- Track system events
- Get history summaries
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


HistoryPeriod = Literal["week", "month", "year", "all"]
EventSeverity = Literal["info", "warning", "error"]


class HistoryAPI:
    """
    Synchronous History API.

    Provides access to historical data:
    - List historical debates
    - View nomic improvement cycles
    - Track system events
    - Get history summaries

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="...")
        >>> debates = client.history.list_debates(limit=10)
        >>> summary = client.history.get_summary()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def list_debates(
        self,
        limit: int | None = None,
        offset: int | None = None,
        since: str | None = None,
        until: str | None = None,
        status: str | None = None,
        agent: str | None = None,
        domain: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List historical debates.

        Args:
            limit: Maximum number of debates
            offset: Number of debates to skip
            since: Start date (ISO 8601)
            until: End date (ISO 8601)
            status: Filter by status
            agent: Filter by agent
            domain: Filter by domain

        Returns:
            List of historical debate records
        """
        params: dict[str, Any] = {}
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        if status:
            params["status"] = status
        if agent:
            params["agent"] = agent
        if domain:
            params["domain"] = domain
        response = self._client.request(
            "GET", "/api/v1/history/debates", params=params if params else None
        )
        return response.get("debates", [])

    def list_cycles(
        self,
        limit: int | None = None,
        offset: int | None = None,
        since: str | None = None,
        until: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List nomic improvement cycles.

        Args:
            limit: Maximum number of cycles
            offset: Number of cycles to skip
            since: Start date (ISO 8601)
            until: End date (ISO 8601)
            status: Filter by status

        Returns:
            List of nomic cycle records
        """
        params: dict[str, Any] = {}
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        if status:
            params["status"] = status
        response = self._client.request(
            "GET", "/api/v1/history/cycles", params=params if params else None
        )
        return response.get("cycles", [])

    def list_events(
        self,
        limit: int | None = None,
        offset: int | None = None,
        since: str | None = None,
        until: str | None = None,
        event_type: str | None = None,
        severity: EventSeverity | None = None,
    ) -> list[dict[str, Any]]:
        """
        List system events.

        Args:
            limit: Maximum number of events
            offset: Number of events to skip
            since: Start date (ISO 8601)
            until: End date (ISO 8601)
            event_type: Filter by event type
            severity: Filter by severity

        Returns:
            List of system event records
        """
        params: dict[str, Any] = {}
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        if event_type:
            params["event_type"] = event_type
        if severity:
            params["severity"] = severity
        response = self._client.request(
            "GET", "/api/v1/history/events", params=params if params else None
        )
        return response.get("events", [])

    def get_summary(self, period: HistoryPeriod | None = None) -> dict[str, Any]:
        """
        Get a summary of historical activity.

        Args:
            period: Time period for summary

        Returns:
            Dict with:
            - total_debates: Total debate count
            - debates_this_week: Debates this week
            - debates_this_month: Debates this month
            - total_nomic_cycles: Nomic cycle count
            - active_agents: Active agent count
            - most_active_agent: Top agent
            - top_domains: Most debated domains
            - consensus_rate: Consensus success rate
            - average_debate_duration_minutes: Average duration
        """
        params = {"period": period} if period else None
        return self._client.request("GET", "/api/v1/history/summary", params=params)

    def get_agent_history(
        self,
        agent_name: str,
        limit: int | None = None,
        offset: int | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get history for a specific agent.

        Args:
            agent_name: The agent name
            limit: Maximum number of debates
            offset: Number to skip
            since: Start date
            until: End date

        Returns:
            List of debates involving the agent
        """
        params: dict[str, Any] = {}
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        response = self._client.request(
            "GET",
            f"/api/v1/agent/{agent_name}/history",
            params=params if params else None,
        )
        return response.get("debates", [])


class AsyncHistoryAPI:
    """
    Asynchronous History API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     debates = await client.history.list_debates(limit=10)
        ...     summary = await client.history.get_summary()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_debates(
        self,
        limit: int | None = None,
        offset: int | None = None,
        since: str | None = None,
        until: str | None = None,
        status: str | None = None,
        agent: str | None = None,
        domain: str | None = None,
    ) -> list[dict[str, Any]]:
        """List historical debates."""
        params: dict[str, Any] = {}
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        if status:
            params["status"] = status
        if agent:
            params["agent"] = agent
        if domain:
            params["domain"] = domain
        response = await self._client.request(
            "GET", "/api/v1/history/debates", params=params if params else None
        )
        return response.get("debates", [])

    async def list_cycles(
        self,
        limit: int | None = None,
        offset: int | None = None,
        since: str | None = None,
        until: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List nomic improvement cycles."""
        params: dict[str, Any] = {}
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        if status:
            params["status"] = status
        response = await self._client.request(
            "GET", "/api/v1/history/cycles", params=params if params else None
        )
        return response.get("cycles", [])

    async def list_events(
        self,
        limit: int | None = None,
        offset: int | None = None,
        since: str | None = None,
        until: str | None = None,
        event_type: str | None = None,
        severity: EventSeverity | None = None,
    ) -> list[dict[str, Any]]:
        """List system events."""
        params: dict[str, Any] = {}
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        if event_type:
            params["event_type"] = event_type
        if severity:
            params["severity"] = severity
        response = await self._client.request(
            "GET", "/api/v1/history/events", params=params if params else None
        )
        return response.get("events", [])

    async def get_summary(self, period: HistoryPeriod | None = None) -> dict[str, Any]:
        """Get a summary of historical activity."""
        params = {"period": period} if period else None
        return await self._client.request("GET", "/api/v1/history/summary", params=params)

    async def get_agent_history(
        self,
        agent_name: str,
        limit: int | None = None,
        offset: int | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get history for a specific agent."""
        params: dict[str, Any] = {}
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        response = await self._client.request(
            "GET",
            f"/api/v1/agent/{agent_name}/history",
            params=params if params else None,
        )
        return response.get("debates", [])
