"""
Replays namespace for debate replay management.

Provides API access to replay debates, analyze historical runs,
and review agent interactions over time.
"""

from __future__ import annotations

from typing import Any, Literal

ReplayFormat = Literal["json", "markdown", "html"]


class ReplaysAPI:
    """Synchronous replays API."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def list(
        self,
        workspace_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List available replays.

        Args:
            workspace_id: Filter by workspace
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of replays with pagination
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if workspace_id:
            params["workspace_id"] = workspace_id

        return self._client._request("GET", "/api/v1/replays", params=params)

    def get(self, replay_id: str) -> dict[str, Any]:
        """
        Get a replay by ID.

        Args:
            replay_id: Replay identifier

        Returns:
            Replay details
        """
        return self._client._request("GET", f"/api/v1/replays/{replay_id}")

    def get_from_debate(self, debate_id: str) -> dict[str, Any]:
        """
        Get the replay for a debate.

        Args:
            debate_id: Debate identifier

        Returns:
            Replay data for the debate
        """
        return self._client._request("GET", f"/api/v1/debates/{debate_id}/replay")

    def get_events(
        self,
        replay_id: str,
        event_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Get replay events.

        Args:
            replay_id: Replay identifier
            event_type: Filter by event type
            limit: Maximum events
            offset: Pagination offset

        Returns:
            Replay events
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if event_type:
            params["event_type"] = event_type

        return self._client._request("GET", f"/api/v1/replays/{replay_id}/events", params=params)

    def export(
        self,
        replay_id: str,
        format: ReplayFormat = "json",
    ) -> dict[str, Any]:
        """
        Export a replay.

        Args:
            replay_id: Replay identifier
            format: Export format

        Returns:
            Exported replay data or download URL
        """
        params: dict[str, Any] = {"format": format}
        return self._client._request("GET", f"/api/v1/replays/{replay_id}/export", params=params)

    def get_summary(self, replay_id: str) -> dict[str, Any]:
        """
        Get replay summary with key moments.

        Args:
            replay_id: Replay identifier

        Returns:
            Summary with key moments and statistics
        """
        return self._client._request("GET", f"/api/v1/replays/{replay_id}/summary")

    def get_evolution(self, replay_id: str) -> dict[str, Any]:
        """
        Get the evolution of positions throughout the debate.

        Args:
            replay_id: Replay identifier

        Returns:
            Position evolution data
        """
        return self._client._request("GET", f"/api/v1/replays/{replay_id}/evolution")

    def fork(
        self,
        replay_id: str,
        from_event_index: int,
        new_input: str | None = None,
    ) -> dict[str, Any]:
        """
        Fork a replay from a specific point.

        Creates a new debate starting from a specific point in the replay,
        allowing exploration of alternative paths.

        Args:
            replay_id: Replay identifier
            from_event_index: Event index to fork from
            new_input: Optional new input for the fork

        Returns:
            Fork result with new debate_id
        """
        data: dict[str, Any] = {"from_event_index": from_event_index}
        if new_input:
            data["new_input"] = new_input

        return self._client._request("POST", f"/api/v1/replays/{replay_id}/fork", json=data)

    def compare(
        self,
        replay_id_1: str,
        replay_id_2: str,
    ) -> dict[str, Any]:
        """
        Compare two replays.

        Args:
            replay_id_1: First replay ID
            replay_id_2: Second replay ID

        Returns:
            Comparison analysis
        """
        return self._client._request(
            "GET",
            "/api/v1/replays/compare",
            params={"replay_id_1": replay_id_1, "replay_id_2": replay_id_2},
        )

    def delete(self, replay_id: str) -> dict[str, Any]:
        """
        Delete a replay.

        Args:
            replay_id: Replay identifier

        Returns:
            Deletion confirmation
        """
        return self._client._request("DELETE", f"/api/v1/replays/{replay_id}")


class AsyncReplaysAPI:
    """Asynchronous replays API."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def list(
        self,
        workspace_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List available replays."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if workspace_id:
            params["workspace_id"] = workspace_id

        return await self._client._request("GET", "/api/v1/replays", params=params)

    async def get(self, replay_id: str) -> dict[str, Any]:
        """Get a replay by ID."""
        return await self._client._request("GET", f"/api/v1/replays/{replay_id}")

    async def get_from_debate(self, debate_id: str) -> dict[str, Any]:
        """Get the replay for a debate."""
        return await self._client._request("GET", f"/api/v1/debates/{debate_id}/replay")

    async def get_events(
        self,
        replay_id: str,
        event_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get replay events."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if event_type:
            params["event_type"] = event_type

        return await self._client._request(
            "GET", f"/api/v1/replays/{replay_id}/events", params=params
        )

    async def export(
        self,
        replay_id: str,
        format: ReplayFormat = "json",
    ) -> dict[str, Any]:
        """Export a replay."""
        params: dict[str, Any] = {"format": format}
        return await self._client._request(
            "GET", f"/api/v1/replays/{replay_id}/export", params=params
        )

    async def get_summary(self, replay_id: str) -> dict[str, Any]:
        """Get replay summary with key moments."""
        return await self._client._request("GET", f"/api/v1/replays/{replay_id}/summary")

    async def get_evolution(self, replay_id: str) -> dict[str, Any]:
        """Get the evolution of positions throughout the debate."""
        return await self._client._request("GET", f"/api/v1/replays/{replay_id}/evolution")

    async def fork(
        self,
        replay_id: str,
        from_event_index: int,
        new_input: str | None = None,
    ) -> dict[str, Any]:
        """Fork a replay from a specific point."""
        data: dict[str, Any] = {"from_event_index": from_event_index}
        if new_input:
            data["new_input"] = new_input

        return await self._client._request("POST", f"/api/v1/replays/{replay_id}/fork", json=data)

    async def compare(
        self,
        replay_id_1: str,
        replay_id_2: str,
    ) -> dict[str, Any]:
        """Compare two replays."""
        return await self._client._request(
            "GET",
            "/api/v1/replays/compare",
            params={"replay_id_1": replay_id_1, "replay_id_2": replay_id_2},
        )

    async def delete(self, replay_id: str) -> dict[str, Any]:
        """Delete a replay."""
        return await self._client._request("DELETE", f"/api/v1/replays/{replay_id}")
