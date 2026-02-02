"""
Plugins Namespace API

Provides methods for plugin management:
- List and search plugins
- Install and run plugins
- Marketplace operations
- Plugin submissions
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class PluginsAPI:
    """
    Synchronous Plugins API.

    Provides methods for managing plugins that extend
    Aragora's functionality.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> plugins = client.plugins.list()
        >>> client.plugins.install("my-plugin")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Plugin Listing
    # ===========================================================================

    def list(self, category: str | None = None) -> dict[str, Any]:
        """
        List available plugins.

        Args:
            category: Filter by category

        Returns:
            Dict with plugins array
        """
        params: dict[str, Any] = {}
        if category:
            params["category"] = category

        return self._client.request("GET", "/api/v1/plugins", params=params)

    def list_installed(self) -> dict[str, Any]:
        """
        List installed plugins.

        Returns:
            Dict with installed plugins array
        """
        return self._client.request("GET", "/api/v1/plugins/installed")

    def get(self, name: str) -> dict[str, Any]:
        """
        Get details for a specific plugin.

        Args:
            name: Plugin name

        Returns:
            Plugin details
        """
        return self._client.request("GET", f"/api/v1/plugins/{name}")

    # ===========================================================================
    # Plugin Operations
    # ===========================================================================

    def install(self, name: str, version: str | None = None) -> dict[str, Any]:
        """
        Install a plugin.

        Args:
            name: Plugin name
            version: Version to install (latest if not specified)

        Returns:
            Installation result
        """
        data: dict[str, Any] = {}
        if version:
            data["version"] = version

        return self._client.request("POST", f"/api/v1/plugins/{name}/install", json=data)

    def uninstall(self, name: str) -> dict[str, Any]:
        """
        Uninstall a plugin.

        Args:
            name: Plugin name

        Returns:
            Uninstallation result
        """
        return self._client.request("DELETE", f"/api/v1/plugins/{name}/install")

    def run(
        self,
        name: str,
        action: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run a plugin action.

        Args:
            name: Plugin name
            action: Action to run (default action if not specified)
            params: Parameters for the action

        Returns:
            Action result
        """
        data: dict[str, Any] = {}
        if action:
            data["action"] = action
        if params:
            data["params"] = params

        return self._client.request("POST", f"/api/v1/plugins/{name}/run", json=data)

    # ===========================================================================
    # Marketplace
    # ===========================================================================

    def list_marketplace(
        self,
        category: str | None = None,
        sort: str = "popular",
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List plugins in the marketplace.

        Args:
            category: Filter by category
            sort: Sort order (popular, newest, updated)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            Dict with plugins array and total count
        """
        params: dict[str, Any] = {"sort": sort, "limit": limit, "offset": offset}
        if category:
            params["category"] = category

        return self._client.request("GET", "/api/v1/plugins/marketplace", params=params)

    # ===========================================================================
    # Plugin Submissions
    # ===========================================================================

    def submit(
        self,
        name: str,
        description: str,
        repository_url: str,
        category: str,
        version: str = "1.0.0",
    ) -> dict[str, Any]:
        """
        Submit a plugin to the marketplace.

        Args:
            name: Plugin name
            description: Plugin description
            repository_url: Git repository URL
            category: Plugin category
            version: Initial version

        Returns:
            Submission result with submission_id
        """
        data: dict[str, Any] = {
            "name": name,
            "description": description,
            "repository_url": repository_url,
            "category": category,
            "version": version,
        }

        return self._client.request("POST", "/api/v1/plugins/submit", json=data)

    def list_submissions(self, status: str | None = None) -> dict[str, Any]:
        """
        List your plugin submissions.

        Args:
            status: Filter by status (pending, approved, rejected)

        Returns:
            Dict with submissions array
        """
        params: dict[str, Any] = {}
        if status:
            params["status"] = status

        return self._client.request("GET", "/api/v1/plugins/submissions", params=params)


class AsyncPluginsAPI:
    """
    Asynchronous Plugins API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     plugins = await client.plugins.list()
        ...     await client.plugins.install("my-plugin")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # Plugin Listing
    async def list(self, category: str | None = None) -> dict[str, Any]:
        """List available plugins."""
        params: dict[str, Any] = {}
        if category:
            params["category"] = category

        return await self._client.request("GET", "/api/v1/plugins", params=params)

    async def list_installed(self) -> dict[str, Any]:
        """List installed plugins."""
        return await self._client.request("GET", "/api/v1/plugins/installed")

    async def get(self, name: str) -> dict[str, Any]:
        """Get details for a specific plugin."""
        return await self._client.request("GET", f"/api/v1/plugins/{name}")

    # Plugin Operations
    async def install(self, name: str, version: str | None = None) -> dict[str, Any]:
        """Install a plugin."""
        data: dict[str, Any] = {}
        if version:
            data["version"] = version

        return await self._client.request("POST", f"/api/v1/plugins/{name}/install", json=data)

    async def uninstall(self, name: str) -> dict[str, Any]:
        """Uninstall a plugin."""
        return await self._client.request("DELETE", f"/api/v1/plugins/{name}/install")

    async def run(
        self,
        name: str,
        action: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a plugin action."""
        data: dict[str, Any] = {}
        if action:
            data["action"] = action
        if params:
            data["params"] = params

        return await self._client.request("POST", f"/api/v1/plugins/{name}/run", json=data)

    # Marketplace
    async def list_marketplace(
        self,
        category: str | None = None,
        sort: str = "popular",
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List plugins in the marketplace."""
        params: dict[str, Any] = {"sort": sort, "limit": limit, "offset": offset}
        if category:
            params["category"] = category

        return await self._client.request("GET", "/api/v1/plugins/marketplace", params=params)

    # Plugin Submissions
    async def submit(
        self,
        name: str,
        description: str,
        repository_url: str,
        category: str,
        version: str = "1.0.0",
    ) -> dict[str, Any]:
        """Submit a plugin to the marketplace."""
        data: dict[str, Any] = {
            "name": name,
            "description": description,
            "repository_url": repository_url,
            "category": category,
            "version": version,
        }

        return await self._client.request("POST", "/api/v1/plugins/submit", json=data)

    async def list_submissions(self, status: str | None = None) -> dict[str, Any]:
        """List your plugin submissions."""
        params: dict[str, Any] = {}
        if status:
            params["status"] = status

        return await self._client.request("GET", "/api/v1/plugins/submissions", params=params)
