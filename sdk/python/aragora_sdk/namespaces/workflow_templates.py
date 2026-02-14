"""
Workflow Templates Namespace API

Provides access to pre-built workflow templates for common automation patterns:
- List available workflow templates
- Get template details
- Get full template packages with definitions and examples
- Run templates with inputs
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class WorkflowTemplatesAPI:
    """
    Synchronous Workflow Templates API.

    Provides access to pre-built workflow templates for common automation patterns.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> templates = client.workflow_templates.list(category="analysis")
        >>> for template in templates["templates"]:
        ...     print(template["name"], template["description"])
        >>> # Run a template
        >>> result = client.workflow_templates.run(
        ...     template_id="tpl_123",
        ...     inputs={"document": "contract.pdf"}
        ... )
        >>> print(result["execution_id"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(
        self,
        category: str | None = None,
        pattern: str | None = None,
        search: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List available workflow templates.

        Args:
            category: Filter by category (e.g., "analysis", "review", "automation")
            pattern: Filter by pattern type
            search: Search query for template name/description
            tags: Filter by tags
            limit: Maximum number of templates to return (default: 50)
            offset: Pagination offset (default: 0)

        Returns:
            Dictionary with 'templates' list and 'total' count
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if category:
            params["category"] = category
        if pattern:
            params["pattern"] = pattern
        if search:
            params["search"] = search
        if tags:
            params["tags"] = tags

        return self._client.request("GET", "/api/v1/workflow/templates", params=params)

    def get(self, template_id: str) -> dict[str, Any]:
        """
        Get a specific workflow template by ID.

        Args:
            template_id: The template ID

        Returns:
            Template details including id, name, description, category,
            pattern, tags, version, and timestamps
        """
        return self._client.request("GET", f"/api/v1/workflow/templates/{template_id}")

    def get_package(self, template_id: str) -> dict[str, Any]:
        """
        Get the full template package including definition and examples.

        Args:
            template_id: The template ID

        Returns:
            Template package including:
            - template: Basic template metadata
            - definition: Full workflow definition
            - dependencies: Required dependencies
            - examples: Usage examples with sample inputs
        """
        return self._client.request("GET", f"/api/v1/workflow/templates/{template_id}/package")


class AsyncWorkflowTemplatesAPI:
    """
    Asynchronous Workflow Templates API.

    Provides access to pre-built workflow templates for common automation patterns.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     templates = await client.workflow_templates.list(category="analysis")
        ...     for template in templates["templates"]:
        ...         print(template["name"], template["description"])
        ...     # Run a template
        ...     result = await client.workflow_templates.run(
        ...         template_id="tpl_123",
        ...         inputs={"document": "contract.pdf"}
        ...     )
        ...     print(result["execution_id"])
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(
        self,
        category: str | None = None,
        pattern: str | None = None,
        search: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List available workflow templates.

        Args:
            category: Filter by category (e.g., "analysis", "review", "automation")
            pattern: Filter by pattern type
            search: Search query for template name/description
            tags: Filter by tags
            limit: Maximum number of templates to return (default: 50)
            offset: Pagination offset (default: 0)

        Returns:
            Dictionary with 'templates' list and 'total' count
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if category:
            params["category"] = category
        if pattern:
            params["pattern"] = pattern
        if search:
            params["search"] = search
        if tags:
            params["tags"] = tags

        return await self._client.request("GET", "/api/v1/workflow/templates", params=params)

    async def get(self, template_id: str) -> dict[str, Any]:
        """
        Get a specific workflow template by ID.

        Args:
            template_id: The template ID

        Returns:
            Template details including id, name, description, category,
            pattern, tags, version, and timestamps
        """
        return await self._client.request("GET", f"/api/v1/workflow/templates/{template_id}")

    async def get_package(self, template_id: str) -> dict[str, Any]:
        """
        Get the full template package including definition and examples.

        Args:
            template_id: The template ID

        Returns:
            Template package including:
            - template: Basic template metadata
            - definition: Full workflow definition
            - dependencies: Required dependencies
            - examples: Usage examples with sample inputs
        """
        return await self._client.request(
            "GET", f"/api/v1/workflow/templates/{template_id}/package"
        )
