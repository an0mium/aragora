"""
Marketplace Client for Aragora.

Provides connectivity to remote marketplace servers for
sharing and discovering templates across the community.
"""

from dataclasses import dataclass
from typing import Any, Optional, Union

import aiohttp

from .models import (
    AgentTemplate,
    DebateTemplate,
    WorkflowTemplate,
    TemplateCategory,
    TemplateMetadata,
)


@dataclass
class MarketplaceConfig:
    """Configuration for the marketplace client."""

    base_url: str = "https://marketplace.aragora.ai/api/v1"
    api_key: Optional[str] = None
    timeout: float = 30.0


class MarketplaceClient:
    """Client for the Aragora template marketplace."""

    def __init__(self, config: Optional[MarketplaceConfig] = None):
        """Initialize the marketplace client."""
        self.config = config or MarketplaceConfig()
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an HTTP session."""
        if self._session is None or self._session.closed:
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "MarketplaceClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Make an HTTP request to the marketplace API."""
        session = await self._get_session()
        url = f"{self.config.base_url}{endpoint}"

        async with session.request(method, url, json=data, params=params) as resp:
            if resp.status >= 400:
                error = await resp.text()
                raise MarketplaceError(f"API error {resp.status}: {error}")
            return await resp.json()

    # Template Operations

    async def search_templates(
        self,
        query: Optional[str] = None,
        category: Optional[TemplateCategory] = None,
        template_type: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Search for templates in the marketplace."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if query:
            params["q"] = query
        if category:
            params["category"] = category.value
        if template_type:
            params["type"] = template_type
        if tags:
            params["tags"] = ",".join(tags)

        result = await self._request("GET", "/templates", params=params)
        return result.get("templates", [])

    async def get_template(self, template_id: str) -> dict[str, Any]:
        """Get a template by ID."""
        return await self._request("GET", f"/templates/{template_id}")

    async def publish_template(
        self,
        template: Union[AgentTemplate, DebateTemplate, WorkflowTemplate],
    ) -> dict[str, Any]:
        """Publish a template to the marketplace."""
        return await self._request("POST", "/templates", data=template.to_dict())

    async def update_template(
        self,
        template_id: str,
        template: Union[AgentTemplate, DebateTemplate, WorkflowTemplate],
    ) -> dict[str, Any]:
        """Update an existing template."""
        return await self._request("PUT", f"/templates/{template_id}", data=template.to_dict())

    async def delete_template(self, template_id: str) -> dict[str, Any]:
        """Delete a template from the marketplace."""
        return await self._request("DELETE", f"/templates/{template_id}")

    async def download_template(
        self, template_id: str
    ) -> Union[AgentTemplate, DebateTemplate, WorkflowTemplate]:
        """Download a template and convert to local object."""
        data = await self.get_template(template_id)
        return self._data_to_template(data)

    def _data_to_template(
        self, data: dict[str, Any]
    ) -> Union[AgentTemplate, DebateTemplate, WorkflowTemplate]:
        """Convert API data to a template object."""
        metadata = TemplateMetadata(
            id=data["metadata"]["id"],
            name=data["metadata"]["name"],
            description=data["metadata"]["description"],
            version=data["metadata"]["version"],
            author=data["metadata"]["author"],
            category=TemplateCategory(data["metadata"]["category"]),
            tags=data["metadata"].get("tags", []),
            downloads=data["metadata"].get("downloads", 0),
            stars=data["metadata"].get("stars", 0),
        )

        if "agent_type" in data:
            return AgentTemplate(
                metadata=metadata,
                agent_type=data["agent_type"],
                system_prompt=data["system_prompt"],
                model_config=data.get("model_config", {}),
                capabilities=data.get("capabilities", []),
                constraints=data.get("constraints", []),
                examples=data.get("examples", []),
            )
        elif "task_template" in data:
            return DebateTemplate(
                metadata=metadata,
                task_template=data["task_template"],
                agent_roles=data["agent_roles"],
                protocol=data["protocol"],
                evaluation_criteria=data.get("evaluation_criteria", []),
                success_metrics=data.get("success_metrics", {}),
            )
        elif "nodes" in data:
            return WorkflowTemplate(
                metadata=metadata,
                nodes=data["nodes"],
                edges=data["edges"],
                inputs=data.get("inputs", {}),
                outputs=data.get("outputs", {}),
                variables=data.get("variables", {}),
            )
        else:
            raise ValueError("Unknown template format")

    # Rating Operations

    async def rate_template(
        self,
        template_id: str,
        score: int,
        review: Optional[str] = None,
    ) -> dict[str, Any]:
        """Rate a template."""
        data = {"score": score}
        if review:
            data["review"] = review
        return await self._request("POST", f"/templates/{template_id}/ratings", data=data)

    async def get_ratings(self, template_id: str) -> list[dict[str, Any]]:
        """Get ratings for a template."""
        result = await self._request("GET", f"/templates/{template_id}/ratings")
        return result.get("ratings", [])

    # Star Operations

    async def star_template(self, template_id: str) -> dict[str, Any]:
        """Star a template."""
        return await self._request("POST", f"/templates/{template_id}/star")

    async def unstar_template(self, template_id: str) -> dict[str, Any]:
        """Remove star from a template."""
        return await self._request("DELETE", f"/templates/{template_id}/star")

    # Category Operations

    async def list_categories(self) -> list[dict[str, Any]]:
        """List all categories with counts."""
        result = await self._request("GET", "/categories")
        return result.get("categories", [])

    # Featured/Popular

    async def get_featured(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get featured templates."""
        result = await self._request("GET", "/templates/featured", params={"limit": limit})
        return result.get("templates", [])

    async def get_popular(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get most popular templates."""
        result = await self._request(
            "GET", "/templates", params={"limit": limit, "sort": "downloads"}
        )
        return result.get("templates", [])

    async def get_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get most recent templates."""
        result = await self._request(
            "GET", "/templates", params={"limit": limit, "sort": "created_at"}
        )
        return result.get("templates", [])

    # User Operations

    async def get_my_templates(self) -> list[dict[str, Any]]:
        """Get templates published by the current user."""
        result = await self._request("GET", "/users/me/templates")
        return result.get("templates", [])

    async def get_my_starred(self) -> list[dict[str, Any]]:
        """Get templates starred by the current user."""
        result = await self._request("GET", "/users/me/starred")
        return result.get("templates", [])


class MarketplaceError(Exception):
    """Error from marketplace operations."""

    pass
