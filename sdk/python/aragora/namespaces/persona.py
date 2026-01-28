"""
Persona Namespace API.

Provides agent identity and persona management:
- List and manage agent personas
- Get grounded persona with ELO ratings
- Track performance and domain expertise
- Generate identity prompts
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

PersonaTrait = Literal[
    "analytical",
    "creative",
    "cautious",
    "bold",
    "collaborative",
    "independent",
    "detail_oriented",
    "big_picture",
    "evidence_focused",
    "intuitive",
]

ExpertiseDomain = Literal[
    "software",
    "legal",
    "healthcare",
    "finance",
    "research",
    "education",
    "marketing",
    "operations",
    "strategy",
    "general",
]

IdentitySection = Literal[
    "core",
    "traits",
    "expertise",
    "performance",
    "guidelines",
    "constraints",
]


class PersonaAPI:
    """
    Synchronous Persona API.

    Provides methods for managing agent personas:
    - List and create personas
    - Get grounded persona with ELO ratings
    - Track performance and domain expertise
    - Generate identity prompts

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> personas = client.persona.list()
        >>> grounded = client.persona.get_grounded("claude-analyst")
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def list(self) -> dict[str, Any]:
        """
        List all personas.

        Returns:
            List of personas with total count.
        """
        return self._client._request("GET", "/api/personas")

    def get_options(self) -> dict[str, Any]:
        """
        Get available persona options (traits, domains, models).

        Returns:
            Available traits, expertise domains, and model options.
        """
        return self._client._request("GET", "/api/personas/options")

    def get(self, agent_name: str) -> dict[str, Any]:
        """
        Get a persona by agent name.

        Args:
            agent_name: Name of the agent.

        Returns:
            Persona definition with traits and expertise.
        """
        return self._client._request("GET", f"/api/agent/{agent_name}/persona")

    def get_grounded(self, agent_name: str) -> dict[str, Any]:
        """
        Get grounded persona with performance metrics.

        Args:
            agent_name: Name of the agent.

        Returns:
            Grounded persona with ELO, domain ratings, and calibration.
        """
        return self._client._request("GET", f"/api/agent/{agent_name}/grounded-persona")

    def get_identity_prompt(
        self,
        agent_name: str,
        sections: list[IdentitySection] | None = None,
    ) -> dict[str, Any]:
        """
        Generate identity prompt for an agent.

        Args:
            agent_name: Name of the agent.
            sections: Sections to include (core, traits, expertise, etc.).

        Returns:
            Generated identity prompt with token count.
        """
        params: dict[str, Any] = {}
        if sections:
            params["sections"] = ",".join(sections)

        return self._client._request(
            "GET", f"/api/agent/{agent_name}/identity-prompt", params=params
        )

    def get_performance(self, agent_name: str) -> dict[str, Any]:
        """
        Get performance summary for an agent.

        Args:
            agent_name: Name of the agent.

        Returns:
            Performance metrics including wins, losses, streaks.
        """
        return self._client._request("GET", f"/api/agent/{agent_name}/performance")

    def get_domains(
        self,
        agent_name: str,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Get domain expertise for an agent.

        Args:
            agent_name: Name of the agent.
            limit: Maximum domains to return.

        Returns:
            List of domain expertise with ELO and win rates.
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit

        return self._client._request("GET", f"/api/agent/{agent_name}/domains", params=params)

    def get_accuracy(self, agent_name: str) -> dict[str, Any]:
        """
        Get position accuracy metrics for an agent.

        Args:
            agent_name: Name of the agent.

        Returns:
            Position accuracy with domain breakdown.
        """
        return self._client._request("GET", f"/api/agent/{agent_name}/accuracy")

    def create(
        self,
        agent_name: str,
        description: str | None = None,
        traits: list[PersonaTrait] | None = None,
        expertise: list[ExpertiseDomain] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """
        Create a new persona.

        Args:
            agent_name: Name for the new persona.
            description: Optional description.
            traits: Personality traits.
            expertise: Areas of expertise.
            model: Model to use.
            temperature: Response temperature.

        Returns:
            Created persona.
        """
        data: dict[str, Any] = {"agent_name": agent_name}
        if description:
            data["description"] = description
        if traits:
            data["traits"] = traits
        if expertise:
            data["expertise"] = expertise
        if model:
            data["model"] = model
        if temperature is not None:
            data["temperature"] = temperature

        return self._client._request("POST", "/api/personas", json=data)

    def update(
        self,
        agent_name: str,
        description: str | None = None,
        traits: list[PersonaTrait] | None = None,
        expertise: list[ExpertiseDomain] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """
        Update an existing persona.

        Args:
            agent_name: Name of the persona to update.
            description: New description.
            traits: New traits.
            expertise: New expertise areas.
            model: New model.
            temperature: New temperature.

        Returns:
            Updated persona.
        """
        data: dict[str, Any] = {}
        if description is not None:
            data["description"] = description
        if traits is not None:
            data["traits"] = traits
        if expertise is not None:
            data["expertise"] = expertise
        if model is not None:
            data["model"] = model
        if temperature is not None:
            data["temperature"] = temperature

        return self._client._request("PUT", f"/api/agent/{agent_name}/persona", json=data)

    def delete(self, agent_name: str) -> dict[str, Any]:
        """
        Delete a persona.

        Args:
            agent_name: Name of the persona to delete.

        Returns:
            Deletion confirmation.
        """
        return self._client._request("DELETE", f"/api/agent/{agent_name}/persona")


class AsyncPersonaAPI:
    """Asynchronous Persona API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def list(self) -> dict[str, Any]:
        """List all personas."""
        return await self._client._request("GET", "/api/personas")

    async def get_options(self) -> dict[str, Any]:
        """Get available persona options."""
        return await self._client._request("GET", "/api/personas/options")

    async def get(self, agent_name: str) -> dict[str, Any]:
        """Get a persona by agent name."""
        return await self._client._request("GET", f"/api/agent/{agent_name}/persona")

    async def get_grounded(self, agent_name: str) -> dict[str, Any]:
        """Get grounded persona with performance metrics."""
        return await self._client._request("GET", f"/api/agent/{agent_name}/grounded-persona")

    async def get_identity_prompt(
        self,
        agent_name: str,
        sections: list[IdentitySection] | None = None,
    ) -> dict[str, Any]:
        """Generate identity prompt for an agent."""
        params: dict[str, Any] = {}
        if sections:
            params["sections"] = ",".join(sections)

        return await self._client._request(
            "GET", f"/api/agent/{agent_name}/identity-prompt", params=params
        )

    async def get_performance(self, agent_name: str) -> dict[str, Any]:
        """Get performance summary for an agent."""
        return await self._client._request("GET", f"/api/agent/{agent_name}/performance")

    async def get_domains(
        self,
        agent_name: str,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Get domain expertise for an agent."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit

        return await self._client._request("GET", f"/api/agent/{agent_name}/domains", params=params)

    async def get_accuracy(self, agent_name: str) -> dict[str, Any]:
        """Get position accuracy metrics for an agent."""
        return await self._client._request("GET", f"/api/agent/{agent_name}/accuracy")

    async def create(
        self,
        agent_name: str,
        description: str | None = None,
        traits: list[PersonaTrait] | None = None,
        expertise: list[ExpertiseDomain] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Create a new persona."""
        data: dict[str, Any] = {"agent_name": agent_name}
        if description:
            data["description"] = description
        if traits:
            data["traits"] = traits
        if expertise:
            data["expertise"] = expertise
        if model:
            data["model"] = model
        if temperature is not None:
            data["temperature"] = temperature

        return await self._client._request("POST", "/api/personas", json=data)

    async def update(
        self,
        agent_name: str,
        description: str | None = None,
        traits: list[PersonaTrait] | None = None,
        expertise: list[ExpertiseDomain] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Update an existing persona."""
        data: dict[str, Any] = {}
        if description is not None:
            data["description"] = description
        if traits is not None:
            data["traits"] = traits
        if expertise is not None:
            data["expertise"] = expertise
        if model is not None:
            data["model"] = model
        if temperature is not None:
            data["temperature"] = temperature

        return await self._client._request("PUT", f"/api/agent/{agent_name}/persona", json=data)

    async def delete(self, agent_name: str) -> dict[str, Any]:
        """Delete a persona."""
        return await self._client._request("DELETE", f"/api/agent/{agent_name}/persona")
