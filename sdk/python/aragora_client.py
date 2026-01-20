"""
Auto-generated Python client for Aragora API.

Do not edit manually.

Usage:
    from aragora_client import AragoraClient

    client = AragoraClient(base_url="http://localhost:8080", api_key="your-key")
    debates = await client.list_debates()
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore


@dataclass
class ApiConfig:
    """Configuration for the API client."""
    base_url: str
    api_key: Optional[str] = None
    timeout: float = 30.0
    headers: Optional[Dict[str, str]] = None


class AragoraClient:
    """Async client for the Aragora API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        if httpx is None:
            raise ImportError("httpx is required for AragoraClient. Install with: pip install httpx")

        self.config = ApiConfig(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=timeout,
            headers=headers or {},
        )
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "AragoraClient":
        await self._ensure_client()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json", **self.config.headers}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=self.config.timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Any:
        client = await self._ensure_client()
        response = await client.request(method, path, params=params, json=json)
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Debates
    # =========================================================================

    async def list_debates(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List debates."""
        return await self._request("GET", "/api/debates", params={"limit": limit, "offset": offset})

    async def get_debate(self, debate_id: str) -> Dict[str, Any]:
        """Get a specific debate."""
        return await self._request("GET", f"/api/debates/{debate_id}")

    async def create_debate(
        self,
        question: str,
        agents: List[str],
        rounds: int = 3,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a new debate."""
        return await self._request(
            "POST",
            "/api/debate",
            json={"question": question, "agents": agents, "rounds": rounds, **kwargs},
        )

    # =========================================================================
    # Explainability
    # =========================================================================

    async def get_explanation(
        self,
        debate_id: str,
        include_factors: bool = True,
        include_counterfactuals: bool = True,
        include_provenance: bool = True,
    ) -> Dict[str, Any]:
        """Get full explanation for a debate decision."""
        return await self._request(
            "GET",
            f"/api/debates/{debate_id}/explainability",
            params={
                "include_factors": include_factors,
                "include_counterfactuals": include_counterfactuals,
                "include_provenance": include_provenance,
            },
        )

    async def get_factors(
        self,
        debate_id: str,
        min_contribution: Optional[float] = None,
        sort_by: str = "contribution",
    ) -> Dict[str, Any]:
        """Get contributing factors for a debate decision."""
        params = {"sort_by": sort_by}
        if min_contribution is not None:
            params["min_contribution"] = min_contribution
        return await self._request("GET", f"/api/debates/{debate_id}/explainability/factors", params=params)

    async def get_counterfactuals(
        self,
        debate_id: str,
        max_scenarios: int = 5,
        min_probability: float = 0.3,
    ) -> Dict[str, Any]:
        """Get counterfactual scenarios for a debate."""
        return await self._request(
            "GET",
            f"/api/debates/{debate_id}/explainability/counterfactual",
            params={"max_scenarios": max_scenarios, "min_probability": min_probability},
        )

    async def generate_counterfactual(
        self,
        debate_id: str,
        hypothesis: str,
        affected_agents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate a custom counterfactual scenario."""
        body = {"hypothesis": hypothesis}
        if affected_agents:
            body["affected_agents"] = affected_agents
        return await self._request("POST", f"/api/debates/{debate_id}/explainability/counterfactual", json=body)

    async def get_provenance(
        self,
        debate_id: str,
        include_timestamps: bool = True,
        include_agents: bool = True,
        include_confidence: bool = True,
    ) -> Dict[str, Any]:
        """Get decision provenance chain."""
        return await self._request(
            "GET",
            f"/api/debates/{debate_id}/explainability/provenance",
            params={
                "include_timestamps": include_timestamps,
                "include_agents": include_agents,
                "include_confidence": include_confidence,
            },
        )

    async def get_narrative(
        self,
        debate_id: str,
        format: str = "detailed",
        language: str = "en",
    ) -> Dict[str, Any]:
        """Get natural language narrative explanation."""
        return await self._request(
            "GET",
            f"/api/debates/{debate_id}/explainability/narrative",
            params={"format": format, "language": language},
        )

    # =========================================================================
    # Workflow Templates
    # =========================================================================

    async def list_workflow_templates(
        self,
        category: Optional[str] = None,
        pattern: Optional[str] = None,
        search: Optional[str] = None,
        tags: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List workflow templates."""
        params = {"limit": limit, "offset": offset}
        if category:
            params["category"] = category
        if pattern:
            params["pattern"] = pattern
        if search:
            params["search"] = search
        if tags:
            params["tags"] = tags
        return await self._request("GET", "/api/workflow/templates", params=params)

    async def get_workflow_template(self, template_id: str) -> Dict[str, Any]:
        """Get workflow template details."""
        return await self._request("GET", f"/api/workflow/templates/{template_id}")

    async def get_workflow_template_package(
        self,
        template_id: str,
        include_examples: bool = True,
    ) -> Dict[str, Any]:
        """Get full workflow template package."""
        return await self._request(
            "GET",
            f"/api/workflow/templates/{template_id}/package",
            params={"include_examples": include_examples},
        )

    async def run_workflow_template(
        self,
        template_id: str,
        inputs: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a workflow template."""
        body: Dict[str, Any] = {}
        if inputs:
            body["inputs"] = inputs
        if config:
            body["config"] = config
        if workspace_id:
            body["workspace_id"] = workspace_id
        return await self._request("POST", f"/api/workflow/templates/{template_id}/run", json=body)

    async def list_workflow_categories(self) -> Dict[str, Any]:
        """List workflow template categories."""
        return await self._request("GET", "/api/workflow/categories")

    async def list_workflow_patterns(self) -> Dict[str, Any]:
        """List workflow patterns."""
        return await self._request("GET", "/api/workflow/patterns")

    async def instantiate_pattern(
        self,
        pattern_id: str,
        name: str,
        description: str,
        category: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        agents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a template from a workflow pattern."""
        body = {"name": name, "description": description}
        if category:
            body["category"] = category
        if config:
            body["config"] = config
        if agents:
            body["agents"] = agents
        return await self._request("POST", f"/api/workflow/patterns/{pattern_id}/instantiate", json=body)

    # =========================================================================
    # Gauntlet
    # =========================================================================

    async def list_gauntlet_receipts(
        self,
        verdict: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List gauntlet receipts."""
        params = {"limit": limit, "offset": offset}
        if verdict:
            params["verdict"] = verdict
        return await self._request("GET", "/api/gauntlet/receipts", params=params)

    async def get_gauntlet_receipt(self, receipt_id: str) -> Dict[str, Any]:
        """Get a specific gauntlet receipt."""
        return await self._request("GET", f"/api/gauntlet/receipts/{receipt_id}")

    async def verify_gauntlet_receipt(self, receipt_id: str) -> Dict[str, Any]:
        """Verify receipt integrity."""
        return await self._request("GET", f"/api/gauntlet/receipts/{receipt_id}/verify")

    async def export_gauntlet_receipt(
        self,
        receipt_id: str,
        format: str = "json",
    ) -> Dict[str, Any]:
        """Export receipt in specified format."""
        return await self._request(
            "GET",
            f"/api/gauntlet/receipts/{receipt_id}/export",
            params={"format": format},
        )

    # =========================================================================
    # Agents
    # =========================================================================

    async def list_agents(self) -> Dict[str, Any]:
        """List available agents."""
        return await self._request("GET", "/api/agents")

    async def get_agent(self, agent_name: str) -> Dict[str, Any]:
        """Get agent details."""
        return await self._request("GET", f"/api/agents/{agent_name}")

    # =========================================================================
    # Health
    # =========================================================================

    async def health(self) -> Dict[str, Any]:
        """Check API health."""
        return await self._request("GET", "/api/health")


# Sync wrapper for convenience
class AragoraClientSync:
    """Synchronous wrapper for AragoraClient."""

    def __init__(self, *args: Any, **kwargs: Any):
        self._async_client = AragoraClient(*args, **kwargs)

    def _run(self, coro: Any) -> Any:
        return asyncio.get_event_loop().run_until_complete(coro)

    def close(self) -> None:
        self._run(self._async_client.close())

    def list_debates(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.list_debates(**kwargs))

    def get_debate(self, debate_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_debate(debate_id))

    def create_debate(self, question: str, agents: List[str], **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.create_debate(question, agents, **kwargs))

    def get_explanation(self, debate_id: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.get_explanation(debate_id, **kwargs))

    def list_workflow_templates(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.list_workflow_templates(**kwargs))

    def health(self) -> Dict[str, Any]:
        return self._run(self._async_client.health())
