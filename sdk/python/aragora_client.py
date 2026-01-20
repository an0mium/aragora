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
    # Batch Explainability
    # =========================================================================

    async def create_batch_explanation(
        self,
        debate_ids: List[str],
        include_evidence: bool = True,
        include_counterfactuals: bool = False,
        include_vote_pivots: bool = False,
        format: str = "full",
    ) -> Dict[str, Any]:
        """Create a batch explanation job for multiple debates.

        Args:
            debate_ids: List of debate IDs to process
            include_evidence: Include evidence chains
            include_counterfactuals: Include counterfactual analysis
            include_vote_pivots: Include vote pivot analysis
            format: Output format ('full', 'summary', or 'minimal')

        Returns:
            Batch job info with batch_id and status URLs
        """
        return await self._request(
            "POST",
            "/api/v1/explainability/batch",
            json={
                "debate_ids": debate_ids,
                "options": {
                    "include_evidence": include_evidence,
                    "include_counterfactuals": include_counterfactuals,
                    "include_vote_pivots": include_vote_pivots,
                    "format": format,
                },
            },
        )

    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get status of a batch explanation job."""
        return await self._request("GET", f"/api/v1/explainability/batch/{batch_id}/status")

    async def get_batch_results(
        self,
        batch_id: str,
        include_partial: bool = False,
        offset: int = 0,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Get results of a batch explanation job.

        Args:
            batch_id: The batch job ID
            include_partial: Include results even if job still processing
            offset: Pagination offset
            limit: Results per page (max 100)
        """
        return await self._request(
            "GET",
            f"/api/v1/explainability/batch/{batch_id}/results",
            params={"include_partial": include_partial, "offset": offset, "limit": limit},
        )

    async def compare_explanations(
        self,
        debate_ids: List[str],
        compare_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare explanations across multiple debates.

        Args:
            debate_ids: 2-10 debate IDs to compare
            compare_fields: Fields to compare (default: confidence, consensus_reached,
                          contributing_factors, evidence_quality)
        """
        body: Dict[str, Any] = {"debate_ids": debate_ids}
        if compare_fields:
            body["compare_fields"] = compare_fields
        return await self._request("POST", "/api/v1/explainability/compare", json=body)

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
    # Template Marketplace
    # =========================================================================

    async def browse_marketplace(
        self,
        category: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: str = "downloads",
        min_rating: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Browse community template marketplace.

        Args:
            category: Filter by category
            search: Search query
            sort_by: Sort by 'downloads', 'rating', or 'newest'
            min_rating: Minimum rating filter
            limit: Results per page
            offset: Pagination offset
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset, "sort_by": sort_by}
        if category:
            params["category"] = category
        if search:
            params["search"] = search
        if min_rating is not None:
            params["min_rating"] = min_rating
        return await self._request("GET", "/api/marketplace/templates", params=params)

    async def get_marketplace_template(self, template_id: str) -> Dict[str, Any]:
        """Get marketplace template details."""
        return await self._request("GET", f"/api/marketplace/templates/{template_id}")

    async def publish_template(
        self,
        template_id: str,
        name: str,
        description: str,
        category: str,
        tags: Optional[List[str]] = None,
        documentation: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Publish a template to the marketplace.

        Args:
            template_id: Source template ID from your workspace
            name: Display name for marketplace
            description: Template description
            category: Category (security, compliance, architecture, etc.)
            tags: Optional tags for discoverability
            documentation: Optional markdown documentation
        """
        body: Dict[str, Any] = {
            "template_id": template_id,
            "name": name,
            "description": description,
            "category": category,
        }
        if tags:
            body["tags"] = tags
        if documentation:
            body["documentation"] = documentation
        return await self._request("POST", "/api/marketplace/templates", json=body)

    async def rate_template(self, template_id: str, rating: int) -> Dict[str, Any]:
        """Rate a marketplace template (1-5 stars)."""
        return await self._request(
            "POST",
            f"/api/marketplace/templates/{template_id}/rate",
            json={"rating": rating},
        )

    async def review_template(
        self,
        template_id: str,
        rating: int,
        title: str,
        content: str,
    ) -> Dict[str, Any]:
        """Write a review for a marketplace template."""
        return await self._request(
            "POST",
            f"/api/marketplace/templates/{template_id}/review",
            json={"rating": rating, "title": title, "content": content},
        )

    async def import_template(
        self,
        template_id: str,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Import a marketplace template to your workspace."""
        body: Dict[str, Any] = {}
        if workspace_id:
            body["workspace_id"] = workspace_id
        return await self._request(
            "POST",
            f"/api/marketplace/templates/{template_id}/import",
            json=body,
        )

    async def get_featured_templates(self) -> Dict[str, Any]:
        """Get featured marketplace templates."""
        return await self._request("GET", "/api/marketplace/featured")

    async def get_trending_templates(self) -> Dict[str, Any]:
        """Get trending marketplace templates."""
        return await self._request("GET", "/api/marketplace/trending")

    async def get_marketplace_categories(self) -> Dict[str, Any]:
        """Get marketplace categories with counts."""
        return await self._request("GET", "/api/marketplace/categories")

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

    # Batch Explainability
    def create_batch_explanation(self, debate_ids: List[str], **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.create_batch_explanation(debate_ids, **kwargs))

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_batch_status(batch_id))

    def get_batch_results(self, batch_id: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.get_batch_results(batch_id, **kwargs))

    def compare_explanations(self, debate_ids: List[str], **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.compare_explanations(debate_ids, **kwargs))

    # Workflow Templates
    def list_workflow_templates(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.list_workflow_templates(**kwargs))

    # Marketplace
    def browse_marketplace(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.browse_marketplace(**kwargs))

    def get_marketplace_template(self, template_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_marketplace_template(template_id))

    def publish_template(self, template_id: str, name: str, description: str, category: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.publish_template(template_id, name, description, category, **kwargs))

    def rate_template(self, template_id: str, rating: int) -> Dict[str, Any]:
        return self._run(self._async_client.rate_template(template_id, rating))

    def import_template(self, template_id: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.import_template(template_id, **kwargs))

    def get_featured_templates(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_featured_templates())

    def get_trending_templates(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_trending_templates())

    def health(self) -> Dict[str, Any]:
        return self._run(self._async_client.health())
