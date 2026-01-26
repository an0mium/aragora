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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.streaming import AragoraWebSocket

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
            raise ImportError(
                "httpx is required for AragoraClient. Install with: pip install httpx"
            )

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
        return await self._request(
            "GET", f"/api/debates/{debate_id}/explainability/factors", params=params
        )

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
        return await self._request(
            "POST", f"/api/debates/{debate_id}/explainability/counterfactual", json=body
        )

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
        return await self._request(
            "POST", f"/api/workflow/patterns/{pattern_id}/instantiate", json=body
        )

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

    async def run_gauntlet(
        self,
        input: str,
        profile: str = "comprehensive",
        agents: Optional[List[str]] = None,
        rounds: int = 3,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a gauntlet evaluation.

        Args:
            input: The input to evaluate
            profile: Evaluation profile ('comprehensive', 'quick', 'deep')
            agents: Optional list of agents to use
            rounds: Number of debate rounds
            timeout: Optional timeout in seconds
        """
        body: Dict[str, Any] = {
            "input": input,
            "profile": profile,
            "rounds": rounds,
        }
        if agents:
            body["agents"] = agents
        if timeout:
            body["timeout"] = timeout
        return await self._request("POST", "/api/v1/gauntlet/run", json=body)

    async def get_gauntlet_status(self, gauntlet_id: str) -> Dict[str, Any]:
        """Get status of a running gauntlet."""
        return await self._request("GET", f"/api/v1/gauntlet/{gauntlet_id}")

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

    async def delete_gauntlet(self, gauntlet_id: str) -> Dict[str, Any]:
        """Delete a gauntlet run.

        Args:
            gauntlet_id: The gauntlet run ID to delete

        Returns:
            Deletion confirmation
        """
        return await self._request("DELETE", f"/api/v1/gauntlet/{gauntlet_id}")

    async def list_gauntlet_personas(
        self,
        category: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """List available gauntlet personas.

        Args:
            category: Filter by category ('adversarial', 'compliance', etc.)
            enabled: Filter by enabled status

        Returns:
            List of available personas with their configurations
        """
        params: Dict[str, Any] = {}
        if category:
            params["category"] = category
        if enabled is not None:
            params["enabled"] = enabled
        return await self._request(
            "GET",
            "/api/v1/gauntlet/personas",
            params=params if params else None,
        )

    async def list_gauntlet_results(
        self,
        gauntlet_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List gauntlet run results.

        Args:
            gauntlet_id: Filter by specific gauntlet ID
            status: Filter by status ('completed', 'failed', etc.)
            limit: Results per page
            offset: Pagination offset

        Returns:
            List of gauntlet results with summaries
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if gauntlet_id:
            params["gauntlet_id"] = gauntlet_id
        if status:
            params["status"] = status
        return await self._request("GET", "/api/v1/gauntlet/results", params=params)

    async def get_gauntlet_heatmap(
        self,
        gauntlet_id: str,
        format: str = "json",
    ) -> Dict[str, Any]:
        """Get heatmap visualization data for a gauntlet run.

        Args:
            gauntlet_id: The gauntlet run ID
            format: Output format ('json' or 'svg')

        Returns:
            Heatmap data showing agent vs persona performance matrix
        """
        return await self._request(
            "GET",
            f"/api/v1/gauntlet/{gauntlet_id}/heatmap",
            params={"format": format},
        )

    async def compare_gauntlets(
        self,
        gauntlet_id_1: str,
        gauntlet_id_2: str,
    ) -> Dict[str, Any]:
        """Compare two gauntlet runs.

        Args:
            gauntlet_id_1: First gauntlet run ID
            gauntlet_id_2: Second gauntlet run ID

        Returns:
            Comparison analysis with differences highlighted
        """
        return await self._request(
            "GET",
            f"/api/v1/gauntlet/{gauntlet_id_1}/compare/{gauntlet_id_2}",
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

    async def get_agent_calibration(self, agent_name: str) -> Dict[str, Any]:
        """Get agent calibration metrics."""
        return await self._request("GET", f"/api/agents/{agent_name}/calibration")

    async def get_agent_performance(
        self,
        agent_name: str,
        timeframe: str = "30d",
    ) -> Dict[str, Any]:
        """Get agent performance metrics over time.

        Args:
            agent_name: Agent identifier
            timeframe: Timeframe ('7d', '30d', '90d', 'all')
        """
        return await self._request(
            "GET",
            f"/api/agents/{agent_name}/performance",
            params={"timeframe": timeframe},
        )

    async def get_agent_head_to_head(
        self,
        agent_name: str,
        opponent_name: str,
    ) -> Dict[str, Any]:
        """Get head-to-head statistics between two agents."""
        return await self._request(
            "GET",
            f"/api/agents/{agent_name}/head-to-head/{opponent_name}",
        )

    async def get_agent_network(self, agent_name: str) -> Dict[str, Any]:
        """Get agent's interaction network."""
        return await self._request("GET", f"/api/agents/{agent_name}/network")

    async def get_agent_positions(
        self,
        agent_name: str,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get agent's position history on topics.

        Args:
            agent_name: Agent identifier
            domain: Optional domain filter
        """
        params = {}
        if domain:
            params["domain"] = domain
        return await self._request(
            "GET",
            f"/api/agents/{agent_name}/positions",
            params=params if params else None,
        )

    async def get_agent_domains(self, agent_name: str) -> Dict[str, Any]:
        """Get agent's domain expertise ratings."""
        return await self._request("GET", f"/api/agents/{agent_name}/domains")

    async def get_agent_consistency(self, agent_name: str) -> Dict[str, Any]:
        """Get agent's position consistency metrics.

        Returns metrics on how consistently the agent maintains positions
        across similar topics and debates.
        """
        return await self._request("GET", f"/api/v1/agent/{agent_name}/consistency")

    async def get_agent_flips(
        self,
        agent_name: str,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get instances where agent changed positions.

        Args:
            agent_name: Agent identifier
            limit: Results per page
            offset: Pagination offset

        Returns:
            List of position flips with context and reasoning
        """
        return await self._request(
            "GET",
            f"/api/v1/agent/{agent_name}/flips",
            params={"limit": limit, "offset": offset},
        )

    async def get_agent_moments(
        self,
        agent_name: str,
        moment_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get notable moments from agent's debate history.

        Args:
            agent_name: Agent identifier
            moment_type: Filter by type ('breakthrough', 'consensus', 'flip', etc.)
            limit: Results per page
            offset: Pagination offset

        Returns:
            List of notable moments with context
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if moment_type:
            params["type"] = moment_type
        return await self._request(
            "GET",
            f"/api/v1/agent/{agent_name}/moments",
            params=params,
        )

    async def get_agent_opponent_briefing(
        self,
        agent_name: str,
        opponent_name: str,
    ) -> Dict[str, Any]:
        """Get tactical briefing for debating against a specific opponent.

        Args:
            agent_name: Agent requesting the briefing
            opponent_name: Target opponent agent

        Returns:
            Briefing with opponent's tendencies, weaknesses, and recommended strategies
        """
        return await self._request(
            "GET",
            f"/api/v1/agent/{agent_name}/opponent-briefing/{opponent_name}",
        )

    async def get_leaderboard(self) -> Dict[str, Any]:
        """Get agent leaderboard rankings."""
        return await self._request("GET", "/api/leaderboard")

    # =========================================================================
    # Health
    # =========================================================================

    async def health(self) -> Dict[str, Any]:
        """Check API health."""
        return await self._request("GET", "/api/health")

    # =========================================================================
    # Streaming / WebSocket
    # =========================================================================

    def create_websocket(
        self,
        ws_url: Optional[str] = None,
        options: Optional[Any] = None,
    ) -> "AragoraWebSocket":
        """Create a WebSocket client for real-time streaming.

        Args:
            ws_url: Optional WebSocket URL (auto-derived from base_url if not provided)
            options: Optional WebSocketOptions for connection settings

        Returns:
            AragoraWebSocket instance

        Example:
            ```python
            ws = client.create_websocket()
            await ws.connect()

            async for event in ws.stream_events(debate_id="debate-123"):
                if event.type == "agent_message":
                    print(event.data)
            ```
        """
        from aragora.streaming import AragoraWebSocket

        return AragoraWebSocket(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            ws_url=ws_url,
            options=options,
        )

    # =========================================================================
    # Control Plane - Agent Registry
    # =========================================================================

    async def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register an agent with the control plane.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (e.g., 'claude', 'gpt-4')
            capabilities: List of agent capabilities
            metadata: Additional agent metadata
        """
        body: Dict[str, Any] = {"agent_id": agent_id, "agent_type": agent_type}
        if capabilities:
            body["capabilities"] = capabilities
        if metadata:
            body["metadata"] = metadata
        return await self._request("POST", "/api/control-plane/agents/register", json=body)

    async def deregister_agent(self, agent_id: str) -> Dict[str, Any]:
        """Deregister an agent from the control plane."""
        return await self._request("POST", f"/api/control-plane/agents/{agent_id}/deregister")

    async def list_registered_agents(
        self,
        agent_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List all registered agents."""
        params: Dict[str, Any] = {}
        if agent_type:
            params["agent_type"] = agent_type
        if status:
            params["status"] = status
        return await self._request("GET", "/api/control-plane/agents", params=params)

    async def get_agent_health(self, agent_id: str) -> Dict[str, Any]:
        """Get health status of a registered agent."""
        return await self._request("GET", f"/api/control-plane/agents/{agent_id}/health")

    async def send_agent_heartbeat(
        self,
        agent_id: str,
        status: str = "healthy",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a heartbeat for an agent."""
        body: Dict[str, Any] = {"status": status}
        if metrics:
            body["metrics"] = metrics
        return await self._request(
            "POST", f"/api/control-plane/agents/{agent_id}/heartbeat", json=body
        )

    # =========================================================================
    # Control Plane - Task Scheduler
    # =========================================================================

    async def schedule_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 5,
        scheduled_at: Optional[str] = None,
        agent_constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Schedule a task for execution.

        Args:
            task_type: Type of task to schedule
            payload: Task payload/parameters
            priority: Task priority (1-10, higher = more urgent)
            scheduled_at: Optional ISO timestamp for delayed execution
            agent_constraints: Optional constraints for agent selection
        """
        body: Dict[str, Any] = {"task_type": task_type, "payload": payload, "priority": priority}
        if scheduled_at:
            body["scheduled_at"] = scheduled_at
        if agent_constraints:
            body["agent_constraints"] = agent_constraints
        return await self._request("POST", "/api/control-plane/tasks/schedule", json=body)

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a scheduled task."""
        return await self._request("GET", f"/api/control-plane/tasks/{task_id}")

    async def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """Cancel a scheduled task."""
        return await self._request("POST", f"/api/control-plane/tasks/{task_id}/cancel")

    async def list_tasks(
        self,
        status: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List scheduled tasks."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if task_type:
            params["task_type"] = task_type
        return await self._request("GET", "/api/control-plane/tasks", params=params)

    # =========================================================================
    # Control Plane - Policies
    # =========================================================================

    async def create_policy(
        self,
        name: str,
        rules: List[Dict[str, Any]],
        description: Optional[str] = None,
        enabled: bool = True,
    ) -> Dict[str, Any]:
        """Create a control plane policy.

        Args:
            name: Policy name
            rules: List of policy rules
            description: Optional policy description
            enabled: Whether policy is active
        """
        body: Dict[str, Any] = {"name": name, "rules": rules, "enabled": enabled}
        if description:
            body["description"] = description
        return await self._request("POST", "/api/control-plane/policies", json=body)

    async def get_policy(self, policy_id: str) -> Dict[str, Any]:
        """Get a policy by ID."""
        return await self._request("GET", f"/api/control-plane/policies/{policy_id}")

    async def update_policy(
        self,
        policy_id: str,
        name: Optional[str] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Update a policy."""
        body: Dict[str, Any] = {}
        if name:
            body["name"] = name
        if rules:
            body["rules"] = rules
        if enabled is not None:
            body["enabled"] = enabled
        return await self._request("PUT", f"/api/control-plane/policies/{policy_id}", json=body)

    async def delete_policy(self, policy_id: str) -> Dict[str, Any]:
        """Delete a policy."""
        return await self._request("DELETE", f"/api/control-plane/policies/{policy_id}")

    async def list_policies(
        self,
        enabled: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List control plane policies."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if enabled is not None:
            params["enabled"] = enabled
        return await self._request("GET", "/api/control-plane/policies", params=params)

    # =========================================================================
    # Graph Debates
    # =========================================================================

    async def create_graph_debate(
        self,
        question: str,
        agents: List[str],
        graph_structure: Dict[str, Any],
        rounds: int = 3,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a graph-structured debate.

        Args:
            question: The debate question
            agents: List of participating agents
            graph_structure: Graph topology definition with nodes and edges
            rounds: Number of debate rounds
        """
        return await self._request(
            "POST",
            "/api/debates/graph",
            json={
                "question": question,
                "agents": agents,
                "graph_structure": graph_structure,
                "rounds": rounds,
                **kwargs,
            },
        )

    async def get_graph_debate_topology(self, debate_id: str) -> Dict[str, Any]:
        """Get the graph topology for a debate."""
        return await self._request("GET", f"/api/debates/{debate_id}/graph/topology")

    async def update_graph_edge(
        self,
        debate_id: str,
        source: str,
        target: str,
        weight: float,
        edge_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update an edge in the debate graph."""
        body: Dict[str, Any] = {"source": source, "target": target, "weight": weight}
        if edge_type:
            body["edge_type"] = edge_type
        return await self._request("PUT", f"/api/debates/{debate_id}/graph/edges", json=body)

    async def get_graph_path(
        self,
        debate_id: str,
        start_node: str,
        end_node: str,
        algorithm: str = "dijkstra",
    ) -> Dict[str, Any]:
        """Find path between nodes in debate graph."""
        return await self._request(
            "GET",
            f"/api/debates/{debate_id}/graph/path",
            params={"start": start_node, "end": end_node, "algorithm": algorithm},
        )

    # =========================================================================
    # Matrix Debates
    # =========================================================================

    async def create_matrix_debate(
        self,
        question: str,
        agents: List[str],
        dimensions: List[str],
        rounds: int = 3,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a matrix-structured debate.

        Args:
            question: The debate question
            agents: List of participating agents
            dimensions: Analysis dimensions (e.g., ['cost', 'risk', 'feasibility'])
            rounds: Number of debate rounds
        """
        return await self._request(
            "POST",
            "/api/debates/matrix",
            json={
                "question": question,
                "agents": agents,
                "dimensions": dimensions,
                "rounds": rounds,
                **kwargs,
            },
        )

    async def get_matrix_analysis(self, debate_id: str) -> Dict[str, Any]:
        """Get the matrix analysis results for a debate."""
        return await self._request("GET", f"/api/debates/{debate_id}/matrix/analysis")

    async def update_matrix_cell(
        self,
        debate_id: str,
        row: str,
        column: str,
        value: Any,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Update a cell in the debate matrix."""
        body: Dict[str, Any] = {"row": row, "column": column, "value": value}
        if confidence is not None:
            body["confidence"] = confidence
        return await self._request("PUT", f"/api/debates/{debate_id}/matrix/cells", json=body)

    async def get_matrix_summary(self, debate_id: str) -> Dict[str, Any]:
        """Get summary statistics for a matrix debate."""
        return await self._request("GET", f"/api/debates/{debate_id}/matrix/summary")

    # =========================================================================
    # Agent Intelligence
    # =========================================================================

    async def get_agent_profile(self, agent_name: str) -> Dict[str, Any]:
        """Get detailed agent intelligence profile."""
        return await self._request("GET", f"/api/agents/{agent_name}/profile")

    async def get_agent_history(
        self,
        agent_name: str,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get agent debate history and performance."""
        return await self._request(
            "GET",
            f"/api/agents/{agent_name}/history",
            params={"limit": limit, "offset": offset},
        )

    async def get_agent_leaderboard(
        self,
        metric: str = "elo",
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Get agent leaderboard rankings.

        Args:
            metric: Ranking metric ('elo', 'win_rate', 'consensus_rate')
            limit: Number of agents to return
        """
        return await self._request(
            "GET",
            "/api/agents/leaderboard",
            params={"metric": metric, "limit": limit},
        )

    async def compare_agents(
        self,
        agent_names: List[str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare multiple agents across metrics.

        Args:
            agent_names: List of agent names to compare
            metrics: Optional list of metrics to compare
        """
        body: Dict[str, Any] = {"agents": agent_names}
        if metrics:
            body["metrics"] = metrics
        return await self._request("POST", "/api/agents/compare", json=body)

    # =========================================================================
    # Verification
    # =========================================================================

    async def verify_claim(
        self,
        claim: str,
        sources: Optional[List[str]] = None,
        confidence_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """Verify a claim using multi-agent analysis.

        Args:
            claim: The claim to verify
            sources: Optional list of source URLs/references
            confidence_threshold: Minimum confidence for verification
        """
        body: Dict[str, Any] = {"claim": claim, "confidence_threshold": confidence_threshold}
        if sources:
            body["sources"] = sources
        return await self._request("POST", "/api/verification/claim", json=body)

    async def get_verification_status(self, verification_id: str) -> Dict[str, Any]:
        """Get status of a verification request."""
        return await self._request("GET", f"/api/verification/{verification_id}")

    async def get_verification_evidence(self, verification_id: str) -> Dict[str, Any]:
        """Get evidence collected for a verification."""
        return await self._request("GET", f"/api/verification/{verification_id}/evidence")

    # =========================================================================
    # Memory (Continuum)
    # =========================================================================

    async def store_memory(
        self,
        content: str,
        tier: str = "medium",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store content in continuum memory.

        Args:
            content: Content to store
            tier: Memory tier ('fast', 'medium', 'slow', 'glacial')
            tags: Optional tags for categorization
            metadata: Optional metadata
        """
        body: Dict[str, Any] = {"content": content, "tier": tier}
        if tags:
            body["tags"] = tags
        if metadata:
            body["metadata"] = metadata
        return await self._request("POST", "/api/memory/continuum/store", json=body)

    async def retrieve_memory(
        self,
        query: str,
        tier: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Retrieve memories matching a query.

        Args:
            query: Search query
            tier: Optional specific tier to search
            limit: Maximum results to return
        """
        params: Dict[str, Any] = {"query": query, "limit": limit}
        if tier:
            params["tier"] = tier
        return await self._request("GET", "/api/memory/continuum/retrieve", params=params)

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return await self._request("GET", "/api/memory/continuum/stats")

    # =========================================================================
    # Knowledge Mound
    # =========================================================================

    async def search_knowledge(
        self,
        query: str,
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Search the knowledge mound.

        Args:
            query: Search query
            limit: Maximum results
            filters: Optional search filters
        """
        body: Dict[str, Any] = {"query": query, "limit": limit}
        if filters:
            body["filters"] = filters
        return await self._request("POST", "/api/knowledge/search", json=body)

    async def add_knowledge(
        self,
        content: str,
        source: str,
        knowledge_type: str = "document",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add content to the knowledge mound.

        Args:
            content: Content to add
            source: Source identifier
            knowledge_type: Type of knowledge
            metadata: Optional metadata
        """
        body: Dict[str, Any] = {
            "content": content,
            "source": source,
            "type": knowledge_type,
        }
        if metadata:
            body["metadata"] = metadata
        return await self._request("POST", "/api/knowledge/add", json=body)

    async def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge mound statistics."""
        return await self._request("GET", "/api/knowledge/stats")

    # =========================================================================
    # ELO Rankings
    # =========================================================================

    async def get_elo_rankings(
        self,
        limit: int = 50,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get ELO rankings for agents.

        Args:
            limit: Number of rankings to return
            category: Optional category filter
        """
        params: Dict[str, Any] = {"limit": limit}
        if category:
            params["category"] = category
        return await self._request("GET", "/api/ranking/elo", params=params)

    async def get_agent_elo(self, agent_name: str) -> Dict[str, Any]:
        """Get ELO rating for a specific agent."""
        return await self._request("GET", f"/api/ranking/elo/{agent_name}")

    async def get_elo_history(
        self,
        agent_name: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get ELO rating history for an agent."""
        return await self._request(
            "GET",
            f"/api/ranking/elo/{agent_name}/history",
            params={"days": days},
        )

    # =========================================================================
    # Tournaments
    # =========================================================================

    async def create_tournament(
        self,
        name: str,
        agents: List[str],
        questions: List[str],
        tournament_type: str = "round_robin",
    ) -> Dict[str, Any]:
        """Create a debate tournament.

        Args:
            name: Tournament name
            agents: Participating agents
            questions: Debate questions
            tournament_type: Type ('round_robin', 'elimination', 'swiss')
        """
        return await self._request(
            "POST",
            "/api/tournaments",
            json={
                "name": name,
                "agents": agents,
                "questions": questions,
                "type": tournament_type,
            },
        )

    async def get_tournament(self, tournament_id: str) -> Dict[str, Any]:
        """Get tournament details."""
        return await self._request("GET", f"/api/tournaments/{tournament_id}")

    async def list_tournaments(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List tournaments."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        return await self._request("GET", "/api/tournaments", params=params)

    async def get_tournament_standings(self, tournament_id: str) -> Dict[str, Any]:
        """Get tournament standings."""
        return await self._request("GET", f"/api/tournaments/{tournament_id}/standings")

    # =========================================================================
    # Authentication
    # =========================================================================

    async def register_user(
        self,
        email: str,
        password: str,
        name: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a new user account.

        Args:
            email: User's email address
            password: User's password
            name: Optional display name
            organization: Optional organization name
        """
        body: Dict[str, Any] = {"email": email, "password": password}
        if name:
            body["name"] = name
        if organization:
            body["organization"] = organization
        return await self._request("POST", "/api/v1/auth/register", json=body)

    async def login(
        self,
        email: str,
        password: str,
        mfa_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Login to get access token.

        Args:
            email: User's email address
            password: User's password
            mfa_code: Optional MFA code if MFA is enabled
        """
        body: Dict[str, Any] = {"email": email, "password": password}
        if mfa_code:
            body["mfa_code"] = mfa_code
        return await self._request("POST", "/api/v1/auth/login", json=body)

    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token using refresh token.

        Args:
            refresh_token: The refresh token
        """
        return await self._request(
            "POST", "/api/v1/auth/refresh", json={"refresh_token": refresh_token}
        )

    async def logout(self) -> None:
        """Logout and invalidate current token."""
        await self._request("POST", "/api/v1/auth/logout", json={})

    async def logout_all(self) -> Dict[str, Any]:
        """Logout from all sessions."""
        return await self._request("POST", "/api/v1/auth/logout-all", json={})

    async def verify_email(self, token: str) -> Dict[str, Any]:
        """Verify email address with token.

        Args:
            token: Email verification token
        """
        return await self._request("POST", "/api/v1/auth/verify-email", json={"token": token})

    async def resend_verification(self, email: str) -> Dict[str, Any]:
        """Resend email verification.

        Args:
            email: Email address to resend verification to
        """
        return await self._request(
            "POST", "/api/v1/auth/resend-verification", json={"email": email}
        )

    async def get_current_user(self) -> Dict[str, Any]:
        """Get current authenticated user profile."""
        return await self._request("GET", "/api/v1/auth/me")

    async def update_profile(
        self,
        name: Optional[str] = None,
        organization: Optional[str] = None,
        avatar_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update current user's profile.

        Args:
            name: New display name
            organization: New organization name
            avatar_url: New avatar URL
            metadata: Additional metadata
        """
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if organization is not None:
            body["organization"] = organization
        if avatar_url is not None:
            body["avatar_url"] = avatar_url
        if metadata is not None:
            body["metadata"] = metadata
        return await self._request("PATCH", "/api/v1/auth/profile", json=body)

    async def change_password(self, current_password: str, new_password: str) -> None:
        """Change user's password.

        Args:
            current_password: Current password
            new_password: New password
        """
        await self._request(
            "POST",
            "/api/v1/auth/change-password",
            json={"current_password": current_password, "new_password": new_password},
        )

    async def request_password_reset(self, email: str) -> None:
        """Request password reset email.

        Args:
            email: Email address for password reset
        """
        await self._request("POST", "/api/v1/auth/forgot-password", json={"email": email})

    async def reset_password(self, token: str, new_password: str) -> None:
        """Reset password using reset token.

        Args:
            token: Password reset token
            new_password: New password
        """
        await self._request(
            "POST",
            "/api/v1/auth/reset-password",
            json={"token": token, "new_password": new_password},
        )

    # =========================================================================
    # MFA (Multi-Factor Authentication)
    # =========================================================================

    async def setup_mfa(
        self, mfa_type: str = "totp", phone_number: Optional[str] = None
    ) -> Dict[str, Any]:
        """Setup MFA for the account.

        Args:
            mfa_type: Type of MFA ('totp', 'sms', 'email')
            phone_number: Phone number for SMS-based MFA
        """
        body: Dict[str, Any] = {"type": mfa_type}
        if phone_number:
            body["phone_number"] = phone_number
        return await self._request("POST", "/api/v1/auth/mfa/setup", json=body)

    async def verify_mfa_setup(self, code: str, mfa_type: str = "totp") -> Dict[str, Any]:
        """Verify MFA setup with code.

        Args:
            code: Verification code from authenticator or SMS
            mfa_type: Type of MFA being verified
        """
        return await self._request(
            "POST",
            "/api/v1/auth/mfa/verify-setup",
            json={"code": code, "type": mfa_type},
        )

    async def enable_mfa(self, code: str) -> Dict[str, Any]:
        """Enable MFA after verification.

        Args:
            code: Final verification code
        """
        return await self._request("POST", "/api/v1/auth/mfa/enable", json={"code": code})

    async def disable_mfa(self) -> None:
        """Disable MFA for the account."""
        await self._request("POST", "/api/v1/auth/mfa/disable", json={})

    async def generate_backup_codes(self) -> Dict[str, Any]:
        """Generate new backup codes for MFA."""
        return await self._request("POST", "/api/v1/auth/mfa/backup-codes", json={})

    # =========================================================================
    # Session Management
    # =========================================================================

    async def list_sessions(self) -> Dict[str, Any]:
        """List all active sessions."""
        return await self._request("GET", "/api/v1/auth/sessions")

    async def revoke_session(self, session_id: str) -> Dict[str, Any]:
        """Revoke a specific session.

        Args:
            session_id: ID of session to revoke
        """
        return await self._request("DELETE", f"/api/v1/auth/sessions/{session_id}")

    # =========================================================================
    # API Key Management
    # =========================================================================

    async def list_api_keys(self) -> Dict[str, Any]:
        """List all API keys for the account."""
        return await self._request("GET", "/api/v1/auth/api-keys")

    async def create_api_key(
        self,
        name: str,
        expires_in: Optional[int] = None,
        scopes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a new API key.

        Args:
            name: Name/description for the API key
            expires_in: Expiration time in seconds (None for no expiration)
            scopes: Optional list of permission scopes
        """
        body: Dict[str, Any] = {"name": name}
        if expires_in is not None:
            body["expires_in"] = expires_in
        if scopes is not None:
            body["scopes"] = scopes
        return await self._request("POST", "/api/v1/auth/api-keys", json=body)

    async def revoke_api_key(self, key_id: str) -> Dict[str, Any]:
        """Revoke an API key.

        Args:
            key_id: ID of the API key to revoke
        """
        return await self._request("DELETE", f"/api/v1/auth/api-keys/{key_id}")

    # =========================================================================
    # OAuth & SSO
    # =========================================================================

    async def get_oauth_url(
        self,
        provider: str,
        redirect_uri: Optional[str] = None,
        state: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get OAuth authorization URL.

        Args:
            provider: OAuth provider ('google', 'github', 'microsoft')
            redirect_uri: Custom redirect URI
            state: Custom state parameter
            scope: OAuth scope
        """
        params: Dict[str, Any] = {"provider": provider}
        if redirect_uri:
            params["redirect_uri"] = redirect_uri
        if state:
            params["state"] = state
        if scope:
            params["scope"] = scope
        return await self._request("GET", "/api/v1/auth/oauth/authorize", params=params)

    async def complete_oauth(self, code: str, state: str, provider: str) -> Dict[str, Any]:
        """Complete OAuth flow with authorization code.

        Args:
            code: Authorization code from OAuth provider
            state: State parameter for verification
            provider: OAuth provider name
        """
        return await self._request(
            "POST",
            "/api/v1/auth/oauth/callback",
            json={"code": code, "state": state, "provider": provider},
        )

    async def list_oauth_providers(self) -> Dict[str, Any]:
        """List available OAuth providers."""
        return await self._request("GET", "/api/v1/auth/oauth/providers")

    async def link_oauth_provider(self, provider: str, code: str) -> Dict[str, Any]:
        """Link an OAuth provider to existing account.

        Args:
            provider: OAuth provider name
            code: Authorization code
        """
        return await self._request(
            "POST",
            "/api/v1/auth/oauth/link",
            json={"provider": provider, "code": code},
        )

    async def unlink_oauth_provider(self, provider: str) -> Dict[str, Any]:
        """Unlink an OAuth provider from account.

        Args:
            provider: OAuth provider to unlink
        """
        return await self._request("DELETE", f"/api/v1/auth/oauth/providers/{provider}")

    async def initiate_sso_login(
        self, domain: Optional[str] = None, redirect_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Initiate SSO login.

        Args:
            domain: Organization domain for SSO
            redirect_url: URL to redirect after SSO
        """
        params: Dict[str, Any] = {}
        if domain:
            params["domain"] = domain
        if redirect_url:
            params["redirect_url"] = redirect_url
        return await self._request("GET", "/api/v1/auth/sso/initiate", params=params)

    async def list_sso_providers(self) -> Dict[str, Any]:
        """List configured SSO providers for organization."""
        return await self._request("GET", "/api/v1/auth/sso/providers")

    # =========================================================================
    # Invitations
    # =========================================================================

    async def invite_team_member(
        self,
        email: str,
        organization_id: str,
        role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Invite a new team member.

        Args:
            email: Email address to invite
            organization_id: Organization to invite to
            role: Role to assign to invitee
        """
        body: Dict[str, Any] = {"email": email, "organization_id": organization_id}
        if role:
            body["role"] = role
        return await self._request("POST", "/api/v1/invitations", json=body)

    async def check_invite(self, token: str) -> Dict[str, Any]:
        """Check if invitation token is valid.

        Args:
            token: Invitation token
        """
        return await self._request("GET", f"/api/v1/invitations/{token}")

    async def accept_invite(self, token: str) -> Dict[str, Any]:
        """Accept an invitation.

        Args:
            token: Invitation token
        """
        return await self._request("POST", f"/api/v1/invitations/{token}/accept", json={})

    # =========================================================================
    # Tenancy
    # =========================================================================

    async def list_tenants(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List tenants/organizations.

        Args:
            limit: Maximum number of results
            offset: Pagination offset
        """
        return await self._request(
            "GET", "/api/v1/tenants", params={"limit": limit, "offset": offset}
        )

    async def get_tenant(self, tenant_id: str) -> Dict[str, Any]:
        """Get tenant details.

        Args:
            tenant_id: Tenant ID
        """
        return await self._request("GET", f"/api/v1/tenants/{tenant_id}")

    async def create_tenant(
        self,
        name: str,
        slug: Optional[str] = None,
        description: Optional[str] = None,
        plan: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new tenant/organization.

        Args:
            name: Tenant name
            slug: URL-friendly slug
            description: Tenant description
            plan: Subscription plan
        """
        body: Dict[str, Any] = {"name": name}
        if slug:
            body["slug"] = slug
        if description:
            body["description"] = description
        if plan:
            body["plan"] = plan
        return await self._request("POST", "/api/v1/tenants", json=body)

    async def update_tenant(
        self,
        tenant_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        plan: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update tenant settings.

        Args:
            tenant_id: Tenant ID
            name: New name
            description: New description
            plan: New plan
            status: New status
        """
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if plan is not None:
            body["plan"] = plan
        if status is not None:
            body["status"] = status
        return await self._request("PATCH", f"/api/v1/tenants/{tenant_id}", json=body)

    async def delete_tenant(self, tenant_id: str) -> None:
        """Delete a tenant.

        Args:
            tenant_id: Tenant ID to delete
        """
        await self._request("DELETE", f"/api/v1/tenants/{tenant_id}")

    async def get_tenant_quotas(self, tenant_id: str) -> Dict[str, Any]:
        """Get tenant resource quotas.

        Args:
            tenant_id: Tenant ID
        """
        return await self._request("GET", f"/api/v1/tenants/{tenant_id}/quotas")

    async def update_tenant_quotas(self, tenant_id: str, **quotas: Any) -> Dict[str, Any]:
        """Update tenant quotas.

        Args:
            tenant_id: Tenant ID
            **quotas: Quota settings to update
        """
        return await self._request("PATCH", f"/api/v1/tenants/{tenant_id}/quotas", json=quotas)

    async def list_tenant_members(
        self, tenant_id: str, limit: int = 50, offset: int = 0
    ) -> Dict[str, Any]:
        """List tenant members.

        Args:
            tenant_id: Tenant ID
            limit: Maximum results
            offset: Pagination offset
        """
        return await self._request(
            "GET",
            f"/api/v1/tenants/{tenant_id}/members",
            params={"limit": limit, "offset": offset},
        )

    async def add_tenant_member(
        self,
        tenant_id: str,
        email: str,
        role: str,
        send_invitation: bool = True,
    ) -> Dict[str, Any]:
        """Add a member to tenant.

        Args:
            tenant_id: Tenant ID
            email: Member's email
            role: Role to assign
            send_invitation: Whether to send invitation email
        """
        return await self._request(
            "POST",
            f"/api/v1/tenants/{tenant_id}/members",
            json={"email": email, "role": role, "send_invitation": send_invitation},
        )

    async def remove_tenant_member(self, tenant_id: str, user_id: str) -> None:
        """Remove a member from tenant.

        Args:
            tenant_id: Tenant ID
            user_id: User ID to remove
        """
        await self._request("DELETE", f"/api/v1/tenants/{tenant_id}/members/{user_id}")

    async def setup_organization(
        self,
        name: str,
        slug: Optional[str] = None,
        plan: Optional[str] = None,
        billing_email: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Setup a new organization (onboarding).

        Args:
            name: Organization name
            slug: URL-friendly slug
            plan: Subscription plan
            billing_email: Billing contact email
        """
        body: Dict[str, Any] = {"name": name}
        if slug:
            body["slug"] = slug
        if plan:
            body["plan"] = plan
        if billing_email:
            body["billing_email"] = billing_email
        return await self._request("POST", "/api/v1/organizations/setup", json=body)

    # =========================================================================
    # RBAC (Role-Based Access Control)
    # =========================================================================

    async def list_roles(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List available roles.

        Args:
            limit: Maximum results
            offset: Pagination offset
        """
        return await self._request(
            "GET", "/api/v1/rbac/roles", params={"limit": limit, "offset": offset}
        )

    async def get_role(self, role_id: str) -> Dict[str, Any]:
        """Get role details.

        Args:
            role_id: Role ID
        """
        return await self._request("GET", f"/api/v1/rbac/roles/{role_id}")

    async def create_role(
        self,
        name: str,
        permissions: List[str],
        description: Optional[str] = None,
        inherits_from: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a new role.

        Args:
            name: Role name
            permissions: List of permission strings
            description: Role description
            inherits_from: List of role IDs to inherit from
        """
        body: Dict[str, Any] = {"name": name, "permissions": permissions}
        if description:
            body["description"] = description
        if inherits_from:
            body["inherits_from"] = inherits_from
        return await self._request("POST", "/api/v1/rbac/roles", json=body)

    async def update_role(
        self,
        role_id: str,
        name: Optional[str] = None,
        permissions: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update a role.

        Args:
            role_id: Role ID
            name: New name
            permissions: New permissions list
            description: New description
        """
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if permissions is not None:
            body["permissions"] = permissions
        if description is not None:
            body["description"] = description
        return await self._request("PATCH", f"/api/v1/rbac/roles/{role_id}", json=body)

    async def delete_role(self, role_id: str) -> None:
        """Delete a role.

        Args:
            role_id: Role ID to delete
        """
        await self._request("DELETE", f"/api/v1/rbac/roles/{role_id}")

    async def list_permissions(self) -> Dict[str, Any]:
        """List all available permissions."""
        return await self._request("GET", "/api/v1/rbac/permissions")

    async def assign_role(
        self,
        user_id: str,
        role_id: str,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Assign a role to a user.

        Args:
            user_id: User ID
            role_id: Role ID to assign
            tenant_id: Optional tenant scope
        """
        body: Dict[str, Any] = {"user_id": user_id, "role_id": role_id}
        if tenant_id:
            body["tenant_id"] = tenant_id
        await self._request("POST", "/api/v1/rbac/assignments", json=body)

    async def revoke_role(
        self,
        user_id: str,
        role_id: str,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Revoke a role from a user.

        Args:
            user_id: User ID
            role_id: Role ID to revoke
            tenant_id: Optional tenant scope
        """
        params: Dict[str, Any] = {"user_id": user_id, "role_id": role_id}
        if tenant_id:
            params["tenant_id"] = tenant_id
        await self._request("DELETE", "/api/v1/rbac/assignments", params=params)

    async def get_user_roles(self, user_id: str) -> Dict[str, Any]:
        """Get all roles assigned to a user.

        Args:
            user_id: User ID
        """
        return await self._request("GET", f"/api/v1/rbac/users/{user_id}/roles")

    async def check_permission(
        self,
        user_id: str,
        permission: str,
        resource: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check if user has a permission.

        Args:
            user_id: User ID
            permission: Permission to check
            resource: Optional resource context
        """
        params: Dict[str, Any] = {"permission": permission}
        if resource:
            params["resource"] = resource
        return await self._request(
            "GET", f"/api/v1/rbac/users/{user_id}/permissions/check", params=params
        )

    async def list_role_assignments(
        self, role_id: str, limit: int = 50, offset: int = 0
    ) -> Dict[str, Any]:
        """List users assigned to a role.

        Args:
            role_id: Role ID
            limit: Maximum results
            offset: Pagination offset
        """
        return await self._request(
            "GET",
            f"/api/v1/rbac/roles/{role_id}/assignments",
            params={"limit": limit, "offset": offset},
        )

    async def bulk_assign_roles(self, assignments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bulk assign roles to users.

        Args:
            assignments: List of {user_id, role_id, tenant_id?} dicts
        """
        return await self._request(
            "POST", "/api/v1/rbac/assignments/bulk", json={"assignments": assignments}
        )

    # =========================================================================
    # Billing
    # =========================================================================

    async def list_billing_plans(self) -> Dict[str, Any]:
        """List available billing plans."""
        return await self._request("GET", "/api/v1/billing/plans")

    async def get_billing_usage(self, period: Optional[str] = None) -> Dict[str, Any]:
        """Get billing usage for current period.

        Args:
            period: Time period ('current', 'previous', or YYYY-MM)
        """
        params: Dict[str, Any] = {}
        if period:
            params["period"] = period
        return await self._request("GET", "/api/v1/billing/usage", params=params)

    async def get_subscription(self) -> Dict[str, Any]:
        """Get current subscription details."""
        return await self._request("GET", "/api/v1/billing/subscription")

    async def create_checkout_session(
        self, plan_id: str, success_url: str, cancel_url: str
    ) -> Dict[str, Any]:
        """Create a checkout session for subscription.

        Args:
            plan_id: Plan to subscribe to
            success_url: URL to redirect on success
            cancel_url: URL to redirect on cancel
        """
        return await self._request(
            "POST",
            "/api/v1/billing/checkout",
            json={
                "plan_id": plan_id,
                "success_url": success_url,
                "cancel_url": cancel_url,
            },
        )

    async def create_billing_portal_session(
        self, return_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a billing portal session.

        Args:
            return_url: URL to return to after portal
        """
        body: Dict[str, Any] = {}
        if return_url:
            body["return_url"] = return_url
        return await self._request("POST", "/api/v1/billing/portal", json=body)

    async def cancel_subscription(self) -> Dict[str, Any]:
        """Cancel current subscription."""
        return await self._request("POST", "/api/v1/billing/subscription/cancel", json={})

    async def resume_subscription(self) -> Dict[str, Any]:
        """Resume a canceled subscription."""
        return await self._request("POST", "/api/v1/billing/subscription/resume", json={})

    async def get_invoice_history(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Get invoice history.

        Args:
            limit: Maximum results
            offset: Pagination offset
        """
        return await self._request(
            "GET", "/api/v1/billing/invoices", params={"limit": limit, "offset": offset}
        )

    async def get_usage_forecast(self) -> Dict[str, Any]:
        """Get usage forecast for current period."""
        return await self._request("GET", "/api/v1/billing/forecast")

    async def export_usage_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        format: str = "csv",
    ) -> Dict[str, Any]:
        """Export usage data.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            format: Export format ('csv', 'json')
        """
        params: Dict[str, Any] = {"format": format}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return await self._request("GET", "/api/v1/billing/export", params=params)

    # =========================================================================
    # Budgets
    # =========================================================================

    async def list_budgets(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List budgets.

        Args:
            limit: Maximum results
            offset: Pagination offset
        """
        return await self._request(
            "GET", "/api/v1/budgets", params={"limit": limit, "offset": offset}
        )

    async def create_budget(
        self,
        name: str,
        limit_amount: float,
        period: str,
        description: Optional[str] = None,
        alert_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create a new budget.

        Args:
            name: Budget name
            limit_amount: Budget limit amount
            period: Budget period ('daily', 'weekly', 'monthly')
            description: Budget description
            alert_threshold: Percentage threshold for alerts (0-1)
        """
        body: Dict[str, Any] = {
            "name": name,
            "limit_amount": limit_amount,
            "period": period,
        }
        if description:
            body["description"] = description
        if alert_threshold is not None:
            body["alert_threshold"] = alert_threshold
        return await self._request("POST", "/api/v1/budgets", json=body)

    async def get_budget(self, budget_id: str) -> Dict[str, Any]:
        """Get budget details.

        Args:
            budget_id: Budget ID
        """
        return await self._request("GET", f"/api/v1/budgets/{budget_id}")

    async def update_budget(
        self,
        budget_id: str,
        name: Optional[str] = None,
        limit_amount: Optional[float] = None,
        alert_threshold: Optional[float] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update a budget.

        Args:
            budget_id: Budget ID
            name: New name
            limit_amount: New limit
            alert_threshold: New alert threshold
            status: New status ('active', 'paused')
        """
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if limit_amount is not None:
            body["limit_amount"] = limit_amount
        if alert_threshold is not None:
            body["alert_threshold"] = alert_threshold
        if status is not None:
            body["status"] = status
        return await self._request("PATCH", f"/api/v1/budgets/{budget_id}", json=body)

    async def delete_budget(self, budget_id: str) -> Dict[str, Any]:
        """Delete a budget.

        Args:
            budget_id: Budget ID to delete
        """
        return await self._request("DELETE", f"/api/v1/budgets/{budget_id}")

    async def get_budget_alerts(
        self, budget_id: str, limit: int = 50, offset: int = 0
    ) -> Dict[str, Any]:
        """Get alerts for a budget.

        Args:
            budget_id: Budget ID
            limit: Maximum results
            offset: Pagination offset
        """
        return await self._request(
            "GET",
            f"/api/v1/budgets/{budget_id}/alerts",
            params={"limit": limit, "offset": offset},
        )

    async def acknowledge_budget_alert(self, budget_id: str, alert_id: str) -> Dict[str, Any]:
        """Acknowledge a budget alert.

        Args:
            budget_id: Budget ID
            alert_id: Alert ID to acknowledge
        """
        return await self._request(
            "POST", f"/api/v1/budgets/{budget_id}/alerts/{alert_id}/acknowledge", json={}
        )

    async def check_budget(
        self,
        operation: str,
        estimated_cost: float,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check if an operation is within budget.

        Args:
            operation: Operation type/name
            estimated_cost: Estimated cost of operation
            user_id: Optional user ID for user-specific budget
        """
        body: Dict[str, Any] = {
            "operation": operation,
            "estimated_cost": estimated_cost,
        }
        if user_id:
            body["user_id"] = user_id
        return await self._request("POST", "/api/v1/budgets/check", json=body)

    async def get_budget_summary(self) -> Dict[str, Any]:
        """Get summary of all budgets and current usage."""
        return await self._request("GET", "/api/v1/budgets/summary")

    async def add_budget_override(
        self,
        budget_id: str,
        user_id: str,
        limit: float,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add a budget override for a user.

        Args:
            budget_id: Budget ID
            user_id: User ID for override
            limit: Override limit
            reason: Reason for override
        """
        body: Dict[str, Any] = {"user_id": user_id, "limit": limit}
        if reason:
            body["reason"] = reason
        return await self._request("POST", f"/api/v1/budgets/{budget_id}/overrides", json=body)

    async def remove_budget_override(self, budget_id: str, user_id: str) -> Dict[str, Any]:
        """Remove a budget override.

        Args:
            budget_id: Budget ID
            user_id: User ID whose override to remove
        """
        return await self._request("DELETE", f"/api/v1/budgets/{budget_id}/overrides/{user_id}")

    async def reset_budget(self, budget_id: str) -> Dict[str, Any]:
        """Reset budget usage to zero.

        Args:
            budget_id: Budget ID to reset
        """
        return await self._request("POST", f"/api/v1/budgets/{budget_id}/reset", json={})

    # =========================================================================
    # Audit
    # =========================================================================

    async def list_audit_events(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        actor_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List audit events.

        Args:
            start_date: Filter from date (ISO format)
            end_date: Filter to date (ISO format)
            actor_id: Filter by actor
            resource_type: Filter by resource type
            action: Filter by action
            limit: Maximum results
            offset: Pagination offset
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if actor_id:
            params["actor_id"] = actor_id
        if resource_type:
            params["resource_type"] = resource_type
        if action:
            params["action"] = action
        return await self._request("GET", "/api/v1/audit/events", params=params)

    async def get_audit_stats(self, period: Optional[str] = None) -> Dict[str, Any]:
        """Get audit statistics.

        Args:
            period: Time period for stats
        """
        params: Dict[str, Any] = {}
        if period:
            params["period"] = period
        return await self._request("GET", "/api/v1/audit/stats", params=params)

    async def export_audit_logs(
        self,
        start_date: str,
        end_date: str,
        format: str = "json",
        filters: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Export audit logs.

        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            format: Export format ('json', 'csv')
            filters: Additional filters
        """
        body: Dict[str, Any] = {
            "start_date": start_date,
            "end_date": end_date,
            "format": format,
        }
        if filters:
            body["filters"] = filters
        return await self._request("POST", "/api/v1/audit/export", json=body)

    async def verify_audit_integrity(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verify audit log integrity.

        Args:
            start_date: Start date for verification
            end_date: End date for verification
        """
        params: Dict[str, Any] = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return await self._request("GET", "/api/v1/audit/verify", params=params)

    async def list_audit_trails(
        self,
        verdict: Optional[str] = None,
        risk_level: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List audit trails.

        Args:
            verdict: Filter by verdict
            risk_level: Filter by risk level
            limit: Maximum results
            offset: Pagination offset
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if verdict:
            params["verdict"] = verdict
        if risk_level:
            params["risk_level"] = risk_level
        return await self._request("GET", "/api/v1/audit/trails", params=params)

    async def get_audit_trail(self, trail_id: str) -> Dict[str, Any]:
        """Get audit trail details.

        Args:
            trail_id: Audit trail ID
        """
        return await self._request("GET", f"/api/v1/audit/trails/{trail_id}")

    async def verify_audit_trail(self, trail_id: str) -> Dict[str, Any]:
        """Verify audit trail integrity.

        Args:
            trail_id: Audit trail ID
        """
        return await self._request("POST", f"/api/v1/audit/trails/{trail_id}/verify", json={})

    async def export_audit_trail(self, trail_id: str, format: str = "json") -> Dict[str, Any]:
        """Export audit trail.

        Args:
            trail_id: Audit trail ID
            format: Export format ('json', 'pdf')
        """
        return await self._request(
            "GET", f"/api/v1/audit/trails/{trail_id}/export", params={"format": format}
        )

    # =========================================================================
    # Notifications
    # =========================================================================

    async def get_notification_status(self) -> Dict[str, Any]:
        """Get notification configuration status."""
        return await self._request("GET", "/api/v1/notifications/status")

    async def configure_email_notifications(
        self,
        provider: str,
        from_email: str,
        **config: Any,
    ) -> Dict[str, Any]:
        """Configure email notifications.

        Args:
            provider: Email provider ('sendgrid', 'ses', 'smtp')
            from_email: From email address
            **config: Provider-specific configuration
        """
        body: Dict[str, Any] = {
            "provider": provider,
            "from_email": from_email,
            **config,
        }
        return await self._request("POST", "/api/v1/notifications/email/configure", json=body)

    async def configure_telegram_notifications(
        self,
        bot_token: str,
        chat_id: str,
        **config: Any,
    ) -> Dict[str, Any]:
        """Configure Telegram notifications.

        Args:
            bot_token: Telegram bot token
            chat_id: Telegram chat ID
            **config: Additional configuration
        """
        body: Dict[str, Any] = {"bot_token": bot_token, "chat_id": chat_id, **config}
        return await self._request("POST", "/api/v1/notifications/telegram/configure", json=body)

    async def get_email_recipients(self) -> Dict[str, Any]:
        """Get list of email notification recipients."""
        return await self._request("GET", "/api/v1/notifications/email/recipients")

    async def add_email_recipient(
        self,
        email: str,
        name: Optional[str] = None,
        preferences: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """Add email notification recipient.

        Args:
            email: Email address
            name: Recipient name
            preferences: Notification preferences by type
        """
        body: Dict[str, Any] = {"email": email}
        if name:
            body["name"] = name
        if preferences:
            body["preferences"] = preferences
        return await self._request("POST", "/api/v1/notifications/email/recipients", json=body)

    async def remove_email_recipient(self, email: str) -> Dict[str, Any]:
        """Remove email notification recipient.

        Args:
            email: Email address to remove
        """
        return await self._request(
            "DELETE", "/api/v1/notifications/email/recipients", params={"email": email}
        )

    async def send_test_notification(self, channel: str) -> Dict[str, Any]:
        """Send a test notification.

        Args:
            channel: Channel to test ('email', 'telegram', 'slack')
        """
        return await self._request("POST", "/api/v1/notifications/test", json={"channel": channel})

    async def send_notification(
        self,
        channel: str,
        message: str,
        subject: Optional[str] = None,
        recipients: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Send a notification.

        Args:
            channel: Notification channel
            message: Message content
            subject: Message subject (for email)
            recipients: Specific recipients (optional)
        """
        body: Dict[str, Any] = {"channel": channel, "message": message}
        if subject:
            body["subject"] = subject
        if recipients:
            body["recipients"] = recipients
        return await self._request("POST", "/api/v1/notifications/send", json=body)

    # =========================================================================
    # Costs
    # =========================================================================

    async def get_cost_dashboard(self, period: Optional[str] = None) -> Dict[str, Any]:
        """Get cost dashboard data.

        Args:
            period: Time period for costs
        """
        params: Dict[str, Any] = {}
        if period:
            params["period"] = period
        return await self._request("GET", "/api/v1/costs/dashboard", params=params)

    async def get_cost_breakdown(
        self,
        period: Optional[str] = None,
        group_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get cost breakdown.

        Args:
            period: Time period
            group_by: Grouping ('agent', 'user', 'operation')
        """
        params: Dict[str, Any] = {}
        if period:
            params["period"] = period
        if group_by:
            params["group_by"] = group_by
        return await self._request("GET", "/api/v1/costs/breakdown", params=params)

    async def get_cost_timeline(
        self,
        period: Optional[str] = None,
        granularity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get cost timeline data.

        Args:
            period: Time period
            granularity: Data granularity ('hour', 'day', 'week')
        """
        params: Dict[str, Any] = {}
        if period:
            params["period"] = period
        if granularity:
            params["granularity"] = granularity
        return await self._request("GET", "/api/v1/costs/timeline", params=params)

    async def get_cost_alerts(self) -> Dict[str, Any]:
        """Get active cost alerts."""
        return await self._request("GET", "/api/v1/costs/alerts")

    async def set_cost_budget(
        self,
        daily_limit: Optional[float] = None,
        monthly_limit: Optional[float] = None,
        alert_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Set cost budget limits.

        Args:
            daily_limit: Daily cost limit
            monthly_limit: Monthly cost limit
            alert_threshold: Alert threshold percentage (0-1)
        """
        body: Dict[str, Any] = {}
        if daily_limit is not None:
            body["daily_limit"] = daily_limit
        if monthly_limit is not None:
            body["monthly_limit"] = monthly_limit
        if alert_threshold is not None:
            body["alert_threshold"] = alert_threshold
        return await self._request("POST", "/api/v1/costs/budget", json=body)

    async def dismiss_cost_alert(self, alert_id: str) -> Dict[str, Any]:
        """Dismiss a cost alert.

        Args:
            alert_id: Alert ID to dismiss
        """
        return await self._request("POST", f"/api/v1/costs/alerts/{alert_id}/dismiss", json={})

    # =========================================================================
    # Onboarding
    # =========================================================================

    async def get_onboarding_status(self) -> Dict[str, Any]:
        """Get onboarding progress status."""
        return await self._request("GET", "/api/v1/onboarding/status")

    async def complete_onboarding(
        self,
        first_debate_id: Optional[str] = None,
        template_used: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mark onboarding as complete.

        Args:
            first_debate_id: ID of first created debate
            template_used: Template used during onboarding
        """
        body: Dict[str, Any] = {}
        if first_debate_id:
            body["first_debate_id"] = first_debate_id
        if template_used:
            body["template_used"] = template_used
        return await self._request("POST", "/api/v1/onboarding/complete", json=body)

    # =========================================================================
    # Decision Receipts
    # =========================================================================

    async def list_decision_receipts(
        self,
        verdict: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List decision receipts.

        Args:
            verdict: Filter by verdict
            limit: Maximum results
            offset: Pagination offset
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if verdict:
            params["verdict"] = verdict
        return await self._request("GET", "/api/v1/receipts", params=params)

    async def get_decision_receipt(self, receipt_id: str) -> Dict[str, Any]:
        """Get decision receipt details.

        Args:
            receipt_id: Receipt ID
        """
        return await self._request("GET", f"/api/v1/receipts/{receipt_id}")

    async def verify_decision_receipt(self, receipt_id: str) -> Dict[str, Any]:
        """Verify decision receipt integrity.

        Args:
            receipt_id: Receipt ID to verify
        """
        return await self._request("POST", f"/api/v1/receipts/{receipt_id}/verify", json={})

    # =========================================================================
    # Analytics
    # =========================================================================

    async def get_disagreement_analytics(
        self,
        period: str = "30d",
    ) -> Dict[str, Any]:
        """Get analytics on agent disagreements.

        Args:
            period: Time period ('7d', '30d', '90d', 'all')

        Returns:
            Disagreement patterns, common topics, and resolution rates
        """
        return await self._request(
            "GET",
            "/api/v1/analytics/disagreements",
            params={"period": period},
        )

    async def get_consensus_quality_analytics(
        self,
        period: str = "30d",
    ) -> Dict[str, Any]:
        """Get analytics on consensus quality.

        Args:
            period: Time period ('7d', '30d', '90d', 'all')

        Returns:
            Consensus quality metrics including strength, stability, and confidence
        """
        return await self._request(
            "GET",
            "/api/v1/analytics/consensus-quality",
            params={"period": period},
        )

    async def get_role_rotation_analytics(
        self,
        period: str = "30d",
    ) -> Dict[str, Any]:
        """Get analytics on role rotation effectiveness.

        Args:
            period: Time period ('7d', '30d', '90d', 'all')

        Returns:
            Role rotation patterns and impact on debate outcomes
        """
        return await self._request(
            "GET",
            "/api/v1/analytics/role-rotation",
            params={"period": period},
        )

    async def get_early_stop_analytics(
        self,
        period: str = "30d",
    ) -> Dict[str, Any]:
        """Get analytics on early stopping patterns.

        Args:
            period: Time period ('7d', '30d', '90d', 'all')

        Returns:
            Early stop rates, triggers, and quality impact
        """
        return await self._request(
            "GET",
            "/api/v1/analytics/early-stops",
            params={"period": period},
        )

    async def get_ranking_stats(self) -> Dict[str, Any]:
        """Get overall ranking statistics.

        Returns:
            Ranking distribution, rating changes, and leaderboard trends
        """
        return await self._request("GET", "/api/v1/ranking/stats")

    # =========================================================================
    # Knowledge Mound
    # =========================================================================

    async def query_knowledge_mound(
        self,
        query: str,
        types: Optional[List[str]] = None,
        depth: int = 2,
        include_relationships: bool = True,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Query the Knowledge Mound graph."""
        return await self._request(
            "POST",
            "/api/v1/knowledge/mound/query",
            json={
                "query": query,
                "types": types,
                "depth": depth,
                "include_relationships": include_relationships,
                "limit": limit,
            },
        )

    async def list_knowledge_nodes(
        self,
        node_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List Knowledge Mound nodes."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if node_type:
            params["type"] = node_type
        return await self._request("GET", "/api/v1/knowledge/mound/nodes", params=params)

    async def create_knowledge_node(
        self,
        content: str,
        node_type: str = "fact",
        confidence: float = 0.8,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        visibility: str = "workspace",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new Knowledge Mound node."""
        return await self._request(
            "POST",
            "/api/v1/knowledge/mound/nodes",
            json={
                "content": content,
                "node_type": node_type,
                "confidence": confidence,
                "source": source,
                "tags": tags or [],
                "visibility": visibility,
                "metadata": metadata or {},
            },
        )

    async def get_knowledge_node(self, node_id: str) -> Dict[str, Any]:
        """Get a specific Knowledge Mound node."""
        return await self._request("GET", f"/api/v1/knowledge/mound/nodes/{node_id}")

    async def get_knowledge_mound_stats(self) -> Dict[str, Any]:
        """Get Knowledge Mound statistics."""
        return await self._request("GET", "/api/v1/knowledge/mound/stats")

    async def create_knowledge_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        strength: float = 1.0,
        confidence: float = 0.8,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a relationship between two knowledge nodes."""
        return await self._request(
            "POST",
            "/api/v1/knowledge/mound/relationships",
            json={
                "source_id": source_id,
                "target_id": target_id,
                "relationship_type": relationship_type,
                "strength": strength,
                "confidence": confidence,
                "metadata": metadata or {},
            },
        )

    async def get_node_relationships(
        self,
        node_id: str,
        direction: str = "both",
    ) -> Dict[str, Any]:
        """Get relationships for a knowledge node."""
        return await self._request(
            "GET",
            f"/api/v1/knowledge/mound/nodes/{node_id}/relationships",
            params={"direction": direction},
        )

    async def traverse_knowledge_graph(
        self,
        node_id: str,
        depth: int = 2,
        direction: str = "both",
    ) -> Dict[str, Any]:
        """Traverse the knowledge graph from a starting node."""
        return await self._request(
            "GET",
            f"/api/v1/knowledge/mound/graph/{node_id}",
            params={"depth": depth, "direction": direction},
        )

    async def get_knowledge_lineage(
        self,
        node_id: str,
        max_depth: int = 5,
    ) -> Dict[str, Any]:
        """Get the derivation lineage of a knowledge node."""
        return await self._request(
            "GET",
            f"/api/v1/knowledge/mound/graph/{node_id}/lineage",
            params={"max_depth": max_depth},
        )

    async def get_stale_knowledge(
        self,
        max_age_days: int = 30,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Get stale knowledge items that need revalidation."""
        return await self._request(
            "GET",
            "/api/v1/knowledge/mound/stale",
            params={"max_age_days": max_age_days, "limit": limit},
        )

    async def revalidate_knowledge(
        self,
        node_id: str,
        valid: bool,
        new_confidence: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Revalidate a knowledge node."""
        body: Dict[str, Any] = {"valid": valid}
        if new_confidence is not None:
            body["new_confidence"] = new_confidence
        if notes:
            body["notes"] = notes
        return await self._request(
            "POST",
            f"/api/v1/knowledge/mound/revalidate/{node_id}",
            json=body,
        )

    # =========================================================================
    # Knowledge Mound - Sharing
    # =========================================================================

    async def share_knowledge(
        self,
        item_id: str,
        target_id: str,
        target_type: str = "workspace",
        permission: str = "read",
        expires_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Share a knowledge item with another workspace or user."""
        body: Dict[str, Any] = {
            "item_id": item_id,
            "target_id": target_id,
            "target_type": target_type,
            "permission": permission,
        }
        if expires_at:
            body["expires_at"] = expires_at
        return await self._request("POST", "/api/v1/knowledge/mound/share", json=body)

    async def get_shared_with_me(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get items shared with the current user/workspace."""
        return await self._request(
            "GET",
            "/api/v1/knowledge/mound/shared-with-me",
            params={"limit": limit, "offset": offset},
        )

    async def get_my_shares(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get items shared by the current user."""
        return await self._request(
            "GET",
            "/api/v1/knowledge/mound/my-shares",
            params={"limit": limit, "offset": offset},
        )

    async def revoke_share(self, share_id: str) -> Dict[str, Any]:
        """Revoke a share."""
        return await self._request(
            "DELETE",
            "/api/v1/knowledge/mound/share",
            json={"share_id": share_id},
        )

    # =========================================================================
    # Knowledge Mound - Federation
    # =========================================================================

    async def list_federated_regions(self) -> Dict[str, Any]:
        """List all federated regions."""
        return await self._request("GET", "/api/v1/knowledge/mound/federation/regions")

    async def get_federation_status(self) -> Dict[str, Any]:
        """Get federation status and health."""
        return await self._request("GET", "/api/v1/knowledge/mound/federation/status")

    async def sync_to_region(
        self,
        region_id: str,
        scope: str = "workspace",
        node_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Push knowledge to a remote region."""
        body: Dict[str, Any] = {"region_id": region_id, "scope": scope}
        if node_ids:
            body["node_ids"] = node_ids
        return await self._request(
            "POST",
            "/api/v1/knowledge/mound/federation/sync/push",
            json=body,
        )

    async def pull_from_region(
        self,
        region_id: str,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Pull knowledge from a remote region."""
        body: Dict[str, Any] = {"region_id": region_id, "limit": limit}
        if since:
            body["since"] = since
        return await self._request(
            "POST",
            "/api/v1/knowledge/mound/federation/sync/pull",
            json=body,
        )

    # =========================================================================
    # Knowledge Mound - Deduplication
    # =========================================================================

    async def get_duplicate_clusters(
        self,
        threshold: float = 0.9,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Find duplicate clusters by similarity threshold."""
        return await self._request(
            "GET",
            "/api/v1/knowledge/mound/dedup/clusters",
            params={"threshold": threshold, "limit": limit},
        )

    async def get_dedup_report(self) -> Dict[str, Any]:
        """Generate deduplication analysis report."""
        return await self._request("GET", "/api/v1/knowledge/mound/dedup/report")

    async def merge_duplicate_cluster(
        self,
        cluster_id: str,
        primary_id: Optional[str] = None,
        strategy: str = "highest_confidence",
    ) -> Dict[str, Any]:
        """Merge a specific duplicate cluster."""
        return await self._request(
            "POST",
            "/api/v1/knowledge/mound/dedup/merge",
            json={
                "cluster_id": cluster_id,
                "primary_id": primary_id,
                "strategy": strategy,
            },
        )

    # =========================================================================
    # Knowledge Mound - Contradictions
    # =========================================================================

    async def detect_contradictions(self, scope: str = "workspace") -> Dict[str, Any]:
        """Trigger contradiction detection scan."""
        return await self._request(
            "POST",
            "/api/v1/knowledge/mound/contradictions/detect",
            json={"scope": scope},
        )

    async def list_contradictions(
        self,
        status: str = "unresolved",
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List contradictions."""
        return await self._request(
            "GET",
            "/api/v1/knowledge/mound/contradictions",
            params={"status": status, "limit": limit, "offset": offset},
        )

    async def resolve_contradiction(
        self,
        contradiction_id: str,
        strategy: str,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolve a contradiction."""
        body: Dict[str, Any] = {"strategy": strategy}
        if notes:
            body["notes"] = notes
        return await self._request(
            "POST",
            f"/api/v1/knowledge/mound/contradictions/{contradiction_id}/resolve",
            json=body,
        )

    # =========================================================================
    # Knowledge Mound - Analytics
    # =========================================================================

    async def analyze_knowledge_coverage(
        self,
        topics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Analyze domain coverage by topic."""
        params: Dict[str, Any] = {}
        if topics:
            params["topics"] = ",".join(topics)
        return await self._request(
            "GET",
            "/api/v1/knowledge/mound/analytics/coverage",
            params=params,
        )

    async def analyze_knowledge_usage(
        self,
        period: str = "week",
        since: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze usage patterns over time."""
        params: Dict[str, Any] = {"period": period}
        if since:
            params["since"] = since
        return await self._request(
            "GET",
            "/api/v1/knowledge/mound/analytics/usage",
            params=params,
        )

    async def get_knowledge_quality_trend(
        self,
        period: str = "week",
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get quality metrics trend over time."""
        params: Dict[str, Any] = {"period": period}
        if metrics:
            params["metrics"] = ",".join(metrics)
        return await self._request(
            "GET",
            "/api/v1/knowledge/mound/analytics/quality/trend",
            params=params,
        )

    # =========================================================================
    # Knowledge Mound - Export
    # =========================================================================

    async def export_knowledge_d3(
        self,
        scope: str = "workspace",
        depth: int = 3,
    ) -> Dict[str, Any]:
        """Export knowledge graph as D3 JSON format for visualization."""
        return await self._request(
            "GET",
            "/api/v1/knowledge/mound/export/d3",
            params={"scope": scope, "depth": depth},
        )

    async def export_knowledge_graphml(
        self,
        scope: str = "workspace",
    ) -> str:
        """Export knowledge graph as GraphML XML format."""
        return await self._request(
            "GET",
            "/api/v1/knowledge/mound/export/graphml",
            params={"scope": scope},
        )

    # =========================================================================
    # Knowledge Mound - Extraction
    # =========================================================================

    async def extract_from_debate(
        self,
        debate_id: str,
        confidence_threshold: float = 0.7,
        auto_promote: bool = False,
    ) -> Dict[str, Any]:
        """Extract claims/knowledge from a debate."""
        return await self._request(
            "POST",
            "/api/v1/knowledge/mound/extraction/debate",
            json={
                "debate_id": debate_id,
                "confidence_threshold": confidence_threshold,
                "auto_promote": auto_promote,
            },
        )

    async def promote_extracted_claims(
        self,
        claim_ids: List[str],
        target_tier: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Promote extracted claims to main knowledge."""
        body: Dict[str, Any] = {"claim_ids": claim_ids}
        if target_tier:
            body["target_tier"] = target_tier
        return await self._request(
            "POST",
            "/api/v1/knowledge/mound/extraction/promote",
            json=body,
        )

    # =========================================================================
    # Knowledge Mound - Dashboard
    # =========================================================================

    async def get_knowledge_dashboard_health(self) -> Dict[str, Any]:
        """Get Knowledge Mound health status and recommendations."""
        return await self._request("GET", "/api/v1/knowledge/mound/dashboard/health")

    async def get_knowledge_dashboard_metrics(self) -> Dict[str, Any]:
        """Get detailed operational metrics."""
        return await self._request("GET", "/api/v1/knowledge/mound/dashboard/metrics")

    async def get_knowledge_dashboard_adapters(self) -> Dict[str, Any]:
        """Get adapter status and health."""
        return await self._request("GET", "/api/v1/knowledge/mound/dashboard/adapters")

    # =========================================================================
    # Integrations
    # =========================================================================

    async def list_integrations(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List all configured integrations."""
        return await self._request(
            "GET",
            "/api/integrations",
            params={"limit": limit, "offset": offset},
        )

    async def get_integration(self, integration_id: str) -> Dict[str, Any]:
        """Get a specific integration."""
        return await self._request("GET", f"/api/integrations/{integration_id}")

    async def create_integration(
        self,
        integration_type: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new integration."""
        body: Dict[str, Any] = {"type": integration_type}
        if name:
            body["name"] = name
        if config:
            body["config"] = config
        return await self._request("POST", "/api/integrations", json=body)

    async def update_integration(
        self,
        integration_id: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Update an integration."""
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if config is not None:
            body["config"] = config
        if enabled is not None:
            body["enabled"] = enabled
        return await self._request("PUT", f"/api/integrations/{integration_id}", json=body)

    async def delete_integration(self, integration_id: str) -> Dict[str, Any]:
        """Delete an integration."""
        return await self._request("DELETE", f"/api/integrations/{integration_id}")

    async def test_integration(self, integration_id: str) -> Dict[str, Any]:
        """Test an integration connection."""
        return await self._request("POST", f"/api/integrations/{integration_id}/test")

    async def sync_integration(self, integration_id: str) -> Dict[str, Any]:
        """Trigger a sync for an integration."""
        return await self._request("POST", f"/api/integrations/{integration_id}/sync")

    async def get_bot_status(self, platform: str) -> Dict[str, Any]:
        """Get bot status for a platform (slack, telegram, whatsapp, discord)."""
        return await self._request("GET", f"/api/v1/bots/{platform}/status")

    async def get_teams_status(self) -> Dict[str, Any]:
        """Get Microsoft Teams integration status."""
        return await self._request("GET", "/api/v1/integrations/teams/status")

    async def list_zapier_apps(self) -> Dict[str, Any]:
        """List Zapier apps."""
        return await self._request("GET", "/api/v1/integrations/zapier/apps")

    async def create_zapier_app(
        self,
        name: str,
        triggers: Optional[List[str]] = None,
        actions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a Zapier app."""
        return await self._request(
            "POST",
            "/api/v1/integrations/zapier/apps",
            json={"name": name, "triggers": triggers, "actions": actions},
        )

    async def list_make_connections(self) -> Dict[str, Any]:
        """List Make (Integromat) connections."""
        return await self._request("GET", "/api/v1/integrations/make/connections")

    async def create_make_connection(
        self,
        name: str,
        credentials: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a Make connection."""
        return await self._request(
            "POST",
            "/api/v1/integrations/make/connections",
            json={"name": name, "credentials": credentials},
        )

    async def list_n8n_credentials(self) -> Dict[str, Any]:
        """List n8n credentials."""
        return await self._request("GET", "/api/v1/integrations/n8n/credentials")

    async def start_integration_wizard(
        self,
        integration_type: str,
    ) -> Dict[str, Any]:
        """Start the integration wizard."""
        return await self._request(
            "POST",
            "/api/v2/integrations/wizard",
            json={"type": integration_type},
        )

    async def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return await self._request("GET", "/api/v2/integrations/stats")

    # =========================================================================
    # Webhooks
    # =========================================================================

    async def list_webhooks(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List all webhooks."""
        return await self._request(
            "GET",
            "/api/webhooks",
            params={"limit": limit, "offset": offset},
        )

    async def get_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """Get a specific webhook."""
        return await self._request("GET", f"/api/webhooks/{webhook_id}")

    async def create_webhook(
        self,
        url: str,
        events: List[str],
        secret: Optional[str] = None,
        active: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new webhook."""
        body: Dict[str, Any] = {"url": url, "events": events, "active": active}
        if secret:
            body["secret"] = secret
        if metadata:
            body["metadata"] = metadata
        return await self._request("POST", "/api/webhooks", json=body)

    async def update_webhook(
        self,
        webhook_id: str,
        url: Optional[str] = None,
        events: Optional[List[str]] = None,
        secret: Optional[str] = None,
        active: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Update a webhook."""
        body: Dict[str, Any] = {}
        if url is not None:
            body["url"] = url
        if events is not None:
            body["events"] = events
        if secret is not None:
            body["secret"] = secret
        if active is not None:
            body["active"] = active
        return await self._request("PUT", f"/api/webhooks/{webhook_id}", json=body)

    async def delete_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """Delete a webhook."""
        return await self._request("DELETE", f"/api/webhooks/{webhook_id}")

    async def test_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """Test a webhook by sending a test event."""
        return await self._request("POST", f"/api/webhooks/{webhook_id}/test")

    async def list_webhook_events(self) -> Dict[str, Any]:
        """List available webhook events."""
        return await self._request("GET", "/api/webhooks/events")

    async def get_webhook_slo_status(self) -> Dict[str, Any]:
        """Get webhook SLO status."""
        return await self._request("GET", "/api/webhooks/slo")

    async def list_webhook_deliveries(
        self,
        webhook_id: str,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List webhook deliveries."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        return await self._request(
            "GET",
            f"/api/v1/webhooks/{webhook_id}/deliveries",
            params=params,
        )

    async def retry_webhook_delivery(
        self,
        webhook_id: str,
        delivery_id: str,
    ) -> Dict[str, Any]:
        """Retry a failed webhook delivery."""
        return await self._request(
            "POST",
            f"/api/v1/webhooks/{webhook_id}/deliveries/{delivery_id}/retry",
        )

    async def rotate_webhook_secret(self, webhook_id: str) -> Dict[str, Any]:
        """Rotate a webhook's secret."""
        return await self._request("POST", f"/api/v1/webhooks/{webhook_id}/rotate-secret")


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

    def publish_template(
        self, template_id: str, name: str, description: str, category: str, **kwargs: Any
    ) -> Dict[str, Any]:
        return self._run(
            self._async_client.publish_template(template_id, name, description, category, **kwargs)
        )

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

    # WebSocket
    def create_websocket(self, **kwargs: Any) -> Any:
        return self._async_client.create_websocket(**kwargs)

    # Control Plane - Agent Registry
    def register_agent(self, agent_id: str, agent_type: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.register_agent(agent_id, agent_type, **kwargs))

    def deregister_agent(self, agent_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.deregister_agent(agent_id))

    def list_registered_agents(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.list_registered_agents(**kwargs))

    def get_agent_health(self, agent_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_agent_health(agent_id))

    def send_agent_heartbeat(self, agent_id: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.send_agent_heartbeat(agent_id, **kwargs))

    # Control Plane - Task Scheduler
    def schedule_task(
        self, task_type: str, payload: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        return self._run(self._async_client.schedule_task(task_type, payload, **kwargs))

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_task_status(task_id))

    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.cancel_task(task_id))

    def list_tasks(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.list_tasks(**kwargs))

    # Control Plane - Policies
    def create_policy(
        self, name: str, rules: List[Dict[str, Any]], **kwargs: Any
    ) -> Dict[str, Any]:
        return self._run(self._async_client.create_policy(name, rules, **kwargs))

    def get_policy(self, policy_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_policy(policy_id))

    def update_policy(self, policy_id: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.update_policy(policy_id, **kwargs))

    def delete_policy(self, policy_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.delete_policy(policy_id))

    def list_policies(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.list_policies(**kwargs))

    # Graph Debates
    def create_graph_debate(
        self, question: str, agents: List[str], graph_structure: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        return self._run(
            self._async_client.create_graph_debate(question, agents, graph_structure, **kwargs)
        )

    def get_graph_debate_topology(self, debate_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_graph_debate_topology(debate_id))

    def update_graph_edge(
        self, debate_id: str, source: str, target: str, weight: float, **kwargs: Any
    ) -> Dict[str, Any]:
        return self._run(
            self._async_client.update_graph_edge(debate_id, source, target, weight, **kwargs)
        )

    def get_graph_path(
        self, debate_id: str, start_node: str, end_node: str, **kwargs: Any
    ) -> Dict[str, Any]:
        return self._run(
            self._async_client.get_graph_path(debate_id, start_node, end_node, **kwargs)
        )

    # Matrix Debates
    def create_matrix_debate(
        self, question: str, agents: List[str], dimensions: List[str], **kwargs: Any
    ) -> Dict[str, Any]:
        return self._run(
            self._async_client.create_matrix_debate(question, agents, dimensions, **kwargs)
        )

    def get_matrix_analysis(self, debate_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_matrix_analysis(debate_id))

    def update_matrix_cell(
        self, debate_id: str, row: str, column: str, value: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        return self._run(
            self._async_client.update_matrix_cell(debate_id, row, column, value, **kwargs)
        )

    def get_matrix_summary(self, debate_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_matrix_summary(debate_id))

    # Agent Intelligence
    def get_agent_profile(self, agent_name: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_agent_profile(agent_name))

    def get_agent_history(self, agent_name: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.get_agent_history(agent_name, **kwargs))

    def get_agent_leaderboard(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.get_agent_leaderboard(**kwargs))

    def compare_agents(self, agent_names: List[str], **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.compare_agents(agent_names, **kwargs))

    # Verification
    def verify_claim(self, claim: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.verify_claim(claim, **kwargs))

    def get_verification_status(self, verification_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_verification_status(verification_id))

    def get_verification_evidence(self, verification_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_verification_evidence(verification_id))

    # Memory
    def store_memory(self, content: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.store_memory(content, **kwargs))

    def retrieve_memory(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.retrieve_memory(query, **kwargs))

    def get_memory_stats(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_memory_stats())

    # Knowledge Mound
    def search_knowledge(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.search_knowledge(query, **kwargs))

    def add_knowledge(self, content: str, source: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.add_knowledge(content, source, **kwargs))

    def get_knowledge_stats(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_knowledge_stats())

    # ELO Rankings
    def get_elo_rankings(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.get_elo_rankings(**kwargs))

    def get_agent_elo(self, agent_name: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_agent_elo(agent_name))

    def get_elo_history(self, agent_name: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.get_elo_history(agent_name, **kwargs))

    # Tournaments
    def create_tournament(
        self, name: str, agents: List[str], questions: List[str], **kwargs: Any
    ) -> Dict[str, Any]:
        return self._run(self._async_client.create_tournament(name, agents, questions, **kwargs))

    def get_tournament(self, tournament_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_tournament(tournament_id))

    def list_tournaments(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.list_tournaments(**kwargs))

    def get_tournament_standings(self, tournament_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_tournament_standings(tournament_id))

    # =========================================================================
    # Authentication
    # =========================================================================

    def register_user(self, email: str, password: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.register_user(email, password, **kwargs))

    def login(self, email: str, password: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.login(email, password, **kwargs))

    def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        return self._run(self._async_client.refresh_token(refresh_token))

    def logout(self) -> None:
        return self._run(self._async_client.logout())

    def logout_all(self) -> None:
        return self._run(self._async_client.logout_all())

    def verify_email(self, token: str) -> Dict[str, Any]:
        return self._run(self._async_client.verify_email(token))

    def resend_verification(self, email: str) -> Dict[str, Any]:
        return self._run(self._async_client.resend_verification(email))

    def get_current_user(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_current_user())

    def update_profile(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.update_profile(**kwargs))

    def change_password(self, current_password: str, new_password: str) -> None:
        return self._run(self._async_client.change_password(current_password, new_password))

    def request_password_reset(self, email: str) -> None:
        return self._run(self._async_client.request_password_reset(email))

    def reset_password(self, token: str, new_password: str) -> None:
        return self._run(self._async_client.reset_password(token, new_password))

    # =========================================================================
    # Multi-Factor Authentication
    # =========================================================================

    def setup_mfa(self, mfa_type: str = "totp") -> Dict[str, Any]:
        return self._run(self._async_client.setup_mfa(mfa_type))

    def verify_mfa_setup(self, code: str) -> Dict[str, Any]:
        return self._run(self._async_client.verify_mfa_setup(code))

    def enable_mfa(self, code: str) -> Dict[str, Any]:
        return self._run(self._async_client.enable_mfa(code))

    def disable_mfa(self) -> None:
        return self._run(self._async_client.disable_mfa())

    def generate_backup_codes(self) -> Dict[str, Any]:
        return self._run(self._async_client.generate_backup_codes())

    # =========================================================================
    # Session Management
    # =========================================================================

    def list_sessions(self) -> Dict[str, Any]:
        return self._run(self._async_client.list_sessions())

    def revoke_session(self, session_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.revoke_session(session_id))

    # =========================================================================
    # API Key Management
    # =========================================================================

    def list_api_keys(self) -> Dict[str, Any]:
        return self._run(self._async_client.list_api_keys())

    def create_api_key(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.create_api_key(name, **kwargs))

    def revoke_api_key(self, key_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.revoke_api_key(key_id))

    # =========================================================================
    # OAuth & SSO
    # =========================================================================

    def get_oauth_url(self, provider: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.get_oauth_url(provider, **kwargs))

    def complete_oauth(self, code: str, state: str, provider: str) -> Dict[str, Any]:
        return self._run(self._async_client.complete_oauth(code, state, provider))

    def list_oauth_providers(self) -> Dict[str, Any]:
        return self._run(self._async_client.list_oauth_providers())

    def link_oauth_provider(self, provider: str, code: str) -> Dict[str, Any]:
        return self._run(self._async_client.link_oauth_provider(provider, code))

    def unlink_oauth_provider(self, provider: str) -> Dict[str, Any]:
        return self._run(self._async_client.unlink_oauth_provider(provider))

    def initiate_sso_login(self, domain: str) -> Dict[str, Any]:
        return self._run(self._async_client.initiate_sso_login(domain))

    def list_sso_providers(self) -> Dict[str, Any]:
        return self._run(self._async_client.list_sso_providers())

    # =========================================================================
    # Invitations
    # =========================================================================

    def invite_team_member(self, email: str, role: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.invite_team_member(email, role, **kwargs))

    def check_invite(self, token: str) -> Dict[str, Any]:
        return self._run(self._async_client.check_invite(token))

    def accept_invite(self, token: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.accept_invite(token, **kwargs))

    # =========================================================================
    # Tenancy
    # =========================================================================

    def list_tenants(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.list_tenants(**kwargs))

    def get_tenant(self, tenant_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_tenant(tenant_id))

    def create_tenant(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.create_tenant(name, **kwargs))

    def update_tenant(self, tenant_id: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.update_tenant(tenant_id, **kwargs))

    def delete_tenant(self, tenant_id: str) -> None:
        return self._run(self._async_client.delete_tenant(tenant_id))

    def get_tenant_quotas(self, tenant_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_tenant_quotas(tenant_id))

    def update_tenant_quotas(self, tenant_id: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.update_tenant_quotas(tenant_id, **kwargs))

    def list_tenant_members(self, tenant_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.list_tenant_members(tenant_id))

    def add_tenant_member(self, tenant_id: str, email: str, role: str) -> Dict[str, Any]:
        return self._run(self._async_client.add_tenant_member(tenant_id, email, role))

    def remove_tenant_member(self, tenant_id: str, user_id: str) -> None:
        return self._run(self._async_client.remove_tenant_member(tenant_id, user_id))

    def setup_organization(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.setup_organization(name, **kwargs))

    # =========================================================================
    # RBAC (Role-Based Access Control)
    # =========================================================================

    def list_roles(self) -> Dict[str, Any]:
        return self._run(self._async_client.list_roles())

    def get_role(self, role_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_role(role_id))

    def create_role(self, name: str, permissions: List[str], **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.create_role(name, permissions, **kwargs))

    def update_role(self, role_id: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.update_role(role_id, **kwargs))

    def delete_role(self, role_id: str) -> None:
        return self._run(self._async_client.delete_role(role_id))

    def list_permissions(self) -> Dict[str, Any]:
        return self._run(self._async_client.list_permissions())

    def assign_role(self, user_id: str, role_id: str) -> None:
        return self._run(self._async_client.assign_role(user_id, role_id))

    def revoke_role(self, user_id: str, role_id: str) -> None:
        return self._run(self._async_client.revoke_role(user_id, role_id))

    def get_user_roles(self, user_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_user_roles(user_id))

    def check_permission(self, user_id: str, permission: str) -> Dict[str, Any]:
        return self._run(self._async_client.check_permission(user_id, permission))

    def list_role_assignments(self, role_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.list_role_assignments(role_id))

    def bulk_assign_roles(self, assignments: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._run(self._async_client.bulk_assign_roles(assignments))

    # =========================================================================
    # Billing
    # =========================================================================

    def list_billing_plans(self) -> Dict[str, Any]:
        return self._run(self._async_client.list_billing_plans())

    def get_billing_usage(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.get_billing_usage(**kwargs))

    def get_subscription(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_subscription())

    def create_checkout_session(
        self, plan_id: str, success_url: str, cancel_url: str
    ) -> Dict[str, Any]:
        return self._run(
            self._async_client.create_checkout_session(plan_id, success_url, cancel_url)
        )

    def create_billing_portal_session(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.create_billing_portal_session(**kwargs))

    def cancel_subscription(self) -> Dict[str, Any]:
        return self._run(self._async_client.cancel_subscription())

    def resume_subscription(self) -> Dict[str, Any]:
        return self._run(self._async_client.resume_subscription())

    def get_invoice_history(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_invoice_history())

    def get_usage_forecast(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_usage_forecast())

    def export_usage_data(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.export_usage_data(**kwargs))

    # =========================================================================
    # Budgets
    # =========================================================================

    def list_budgets(self) -> Dict[str, Any]:
        return self._run(self._async_client.list_budgets())

    def create_budget(
        self, name: str, limit_amount: float, period: str, **kwargs: Any
    ) -> Dict[str, Any]:
        return self._run(self._async_client.create_budget(name, limit_amount, period, **kwargs))

    def get_budget(self, budget_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_budget(budget_id))

    def update_budget(self, budget_id: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.update_budget(budget_id, **kwargs))

    def delete_budget(self, budget_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.delete_budget(budget_id))

    def get_budget_alerts(self, budget_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_budget_alerts(budget_id))

    def acknowledge_budget_alert(self, budget_id: str, alert_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.acknowledge_budget_alert(budget_id, alert_id))

    def check_budget(self, operation: str, estimated_cost: float) -> Dict[str, Any]:
        return self._run(self._async_client.check_budget(operation, estimated_cost))

    def get_budget_summary(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_budget_summary())

    def add_budget_override(self, budget_id: str, user_id: str, limit: float) -> Dict[str, Any]:
        return self._run(self._async_client.add_budget_override(budget_id, user_id, limit))

    def remove_budget_override(self, budget_id: str, user_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.remove_budget_override(budget_id, user_id))

    def reset_budget(self, budget_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.reset_budget(budget_id))

    # =========================================================================
    # Audit
    # =========================================================================

    def list_audit_events(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.list_audit_events(**kwargs))

    def get_audit_stats(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.get_audit_stats(**kwargs))

    def export_audit_logs(self, start_date: str, end_date: str) -> Dict[str, Any]:
        return self._run(self._async_client.export_audit_logs(start_date, end_date))

    def verify_audit_integrity(self) -> Dict[str, Any]:
        return self._run(self._async_client.verify_audit_integrity())

    def list_audit_trails(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.list_audit_trails(**kwargs))

    def get_audit_trail(self, trail_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_audit_trail(trail_id))

    def verify_audit_trail(self, trail_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.verify_audit_trail(trail_id))

    def export_audit_trail(self, trail_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.export_audit_trail(trail_id))

    # =========================================================================
    # Notifications
    # =========================================================================

    def get_notification_status(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_notification_status())

    def configure_email_notifications(
        self, provider: str, from_email: str, **kwargs: Any
    ) -> Dict[str, Any]:
        return self._run(
            self._async_client.configure_email_notifications(provider, from_email, **kwargs)
        )

    def configure_telegram_notifications(self, bot_token: str, chat_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.configure_telegram_notifications(bot_token, chat_id))

    def get_email_recipients(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_email_recipients())

    def add_email_recipient(self, email: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.add_email_recipient(email, **kwargs))

    def remove_email_recipient(self, recipient_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.remove_email_recipient(recipient_id))

    def send_test_notification(self, channel: str) -> Dict[str, Any]:
        return self._run(self._async_client.send_test_notification(channel))

    def send_notification(self, channel: str, message: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.send_notification(channel, message, **kwargs))

    # =========================================================================
    # Costs
    # =========================================================================

    def get_cost_dashboard(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.get_cost_dashboard(**kwargs))

    def get_cost_breakdown(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.get_cost_breakdown(**kwargs))

    def get_cost_timeline(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.get_cost_timeline(**kwargs))

    def get_cost_alerts(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_cost_alerts())

    def set_cost_budget(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.set_cost_budget(**kwargs))

    def dismiss_cost_alert(self, alert_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.dismiss_cost_alert(alert_id))

    # =========================================================================
    # Onboarding
    # =========================================================================

    def get_onboarding_status(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_onboarding_status())

    def complete_onboarding(self, step: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.complete_onboarding(step, **kwargs))

    # =========================================================================
    # Decision Receipts
    # =========================================================================

    def list_decision_receipts(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.list_decision_receipts(**kwargs))

    def get_decision_receipt(self, receipt_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_decision_receipt(receipt_id))

    def verify_decision_receipt(self, receipt_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.verify_decision_receipt(receipt_id))

    # =========================================================================
    # Gauntlet (Extended)
    # =========================================================================

    def run_gauntlet(self, input: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.run_gauntlet(input, **kwargs))

    def get_gauntlet_status(self, gauntlet_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_gauntlet_status(gauntlet_id))

    def delete_gauntlet(self, gauntlet_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.delete_gauntlet(gauntlet_id))

    def list_gauntlet_personas(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.list_gauntlet_personas(**kwargs))

    def list_gauntlet_results(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.list_gauntlet_results(**kwargs))

    def get_gauntlet_heatmap(self, gauntlet_id: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.get_gauntlet_heatmap(gauntlet_id, **kwargs))

    def compare_gauntlets(self, gauntlet_id_1: str, gauntlet_id_2: str) -> Dict[str, Any]:
        return self._run(self._async_client.compare_gauntlets(gauntlet_id_1, gauntlet_id_2))

    def list_gauntlet_receipts(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.list_gauntlet_receipts(**kwargs))

    def get_gauntlet_receipt(self, receipt_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_gauntlet_receipt(receipt_id))

    def verify_gauntlet_receipt(self, receipt_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.verify_gauntlet_receipt(receipt_id))

    def export_gauntlet_receipt(self, receipt_id: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.export_gauntlet_receipt(receipt_id, **kwargs))

    # =========================================================================
    # Agent Deep Dive
    # =========================================================================

    def get_agent_consistency(self, agent_name: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_agent_consistency(agent_name))

    def get_agent_flips(self, agent_name: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.get_agent_flips(agent_name, **kwargs))

    def get_agent_moments(self, agent_name: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.get_agent_moments(agent_name, **kwargs))

    def get_agent_opponent_briefing(self, agent_name: str, opponent_name: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_agent_opponent_briefing(agent_name, opponent_name))

    def get_leaderboard(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_leaderboard())

    # =========================================================================
    # Analytics
    # =========================================================================

    def get_disagreement_analytics(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.get_disagreement_analytics(**kwargs))

    def get_consensus_quality_analytics(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.get_consensus_quality_analytics(**kwargs))

    def get_role_rotation_analytics(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.get_role_rotation_analytics(**kwargs))

    def get_early_stop_analytics(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.get_early_stop_analytics(**kwargs))

    def get_ranking_stats(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_ranking_stats())

    # =========================================================================
    # Integrations
    # =========================================================================

    def list_integrations(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        return self._run(self._async_client.list_integrations(limit, offset))

    def get_integration(self, integration_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_integration(integration_id))

    def create_integration(self, integration_type: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.create_integration(integration_type, **kwargs))

    def update_integration(self, integration_id: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.update_integration(integration_id, **kwargs))

    def delete_integration(self, integration_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.delete_integration(integration_id))

    def test_integration(self, integration_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.test_integration(integration_id))

    def sync_integration(self, integration_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.sync_integration(integration_id))

    def get_bot_status(self, platform: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_bot_status(platform))

    def get_teams_status(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_teams_status())

    def list_zapier_apps(self) -> Dict[str, Any]:
        return self._run(self._async_client.list_zapier_apps())

    def create_zapier_app(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.create_zapier_app(name, **kwargs))

    def list_make_connections(self) -> Dict[str, Any]:
        return self._run(self._async_client.list_make_connections())

    def create_make_connection(self, name: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        return self._run(self._async_client.create_make_connection(name, credentials))

    def list_n8n_credentials(self) -> Dict[str, Any]:
        return self._run(self._async_client.list_n8n_credentials())

    def start_integration_wizard(self, integration_type: str) -> Dict[str, Any]:
        return self._run(self._async_client.start_integration_wizard(integration_type))

    def get_integration_stats(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_integration_stats())

    # =========================================================================
    # Webhooks
    # =========================================================================

    def list_webhooks(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        return self._run(self._async_client.list_webhooks(limit, offset))

    def get_webhook(self, webhook_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_webhook(webhook_id))

    def create_webhook(self, url: str, events: List[str], **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.create_webhook(url, events, **kwargs))

    def update_webhook(self, webhook_id: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.update_webhook(webhook_id, **kwargs))

    def delete_webhook(self, webhook_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.delete_webhook(webhook_id))

    def test_webhook(self, webhook_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.test_webhook(webhook_id))

    def list_webhook_events(self) -> Dict[str, Any]:
        return self._run(self._async_client.list_webhook_events())

    def get_webhook_slo_status(self) -> Dict[str, Any]:
        return self._run(self._async_client.get_webhook_slo_status())

    def list_webhook_deliveries(self, webhook_id: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.list_webhook_deliveries(webhook_id, **kwargs))

    def retry_webhook_delivery(self, webhook_id: str, delivery_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.retry_webhook_delivery(webhook_id, delivery_id))

    def rotate_webhook_secret(self, webhook_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.rotate_webhook_secret(webhook_id))
