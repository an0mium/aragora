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
