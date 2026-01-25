"""Main client for the Aragora SDK."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

import httpx

from aragora_client.audit import AuditAPI
from aragora_client.auth import AuthAPI
from aragora_client.control_plane import ControlPlaneAPI
from aragora_client.cross_pollination import CrossPollinationAPI
from aragora_client.decisions import DecisionsAPI
from aragora_client.exceptions import (
    AragoraAuthenticationError,
    AragoraConnectionError,
    AragoraError,
    AragoraNotFoundError,
    AragoraTimeoutError,
    AragoraValidationError,
)
from aragora_client.explainability import ExplainabilityAPI
from aragora_client.knowledge import KnowledgeAPI
from aragora_client.marketplace import MarketplaceAPI
from aragora_client.onboarding import OnboardingAPI
from aragora_client.rbac import RBACAPI
from aragora_client.replay import ReplayAPI
from aragora_client.tenancy import TenancyAPI
from aragora_client.threat_intel import ThreatIntelAPI
from aragora_client.tournaments import TournamentsAPI
from aragora_client.types import (
    AgentProfile,
    AgentScore,
    CreateDebateRequest,
    CreateGraphDebateRequest,
    CreateMatrixDebateRequest,
    Debate,
    GauntletReceipt,
    GraphBranch,
    GraphDebate,
    HealthStatus,
    MatrixConclusion,
    MatrixDebate,
    MemoryAnalytics,
    MemoryTierStats,
    RunGauntletRequest,
    ScoreAgentsRequest,
    SelectionPlugins,
    SelectTeamRequest,
    TeamSelection,
    VerificationResult,
    VerifyClaimRequest,
)
from aragora_client.workflows import WorkflowsAPI


class DebatesAPI:
    """API for debate operations."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def create(
        self,
        task: str,
        *,
        agents: list[str] | None = None,
        max_rounds: int = 5,
        consensus_threshold: float = 0.8,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a new debate."""
        request = CreateDebateRequest(
            task=task,
            agents=agents,
            max_rounds=max_rounds,
            consensus_threshold=consensus_threshold,
            metadata=kwargs.get("metadata", {}),
        )
        return await self._client._post("/api/v1/debates", request.model_dump())

    async def get(self, debate_id: str) -> Debate:
        """Get a debate by ID."""
        data = await self._client._get(f"/api/v1/debates/{debate_id}")
        return Debate.model_validate(data)

    async def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
    ) -> list[Debate]:
        """List debates."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        data = await self._client._get("/api/v1/debates", params=params)
        return [Debate.model_validate(d) for d in data.get("debates", [])]

    async def run(
        self,
        task: str,
        *,
        agents: list[str] | None = None,
        max_rounds: int = 5,
        consensus_threshold: float = 0.8,
        poll_interval: float = 1.0,
        timeout: float = 300.0,
        **kwargs: Any,
    ) -> Debate:
        """Run a debate and wait for completion."""
        response = await self.create(
            task,
            agents=agents,
            max_rounds=max_rounds,
            consensus_threshold=consensus_threshold,
            **kwargs,
        )
        debate_id = response["id"]

        elapsed = 0.0
        while elapsed < timeout:
            debate = await self.get(debate_id)
            if debate.status.value in ("completed", "failed", "cancelled"):
                return debate
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise AragoraTimeoutError(
            f"Debate {debate_id} did not complete within {timeout}s"
        )

    async def update(
        self,
        debate_id: str,
        *,
        title: str | None = None,
        status: str | None = None,
        tags: list[str] | None = None,
    ) -> Debate:
        """Update debate metadata.

        Args:
            debate_id: Debate ID
            title: New title
            status: New status (active, paused, concluded, archived)
            tags: Updated tags

        Returns:
            Updated debate
        """
        body: dict[str, Any] = {}
        if title is not None:
            body["title"] = title
        if status is not None:
            body["status"] = status
        if tags is not None:
            body["tags"] = tags

        data = await self._client._patch(f"/api/v1/debates/{debate_id}", body)
        return Debate.model_validate(data)

    async def cancel(self, debate_id: str) -> dict[str, Any]:
        """Cancel a running debate.

        Args:
            debate_id: Debate ID

        Returns:
            Cancellation result
        """
        return await self._client._post(f"/api/v1/debates/{debate_id}/cancel", {})

    async def get_messages(
        self,
        debate_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get paginated message history for a debate.

        Args:
            debate_id: Debate ID
            limit: Maximum messages to return
            offset: Pagination offset

        Returns:
            List of debate messages
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        data = await self._client._get(
            f"/api/v1/debates/{debate_id}/messages", params=params
        )
        return data.get("messages", [])

    async def get_convergence(self, debate_id: str) -> dict[str, Any]:
        """Get convergence status for a debate.

        Args:
            debate_id: Debate ID

        Returns:
            Convergence analysis
        """
        return await self._client._get(f"/api/v1/debates/{debate_id}/convergence")

    async def get_citations(self, debate_id: str) -> list[dict[str, Any]]:
        """Get evidence citations for a debate.

        Args:
            debate_id: Debate ID

        Returns:
            List of citations
        """
        data = await self._client._get(f"/api/v1/debates/{debate_id}/citations")
        return data.get("citations", [])

    async def get_evidence(self, debate_id: str) -> dict[str, Any]:
        """Get comprehensive evidence trail for a debate.

        Args:
            debate_id: Debate ID

        Returns:
            Evidence chain
        """
        return await self._client._get(f"/api/v1/debates/{debate_id}/evidence")

    async def get_impasse(self, debate_id: str) -> dict[str, Any]:
        """Detect debate impasse.

        Args:
            debate_id: Debate ID

        Returns:
            Impasse analysis
        """
        return await self._client._get(f"/api/v1/debates/{debate_id}/impasse")

    async def get_summary(self, debate_id: str) -> dict[str, Any]:
        """Get human-readable summary of a debate.

        Args:
            debate_id: Debate ID

        Returns:
            Debate summary
        """
        return await self._client._get(f"/api/v1/debates/{debate_id}/summary")

    async def get_verification_report(self, debate_id: str) -> dict[str, Any]:
        """Get verification feedback for a debate.

        Args:
            debate_id: Debate ID

        Returns:
            Verification report
        """
        return await self._client._get(
            f"/api/v1/debates/{debate_id}/verification-report"
        )

    async def fork(
        self,
        debate_id: str,
        *,
        branch_point: int,
        modified_context: str | None = None,
    ) -> dict[str, Any]:
        """Fork a debate at a specific branch point.

        Args:
            debate_id: Debate ID
            branch_point: Round number to fork at
            modified_context: Alternative context for the fork

        Returns:
            Forked debate info
        """
        body: dict[str, Any] = {"branch_point": branch_point}
        if modified_context:
            body["modified_context"] = modified_context

        return await self._client._post(f"/api/v1/debates/{debate_id}/fork", body)

    async def list_forks(self, debate_id: str) -> list[dict[str, Any]]:
        """List all forks of a debate.

        Args:
            debate_id: Debate ID

        Returns:
            List of forked debates
        """
        data = await self._client._get(f"/api/v1/debates/{debate_id}/forks")
        return data.get("forks", [])

    async def create_followup(
        self,
        debate_id: str,
        *,
        crux: str,
        question: str | None = None,
    ) -> dict[str, Any]:
        """Create a crux-driven follow-up debate.

        Args:
            debate_id: Parent debate ID
            crux: The crux point to explore
            question: Optional alternative question

        Returns:
            Follow-up debate info
        """
        body: dict[str, Any] = {"crux": crux}
        if question:
            body["question"] = question

        return await self._client._post(f"/api/v1/debates/{debate_id}/followup", body)

    async def get_followup_suggestions(self, debate_id: str) -> list[dict[str, Any]]:
        """Get follow-up debate suggestions.

        Args:
            debate_id: Debate ID

        Returns:
            List of suggested follow-up topics
        """
        data = await self._client._get(f"/api/v1/debates/{debate_id}/followups")
        return data.get("suggestions", [])

    async def export(
        self,
        debate_id: str,
        format: str = "json",
        *,
        tables: list[str] | None = None,
    ) -> bytes:
        """Export debate in various formats.

        Args:
            debate_id: Debate ID
            format: Export format (json, csv, html, txt, md)
            tables: Tables to include (summary, messages, critiques, votes)

        Returns:
            Exported data
        """
        params: dict[str, Any] = {}
        if tables:
            params["tables"] = ",".join(tables)

        return await self._client._get_raw(
            f"/api/v1/debates/{debate_id}/export/{format}",
            params=params or None,
        )

    async def get_rhetorical(self, debate_id: str) -> dict[str, Any]:
        """Get rhetorical pattern observations for a debate.

        Args:
            debate_id: Debate ID

        Returns:
            Rhetorical analysis
        """
        return await self._client._get(f"/api/v1/debates/{debate_id}/rhetorical")

    async def get_trickster(self, debate_id: str) -> dict[str, Any]:
        """Get trickster hollow consensus status for a debate.

        Args:
            debate_id: Debate ID

        Returns:
            Trickster status
        """
        return await self._client._get(f"/api/v1/debates/{debate_id}/trickster")

    async def get_meta_critique(self, debate_id: str) -> dict[str, Any]:
        """Get meta-level debate analysis.

        Args:
            debate_id: Debate ID

        Returns:
            Meta-critique analysis
        """
        return await self._client._get(f"/api/v1/debates/{debate_id}/meta-critique")

    async def get_graph_stats(self, debate_id: str) -> dict[str, Any]:
        """Get argument graph statistics.

        Args:
            debate_id: Debate ID

        Returns:
            Graph statistics
        """
        return await self._client._get(f"/api/v1/debates/{debate_id}/graph/stats")

    async def search(
        self,
        query: str,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Search across debates.

        Args:
            query: Search query
            limit: Maximum results
            offset: Pagination offset

        Returns:
            Search results
        """
        params: dict[str, Any] = {"query": query, "limit": limit, "offset": offset}
        data = await self._client._get("/api/v1/search", params=params)
        return data.get("results", [])

    async def batch_submit(
        self,
        items: list[dict[str, Any]],
        *,
        webhook_url: str | None = None,
        max_parallel: int = 10,
    ) -> dict[str, Any]:
        """Submit a batch of debates.

        Args:
            items: List of debate specifications
            webhook_url: URL to notify on completion
            max_parallel: Maximum parallel debates

        Returns:
            Batch submission info with status URL
        """
        body: dict[str, Any] = {"items": items, "max_parallel": max_parallel}
        if webhook_url:
            body["webhook_url"] = webhook_url

        return await self._client._post("/api/v1/debates/batch", body)

    async def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        """Get status of a batch debate submission.

        Args:
            batch_id: Batch ID

        Returns:
            Batch status
        """
        return await self._client._get(f"/api/v1/debates/batch/{batch_id}/status")

    async def get_queue_status(self) -> dict[str, Any]:
        """Get debate queue status.

        Returns:
            Queue status and metrics
        """
        return await self._client._get("/api/v1/debates/queue/status")


class GraphDebatesAPI:
    """API for graph debate operations."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def create(
        self,
        task: str,
        *,
        agents: list[str] | None = None,
        max_rounds: int = 5,
        branch_threshold: float = 0.5,
        max_branches: int = 10,
    ) -> dict[str, Any]:
        """Create a new graph debate."""
        request = CreateGraphDebateRequest(
            task=task,
            agents=agents,
            max_rounds=max_rounds,
            branch_threshold=branch_threshold,
            max_branches=max_branches,
        )
        return await self._client._post("/api/v1/graph-debates", request.model_dump())

    async def get(self, debate_id: str) -> GraphDebate:
        """Get a graph debate by ID."""
        data = await self._client._get(f"/api/v1/graph-debates/{debate_id}")
        return GraphDebate.model_validate(data)

    async def get_branches(self, debate_id: str) -> list[GraphBranch]:
        """Get branches for a graph debate."""
        data = await self._client._get(f"/api/v1/graph-debates/{debate_id}/branches")
        return [GraphBranch.model_validate(b) for b in data.get("branches", [])]


class MatrixDebatesAPI:
    """API for matrix debate operations."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def create(
        self,
        task: str,
        scenarios: list[dict[str, Any]],
        *,
        agents: list[str] | None = None,
        max_rounds: int = 3,
    ) -> dict[str, Any]:
        """Create a new matrix debate."""
        request = CreateMatrixDebateRequest(
            task=task,
            scenarios=scenarios,
            agents=agents,
            max_rounds=max_rounds,
        )
        return await self._client._post("/api/v1/matrix-debates", request.model_dump())

    async def get(self, debate_id: str) -> MatrixDebate:
        """Get a matrix debate by ID."""
        data = await self._client._get(f"/api/v1/matrix-debates/{debate_id}")
        return MatrixDebate.model_validate(data)

    async def get_conclusions(self, debate_id: str) -> MatrixConclusion:
        """Get conclusions for a matrix debate."""
        data = await self._client._get(
            f"/api/v1/matrix-debates/{debate_id}/conclusions"
        )
        return MatrixConclusion.model_validate(data)


class AgentsAPI:
    """API for agent operations."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def list(self) -> list[AgentProfile]:
        """List all available agents."""
        data = await self._client._get("/api/v1/agents")
        return [AgentProfile.model_validate(a) for a in data.get("agents", [])]

    async def get(self, agent_id: str) -> AgentProfile:
        """Get an agent profile."""
        data = await self._client._get(f"/api/v1/agents/{agent_id}")
        return AgentProfile.model_validate(data)

    async def history(self, agent_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Get match history for an agent."""
        data = await self._client._get(
            f"/api/v1/agents/{agent_id}/history", params={"limit": limit}
        )
        return data.get("matches", [])

    async def rivals(self, agent_id: str) -> list[dict[str, Any]]:
        """Get rivals for an agent."""
        data = await self._client._get(f"/api/v1/agents/{agent_id}/rivals")
        return data.get("rivals", [])

    async def allies(self, agent_id: str) -> list[dict[str, Any]]:
        """Get allies for an agent."""
        data = await self._client._get(f"/api/v1/agents/{agent_id}/allies")
        return data.get("allies", [])

    async def get_calibration(self, agent_id: str) -> dict[str, Any]:
        """Get calibration data for an agent."""
        return await self._client._get(f"/api/v1/agents/{agent_id}/calibration")

    async def get_performance(self, agent_id: str) -> dict[str, Any]:
        """Get performance metrics for an agent."""
        return await self._client._get(f"/api/v1/agents/{agent_id}/performance")

    async def get_head_to_head(self, agent_id: str, opponent: str) -> dict[str, Any]:
        """Get head-to-head statistics against an opponent."""
        return await self._client._get(
            f"/api/v1/agents/{agent_id}/head-to-head/{opponent}"
        )

    async def get_opponent_briefing(
        self, agent_id: str, opponent: str
    ) -> dict[str, Any]:
        """Get opponent briefing for strategic planning."""
        return await self._client._get(
            f"/api/v1/agents/{agent_id}/opponent-briefing/{opponent}"
        )

    async def get_consistency(self, agent_id: str) -> dict[str, Any]:
        """Get consistency metrics for an agent."""
        return await self._client._get(f"/api/v1/agents/{agent_id}/consistency")

    async def get_flips(
        self,
        agent_id: str,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get position flips for an agent."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        data = await self._client._get(
            f"/api/v1/agents/{agent_id}/flips", params=params
        )
        return data.get("flips", [])

    async def get_network(self, agent_id: str) -> dict[str, Any]:
        """Get agent network graph."""
        return await self._client._get(f"/api/v1/agents/{agent_id}/network")

    async def get_moments(
        self,
        agent_id: str,
        *,
        type: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get notable moments for an agent."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if type:
            params["type"] = type
        data = await self._client._get(
            f"/api/v1/agents/{agent_id}/moments", params=params
        )
        return data.get("moments", [])

    async def get_positions(
        self,
        agent_id: str,
        *,
        topic: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get positions held by an agent."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if topic:
            params["topic"] = topic
        data = await self._client._get(
            f"/api/v1/agents/{agent_id}/positions", params=params
        )
        return data.get("positions", [])

    async def get_domains(self, agent_id: str) -> list[dict[str, Any]]:
        """Get domain expertise ratings for an agent."""
        data = await self._client._get(f"/api/v1/agents/{agent_id}/domains")
        return data.get("domains", [])

    async def get_elo(self, agent_id: str) -> dict[str, Any]:
        """Get ELO rating and history for an agent."""
        return await self._client._get(f"/api/v1/agents/{agent_id}/elo")

    async def get_leaderboard(self) -> list[AgentProfile]:
        """Get the agent leaderboard."""
        data = await self._client._get("/api/v1/leaderboard")
        return [AgentProfile.model_validate(a) for a in data.get("agents", [])]

    async def compare(self, agents: list[str]) -> dict[str, Any]:
        """Compare multiple agents."""
        params = {"agents": ",".join(agents)}
        return await self._client._get("/api/v1/agents/compare", params=params)

    async def get_relationship(self, agent_a: str, agent_b: str) -> dict[str, Any]:
        """Get relationship analysis between two agents."""
        return await self._client._get(
            f"/api/v1/agents/{agent_a}/relationship/{agent_b}"
        )


class VerificationAPI:
    """API for formal verification."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def verify(
        self,
        claim: str,
        *,
        backend: str = "z3",
        timeout: int = 30,
    ) -> VerificationResult:
        """Verify a claim using formal methods."""
        request = VerifyClaimRequest(claim=claim, backend=backend, timeout=timeout)
        data = await self._client._post(
            "/api/v1/verification/verify", request.model_dump()
        )
        return VerificationResult.model_validate(data)

    async def status(self) -> dict[str, Any]:
        """Get verification backend status."""
        return await self._client._get("/api/v1/verification/status")


class GauntletAPI:
    """API for gauntlet validation."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def run(
        self,
        input_content: str,
        *,
        input_type: str = "spec",
        persona: str = "security",
    ) -> dict[str, Any]:
        """Run gauntlet validation."""
        request = RunGauntletRequest(
            input_content=input_content,
            input_type=input_type,
            persona=persona,
        )
        return await self._client._post("/api/v1/gauntlet/run", request.model_dump())

    async def get_receipt(self, gauntlet_id: str) -> GauntletReceipt:
        """Get a gauntlet receipt."""
        data = await self._client._get(f"/api/v1/gauntlet/{gauntlet_id}/receipt")
        return GauntletReceipt.model_validate(data)

    async def run_and_wait(
        self,
        input_content: str,
        *,
        input_type: str = "spec",
        persona: str = "security",
        poll_interval: float = 1.0,
        timeout: float = 120.0,
    ) -> GauntletReceipt:
        """Run gauntlet and wait for completion."""
        response = await self.run(input_content, input_type=input_type, persona=persona)
        gauntlet_id = response["gauntlet_id"]

        elapsed = 0.0
        while elapsed < timeout:
            try:
                return await self.get_receipt(gauntlet_id)
            except AragoraNotFoundError:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

        raise AragoraTimeoutError(
            f"Gauntlet {gauntlet_id} did not complete within {timeout}s"
        )

    async def get(self, gauntlet_id: str) -> dict[str, Any]:
        """Get gauntlet run details."""
        return await self._client._get(f"/api/v1/gauntlet/{gauntlet_id}")

    async def delete(self, gauntlet_id: str) -> dict[str, bool]:
        """Delete a gauntlet run."""
        await self._client._delete(f"/api/v1/gauntlet/{gauntlet_id}")
        return {"deleted": True}

    async def list_receipts(
        self,
        *,
        verdict: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[GauntletReceipt]:
        """List gauntlet receipts."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if verdict:
            params["verdict"] = verdict
        data = await self._client._get("/api/v1/gauntlet/receipts", params=params)
        return [GauntletReceipt.model_validate(r) for r in data.get("receipts", [])]

    async def verify_receipt(self, receipt_id: str) -> dict[str, Any]:
        """Verify a gauntlet receipt's integrity."""
        return await self._client._get(f"/api/v1/gauntlet/receipts/{receipt_id}/verify")

    async def export_receipt(
        self,
        receipt_id: str,
        format: str = "json",
    ) -> bytes:
        """Export a gauntlet receipt."""
        return await self._client._get_raw(
            f"/api/v1/gauntlet/receipts/{receipt_id}/export",
            params={"format": format},
        )

    async def list_personas(
        self,
        *,
        category: str | None = None,
        enabled: bool | None = None,
    ) -> list[dict[str, Any]]:
        """List available gauntlet personas."""
        params: dict[str, Any] = {}
        if category:
            params["category"] = category
        if enabled is not None:
            params["enabled"] = enabled
        data = await self._client._get(
            "/api/v1/gauntlet/personas", params=params or None
        )
        return data.get("personas", [])

    async def list_results(
        self,
        *,
        gauntlet_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List gauntlet results."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if gauntlet_id:
            params["gauntlet_id"] = gauntlet_id
        if status:
            params["status"] = status
        data = await self._client._get("/api/v1/gauntlet/results", params=params)
        return data.get("results", [])

    async def get_heatmap(
        self,
        gauntlet_id: str,
        format: str = "json",
    ) -> dict[str, Any]:
        """Get gauntlet risk heatmap."""
        params: dict[str, Any] = {"format": format}
        return await self._client._get(
            f"/api/v1/gauntlet/{gauntlet_id}/heatmap", params=params
        )

    async def compare(
        self,
        gauntlet_id_1: str,
        gauntlet_id_2: str,
    ) -> dict[str, Any]:
        """Compare two gauntlet runs."""
        return await self._client._get(
            f"/api/v1/gauntlet/compare/{gauntlet_id_1}/{gauntlet_id_2}"
        )

    async def list_heatmaps(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List risk heatmaps."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        data = await self._client._get("/api/v1/gauntlet/heatmaps", params=params)
        return data.get("heatmaps", [])

    async def get_risk_heatmap(self, heatmap_id: str) -> dict[str, Any]:
        """Get a specific risk heatmap."""
        return await self._client._get(f"/api/v1/gauntlet/heatmaps/{heatmap_id}")


class MemoryAPI:
    """API for memory system."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def analytics(self, days: int = 30) -> MemoryAnalytics:
        """Get memory analytics."""
        data = await self._client._get(
            "/api/v1/memory/analytics", params={"days": days}
        )
        return MemoryAnalytics.model_validate(data)

    async def tier_stats(self, tier: str) -> MemoryTierStats:
        """Get stats for a specific memory tier."""
        data = await self._client._get(f"/api/v1/memory/tiers/{tier}")
        return MemoryTierStats.model_validate(data)

    async def snapshot(self) -> dict[str, Any]:
        """Take a manual memory snapshot."""
        return await self._client._post("/api/v1/memory/snapshot", {})


class SelectionAPI:
    """API for agent selection plugins."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def list_plugins(self) -> SelectionPlugins:
        """List available selection plugins."""
        data = await self._client._get("/api/v1/selection/plugins")
        return SelectionPlugins.model_validate(data)

    async def get_defaults(self) -> dict[str, str]:
        """Get default plugin configuration."""
        return await self._client._get("/api/v1/selection/defaults")

    async def score_agents(
        self,
        task_description: str,
        *,
        primary_domain: str | None = None,
        scorer: str | None = None,
    ) -> list[AgentScore]:
        """Score agents for a task."""
        request = ScoreAgentsRequest(
            task_description=task_description,
            primary_domain=primary_domain,
            scorer=scorer,
        )
        data = await self._client._post("/api/v1/selection/score", request.model_dump())
        return [AgentScore.model_validate(a) for a in data.get("agents", [])]

    async def select_team(
        self,
        task_description: str,
        *,
        min_agents: int = 2,
        max_agents: int = 5,
        diversity_preference: float = 0.5,
        quality_priority: float = 0.5,
        scorer: str | None = None,
        team_selector: str | None = None,
        role_assigner: str | None = None,
    ) -> TeamSelection:
        """Select an optimal team for a task."""
        request = SelectTeamRequest(
            task_description=task_description,
            min_agents=min_agents,
            max_agents=max_agents,
            diversity_preference=diversity_preference,
            quality_priority=quality_priority,
            scorer=scorer,
            team_selector=team_selector,
            role_assigner=role_assigner,
        )
        data = await self._client._post("/api/v1/selection/team", request.model_dump())
        return TeamSelection.model_validate(data)


class CodebaseAPI:
    """API for codebase analysis and security scans."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def start_scan(self, repo: str, body: dict[str, Any]) -> dict[str, Any]:
        """Start dependency scan."""
        return await self._client._post(f"/api/v1/codebase/{repo}/scan", body)

    async def latest_scan(self, repo: str) -> dict[str, Any]:
        """Get latest dependency scan."""
        return await self._client._get(f"/api/v1/codebase/{repo}/scan/latest")

    async def get_scan(self, repo: str, scan_id: str) -> dict[str, Any]:
        """Get dependency scan by ID."""
        return await self._client._get(f"/api/v1/codebase/{repo}/scan/{scan_id}")

    async def list_scans(
        self,
        repo: str,
        *,
        status: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """List dependency scans."""
        params = {
            k: v
            for k, v in {
                "status": status,
                "limit": limit,
                "offset": offset,
            }.items()
            if v is not None
        }
        return await self._client._get(
            f"/api/v1/codebase/{repo}/scans", params=params or None
        )

    async def list_vulnerabilities(
        self,
        repo: str,
        *,
        severity: str | None = None,
        package: str | None = None,
        ecosystem: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """List vulnerabilities from latest scan."""
        params = {
            k: v
            for k, v in {
                "severity": severity,
                "package": package,
                "ecosystem": ecosystem,
                "limit": limit,
                "offset": offset,
            }.items()
            if v is not None
        }
        return await self._client._get(
            f"/api/v1/codebase/{repo}/vulnerabilities", params=params or None
        )

    async def package_vulnerabilities(
        self,
        ecosystem: str,
        package_name: str,
        *,
        version: str | None = None,
    ) -> dict[str, Any]:
        """Query package vulnerabilities."""
        params = {"version": version} if version else None
        return await self._client._get(
            f"/api/v1/codebase/package/{ecosystem}/{package_name}/vulnerabilities",
            params=params,
        )

    async def get_cve(self, cve_id: str) -> dict[str, Any]:
        """Get CVE details."""
        return await self._client._get(f"/api/v1/cve/{cve_id}")

    async def analyze_dependencies(self, body: dict[str, Any]) -> dict[str, Any]:
        """Analyze dependency graph."""
        return await self._client._post("/api/v1/codebase/analyze-dependencies", body)

    async def scan_vulnerabilities(self, body: dict[str, Any]) -> dict[str, Any]:
        """Scan dependencies for CVEs."""
        return await self._client._post("/api/v1/codebase/scan-vulnerabilities", body)

    async def check_licenses(self, body: dict[str, Any]) -> dict[str, Any]:
        """Check license compatibility."""
        return await self._client._post("/api/v1/codebase/check-licenses", body)

    async def generate_sbom(self, body: dict[str, Any]) -> dict[str, Any]:
        """Generate SBOM for a repository."""
        return await self._client._post("/api/v1/codebase/sbom", body)

    async def clear_cache(self) -> dict[str, Any]:
        """Clear dependency cache."""
        return await self._client._post("/api/v1/codebase/clear-cache", {})

    async def start_secrets_scan(
        self, repo: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """Trigger secrets scan."""
        return await self._client._post(f"/api/v1/codebase/{repo}/scan/secrets", body)

    async def latest_secrets_scan(self, repo: str) -> dict[str, Any]:
        """Get latest secrets scan."""
        return await self._client._get(f"/api/v1/codebase/{repo}/scan/secrets/latest")

    async def get_secrets_scan(self, repo: str, scan_id: str) -> dict[str, Any]:
        """Get secrets scan by ID."""
        return await self._client._get(
            f"/api/v1/codebase/{repo}/scan/secrets/{scan_id}"
        )

    async def list_secrets(self, repo: str) -> dict[str, Any]:
        """List secrets from latest scan."""
        return await self._client._get(f"/api/v1/codebase/{repo}/secrets")

    async def list_secrets_scans(self, repo: str) -> dict[str, Any]:
        """List secrets scan history."""
        return await self._client._get(f"/api/v1/codebase/{repo}/scans/secrets")

    async def start_sast_scan(self, repo: str, body: dict[str, Any]) -> dict[str, Any]:
        """Trigger SAST scan."""
        return await self._client._post(f"/api/v1/codebase/{repo}/scan/sast", body)

    async def get_sast_scan(self, repo: str, scan_id: str) -> dict[str, Any]:
        """Get SAST scan by ID."""
        return await self._client._get(f"/api/v1/codebase/{repo}/scan/sast/{scan_id}")

    async def list_sast_findings(
        self,
        repo: str,
        *,
        severity: str | None = None,
        owasp_category: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """List SAST findings."""
        params = {
            k: v
            for k, v in {
                "severity": severity,
                "owasp_category": owasp_category,
                "limit": limit,
                "offset": offset,
            }.items()
            if v is not None
        }
        return await self._client._get(
            f"/api/v1/codebase/{repo}/sast/findings", params=params or None
        )

    async def get_sast_owasp_summary(self, repo: str) -> dict[str, Any]:
        """Summarize SAST findings by OWASP category."""
        return await self._client._get(f"/api/v1/codebase/{repo}/sast/owasp-summary")

    async def run_metrics_analysis(
        self, repo: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """Run codebase metrics analysis."""
        return await self._client._post(
            f"/api/v1/codebase/{repo}/metrics/analyze", body
        )

    async def latest_metrics(self, repo: str) -> dict[str, Any]:
        """Get latest metrics report."""
        return await self._client._get(f"/api/v1/codebase/{repo}/metrics")

    async def get_metrics(self, repo: str, analysis_id: str) -> dict[str, Any]:
        """Get metrics report by ID."""
        return await self._client._get(f"/api/v1/codebase/{repo}/metrics/{analysis_id}")

    async def list_metrics_history(
        self,
        repo: str,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """List metrics history."""
        params = {
            k: v
            for k, v in {
                "limit": limit,
                "offset": offset,
            }.items()
            if v is not None
        }
        return await self._client._get(
            f"/api/v1/codebase/{repo}/metrics/history", params=params or None
        )

    async def get_hotspots(
        self,
        repo: str,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """Get hotspot analysis."""
        params = {
            k: v
            for k, v in {
                "limit": limit,
                "offset": offset,
            }.items()
            if v is not None
        }
        return await self._client._get(
            f"/api/v1/codebase/{repo}/hotspots", params=params or None
        )

    async def get_duplicates(
        self,
        repo: str,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """Get duplicate block analysis."""
        params = {
            k: v
            for k, v in {
                "limit": limit,
                "offset": offset,
            }.items()
            if v is not None
        }
        return await self._client._get(
            f"/api/v1/codebase/{repo}/duplicates", params=params or None
        )

    async def get_file_metrics(self, repo: str, file_path: str) -> dict[str, Any]:
        """Get metrics for a specific file."""
        encoded = quote(file_path, safe="")
        return await self._client._get(
            f"/api/v1/codebase/{repo}/metrics/file/{encoded}"
        )

    async def analyze_codebase(self, repo: str, body: dict[str, Any]) -> dict[str, Any]:
        """Run code intelligence analysis."""
        return await self._client._post(f"/api/v1/codebase/{repo}/analyze", body)

    async def get_symbols(
        self, repo: str, *, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Get codebase symbols."""
        return await self._client._get(
            f"/api/v1/codebase/{repo}/symbols", params=params
        )

    async def get_callgraph(
        self, repo: str, *, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Get codebase call graph."""
        return await self._client._get(
            f"/api/v1/codebase/{repo}/callgraph", params=params
        )

    async def get_deadcode(
        self, repo: str, *, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Get dead code report."""
        return await self._client._get(
            f"/api/v1/codebase/{repo}/deadcode", params=params
        )

    async def analyze_impact(self, repo: str, body: dict[str, Any]) -> dict[str, Any]:
        """Analyze change impact."""
        return await self._client._post(f"/api/v1/codebase/{repo}/impact", body)

    async def understand(self, repo: str, body: dict[str, Any]) -> dict[str, Any]:
        """Explain codebase components."""
        return await self._client._post(f"/api/v1/codebase/{repo}/understand", body)

    async def start_audit(self, repo: str, body: dict[str, Any]) -> dict[str, Any]:
        """Start a codebase audit."""
        return await self._client._post(f"/api/v1/codebase/{repo}/audit", body)

    async def get_audit(self, repo: str, audit_id: str) -> dict[str, Any]:
        """Get codebase audit results."""
        return await self._client._get(f"/api/v1/codebase/{repo}/audit/{audit_id}")

    async def start_quick_scan(self, body: dict[str, Any]) -> dict[str, Any]:
        """Run a quick security scan."""
        return await self._client._post("/api/v1/codebase/quick-scan", body)

    async def get_quick_scan(self, scan_id: str) -> dict[str, Any]:
        """Get quick scan result."""
        return await self._client._get(f"/api/v1/codebase/quick-scan/{scan_id}")

    async def list_quick_scans(self) -> dict[str, Any]:
        """List quick scans."""
        return await self._client._get("/api/v1/codebase/quick-scans")


class GmailAPI:
    """API for Gmail operations."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def list_labels(self, *, user_id: str | None = None) -> dict[str, Any]:
        """List Gmail labels."""
        params = {"user_id": user_id} if user_id else None
        return await self._client._get("/api/v1/gmail/labels", params=params)

    async def create_label(self, body: dict[str, Any]) -> dict[str, Any]:
        """Create a Gmail label."""
        return await self._client._post("/api/v1/gmail/labels", body)

    async def update_label(self, label_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Update a Gmail label."""
        return await self._client._patch(f"/api/v1/gmail/labels/{label_id}", body)

    async def delete_label(
        self, label_id: str, *, user_id: str | None = None
    ) -> dict[str, Any]:
        """Delete a Gmail label."""
        params = {"user_id": user_id} if user_id else None
        response = await self._client._request(
            "DELETE", f"/api/v1/gmail/labels/{label_id}", params=params
        )
        return response.json()

    async def list_filters(self, *, user_id: str | None = None) -> dict[str, Any]:
        """List Gmail filters."""
        params = {"user_id": user_id} if user_id else None
        return await self._client._get("/api/v1/gmail/filters", params=params)

    async def create_filter(self, body: dict[str, Any]) -> dict[str, Any]:
        """Create a Gmail filter."""
        return await self._client._post("/api/v1/gmail/filters", body)

    async def delete_filter(
        self, filter_id: str, *, user_id: str | None = None
    ) -> dict[str, Any]:
        """Delete a Gmail filter."""
        params = {"user_id": user_id} if user_id else None
        response = await self._client._request(
            "DELETE", f"/api/v1/gmail/filters/{filter_id}", params=params
        )
        return response.json()

    async def modify_message_labels(
        self, message_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """Modify labels for a message."""
        return await self._client._post(
            f"/api/v1/gmail/messages/{message_id}/labels", body
        )

    async def mark_message_read(
        self, message_id: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Mark a message read/unread."""
        return await self._client._post(
            f"/api/v1/gmail/messages/{message_id}/read", body or {}
        )

    async def mark_message_star(
        self, message_id: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Star or unstar a message."""
        return await self._client._post(
            f"/api/v1/gmail/messages/{message_id}/star", body or {}
        )

    async def archive_message(self, message_id: str) -> dict[str, Any]:
        """Archive a message."""
        return await self._client._post(
            f"/api/v1/gmail/messages/{message_id}/archive", {}
        )

    async def trash_message(
        self, message_id: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Trash or untrash a message."""
        return await self._client._post(
            f"/api/v1/gmail/messages/{message_id}/trash", body or {}
        )

    async def get_attachment(
        self,
        message_id: str,
        attachment_id: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Get a message attachment."""
        params = {"user_id": user_id} if user_id else None
        return await self._client._get(
            f"/api/v1/gmail/messages/{message_id}/attachments/{attachment_id}",
            params=params,
        )

    async def list_threads(
        self,
        *,
        user_id: str | None = None,
        q: str | None = None,
        label_ids: str | None = None,
        limit: int | None = None,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        """List Gmail threads."""
        params = {
            k: v
            for k, v in {
                "user_id": user_id,
                "q": q,
                "label_ids": label_ids,
                "limit": limit,
                "page_token": page_token,
            }.items()
            if v is not None
        }
        return await self._client._get("/api/v1/gmail/threads", params=params or None)

    async def get_thread(
        self, thread_id: str, *, user_id: str | None = None
    ) -> dict[str, Any]:
        """Get a Gmail thread."""
        params = {"user_id": user_id} if user_id else None
        return await self._client._get(
            f"/api/v1/gmail/threads/{thread_id}", params=params
        )

    async def archive_thread(self, thread_id: str) -> dict[str, Any]:
        """Archive a Gmail thread."""
        return await self._client._post(
            f"/api/v1/gmail/threads/{thread_id}/archive", {}
        )

    async def trash_thread(
        self, thread_id: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Trash or untrash a Gmail thread."""
        return await self._client._post(
            f"/api/v1/gmail/threads/{thread_id}/trash", body or {}
        )

    async def modify_thread_labels(
        self, thread_id: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """Modify labels for a thread."""
        return await self._client._post(
            f"/api/v1/gmail/threads/{thread_id}/labels", body
        )

    async def list_drafts(
        self,
        *,
        user_id: str | None = None,
        limit: int | None = None,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        """List Gmail drafts."""
        params = {
            k: v
            for k, v in {
                "user_id": user_id,
                "limit": limit,
                "page_token": page_token,
            }.items()
            if v is not None
        }
        return await self._client._get("/api/v1/gmail/drafts", params=params or None)

    async def create_draft(self, body: dict[str, Any]) -> dict[str, Any]:
        """Create a Gmail draft."""
        return await self._client._post("/api/v1/gmail/drafts", body)

    async def get_draft(
        self, draft_id: str, *, user_id: str | None = None
    ) -> dict[str, Any]:
        """Get a Gmail draft."""
        params = {"user_id": user_id} if user_id else None
        return await self._client._get(
            f"/api/v1/gmail/drafts/{draft_id}", params=params
        )

    async def update_draft(self, draft_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Update a Gmail draft."""
        return await self._client._put(f"/api/v1/gmail/drafts/{draft_id}", body)

    async def delete_draft(
        self, draft_id: str, *, user_id: str | None = None
    ) -> dict[str, Any]:
        """Delete a Gmail draft."""
        params = {"user_id": user_id} if user_id else None
        response = await self._client._request(
            "DELETE", f"/api/v1/gmail/drafts/{draft_id}", params=params
        )
        return response.json()

    async def send_draft(self, draft_id: str) -> dict[str, Any]:
        """Send a Gmail draft."""
        return await self._client._post(f"/api/v1/gmail/drafts/{draft_id}/send", {})


class ReplaysAPI:
    """API for replay management."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def list(self, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """List replays."""
        data = await self._client._get(
            "/api/v1/replays", params={"limit": limit, "offset": offset}
        )
        return data.get("replays", [])

    async def get(self, replay_id: str) -> dict[str, Any]:
        """Get a replay by ID."""
        return await self._client._get(f"/api/v1/replays/{replay_id}")

    async def export(self, replay_id: str, format: str = "json") -> bytes:
        """Export a replay."""
        return await self._client._get_raw(
            f"/api/v1/replays/{replay_id}/export", params={"format": format}
        )

    async def delete(self, replay_id: str) -> None:
        """Delete a replay."""
        await self._client._delete(f"/api/v1/replays/{replay_id}")


class AragoraClient:
    """
    Client for the Aragora API.

    Example:
        >>> client = AragoraClient("http://localhost:8080")
        >>> debate = await client.debates.run(task="Should we use microservices?")
        >>> print(debate.consensus.conclusion)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize the Aragora client.

        Args:
            base_url: Base URL of the Aragora server.
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds.
            headers: Optional additional headers.
        """
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

        default_headers = {"User-Agent": "aragora-client-python/2.0.0"}
        if api_key:
            default_headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            default_headers.update(headers)

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=default_headers,
            timeout=timeout,
        )

        # Initialize API namespaces
        self.debates = DebatesAPI(self)
        self.graph_debates = GraphDebatesAPI(self)
        self.matrix_debates = MatrixDebatesAPI(self)
        self.agents = AgentsAPI(self)
        self.verification = VerificationAPI(self)
        self.gauntlet = GauntletAPI(self)
        self.memory = MemoryAPI(self)
        self.selection = SelectionAPI(self)
        self.replays = ReplaysAPI(self)
        self.control_plane = ControlPlaneAPI(self)
        self.codebase = CodebaseAPI(self)
        self.gmail = GmailAPI(self)
        # Enterprise APIs
        self.auth = AuthAPI(self)
        self.tenants = TenancyAPI(self)
        self.rbac = RBACAPI(self)
        self.tournaments = TournamentsAPI(self)
        self.audit = AuditAPI(self)
        self.onboarding = OnboardingAPI(self)
        self.knowledge = KnowledgeAPI(self)
        self.workflows = WorkflowsAPI(self)
        self.explainability = ExplainabilityAPI(self)
        self.marketplace = MarketplaceAPI(self)
        self.cross_pollination = CrossPollinationAPI(self)
        self.threat_intel = ThreatIntelAPI(self)
        self.decisions = DecisionsAPI(self)
        self.replays = ReplayAPI(self)

    async def __aenter__(self) -> AragoraClient:
        """Enter async context."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context."""
        await self.close()

    async def close(self) -> None:
        """Close the client."""
        await self._client.aclose()

    async def health(self) -> HealthStatus:
        """Get server health status."""
        data = await self._get("/api/v1/health")
        return HealthStatus.model_validate(data)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Make an HTTP request."""
        try:
            response = await self._client.request(
                method, path, params=params, json=json
            )
        except httpx.ConnectError as e:
            raise AragoraConnectionError(str(e)) from e
        except httpx.TimeoutException as e:
            raise AragoraTimeoutError(str(e)) from e

        if response.status_code == 401:
            raise AragoraAuthenticationError()
        if response.status_code == 404:
            raise AragoraNotFoundError("Resource", path)
        if response.status_code == 400:
            data = response.json() if response.content else {}
            raise AragoraValidationError(
                data.get("error", "Validation error"),
                details=data.get("details"),
            )
        if response.status_code >= 400:
            data = response.json() if response.content else {}
            raise AragoraError(
                data.get("error", f"Request failed with status {response.status_code}"),
                code=data.get("code"),
                status=response.status_code,
                details=data.get("details"),
            )

        return response

    async def _get(
        self, path: str, *, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make a GET request."""
        response = await self._request("GET", path, params=params)
        return response.json()

    async def _get_raw(
        self, path: str, *, params: dict[str, Any] | None = None
    ) -> bytes:
        """Make a GET request and return raw bytes."""
        response = await self._request("GET", path, params=params)
        return response.content

    async def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make a POST request."""
        response = await self._request("POST", path, json=data)
        return response.json()

    async def _patch(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make a PATCH request."""
        response = await self._request("PATCH", path, json=data)
        return response.json()

    async def _put(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make a PUT request."""
        response = await self._request("PUT", path, json=data)
        return response.json()

    async def _delete(self, path: str) -> None:
        """Make a DELETE request."""
        await self._request("DELETE", path)

    async def _delete_with_body(self, path: str, data: dict[str, Any]) -> None:
        """Make a DELETE request with a body."""
        await self._request("DELETE", path, json=data)
