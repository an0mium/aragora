"""
Debates Namespace API

Provides methods for creating, managing, and analyzing debates.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient
    from ..pagination import AsyncPaginator, SyncPaginator
    from ..websocket import WebSocketEvent


_List = list  # Preserve builtin list for type annotations


class DebatesAPI:
    """
    Synchronous Debates API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> debate = client.debates.create(task="Should we use microservices?")
        >>> messages = client.debates.get_messages(debate["debate_id"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def create(
        self,
        task: str,
        agents: _List[str] | None = None,
        protocol: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Create a new debate.

        Args:
            task: The topic or question to debate
            agents: List of agent names to participate (optional)
            protocol: Debate protocol configuration (optional)
            **kwargs: Additional debate options

        Returns:
            Created debate with debate_id
        """
        data = {"task": task, **kwargs}
        if agents:
            data["agents"] = agents
        if protocol:
            data["protocol"] = protocol

        return self._client.request("POST", "/api/v1/debates", json=data)

    def get(self, debate_id: str) -> dict[str, Any]:
        """
        Get a debate by ID.

        Args:
            debate_id: The debate ID

        Returns:
            Debate details
        """
        return self._client.request("GET", f"/api/v1/debates/{debate_id}")

    def list(
        self,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
    ) -> dict[str, Any]:
        """
        List debates with pagination.

        Args:
            limit: Maximum number of debates to return
            offset: Number of debates to skip
            status: Filter by status (active, completed, etc.)

        Returns:
            List of debates with pagination info
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        return self._client.request("GET", "/api/v1/debates", params=params)

    def list_all(
        self,
        status: str | None = None,
        page_size: int = 20,
    ) -> SyncPaginator:
        """
        Iterate through all debates with automatic pagination.

        Args:
            status: Filter by status (active, completed, etc.)
            page_size: Number of debates per page (default 20)

        Returns:
            SyncPaginator yielding debate dictionaries

        Example::

            for debate in client.debates.list_all(status="active"):
                print(debate["id"])
        """
        from ..pagination import SyncPaginator

        params: dict[str, Any] = {}
        if status:
            params["status"] = status

        return SyncPaginator(self._client, "/api/v1/debates", params, page_size)

    def get_messages(self, debate_id: str) -> dict[str, Any]:
        """
        Get messages from a debate.

        Args:
            debate_id: The debate ID

        Returns:
            List of messages
        """
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/messages")

    def add_message(
        self,
        debate_id: str,
        content: str,
        role: str = "user",
    ) -> dict[str, Any]:
        """
        Add a message to a debate.

        Args:
            debate_id: The debate ID
            content: Message content
            role: Message role (user, system, etc.)

        Returns:
            Created message
        """
        return self._client.request(
            "POST",
            f"/api/v1/debates/{debate_id}/messages",
            json={"content": content, "role": role},
        )

    def get_consensus(self, debate_id: str) -> dict[str, Any]:
        """
        Get consensus information for a debate.

        Args:
            debate_id: The debate ID

        Returns:
            Consensus details
        """
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/consensus")

    def get_export(
        self,
        debate_id: str,
        format: str = "json",
    ) -> dict[str, Any]:
        """
        Export a debate.

        Args:
            debate_id: The debate ID
            format: Export format (json, pdf, etc.)

        Returns:
            Exported debate data
        """
        return self._client.request(
            "GET",
            f"/api/v1/debates/{debate_id}/export",
            params={"format": format},
        )

    def cancel(self, debate_id: str) -> dict[str, Any]:
        """
        Cancel a debate.

        Args:
            debate_id: The debate ID

        Returns:
            Cancellation result
        """
        return self._client.request("POST", f"/api/v1/debates/{debate_id}/cancel")

    def get_by_slug(self, slug: str) -> dict[str, Any]:
        """Get a debate by slug."""
        return self._client.request("GET", f"/api/v1/debates/slug/{slug}")

    def get_explainability(self, debate_id: str) -> dict[str, Any]:
        """Get explainability data for a debate."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/explainability")

    def get_explainability_factors(self, debate_id: str) -> dict[str, Any]:
        """Get factor decomposition for a debate decision."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/explainability/factors")

    def get_explainability_narrative(self, debate_id: str) -> dict[str, Any]:
        """Get natural language narrative explanation."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/explainability/narrative")

    def get_explainability_provenance(self, debate_id: str) -> dict[str, Any]:
        """Get provenance chain for debate claims."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/explainability/provenance")

    def get_explainability_counterfactual(self, debate_id: str) -> dict[str, Any]:
        """Get counterfactual analysis."""
        return self._client.request(
            "GET", f"/api/v1/debates/{debate_id}/explainability/counterfactual"
        )

    def create_counterfactual(self, debate_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        """Create a counterfactual scenario."""
        return self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/explainability/counterfactual", json=changes
        )

    def get_citations(self, debate_id: str) -> dict[str, Any]:
        """Get citations used in debate."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/citations")

    def get_convergence(self, debate_id: str) -> dict[str, Any]:
        """Get convergence analysis."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/convergence")

    def get_evidence(self, debate_id: str) -> dict[str, Any]:
        """Get evidence collected during debate."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/evidence")

    def get_graph_stats(self, debate_id: str) -> dict[str, Any]:
        """Get graph statistics for debate."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/graph/stats")

    def get_impasse(self, debate_id: str) -> dict[str, Any]:
        """Get impasse analysis if debate is stuck."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/impasse")

    def get_meta_critique(self, debate_id: str) -> dict[str, Any]:
        """Get meta-critique of debate process."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/meta-critique")

    def get_red_team(self, debate_id: str) -> dict[str, Any]:
        """Get red team analysis."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/red-team")

    def capability_probe(self, task: str, agents: _List[str] | None = None) -> dict[str, Any]:
        """Run a capability probe debate."""
        data: dict[str, Any] = {"task": task}
        if agents:
            data["agents"] = agents
        return self._client.request("POST", "/api/v1/debates/capability-probe", json=data)

    def deep_audit(self, task: str, agents: _List[str] | None = None) -> dict[str, Any]:
        """Run a deep audit debate."""
        data: dict[str, Any] = {"task": task}
        if agents:
            data["agents"] = agents
        return self._client.request("POST", "/api/v1/debates/deep-audit", json=data)

    def broadcast(self, debate_id: str, channels: _List[str] | None = None) -> dict[str, Any]:
        """Broadcast debate to channels."""
        data: dict[str, Any] = {"channels": channels} if channels else {}
        return self._client.request("POST", f"/api/v1/debates/{debate_id}/broadcast", json=data)

    def fork(self, debate_id: str, changes: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fork a debate with optional changes."""
        return self._client.request("POST", f"/api/v1/debates/{debate_id}/fork", json=changes or {})

    def publish_twitter(self, debate_id: str, message: str | None = None) -> dict[str, Any]:
        """Publish debate summary to Twitter."""
        data: dict[str, Any] = {"message": message} if message else {}
        return self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/publish/twitter", json=data
        )

    def publish_youtube(self, debate_id: str, title: str | None = None) -> dict[str, Any]:
        """Publish debate to YouTube."""
        data: dict[str, Any] = {"title": title} if title else {}
        return self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/publish/youtube", json=data
        )

    def get_dashboard(self) -> dict[str, Any]:
        """Get debates dashboard view."""
        return self._client.request("GET", "/api/v1/dashboard/debates")

    def get_debate_graph_stats(self, debate_id: str) -> dict[str, Any]:
        """Get graph stats via debate endpoint."""
        return self._client.request("GET", f"/api/v1/debate/{debate_id}/graph-stats")

    def get_history(self, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        """Get debate history."""
        return self._client.request(
            "GET", "/api/v1/history/debates", params={"limit": limit, "offset": offset}
        )

    # ========== Additional CRUD ==========

    def update(self, debate_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        """Update an existing debate."""
        return self._client.request("PUT", f"/api/v1/debates/{debate_id}", json=updates)

    def delete(self, debate_id: str) -> dict[str, Any]:
        """Delete a debate."""
        return self._client.request("DELETE", f"/api/v1/debates/{debate_id}")

    def clone(
        self,
        debate_id: str,
        preserve_agents: bool = True,
        preserve_context: bool = False,
    ) -> dict[str, Any]:
        """Clone a debate (create a copy with fresh state)."""
        return self._client.request(
            "POST",
            f"/api/v1/debates/{debate_id}/clone",
            json={"preserveAgents": preserve_agents, "preserveContext": preserve_context},
        )

    def archive(self, debate_id: str) -> dict[str, Any]:
        """Archive a debate."""
        return self._client.request("POST", f"/api/v1/debates/{debate_id}/archive")

    # ========== Lifecycle ==========

    def start(self, debate_id: str) -> dict[str, Any]:
        """Start a debate."""
        return self._client.request("POST", f"/api/v1/debates/{debate_id}/start")

    def stop(self, debate_id: str) -> dict[str, Any]:
        """Stop a running debate."""
        return self._client.request("POST", f"/api/v1/debates/{debate_id}/stop")

    def pause(self, debate_id: str) -> dict[str, Any]:
        """Pause a running debate."""
        return self._client.request("POST", f"/api/v1/debates/{debate_id}/pause")

    def resume(self, debate_id: str) -> dict[str, Any]:
        """Resume a paused debate."""
        return self._client.request("POST", f"/api/v1/debates/{debate_id}/resume")

    # ========== Analysis ==========

    def get_rhetorical(self, debate_id: str) -> dict[str, Any]:
        """Get rhetorical pattern observations for a debate."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/rhetorical")

    def get_trickster(self, debate_id: str) -> dict[str, Any]:
        """Get trickster hollow consensus detection status."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/trickster")

    def get_summary(self, debate_id: str) -> dict[str, Any]:
        """Get a human-readable summary of the debate."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/summary")

    def get_verification_report(self, debate_id: str) -> dict[str, Any]:
        """Get the verification report for debate conclusions."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/verification-report")

    def verify_claim(
        self,
        debate_id: str,
        claim_id: str,
        evidence: str | None = None,
    ) -> dict[str, Any]:
        """Verify a specific claim from the debate."""
        payload: dict[str, Any] = {"claim_id": claim_id}
        if evidence:
            payload["evidence"] = evidence
        return self._client.request("POST", f"/api/v1/debates/{debate_id}/verify", json=payload)

    # ========== Follow-up & Continuation ==========

    def get_followup_suggestions(self, debate_id: str) -> dict[str, Any]:
        """Get suggestions for follow-up debates."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/followups")

    def follow_up(
        self,
        debate_id: str,
        crux_id: str | None = None,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Create a follow-up debate from an existing one."""
        payload: dict[str, Any] = {}
        if crux_id:
            payload["cruxId"] = crux_id
        if context:
            payload["context"] = context
        return self._client.request("POST", f"/api/v1/debates/{debate_id}/followup", json=payload)

    def list_forks(self, debate_id: str) -> dict[str, Any]:
        """List all forks created from a debate."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/forks")

    # ========== Rounds, Agents, Votes ==========

    def get_rounds(self, debate_id: str) -> dict[str, Any]:
        """Get rounds from a debate."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/rounds")

    def get_agents(self, debate_id: str) -> dict[str, Any]:
        """Get agents participating in a debate."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/agents")

    def get_votes(self, debate_id: str) -> dict[str, Any]:
        """Get votes from a debate."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/votes")

    def add_user_input(
        self,
        debate_id: str,
        input_text: str,
        input_type: str = "suggestion",  # 'suggestion' | 'vote' | 'question' | 'context'
    ) -> dict[str, Any]:
        """Add user input to a debate."""
        return self._client.request(
            "POST",
            f"/api/v1/debates/{debate_id}/user-input",
            json={"input": input_text, "type": input_type},
        )

    def get_timeline(self, debate_id: str) -> dict[str, Any]:
        """Get the timeline of events in a debate."""
        return self._client.request("GET", f"/api/v1/debates/{debate_id}/timeline")

    def add_evidence(
        self,
        debate_id: str,
        evidence: str,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add evidence to a debate."""
        payload: dict[str, Any] = {"evidence": evidence}
        if source:
            payload["source"] = source
        if metadata:
            payload["metadata"] = metadata
        return self._client.request("POST", f"/api/v1/debates/{debate_id}/evidence", json=payload)

    # ========== Search ==========

    def search(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
        domain: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        """Search across all debates."""
        params: dict[str, Any] = {"query": query, "limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if domain:
            params["domain"] = domain
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        return self._client.request("GET", "/api/v1/search", params=params)

    # ========== Batch Operations ==========

    def submit_batch(self, requests: _List[dict[str, Any]]) -> dict[str, Any]:
        """Submit multiple debates for batch processing."""
        return self._client.request("POST", "/api/v1/debates/batch", json={"requests": requests})

    def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        """Get the status of a batch job."""
        return self._client.request("GET", f"/api/v1/debates/batch/{batch_id}/status")

    def list_batches(
        self,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
    ) -> dict[str, Any]:
        """List all batch jobs."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        return self._client.request("GET", "/api/v1/debates/batch", params=params)

    def get_queue_status(self) -> dict[str, Any]:
        """Get the current queue status."""
        return self._client.request("GET", "/api/v1/debates/queue/status")

    # ========== Graph & Visualization ==========

    def get_graph(self, debate_id: str) -> dict[str, Any]:
        """Get the argument graph for a debate."""
        return self._client.request("GET", f"/api/v1/debates/graph/{debate_id}")

    def get_graph_branches(self, debate_id: str) -> dict[str, Any]:
        """Get branches in the argument graph."""
        return self._client.request("GET", f"/api/v1/debates/graph/{debate_id}/branches")

    def get_matrix_comparison(self, debate_id: str) -> dict[str, Any]:
        """Get matrix comparison for a multi-scenario debate."""
        return self._client.request("GET", f"/api/v1/debates/matrix/{debate_id}")

    # ========== Streaming ==========

    def stream_debates(self) -> dict[str, Any]:
        """Stream debate events via Server-Sent Events.

        Returns an SSE stream of real-time debate events including
        debate_started, round_start, agent_message, consensus, and debate_end.

        Returns:
            SSE stream response with debate events
        """
        return self._client.request("GET", "/api/v1/debates/stream")

    # ========== RLM / Compression ==========

    def compress(
        self,
        debate_id: str,
        target_levels: _List[str] | None = None,
        compression_ratio: float = 0.3,
    ) -> dict[str, Any]:
        """Compress debate context using RLM hierarchical abstraction.

        Args:
            debate_id: The debate ID
            target_levels: Abstraction levels to generate (e.g. ["ABSTRACT", "SUMMARY", "DETAILED"])
            compression_ratio: Target compression ratio (0.0-1.0, default 0.3)

        Returns:
            Compression result with original_tokens, compressed_tokens, compression_ratios,
            time_seconds, and levels_created
        """
        payload: dict[str, Any] = {"compression_ratio": compression_ratio}
        if target_levels:
            payload["target_levels"] = target_levels
        return self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/compress", json=payload
        )

    def get_context_level(self, debate_id: str, level: str) -> dict[str, Any]:
        """Get debate content at a specific abstraction level.

        Args:
            debate_id: The debate ID
            level: Abstraction level (ABSTRACT, SUMMARY, DETAILED, or RAW)

        Returns:
            Context at the requested level with content, token_count, and nodes
        """
        return self._client.request(
            "GET", f"/api/v1/debates/{debate_id}/context/{level}"
        )

    def query_rlm(
        self,
        debate_id: str,
        query: str,
        strategy: str = "auto",
        max_iterations: int = 3,
        start_level: str = "SUMMARY",
    ) -> dict[str, Any]:
        """Query a debate using RLM with iterative refinement.

        Args:
            debate_id: The debate ID
            query: The question to ask about the debate
            strategy: Query strategy (auto, peek, grep, partition_map, summarize, hierarchical)
            max_iterations: Maximum refinement iterations (1-10, default 3)
            start_level: Starting abstraction level (ABSTRACT, SUMMARY, DETAILED, RAW)

        Returns:
            Query result with answer, confidence, refinement_history, and token usage
        """
        return self._client.request(
            "POST",
            f"/api/v1/debates/{debate_id}/query-rlm",
            json={
                "query": query,
                "strategy": strategy,
                "max_iterations": max_iterations,
                "start_level": start_level,
            },
        )

    def get_refinement_status(self, debate_id: str) -> dict[str, Any]:
        """Get the status of an ongoing RLM refinement process.

        Args:
            debate_id: The debate ID

        Returns:
            Refinement status with active_queries, cached_contexts, and status
        """
        return self._client.request(
            "GET", f"/api/v1/debates/{debate_id}/refinement-status"
        )

    # ========== Decision Integrity ==========

    def get_decision_integrity(self, debate_id: str) -> dict[str, Any]:
        """Get the decision integrity package for a debate.

        Generates a decision receipt and implementation plan bundle
        containing audit-ready documentation of the debate outcome.

        Args:
            debate_id: The debate ID

        Returns:
            Decision integrity package with receipt and implementation plan
        """
        return self._client.request(
            "GET", f"/api/v1/debates/{debate_id}/decision-integrity"
        )

    # ========== Intervention ==========

    def intervention_inject(
        self,
        debate_id: str,
        content: str,
        injection_type: str = "argument",
        source: str = "user",
    ) -> dict[str, Any]:
        """Inject a user argument or follow-up question into a debate.

        The injected content will be included in the next round's context
        and considered by all agents.

        Args:
            debate_id: The active debate ID
            content: The argument or question to inject
            injection_type: Type of injection ("argument" or "follow_up")
            source: Source identifier (default "user")

        Returns:
            Injection result with injection_id and confirmation
        """
        return self._client.request(
            "POST",
            f"/api/v1/debates/{debate_id}/intervention/inject",
            json={"content": content, "type": injection_type, "source": source},
        )

    def intervention_log(self, debate_id: str, limit: int = 50) -> dict[str, Any]:
        """Get the intervention audit log for a debate.

        Returns all interventions with timestamps for compliance and audit purposes.

        Args:
            debate_id: The debate ID
            limit: Maximum number of log entries to return (default 50)

        Returns:
            Intervention log with total_interventions and interventions list
        """
        return self._client.request(
            "GET",
            f"/api/v1/debates/{debate_id}/intervention/log",
            params={"limit": limit},
        )

    def intervention_pause(self, debate_id: str) -> dict[str, Any]:
        """Pause an active debate.

        Pausing stops agent responses but preserves all debate state.
        The debate can be resumed at any point.

        Args:
            debate_id: The active debate ID

        Returns:
            Pause result with is_paused status and paused_at timestamp
        """
        return self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/intervention/pause"
        )

    def intervention_resume(self, debate_id: str) -> dict[str, Any]:
        """Resume a paused debate.

        Resumes agent responses from where they left off.

        Args:
            debate_id: The paused debate ID

        Returns:
            Resume result with resumed_at timestamp and pause_duration_seconds
        """
        return self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/intervention/resume"
        )

    def intervention_state(self, debate_id: str) -> dict[str, Any]:
        """Get the current intervention state for a debate.

        Returns pause status, agent weights, consensus threshold,
        and counts of pending injections and follow-ups.

        Args:
            debate_id: The debate ID

        Returns:
            Intervention state with is_paused, consensus_threshold, agent_weights,
            pending_injections, and pending_follow_ups
        """
        return self._client.request(
            "GET", f"/api/v1/debates/{debate_id}/intervention/state"
        )

    def intervention_update_threshold(
        self, debate_id: str, threshold: float
    ) -> dict[str, Any]:
        """Update the consensus threshold for a debate.

        Threshold is the minimum agreement level required for consensus:
        0.5 = simple majority, 0.75 = strong majority (default), 1.0 = unanimous.

        Args:
            debate_id: The active debate ID
            threshold: New threshold value (0.5 to 1.0)

        Returns:
            Update result with old_threshold and new_threshold
        """
        return self._client.request(
            "PUT",
            f"/api/v1/debates/{debate_id}/intervention/threshold",
            json={"threshold": threshold},
        )

    def intervention_update_weights(
        self, debate_id: str, agent: str, weight: float
    ) -> dict[str, Any]:
        """Update an agent's influence weight in a debate.

        Weight affects how much the agent's vote counts in consensus:
        0.0 = muted, 1.0 = normal influence, 2.0 = double influence.

        Args:
            debate_id: The active debate ID
            agent: Agent name or ID
            weight: New weight value (0.0 to 2.0)

        Returns:
            Update result with agent, old_weight, and new_weight
        """
        return self._client.request(
            "PUT",
            f"/api/v1/debates/{debate_id}/intervention/weights",
            json={"agent": agent, "weight": weight},
        )


class AsyncDebatesAPI:
    """
    Asynchronous Debates API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     debate = await client.debates.create(task="Should we use microservices?")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def create(
        self,
        task: str,
        agents: _List[str] | None = None,
        protocol: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a new debate."""
        data = {"task": task, **kwargs}
        if agents:
            data["agents"] = agents
        if protocol:
            data["protocol"] = protocol

        return await self._client.request("POST", "/api/v1/debates", json=data)

    async def get(self, debate_id: str) -> dict[str, Any]:
        """Get a debate by ID."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}")

    async def list(
        self,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
    ) -> dict[str, Any]:
        """List debates with pagination."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        return await self._client.request("GET", "/api/v1/debates", params=params)

    def list_all(
        self,
        status: str | None = None,
        page_size: int = 20,
    ) -> AsyncPaginator:
        """
        Iterate through all debates with automatic pagination.

        Args:
            status: Filter by status (active, completed, etc.)
            page_size: Number of debates per page (default 20)

        Returns:
            AsyncPaginator yielding debate dictionaries

        Example::

            async for debate in client.debates.list_all(status="active"):
                print(debate["id"])
        """
        from ..pagination import AsyncPaginator

        params = {}
        if status:
            params["status"] = status

        return AsyncPaginator(self._client, "/api/v1/debates", params, page_size)

    async def stream(self, debate_id: str) -> AsyncIterator[WebSocketEvent]:
        """
        Stream debate events via WebSocket.

        Yields events as they occur during the debate. Automatically connects
        and disconnects the WebSocket.

        Args:
            debate_id: The ID of the debate to stream

        Yields:
            WebSocketEvent objects with typed_data for structured access

        Example::

            async for event in client.debates.stream(debate_id):
                print(event.type, event.data)
                if event.type == "debate_end":
                    break
        """
        from ..websocket import AragoraWebSocket

        ws = AragoraWebSocket(
            base_url=self._client.base_url,
            api_key=self._client.api_key,
        )
        await ws.connect(debate_id=debate_id)
        try:
            async for event in ws.events():
                yield event
                if event.type in ("debate_end", "error"):
                    break
        finally:
            await ws.close()

    async def get_messages(self, debate_id: str) -> dict[str, Any]:
        """Get messages from a debate."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/messages")

    async def add_message(
        self,
        debate_id: str,
        content: str,
        role: str = "user",
    ) -> dict[str, Any]:
        """Add a message to a debate."""
        return await self._client.request(
            "POST",
            f"/api/v1/debates/{debate_id}/messages",
            json={"content": content, "role": role},
        )

    async def get_consensus(self, debate_id: str) -> dict[str, Any]:
        """Get consensus information for a debate."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/consensus")

    async def get_export(
        self,
        debate_id: str,
        format: str = "json",
    ) -> dict[str, Any]:
        """Export a debate."""
        return await self._client.request(
            "GET",
            f"/api/v1/debates/{debate_id}/export",
            params={"format": format},
        )

    async def cancel(self, debate_id: str) -> dict[str, Any]:
        """Cancel a debate."""
        return await self._client.request("POST", f"/api/v1/debates/{debate_id}/cancel")

    async def get_by_slug(self, slug: str) -> dict[str, Any]:
        """Get a debate by slug."""
        return await self._client.request("GET", f"/api/v1/debates/slug/{slug}")

    async def get_explainability(self, debate_id: str) -> dict[str, Any]:
        """Get explainability data for a debate."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/explainability")

    async def get_explainability_factors(self, debate_id: str) -> dict[str, Any]:
        """Get factor decomposition for a debate decision."""
        return await self._client.request(
            "GET", f"/api/v1/debates/{debate_id}/explainability/factors"
        )

    async def get_explainability_narrative(self, debate_id: str) -> dict[str, Any]:
        """Get natural language narrative explanation."""
        return await self._client.request(
            "GET", f"/api/v1/debates/{debate_id}/explainability/narrative"
        )

    async def get_explainability_provenance(self, debate_id: str) -> dict[str, Any]:
        """Get provenance chain for debate claims."""
        return await self._client.request(
            "GET", f"/api/v1/debates/{debate_id}/explainability/provenance"
        )

    async def get_explainability_counterfactual(self, debate_id: str) -> dict[str, Any]:
        """Get counterfactual analysis."""
        return await self._client.request(
            "GET", f"/api/v1/debates/{debate_id}/explainability/counterfactual"
        )

    async def create_counterfactual(
        self, debate_id: str, changes: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a counterfactual scenario."""
        return await self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/explainability/counterfactual", json=changes
        )

    async def get_citations(self, debate_id: str) -> dict[str, Any]:
        """Get citations used in debate."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/citations")

    async def get_convergence(self, debate_id: str) -> dict[str, Any]:
        """Get convergence analysis."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/convergence")

    async def get_evidence(self, debate_id: str) -> dict[str, Any]:
        """Get evidence collected during debate."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/evidence")

    async def get_graph_stats(self, debate_id: str) -> dict[str, Any]:
        """Get graph statistics for debate."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/graph/stats")

    async def get_impasse(self, debate_id: str) -> dict[str, Any]:
        """Get impasse analysis if debate is stuck."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/impasse")

    async def get_meta_critique(self, debate_id: str) -> dict[str, Any]:
        """Get meta-critique of debate process."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/meta-critique")

    async def get_red_team(self, debate_id: str) -> dict[str, Any]:
        """Get red team analysis."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/red-team")

    async def capability_probe(self, task: str, agents: _List[str] | None = None) -> dict[str, Any]:
        """Run a capability probe debate."""
        data: dict[str, Any] = {"task": task}
        if agents:
            data["agents"] = agents
        return await self._client.request("POST", "/api/v1/debates/capability-probe", json=data)

    async def deep_audit(self, task: str, agents: _List[str] | None = None) -> dict[str, Any]:
        """Run a deep audit debate."""
        data: dict[str, Any] = {"task": task}
        if agents:
            data["agents"] = agents
        return await self._client.request("POST", "/api/v1/debates/deep-audit", json=data)

    async def broadcast(self, debate_id: str, channels: _List[str] | None = None) -> dict[str, Any]:
        """Broadcast debate to channels."""
        data: dict[str, Any] = {"channels": channels} if channels else {}
        return await self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/broadcast", json=data
        )

    async def fork(self, debate_id: str, changes: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fork a debate with optional changes."""
        return await self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/fork", json=changes or {}
        )

    async def publish_twitter(self, debate_id: str, message: str | None = None) -> dict[str, Any]:
        """Publish debate summary to Twitter."""
        data: dict[str, Any] = {"message": message} if message else {}
        return await self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/publish/twitter", json=data
        )

    async def publish_youtube(self, debate_id: str, title: str | None = None) -> dict[str, Any]:
        """Publish debate to YouTube."""
        data: dict[str, Any] = {"title": title} if title else {}
        return await self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/publish/youtube", json=data
        )

    async def get_dashboard(self) -> dict[str, Any]:
        """Get debates dashboard view."""
        return await self._client.request("GET", "/api/v1/dashboard/debates")

    async def get_debate_graph_stats(self, debate_id: str) -> dict[str, Any]:
        """Get graph stats via debate endpoint."""
        return await self._client.request("GET", f"/api/v1/debate/{debate_id}/graph-stats")

    async def get_history(self, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        """Get debate history."""
        return await self._client.request(
            "GET", "/api/v1/history/debates", params={"limit": limit, "offset": offset}
        )

    # ========== Additional CRUD ==========

    async def update(self, debate_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        """Update an existing debate."""
        return await self._client.request("PUT", f"/api/v1/debates/{debate_id}", json=updates)

    async def delete(self, debate_id: str) -> dict[str, Any]:
        """Delete a debate."""
        return await self._client.request("DELETE", f"/api/v1/debates/{debate_id}")

    async def clone(
        self,
        debate_id: str,
        preserve_agents: bool = True,
        preserve_context: bool = False,
    ) -> dict[str, Any]:
        """Clone a debate (create a copy with fresh state)."""
        return await self._client.request(
            "POST",
            f"/api/v1/debates/{debate_id}/clone",
            json={"preserveAgents": preserve_agents, "preserveContext": preserve_context},
        )

    async def archive(self, debate_id: str) -> dict[str, Any]:
        """Archive a debate."""
        return await self._client.request("POST", f"/api/v1/debates/{debate_id}/archive")

    # ========== Lifecycle ==========

    async def start(self, debate_id: str) -> dict[str, Any]:
        """Start a debate."""
        return await self._client.request("POST", f"/api/v1/debates/{debate_id}/start")

    async def stop(self, debate_id: str) -> dict[str, Any]:
        """Stop a running debate."""
        return await self._client.request("POST", f"/api/v1/debates/{debate_id}/stop")

    async def pause(self, debate_id: str) -> dict[str, Any]:
        """Pause a running debate."""
        return await self._client.request("POST", f"/api/v1/debates/{debate_id}/pause")

    async def resume(self, debate_id: str) -> dict[str, Any]:
        """Resume a paused debate."""
        return await self._client.request("POST", f"/api/v1/debates/{debate_id}/resume")

    # ========== Analysis ==========

    async def get_rhetorical(self, debate_id: str) -> dict[str, Any]:
        """Get rhetorical pattern observations for a debate."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/rhetorical")

    async def get_trickster(self, debate_id: str) -> dict[str, Any]:
        """Get trickster hollow consensus detection status."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/trickster")

    async def get_summary(self, debate_id: str) -> dict[str, Any]:
        """Get a human-readable summary of the debate."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/summary")

    async def get_verification_report(self, debate_id: str) -> dict[str, Any]:
        """Get the verification report for debate conclusions."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/verification-report")

    async def verify_claim(
        self,
        debate_id: str,
        claim_id: str,
        evidence: str | None = None,
    ) -> dict[str, Any]:
        """Verify a specific claim from the debate."""
        payload: dict[str, Any] = {"claim_id": claim_id}
        if evidence:
            payload["evidence"] = evidence
        return await self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/verify", json=payload
        )

    # ========== Follow-up & Continuation ==========

    async def get_followup_suggestions(self, debate_id: str) -> dict[str, Any]:
        """Get suggestions for follow-up debates."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/followups")

    async def follow_up(
        self,
        debate_id: str,
        crux_id: str | None = None,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Create a follow-up debate from an existing one."""
        payload: dict[str, Any] = {}
        if crux_id:
            payload["cruxId"] = crux_id
        if context:
            payload["context"] = context
        return await self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/followup", json=payload
        )

    async def list_forks(self, debate_id: str) -> dict[str, Any]:
        """List all forks created from a debate."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/forks")

    # ========== Rounds, Agents, Votes ==========

    async def get_rounds(self, debate_id: str) -> dict[str, Any]:
        """Get rounds from a debate."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/rounds")

    async def get_agents(self, debate_id: str) -> dict[str, Any]:
        """Get agents participating in a debate."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/agents")

    async def get_votes(self, debate_id: str) -> dict[str, Any]:
        """Get votes from a debate."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/votes")

    async def add_user_input(
        self,
        debate_id: str,
        input_text: str,
        input_type: str = "suggestion",  # 'suggestion' | 'vote' | 'question' | 'context'
    ) -> dict[str, Any]:
        """Add user input to a debate."""
        return await self._client.request(
            "POST",
            f"/api/v1/debates/{debate_id}/user-input",
            json={"input": input_text, "type": input_type},
        )

    async def get_timeline(self, debate_id: str) -> dict[str, Any]:
        """Get the timeline of events in a debate."""
        return await self._client.request("GET", f"/api/v1/debates/{debate_id}/timeline")

    async def add_evidence(
        self,
        debate_id: str,
        evidence: str,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add evidence to a debate."""
        payload: dict[str, Any] = {"evidence": evidence}
        if source:
            payload["source"] = source
        if metadata:
            payload["metadata"] = metadata
        return await self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/evidence", json=payload
        )

    # ========== Search ==========

    async def search(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
        domain: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        """Search across all debates."""
        params: dict[str, Any] = {"query": query, "limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if domain:
            params["domain"] = domain
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        return await self._client.request("GET", "/api/v1/search", params=params)

    # ========== Batch Operations ==========

    async def submit_batch(self, requests: _List[dict[str, Any]]) -> dict[str, Any]:
        """Submit multiple debates for batch processing."""
        return await self._client.request(
            "POST", "/api/v1/debates/batch", json={"requests": requests}
        )

    async def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        """Get the status of a batch job."""
        return await self._client.request("GET", f"/api/v1/debates/batch/{batch_id}/status")

    async def list_batches(
        self,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
    ) -> dict[str, Any]:
        """List all batch jobs."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        return await self._client.request("GET", "/api/v1/debates/batch", params=params)

    async def get_queue_status(self) -> dict[str, Any]:
        """Get the current queue status."""
        return await self._client.request("GET", "/api/v1/debates/queue/status")

    # ========== Graph & Visualization ==========

    async def get_graph(self, debate_id: str) -> dict[str, Any]:
        """Get the argument graph for a debate."""
        return await self._client.request("GET", f"/api/v1/debates/graph/{debate_id}")

    async def get_graph_branches(self, debate_id: str) -> dict[str, Any]:
        """Get branches in the argument graph."""
        return await self._client.request("GET", f"/api/v1/debates/graph/{debate_id}/branches")

    async def get_matrix_comparison(self, debate_id: str) -> dict[str, Any]:
        """Get matrix comparison for a multi-scenario debate."""
        return await self._client.request("GET", f"/api/v1/debates/matrix/{debate_id}")

    # ========== Streaming ==========

    async def stream_debates(self) -> dict[str, Any]:
        """Stream debate events via Server-Sent Events.

        Returns an SSE stream of real-time debate events including
        debate_started, round_start, agent_message, consensus, and debate_end.

        Returns:
            SSE stream response with debate events
        """
        return await self._client.request("GET", "/api/v1/debates/stream")

    # ========== RLM / Compression ==========

    async def compress(
        self,
        debate_id: str,
        target_levels: _List[str] | None = None,
        compression_ratio: float = 0.3,
    ) -> dict[str, Any]:
        """Compress debate context using RLM hierarchical abstraction.

        Args:
            debate_id: The debate ID
            target_levels: Abstraction levels to generate (e.g. ["ABSTRACT", "SUMMARY", "DETAILED"])
            compression_ratio: Target compression ratio (0.0-1.0, default 0.3)

        Returns:
            Compression result with original_tokens, compressed_tokens, compression_ratios,
            time_seconds, and levels_created
        """
        payload: dict[str, Any] = {"compression_ratio": compression_ratio}
        if target_levels:
            payload["target_levels"] = target_levels
        return await self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/compress", json=payload
        )

    async def get_context_level(self, debate_id: str, level: str) -> dict[str, Any]:
        """Get debate content at a specific abstraction level.

        Args:
            debate_id: The debate ID
            level: Abstraction level (ABSTRACT, SUMMARY, DETAILED, or RAW)

        Returns:
            Context at the requested level with content, token_count, and nodes
        """
        return await self._client.request(
            "GET", f"/api/v1/debates/{debate_id}/context/{level}"
        )

    async def query_rlm(
        self,
        debate_id: str,
        query: str,
        strategy: str = "auto",
        max_iterations: int = 3,
        start_level: str = "SUMMARY",
    ) -> dict[str, Any]:
        """Query a debate using RLM with iterative refinement.

        Args:
            debate_id: The debate ID
            query: The question to ask about the debate
            strategy: Query strategy (auto, peek, grep, partition_map, summarize, hierarchical)
            max_iterations: Maximum refinement iterations (1-10, default 3)
            start_level: Starting abstraction level (ABSTRACT, SUMMARY, DETAILED, RAW)

        Returns:
            Query result with answer, confidence, refinement_history, and token usage
        """
        return await self._client.request(
            "POST",
            f"/api/v1/debates/{debate_id}/query-rlm",
            json={
                "query": query,
                "strategy": strategy,
                "max_iterations": max_iterations,
                "start_level": start_level,
            },
        )

    async def get_refinement_status(self, debate_id: str) -> dict[str, Any]:
        """Get the status of an ongoing RLM refinement process.

        Args:
            debate_id: The debate ID

        Returns:
            Refinement status with active_queries, cached_contexts, and status
        """
        return await self._client.request(
            "GET", f"/api/v1/debates/{debate_id}/refinement-status"
        )

    # ========== Decision Integrity ==========

    async def get_decision_integrity(self, debate_id: str) -> dict[str, Any]:
        """Get the decision integrity package for a debate.

        Generates a decision receipt and implementation plan bundle
        containing audit-ready documentation of the debate outcome.

        Args:
            debate_id: The debate ID

        Returns:
            Decision integrity package with receipt and implementation plan
        """
        return await self._client.request(
            "GET", f"/api/v1/debates/{debate_id}/decision-integrity"
        )

    # ========== Intervention ==========

    async def intervention_inject(
        self,
        debate_id: str,
        content: str,
        injection_type: str = "argument",
        source: str = "user",
    ) -> dict[str, Any]:
        """Inject a user argument or follow-up question into a debate.

        The injected content will be included in the next round's context
        and considered by all agents.

        Args:
            debate_id: The active debate ID
            content: The argument or question to inject
            injection_type: Type of injection ("argument" or "follow_up")
            source: Source identifier (default "user")

        Returns:
            Injection result with injection_id and confirmation
        """
        return await self._client.request(
            "POST",
            f"/api/v1/debates/{debate_id}/intervention/inject",
            json={"content": content, "type": injection_type, "source": source},
        )

    async def intervention_log(self, debate_id: str, limit: int = 50) -> dict[str, Any]:
        """Get the intervention audit log for a debate.

        Returns all interventions with timestamps for compliance and audit purposes.

        Args:
            debate_id: The debate ID
            limit: Maximum number of log entries to return (default 50)

        Returns:
            Intervention log with total_interventions and interventions list
        """
        return await self._client.request(
            "GET",
            f"/api/v1/debates/{debate_id}/intervention/log",
            params={"limit": limit},
        )

    async def intervention_pause(self, debate_id: str) -> dict[str, Any]:
        """Pause an active debate.

        Pausing stops agent responses but preserves all debate state.
        The debate can be resumed at any point.

        Args:
            debate_id: The active debate ID

        Returns:
            Pause result with is_paused status and paused_at timestamp
        """
        return await self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/intervention/pause"
        )

    async def intervention_resume(self, debate_id: str) -> dict[str, Any]:
        """Resume a paused debate.

        Resumes agent responses from where they left off.

        Args:
            debate_id: The paused debate ID

        Returns:
            Resume result with resumed_at timestamp and pause_duration_seconds
        """
        return await self._client.request(
            "POST", f"/api/v1/debates/{debate_id}/intervention/resume"
        )

    async def intervention_state(self, debate_id: str) -> dict[str, Any]:
        """Get the current intervention state for a debate.

        Returns pause status, agent weights, consensus threshold,
        and counts of pending injections and follow-ups.

        Args:
            debate_id: The debate ID

        Returns:
            Intervention state with is_paused, consensus_threshold, agent_weights,
            pending_injections, and pending_follow_ups
        """
        return await self._client.request(
            "GET", f"/api/v1/debates/{debate_id}/intervention/state"
        )

    async def intervention_update_threshold(
        self, debate_id: str, threshold: float
    ) -> dict[str, Any]:
        """Update the consensus threshold for a debate.

        Threshold is the minimum agreement level required for consensus:
        0.5 = simple majority, 0.75 = strong majority (default), 1.0 = unanimous.

        Args:
            debate_id: The active debate ID
            threshold: New threshold value (0.5 to 1.0)

        Returns:
            Update result with old_threshold and new_threshold
        """
        return await self._client.request(
            "PUT",
            f"/api/v1/debates/{debate_id}/intervention/threshold",
            json={"threshold": threshold},
        )

    async def intervention_update_weights(
        self, debate_id: str, agent: str, weight: float
    ) -> dict[str, Any]:
        """Update an agent's influence weight in a debate.

        Weight affects how much the agent's vote counts in consensus:
        0.0 = muted, 1.0 = normal influence, 2.0 = double influence.

        Args:
            debate_id: The active debate ID
            agent: Agent name or ID
            weight: New weight value (0.0 to 2.0)

        Returns:
            Update result with agent, old_weight, and new_weight
        """
        return await self._client.request(
            "PUT",
            f"/api/v1/debates/{debate_id}/intervention/weights",
            json={"agent": agent, "weight": weight},
        )
