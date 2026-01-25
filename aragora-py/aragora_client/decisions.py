"""Decisions API for the Aragora SDK.

Provides a unified interface for creating and tracking decisions:
- Debate-based decisions
- Workflow-based decisions
- Gauntlet verification decisions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora_client.client import AragoraClient


class DecisionType(str, Enum):
    """Type of decision request."""

    AUTO = "auto"
    DEBATE = "debate"
    WORKFLOW = "workflow"
    GAUNTLET = "gauntlet"
    QUICK = "quick"


class DecisionStatus(str, Enum):
    """Status of a decision request."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"
    UNKNOWN = "unknown"


def _parse_decision_type(raw: str | None) -> DecisionType:
    if not raw:
        return DecisionType.AUTO
    try:
        return DecisionType(raw)
    except ValueError:
        return DecisionType.AUTO


def _parse_decision_status(raw: str | None) -> DecisionStatus:
    if not raw:
        return DecisionStatus.UNKNOWN
    try:
        return DecisionStatus(raw)
    except ValueError:
        return DecisionStatus.UNKNOWN


@dataclass
class DecisionResult:
    """Result of a decision request."""

    request_id: str
    decision_type: DecisionType
    status: DecisionStatus
    answer: str | None = None
    confidence: float | None = None
    consensus_reached: bool | None = None
    reasoning: str | None = None
    evidence_used: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float | None = None
    created_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    task: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionResult:
        payload = data.get("result", {})
        decision_type = _parse_decision_type(
            data.get("decision_type")
            or payload.get("decision_type")
            or data.get("type")
        )
        status = _parse_decision_status(data.get("status") or payload.get("status"))
        return cls(
            request_id=data.get("request_id")
            or data.get("id")
            or payload.get("request_id", ""),
            decision_type=decision_type,
            status=status,
            answer=data.get("answer")
            or payload.get("answer")
            or data.get("conclusion"),
            confidence=data.get("confidence") or payload.get("confidence"),
            consensus_reached=data.get("consensus_reached")
            if "consensus_reached" in data
            else payload.get("consensus_reached"),
            reasoning=data.get("reasoning") or payload.get("reasoning"),
            evidence_used=data.get("evidence_used")
            or payload.get("evidence_used")
            or [],
            duration_seconds=data.get("duration_seconds")
            or payload.get("duration_seconds"),
            created_at=data.get("created_at") or payload.get("created_at"),
            completed_at=data.get("completed_at") or payload.get("completed_at"),
            error=data.get("error") or payload.get("error"),
            metadata=data.get("metadata") or payload.get("metadata") or {},
            task=data.get("task") or data.get("content"),
        )

    @property
    def id(self) -> str:
        return self.request_id

    @property
    def type(self) -> DecisionType:
        return self.decision_type

    @property
    def conclusion(self) -> str | None:
        return self.answer


@dataclass
class DecisionStatusInfo:
    """Status information for polling."""

    request_id: str
    status: DecisionStatus
    progress: float  # 0-100
    current_phase: str | None
    eta_seconds: int | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionStatusInfo:
        return cls(
            request_id=data.get("request_id") or data.get("id", ""),
            status=_parse_decision_status(data.get("status")),
            progress=data.get("progress", 0.0),
            current_phase=data.get("current_phase"),
            eta_seconds=data.get("eta_seconds"),
        )

    @property
    def id(self) -> str:
        return self.request_id


class DecisionsAPI:
    """API for unified decision operations.

    Provides a single interface for creating decisions that can be
    resolved through debates, workflows, or gauntlet verification.
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Decision Creation
    # =========================================================================

    async def create(
        self,
        task: str,
        *,
        decision_type: DecisionType = DecisionType.DEBATE,
        agents: list[str] | None = None,
        max_rounds: int = 5,
        consensus_threshold: float = 0.8,
        workflow_id: str | None = None,
        gauntlet_checks: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        async_mode: bool = True,
    ) -> DecisionResult:
        """Create a new decision request.

        Creates a decision that will be resolved using the specified method
        (debate, workflow, or gauntlet verification).

        Args:
            task: The decision question or task to resolve
            decision_type: Method to use (debate, workflow, gauntlet)
            agents: Agents to participate (for debates)
            max_rounds: Maximum debate rounds (for debates)
            consensus_threshold: Required consensus level (for debates)
            workflow_id: Workflow template ID (for workflow decisions)
            gauntlet_checks: Verification checks to run (for gauntlet)
            metadata: Additional metadata to attach
            async_mode: If True, returns immediately; poll status for result

        Returns:
            DecisionResult with initial status

        Example:
            # Create a debate-based decision
            result = await client.decisions.create(
                "Should we migrate to microservices?",
                decision_type=DecisionType.DEBATE,
                agents=["claude", "gpt-4"],
                max_rounds=3
            )

            # Create a workflow-based decision
            result = await client.decisions.create(
                "Evaluate Q4 budget proposal",
                decision_type=DecisionType.WORKFLOW,
                workflow_id="budget-review-v2"
            )
        """
        data: dict[str, Any] = {
            "content": task,
            "decision_type": decision_type.value,
        }

        config: dict[str, Any] = {}
        if agents:
            config["agents"] = agents
        if max_rounds != 5:
            config["rounds"] = max_rounds
        if consensus_threshold != 0.8:
            config["consensus_threshold"] = consensus_threshold
        if workflow_id:
            config["workflow_id"] = workflow_id
        if gauntlet_checks:
            config["attack_categories"] = gauntlet_checks
        if async_mode is False:
            config["async"] = False
        if config:
            data["config"] = config
        if metadata:
            data["context"] = {"metadata": metadata}

        response = await self._client._post("/api/v1/decisions", data)
        result = DecisionResult.from_dict(response)
        if result.task is None:
            result.task = task
        return result

    # =========================================================================
    # Decision Retrieval
    # =========================================================================

    async def get(self, decision_id: str) -> DecisionResult:
        """Get a decision result by ID.

        Args:
            decision_id: The decision request ID

        Returns:
            DecisionResult with full result if completed

        Example:
            result = await client.decisions.get("dec_abc123")
            if result.status == DecisionStatus.COMPLETED:
                print(f"Conclusion: {result.conclusion}")
        """
        response = await self._client._get(f"/api/v1/decisions/{decision_id}")
        return DecisionResult.from_dict(response)

    async def get_status(self, decision_id: str) -> DecisionStatusInfo:
        """Get decision status for polling.

        Lightweight endpoint for checking progress without full result.

        Args:
            decision_id: The decision request ID

        Returns:
            DecisionStatusInfo with progress information

        Example:
            while True:
                status = await client.decisions.get_status("dec_abc123")
                print(f"Progress: {status.progress}%")
                if status.status in [DecisionStatus.COMPLETED, DecisionStatus.FAILED]:
                    break
                await asyncio.sleep(2)
        """
        response = await self._client._get(f"/api/v1/decisions/{decision_id}/status")
        return DecisionStatusInfo.from_dict(response)

    async def list(
        self,
        *,
        status: DecisionStatus | None = None,
        decision_type: DecisionType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[DecisionResult]:
        """List decisions with optional filtering.

        Args:
            status: Filter by status
            decision_type: Filter by type
            limit: Maximum results to return
            offset: Offset for pagination

        Returns:
            List of DecisionResult objects

        Example:
            # Get recent completed decisions
            decisions = await client.decisions.list(
                status=DecisionStatus.COMPLETED,
                limit=10
            )
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status.value
        if decision_type:
            params["type"] = decision_type.value

        response = await self._client._get("/api/v1/decisions", params=params)
        return [DecisionResult.from_dict(d) for d in response.get("decisions", [])]

    # =========================================================================
    # Decision Management
    # =========================================================================

    async def cancel(self, decision_id: str) -> DecisionResult:
        """Cancel a pending or running decision.

        Args:
            decision_id: The decision request ID

        Returns:
            Updated DecisionResult with cancelled status

        Example:
            result = await client.decisions.cancel("dec_abc123")
            assert result.status == DecisionStatus.CANCELLED
        """
        response = await self._client._post(
            f"/api/v1/decisions/{decision_id}/cancel", {}
        )
        return DecisionResult.from_dict(response)

    async def retry(self, decision_id: str) -> DecisionResult:
        """Retry a failed decision.

        Creates a new decision with the same parameters as the failed one.

        Args:
            decision_id: The failed decision ID

        Returns:
            New DecisionResult for the retry

        Example:
            # Retry a failed decision
            original = await client.decisions.get("dec_abc123")
            if original.status == DecisionStatus.FAILED:
                new_result = await client.decisions.retry("dec_abc123")
        """
        response = await self._client._post(
            f"/api/v1/decisions/{decision_id}/retry", {}
        )
        return DecisionResult.from_dict(response)

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def wait_for_completion(
        self,
        decision_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: float | None = None,
    ) -> DecisionResult:
        """Wait for a decision to complete.

        Polls the decision status until it reaches a terminal state.

        Args:
            decision_id: The decision request ID
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait (None for no timeout)

        Returns:
            Final DecisionResult

        Raises:
            TimeoutError: If timeout is reached

        Example:
            result = await client.decisions.create("Should we...?")
            final = await client.decisions.wait_for_completion(
                result.id,
                timeout=300  # 5 minute timeout
            )
        """
        import asyncio
        import time

        start_time = time.time()
        terminal_states = {
            DecisionStatus.COMPLETED,
            DecisionStatus.FAILED,
            DecisionStatus.CANCELLED,
        }

        while True:
            status = await self.get_status(decision_id)
            if status.status in terminal_states:
                return await self.get(decision_id)

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(
                    f"Decision {decision_id} did not complete within {timeout}s"
                )

            await asyncio.sleep(poll_interval)
