"""
Nomic Namespace API

Provides methods for nomic loop control and monitoring:
- Get loop state, health, and metrics
- Start, stop, pause, and resume the loop
- Manage improvement proposals
- Access risk register and logs
- Operational modes management
- Gas Town witness and mayor monitoring

The nomic loop is the autonomous self-improvement cycle that enables
the system to propose, design, implement, and verify its own improvements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class NomicAPI:
    """
    Synchronous Nomic API.

    Provides methods for nomic loop control and monitoring:
    - Get loop state, health, and metrics
    - Start, stop, pause, and resume the loop
    - Manage improvement proposals
    - Access risk register and logs
    - Operational modes management
    - Gas Town witness and mayor monitoring

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> health = client.nomic.get_health()
        >>> print(f"Status: {health['status']}, Phase: {health['phase']}")
        >>> client.nomic.start(cycles=3, auto_approve=False)
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # State & Monitoring
    # ===========================================================================

    def get_state(self) -> dict[str, Any]:
        """
        Get current nomic loop state.

        Returns:
            Dict with running, cycle, phase, paused, started_at, last_update,
            target_cycles, and auto_approve fields
        """
        return self._client.request("GET", "/api/v1/nomic/state")

    def get_health(self) -> dict[str, Any]:
        """
        Get nomic loop health with stall detection.

        Returns:
            Dict with status (healthy/stalled/not_running/error), cycle, phase,
            last_activity, stall_duration_seconds, and warnings
        """
        return self._client.request("GET", "/api/v1/nomic/health")

    def get_metrics(self) -> dict[str, Any]:
        """
        Get nomic loop metrics summary.

        Returns:
            Dict with summary, stuck_detection, and status
        """
        return self._client.request("GET", "/api/v1/nomic/metrics")

    def get_logs(self, lines: int = 100) -> dict[str, Any]:
        """
        Get recent nomic loop log lines.

        Args:
            lines: Number of log lines to return (default: 100, max: 1000)

        Returns:
            Dict with lines array, total, and showing count
        """
        return self._client.request("GET", "/api/v1/nomic/log", params={"lines": lines})

    def get_risk_register(self, limit: int | None = None) -> dict[str, Any]:
        """
        Get risk register entries.

        Args:
            limit: Maximum entries to return

        Returns:
            Dict with risks array, total, critical_count, and high_count
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        return self._client.request("GET", "/api/v1/nomic/risk-register", params=params)

    # ===========================================================================
    # Loop Control
    # ===========================================================================

    def start(
        self,
        cycles: int | None = None,
        max_cycles: int | None = None,
        auto_approve: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Start the nomic loop.

        Args:
            cycles: Target number of cycles to run
            max_cycles: Alias for cycles (deprecated)
            auto_approve: Auto-approve proposals without human review
            dry_run: Preview changes without executing

        Returns:
            Dict with status, pid, and target_cycles
        """
        data: dict[str, Any] = {
            "auto_approve": auto_approve,
            "dry_run": dry_run,
        }
        if cycles is not None:
            data["cycles"] = cycles
        elif max_cycles is not None:
            data["max_cycles"] = max_cycles
        return self._client.request("POST", "/api/v1/nomic/control/start", json=data)

    def stop(self, graceful: bool = True) -> dict[str, Any]:
        """
        Stop the running nomic loop.

        Args:
            graceful: Allow current phase to complete (default: True)

        Returns:
            Dict with status and pid
        """
        data: dict[str, Any] = {"graceful": graceful}
        return self._client.request("POST", "/api/v1/nomic/control/stop", json=data)

    def pause(self) -> dict[str, Any]:
        """
        Pause the nomic loop at the current phase.

        Returns:
            Dict with status, cycle, and phase
        """
        return self._client.request("POST", "/api/v1/nomic/control/pause")

    def resume(self) -> dict[str, Any]:
        """
        Resume a paused nomic loop.

        Returns:
            Dict with status, cycle, and phase
        """
        return self._client.request("POST", "/api/v1/nomic/control/resume")

    def skip_phase(self) -> dict[str, Any]:
        """
        Skip the current phase and move to the next.

        Returns:
            Dict with status, previous_phase, next_phase, and cycle
        """
        return self._client.request("POST", "/api/v1/nomic/control/skip-phase")

    # ===========================================================================
    # Proposal Management
    # ===========================================================================

    def get_proposals(self) -> dict[str, Any]:
        """
        Get pending improvement proposals.

        Returns:
            Dict with proposals array, total, and all_proposals count
        """
        return self._client.request("GET", "/api/v1/nomic/proposals")

    def approve_proposal(
        self,
        proposal_id: str,
        approved_by: str | None = None,
    ) -> dict[str, Any]:
        """
        Approve a pending proposal.

        Args:
            proposal_id: The proposal ID to approve
            approved_by: Identifier of the approver

        Returns:
            Dict with status and proposal_id
        """
        data: dict[str, Any] = {"proposal_id": proposal_id}
        if approved_by:
            data["approved_by"] = approved_by
        return self._client.request("POST", "/api/v1/nomic/proposals/approve", json=data)

    def reject_proposal(
        self,
        proposal_id: str,
        rejected_by: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """
        Reject a pending proposal.

        Args:
            proposal_id: The proposal ID to reject
            rejected_by: Identifier of the rejector
            reason: Reason for rejection

        Returns:
            Dict with status and proposal_id
        """
        data: dict[str, Any] = {"proposal_id": proposal_id}
        if rejected_by:
            data["rejected_by"] = rejected_by
        if reason:
            data["reason"] = reason
        return self._client.request("POST", "/api/v1/nomic/proposals/reject", json=data)

    # ===========================================================================
    # Operational Modes
    # ===========================================================================

    def get_modes(self) -> dict[str, Any]:
        """
        Get available operational modes (builtin + custom).

        Returns:
            Dict with modes array and total count
        """
        return self._client.request("GET", "/api/v1/modes")

    # ===========================================================================
    # Gas Town (Witness & Mayor)
    # ===========================================================================

    def get_witness_status(self) -> dict[str, Any]:
        """
        Get Gas Town witness patrol status.

        Witnesses monitor the nomic loop for violations and irregularities.

        Returns:
            Dict with active, patrol_count, last_patrol, violations_detected,
            current_focus, and witnesses array
        """
        return self._client.request("GET", "/api/v1/nomic/witness/status")

    def get_mayor_current(self) -> dict[str, Any]:
        """
        Get current Gas Town mayor information.

        The mayor is the elected leader who can enact policies and use emergency powers.

        Returns:
            Dict with current_mayor, elected_at, term_ends, approval_rating,
            policies_enacted, vetoes, and emergency_powers_active
        """
        return self._client.request("GET", "/api/v1/nomic/mayor/current")

    # ===========================================================================
    # Convenience Methods
    # ===========================================================================

    def state(self) -> dict[str, Any]:
        """Alias for get_state()."""
        return self.get_state()

    def health(self) -> dict[str, Any]:
        """Alias for get_health()."""
        return self.get_health()

    def is_running(self) -> bool:
        """
        Check if the nomic loop is currently running.

        Returns:
            True if running and not paused
        """
        state = self.get_state()
        return state.get("running", False) and not state.get("paused", False)

    def get_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        """
        Get a specific proposal by ID.

        Args:
            proposal_id: The proposal ID to find

        Returns:
            Proposal dict or None if not found
        """
        response = self.get_proposals()
        for proposal in response.get("proposals", []):
            if proposal.get("id") == proposal_id:
                return proposal
        return None


class AsyncNomicAPI:
    """
    Asynchronous Nomic API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     health = await client.nomic.get_health()
        ...     await client.nomic.start(cycles=3, auto_approve=False)
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # State & Monitoring
    # ===========================================================================

    async def get_state(self) -> dict[str, Any]:
        """Get current nomic loop state."""
        return await self._client.request("GET", "/api/v1/nomic/state")

    async def get_health(self) -> dict[str, Any]:
        """Get nomic loop health with stall detection."""
        return await self._client.request("GET", "/api/v1/nomic/health")

    async def get_metrics(self) -> dict[str, Any]:
        """Get nomic loop metrics summary."""
        return await self._client.request("GET", "/api/v1/nomic/metrics")

    async def get_logs(self, lines: int = 100) -> dict[str, Any]:
        """Get recent nomic loop log lines."""
        return await self._client.request("GET", "/api/v1/nomic/log", params={"lines": lines})

    async def get_risk_register(self, limit: int | None = None) -> dict[str, Any]:
        """Get risk register entries."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        return await self._client.request("GET", "/api/v1/nomic/risk-register", params=params)

    # ===========================================================================
    # Loop Control
    # ===========================================================================

    async def start(
        self,
        cycles: int | None = None,
        max_cycles: int | None = None,
        auto_approve: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Start the nomic loop."""
        data: dict[str, Any] = {
            "auto_approve": auto_approve,
            "dry_run": dry_run,
        }
        if cycles is not None:
            data["cycles"] = cycles
        elif max_cycles is not None:
            data["max_cycles"] = max_cycles
        return await self._client.request("POST", "/api/v1/nomic/control/start", json=data)

    async def stop(self, graceful: bool = True) -> dict[str, Any]:
        """Stop the running nomic loop."""
        data: dict[str, Any] = {"graceful": graceful}
        return await self._client.request("POST", "/api/v1/nomic/control/stop", json=data)

    async def pause(self) -> dict[str, Any]:
        """Pause the nomic loop at the current phase."""
        return await self._client.request("POST", "/api/v1/nomic/control/pause")

    async def resume(self) -> dict[str, Any]:
        """Resume a paused nomic loop."""
        return await self._client.request("POST", "/api/v1/nomic/control/resume")

    async def skip_phase(self) -> dict[str, Any]:
        """Skip the current phase and move to the next."""
        return await self._client.request("POST", "/api/v1/nomic/control/skip-phase")

    # ===========================================================================
    # Proposal Management
    # ===========================================================================

    async def get_proposals(self) -> dict[str, Any]:
        """Get pending improvement proposals."""
        return await self._client.request("GET", "/api/v1/nomic/proposals")

    async def approve_proposal(
        self,
        proposal_id: str,
        approved_by: str | None = None,
    ) -> dict[str, Any]:
        """Approve a pending proposal."""
        data: dict[str, Any] = {"proposal_id": proposal_id}
        if approved_by:
            data["approved_by"] = approved_by
        return await self._client.request("POST", "/api/v1/nomic/proposals/approve", json=data)

    async def reject_proposal(
        self,
        proposal_id: str,
        rejected_by: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Reject a pending proposal."""
        data: dict[str, Any] = {"proposal_id": proposal_id}
        if rejected_by:
            data["rejected_by"] = rejected_by
        if reason:
            data["reason"] = reason
        return await self._client.request("POST", "/api/v1/nomic/proposals/reject", json=data)

    # ===========================================================================
    # Operational Modes
    # ===========================================================================

    async def get_modes(self) -> dict[str, Any]:
        """Get available operational modes."""
        return await self._client.request("GET", "/api/v1/modes")

    # ===========================================================================
    # Gas Town (Witness & Mayor)
    # ===========================================================================

    async def get_witness_status(self) -> dict[str, Any]:
        """Get Gas Town witness patrol status."""
        return await self._client.request("GET", "/api/v1/nomic/witness/status")

    async def get_mayor_current(self) -> dict[str, Any]:
        """Get current Gas Town mayor information."""
        return await self._client.request("GET", "/api/v1/nomic/mayor/current")

    # ===========================================================================
    # Convenience Methods
    # ===========================================================================

    async def state(self) -> dict[str, Any]:
        """Alias for get_state()."""
        return await self.get_state()

    async def health(self) -> dict[str, Any]:
        """Alias for get_health()."""
        return await self.get_health()

    async def is_running(self) -> bool:
        """Check if the nomic loop is currently running."""
        state = await self.get_state()
        return state.get("running", False) and not state.get("paused", False)

    async def get_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        """Get a specific proposal by ID."""
        response = await self.get_proposals()
        for proposal in response.get("proposals", []):
            if proposal.get("id") == proposal_id:
                return proposal
        return None
