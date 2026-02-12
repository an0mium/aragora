"""NomicAPI resource for the Aragora client."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ..client import AragoraClient


class NomicAPI:
    """API interface for the Nomic self-improvement loop."""

    def __init__(self, client: AragoraClient):
        self._client = client

    # ------------------------------------------------------------------
    # Read-only endpoints
    # ------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """Get the current nomic loop state."""
        return cast(dict[str, Any], self._client._get("/api/v1/nomic/state"))

    async def state_async(self) -> dict[str, Any]:
        """Async version of state()."""
        return cast(dict[str, Any], await self._client._get_async("/api/v1/nomic/state"))

    def health(self) -> dict[str, Any]:
        """Get nomic loop health with stall detection."""
        return cast(dict[str, Any], self._client._get("/api/v1/nomic/health"))

    async def health_async(self) -> dict[str, Any]:
        """Async version of health()."""
        return cast(dict[str, Any], await self._client._get_async("/api/v1/nomic/health"))

    def metrics(self) -> dict[str, Any]:
        """Get nomic loop metrics summary."""
        return cast(dict[str, Any], self._client._get("/api/v1/nomic/metrics"))

    async def metrics_async(self) -> dict[str, Any]:
        """Async version of metrics()."""
        return cast(dict[str, Any], await self._client._get_async("/api/v1/nomic/metrics"))

    def log(self, lines: int = 100) -> dict[str, Any]:
        """Get recent nomic loop log lines."""
        return cast(
            dict[str, Any],
            self._client._get("/api/v1/nomic/log", params={"lines": lines}),
        )

    async def log_async(self, lines: int = 100) -> dict[str, Any]:
        """Async version of log()."""
        return cast(
            dict[str, Any],
            await self._client._get_async("/api/v1/nomic/log", params={"lines": lines}),
        )

    def risk_register(self, limit: int = 50) -> dict[str, Any]:
        """Get nomic loop risk register entries."""
        return cast(
            dict[str, Any],
            self._client._get("/api/v1/nomic/risk-register", params={"limit": limit}),
        )

    async def risk_register_async(self, limit: int = 50) -> dict[str, Any]:
        """Async version of risk_register()."""
        return cast(
            dict[str, Any],
            await self._client._get_async("/api/v1/nomic/risk-register", params={"limit": limit}),
        )

    def witness_status(self) -> dict[str, Any]:
        """Get the current witness patrol status."""
        return cast(dict[str, Any], self._client._get("/api/v1/nomic/witness/status"))

    async def witness_status_async(self) -> dict[str, Any]:
        """Async version of witness_status()."""
        return cast(
            dict[str, Any],
            await self._client._get_async("/api/v1/nomic/witness/status"),
        )

    def mayor_current(self) -> dict[str, Any]:
        """Get current mayor information."""
        return cast(dict[str, Any], self._client._get("/api/v1/nomic/mayor/current"))

    async def mayor_current_async(self) -> dict[str, Any]:
        """Async version of mayor_current()."""
        return cast(
            dict[str, Any],
            await self._client._get_async("/api/v1/nomic/mayor/current"),
        )

    def proposals(self) -> dict[str, Any]:
        """Get pending nomic proposals."""
        return cast(dict[str, Any], self._client._get("/api/v1/nomic/proposals"))

    async def proposals_async(self) -> dict[str, Any]:
        """Async version of proposals()."""
        return cast(
            dict[str, Any],
            await self._client._get_async("/api/v1/nomic/proposals"),
        )

    def modes(self) -> dict[str, Any]:
        """Get available operational modes."""
        return cast(dict[str, Any], self._client._get("/api/v1/modes"))

    async def modes_async(self) -> dict[str, Any]:
        """Async version of modes()."""
        return cast(dict[str, Any], await self._client._get_async("/api/v1/modes"))

    # ------------------------------------------------------------------
    # Control endpoints
    # ------------------------------------------------------------------

    def start(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Start the nomic loop with optional configuration."""
        return cast(
            dict[str, Any],
            self._client._post("/api/v1/nomic/control/start", data=config or {}),
        )

    async def start_async(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Async version of start()."""
        return cast(
            dict[str, Any],
            await self._client._post_async("/api/v1/nomic/control/start", data=config or {}),
        )

    def stop(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Stop the nomic loop."""
        return cast(
            dict[str, Any],
            self._client._post("/api/v1/nomic/control/stop", data=payload or {}),
        )

    async def stop_async(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Async version of stop()."""
        return cast(
            dict[str, Any],
            await self._client._post_async("/api/v1/nomic/control/stop", data=payload or {}),
        )

    def pause(self) -> dict[str, Any]:
        """Pause the nomic loop."""
        return cast(dict[str, Any], self._client._post("/api/v1/nomic/control/pause", {}))

    async def pause_async(self) -> dict[str, Any]:
        """Async version of pause()."""
        return cast(
            dict[str, Any],
            await self._client._post_async("/api/v1/nomic/control/pause", {}),
        )

    def resume(self) -> dict[str, Any]:
        """Resume the nomic loop."""
        return cast(dict[str, Any], self._client._post("/api/v1/nomic/control/resume", {}))

    async def resume_async(self) -> dict[str, Any]:
        """Async version of resume()."""
        return cast(
            dict[str, Any],
            await self._client._post_async("/api/v1/nomic/control/resume", {}),
        )

    def skip_phase(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Skip the current nomic phase."""
        return cast(
            dict[str, Any],
            self._client._post("/api/v1/nomic/control/skip-phase", data=payload or {}),
        )

    async def skip_phase_async(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Async version of skip_phase()."""
        return cast(
            dict[str, Any],
            await self._client._post_async("/api/v1/nomic/control/skip-phase", data=payload or {}),
        )

    def approve_proposal(self, proposal_id: str | None = None) -> dict[str, Any]:
        """Approve a nomic proposal."""
        payload: dict[str, Any] = {}
        if proposal_id:
            payload["proposal_id"] = proposal_id
        return cast(
            dict[str, Any],
            self._client._post("/api/v1/nomic/proposals/approve", data=payload),
        )

    async def approve_proposal_async(self, proposal_id: str | None = None) -> dict[str, Any]:
        """Async version of approve_proposal()."""
        payload: dict[str, Any] = {}
        if proposal_id:
            payload["proposal_id"] = proposal_id
        return cast(
            dict[str, Any],
            await self._client._post_async("/api/v1/nomic/proposals/approve", data=payload),
        )

    def reject_proposal(
        self, proposal_id: str | None = None, reason: str | None = None
    ) -> dict[str, Any]:
        """Reject a nomic proposal."""
        payload: dict[str, Any] = {}
        if proposal_id:
            payload["proposal_id"] = proposal_id
        if reason:
            payload["reason"] = reason
        return cast(
            dict[str, Any],
            self._client._post("/api/v1/nomic/proposals/reject", data=payload),
        )

    async def reject_proposal_async(
        self, proposal_id: str | None = None, reason: str | None = None
    ) -> dict[str, Any]:
        """Async version of reject_proposal()."""
        payload: dict[str, Any] = {}
        if proposal_id:
            payload["proposal_id"] = proposal_id
        if reason:
            payload["reason"] = reason
        return cast(
            dict[str, Any],
            await self._client._post_async("/api/v1/nomic/proposals/reject", data=payload),
        )
