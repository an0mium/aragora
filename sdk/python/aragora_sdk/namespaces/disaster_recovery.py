"""
Disaster Recovery Namespace API

Provides methods for disaster recovery operations:
- Checking DR readiness status
- Running DR drills (tabletop, simulation, full)
- Getting and monitoring RPO/RTO objectives
- Validating DR configuration
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class DisasterRecoveryAPI:
    """
    Synchronous Disaster Recovery API.

    Provides methods for enterprise disaster recovery operations:
    - Checking DR readiness status
    - Running DR drills (tabletop, simulation, full)
    - Getting and monitoring RPO/RTO objectives
    - Validating DR configuration

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> status = client.dr.get_status()
        >>> print(f"DR ready: {status['ready']}, Health: {status['overall_health']}")
        >>>
        >>> # Run a tabletop drill
        >>> drill = client.dr.run_drill(type="tabletop", notify_team=True)
        >>>
        >>> # Check recovery objectives
        >>> objectives = client.dr.get_objectives()
        >>> print(f"RPO: {objectives['rpo_minutes']}min, RTO: {objectives['rto_minutes']}min")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # DR Status and Objectives
    # ===========================================================================

    def is_ready(self) -> bool:
        """
        Check if DR is ready for production.

        Convenience method that checks status and validates configuration.

        Returns:
            True if DR is ready and configuration is valid, False otherwise
        """
        status = self.get_status()
        validation = self.validate()
        return bool(status.get("ready")) and bool(validation.get("valid"))

    def get_health_summary(self) -> dict[str, Any]:
        """
        Get a summary of DR health.

        Convenience method that aggregates status and objectives.

        Returns:
            Dict with health summary including:
            - ready: Whether DR is ready
            - health: Overall health status
            - rpo_compliant: Whether RPO target is met
            - rto_compliant: Whether RTO target is met
            - issues_count: Number of outstanding issues
        """
        status = self.get_status()
        objectives = self.get_objectives()
        return {
            "ready": status.get("ready", False),
            "health": status.get("overall_health", "unknown"),
            "rpo_compliant": objectives.get("rpo_compliant", False),
            "rto_compliant": objectives.get("rto_compliant", False),
            "issues_count": len(status.get("issues", [])),
        }

class AsyncDisasterRecoveryAPI:
    """
    Asynchronous Disaster Recovery API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     status = await client.dr.get_status()
        ...     print(f"DR ready: {status['ready']}, Health: {status['overall_health']}")
        ...
        ...     # Run a tabletop drill
        ...     drill = await client.dr.run_drill(type="tabletop", notify_team=True)
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # DR Status and Objectives
    # ===========================================================================

    async def is_ready(self) -> bool:
        """
        Check if DR is ready for production.

        Convenience method that checks status and validates configuration.

        Returns:
            True if DR is ready and configuration is valid, False otherwise
        """
        status = await self.get_status()
        validation = await self.validate()
        return bool(status.get("ready")) and bool(validation.get("valid"))

    async def get_health_summary(self) -> dict[str, Any]:
        """
        Get a summary of DR health.

        Convenience method that aggregates status and objectives.

        Returns:
            Dict with health summary including ready, health, rpo_compliant,
            rto_compliant, and issues_count.
        """
        status = await self.get_status()
        objectives = await self.get_objectives()
        return {
            "ready": status.get("ready", False),
            "health": status.get("overall_health", "unknown"),
            "rpo_compliant": objectives.get("rpo_compliant", False),
            "rto_compliant": objectives.get("rto_compliant", False),
            "issues_count": len(status.get("issues", [])),
        }
