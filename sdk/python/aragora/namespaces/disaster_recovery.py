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

    def get_status(self) -> dict[str, Any]:
        """
        Get current DR readiness status.

        Returns:
            Dict with DR status including:
            - ready: Whether DR is ready
            - last_backup_at: Timestamp of last backup
            - last_drill_at: Timestamp of last drill
            - rpo_met: Whether RPO is met
            - rto_met: Whether RTO is met
            - issues: List of DR issues
            - overall_health: 'healthy', 'degraded', or 'critical'
        """
        return self._client.request("GET", "/api/v2/dr/status")

    def get_objectives(self) -> dict[str, Any]:
        """
        Get current RPO/RTO objectives and compliance status.

        Returns:
            Dict with objectives including:
            - rpo_minutes: Target RPO in minutes
            - rto_minutes: Target RTO in minutes
            - current_rpo_minutes: Current RPO achievement
            - current_rto_minutes: Current RTO achievement
            - rpo_compliant: Whether RPO target is met
            - rto_compliant: Whether RTO target is met
            - last_measured_at: Timestamp of last measurement
        """
        return self._client.request("GET", "/api/v2/dr/objectives")

    # ===========================================================================
    # DR Drills
    # ===========================================================================

    def run_drill(
        self,
        type: str,
        components: list[str] | None = None,
        notify_team: bool | None = None,
        dry_run: bool | None = None,
    ) -> dict[str, Any]:
        """
        Run a DR drill.

        Args:
            type: Drill type ('tabletop', 'simulation', or 'full')
            components: Optional list of components to include in drill
            notify_team: Whether to notify the team about the drill
            dry_run: Whether to perform a dry run without actual changes

        Returns:
            Dict with drill results including:
            - drill_id: Unique drill identifier
            - type: Drill type
            - started_at: Start timestamp
            - completed_at: Completion timestamp
            - duration_seconds: Total duration
            - success: Whether drill succeeded
            - recovery_time_seconds: Measured recovery time
            - data_loss_seconds: Measured data loss
            - steps: List of drill steps with status
            - recommendations: List of improvement recommendations
        """
        data: dict[str, Any] = {"type": type}
        if components is not None:
            data["components"] = components
        if notify_team is not None:
            data["notify_team"] = notify_team
        if dry_run is not None:
            data["dry_run"] = dry_run
        return self._client.request("POST", "/api/v2/dr/drill", json=data)

    # ===========================================================================
    # DR Validation
    # ===========================================================================

    def validate(
        self,
        check_backups: bool | None = None,
        check_replication: bool | None = None,
        check_failover: bool | None = None,
        check_dns: bool | None = None,
    ) -> dict[str, Any]:
        """
        Validate DR configuration.

        Args:
            check_backups: Whether to check backup configuration
            check_replication: Whether to check replication status
            check_failover: Whether to check failover configuration
            check_dns: Whether to check DNS configuration

        Returns:
            Dict with validation results including:
            - valid: Whether configuration is valid
            - checks: List of validation checks with status
            - overall_score: Numeric score (0-100)
            - last_validated_at: Timestamp of validation
        """
        data: dict[str, Any] = {}
        if check_backups is not None:
            data["check_backups"] = check_backups
        if check_replication is not None:
            data["check_replication"] = check_replication
        if check_failover is not None:
            data["check_failover"] = check_failover
        if check_dns is not None:
            data["check_dns"] = check_dns
        return self._client.request("POST", "/api/v2/dr/validate", json=data)

    # ===========================================================================
    # Convenience Methods
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

    async def get_status(self) -> dict[str, Any]:
        """
        Get current DR readiness status.

        Returns:
            Dict with DR status including ready, last_backup_at, last_drill_at,
            rpo_met, rto_met, issues, and overall_health.
        """
        return await self._client.request("GET", "/api/v2/dr/status")

    async def get_objectives(self) -> dict[str, Any]:
        """
        Get current RPO/RTO objectives and compliance status.

        Returns:
            Dict with objectives including rpo_minutes, rto_minutes,
            current_rpo_minutes, current_rto_minutes, rpo_compliant,
            rto_compliant, and last_measured_at.
        """
        return await self._client.request("GET", "/api/v2/dr/objectives")

    # ===========================================================================
    # DR Drills
    # ===========================================================================

    async def run_drill(
        self,
        type: str,
        components: list[str] | None = None,
        notify_team: bool | None = None,
        dry_run: bool | None = None,
    ) -> dict[str, Any]:
        """
        Run a DR drill.

        Args:
            type: Drill type ('tabletop', 'simulation', or 'full')
            components: Optional list of components to include in drill
            notify_team: Whether to notify the team about the drill
            dry_run: Whether to perform a dry run without actual changes

        Returns:
            Dict with drill results including drill_id, success, recovery_time_seconds,
            data_loss_seconds, steps, and recommendations.
        """
        data: dict[str, Any] = {"type": type}
        if components is not None:
            data["components"] = components
        if notify_team is not None:
            data["notify_team"] = notify_team
        if dry_run is not None:
            data["dry_run"] = dry_run
        return await self._client.request("POST", "/api/v2/dr/drill", json=data)

    # ===========================================================================
    # DR Validation
    # ===========================================================================

    async def validate(
        self,
        check_backups: bool | None = None,
        check_replication: bool | None = None,
        check_failover: bool | None = None,
        check_dns: bool | None = None,
    ) -> dict[str, Any]:
        """
        Validate DR configuration.

        Args:
            check_backups: Whether to check backup configuration
            check_replication: Whether to check replication status
            check_failover: Whether to check failover configuration
            check_dns: Whether to check DNS configuration

        Returns:
            Dict with validation results including valid, checks,
            overall_score, and last_validated_at.
        """
        data: dict[str, Any] = {}
        if check_backups is not None:
            data["check_backups"] = check_backups
        if check_replication is not None:
            data["check_replication"] = check_replication
        if check_failover is not None:
            data["check_failover"] = check_failover
        if check_dns is not None:
            data["check_dns"] = check_dns
        return await self._client.request("POST", "/api/v2/dr/validate", json=data)

    # ===========================================================================
    # Convenience Methods
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
