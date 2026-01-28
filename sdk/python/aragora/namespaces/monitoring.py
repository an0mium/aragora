"""
Monitoring namespace for observability operations.

Provides API access to metrics, alerts, dashboards, and
system health monitoring.
"""

from __future__ import annotations

from typing import Any


class MonitoringAPI:
    """Synchronous monitoring API."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def get_metrics(
        self,
        metric_names: list[str] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        step: str = "1m",
    ) -> dict[str, Any]:
        """
        Get system metrics.

        Args:
            metric_names: Specific metrics to retrieve
            start_time: Start time (ISO format)
            end_time: End time (ISO format)
            step: Query resolution

        Returns:
            Metrics data
        """
        params: dict[str, Any] = {"step": step}
        if metric_names:
            params["metrics"] = ",".join(metric_names)
        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        return self._client._request("GET", "/api/v1/monitoring/metrics", params=params)

    def get_health(self) -> dict[str, Any]:
        """
        Get system health status.

        Returns:
            Health status for all components
        """
        return self._client._request("GET", "/api/v1/monitoring/health")

    def list_alerts(
        self,
        status: str | None = None,
        severity: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        List active alerts.

        Args:
            status: Filter by status (firing, resolved)
            severity: Filter by severity (critical, warning, info)
            limit: Maximum alerts to return

        Returns:
            List of alerts
        """
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if severity:
            params["severity"] = severity

        return self._client._request("GET", "/api/v1/monitoring/alerts", params=params)

    def acknowledge_alert(self, alert_id: str, comment: str | None = None) -> dict[str, Any]:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert identifier
            comment: Acknowledgment comment

        Returns:
            Updated alert
        """
        data: dict[str, Any] = {}
        if comment:
            data["comment"] = comment

        return self._client._request(
            "POST", f"/api/v1/monitoring/alerts/{alert_id}/acknowledge", json=data
        )

    def resolve_alert(self, alert_id: str, resolution: str | None = None) -> dict[str, Any]:
        """
        Resolve an alert.

        Args:
            alert_id: Alert identifier
            resolution: Resolution notes

        Returns:
            Updated alert
        """
        data: dict[str, Any] = {}
        if resolution:
            data["resolution"] = resolution

        return self._client._request(
            "POST", f"/api/v1/monitoring/alerts/{alert_id}/resolve", json=data
        )

    def list_dashboards(self) -> list[dict[str, Any]]:
        """
        List available dashboards.

        Returns:
            List of dashboards
        """
        return self._client._request("GET", "/api/v1/monitoring/dashboards")

    def get_dashboard(self, dashboard_id: str) -> dict[str, Any]:
        """
        Get dashboard details.

        Args:
            dashboard_id: Dashboard identifier

        Returns:
            Dashboard configuration
        """
        return self._client._request("GET", f"/api/v1/monitoring/dashboards/{dashboard_id}")

    def get_logs(
        self,
        query: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query logs.

        Args:
            query: Log query (LogQL-like syntax)
            start_time: Start time (ISO format)
            end_time: End time (ISO format)
            limit: Maximum log entries to return

        Returns:
            Log entries
        """
        params: dict[str, Any] = {"limit": limit}
        if query:
            params["query"] = query
        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        return self._client._request("GET", "/api/v1/monitoring/logs", params=params)

    def get_traces(
        self,
        service: str | None = None,
        operation: str | None = None,
        min_duration: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Query distributed traces.

        Args:
            service: Filter by service name
            operation: Filter by operation name
            min_duration: Minimum duration (e.g., "100ms")
            limit: Maximum traces to return

        Returns:
            Trace data
        """
        params: dict[str, Any] = {"limit": limit}
        if service:
            params["service"] = service
        if operation:
            params["operation"] = operation
        if min_duration:
            params["min_duration"] = min_duration

        return self._client._request("GET", "/api/v1/monitoring/traces", params=params)

    def get_slos(self) -> list[dict[str, Any]]:
        """
        Get SLO status.

        Returns:
            List of SLOs with current status
        """
        return self._client._request("GET", "/api/v1/monitoring/slos")


class AsyncMonitoringAPI:
    """Asynchronous monitoring API."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def get_metrics(
        self,
        metric_names: list[str] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        step: str = "1m",
    ) -> dict[str, Any]:
        """Get system metrics."""
        params: dict[str, Any] = {"step": step}
        if metric_names:
            params["metrics"] = ",".join(metric_names)
        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        return await self._client._request("GET", "/api/v1/monitoring/metrics", params=params)

    async def get_health(self) -> dict[str, Any]:
        """Get system health status."""
        return await self._client._request("GET", "/api/v1/monitoring/health")

    async def list_alerts(
        self,
        status: str | None = None,
        severity: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List active alerts."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if severity:
            params["severity"] = severity

        return await self._client._request("GET", "/api/v1/monitoring/alerts", params=params)

    async def acknowledge_alert(self, alert_id: str, comment: str | None = None) -> dict[str, Any]:
        """Acknowledge an alert."""
        data: dict[str, Any] = {}
        if comment:
            data["comment"] = comment

        return await self._client._request(
            "POST", f"/api/v1/monitoring/alerts/{alert_id}/acknowledge", json=data
        )

    async def resolve_alert(self, alert_id: str, resolution: str | None = None) -> dict[str, Any]:
        """Resolve an alert."""
        data: dict[str, Any] = {}
        if resolution:
            data["resolution"] = resolution

        return await self._client._request(
            "POST", f"/api/v1/monitoring/alerts/{alert_id}/resolve", json=data
        )

    async def list_dashboards(self) -> list[dict[str, Any]]:
        """List available dashboards."""
        return await self._client._request("GET", "/api/v1/monitoring/dashboards")

    async def get_dashboard(self, dashboard_id: str) -> dict[str, Any]:
        """Get dashboard details."""
        return await self._client._request("GET", f"/api/v1/monitoring/dashboards/{dashboard_id}")

    async def get_logs(
        self,
        query: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query logs."""
        params: dict[str, Any] = {"limit": limit}
        if query:
            params["query"] = query
        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        return await self._client._request("GET", "/api/v1/monitoring/logs", params=params)

    async def get_traces(
        self,
        service: str | None = None,
        operation: str | None = None,
        min_duration: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Query distributed traces."""
        params: dict[str, Any] = {"limit": limit}
        if service:
            params["service"] = service
        if operation:
            params["operation"] = operation
        if min_duration:
            params["min_duration"] = min_duration

        return await self._client._request("GET", "/api/v1/monitoring/traces", params=params)

    async def get_slos(self) -> list[dict[str, Any]]:
        """Get SLO status."""
        return await self._client._request("GET", "/api/v1/monitoring/slos")
