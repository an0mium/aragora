"""
DevOps Namespace API

Provides methods for DevOps integration:
- CI/CD pipelines
- Deployment management
- Infrastructure monitoring
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class DevopsAPI:
    """Synchronous DevOps API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list_pipelines(self, status: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List CI/CD pipelines."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return self._client.request("GET", "/api/v1/devops/pipelines", params=params)

    def get_pipeline(self, pipeline_id: str) -> dict[str, Any]:
        """Get pipeline by ID."""
        return self._client.request("GET", f"/api/v1/devops/pipelines/{pipeline_id}")

    def trigger_pipeline(
        self, pipeline_id: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Trigger a pipeline."""
        data: dict[str, Any] = {}
        if params:
            data["params"] = params
        return self._client.request(
            "POST", f"/api/v1/devops/pipelines/{pipeline_id}/trigger", json=data
        )

    def list_deployments(self, environment: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List deployments."""
        params_dict: dict[str, Any] = {"limit": limit}
        if environment:
            params_dict["environment"] = environment
        return self._client.request("GET", "/api/v1/devops/deployments", params=params_dict)

    def get_deployment(self, deployment_id: str) -> dict[str, Any]:
        """Get deployment by ID."""
        return self._client.request("GET", f"/api/v1/devops/deployments/{deployment_id}")

    def create_deployment(self, environment: str, version: str, service: str) -> dict[str, Any]:
        """Create a deployment."""
        return self._client.request(
            "POST",
            "/api/v1/devops/deployments",
            json={
                "environment": environment,
                "version": version,
                "service": service,
            },
        )

    def rollback(self, deployment_id: str) -> dict[str, Any]:
        """Rollback a deployment."""
        return self._client.request("POST", f"/api/v1/devops/deployments/{deployment_id}/rollback")

    def get_infrastructure_status(self) -> dict[str, Any]:
        """Get infrastructure status."""
        return self._client.request("GET", "/api/v1/devops/infrastructure/status")

    def get_alerts(self, severity: str | None = None) -> dict[str, Any]:
        """Get DevOps alerts."""
        params_dict: dict[str, Any] = {}
        if severity:
            params_dict["severity"] = severity
        return self._client.request("GET", "/api/v1/devops/alerts", params=params_dict)


class AsyncDevopsAPI:
    """Asynchronous DevOps API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_pipelines(self, status: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List CI/CD pipelines."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return await self._client.request("GET", "/api/v1/devops/pipelines", params=params)

    async def get_pipeline(self, pipeline_id: str) -> dict[str, Any]:
        """Get pipeline by ID."""
        return await self._client.request("GET", f"/api/v1/devops/pipelines/{pipeline_id}")

    async def trigger_pipeline(
        self, pipeline_id: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Trigger a pipeline."""
        data: dict[str, Any] = {}
        if params:
            data["params"] = params
        return await self._client.request(
            "POST", f"/api/v1/devops/pipelines/{pipeline_id}/trigger", json=data
        )

    async def list_deployments(
        self, environment: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        """List deployments."""
        params_dict: dict[str, Any] = {"limit": limit}
        if environment:
            params_dict["environment"] = environment
        return await self._client.request("GET", "/api/v1/devops/deployments", params=params_dict)

    async def get_deployment(self, deployment_id: str) -> dict[str, Any]:
        """Get deployment by ID."""
        return await self._client.request("GET", f"/api/v1/devops/deployments/{deployment_id}")

    async def create_deployment(
        self, environment: str, version: str, service: str
    ) -> dict[str, Any]:
        """Create a deployment."""
        return await self._client.request(
            "POST",
            "/api/v1/devops/deployments",
            json={
                "environment": environment,
                "version": version,
                "service": service,
            },
        )

    async def rollback(self, deployment_id: str) -> dict[str, Any]:
        """Rollback a deployment."""
        return await self._client.request(
            "POST", f"/api/v1/devops/deployments/{deployment_id}/rollback"
        )

    async def get_infrastructure_status(self) -> dict[str, Any]:
        """Get infrastructure status."""
        return await self._client.request("GET", "/api/v1/devops/infrastructure/status")

    async def get_alerts(self, severity: str | None = None) -> dict[str, Any]:
        """Get DevOps alerts."""
        params_dict: dict[str, Any] = {}
        if severity:
            params_dict["severity"] = severity
        return await self._client.request("GET", "/api/v1/devops/alerts", params=params_dict)
