"""
Decisions namespace for unified decision management.

Provides a high-level API for submitting decisions and retrieving results
across all decision types (debates, gauntlets, workflows).
"""

from __future__ import annotations

from typing import Any, Literal

DecisionType = Literal["debate", "gauntlet", "workflow", "quick"]
DecisionPriority = Literal["low", "normal", "high", "urgent"]
DecisionStatus = Literal["pending", "processing", "completed", "failed", "cancelled"]


class DecisionsAPI:
    """Synchronous decisions API."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def submit(
        self,
        input: str,
        decision_type: DecisionType = "debate",
        priority: DecisionPriority = "normal",
        context: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        callback_url: str | None = None,
    ) -> dict[str, Any]:
        """
        Submit a decision request.

        This is the unified entry point for all decision types.
        The system will route to the appropriate handler based on type.

        Args:
            input: The decision input (question, claim, task)
            decision_type: Type of decision process
            priority: Processing priority
            context: Optional context for the decision
            config: Type-specific configuration
            callback_url: Webhook URL for completion notification

        Returns:
            Decision submission result with decision_id
        """
        data: dict[str, Any] = {
            "input": input,
            "decision_type": decision_type,
            "priority": priority,
        }
        if context:
            data["context"] = context
        if config:
            data["config"] = config
        if callback_url:
            data["callback_url"] = callback_url

        return self._client._request("POST", "/api/v1/decisions", json=data)

    def get(self, decision_id: str) -> dict[str, Any]:
        """
        Get a decision by ID.

        Args:
            decision_id: Decision identifier

        Returns:
            Decision details including status and result
        """
        return self._client._request("GET", f"/api/v1/decisions/{decision_id}")

    def get_status(self, decision_id: str) -> dict[str, Any]:
        """
        Get the status of a decision.

        Lightweight endpoint for polling decision completion.

        Args:
            decision_id: Decision identifier

        Returns:
            Decision status
        """
        return self._client._request("GET", f"/api/v1/decisions/{decision_id}/status")

    def list(
        self,
        status: DecisionStatus | None = None,
        decision_type: DecisionType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List decisions with optional filtering.

        Args:
            status: Filter by status
            decision_type: Filter by decision type
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of decisions with pagination
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if decision_type:
            params["decision_type"] = decision_type

        return self._client._request("GET", "/api/v1/decisions", params=params)

    def cancel(self, decision_id: str, reason: str | None = None) -> dict[str, Any]:
        """
        Cancel a pending or processing decision.

        Args:
            decision_id: Decision identifier
            reason: Cancellation reason

        Returns:
            Cancellation confirmation
        """
        data: dict[str, Any] = {}
        if reason:
            data["reason"] = reason

        return self._client._request("POST", f"/api/v1/decisions/{decision_id}/cancel", json=data)

    def retry(self, decision_id: str) -> dict[str, Any]:
        """
        Retry a failed decision.

        Args:
            decision_id: Decision identifier

        Returns:
            Retry result with new decision_id
        """
        return self._client._request("POST", f"/api/v1/decisions/{decision_id}/retry")

    def get_receipt(self, decision_id: str) -> dict[str, Any]:
        """
        Get the receipt for a completed decision.

        Args:
            decision_id: Decision identifier

        Returns:
            Decision receipt
        """
        return self._client._request("GET", f"/api/v1/decisions/{decision_id}/receipt")

    def get_explanation(self, decision_id: str) -> dict[str, Any]:
        """
        Get the explanation for a completed decision.

        Args:
            decision_id: Decision identifier

        Returns:
            Decision explanation with factors and reasoning
        """
        return self._client._request("GET", f"/api/v1/decisions/{decision_id}/explain")

    def submit_feedback(
        self,
        decision_id: str,
        rating: int,
        comment: str | None = None,
    ) -> dict[str, Any]:
        """
        Submit feedback on a decision.

        Args:
            decision_id: Decision identifier
            rating: Rating (1-5)
            comment: Optional comment

        Returns:
            Feedback submission confirmation
        """
        data: dict[str, Any] = {"rating": rating}
        if comment:
            data["comment"] = comment

        return self._client._request("POST", f"/api/v1/decisions/{decision_id}/feedback", json=data)


class AsyncDecisionsAPI:
    """Asynchronous decisions API."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def submit(
        self,
        input: str,
        decision_type: DecisionType = "debate",
        priority: DecisionPriority = "normal",
        context: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        callback_url: str | None = None,
    ) -> dict[str, Any]:
        """Submit a decision request."""
        data: dict[str, Any] = {
            "input": input,
            "decision_type": decision_type,
            "priority": priority,
        }
        if context:
            data["context"] = context
        if config:
            data["config"] = config
        if callback_url:
            data["callback_url"] = callback_url

        return await self._client._request("POST", "/api/v1/decisions", json=data)

    async def get(self, decision_id: str) -> dict[str, Any]:
        """Get a decision by ID."""
        return await self._client._request("GET", f"/api/v1/decisions/{decision_id}")

    async def get_status(self, decision_id: str) -> dict[str, Any]:
        """Get the status of a decision."""
        return await self._client._request("GET", f"/api/v1/decisions/{decision_id}/status")

    async def list(
        self,
        status: DecisionStatus | None = None,
        decision_type: DecisionType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List decisions with optional filtering."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if decision_type:
            params["decision_type"] = decision_type

        return await self._client._request("GET", "/api/v1/decisions", params=params)

    async def cancel(self, decision_id: str, reason: str | None = None) -> dict[str, Any]:
        """Cancel a pending or processing decision."""
        data: dict[str, Any] = {}
        if reason:
            data["reason"] = reason

        return await self._client._request(
            "POST", f"/api/v1/decisions/{decision_id}/cancel", json=data
        )

    async def retry(self, decision_id: str) -> dict[str, Any]:
        """Retry a failed decision."""
        return await self._client._request("POST", f"/api/v1/decisions/{decision_id}/retry")

    async def get_receipt(self, decision_id: str) -> dict[str, Any]:
        """Get the receipt for a completed decision."""
        return await self._client._request("GET", f"/api/v1/decisions/{decision_id}/receipt")

    async def get_explanation(self, decision_id: str) -> dict[str, Any]:
        """Get the explanation for a completed decision."""
        return await self._client._request("GET", f"/api/v1/decisions/{decision_id}/explain")

    async def submit_feedback(
        self,
        decision_id: str,
        rating: int,
        comment: str | None = None,
    ) -> dict[str, Any]:
        """Submit feedback on a decision."""
        data: dict[str, Any] = {"rating": rating}
        if comment:
            data["comment"] = comment

        return await self._client._request(
            "POST", f"/api/v1/decisions/{decision_id}/feedback", json=data
        )
