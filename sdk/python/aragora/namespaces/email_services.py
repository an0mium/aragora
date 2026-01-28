"""
Email Services Namespace API

Provides methods for advanced email management services including
follow-up tracking, snooze management, and email categorization.

Features:
- Follow-up tracking (mark, list, resolve, check replies)
- Snooze recommendations and management
- Email categorization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class EmailServicesAPI:
    """
    Synchronous Email Services API.

    Provides methods for advanced email management:
    - Follow-up tracking
    - Snooze management
    - Email categorization

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="...")
        >>> pending = client.email_services.list_pending_followups()
        >>> client.email_services.mark_followup(email_id="abc123")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Follow-up Tracking
    # ===========================================================================

    def mark_followup(
        self,
        email_id: str,
        expected_reply_by: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """
        Mark an email as awaiting reply.

        Args:
            email_id: ID of the email to track
            expected_reply_by: Optional ISO datetime for expected reply
            notes: Optional notes about the follow-up

        Returns:
            Dict with followup_id and status
        """
        data: dict[str, Any] = {"email_id": email_id}
        if expected_reply_by:
            data["expected_reply_by"] = expected_reply_by
        if notes:
            data["notes"] = notes

        return self._client.request("POST", "/api/v1/email/followups/mark", json=data)

    def list_pending_followups(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List pending follow-ups.

        Args:
            limit: Maximum number of results (default: 50)
            offset: Pagination offset

        Returns:
            Dict with followups array and pagination info
        """
        return self._client.request(
            "GET",
            "/api/v1/email/followups/pending",
            params={"limit": limit, "offset": offset},
        )

    def resolve_followup(
        self,
        followup_id: str,
        resolution: str | None = None,
    ) -> dict[str, Any]:
        """
        Resolve a follow-up.

        Args:
            followup_id: ID of the follow-up to resolve
            resolution: Optional resolution notes

        Returns:
            Dict with success status
        """
        data = {}
        if resolution:
            data["resolution"] = resolution

        return self._client.request(
            "POST",
            f"/api/v1/email/followups/{followup_id}/resolve",
            json=data,
        )

    def check_replies(self) -> dict[str, Any]:
        """
        Check for replies to tracked emails.

        Returns:
            Dict with replied_count and updated followups
        """
        return self._client.request("POST", "/api/v1/email/followups/check-replies")

    # ===========================================================================
    # Snooze Management
    # ===========================================================================

    def get_snooze_suggestions(self, email_id: str) -> dict[str, Any]:
        """
        Get snooze recommendations for an email.

        Args:
            email_id: ID of the email

        Returns:
            Dict with suggested snooze times and reasons
        """
        return self._client.request("GET", f"/api/v1/email/{email_id}/snooze-suggestions")

    def snooze_email(
        self,
        email_id: str,
        snooze_until: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """
        Apply snooze to an email.

        Args:
            email_id: ID of the email to snooze
            snooze_until: ISO datetime when snooze ends
            reason: Optional reason for snoozing

        Returns:
            Dict with snooze_id and scheduled_unsnooze
        """
        data: dict[str, Any] = {"snooze_until": snooze_until}
        if reason:
            data["reason"] = reason

        return self._client.request("POST", f"/api/v1/email/{email_id}/snooze", json=data)

    def cancel_snooze(self, email_id: str) -> dict[str, Any]:
        """
        Cancel a snooze on an email.

        Args:
            email_id: ID of the snoozed email

        Returns:
            Dict with success status
        """
        return self._client.request("DELETE", f"/api/v1/email/{email_id}/snooze")

    def list_snoozed(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List snoozed emails.

        Args:
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            Dict with snoozed emails array
        """
        return self._client.request(
            "GET",
            "/api/v1/email/snoozed",
            params={"limit": limit, "offset": offset},
        )

    # ===========================================================================
    # Email Categorization
    # ===========================================================================

    def list_categories(self) -> dict[str, Any]:
        """
        List available email categories.

        Returns:
            Dict with categories array
        """
        return self._client.request("GET", "/api/v1/email/categories")

    def submit_category_feedback(
        self,
        email_id: str,
        suggested_category: str,
        is_correct: bool,
        correct_category: str | None = None,
    ) -> dict[str, Any]:
        """
        Submit category feedback to improve categorization.

        Args:
            email_id: ID of the email
            suggested_category: The category that was suggested
            is_correct: Whether the suggestion was correct
            correct_category: The correct category if suggestion was wrong

        Returns:
            Dict with success status
        """
        data: dict[str, Any] = {
            "email_id": email_id,
            "suggested_category": suggested_category,
            "is_correct": is_correct,
        }
        if correct_category:
            data["correct_category"] = correct_category

        return self._client.request("POST", "/api/v1/email/categories/learn", json=data)


class AsyncEmailServicesAPI:
    """
    Asynchronous Email Services API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     pending = await client.email_services.list_pending_followups()
        ...     await client.email_services.mark_followup(email_id="abc123")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Follow-up Tracking
    # ===========================================================================

    async def mark_followup(
        self,
        email_id: str,
        expected_reply_by: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Mark an email as awaiting reply."""
        data: dict[str, Any] = {"email_id": email_id}
        if expected_reply_by:
            data["expected_reply_by"] = expected_reply_by
        if notes:
            data["notes"] = notes

        return await self._client.request("POST", "/api/v1/email/followups/mark", json=data)

    async def list_pending_followups(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List pending follow-ups."""
        return await self._client.request(
            "GET",
            "/api/v1/email/followups/pending",
            params={"limit": limit, "offset": offset},
        )

    async def resolve_followup(
        self,
        followup_id: str,
        resolution: str | None = None,
    ) -> dict[str, Any]:
        """Resolve a follow-up."""
        data = {}
        if resolution:
            data["resolution"] = resolution

        return await self._client.request(
            "POST",
            f"/api/v1/email/followups/{followup_id}/resolve",
            json=data,
        )

    async def check_replies(self) -> dict[str, Any]:
        """Check for replies to tracked emails."""
        return await self._client.request("POST", "/api/v1/email/followups/check-replies")

    # ===========================================================================
    # Snooze Management
    # ===========================================================================

    async def get_snooze_suggestions(self, email_id: str) -> dict[str, Any]:
        """Get snooze recommendations for an email."""
        return await self._client.request("GET", f"/api/v1/email/{email_id}/snooze-suggestions")

    async def snooze_email(
        self,
        email_id: str,
        snooze_until: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Apply snooze to an email."""
        data: dict[str, Any] = {"snooze_until": snooze_until}
        if reason:
            data["reason"] = reason

        return await self._client.request("POST", f"/api/v1/email/{email_id}/snooze", json=data)

    async def cancel_snooze(self, email_id: str) -> dict[str, Any]:
        """Cancel a snooze on an email."""
        return await self._client.request("DELETE", f"/api/v1/email/{email_id}/snooze")

    async def list_snoozed(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List snoozed emails."""
        return await self._client.request(
            "GET",
            "/api/v1/email/snoozed",
            params={"limit": limit, "offset": offset},
        )

    # ===========================================================================
    # Email Categorization
    # ===========================================================================

    async def list_categories(self) -> dict[str, Any]:
        """List available email categories."""
        return await self._client.request("GET", "/api/v1/email/categories")

    async def submit_category_feedback(
        self,
        email_id: str,
        suggested_category: str,
        is_correct: bool,
        correct_category: str | None = None,
    ) -> dict[str, Any]:
        """Submit category feedback to improve categorization."""
        data: dict[str, Any] = {
            "email_id": email_id,
            "suggested_category": suggested_category,
            "is_correct": is_correct,
        }
        if correct_category:
            data["correct_category"] = correct_category

        return await self._client.request("POST", "/api/v1/email/categories/learn", json=data)
