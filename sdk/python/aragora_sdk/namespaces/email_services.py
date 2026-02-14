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

    def get_context(self, email_id: str) -> dict[str, Any]:
        """Get email context."""
        return self._client.request("GET", f"/api/v1/email/context/{email_id}")

    def delete_followup(self, followup_id: str) -> dict[str, Any]:
        """DELETE a followup."""
        return self._client.request("DELETE", f"/api/v1/email/followups/{followup_id}")

    def delete_email(self, email_id: str) -> dict[str, Any]:
        """DELETE an email."""
        return self._client.request("DELETE", f"/api/v1/email/{email_id}")

    def get_followup(self, followup_id: str) -> dict[str, Any]:
        """GET a followup."""
        return self._client.request("GET", f"/api/v1/email/followups/{followup_id}")

    def get_email(self, email_id: str) -> dict[str, Any]:
        """GET an email."""
        return self._client.request("GET", f"/api/v1/email/{email_id}")

    def post_followup(self, followup_id: str) -> dict[str, Any]:
        """POST a followup."""
        return self._client.request("POST", f"/api/v1/email/followups/{followup_id}")

    def post_email(self, email_id: str) -> dict[str, Any]:
        """POST an email."""
        return self._client.request("POST", f"/api/v1/email/{email_id}")


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

    async def check_replies(self) -> dict[str, Any]:
        """Check for replies to tracked emails."""
        return await self._client.request("POST", "/api/v1/email/followups/check-replies")

    # ===========================================================================
    # Snooze Management
    # ===========================================================================

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

    async def get_context(self, email_id: str) -> dict[str, Any]:
        """Get email context."""
        return await self._client.request("GET", f"/api/v1/email/context/{email_id}")

    async def delete_followup(self, followup_id: str) -> dict[str, Any]:
        """DELETE a followup."""
        return await self._client.request("DELETE", f"/api/v1/email/followups/{followup_id}")

    async def delete_email(self, email_id: str) -> dict[str, Any]:
        """DELETE an email."""
        return await self._client.request("DELETE", f"/api/v1/email/{email_id}")

    async def get_followup(self, followup_id: str) -> dict[str, Any]:
        """GET a followup."""
        return await self._client.request("GET", f"/api/v1/email/followups/{followup_id}")

    async def get_email(self, email_id: str) -> dict[str, Any]:
        """GET an email."""
        return await self._client.request("GET", f"/api/v1/email/{email_id}")

    async def post_followup(self, followup_id: str) -> dict[str, Any]:
        """POST a followup."""
        return await self._client.request("POST", f"/api/v1/email/followups/{followup_id}")

    async def post_email(self, email_id: str) -> dict[str, Any]:
        """POST an email."""
        return await self._client.request("POST", f"/api/v1/email/{email_id}")
