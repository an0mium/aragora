"""
Email Debate Namespace API

Provides methods for multi-agent email prioritization and triage:
- Single email prioritization
- Batch email prioritization
- Full inbox triage with categorization
- Prioritization history retrieval
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class EmailDebateAPI:
    """
    Synchronous Email Debate API.

    Provides AI-powered email prioritization using multi-agent debate:
    - Prioritize single emails with reasoning and suggested actions
    - Batch prioritize multiple emails efficiently
    - Full inbox triage with categorization and auto-reply drafts
    - Retrieve prioritization history

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> result = client.email_debate.prioritize(
        ...     email={
        ...         "subject": "Q4 Budget Review - Action Required",
        ...         "body": "Please review the attached budget proposal...",
        ...         "sender": "cfo@company.com",
        ...     },
        ...     user_id="user_123",
        ... )
        >>> print(f"Priority: {result['priority']}, Score: {result['score']}")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Email Prioritization
    # ===========================================================================

    def prioritize(
        self,
        email: dict[str, Any],
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Prioritize a single email using multi-agent debate.

        Agents analyze the email content, sender reputation, and context
        to determine priority and suggest actions.

        Args:
            email: Email to prioritize. Should include:
                - subject: Email subject line (optional)
                - body: Email body content (optional)
                - sender: Sender email address (required)
                - received_at: ISO timestamp when received (optional)
                - message_id: Unique message identifier (optional)
                - user_id: User ID for the email owner (optional)
                - thread_id: Thread/conversation ID (optional)
                - labels: List of labels/tags (optional)
                - attachments: List of attachment dicts with filename,
                  mime_type, size_bytes (optional)
            user_id: User ID for personalized prioritization

        Returns:
            Dict with prioritization result including:
                - priority: Priority level (critical/high/medium/low/none)
                - score: Numeric priority score
                - confidence: Confidence in the prioritization
                - reasoning: Explanation for the priority assignment
                - suggested_actions: List of recommended actions
                - category: Email category (optional)
                - response_urgency: When to respond (optional)
                - agents_consulted: List of agents that participated
                - consensus_reached: Whether agents reached consensus
                - debate_id: ID of the debate session (optional)
        """
        data: dict[str, Any] = {**email}
        if user_id is not None:
            data["user_id"] = user_id
        return self._client.request("POST", "/api/v1/email/prioritize", json=data)

    def prioritize_batch(
        self,
        emails: list[dict[str, Any]],
        user_id: str | None = None,
        parallel: bool = True,
    ) -> dict[str, Any]:
        """
        Prioritize multiple emails in batch.

        More efficient than calling prioritize() multiple times.

        Args:
            emails: List of email dicts to prioritize. Each email should
                include subject, body, sender, and optionally received_at,
                message_id, user_id, thread_id, labels, and attachments.
            user_id: User ID for personalized prioritization
            parallel: Process emails in parallel (default: True)

        Returns:
            Dict with batch prioritization results including:
                - results: List of prioritization results for each email
                - total_processed: Number of emails processed
                - processing_time_ms: Total processing time in milliseconds
                - summary: Counts by priority level (critical/high/medium/low/none)
        """
        data: dict[str, Any] = {
            "emails": emails,
            "parallel": parallel,
        }
        if user_id is not None:
            data["user_id"] = user_id
        return self._client.request("POST", "/api/v1/email/prioritize/batch", json=data)

    # ===========================================================================
    # Inbox Triage
    # ===========================================================================

    def triage_inbox(
        self,
        emails: list[dict[str, Any]],
        user_id: str | None = None,
        include_auto_replies: bool = False,
    ) -> dict[str, Any]:
        """
        Triage inbox with full categorization.

        Provides comprehensive email triage including:
        - Category assignment (action_required, fyi, meeting, follow_up,
          newsletter, spam, personal, finance, legal, other)
        - Priority scoring
        - Folder suggestions
        - Auto-reply drafts (if enabled)
        - Delegation recommendations

        Args:
            emails: List of email dicts to triage. Each email should include
                subject, body, sender, and optionally received_at, message_id,
                user_id, thread_id, labels, and attachments.
            user_id: User ID for personalized triage
            include_auto_replies: Generate auto-reply drafts (default: False)

        Returns:
            Dict with triage results including:
                - results: List of triage results for each email with category,
                  priority, suggested_folder, suggested_labels, auto_reply_suggested,
                  auto_reply_draft, delegate_to, snooze_until, and reasoning
                - total_triaged: Number of emails triaged
                - processing_time_ms: Total processing time in milliseconds
                - summary: Aggregated stats by category and priority
        """
        data: dict[str, Any] = {
            "emails": emails,
            "include_auto_replies": include_auto_replies,
        }
        if user_id is not None:
            data["user_id"] = user_id
        return self._client.request("POST", "/api/v1/email/triage", json=data)

    # ===========================================================================
    # History
    # ===========================================================================

    def get_history(
        self,
        user_id: str,
        limit: int | None = None,
        since: str | None = None,
    ) -> dict[str, Any]:
        """
        Get prioritization history for a user.

        Args:
            user_id: User ID to get history for
            limit: Maximum number of results (default: 50)
            since: Filter to emails since this ISO timestamp

        Returns:
            Dict with list of past prioritization results
        """
        params: dict[str, Any] = {"user_id": user_id}
        if limit is not None:
            params["limit"] = limit
        if since is not None:
            params["since"] = since
        return self._client.request("GET", "/api/v1/email/prioritize/history", params=params)


class AsyncEmailDebateAPI:
    """
    Asynchronous Email Debate API.

    Provides AI-powered email prioritization using multi-agent debate.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     result = await client.email_debate.prioritize(
        ...         email={
        ...             "subject": "Urgent: Server Down",
        ...             "body": "Production server is unresponsive...",
        ...             "sender": "alerts@monitoring.com",
        ...         },
        ...         user_id="user_123",
        ...     )
        ...     print(f"Priority: {result['priority']}")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Email Prioritization
    # ===========================================================================

    async def prioritize(
        self,
        email: dict[str, Any],
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Prioritize a single email using multi-agent debate.

        Args:
            email: Email to prioritize with subject, body, sender, etc.
            user_id: User ID for personalized prioritization

        Returns:
            Dict with prioritization result including priority, score,
            confidence, reasoning, and suggested actions
        """
        data: dict[str, Any] = {**email}
        if user_id is not None:
            data["user_id"] = user_id
        return await self._client.request("POST", "/api/v1/email/prioritize", json=data)

    async def prioritize_batch(
        self,
        emails: list[dict[str, Any]],
        user_id: str | None = None,
        parallel: bool = True,
    ) -> dict[str, Any]:
        """
        Prioritize multiple emails in batch.

        Args:
            emails: List of email dicts to prioritize
            user_id: User ID for personalized prioritization
            parallel: Process emails in parallel (default: True)

        Returns:
            Dict with batch results, total_processed, and summary by priority
        """
        data: dict[str, Any] = {
            "emails": emails,
            "parallel": parallel,
        }
        if user_id is not None:
            data["user_id"] = user_id
        return await self._client.request("POST", "/api/v1/email/prioritize/batch", json=data)

    # ===========================================================================
    # Inbox Triage
    # ===========================================================================

    async def triage_inbox(
        self,
        emails: list[dict[str, Any]],
        user_id: str | None = None,
        include_auto_replies: bool = False,
    ) -> dict[str, Any]:
        """
        Triage inbox with full categorization.

        Args:
            emails: List of email dicts to triage
            user_id: User ID for personalized triage
            include_auto_replies: Generate auto-reply drafts (default: False)

        Returns:
            Dict with triage results including category, priority,
            suggested_folder, and optional auto-reply drafts
        """
        data: dict[str, Any] = {
            "emails": emails,
            "include_auto_replies": include_auto_replies,
        }
        if user_id is not None:
            data["user_id"] = user_id
        return await self._client.request("POST", "/api/v1/email/triage", json=data)

    # ===========================================================================
    # History
    # ===========================================================================

    async def get_history(
        self,
        user_id: str,
        limit: int | None = None,
        since: str | None = None,
    ) -> dict[str, Any]:
        """
        Get prioritization history for a user.

        Args:
            user_id: User ID to get history for
            limit: Maximum number of results (default: 50)
            since: Filter to emails since this ISO timestamp

        Returns:
            Dict with list of past prioritization results
        """
        params: dict[str, Any] = {"user_id": user_id}
        if limit is not None:
            params["limit"] = limit
        if since is not None:
            params["since"] = since
        return await self._client.request("GET", "/api/v1/email/prioritize/history", params=params)
