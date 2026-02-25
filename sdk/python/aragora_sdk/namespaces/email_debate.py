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
    ) -> dict[str, Any]:
        """
        Prioritize multiple emails in a single batch request.

        Args:
            emails: List of email dicts to prioritize (same format as prioritize())
            user_id: User ID for personalized prioritization

        Returns:
            Dict with batch prioritization results
        """
        data: dict[str, Any] = {"emails": emails}
        if user_id is not None:
            data["user_id"] = user_id
        return self._client.request("POST", "/api/v1/email/prioritize/batch", json=data)

    def triage(
        self,
        emails: list[dict[str, Any]] | None = None,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Full inbox triage with categorization and auto-reply drafts.

        Args:
            emails: Optional list of emails to triage
            user_id: User ID for personalized triage
            **kwargs: Additional triage options

        Returns:
            Dict with triage results including categories and suggested actions
        """
        data: dict[str, Any] = {**kwargs}
        if emails is not None:
            data["emails"] = emails
        if user_id is not None:
            data["user_id"] = user_id
        return self._client.request("POST", "/api/v1/email/triage", json=data)


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
    ) -> dict[str, Any]:
        """Prioritize multiple emails in a single batch request.

        Args:
            emails: List of email dicts to prioritize
            user_id: User ID for personalized prioritization

        Returns:
            Dict with batch prioritization results
        """
        data: dict[str, Any] = {"emails": emails}
        if user_id is not None:
            data["user_id"] = user_id
        return await self._client.request("POST", "/api/v1/email/prioritize/batch", json=data)

    async def triage(
        self,
        emails: list[dict[str, Any]] | None = None,
        user_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Full inbox triage with categorization and auto-reply drafts.

        Args:
            emails: Optional list of emails to triage
            user_id: User ID for personalized triage
            **kwargs: Additional triage options

        Returns:
            Dict with triage results including categories and suggested actions
        """
        data: dict[str, Any] = {**kwargs}
        if emails is not None:
            data["emails"] = emails
        if user_id is not None:
            data["user_id"] = user_id
        return await self._client.request("POST", "/api/v1/email/triage", json=data)
