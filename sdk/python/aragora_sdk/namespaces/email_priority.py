"""
Email Priority Namespace API

Provides email prioritization and inbox management:
- Email priority scoring with ML/LLM tiers
- Inbox ranking and categorization
- User feedback for learning
- VIP sender management
- Gmail OAuth integration

Features:
- Multi-tier priority scoring
- Smart email categorization
- Cross-channel context boosts
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


ScoringTier = Literal["tier_1_rules", "tier_2_ml", "tier_3_llm"]
UserAction = Literal["read", "archived", "deleted", "replied", "starred", "important", "snoozed"]
EmailCategory = Literal[
    "invoices", "receipts", "newsletters", "promotions", "personal", "work", "other"
]
GmailScopes = Literal["readonly", "full"]


class EmailPriorityAPI:
    """
    Synchronous Email Priority API.

    Provides methods for email prioritization and inbox management:
    - Score and rank emails by priority
    - Categorize emails into smart folders
    - Record user feedback for ML learning
    - Manage VIP senders
    - Connect Gmail via OAuth

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="...")
        >>> result = client.email_priority.prioritize(email)
        >>> ranked = client.email_priority.rank_inbox(emails)
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # =========================================================================
    # Priority Scoring
    # =========================================================================

    def prioritize(
        self,
        email: dict[str, Any],
        force_tier: ScoringTier | None = None,
    ) -> dict[str, Any]:
        """
        Score a single email for priority.

        Args:
            email: Email message dict with subject, from_address, body, etc.
            force_tier: Force specific scoring tier

        Returns:
            Dict with priority result including score, confidence, rationale
        """
        data: dict[str, Any] = {"email": email}
        if force_tier:
            data["force_tier"] = force_tier
        return self._client.request("POST", "/api/v1/email/prioritize", json=data)

    def rank_inbox(
        self,
        emails: list[dict[str, Any]],
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Rank multiple emails by priority.

        Args:
            emails: List of email message dicts
            limit: Maximum results to return

        Returns:
            Dict with ranked results and total count
        """
        data: dict[str, Any] = {"emails": emails}
        if limit:
            data["limit"] = limit
        return self._client.request("POST", "/api/v1/email/rank-inbox", json=data)

    # =========================================================================
    # Feedback and Learning
    # =========================================================================

    def record_feedback(
        self,
        email_id: str,
        action: UserAction,
        email: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Record user action for ML learning.

        Args:
            email_id: Email message ID
            action: User action taken
            email: Optional email data for context

        Returns:
            Dict with success status
        """
        data: dict[str, Any] = {"email_id": email_id, "action": action}
        if email:
            data["email"] = email
        return self._client.request("POST", "/api/v1/email/feedback", json=data)

    def record_feedback_batch(
        self,
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Record batch of user actions.

        Args:
            items: List of feedback items with email_id, action, etc.

        Returns:
            Dict with recorded/errors counts
        """
        return self._client.request("POST", "/api/v1/email/feedback/batch", json={"items": items})

    # =========================================================================
    # Categorization
    # =========================================================================

    def categorize(self, email: dict[str, Any]) -> dict[str, Any]:
        """
        Categorize an email to a smart folder.

        Args:
            email: Email message dict

        Returns:
            Dict with category result and confidence
        """
        return self._client.request("POST", "/api/v1/email/categorize", json={"email": email})

    def categorize_batch(
        self,
        emails: list[dict[str, Any]],
        concurrency: int | None = None,
    ) -> dict[str, Any]:
        """
        Categorize multiple emails.

        Args:
            emails: List of email message dicts
            concurrency: Parallel processing level

        Returns:
            Dict with results and category stats
        """
        data: dict[str, Any] = {"emails": emails}
        if concurrency:
            data["concurrency"] = concurrency
        return self._client.request("POST", "/api/v1/email/categorize/batch", json=data)

    def apply_label(
        self,
        email_id: str,
        category: EmailCategory,
    ) -> dict[str, Any]:
        """
        Apply Gmail label based on category.

        Args:
            email_id: Email message ID
            category: Category to apply

        Returns:
            Dict with success status and applied label
        """
        return self._client.request(
            "POST",
            "/api/v1/email/categorize/apply-label",
            json={"email_id": email_id, "category": category},
        )

    # =========================================================================
    # Inbox Management
    # =========================================================================

    def fetch_inbox(
        self,
        limit: int | None = None,
        offset: int | None = None,
        labels: list[str] | None = None,
        unread_only: bool | None = None,
    ) -> dict[str, Any]:
        """
        Fetch and rank inbox emails.

        Args:
            limit: Maximum emails to fetch
            offset: Number to skip
            labels: Filter by Gmail labels
            unread_only: Only unread emails

        Returns:
            Dict with ranked inbox items
        """
        params: dict[str, Any] = {}
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        if labels:
            params["labels"] = labels
        if unread_only is not None:
            params["unread_only"] = unread_only
        return self._client.request("GET", "/api/v1/email/inbox", params=params if params else None)

    # =========================================================================
    # Configuration
    # =========================================================================

    def get_config(self) -> dict[str, Any]:
        """
        Get email prioritization configuration.

        Returns:
            Dict with VIP lists, thresholds, signal settings
        """
        return self._client.request("GET", "/api/v1/email/config")

    def update_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        """
        Update email prioritization configuration.

        Args:
            updates: Configuration updates

        Returns:
            Dict with updated config
        """
        return self._client.request("PUT", "/api/v1/email/config", json=updates)

    # =========================================================================
    # VIP Management
    # =========================================================================

    def add_vip(
        self,
        email: str | None = None,
        domain: str | None = None,
    ) -> dict[str, Any]:
        """
        Add a VIP email address or domain.

        Args:
            email: VIP email address
            domain: VIP domain

        Returns:
            Dict with success status
        """
        data: dict[str, Any] = {}
        if email:
            data["email"] = email
        if domain:
            data["domain"] = domain
        return self._client.request("POST", "/api/v1/email/vip", json=data)

    def remove_vip(
        self,
        email: str | None = None,
        domain: str | None = None,
    ) -> dict[str, Any]:
        """
        Remove a VIP email address or domain.

        Args:
            email: VIP email to remove
            domain: VIP domain to remove

        Returns:
            Dict with success status
        """
        data: dict[str, Any] = {}
        if email:
            data["email"] = email
        if domain:
            data["domain"] = domain
        return self._client.request("DELETE", "/api/v1/email/vip", json=data)

    # =========================================================================
    # Gmail OAuth
    # =========================================================================

    def get_gmail_oauth_url(
        self,
        redirect_uri: str,
        state: str | None = None,
        scopes: GmailScopes | None = None,
    ) -> dict[str, Any]:
        """
        Get Gmail OAuth authorization URL.

        Args:
            redirect_uri: OAuth callback URL
            state: Optional state parameter
            scopes: Gmail permission scopes

        Returns:
            Dict with oauth_url and scopes
        """
        data: dict[str, Any] = {"redirect_uri": redirect_uri}
        if state:
            data["state"] = state
        if scopes:
            data["scopes"] = scopes
        return self._client.request("POST", "/api/v1/email/gmail/oauth/url", json=data)

    def handle_gmail_oauth_callback(
        self,
        code: str,
        redirect_uri: str,
    ) -> dict[str, Any]:
        """
        Handle Gmail OAuth callback.

        Args:
            code: Authorization code
            redirect_uri: Original redirect URI

        Returns:
            Dict with success status and email
        """
        return self._client.request(
            "POST",
            "/api/v1/email/gmail/oauth/callback",
            json={"code": code, "redirect_uri": redirect_uri},
        )

    def get_gmail_status(self) -> dict[str, Any]:
        """
        Check Gmail connection status.

        Returns:
            Dict with authentication status, email, scopes
        """
        return self._client.request("GET", "/api/v1/email/gmail/status")

    # =========================================================================
    # Cross-Channel Context
    # =========================================================================

    def get_context(self, email_address: str) -> dict[str, Any]:
        """
        Get cross-channel context for an email address.

        Args:
            email_address: Email to look up

        Returns:
            Dict with Slack, calendar, drive activity
        """
        return self._client.request("GET", f"/api/v1/email/context/{email_address}")

    def get_context_boost(self, email: dict[str, Any]) -> dict[str, Any]:
        """
        Get context-based priority boosts for an email.

        Args:
            email: Email message dict

        Returns:
            Dict with boost factors from various signals
        """
        return self._client.request("POST", "/api/v1/email/context/boost", json={"email": email})


class AsyncEmailPriorityAPI:
    """
    Asynchronous Email Priority API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     result = await client.email_priority.prioritize(email)
        ...     await client.email_priority.record_feedback(email_id, "replied")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def prioritize(
        self,
        email: dict[str, Any],
        force_tier: ScoringTier | None = None,
    ) -> dict[str, Any]:
        """Score a single email for priority."""
        data: dict[str, Any] = {"email": email}
        if force_tier:
            data["force_tier"] = force_tier
        return await self._client.request("POST", "/api/v1/email/prioritize", json=data)

    async def rank_inbox(
        self,
        emails: list[dict[str, Any]],
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Rank multiple emails by priority."""
        data: dict[str, Any] = {"emails": emails}
        if limit:
            data["limit"] = limit
        return await self._client.request("POST", "/api/v1/email/rank-inbox", json=data)

    async def record_feedback(
        self,
        email_id: str,
        action: UserAction,
        email: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record user action for ML learning."""
        data: dict[str, Any] = {"email_id": email_id, "action": action}
        if email:
            data["email"] = email
        return await self._client.request("POST", "/api/v1/email/feedback", json=data)

    async def record_feedback_batch(
        self,
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Record batch of user actions."""
        return await self._client.request(
            "POST", "/api/v1/email/feedback/batch", json={"items": items}
        )

    async def categorize(self, email: dict[str, Any]) -> dict[str, Any]:
        """Categorize an email to a smart folder."""
        return await self._client.request("POST", "/api/v1/email/categorize", json={"email": email})

    async def categorize_batch(
        self,
        emails: list[dict[str, Any]],
        concurrency: int | None = None,
    ) -> dict[str, Any]:
        """Categorize multiple emails."""
        data: dict[str, Any] = {"emails": emails}
        if concurrency:
            data["concurrency"] = concurrency
        return await self._client.request("POST", "/api/v1/email/categorize/batch", json=data)

    async def apply_label(
        self,
        email_id: str,
        category: EmailCategory,
    ) -> dict[str, Any]:
        """Apply Gmail label based on category."""
        return await self._client.request(
            "POST",
            "/api/v1/email/categorize/apply-label",
            json={"email_id": email_id, "category": category},
        )

    async def fetch_inbox(
        self,
        limit: int | None = None,
        offset: int | None = None,
        labels: list[str] | None = None,
        unread_only: bool | None = None,
    ) -> dict[str, Any]:
        """Fetch and rank inbox emails."""
        params: dict[str, Any] = {}
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        if labels:
            params["labels"] = labels
        if unread_only is not None:
            params["unread_only"] = unread_only
        return await self._client.request(
            "GET", "/api/v1/email/inbox", params=params if params else None
        )

    async def get_config(self) -> dict[str, Any]:
        """Get email prioritization configuration."""
        return await self._client.request("GET", "/api/v1/email/config")

    async def update_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Update email prioritization configuration."""
        return await self._client.request("PUT", "/api/v1/email/config", json=updates)

    async def add_vip(
        self,
        email: str | None = None,
        domain: str | None = None,
    ) -> dict[str, Any]:
        """Add a VIP email address or domain."""
        data: dict[str, Any] = {}
        if email:
            data["email"] = email
        if domain:
            data["domain"] = domain
        return await self._client.request("POST", "/api/v1/email/vip", json=data)

    async def remove_vip(
        self,
        email: str | None = None,
        domain: str | None = None,
    ) -> dict[str, Any]:
        """Remove a VIP email address or domain."""
        data: dict[str, Any] = {}
        if email:
            data["email"] = email
        if domain:
            data["domain"] = domain
        return await self._client.request("DELETE", "/api/v1/email/vip", json=data)

    async def get_gmail_oauth_url(
        self,
        redirect_uri: str,
        state: str | None = None,
        scopes: GmailScopes | None = None,
    ) -> dict[str, Any]:
        """Get Gmail OAuth authorization URL."""
        data: dict[str, Any] = {"redirect_uri": redirect_uri}
        if state:
            data["state"] = state
        if scopes:
            data["scopes"] = scopes
        return await self._client.request("POST", "/api/v1/email/gmail/oauth/url", json=data)

    async def handle_gmail_oauth_callback(
        self,
        code: str,
        redirect_uri: str,
    ) -> dict[str, Any]:
        """Handle Gmail OAuth callback."""
        return await self._client.request(
            "POST",
            "/api/v1/email/gmail/oauth/callback",
            json={"code": code, "redirect_uri": redirect_uri},
        )

    async def get_gmail_status(self) -> dict[str, Any]:
        """Check Gmail connection status."""
        return await self._client.request("GET", "/api/v1/email/gmail/status")

    async def get_context(self, email_address: str) -> dict[str, Any]:
        """Get cross-channel context for an email address."""
        return await self._client.request("GET", f"/api/v1/email/context/{email_address}")

    async def get_context_boost(self, email: dict[str, Any]) -> dict[str, Any]:
        """Get context-based priority boosts for an email."""
        return await self._client.request(
            "POST", "/api/v1/email/context/boost", json={"email": email}
        )
