"""
Privacy Namespace API

Provides methods for GDPR/CCPA compliant data operations.

Features:
- User data export (GDPR Article 15, CCPA Right to Know)
- Data inventory and categories
- Account deletion (GDPR Article 17, CCPA Right to Delete)
- Privacy preferences management (CCPA Do Not Sell)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class PrivacyAPI:
    """
    Synchronous Privacy API.

    Provides methods for GDPR/CCPA compliant data operations:
    - Data export
    - Data inventory
    - Account deletion
    - Privacy preferences

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="...")
        >>> export = client.privacy.export_data()
        >>> prefs = client.privacy.get_preferences()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Data Export (GDPR Article 15 / CCPA Right to Know)
    # ===========================================================================

    def export_data(self, format: str = "json") -> dict[str, Any]:
        """
        Export all user data in GDPR-compliant format.

        GDPR Article 15: Right of access
        CCPA: Right to know

        Args:
            format: Export format - "json" (default) or "csv"

        Returns:
            Dict containing all user data with export metadata
        """
        return self._client.request(
            "GET",
            "/api/v1/privacy/export",
            params={"format": format},
        )

    # ===========================================================================
    # Data Inventory
    # ===========================================================================

    def get_data_inventory(self) -> dict[str, Any]:
        """
        Get inventory of data categories collected about the user.

        CCPA: Disclosure of data categories

        Returns:
            Dict with categories, third_party_sharing info, and opt_out status
        """
        return self._client.request("GET", "/api/v1/privacy/data-inventory")

    # ===========================================================================
    # Account Deletion (GDPR Article 17 / CCPA Right to Delete)
    # ===========================================================================

    def delete_account(
        self,
        password: str,
        confirm: bool = True,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """
        Delete user account and associated data.

        GDPR Article 17: Right to erasure
        CCPA: Right to delete

        Args:
            password: Current password for verification
            confirm: Must be True to confirm deletion
            reason: Optional reason for deletion

        Returns:
            Dict with deletion_id and data_deleted summary

        Note:
            Audit logs are retained for 7 years per compliance requirements.
        """
        data = {"password": password, "confirm": confirm}
        if reason:
            data["reason"] = reason

        return self._client.request(
            "DELETE",
            "/api/v1/privacy/account",
            json=data,
        )

    # ===========================================================================
    # Privacy Preferences
    # ===========================================================================

    def get_preferences(self) -> dict[str, Any]:
        """
        Get user's privacy preferences.

        Returns:
            Dict with do_not_sell, marketing_opt_out, analytics_opt_out,
            and third_party_sharing settings
        """
        return self._client.request("GET", "/api/v1/privacy/preferences")

    def update_preferences(
        self,
        do_not_sell: bool | None = None,
        marketing_opt_out: bool | None = None,
        analytics_opt_out: bool | None = None,
        third_party_sharing: bool | None = None,
    ) -> dict[str, Any]:
        """
        Update user's privacy preferences.

        CCPA: Do Not Sell My Personal Information

        Args:
            do_not_sell: Opt out of data selling
            marketing_opt_out: Opt out of marketing communications
            analytics_opt_out: Opt out of analytics tracking
            third_party_sharing: Control third-party data sharing

        Returns:
            Dict with updated preferences
        """
        data = {}
        if do_not_sell is not None:
            data["do_not_sell"] = do_not_sell
        if marketing_opt_out is not None:
            data["marketing_opt_out"] = marketing_opt_out
        if analytics_opt_out is not None:
            data["analytics_opt_out"] = analytics_opt_out
        if third_party_sharing is not None:
            data["third_party_sharing"] = third_party_sharing

        return self._client.request(
            "POST",
            "/api/v1/privacy/preferences",
            json=data,
        )


class AsyncPrivacyAPI:
    """
    Asynchronous Privacy API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     export = await client.privacy.export_data()
        ...     prefs = await client.privacy.get_preferences()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Data Export
    # ===========================================================================

    async def export_data(self, format: str = "json") -> dict[str, Any]:
        """Export all user data in GDPR-compliant format."""
        return await self._client.request(
            "GET",
            "/api/v1/privacy/export",
            params={"format": format},
        )

    # ===========================================================================
    # Data Inventory
    # ===========================================================================

    async def get_data_inventory(self) -> dict[str, Any]:
        """Get inventory of data categories collected about the user."""
        return await self._client.request("GET", "/api/v1/privacy/data-inventory")

    # ===========================================================================
    # Account Deletion
    # ===========================================================================

    async def delete_account(
        self,
        password: str,
        confirm: bool = True,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Delete user account and associated data."""
        data = {"password": password, "confirm": confirm}
        if reason:
            data["reason"] = reason

        return await self._client.request(
            "DELETE",
            "/api/v1/privacy/account",
            json=data,
        )

    # ===========================================================================
    # Privacy Preferences
    # ===========================================================================

    async def get_preferences(self) -> dict[str, Any]:
        """Get user's privacy preferences."""
        return await self._client.request("GET", "/api/v1/privacy/preferences")

    async def update_preferences(
        self,
        do_not_sell: bool | None = None,
        marketing_opt_out: bool | None = None,
        analytics_opt_out: bool | None = None,
        third_party_sharing: bool | None = None,
    ) -> dict[str, Any]:
        """Update user's privacy preferences."""
        data = {}
        if do_not_sell is not None:
            data["do_not_sell"] = do_not_sell
        if marketing_opt_out is not None:
            data["marketing_opt_out"] = marketing_opt_out
        if analytics_opt_out is not None:
            data["analytics_opt_out"] = analytics_opt_out
        if third_party_sharing is not None:
            data["third_party_sharing"] = third_party_sharing

        return await self._client.request(
            "POST",
            "/api/v1/privacy/preferences",
            json=data,
        )
