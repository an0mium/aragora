"""
Advertising Namespace API

Provides methods for managing advertising platform integrations:
- Connect and manage advertising platforms (Google Ads, Meta, etc.)
- View and manage campaigns
- Analyze performance and get budget recommendations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class AdvertisingAPI:
    """
    Synchronous Advertising API.

    Provides methods for advertising platform management:
    - Platform connections
    - Campaign management
    - Performance analytics
    - Budget recommendations

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> platforms = client.advertising.list_platforms()
        >>> performance = client.advertising.get_performance()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Platform Management
    # ===========================================================================

    def list_platforms(self) -> dict[str, Any]:
        """
        List available advertising platforms.

        Returns:
            Dict with platforms array containing supported advertising platforms
        """
        return self._client.request("GET", "/api/v1/advertising/platforms")

    def connect(
        self,
        platform: str,
        credentials: dict[str, Any],
        account_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Connect an advertising platform.

        Args:
            platform: Platform name (google_ads, meta, linkedin, etc.)
            credentials: Platform-specific credentials
            account_id: Optional account ID to connect

        Returns:
            Connection result with platform details
        """
        data: dict[str, Any] = {"platform": platform, "credentials": credentials}
        if account_id:
            data["account_id"] = account_id

        return self._client.request("POST", "/api/v1/advertising/connect", json=data)

    def disconnect(self, platform: str) -> dict[str, Any]:
        """
        Disconnect an advertising platform.

        Args:
            platform: Platform name to disconnect

        Returns:
            Disconnection result
        """
        return self._client.request("DELETE", f"/api/v1/advertising/{platform}")

    # ===========================================================================
    # Campaign Management
    # ===========================================================================

    def list_campaigns(
        self,
        platform: str | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List advertising campaigns.

        Args:
            platform: Filter by platform (optional, returns all if not specified)
            status: Filter by status (active, paused, ended)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            Dict with campaigns array
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        if platform:
            return self._client.request(
                "GET", f"/api/v1/advertising/{platform}/campaigns", params=params
            )
        return self._client.request("GET", "/api/v1/advertising/campaigns", params=params)

    def get_campaign(self, platform: str, campaign_id: str) -> dict[str, Any]:
        """
        Get a specific campaign.

        Args:
            platform: Platform name
            campaign_id: Campaign ID

        Returns:
            Campaign details
        """
        return self._client.request(
            "GET", f"/api/v1/advertising/{platform}/campaigns/{campaign_id}"
        )

    def create_campaign(
        self,
        platform: str,
        name: str,
        budget: float,
        objective: str,
        targeting: dict[str, Any] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Create a new advertising campaign.

        Args:
            platform: Platform name
            name: Campaign name
            budget: Daily or total budget
            objective: Campaign objective (awareness, traffic, conversions, etc.)
            targeting: Targeting configuration
            start_date: Start date (ISO 8601)
            end_date: End date (ISO 8601)
            **kwargs: Additional platform-specific parameters

        Returns:
            Created campaign details
        """
        data: dict[str, Any] = {
            "name": name,
            "budget": budget,
            "objective": objective,
            **kwargs,
        }
        if targeting:
            data["targeting"] = targeting
        if start_date:
            data["start_date"] = start_date
        if end_date:
            data["end_date"] = end_date

        return self._client.request("POST", f"/api/v1/advertising/{platform}/campaigns", json=data)

    def update_campaign(
        self,
        platform: str,
        campaign_id: str,
        name: str | None = None,
        budget: float | None = None,
        status: str | None = None,
        targeting: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Update an existing campaign.

        Args:
            platform: Platform name
            campaign_id: Campaign ID
            name: New campaign name
            budget: New budget
            status: New status (active, paused)
            targeting: Updated targeting
            **kwargs: Additional parameters

        Returns:
            Updated campaign details
        """
        data: dict[str, Any] = {**kwargs}
        if name is not None:
            data["name"] = name
        if budget is not None:
            data["budget"] = budget
        if status is not None:
            data["status"] = status
        if targeting is not None:
            data["targeting"] = targeting

        return self._client.request(
            "PUT", f"/api/v1/advertising/{platform}/campaigns/{campaign_id}", json=data
        )

    # ===========================================================================
    # Analytics & Recommendations
    # ===========================================================================

    def get_performance(
        self,
        platform: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Get advertising performance metrics.

        Args:
            platform: Filter by platform (optional, returns aggregated if not specified)
            start_date: Start date for metrics (ISO 8601)
            end_date: End date for metrics (ISO 8601)
            metrics: Specific metrics to include

        Returns:
            Performance data with metrics
        """
        params: dict[str, Any] = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if metrics:
            params["metrics"] = ",".join(metrics)

        if platform:
            return self._client.request(
                "GET", f"/api/v1/advertising/{platform}/performance", params=params
            )
        return self._client.request("GET", "/api/v1/advertising/performance", params=params)

    def analyze(
        self,
        campaign_ids: list[str] | None = None,
        analysis_type: str = "performance",
    ) -> dict[str, Any]:
        """
        Analyze advertising campaigns.

        Args:
            campaign_ids: Campaign IDs to analyze (optional, analyzes all if not specified)
            analysis_type: Type of analysis (performance, audience, creative, etc.)

        Returns:
            Analysis results with insights and recommendations
        """
        data: dict[str, Any] = {"analysis_type": analysis_type}
        if campaign_ids:
            data["campaign_ids"] = campaign_ids

        return self._client.request("POST", "/api/v1/advertising/analyze", json=data)

    def get_budget_recommendations(
        self,
        goal: str | None = None,
        timeframe: str | None = None,
    ) -> dict[str, Any]:
        """
        Get AI-powered budget recommendations.

        Args:
            goal: Optimization goal (maximize_conversions, maximize_reach, etc.)
            timeframe: Timeframe for recommendations (weekly, monthly, quarterly)

        Returns:
            Budget recommendations by platform and campaign
        """
        params: dict[str, Any] = {}
        if goal:
            params["goal"] = goal
        if timeframe:
            params["timeframe"] = timeframe

        return self._client.request(
            "GET", "/api/v1/advertising/budget-recommendations", params=params
        )


class AsyncAdvertisingAPI:
    """
    Asynchronous Advertising API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     platforms = await client.advertising.list_platforms()
        ...     performance = await client.advertising.get_performance()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Platform Management
    # ===========================================================================

    async def list_platforms(self) -> dict[str, Any]:
        """List available advertising platforms."""
        return await self._client.request("GET", "/api/v1/advertising/platforms")

    async def connect(
        self,
        platform: str,
        credentials: dict[str, Any],
        account_id: str | None = None,
    ) -> dict[str, Any]:
        """Connect an advertising platform."""
        data: dict[str, Any] = {"platform": platform, "credentials": credentials}
        if account_id:
            data["account_id"] = account_id

        return await self._client.request("POST", "/api/v1/advertising/connect", json=data)

    async def disconnect(self, platform: str) -> dict[str, Any]:
        """Disconnect an advertising platform."""
        return await self._client.request("DELETE", f"/api/v1/advertising/{platform}")

    # ===========================================================================
    # Campaign Management
    # ===========================================================================

    async def list_campaigns(
        self,
        platform: str | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List advertising campaigns."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        if platform:
            return await self._client.request(
                "GET", f"/api/v1/advertising/{platform}/campaigns", params=params
            )
        return await self._client.request("GET", "/api/v1/advertising/campaigns", params=params)

    async def get_campaign(self, platform: str, campaign_id: str) -> dict[str, Any]:
        """Get a specific campaign."""
        return await self._client.request(
            "GET", f"/api/v1/advertising/{platform}/campaigns/{campaign_id}"
        )

    async def create_campaign(
        self,
        platform: str,
        name: str,
        budget: float,
        objective: str,
        targeting: dict[str, Any] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a new advertising campaign."""
        data: dict[str, Any] = {
            "name": name,
            "budget": budget,
            "objective": objective,
            **kwargs,
        }
        if targeting:
            data["targeting"] = targeting
        if start_date:
            data["start_date"] = start_date
        if end_date:
            data["end_date"] = end_date

        return await self._client.request(
            "POST", f"/api/v1/advertising/{platform}/campaigns", json=data
        )

    async def update_campaign(
        self,
        platform: str,
        campaign_id: str,
        name: str | None = None,
        budget: float | None = None,
        status: str | None = None,
        targeting: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Update an existing campaign."""
        data: dict[str, Any] = {**kwargs}
        if name is not None:
            data["name"] = name
        if budget is not None:
            data["budget"] = budget
        if status is not None:
            data["status"] = status
        if targeting is not None:
            data["targeting"] = targeting

        return await self._client.request(
            "PUT", f"/api/v1/advertising/{platform}/campaigns/{campaign_id}", json=data
        )

    # ===========================================================================
    # Analytics & Recommendations
    # ===========================================================================

    async def get_performance(
        self,
        platform: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get advertising performance metrics."""
        params: dict[str, Any] = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if metrics:
            params["metrics"] = ",".join(metrics)

        if platform:
            return await self._client.request(
                "GET", f"/api/v1/advertising/{platform}/performance", params=params
            )
        return await self._client.request("GET", "/api/v1/advertising/performance", params=params)

    async def analyze(
        self,
        campaign_ids: list[str] | None = None,
        analysis_type: str = "performance",
    ) -> dict[str, Any]:
        """Analyze advertising campaigns."""
        data: dict[str, Any] = {"analysis_type": analysis_type}
        if campaign_ids:
            data["campaign_ids"] = campaign_ids

        return await self._client.request("POST", "/api/v1/advertising/analyze", json=data)

    async def get_budget_recommendations(
        self,
        goal: str | None = None,
        timeframe: str | None = None,
    ) -> dict[str, Any]:
        """Get AI-powered budget recommendations."""
        params: dict[str, Any] = {}
        if goal:
            params["goal"] = goal
        if timeframe:
            params["timeframe"] = timeframe

        return await self._client.request(
            "GET", "/api/v1/advertising/budget-recommendations", params=params
        )
