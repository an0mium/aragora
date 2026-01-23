"""
TikTok Ads Connector.

Integration with TikTok Ads API:
- Campaign management
- Ad group management
- Creative management
- Audience targeting
- Performance analytics
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any

import httpx


class CampaignStatus(Enum):
    """TikTok campaign status."""

    ENABLE = "ENABLE"
    DISABLE = "DISABLE"
    DELETE = "DELETE"


class CampaignObjective(Enum):
    """TikTok campaign objectives."""

    TRAFFIC = "TRAFFIC"
    REACH = "REACH"
    VIDEO_VIEWS = "VIDEO_VIEWS"
    CONVERSIONS = "CONVERSIONS"
    APP_INSTALL = "APP_INSTALL"
    LEAD_GENERATION = "LEAD_GENERATION"
    CATALOG_SALES = "CATALOG_SALES"
    ENGAGEMENT = "ENGAGEMENT"


class AdGroupStatus(Enum):
    """TikTok ad group status."""

    ENABLE = "ENABLE"
    DISABLE = "DISABLE"
    DELETE = "DELETE"


class OptimizationGoal(Enum):
    """TikTok optimization goals."""

    CLICK = "CLICK"
    SHOW = "SHOW"
    REACH = "REACH"
    CONVERT = "CONVERT"
    VIDEO_VIEW = "VIDEO_VIEW"
    LEAD = "LEAD"
    APP_INSTALL = "APP_INSTALL"
    ENGAGED_VIEW = "ENGAGED_VIEW"


class PlacementType(Enum):
    """TikTok placement types."""

    PLACEMENT_TYPE_TIKTOK = "PLACEMENT_TYPE_TIKTOK"
    PLACEMENT_TYPE_PANGLE = "PLACEMENT_TYPE_PANGLE"
    PLACEMENT_TYPE_AUTOMATIC = "PLACEMENT_TYPE_AUTOMATIC"


class BillingEvent(Enum):
    """TikTok billing events."""

    CPM = "CPM"
    CPC = "CPC"
    OCPM = "OCPM"
    CPV = "CPV"


@dataclass
class TikTokAdsCredentials:
    """TikTok Ads API credentials."""

    access_token: str
    advertiser_id: str
    app_id: str | None = None
    secret: str | None = None


@dataclass
class Campaign:
    """TikTok advertising campaign."""

    id: str
    name: str
    advertiser_id: str
    status: CampaignStatus
    objective: CampaignObjective
    budget: float | None = None
    budget_mode: str = "BUDGET_MODE_DAY"  # BUDGET_MODE_DAY or BUDGET_MODE_TOTAL
    create_time: datetime | None = None
    modify_time: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "Campaign":
        """Create from TikTok API response."""
        return cls(
            id=str(data.get("campaign_id", "")),
            name=data.get("campaign_name", ""),
            advertiser_id=str(data.get("advertiser_id", "")),
            status=CampaignStatus(data.get("status", "DISABLE")),
            objective=CampaignObjective(data.get("objective_type", "TRAFFIC")),
            budget=float(data.get("budget", 0)) if data.get("budget") else None,
            budget_mode=data.get("budget_mode", "BUDGET_MODE_DAY"),
            create_time=datetime.fromisoformat(data["create_time"].replace("Z", "+00:00"))
            if data.get("create_time")
            else None,
            modify_time=datetime.fromisoformat(data["modify_time"].replace("Z", "+00:00"))
            if data.get("modify_time")
            else None,
        )


@dataclass
class AdGroup:
    """TikTok ad group."""

    id: str
    campaign_id: str
    name: str
    advertiser_id: str
    status: AdGroupStatus
    optimization_goal: OptimizationGoal
    placement_type: PlacementType
    billing_event: BillingEvent
    budget: float | None = None
    bid_price: float | None = None
    schedule_start_time: datetime | None = None
    schedule_end_time: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "AdGroup":
        """Create from TikTok API response."""
        return cls(
            id=str(data.get("adgroup_id", "")),
            campaign_id=str(data.get("campaign_id", "")),
            name=data.get("adgroup_name", ""),
            advertiser_id=str(data.get("advertiser_id", "")),
            status=AdGroupStatus(data.get("status", "DISABLE")),
            optimization_goal=OptimizationGoal(data.get("optimization_goal", "CLICK")),
            placement_type=PlacementType(data.get("placement_type", "PLACEMENT_TYPE_AUTOMATIC")),
            billing_event=BillingEvent(data.get("billing_event", "CPC")),
            budget=float(data.get("budget", 0)) if data.get("budget") else None,
            bid_price=float(data.get("bid_price", 0)) if data.get("bid_price") else None,
            schedule_start_time=datetime.fromisoformat(
                data["schedule_start_time"].replace("Z", "+00:00")
            )
            if data.get("schedule_start_time")
            else None,
            schedule_end_time=datetime.fromisoformat(
                data["schedule_end_time"].replace("Z", "+00:00")
            )
            if data.get("schedule_end_time")
            else None,
        )


@dataclass
class Ad:
    """TikTok ad creative."""

    id: str
    adgroup_id: str
    name: str
    status: str
    call_to_action: str | None = None
    landing_page_url: str | None = None
    video_id: str | None = None
    image_ids: list[str] = field(default_factory=list)
    create_time: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "Ad":
        """Create from TikTok API response."""
        return cls(
            id=str(data.get("ad_id", "")),
            adgroup_id=str(data.get("adgroup_id", "")),
            name=data.get("ad_name", ""),
            status=data.get("status", ""),
            call_to_action=data.get("call_to_action"),
            landing_page_url=data.get("landing_page_url"),
            video_id=data.get("video_id"),
            image_ids=data.get("image_ids", []),
            create_time=datetime.fromisoformat(data["create_time"].replace("Z", "+00:00"))
            if data.get("create_time")
            else None,
        )


@dataclass
class AdGroupMetrics:
    """TikTok ad group performance metrics."""

    adgroup_id: str
    adgroup_name: str
    campaign_id: str
    start_date: date
    end_date: date
    impressions: int = 0
    clicks: int = 0
    spend: float = 0.0
    conversions: int = 0
    conversion_rate: float = 0.0
    video_views: int = 0
    video_views_p25: int = 0
    video_views_p50: int = 0
    video_views_p75: int = 0
    video_views_p100: int = 0
    reach: int = 0
    frequency: float = 0.0
    ctr: float = 0.0
    cpc: float = 0.0
    cpm: float = 0.0
    cost_per_conversion: float = 0.0

    @classmethod
    def from_api(
        cls,
        data: dict[str, Any],
        dimensions: dict[str, Any],
        start_date: date,
        end_date: date,
    ) -> "AdGroupMetrics":
        """Create from TikTok API response."""
        metrics = data.get("metrics", {})
        impressions = int(metrics.get("impressions", 0))
        clicks = int(metrics.get("clicks", 0))
        spend = float(metrics.get("spend", 0))
        conversions = int(metrics.get("conversion", 0))

        return cls(
            adgroup_id=dimensions.get("adgroup_id", ""),
            adgroup_name=dimensions.get("adgroup_name", ""),
            campaign_id=dimensions.get("campaign_id", ""),
            start_date=start_date,
            end_date=end_date,
            impressions=impressions,
            clicks=clicks,
            spend=spend,
            conversions=conversions,
            conversion_rate=(conversions / clicks * 100) if clicks > 0 else 0.0,
            video_views=int(metrics.get("video_views", 0)),
            video_views_p25=int(metrics.get("video_views_p25", 0)),
            video_views_p50=int(metrics.get("video_views_p50", 0)),
            video_views_p75=int(metrics.get("video_views_p75", 0)),
            video_views_p100=int(metrics.get("video_views_p100", 0)),
            reach=int(metrics.get("reach", 0)),
            frequency=float(metrics.get("frequency", 0)),
            ctr=(clicks / impressions * 100) if impressions > 0 else 0.0,
            cpc=(spend / clicks) if clicks > 0 else 0.0,
            cpm=(spend / impressions * 1000) if impressions > 0 else 0.0,
            cost_per_conversion=(spend / conversions) if conversions > 0 else 0.0,
        )


@dataclass
class CustomAudience:
    """TikTok custom audience."""

    id: str
    name: str
    advertiser_id: str
    audience_type: str
    audience_sub_type: str | None = None
    is_valid: bool = True
    audience_size: int | None = None
    create_time: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "CustomAudience":
        """Create from TikTok API response."""
        return cls(
            id=str(data.get("audience_id", "")),
            name=data.get("name", ""),
            advertiser_id=str(data.get("advertiser_id", "")),
            audience_type=data.get("audience_type", ""),
            audience_sub_type=data.get("audience_sub_type"),
            is_valid=data.get("is_valid", True),
            audience_size=data.get("audience_size"),
            create_time=datetime.fromisoformat(data["create_time"].replace("Z", "+00:00"))
            if data.get("create_time")
            else None,
        )


@dataclass
class Pixel:
    """TikTok tracking pixel."""

    id: str
    name: str
    advertiser_id: str
    code: str | None = None
    create_time: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "Pixel":
        """Create from TikTok API response."""
        return cls(
            id=str(data.get("pixel_id", "")),
            name=data.get("pixel_name", ""),
            advertiser_id=str(data.get("advertiser_id", "")),
            code=data.get("pixel_code"),
            create_time=datetime.fromisoformat(data["create_time"].replace("Z", "+00:00"))
            if data.get("create_time")
            else None,
        )


class TikTokAdsError(Exception):
    """TikTok Ads API error."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_code: int | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class TikTokAdsConnector:
    """TikTok Ads API connector."""

    BASE_URL = "https://business-api.tiktok.com/open_api/v1.3"

    def __init__(self, credentials: TikTokAdsCredentials):
        """Initialize with credentials."""
        self.credentials = credentials
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "Access-Token": self.credentials.access_token,
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make API request."""
        client = await self._get_client()
        url = f"{self.BASE_URL}/{endpoint}"

        if json_data and "advertiser_id" not in json_data:
            json_data["advertiser_id"] = self.credentials.advertiser_id

        if params and "advertiser_id" not in params:
            params["advertiser_id"] = self.credentials.advertiser_id

        response = await client.request(
            method=method,
            url=url,
            params=params,
            json=json_data,
        )

        data = response.json()

        if data.get("code") != 0:
            raise TikTokAdsError(
                message=data.get("message", f"API error: {response.status_code}"),
                status_code=response.status_code,
                error_code=data.get("code"),
            )

        return data.get("data", {})

    # Campaign Operations

    async def get_campaigns(
        self,
        status: CampaignStatus | None = None,
        page: int = 1,
        page_size: int = 100,
    ) -> list[Campaign]:
        """Get campaigns for the advertiser."""
        params = {
            "advertiser_id": self.credentials.advertiser_id,
            "page": page,
            "page_size": page_size,
        }

        if status:
            params["primary_status"] = status.value

        data = await self._request("GET", "campaign/get/", params=params)
        return [Campaign.from_api(c) for c in data.get("list", [])]

    async def get_campaign(self, campaign_id: str) -> Campaign:
        """Get a specific campaign by ID."""
        params = {
            "advertiser_id": self.credentials.advertiser_id,
            "campaign_ids": [campaign_id],
        }

        data = await self._request("GET", "campaign/get/", params=params)
        campaigns = data.get("list", [])
        if not campaigns:
            raise TikTokAdsError(f"Campaign {campaign_id} not found")
        return Campaign.from_api(campaigns[0])

    async def create_campaign(
        self,
        name: str,
        objective: CampaignObjective,
        budget: float | None = None,
        budget_mode: str = "BUDGET_MODE_DAY",
    ) -> Campaign:
        """Create a new campaign."""
        json_data = {
            "advertiser_id": self.credentials.advertiser_id,
            "campaign_name": name,
            "objective_type": objective.value,
            "budget_mode": budget_mode,
        }

        if budget:
            json_data["budget"] = budget

        data = await self._request("POST", "campaign/create/", json_data=json_data)
        return await self.get_campaign(str(data.get("campaign_id", "")))

    async def update_campaign_status(
        self,
        campaign_id: str,
        status: CampaignStatus,
    ) -> None:
        """Update campaign status."""
        json_data = {
            "advertiser_id": self.credentials.advertiser_id,
            "campaign_ids": [campaign_id],
            "operation_status": status.value,
        }

        await self._request("POST", "campaign/status/update/", json_data=json_data)

    # Ad Group Operations

    async def get_ad_groups(
        self,
        campaign_id: str | None = None,
        page: int = 1,
        page_size: int = 100,
    ) -> list[AdGroup]:
        """Get ad groups for the advertiser."""
        params = {
            "advertiser_id": self.credentials.advertiser_id,
            "page": page,
            "page_size": page_size,
        }

        if campaign_id:
            params["campaign_ids"] = [campaign_id]

        data = await self._request("GET", "adgroup/get/", params=params)
        return [AdGroup.from_api(ag) for ag in data.get("list", [])]

    async def create_ad_group(
        self,
        campaign_id: str,
        name: str,
        optimization_goal: OptimizationGoal,
        placement_type: PlacementType = PlacementType.PLACEMENT_TYPE_AUTOMATIC,
        billing_event: BillingEvent = BillingEvent.OCPM,
        budget: float | None = None,
        bid_price: float | None = None,
    ) -> AdGroup:
        """Create a new ad group."""
        json_data = {
            "advertiser_id": self.credentials.advertiser_id,
            "campaign_id": campaign_id,
            "adgroup_name": name,
            "optimization_goal": optimization_goal.value,
            "placement_type": placement_type.value,
            "billing_event": billing_event.value,
        }

        if budget:
            json_data["budget"] = budget
        if bid_price:
            json_data["bid_price"] = bid_price

        data = await self._request("POST", "adgroup/create/", json_data=json_data)
        ad_groups = await self.get_ad_groups(campaign_id=campaign_id)
        return next(
            (ag for ag in ad_groups if ag.id == str(data.get("adgroup_id", ""))), ad_groups[0]
        )

    # Ad Operations

    async def get_ads(
        self,
        adgroup_id: str | None = None,
        page: int = 1,
        page_size: int = 100,
    ) -> list[Ad]:
        """Get ads for the advertiser."""
        params = {
            "advertiser_id": self.credentials.advertiser_id,
            "page": page,
            "page_size": page_size,
        }

        if adgroup_id:
            params["adgroup_ids"] = [adgroup_id]

        data = await self._request("GET", "ad/get/", params=params)
        return [Ad.from_api(ad) for ad in data.get("list", [])]

    # Analytics

    async def get_ad_group_metrics(
        self,
        start_date: date,
        end_date: date,
        campaign_id: str | None = None,
        adgroup_ids: list[str] | None = None,
    ) -> list[AdGroupMetrics]:
        """Get performance metrics for ad groups."""
        json_data = {
            "advertiser_id": self.credentials.advertiser_id,
            "report_type": "BASIC",
            "dimensions": ["adgroup_id"],
            "data_level": "AUCTION_ADGROUP",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "metrics": [
                "impressions",
                "clicks",
                "spend",
                "conversion",
                "video_views",
                "video_views_p25",
                "video_views_p50",
                "video_views_p75",
                "video_views_p100",
                "reach",
                "frequency",
            ],
        }

        if campaign_id:
            json_data["filters"] = [
                {"field_name": "campaign_id", "filter_type": "IN", "filter_value": [campaign_id]}
            ]
        if adgroup_ids:
            json_data["filters"] = [
                {"field_name": "adgroup_id", "filter_type": "IN", "filter_value": adgroup_ids}
            ]

        data = await self._request("POST", "report/integrated/get/", json_data=json_data)

        results = []
        for item in data.get("list", []):
            dimensions = item.get("dimensions", {})
            results.append(AdGroupMetrics.from_api(item, dimensions, start_date, end_date))

        return results

    # Audiences

    async def get_custom_audiences(self) -> list[CustomAudience]:
        """Get custom audiences for the advertiser."""
        params = {
            "advertiser_id": self.credentials.advertiser_id,
        }

        data = await self._request("GET", "audience/custom/list/", params=params)
        return [CustomAudience.from_api(ca) for ca in data.get("list", [])]

    async def create_custom_audience(
        self,
        name: str,
        audience_type: str = "CUSTOMER_FILE",
    ) -> CustomAudience:
        """Create a custom audience."""
        json_data = {
            "advertiser_id": self.credentials.advertiser_id,
            "custom_audience_name": name,
            "audience_type": audience_type,
        }

        data = await self._request("POST", "audience/custom/create/", json_data=json_data)

        audiences = await self.get_custom_audiences()
        return next(
            (a for a in audiences if a.id == str(data.get("audience_id", ""))), audiences[0]
        )

    # Pixels

    async def get_pixels(self) -> list[Pixel]:
        """Get tracking pixels for the advertiser."""
        params = {
            "advertiser_id": self.credentials.advertiser_id,
        }

        data = await self._request("GET", "pixel/list/", params=params)
        return [Pixel.from_api(p) for p in data.get("pixels", [])]

    async def create_pixel(self, name: str) -> Pixel:
        """Create a tracking pixel."""
        json_data = {
            "advertiser_id": self.credentials.advertiser_id,
            "pixel_name": name,
        }

        data = await self._request("POST", "pixel/create/", json_data=json_data)

        pixels = await self.get_pixels()
        return next((p for p in pixels if p.id == str(data.get("pixel_id", ""))), pixels[0])


def get_mock_campaign() -> Campaign:
    """Get mock campaign for testing."""
    return Campaign(
        id="123456789",
        name="Test TikTok Campaign",
        advertiser_id="987654321",
        status=CampaignStatus.ENABLE,
        objective=CampaignObjective.TRAFFIC,
        budget=500.0,
        budget_mode="BUDGET_MODE_DAY",
        create_time=datetime.now(),
    )


def get_mock_metrics() -> AdGroupMetrics:
    """Get mock metrics for testing."""
    return AdGroupMetrics(
        adgroup_id="456789",
        adgroup_name="Test Ad Group",
        campaign_id="123456789",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        impressions=500000,
        clicks=15000,
        spend=3000.0,
        conversions=450,
        conversion_rate=3.0,
        video_views=200000,
        video_views_p25=150000,
        video_views_p50=100000,
        video_views_p75=50000,
        video_views_p100=25000,
        reach=300000,
        frequency=1.67,
        ctr=3.0,
        cpc=0.20,
        cpm=6.0,
        cost_per_conversion=6.67,
    )
