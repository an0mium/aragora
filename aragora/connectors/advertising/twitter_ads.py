"""
Twitter/X Ads Connector.

Integration with X (formerly Twitter) Ads API:
- Campaign management
- Promoted Tweets
- Audience targeting
- Conversion tracking
- Analytics and reporting
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any

import httpx


class CampaignStatus(Enum):
    """Twitter campaign status."""

    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    DRAFT = "DRAFT"
    SCHEDULED = "SCHEDULED"


class CampaignObjective(Enum):
    """Twitter campaign objectives."""

    AWARENESS = "AWARENESS"
    TWEET_ENGAGEMENTS = "TWEET_ENGAGEMENTS"
    VIDEO_VIEWS = "VIDEO_VIEWS"
    WEBSITE_CLICKS = "WEBSITE_CLICKS"
    APP_INSTALLS = "APP_INSTALLS"
    APP_ENGAGEMENTS = "APP_ENGAGEMENTS"
    FOLLOWERS = "FOLLOWERS"
    REACH = "REACH"


class LineItemType(Enum):
    """Line item (ad group) types."""

    PROMOTED_TWEETS = "PROMOTED_TWEETS"
    PROMOTED_ACCOUNT = "PROMOTED_ACCOUNT"
    PROMOTED_TREND = "PROMOTED_TREND"


class PlacementType(Enum):
    """Ad placement types."""

    ALL_ON_TWITTER = "ALL_ON_TWITTER"
    TWITTER_PROFILE = "TWITTER_PROFILE"
    TWITTER_TIMELINE = "TWITTER_TIMELINE"
    TWITTER_SEARCH = "TWITTER_SEARCH"
    PUBLISHER_NETWORK = "PUBLISHER_NETWORK"


@dataclass
class TwitterAdsCredentials:
    """Twitter Ads API credentials."""

    consumer_key: str
    consumer_secret: str
    access_token: str
    access_token_secret: str
    ads_account_id: str


@dataclass
class FundingInstrument:
    """Twitter ads funding instrument (payment method)."""

    id: str
    account_id: str
    currency: str
    funded_amount: float
    credit_remaining: float
    start_time: datetime | None = None
    end_time: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "FundingInstrument":
        """Create from Twitter API response."""
        return cls(
            id=data.get("id", ""),
            account_id=data.get("account_id", ""),
            currency=data.get("currency", "USD"),
            funded_amount=float(data.get("funded_amount_local_micro", 0)) / 1_000_000,
            credit_remaining=float(data.get("credit_remaining_local_micro", 0)) / 1_000_000,
            start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
        )


@dataclass
class Campaign:
    """Twitter advertising campaign."""

    id: str
    name: str
    account_id: str
    status: CampaignStatus
    objective: CampaignObjective
    funding_instrument_id: str
    daily_budget: float | None = None
    total_budget: float | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "Campaign":
        """Create from Twitter API response."""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            account_id=data.get("account_id", ""),
            status=CampaignStatus(data.get("entity_status", "PAUSED")),
            objective=CampaignObjective(data.get("objective", "WEBSITE_CLICKS")),
            funding_instrument_id=data.get("funding_instrument_id", ""),
            daily_budget=float(data.get("daily_budget_amount_local_micro", 0)) / 1_000_000 if data.get("daily_budget_amount_local_micro") else None,
            total_budget=float(data.get("total_budget_amount_local_micro", 0)) / 1_000_000 if data.get("total_budget_amount_local_micro") else None,
            start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
        )


@dataclass
class LineItem:
    """Twitter line item (ad group)."""

    id: str
    campaign_id: str
    name: str
    status: str
    line_item_type: LineItemType
    placements: list[PlacementType]
    bid_amount: float | None = None
    total_budget: float | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "LineItem":
        """Create from Twitter API response."""
        return cls(
            id=data.get("id", ""),
            campaign_id=data.get("campaign_id", ""),
            name=data.get("name", ""),
            status=data.get("entity_status", "PAUSED"),
            line_item_type=LineItemType(data.get("product_type", "PROMOTED_TWEETS")),
            placements=[PlacementType(p) for p in data.get("placements", [])],
            bid_amount=float(data.get("bid_amount_local_micro", 0)) / 1_000_000 if data.get("bid_amount_local_micro") else None,
            total_budget=float(data.get("total_budget_amount_local_micro", 0)) / 1_000_000 if data.get("total_budget_amount_local_micro") else None,
            start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
        )


@dataclass
class PromotedTweet:
    """Promoted tweet (ad creative)."""

    id: str
    line_item_id: str
    tweet_id: str
    status: str
    approval_status: str
    created_at: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "PromotedTweet":
        """Create from Twitter API response."""
        return cls(
            id=data.get("id", ""),
            line_item_id=data.get("line_item_id", ""),
            tweet_id=data.get("tweet_id", ""),
            status=data.get("entity_status", ""),
            approval_status=data.get("approval_status", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
        )


@dataclass
class CampaignAnalytics:
    """Twitter campaign analytics."""

    campaign_id: str
    start_date: date
    end_date: date
    impressions: int = 0
    engagements: int = 0
    clicks: int = 0
    spend: float = 0.0
    retweets: int = 0
    replies: int = 0
    likes: int = 0
    follows: int = 0
    video_views: int = 0
    app_installs: int = 0
    engagement_rate: float = 0.0
    cpe: float = 0.0  # Cost per engagement
    cpc: float = 0.0

    @classmethod
    def from_api(
        cls,
        data: dict[str, Any],
        campaign_id: str,
        start_date: date,
        end_date: date,
    ) -> "CampaignAnalytics":
        """Create from Twitter API response."""
        metrics = data.get("metrics", {})
        impressions = metrics.get("impressions", [0])[0] if metrics.get("impressions") else 0
        engagements = metrics.get("engagements", [0])[0] if metrics.get("engagements") else 0
        clicks = metrics.get("clicks", [0])[0] if metrics.get("clicks") else 0
        spend = float(metrics.get("billed_charge_local_micro", [0])[0] or 0) / 1_000_000

        return cls(
            campaign_id=campaign_id,
            start_date=start_date,
            end_date=end_date,
            impressions=impressions,
            engagements=engagements,
            clicks=clicks,
            spend=spend,
            retweets=metrics.get("retweets", [0])[0] if metrics.get("retweets") else 0,
            replies=metrics.get("replies", [0])[0] if metrics.get("replies") else 0,
            likes=metrics.get("likes", [0])[0] if metrics.get("likes") else 0,
            follows=metrics.get("follows", [0])[0] if metrics.get("follows") else 0,
            video_views=metrics.get("video_views", [0])[0] if metrics.get("video_views") else 0,
            app_installs=metrics.get("app_installs", [0])[0] if metrics.get("app_installs") else 0,
            engagement_rate=(engagements / impressions * 100) if impressions > 0 else 0.0,
            cpe=(spend / engagements) if engagements > 0 else 0.0,
            cpc=(spend / clicks) if clicks > 0 else 0.0,
        )


@dataclass
class TailoredAudience:
    """Twitter tailored audience (custom audience)."""

    id: str
    name: str
    audience_type: str
    audience_size: int | None = None
    targetable: bool = True
    created_at: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "TailoredAudience":
        """Create from Twitter API response."""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            audience_type=data.get("audience_type", ""),
            audience_size=data.get("audience_size"),
            targetable=data.get("targetable", True),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
        )


class TwitterAdsError(Exception):
    """Twitter Ads API error."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_code: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class TwitterAdsConnector:
    """Twitter/X Ads API connector."""

    BASE_URL = "https://ads-api.twitter.com/12"

    def __init__(self, credentials: TwitterAdsCredentials):
        """Initialize with credentials."""
        self.credentials = credentials
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with OAuth 1.0a authentication."""
        if self._client is None:
            # Note: In production, use a proper OAuth 1.0a library like authlib
            self._client = httpx.AsyncClient(
                headers={
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

    def _oauth_header(self, method: str, url: str) -> str:
        """Generate OAuth 1.0a authorization header."""
        # Simplified - in production use proper OAuth library
        import time
        import hashlib
        import hmac
        import base64
        import urllib.parse
        from uuid import uuid4

        oauth_params = {
            "oauth_consumer_key": self.credentials.consumer_key,
            "oauth_token": self.credentials.access_token,
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_timestamp": str(int(time.time())),
            "oauth_nonce": str(uuid4()),
            "oauth_version": "1.0",
        }

        # Create signature base string
        sorted_params = "&".join(
            f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(str(v), safe='')}"
            for k, v in sorted(oauth_params.items())
        )
        base_string = f"{method.upper()}&{urllib.parse.quote(url, safe='')}&{urllib.parse.quote(sorted_params, safe='')}"

        # Create signing key
        signing_key = f"{urllib.parse.quote(self.credentials.consumer_secret, safe='')}&{urllib.parse.quote(self.credentials.access_token_secret, safe='')}"

        # Generate signature
        signature = base64.b64encode(
            hmac.new(
                signing_key.encode(),
                base_string.encode(),
                hashlib.sha1,
            ).digest()
        ).decode()

        oauth_params["oauth_signature"] = signature

        # Build header
        header_params = ", ".join(
            f'{k}="{urllib.parse.quote(str(v), safe="")}"'
            for k, v in sorted(oauth_params.items())
        )
        return f"OAuth {header_params}"

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make API request."""
        client = await self._get_client()
        url = f"{self.BASE_URL}/accounts/{self.credentials.ads_account_id}/{endpoint}"

        headers = {
            "Authorization": self._oauth_header(method, url),
        }

        response = await client.request(
            method=method,
            url=url,
            params=params,
            json=json_data,
            headers=headers,
        )

        if response.status_code >= 400:
            error_data = response.json() if response.content else {}
            errors = error_data.get("errors", [{}])
            raise TwitterAdsError(
                message=errors[0].get("message", f"API error: {response.status_code}"),
                status_code=response.status_code,
                error_code=errors[0].get("code"),
            )

        return response.json()

    # Campaign Operations

    async def get_campaigns(
        self,
        status: CampaignStatus | None = None,
    ) -> list[Campaign]:
        """Get all campaigns for the account."""
        params = {}
        if status:
            params["entity_status"] = status.value

        response = await self._request("GET", "campaigns", params=params)
        return [Campaign.from_api(c) for c in response.get("data", [])]

    async def get_campaign(self, campaign_id: str) -> Campaign:
        """Get a specific campaign by ID."""
        response = await self._request("GET", f"campaigns/{campaign_id}")
        return Campaign.from_api(response.get("data", {}))

    async def create_campaign(
        self,
        name: str,
        funding_instrument_id: str,
        objective: CampaignObjective = CampaignObjective.WEBSITE_CLICKS,
        daily_budget: float | None = None,
        status: CampaignStatus = CampaignStatus.PAUSED,
    ) -> Campaign:
        """Create a new campaign."""
        data = {
            "name": name,
            "funding_instrument_id": funding_instrument_id,
            "objective": objective.value,
            "entity_status": status.value,
        }

        if daily_budget:
            data["daily_budget_amount_local_micro"] = int(daily_budget * 1_000_000)

        response = await self._request("POST", "campaigns", json_data=data)
        return Campaign.from_api(response.get("data", {}))

    async def update_campaign_status(
        self,
        campaign_id: str,
        status: CampaignStatus,
    ) -> Campaign:
        """Update campaign status."""
        data = {"entity_status": status.value}
        response = await self._request("PUT", f"campaigns/{campaign_id}", json_data=data)
        return Campaign.from_api(response.get("data", {}))

    # Line Item (Ad Group) Operations

    async def get_line_items(self, campaign_id: str) -> list[LineItem]:
        """Get line items for a campaign."""
        params = {"campaign_ids": campaign_id}
        response = await self._request("GET", "line_items", params=params)
        return [LineItem.from_api(li) for li in response.get("data", [])]

    async def create_line_item(
        self,
        campaign_id: str,
        name: str,
        line_item_type: LineItemType,
        placements: list[PlacementType],
        bid_amount: float,
    ) -> LineItem:
        """Create a new line item."""
        data = {
            "campaign_id": campaign_id,
            "name": name,
            "product_type": line_item_type.value,
            "placements": [p.value for p in placements],
            "bid_amount_local_micro": int(bid_amount * 1_000_000),
        }

        response = await self._request("POST", "line_items", json_data=data)
        return LineItem.from_api(response.get("data", {}))

    # Promoted Tweets

    async def get_promoted_tweets(self, line_item_id: str) -> list[PromotedTweet]:
        """Get promoted tweets for a line item."""
        params = {"line_item_ids": line_item_id}
        response = await self._request("GET", "promoted_tweets", params=params)
        return [PromotedTweet.from_api(pt) for pt in response.get("data", [])]

    async def create_promoted_tweet(
        self,
        line_item_id: str,
        tweet_id: str,
    ) -> PromotedTweet:
        """Create a promoted tweet."""
        data = {
            "line_item_id": line_item_id,
            "tweet_ids": [tweet_id],
        }

        response = await self._request("POST", "promoted_tweets", json_data=data)
        return PromotedTweet.from_api(response.get("data", [{}])[0])

    # Analytics

    async def get_campaign_analytics(
        self,
        campaign_ids: list[str],
        start_date: date,
        end_date: date,
        granularity: str = "TOTAL",
    ) -> list[CampaignAnalytics]:
        """Get analytics for campaigns."""
        params = {
            "entity": "CAMPAIGN",
            "entity_ids": ",".join(campaign_ids),
            "start_time": f"{start_date.isoformat()}T00:00:00Z",
            "end_time": f"{end_date.isoformat()}T23:59:59Z",
            "granularity": granularity,
            "metric_groups": "ENGAGEMENT,BILLING,VIDEO,MOBILE_CONVERSION",
        }

        response = await self._request("GET", "stats/accounts/" + self.credentials.ads_account_id, params=params)

        results = []
        for item in response.get("data", []):
            campaign_id = item.get("id", "")
            results.append(
                CampaignAnalytics.from_api(item, campaign_id, start_date, end_date)
            )

        return results

    # Audiences

    async def get_tailored_audiences(self) -> list[TailoredAudience]:
        """Get tailored audiences for the account."""
        response = await self._request("GET", "tailored_audiences")
        return [TailoredAudience.from_api(ta) for ta in response.get("data", [])]

    async def create_tailored_audience(
        self,
        name: str,
        audience_type: str = "CRM",
    ) -> TailoredAudience:
        """Create a tailored audience."""
        data = {
            "name": name,
            "audience_type": audience_type,
        }

        response = await self._request("POST", "tailored_audiences", json_data=data)
        return TailoredAudience.from_api(response.get("data", {}))

    # Funding

    async def get_funding_instruments(self) -> list[FundingInstrument]:
        """Get funding instruments for the account."""
        response = await self._request("GET", "funding_instruments")
        return [FundingInstrument.from_api(fi) for fi in response.get("data", [])]


def get_mock_campaign() -> Campaign:
    """Get mock campaign for testing."""
    return Campaign(
        id="abc123",
        name="Test Twitter Campaign",
        account_id="123456789",
        status=CampaignStatus.ACTIVE,
        objective=CampaignObjective.WEBSITE_CLICKS,
        funding_instrument_id="fi_123",
        daily_budget=100.0,
        created_at=datetime.now(),
    )


def get_mock_analytics() -> CampaignAnalytics:
    """Get mock analytics for testing."""
    return CampaignAnalytics(
        campaign_id="abc123",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        impressions=100000,
        engagements=5000,
        clicks=2500,
        spend=500.0,
        retweets=200,
        replies=100,
        likes=2000,
        follows=50,
        engagement_rate=5.0,
        cpe=0.10,
        cpc=0.20,
    )
