"""
Microsoft Advertising (Bing Ads) Connector.

Integration with Microsoft Advertising API for Bing/Microsoft search ads:
- Campaign and ad group management
- Keyword targeting
- Audience targeting (remarketing, in-market)
- Conversion tracking
- Performance reporting
- Import from Google Ads
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any

import httpx


class CampaignStatus(Enum):
    """Microsoft Ads campaign status."""

    ACTIVE = "Active"
    PAUSED = "Paused"
    BUDGET_PAUSED = "BudgetPaused"
    BUDGET_AND_MANUAL_PAUSED = "BudgetAndManualPaused"
    DELETED = "Deleted"
    SUSPENDED = "Suspended"


class CampaignType(Enum):
    """Microsoft Ads campaign types."""

    SEARCH = "Search"
    SHOPPING = "Shopping"
    AUDIENCE = "Audience"
    DYNAMIC_SEARCH_ADS = "DynamicSearchAds"
    PERFORMANCE_MAX = "PerformanceMax"
    HOTEL = "Hotel"


class BudgetType(Enum):
    """Budget types."""

    DAILY_BUDGET_ACCELERATED = "DailyBudgetAccelerated"
    DAILY_BUDGET_STANDARD = "DailyBudgetStandard"
    MONTHLY_BUDGET_SPEND_UNTIL_DEPLETED = "MonthlyBudgetSpendUntilDepleted"


class BiddingScheme(Enum):
    """Bidding strategies."""

    ENHANCED_CPC = "EnhancedCpc"
    MANUAL_CPC = "ManualCpc"
    MANUAL_CPM = "ManualCpm"
    MANUAL_CPV = "ManualCpv"
    MAXIMIZE_CLICKS = "MaxClicks"
    MAXIMIZE_CONVERSIONS = "MaxConversions"
    MAXIMIZE_CONVERSION_VALUE = "MaxConversionValue"
    TARGET_CPA = "TargetCpa"
    TARGET_ROAS = "TargetRoas"
    TARGET_IMPRESSION_SHARE = "TargetImpressionShare"


class AdGroupStatus(Enum):
    """Ad group status."""

    ACTIVE = "Active"
    PAUSED = "Paused"
    DELETED = "Deleted"
    EXPIRED = "Expired"


class AdStatus(Enum):
    """Ad status."""

    ACTIVE = "Active"
    PAUSED = "Paused"
    DELETED = "Deleted"
    INACTIVE = "Inactive"


class AdType(Enum):
    """Ad types."""

    TEXT = "Text"
    EXPANDED_TEXT = "ExpandedText"
    RESPONSIVE_SEARCH = "ResponsiveSearch"
    DYNAMIC_SEARCH = "DynamicSearch"
    PRODUCT = "Product"
    RESPONSIVE = "Responsive"
    APP_INSTALL = "AppInstall"


class KeywordMatchType(Enum):
    """Keyword match types."""

    BROAD = "Broad"
    EXACT = "Exact"
    PHRASE = "Phrase"


class KeywordStatus(Enum):
    """Keyword status."""

    ACTIVE = "Active"
    PAUSED = "Paused"
    DELETED = "Deleted"
    INACTIVE = "Inactive"


@dataclass
class MicrosoftAdsCredentials:
    """Microsoft Advertising API credentials."""

    developer_token: str
    client_id: str
    client_secret: str
    refresh_token: str
    account_id: str
    customer_id: str
    access_token: str | None = None
    token_expires_at: datetime | None = None


@dataclass
class Campaign:
    """Microsoft Ads campaign."""

    id: str
    name: str
    status: CampaignStatus
    campaign_type: CampaignType
    budget_type: BudgetType
    daily_budget: float
    bidding_scheme: BiddingScheme
    target_cpa: float | None = None
    target_roas: float | None = None
    time_zone: str = "PacificTimeUSCanadaTijuana"
    languages: list[str] = field(default_factory=list)
    tracking_template: str | None = None
    start_date: date | None = None
    end_date: date | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "Campaign":
        """Create from Microsoft Ads API response."""
        return cls(
            id=str(data.get("Id", "")),
            name=data.get("Name", ""),
            status=CampaignStatus(data.get("Status", "Paused")),
            campaign_type=CampaignType(data.get("CampaignType", "Search")),
            budget_type=BudgetType(data.get("BudgetType", "DailyBudgetStandard")),
            daily_budget=float(data.get("DailyBudget", 0)),
            bidding_scheme=BiddingScheme(data.get("BiddingScheme", {}).get("Type", "EnhancedCpc")),
            target_cpa=data.get("BiddingScheme", {}).get("TargetCpa"),
            target_roas=data.get("BiddingScheme", {}).get("TargetRoas"),
            time_zone=data.get("TimeZone", "PacificTimeUSCanadaTijuana"),
            languages=data.get("Languages", {}).get("string", []),
            tracking_template=data.get("TrackingUrlTemplate"),
            start_date=(
                date.fromisoformat(data["StartDate"]["Date"]) if data.get("StartDate") else None
            ),
            end_date=date.fromisoformat(data["EndDate"]["Date"]) if data.get("EndDate") else None,
        )


@dataclass
class AdGroup:
    """Microsoft Ads ad group."""

    id: str
    campaign_id: str
    name: str
    status: AdGroupStatus
    cpc_bid: float | None = None
    start_date: date | None = None
    end_date: date | None = None
    tracking_template: str | None = None
    ad_rotation: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "AdGroup":
        """Create from Microsoft Ads API response."""
        return cls(
            id=str(data.get("Id", "")),
            campaign_id=str(data.get("CampaignId", "")),
            name=data.get("Name", ""),
            status=AdGroupStatus(data.get("Status", "Paused")),
            cpc_bid=data.get("CpcBid", {}).get("Amount"),
            start_date=(
                date.fromisoformat(data["StartDate"]["Date"]) if data.get("StartDate") else None
            ),
            end_date=date.fromisoformat(data["EndDate"]["Date"]) if data.get("EndDate") else None,
            tracking_template=data.get("TrackingUrlTemplate"),
            ad_rotation=data.get("AdRotation", {}).get("Type"),
        )


@dataclass
class ResponsiveSearchAd:
    """Microsoft Ads responsive search ad."""

    id: str
    ad_group_id: str
    status: AdStatus
    headlines: list[dict[str, Any]]  # List of {Text, PinnedField}
    descriptions: list[dict[str, Any]]  # List of {Text, PinnedField}
    path1: str | None = None
    path2: str | None = None
    final_urls: list[str] = field(default_factory=list)
    final_mobile_urls: list[str] = field(default_factory=list)
    tracking_template: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "ResponsiveSearchAd":
        """Create from Microsoft Ads API response."""
        return cls(
            id=str(data.get("Id", "")),
            ad_group_id=str(data.get("AdGroupId", "")),
            status=AdStatus(data.get("Status", "Paused")),
            headlines=data.get("Headlines", {}).get("AssetLink", []),
            descriptions=data.get("Descriptions", {}).get("AssetLink", []),
            path1=data.get("Path1"),
            path2=data.get("Path2"),
            final_urls=data.get("FinalUrls", {}).get("string", []),
            final_mobile_urls=data.get("FinalMobileUrls", {}).get("string", []),
            tracking_template=data.get("TrackingUrlTemplate"),
        )


@dataclass
class Keyword:
    """Microsoft Ads keyword."""

    id: str
    ad_group_id: str
    text: str
    match_type: KeywordMatchType
    status: KeywordStatus
    bid: float | None = None
    final_urls: list[str] = field(default_factory=list)
    destination_url: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "Keyword":
        """Create from Microsoft Ads API response."""
        return cls(
            id=str(data.get("Id", "")),
            ad_group_id=str(data.get("AdGroupId", "")),
            text=data.get("Text", ""),
            match_type=KeywordMatchType(data.get("MatchType", "Broad")),
            status=KeywordStatus(data.get("Status", "Paused")),
            bid=data.get("Bid", {}).get("Amount"),
            final_urls=data.get("FinalUrls", {}).get("string", []),
            destination_url=data.get("DestinationUrl"),
        )


@dataclass
class CampaignPerformance:
    """Microsoft Ads campaign performance report."""

    campaign_id: str
    campaign_name: str
    date: date
    impressions: int = 0
    clicks: int = 0
    spend: float = 0.0
    conversions: int = 0
    conversion_value: float = 0.0
    ctr: float = 0.0
    average_cpc: float = 0.0
    average_position: float = 0.0
    quality_score: int | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "CampaignPerformance":
        """Create from Microsoft Ads reporting response."""
        impressions = int(data.get("Impressions", 0))
        clicks = int(data.get("Clicks", 0))
        spend = float(data.get("Spend", 0))

        return cls(
            campaign_id=str(data.get("CampaignId", "")),
            campaign_name=data.get("CampaignName", ""),
            date=date.fromisoformat(data["TimePeriod"]) if data.get("TimePeriod") else date.today(),
            impressions=impressions,
            clicks=clicks,
            spend=spend,
            conversions=int(data.get("Conversions", 0)),
            conversion_value=float(data.get("Revenue", 0)),
            ctr=(clicks / impressions * 100) if impressions > 0 else 0.0,
            average_cpc=(spend / clicks) if clicks > 0 else 0.0,
            average_position=float(data.get("AveragePosition", 0)),
            quality_score=int(data.get("QualityScore")) if data.get("QualityScore") else None,
        )


@dataclass
class AudienceList:
    """Microsoft Ads audience list (remarketing, custom, in-market)."""

    id: str
    name: str
    audience_type: str
    description: str | None = None
    membership_duration: int | None = None
    scope: str | None = None  # Account, Customer
    audience_size: int | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "AudienceList":
        """Create from Microsoft Ads API response."""
        return cls(
            id=str(data.get("Id", "")),
            name=data.get("Name", ""),
            audience_type=data.get("Type", ""),
            description=data.get("Description"),
            membership_duration=data.get("MembershipDuration"),
            scope=data.get("Scope"),
            audience_size=data.get("SearchSize"),
        )


@dataclass
class ConversionGoal:
    """Microsoft Ads conversion goal."""

    id: str
    name: str
    goal_type: str
    status: str
    revenue_type: str | None = None
    revenue_value: float | None = None
    conversion_window: int | None = None  # in days

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "ConversionGoal":
        """Create from Microsoft Ads API response."""
        return cls(
            id=str(data.get("Id", "")),
            name=data.get("Name", ""),
            goal_type=data.get("Type", ""),
            status=data.get("Status", ""),
            revenue_type=data.get("Revenue", {}).get("Type"),
            revenue_value=data.get("Revenue", {}).get("Value"),
            conversion_window=(
                data.get("ConversionWindowInMinutes", 0) // 1440
                if data.get("ConversionWindowInMinutes")
                else None
            ),
        )


@dataclass
class NegativeKeyword:
    """Microsoft Ads negative keyword."""

    id: str
    text: str
    match_type: KeywordMatchType

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "NegativeKeyword":
        """Create from Microsoft Ads API response."""
        return cls(
            id=str(data.get("Id", "")),
            text=data.get("Text", ""),
            match_type=KeywordMatchType(data.get("MatchType", "Broad")),
        )


class MicrosoftAdsError(Exception):
    """Microsoft Ads API error."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        operation_errors: list[dict[str, Any]] | None = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.operation_errors = operation_errors or []


class MicrosoftAdsConnector:
    """Microsoft Advertising API connector."""

    AUTH_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
    API_URL = "https://campaign.api.bingads.microsoft.com/Api/Advertiser/CampaignManagement/v13/CampaignManagementService.svc"
    BULK_URL = "https://bulk.api.bingads.microsoft.com/Api/Advertiser/CampaignManagement/v13/BulkService.svc"
    REPORTING_URL = "https://reporting.api.bingads.microsoft.com/Api/Advertiser/Reporting/v13/ReportingService.svc"

    def __init__(self, credentials: MicrosoftAdsCredentials):
        """Initialize with credentials."""
        self.credentials = credentials
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _refresh_token(self) -> None:
        """Refresh OAuth2 access token."""
        client = await self._get_client()

        response = await client.post(
            self.AUTH_URL,
            data={
                "client_id": self.credentials.client_id,
                "client_secret": self.credentials.client_secret,
                "refresh_token": self.credentials.refresh_token,
                "grant_type": "refresh_token",
                "scope": "https://ads.microsoft.com/msads.manage offline_access",
            },
        )

        if response.status_code != 200:
            raise MicrosoftAdsError(f"Token refresh failed: {response.text}")

        data = response.json()
        self.credentials.access_token = data["access_token"]
        self.credentials.token_expires_at = datetime.now()

    async def _ensure_token(self) -> None:
        """Ensure we have a valid access token."""
        if not self.credentials.access_token:
            await self._refresh_token()
        elif self.credentials.token_expires_at:
            # Refresh if token is close to expiry (within 5 minutes)
            from datetime import timedelta

            if datetime.now() >= self.credentials.token_expires_at - timedelta(minutes=5):
                await self._refresh_token()

    def _build_soap_envelope(
        self,
        operation: str,
        body: str,
        service: str = "CampaignManagement",
    ) -> str:
        """Build SOAP envelope for Microsoft Ads API."""
        return f"""<?xml version="1.0" encoding="utf-8"?>
<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">
  <s:Header>
    <AuthenticationToken xmlns="https://bingads.microsoft.com/CampaignManagement/v13">{self.credentials.access_token}</AuthenticationToken>
    <CustomerAccountId xmlns="https://bingads.microsoft.com/CampaignManagement/v13">{self.credentials.account_id}</CustomerAccountId>
    <CustomerId xmlns="https://bingads.microsoft.com/CampaignManagement/v13">{self.credentials.customer_id}</CustomerId>
    <DeveloperToken xmlns="https://bingads.microsoft.com/CampaignManagement/v13">{self.credentials.developer_token}</DeveloperToken>
  </s:Header>
  <s:Body>
    <{operation}Request xmlns="https://bingads.microsoft.com/{service}/v13">
      {body}
    </{operation}Request>
  </s:Body>
</s:Envelope>"""

    async def _soap_request(
        self,
        operation: str,
        body: str,
        service: str = "CampaignManagement",
    ) -> dict[str, Any]:
        """Make SOAP request to Microsoft Ads API."""
        await self._ensure_token()
        client = await self._get_client()

        # Select appropriate endpoint
        if service == "Reporting":
            url = self.REPORTING_URL
        elif service == "Bulk":
            url = self.BULK_URL
        else:
            url = self.API_URL

        envelope = self._build_soap_envelope(operation, body, service)

        response = await client.post(
            url,
            content=envelope,
            headers={
                "Content-Type": "text/xml; charset=utf-8",
                "SOAPAction": f"https://bingads.microsoft.com/{service}/v13/I{service}Service/{operation}",
            },
        )

        if response.status_code >= 400:
            raise MicrosoftAdsError(
                message=f"API error: {response.status_code}",
                error_code=str(response.status_code),
            )

        # Parse SOAP response (simplified - in production use proper XML parsing)
        import xml.etree.ElementTree as ET

        root = ET.fromstring(response.text)

        # Find response body
        ns = {
            "s": "http://schemas.xmlsoap.org/soap/envelope/",
            "cm": f"https://bingads.microsoft.com/{service}/v13",
        }

        body_elem = root.find(".//s:Body", ns)
        if body_elem is None:
            return {}

        # Convert XML to dict (simplified)
        return self._xml_to_dict(body_elem)

    def _xml_to_dict(self, elem: Any) -> dict[str, Any]:
        """Convert XML element to dictionary (simplified)."""
        result: dict[str, Any] = {}
        for child in elem:
            tag = child.tag.split("}")[-1]  # Remove namespace
            if len(child) > 0:
                result[tag] = self._xml_to_dict(child)
            else:
                result[tag] = child.text
        return result

    # Campaign Operations

    async def get_campaigns(
        self,
        campaign_type: CampaignType | None = None,
    ) -> list[Campaign]:
        """Get all campaigns for the account."""
        body = "<AccountId>{}</AccountId>".format(self.credentials.account_id)
        if campaign_type:
            body += f"<CampaignType>{campaign_type.value}</CampaignType>"

        response = await self._soap_request("GetCampaignsByAccountId", body)

        campaigns = (
            response.get("GetCampaignsByAccountIdResponse", {})
            .get("Campaigns", {})
            .get("Campaign", [])
        )
        if isinstance(campaigns, dict):
            campaigns = [campaigns]

        return [Campaign.from_api(c) for c in campaigns]

    async def get_campaign(self, campaign_id: str) -> Campaign:
        """Get a specific campaign by ID."""
        body = f"""
        <AccountId>{self.credentials.account_id}</AccountId>
        <CampaignIds>
            <long>{campaign_id}</long>
        </CampaignIds>
        """

        response = await self._soap_request("GetCampaignsByIds", body)

        campaigns = (
            response.get("GetCampaignsByIdsResponse", {}).get("Campaigns", {}).get("Campaign", [])
        )
        if isinstance(campaigns, dict):
            campaigns = [campaigns]

        if not campaigns:
            raise MicrosoftAdsError(f"Campaign {campaign_id} not found")

        return Campaign.from_api(campaigns[0])

    async def create_campaign(
        self,
        name: str,
        campaign_type: CampaignType,
        daily_budget: float,
        bidding_scheme: BiddingScheme = BiddingScheme.ENHANCED_CPC,
        status: CampaignStatus = CampaignStatus.PAUSED,
        time_zone: str = "PacificTimeUSCanadaTijuana",
    ) -> str:
        """Create a new campaign. Returns campaign ID."""
        body = f"""
        <AccountId>{self.credentials.account_id}</AccountId>
        <Campaigns>
            <Campaign>
                <Name>{name}</Name>
                <CampaignType>{campaign_type.value}</CampaignType>
                <DailyBudget>{daily_budget}</DailyBudget>
                <BudgetType>DailyBudgetStandard</BudgetType>
                <Status>{status.value}</Status>
                <TimeZone>{time_zone}</TimeZone>
                <BiddingScheme>
                    <Type>{bidding_scheme.value}</Type>
                </BiddingScheme>
            </Campaign>
        </Campaigns>
        """

        response = await self._soap_request("AddCampaigns", body)

        campaign_ids = (
            response.get("AddCampaignsResponse", {}).get("CampaignIds", {}).get("long", [])
        )
        if isinstance(campaign_ids, str):
            return campaign_ids
        return campaign_ids[0] if campaign_ids else ""

    async def update_campaign_status(
        self,
        campaign_id: str,
        status: CampaignStatus,
    ) -> None:
        """Update campaign status."""
        body = f"""
        <AccountId>{self.credentials.account_id}</AccountId>
        <Campaigns>
            <Campaign>
                <Id>{campaign_id}</Id>
                <Status>{status.value}</Status>
            </Campaign>
        </Campaigns>
        """

        await self._soap_request("UpdateCampaigns", body)

    async def update_campaign_budget(
        self,
        campaign_id: str,
        daily_budget: float,
    ) -> None:
        """Update campaign daily budget."""
        body = f"""
        <AccountId>{self.credentials.account_id}</AccountId>
        <Campaigns>
            <Campaign>
                <Id>{campaign_id}</Id>
                <DailyBudget>{daily_budget}</DailyBudget>
            </Campaign>
        </Campaigns>
        """

        await self._soap_request("UpdateCampaigns", body)

    # Ad Group Operations

    async def get_ad_groups(self, campaign_id: str) -> list[AdGroup]:
        """Get ad groups for a campaign."""
        body = f"<CampaignId>{campaign_id}</CampaignId>"

        response = await self._soap_request("GetAdGroupsByCampaignId", body)

        ad_groups = (
            response.get("GetAdGroupsByCampaignIdResponse", {})
            .get("AdGroups", {})
            .get("AdGroup", [])
        )
        if isinstance(ad_groups, dict):
            ad_groups = [ad_groups]

        return [AdGroup.from_api(ag) for ag in ad_groups]

    async def create_ad_group(
        self,
        campaign_id: str,
        name: str,
        cpc_bid: float,
        status: AdGroupStatus = AdGroupStatus.PAUSED,
    ) -> str:
        """Create a new ad group. Returns ad group ID."""
        body = f"""
        <CampaignId>{campaign_id}</CampaignId>
        <AdGroups>
            <AdGroup>
                <Name>{name}</Name>
                <Status>{status.value}</Status>
                <CpcBid>
                    <Amount>{cpc_bid}</Amount>
                </CpcBid>
            </AdGroup>
        </AdGroups>
        """

        response = await self._soap_request("AddAdGroups", body)

        ad_group_ids = response.get("AddAdGroupsResponse", {}).get("AdGroupIds", {}).get("long", [])
        if isinstance(ad_group_ids, str):
            return ad_group_ids
        return ad_group_ids[0] if ad_group_ids else ""

    # Ad Operations

    async def get_ads(self, ad_group_id: str) -> list[ResponsiveSearchAd]:
        """Get ads for an ad group."""
        body = f"""
        <AdGroupId>{ad_group_id}</AdGroupId>
        <AdTypes>
            <AdType>ResponsiveSearch</AdType>
        </AdTypes>
        """

        response = await self._soap_request("GetAdsByAdGroupId", body)

        ads = response.get("GetAdsByAdGroupIdResponse", {}).get("Ads", {}).get("Ad", [])
        if isinstance(ads, dict):
            ads = [ads]

        return [ResponsiveSearchAd.from_api(ad) for ad in ads]

    async def create_responsive_search_ad(
        self,
        ad_group_id: str,
        headlines: list[str],
        descriptions: list[str],
        final_urls: list[str],
        path1: str | None = None,
        path2: str | None = None,
    ) -> str:
        """Create a responsive search ad. Returns ad ID."""
        headlines_xml = "".join(
            [f"<AssetLink><Asset><Text>{h}</Text></Asset></AssetLink>" for h in headlines]
        )
        descriptions_xml = "".join(
            [f"<AssetLink><Asset><Text>{d}</Text></Asset></AssetLink>" for d in descriptions]
        )
        urls_xml = "".join([f"<string>{url}</string>" for url in final_urls])

        body = f"""
        <AdGroupId>{ad_group_id}</AdGroupId>
        <Ads>
            <Ad xsi:type="ResponsiveSearchAd" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                <FinalUrls>{urls_xml}</FinalUrls>
                <Headlines>{headlines_xml}</Headlines>
                <Descriptions>{descriptions_xml}</Descriptions>
                {f"<Path1>{path1}</Path1>" if path1 else ""}
                {f"<Path2>{path2}</Path2>" if path2 else ""}
            </Ad>
        </Ads>
        """

        response = await self._soap_request("AddAds", body)

        ad_ids = response.get("AddAdsResponse", {}).get("AdIds", {}).get("long", [])
        if isinstance(ad_ids, str):
            return ad_ids
        return ad_ids[0] if ad_ids else ""

    # Keyword Operations

    async def get_keywords(self, ad_group_id: str) -> list[Keyword]:
        """Get keywords for an ad group."""
        body = f"<AdGroupId>{ad_group_id}</AdGroupId>"

        response = await self._soap_request("GetKeywordsByAdGroupId", body)

        keywords = (
            response.get("GetKeywordsByAdGroupIdResponse", {})
            .get("Keywords", {})
            .get("Keyword", [])
        )
        if isinstance(keywords, dict):
            keywords = [keywords]

        return [Keyword.from_api(kw) for kw in keywords]

    async def add_keywords(
        self,
        ad_group_id: str,
        keywords: list[tuple[str, KeywordMatchType, float | None]],
    ) -> list[str]:
        """Add keywords to an ad group. Returns keyword IDs."""
        keywords_xml = ""
        for text, match_type, bid in keywords:
            bid_xml = f"<Bid><Amount>{bid}</Amount></Bid>" if bid else ""
            keywords_xml += f"""
            <Keyword>
                <Text>{text}</Text>
                <MatchType>{match_type.value}</MatchType>
                {bid_xml}
            </Keyword>
            """

        body = f"""
        <AdGroupId>{ad_group_id}</AdGroupId>
        <Keywords>{keywords_xml}</Keywords>
        """

        response = await self._soap_request("AddKeywords", body)

        keyword_ids = response.get("AddKeywordsResponse", {}).get("KeywordIds", {}).get("long", [])
        if isinstance(keyword_ids, str):
            return [keyword_ids]
        return keyword_ids if keyword_ids else []

    async def update_keyword_bids(
        self,
        ad_group_id: str,
        keyword_bids: list[tuple[str, float]],
    ) -> None:
        """Update keyword bids."""
        keywords_xml = ""
        for keyword_id, bid in keyword_bids:
            keywords_xml += f"""
            <Keyword>
                <Id>{keyword_id}</Id>
                <Bid><Amount>{bid}</Amount></Bid>
            </Keyword>
            """

        body = f"""
        <AdGroupId>{ad_group_id}</AdGroupId>
        <Keywords>{keywords_xml}</Keywords>
        """

        await self._soap_request("UpdateKeywords", body)

    # Negative Keywords

    async def add_negative_keywords(
        self,
        campaign_id: str,
        keywords: list[tuple[str, KeywordMatchType]],
    ) -> list[str]:
        """Add negative keywords to a campaign."""
        keywords_xml = ""
        for text, match_type in keywords:
            keywords_xml += f"""
            <NegativeKeyword>
                <Text>{text}</Text>
                <MatchType>{match_type.value}</MatchType>
            </NegativeKeyword>
            """

        body = f"""
        <CampaignId>{campaign_id}</CampaignId>
        <NegativeKeywords>{keywords_xml}</NegativeKeywords>
        """

        response = await self._soap_request("AddNegativeKeywordsToCampaigns", body)

        ids = (
            response.get("AddNegativeKeywordsToCampaignsResponse", {})
            .get("NegativeKeywordIds", {})
            .get("long", [])
        )
        if isinstance(ids, str):
            return [ids]
        return ids if ids else []

    # Audience Operations

    async def get_audiences(self) -> list[AudienceList]:
        """Get audience lists for the account."""
        body = f"""
        <Type>RemarketingList</Type>
        <CustomerId>{self.credentials.customer_id}</CustomerId>
        """

        response = await self._soap_request("GetAudiencesByIds", body)

        audiences = (
            response.get("GetAudiencesByIdsResponse", {}).get("Audiences", {}).get("Audience", [])
        )
        if isinstance(audiences, dict):
            audiences = [audiences]

        return [AudienceList.from_api(a) for a in audiences]

    # Conversion Goals

    async def get_conversion_goals(self) -> list[ConversionGoal]:
        """Get conversion goals for the account."""
        body = "<ConversionGoalTypes>Url Event Duration PagesViewedPerVisit InStoreTransaction</ConversionGoalTypes>"

        response = await self._soap_request("GetConversionGoalsByIds", body)

        goals = (
            response.get("GetConversionGoalsByIdsResponse", {})
            .get("ConversionGoals", {})
            .get("ConversionGoal", [])
        )
        if isinstance(goals, dict):
            goals = [goals]

        return [ConversionGoal.from_api(g) for g in goals]

    # Reporting

    async def get_campaign_performance(
        self,
        start_date: date,
        end_date: date,
        campaign_ids: list[str] | None = None,
    ) -> list[CampaignPerformance]:
        """Get campaign performance report."""
        campaign_filter = ""
        if campaign_ids:
            ids_xml = "".join([f"<long>{cid}</long>" for cid in campaign_ids])
            campaign_filter = f"<CampaignIds>{ids_xml}</CampaignIds>"

        body = f"""
        <ReportRequest xsi:type="CampaignPerformanceReportRequest" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            <Format>Xml</Format>
            <ReportName>CampaignPerformance</ReportName>
            <Time>
                <CustomDateRangeStart>
                    <Day>{start_date.day}</Day>
                    <Month>{start_date.month}</Month>
                    <Year>{start_date.year}</Year>
                </CustomDateRangeStart>
                <CustomDateRangeEnd>
                    <Day>{end_date.day}</Day>
                    <Month>{end_date.month}</Month>
                    <Year>{end_date.year}</Year>
                </CustomDateRangeEnd>
            </Time>
            <Scope>
                <AccountIds>
                    <long>{self.credentials.account_id}</long>
                </AccountIds>
                {campaign_filter}
            </Scope>
            <Columns>
                <CampaignPerformanceReportColumn>CampaignId</CampaignPerformanceReportColumn>
                <CampaignPerformanceReportColumn>CampaignName</CampaignPerformanceReportColumn>
                <CampaignPerformanceReportColumn>Impressions</CampaignPerformanceReportColumn>
                <CampaignPerformanceReportColumn>Clicks</CampaignPerformanceReportColumn>
                <CampaignPerformanceReportColumn>Spend</CampaignPerformanceReportColumn>
                <CampaignPerformanceReportColumn>Conversions</CampaignPerformanceReportColumn>
                <CampaignPerformanceReportColumn>Revenue</CampaignPerformanceReportColumn>
            </Columns>
        </ReportRequest>
        """

        await self._soap_request("SubmitGenerateReport", body, service="Reporting")

        # In production, would poll for report completion and download
        # Simplified version returns empty list
        return []

    # Import from Google Ads

    async def import_google_ads_campaigns(
        self,
        google_account_id: str,
        campaign_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Import campaigns from Google Ads.

        Note: This is a placeholder. Actual implementation requires
        Google Ads export and careful mapping of settings.
        """
        return {
            "status": "pending",
            "message": "Google Ads import initiated",
            "google_account_id": google_account_id,
            "campaign_ids": campaign_ids,
        }


def get_mock_campaign() -> Campaign:
    """Get mock campaign for testing."""
    return Campaign(
        id="123456789",
        name="Test Bing Campaign",
        status=CampaignStatus.ACTIVE,
        campaign_type=CampaignType.SEARCH,
        budget_type=BudgetType.DAILY_BUDGET_STANDARD,
        daily_budget=50.0,
        bidding_scheme=BiddingScheme.ENHANCED_CPC,
        time_zone="PacificTimeUSCanadaTijuana",
        languages=["English"],
        start_date=date(2024, 1, 1),
    )


def get_mock_performance() -> CampaignPerformance:
    """Get mock performance report for testing."""
    return CampaignPerformance(
        campaign_id="123456789",
        campaign_name="Test Bing Campaign",
        date=date(2024, 1, 15),
        impressions=10000,
        clicks=250,
        spend=125.0,
        conversions=15,
        conversion_value=450.0,
        ctr=2.5,
        average_cpc=0.50,
        average_position=2.3,
        quality_score=7,
    )
