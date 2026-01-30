"""
Comprehensive tests for Microsoft Advertising (Bing Ads) Connector.

Tests for Microsoft Advertising API integration including:
- Client initialization and token management
- Campaign operations (CRUD, status updates, budget management)
- Ad group operations
- Ad operations (responsive search ads)
- Keyword management
- Negative keyword management
- Audience operations
- Conversion goal management
- Reporting and performance metrics
- Error handling
"""

import pytest
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def microsoft_ads_credentials():
    """Create test credentials."""
    from aragora.connectors.advertising.microsoft_ads import MicrosoftAdsCredentials

    return MicrosoftAdsCredentials(
        developer_token="test_dev_token",
        client_id="test_client_id",
        client_secret="test_client_secret",
        refresh_token="test_refresh_token",
        account_id="123456789",
        customer_id="987654321",
        access_token="test_access_token",
        token_expires_at=datetime.now() + timedelta(hours=1),
    )


@pytest.fixture
def microsoft_ads_connector(microsoft_ads_credentials):
    """Create test connector."""
    from aragora.connectors.advertising.microsoft_ads import MicrosoftAdsConnector

    return MicrosoftAdsConnector(microsoft_ads_credentials)


@pytest.fixture
def mock_campaign_response():
    """Mock campaign API response."""
    return {
        "Id": "12345678901",
        "Name": "Test Bing Campaign",
        "Status": "Active",
        "CampaignType": "Search",
        "BudgetType": "DailyBudgetStandard",
        "DailyBudget": 50.0,
        "BiddingScheme": {"Type": "EnhancedCpc", "TargetCpa": None},
        "TimeZone": "PacificTimeUSCanadaTijuana",
        "Languages": {"string": ["English"]},
        "TrackingUrlTemplate": "https://track.example.com?id={campaignid}",
        "StartDate": {"Date": "2024-01-01"},
        "EndDate": {"Date": "2024-12-31"},
    }


@pytest.fixture
def mock_ad_group_response():
    """Mock ad group API response."""
    return {
        "Id": "987654321",
        "CampaignId": "12345678901",
        "Name": "Test Ad Group",
        "Status": "Active",
        "CpcBid": {"Amount": 1.50},
        "StartDate": {"Date": "2024-01-01"},
        "EndDate": {"Date": "2024-12-31"},
        "TrackingUrlTemplate": None,
        "AdRotation": {"Type": "OptimizeForClicks"},
    }


@pytest.fixture
def mock_ad_response():
    """Mock responsive search ad API response."""
    return {
        "Id": "111222333",
        "AdGroupId": "987654321",
        "Status": "Active",
        "Headlines": {
            "AssetLink": [
                {"Asset": {"Text": "Headline 1"}},
                {"Asset": {"Text": "Headline 2"}},
                {"Asset": {"Text": "Headline 3"}},
            ]
        },
        "Descriptions": {
            "AssetLink": [
                {"Asset": {"Text": "Description 1"}},
                {"Asset": {"Text": "Description 2"}},
            ]
        },
        "Path1": "products",
        "Path2": "deals",
        "FinalUrls": {"string": ["https://example.com"]},
        "FinalMobileUrls": {"string": []},
        "TrackingUrlTemplate": None,
    }


@pytest.fixture
def mock_keyword_response():
    """Mock keyword API response."""
    return {
        "Id": "555666777",
        "AdGroupId": "987654321",
        "Text": "test keyword",
        "MatchType": "Exact",
        "Status": "Active",
        "Bid": {"Amount": 0.75},
        "FinalUrls": {"string": ["https://example.com/keyword"]},
        "DestinationUrl": None,
    }


@pytest.fixture
def mock_performance_response():
    """Mock campaign performance report response."""
    return {
        "CampaignId": "12345678901",
        "CampaignName": "Test Bing Campaign",
        "TimePeriod": "2024-01-15",
        "Impressions": "10000",
        "Clicks": "250",
        "Spend": "125.00",
        "Conversions": "15",
        "Revenue": "450.00",
        "AveragePosition": "2.3",
        "QualityScore": "7",
    }


@pytest.fixture
def mock_audience_response():
    """Mock audience list API response."""
    return {
        "Id": "aud_123456",
        "Name": "Website Visitors",
        "Type": "RemarketingList",
        "Description": "Visitors in past 30 days",
        "MembershipDuration": 30,
        "Scope": "Account",
        "SearchSize": 25000,
    }


@pytest.fixture
def mock_conversion_goal_response():
    """Mock conversion goal API response."""
    return {
        "Id": "goal_123",
        "Name": "Purchase",
        "Type": "Event",
        "Status": "Active",
        "Revenue": {"Type": "FixedValue", "Value": 50.0},
        "ConversionWindowInMinutes": 43200,  # 30 days
    }


# =============================================================================
# Credentials Tests
# =============================================================================


class TestMicrosoftAdsCredentials:
    """Tests for MicrosoftAdsCredentials dataclass."""

    def test_credentials_creation(self):
        """Test credentials creation with required fields."""
        from aragora.connectors.advertising.microsoft_ads import MicrosoftAdsCredentials

        creds = MicrosoftAdsCredentials(
            developer_token="dev_token",
            client_id="client_id",
            client_secret="client_secret",
            refresh_token="refresh_token",
            account_id="123456789",
            customer_id="987654321",
        )

        assert creds.developer_token == "dev_token"
        assert creds.client_id == "client_id"
        assert creds.client_secret == "client_secret"
        assert creds.refresh_token == "refresh_token"
        assert creds.account_id == "123456789"
        assert creds.customer_id == "987654321"
        assert creds.access_token is None
        assert creds.token_expires_at is None

    def test_credentials_with_token(self):
        """Test credentials with access token."""
        from aragora.connectors.advertising.microsoft_ads import MicrosoftAdsCredentials

        expires = datetime.now() + timedelta(hours=1)
        creds = MicrosoftAdsCredentials(
            developer_token="dev_token",
            client_id="client_id",
            client_secret="client_secret",
            refresh_token="refresh_token",
            account_id="123456789",
            customer_id="987654321",
            access_token="access_token",
            token_expires_at=expires,
        )

        assert creds.access_token == "access_token"
        assert creds.token_expires_at == expires


# =============================================================================
# Connector Initialization Tests
# =============================================================================


class TestMicrosoftAdsConnectorInit:
    """Tests for MicrosoftAdsConnector initialization."""

    def test_connector_initialization(self, microsoft_ads_credentials):
        """Test connector initializes correctly."""
        from aragora.connectors.advertising.microsoft_ads import MicrosoftAdsConnector

        connector = MicrosoftAdsConnector(microsoft_ads_credentials)

        assert connector.credentials == microsoft_ads_credentials
        assert connector._client is None
        assert "campaign.api.bingads.microsoft.com" in connector.API_URL

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self, microsoft_ads_connector):
        """Test _get_client creates HTTP client."""
        client = await microsoft_ads_connector._get_client()

        assert client is not None
        assert isinstance(client, httpx.AsyncClient)

        await microsoft_ads_connector.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_client(self, microsoft_ads_connector):
        """Test _get_client reuses existing client."""
        client1 = await microsoft_ads_connector._get_client()
        client2 = await microsoft_ads_connector._get_client()

        assert client1 is client2

        await microsoft_ads_connector.close()

    @pytest.mark.asyncio
    async def test_close_closes_client(self, microsoft_ads_connector):
        """Test close() closes HTTP client."""
        await microsoft_ads_connector._get_client()
        assert microsoft_ads_connector._client is not None

        await microsoft_ads_connector.close()
        assert microsoft_ads_connector._client is None


# =============================================================================
# Token Management Tests
# =============================================================================


class TestTokenManagement:
    """Tests for OAuth token management."""

    @pytest.mark.asyncio
    async def test_ensure_token_with_valid_token(self, microsoft_ads_connector):
        """Test _ensure_token with valid existing token."""
        # Token is already set in fixture with future expiry
        await microsoft_ads_connector._ensure_token()

        # Should not have changed
        assert microsoft_ads_connector.credentials.access_token == "test_access_token"

        await microsoft_ads_connector.close()

    @pytest.mark.asyncio
    async def test_ensure_token_refreshes_expired_token(self, microsoft_ads_credentials):
        """Test _ensure_token refreshes expired token."""
        from aragora.connectors.advertising.microsoft_ads import MicrosoftAdsConnector

        # Set expired token
        microsoft_ads_credentials.access_token = None
        microsoft_ads_credentials.token_expires_at = None
        connector = MicrosoftAdsConnector(microsoft_ads_credentials)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "access_token": "new_access_token",
                "expires_in": 3600,
            }
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await connector._ensure_token()

            assert connector.credentials.access_token == "new_access_token"

        await connector.close()

    @pytest.mark.asyncio
    async def test_refresh_token_failure(self, microsoft_ads_credentials):
        """Test _refresh_token handles failure."""
        from aragora.connectors.advertising.microsoft_ads import (
            MicrosoftAdsConnector,
            MicrosoftAdsError,
        )

        microsoft_ads_credentials.access_token = None
        connector = MicrosoftAdsConnector(microsoft_ads_credentials)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "Invalid refresh token"
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(MicrosoftAdsError, match="Token refresh failed"):
                await connector._refresh_token()

        await connector.close()


# =============================================================================
# Campaign Dataclass Tests
# =============================================================================


class TestCampaignDataclass:
    """Tests for Campaign dataclass."""

    def test_campaign_from_api(self, mock_campaign_response):
        """Test Campaign.from_api parsing."""
        from aragora.connectors.advertising.microsoft_ads import (
            Campaign,
            CampaignStatus,
            CampaignType,
            BudgetType,
            BiddingScheme,
        )

        campaign = Campaign.from_api(mock_campaign_response)

        assert campaign.id == "12345678901"
        assert campaign.name == "Test Bing Campaign"
        assert campaign.status == CampaignStatus.ACTIVE
        assert campaign.campaign_type == CampaignType.SEARCH
        assert campaign.budget_type == BudgetType.DAILY_BUDGET_STANDARD
        assert campaign.daily_budget == 50.0
        assert campaign.bidding_scheme == BiddingScheme.ENHANCED_CPC
        assert campaign.time_zone == "PacificTimeUSCanadaTijuana"
        assert "English" in campaign.languages
        assert campaign.start_date == date(2024, 1, 1)
        assert campaign.end_date == date(2024, 12, 31)

    def test_campaign_from_api_minimal(self):
        """Test Campaign.from_api with minimal data."""
        from aragora.connectors.advertising.microsoft_ads import Campaign

        data = {
            "Id": "123",
            "Name": "Minimal Campaign",
            "Status": "Paused",
            "CampaignType": "Search",
            "BudgetType": "DailyBudgetStandard",
            "DailyBudget": 0,
            "BiddingScheme": {},
        }

        campaign = Campaign.from_api(data)

        assert campaign.id == "123"
        assert campaign.name == "Minimal Campaign"
        assert campaign.start_date is None
        assert campaign.end_date is None


# =============================================================================
# Campaign Operations Tests
# =============================================================================


class TestCampaignOperations:
    """Tests for campaign operations."""

    @pytest.mark.asyncio
    async def test_get_campaigns(self, microsoft_ads_connector, mock_campaign_response):
        """Test get_campaigns returns list of campaigns."""
        with patch.object(microsoft_ads_connector, "_soap_request") as mock_request:
            mock_request.return_value = {
                "GetCampaignsByAccountIdResponse": {
                    "Campaigns": {"Campaign": [mock_campaign_response]}
                }
            }

            campaigns = await microsoft_ads_connector.get_campaigns()

            assert len(campaigns) == 1
            assert campaigns[0].id == "12345678901"

    @pytest.mark.asyncio
    async def test_get_campaigns_single_result(
        self, microsoft_ads_connector, mock_campaign_response
    ):
        """Test get_campaigns handles single result (dict instead of list)."""
        with patch.object(microsoft_ads_connector, "_soap_request") as mock_request:
            mock_request.return_value = {
                "GetCampaignsByAccountIdResponse": {
                    "Campaigns": {"Campaign": mock_campaign_response}
                }
            }

            campaigns = await microsoft_ads_connector.get_campaigns()

            assert len(campaigns) == 1

    @pytest.mark.asyncio
    async def test_get_campaigns_with_type_filter(self, microsoft_ads_connector):
        """Test get_campaigns with campaign type filter."""
        from aragora.connectors.advertising.microsoft_ads import CampaignType

        with patch.object(microsoft_ads_connector, "_soap_request") as mock_request:
            mock_request.return_value = {
                "GetCampaignsByAccountIdResponse": {"Campaigns": {"Campaign": []}}
            }

            await microsoft_ads_connector.get_campaigns(campaign_type=CampaignType.SEARCH)

            call_args = mock_request.call_args[0]
            assert "Search" in call_args[1]

    @pytest.mark.asyncio
    async def test_get_campaign_by_id(self, microsoft_ads_connector, mock_campaign_response):
        """Test get_campaign returns single campaign."""
        with patch.object(microsoft_ads_connector, "_soap_request") as mock_request:
            mock_request.return_value = {
                "GetCampaignsByIdsResponse": {"Campaigns": {"Campaign": [mock_campaign_response]}}
            }

            campaign = await microsoft_ads_connector.get_campaign("12345678901")

            assert campaign.id == "12345678901"

    @pytest.mark.asyncio
    async def test_get_campaign_not_found(self, microsoft_ads_connector):
        """Test get_campaign raises error when not found."""
        from aragora.connectors.advertising.microsoft_ads import MicrosoftAdsError

        with patch.object(microsoft_ads_connector, "_soap_request") as mock_request:
            mock_request.return_value = {
                "GetCampaignsByIdsResponse": {"Campaigns": {"Campaign": []}}
            }

            with pytest.raises(MicrosoftAdsError, match="not found"):
                await microsoft_ads_connector.get_campaign("nonexistent")

    @pytest.mark.asyncio
    async def test_create_campaign(self, microsoft_ads_connector):
        """Test create_campaign."""
        from aragora.connectors.advertising.microsoft_ads import CampaignType

        with patch.object(microsoft_ads_connector, "_soap_request") as mock_request:
            mock_request.return_value = {
                "AddCampaignsResponse": {"CampaignIds": {"long": ["12345678901"]}}
            }

            campaign_id = await microsoft_ads_connector.create_campaign(
                name="New Campaign",
                campaign_type=CampaignType.SEARCH,
                daily_budget=50.0,
            )

            assert campaign_id == "12345678901"

    @pytest.mark.asyncio
    async def test_update_campaign_status(self, microsoft_ads_connector):
        """Test update_campaign_status."""
        from aragora.connectors.advertising.microsoft_ads import CampaignStatus

        with patch.object(microsoft_ads_connector, "_soap_request") as mock_request:
            mock_request.return_value = {}

            await microsoft_ads_connector.update_campaign_status(
                "12345678901", CampaignStatus.PAUSED
            )

            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_campaign_budget(self, microsoft_ads_connector):
        """Test update_campaign_budget."""
        with patch.object(microsoft_ads_connector, "_soap_request") as mock_request:
            mock_request.return_value = {}

            await microsoft_ads_connector.update_campaign_budget("12345678901", 100.0)

            mock_request.assert_called_once()
            call_args = mock_request.call_args[0]
            assert "100.0" in call_args[1]


# =============================================================================
# Ad Group Tests
# =============================================================================


class TestAdGroupDataclass:
    """Tests for AdGroup dataclass."""

    def test_ad_group_from_api(self, mock_ad_group_response):
        """Test AdGroup.from_api parsing."""
        from aragora.connectors.advertising.microsoft_ads import AdGroup, AdGroupStatus

        ad_group = AdGroup.from_api(mock_ad_group_response)

        assert ad_group.id == "987654321"
        assert ad_group.campaign_id == "12345678901"
        assert ad_group.name == "Test Ad Group"
        assert ad_group.status == AdGroupStatus.ACTIVE
        assert ad_group.cpc_bid == 1.50


class TestAdGroupOperations:
    """Tests for ad group operations."""

    @pytest.mark.asyncio
    async def test_get_ad_groups(self, microsoft_ads_connector, mock_ad_group_response):
        """Test get_ad_groups returns list of ad groups."""
        with patch.object(microsoft_ads_connector, "_soap_request") as mock_request:
            mock_request.return_value = {
                "GetAdGroupsByCampaignIdResponse": {
                    "AdGroups": {"AdGroup": [mock_ad_group_response]}
                }
            }

            ad_groups = await microsoft_ads_connector.get_ad_groups("12345678901")

            assert len(ad_groups) == 1
            assert ad_groups[0].id == "987654321"

    @pytest.mark.asyncio
    async def test_create_ad_group(self, microsoft_ads_connector):
        """Test create_ad_group."""
        with patch.object(microsoft_ads_connector, "_soap_request") as mock_request:
            mock_request.return_value = {
                "AddAdGroupsResponse": {"AdGroupIds": {"long": ["987654321"]}}
            }

            ad_group_id = await microsoft_ads_connector.create_ad_group(
                campaign_id="12345678901",
                name="New Ad Group",
                cpc_bid=1.50,
            )

            assert ad_group_id == "987654321"


# =============================================================================
# Ad Tests
# =============================================================================


class TestResponsiveSearchAdDataclass:
    """Tests for ResponsiveSearchAd dataclass."""

    def test_responsive_search_ad_from_api(self, mock_ad_response):
        """Test ResponsiveSearchAd.from_api parsing."""
        from aragora.connectors.advertising.microsoft_ads import ResponsiveSearchAd, AdStatus

        ad = ResponsiveSearchAd.from_api(mock_ad_response)

        assert ad.id == "111222333"
        assert ad.ad_group_id == "987654321"
        assert ad.status == AdStatus.ACTIVE
        assert len(ad.headlines) == 3
        assert len(ad.descriptions) == 2
        assert ad.path1 == "products"
        assert ad.path2 == "deals"
        assert "https://example.com" in ad.final_urls


class TestAdOperations:
    """Tests for ad operations."""

    @pytest.mark.asyncio
    async def test_get_ads(self, microsoft_ads_connector, mock_ad_response):
        """Test get_ads returns list of ads."""
        with patch.object(microsoft_ads_connector, "_soap_request") as mock_request:
            mock_request.return_value = {
                "GetAdsByAdGroupIdResponse": {"Ads": {"Ad": [mock_ad_response]}}
            }

            ads = await microsoft_ads_connector.get_ads("987654321")

            assert len(ads) == 1
            assert ads[0].id == "111222333"

    @pytest.mark.asyncio
    async def test_create_responsive_search_ad(self, microsoft_ads_connector):
        """Test create_responsive_search_ad."""
        with patch.object(microsoft_ads_connector, "_soap_request") as mock_request:
            mock_request.return_value = {"AddAdsResponse": {"AdIds": {"long": ["111222333"]}}}

            ad_id = await microsoft_ads_connector.create_responsive_search_ad(
                ad_group_id="987654321",
                headlines=["Headline 1", "Headline 2", "Headline 3"],
                descriptions=["Description 1", "Description 2"],
                final_urls=["https://example.com"],
                path1="products",
                path2="deals",
            )

            assert ad_id == "111222333"


# =============================================================================
# Keyword Tests
# =============================================================================


class TestKeywordDataclass:
    """Tests for Keyword dataclass."""

    def test_keyword_from_api(self, mock_keyword_response):
        """Test Keyword.from_api parsing."""
        from aragora.connectors.advertising.microsoft_ads import (
            Keyword,
            KeywordMatchType,
            KeywordStatus,
        )

        keyword = Keyword.from_api(mock_keyword_response)

        assert keyword.id == "555666777"
        assert keyword.ad_group_id == "987654321"
        assert keyword.text == "test keyword"
        assert keyword.match_type == KeywordMatchType.EXACT
        assert keyword.status == KeywordStatus.ACTIVE
        assert keyword.bid == 0.75


class TestKeywordOperations:
    """Tests for keyword operations."""

    @pytest.mark.asyncio
    async def test_get_keywords(self, microsoft_ads_connector, mock_keyword_response):
        """Test get_keywords returns list of keywords."""
        with patch.object(microsoft_ads_connector, "_soap_request") as mock_request:
            mock_request.return_value = {
                "GetKeywordsByAdGroupIdResponse": {"Keywords": {"Keyword": [mock_keyword_response]}}
            }

            keywords = await microsoft_ads_connector.get_keywords("987654321")

            assert len(keywords) == 1
            assert keywords[0].text == "test keyword"

    @pytest.mark.asyncio
    async def test_add_keywords(self, microsoft_ads_connector):
        """Test add_keywords."""
        from aragora.connectors.advertising.microsoft_ads import KeywordMatchType

        with patch.object(microsoft_ads_connector, "_soap_request") as mock_request:
            mock_request.return_value = {
                "AddKeywordsResponse": {"KeywordIds": {"long": ["555666777", "555666778"]}}
            }

            keyword_ids = await microsoft_ads_connector.add_keywords(
                ad_group_id="987654321",
                keywords=[
                    ("keyword one", KeywordMatchType.EXACT, 0.75),
                    ("keyword two", KeywordMatchType.PHRASE, 0.50),
                ],
            )

            assert len(keyword_ids) == 2

    @pytest.mark.asyncio
    async def test_update_keyword_bids(self, microsoft_ads_connector):
        """Test update_keyword_bids."""
        with patch.object(microsoft_ads_connector, "_soap_request") as mock_request:
            mock_request.return_value = {}

            await microsoft_ads_connector.update_keyword_bids(
                ad_group_id="987654321",
                keyword_bids=[("555666777", 1.00), ("555666778", 0.80)],
            )

            mock_request.assert_called_once()


# =============================================================================
# Negative Keyword Tests
# =============================================================================


class TestNegativeKeywordDataclass:
    """Tests for NegativeKeyword dataclass."""

    def test_negative_keyword_from_api(self):
        """Test NegativeKeyword.from_api parsing."""
        from aragora.connectors.advertising.microsoft_ads import (
            NegativeKeyword,
            KeywordMatchType,
        )

        data = {
            "Id": "neg_123",
            "Text": "free",
            "MatchType": "Phrase",
        }

        negative_keyword = NegativeKeyword.from_api(data)

        assert negative_keyword.id == "neg_123"
        assert negative_keyword.text == "free"
        assert negative_keyword.match_type == KeywordMatchType.PHRASE


class TestNegativeKeywordOperations:
    """Tests for negative keyword operations."""

    @pytest.mark.asyncio
    async def test_add_negative_keywords(self, microsoft_ads_connector):
        """Test add_negative_keywords."""
        from aragora.connectors.advertising.microsoft_ads import KeywordMatchType

        with patch.object(microsoft_ads_connector, "_soap_request") as mock_request:
            mock_request.return_value = {
                "AddNegativeKeywordsToCampaignsResponse": {
                    "NegativeKeywordIds": {"long": ["neg_123"]}
                }
            }

            ids = await microsoft_ads_connector.add_negative_keywords(
                campaign_id="12345678901",
                keywords=[
                    ("free", KeywordMatchType.PHRASE),
                    ("cheap", KeywordMatchType.EXACT),
                ],
            )

            assert len(ids) == 1


# =============================================================================
# Audience Tests
# =============================================================================


class TestAudienceListDataclass:
    """Tests for AudienceList dataclass."""

    def test_audience_list_from_api(self, mock_audience_response):
        """Test AudienceList.from_api parsing."""
        from aragora.connectors.advertising.microsoft_ads import AudienceList

        audience = AudienceList.from_api(mock_audience_response)

        assert audience.id == "aud_123456"
        assert audience.name == "Website Visitors"
        assert audience.audience_type == "RemarketingList"
        assert audience.membership_duration == 30
        assert audience.scope == "Account"
        assert audience.audience_size == 25000


class TestAudienceOperations:
    """Tests for audience operations."""

    @pytest.mark.asyncio
    async def test_get_audiences(self, microsoft_ads_connector, mock_audience_response):
        """Test get_audiences."""
        with patch.object(microsoft_ads_connector, "_soap_request") as mock_request:
            mock_request.return_value = {
                "GetAudiencesByIdsResponse": {"Audiences": {"Audience": [mock_audience_response]}}
            }

            audiences = await microsoft_ads_connector.get_audiences()

            assert len(audiences) == 1
            assert audiences[0].name == "Website Visitors"


# =============================================================================
# Conversion Goal Tests
# =============================================================================


class TestConversionGoalDataclass:
    """Tests for ConversionGoal dataclass."""

    def test_conversion_goal_from_api(self, mock_conversion_goal_response):
        """Test ConversionGoal.from_api parsing."""
        from aragora.connectors.advertising.microsoft_ads import ConversionGoal

        goal = ConversionGoal.from_api(mock_conversion_goal_response)

        assert goal.id == "goal_123"
        assert goal.name == "Purchase"
        assert goal.goal_type == "Event"
        assert goal.status == "Active"
        assert goal.revenue_type == "FixedValue"
        assert goal.revenue_value == 50.0
        assert goal.conversion_window == 30  # 43200 / 1440


class TestConversionGoalOperations:
    """Tests for conversion goal operations."""

    @pytest.mark.asyncio
    async def test_get_conversion_goals(
        self, microsoft_ads_connector, mock_conversion_goal_response
    ):
        """Test get_conversion_goals."""
        with patch.object(microsoft_ads_connector, "_soap_request") as mock_request:
            mock_request.return_value = {
                "GetConversionGoalsByIdsResponse": {
                    "ConversionGoals": {"ConversionGoal": [mock_conversion_goal_response]}
                }
            }

            goals = await microsoft_ads_connector.get_conversion_goals()

            assert len(goals) == 1
            assert goals[0].name == "Purchase"


# =============================================================================
# Reporting Tests
# =============================================================================


class TestCampaignPerformanceDataclass:
    """Tests for CampaignPerformance dataclass."""

    def test_campaign_performance_from_api(self, mock_performance_response):
        """Test CampaignPerformance.from_api parsing."""
        from aragora.connectors.advertising.microsoft_ads import CampaignPerformance

        perf = CampaignPerformance.from_api(mock_performance_response)

        assert perf.campaign_id == "12345678901"
        assert perf.campaign_name == "Test Bing Campaign"
        assert perf.impressions == 10000
        assert perf.clicks == 250
        assert perf.spend == 125.0
        assert perf.conversions == 15
        assert perf.conversion_value == 450.0
        assert perf.ctr == 2.5  # 250/10000 * 100
        assert perf.average_cpc == 0.5  # 125/250
        assert perf.quality_score == 7

    def test_campaign_performance_zero_impressions(self):
        """Test CampaignPerformance with zero impressions."""
        from aragora.connectors.advertising.microsoft_ads import CampaignPerformance

        data = {
            "CampaignId": "123",
            "CampaignName": "Test",
            "Impressions": "0",
            "Clicks": "0",
            "Spend": "0",
        }

        perf = CampaignPerformance.from_api(data)

        assert perf.ctr == 0.0
        assert perf.average_cpc == 0.0


class TestReportingOperations:
    """Tests for reporting operations."""

    @pytest.mark.asyncio
    async def test_get_campaign_performance(self, microsoft_ads_connector):
        """Test get_campaign_performance."""
        with patch.object(microsoft_ads_connector, "_soap_request") as mock_request:
            mock_request.return_value = {}

            result = await microsoft_ads_connector.get_campaign_performance(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
            )

            # Returns empty list in simplified implementation
            assert result == []

    @pytest.mark.asyncio
    async def test_get_campaign_performance_with_ids(self, microsoft_ads_connector):
        """Test get_campaign_performance with campaign IDs filter."""
        with patch.object(microsoft_ads_connector, "_soap_request") as mock_request:
            mock_request.return_value = {}

            await microsoft_ads_connector.get_campaign_performance(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                campaign_ids=["123", "456"],
            )

            call_args = mock_request.call_args[0]
            assert "123" in call_args[1]


# =============================================================================
# Google Ads Import Tests
# =============================================================================


class TestGoogleAdsImport:
    """Tests for Google Ads import functionality."""

    @pytest.mark.asyncio
    async def test_import_google_ads_campaigns(self, microsoft_ads_connector):
        """Test import_google_ads_campaigns."""
        result = await microsoft_ads_connector.import_google_ads_campaigns(
            google_account_id="google_123",
            campaign_ids=["camp_1", "camp_2"],
        )

        assert result["status"] == "pending"
        assert result["google_account_id"] == "google_123"
        assert result["campaign_ids"] == ["camp_1", "camp_2"]


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestMicrosoftAdsError:
    """Tests for MicrosoftAdsError exception."""

    def test_error_creation(self):
        """Test MicrosoftAdsError creation."""
        from aragora.connectors.advertising.microsoft_ads import MicrosoftAdsError

        error = MicrosoftAdsError(
            message="Invalid operation",
            error_code="1001",
            operation_errors=[{"Code": "100", "Message": "Detailed error"}],
        )

        assert str(error) == "Invalid operation"
        assert error.error_code == "1001"
        assert len(error.operation_errors) == 1

    def test_error_minimal(self):
        """Test MicrosoftAdsError with minimal info."""
        from aragora.connectors.advertising.microsoft_ads import MicrosoftAdsError

        error = MicrosoftAdsError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.error_code is None
        assert error.operation_errors == []


class TestAPIErrorHandling:
    """Tests for API error handling."""

    @pytest.mark.asyncio
    async def test_soap_request_handles_http_error(self, microsoft_ads_connector):
        """Test _soap_request handles HTTP errors."""
        from aragora.connectors.advertising.microsoft_ads import MicrosoftAdsError

        with patch.object(microsoft_ads_connector, "_ensure_token"):
            with patch.object(microsoft_ads_connector, "_get_client") as mock_get_client:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.status_code = 500
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_get_client.return_value = mock_client

                with pytest.raises(MicrosoftAdsError, match="API error: 500"):
                    await microsoft_ads_connector._soap_request("TestOperation", "<body/>")


# =============================================================================
# SOAP Envelope Tests
# =============================================================================


class TestSOAPEnvelope:
    """Tests for SOAP envelope building."""

    def test_build_soap_envelope(self, microsoft_ads_connector):
        """Test _build_soap_envelope."""
        envelope = microsoft_ads_connector._build_soap_envelope(
            operation="GetCampaigns",
            body="<AccountId>123</AccountId>",
            service="CampaignManagement",
        )

        assert "<?xml version" in envelope
        assert "soap:Envelope" in envelope or "s:Envelope" in envelope
        assert "GetCampaignsRequest" in envelope
        assert "test_access_token" in envelope
        assert "123456789" in envelope  # account_id
        assert "987654321" in envelope  # customer_id
        assert "test_dev_token" in envelope


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_campaign_status_values(self):
        """Test CampaignStatus enum values."""
        from aragora.connectors.advertising.microsoft_ads import CampaignStatus

        assert CampaignStatus.ACTIVE.value == "Active"
        assert CampaignStatus.PAUSED.value == "Paused"
        assert CampaignStatus.DELETED.value == "Deleted"
        assert CampaignStatus.SUSPENDED.value == "Suspended"

    def test_campaign_type_values(self):
        """Test CampaignType enum values."""
        from aragora.connectors.advertising.microsoft_ads import CampaignType

        assert CampaignType.SEARCH.value == "Search"
        assert CampaignType.SHOPPING.value == "Shopping"
        assert CampaignType.AUDIENCE.value == "Audience"
        assert CampaignType.PERFORMANCE_MAX.value == "PerformanceMax"

    def test_bidding_scheme_values(self):
        """Test BiddingScheme enum values."""
        from aragora.connectors.advertising.microsoft_ads import BiddingScheme

        assert BiddingScheme.ENHANCED_CPC.value == "EnhancedCpc"
        assert BiddingScheme.MANUAL_CPC.value == "ManualCpc"
        assert BiddingScheme.MAXIMIZE_CONVERSIONS.value == "MaxConversions"
        assert BiddingScheme.TARGET_CPA.value == "TargetCpa"

    def test_keyword_match_type_values(self):
        """Test KeywordMatchType enum values."""
        from aragora.connectors.advertising.microsoft_ads import KeywordMatchType

        assert KeywordMatchType.BROAD.value == "Broad"
        assert KeywordMatchType.EXACT.value == "Exact"
        assert KeywordMatchType.PHRASE.value == "Phrase"


# =============================================================================
# Mock Data Tests
# =============================================================================


class TestMockData:
    """Tests for mock data generation."""

    def test_get_mock_campaign(self):
        """Test get_mock_campaign."""
        from aragora.connectors.advertising.microsoft_ads import get_mock_campaign

        campaign = get_mock_campaign()

        assert campaign.id == "123456789"
        assert campaign.name == "Test Bing Campaign"
        assert campaign.daily_budget == 50.0
        assert "English" in campaign.languages

    def test_get_mock_performance(self):
        """Test get_mock_performance."""
        from aragora.connectors.advertising.microsoft_ads import get_mock_performance

        perf = get_mock_performance()

        assert perf.campaign_id == "123456789"
        assert perf.impressions == 10000
        assert perf.clicks == 250
        assert perf.spend == 125.0
        assert perf.quality_score == 7
