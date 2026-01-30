"""
Comprehensive tests for Google Ads Connector.

Tests for Google Ads API integration including:
- Client initialization and token management
- Campaign operations (CRUD, status updates, budget management)
- Ad group operations
- Ad operations
- Keyword management
- Reporting and metrics
- Error handling
"""

import pytest
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def google_ads_credentials():
    """Create test credentials."""
    from aragora.connectors.advertising.google_ads import GoogleAdsCredentials

    return GoogleAdsCredentials(
        developer_token="test_dev_token",
        client_id="test_client_id",
        client_secret="test_client_secret",
        refresh_token="test_refresh_token",
        customer_id="1234567890",
        login_customer_id="0987654321",
        access_token="test_access_token",
    )


@pytest.fixture
def google_ads_connector(google_ads_credentials):
    """Create test connector."""
    from aragora.connectors.advertising.google_ads import GoogleAdsConnector

    return GoogleAdsConnector(google_ads_credentials)


@pytest.fixture
def mock_campaign_response():
    """Mock campaign API response."""
    return {
        "campaign": {
            "id": "12345678901",
            "name": "Test Campaign",
            "status": "ENABLED",
            "advertisingChannelType": "SEARCH",
            "resourceName": "customers/1234567890/campaigns/12345678901",
            "startDate": "2024-01-01",
            "endDate": "2024-12-31",
            "biddingStrategyType": "TARGET_CPA",
            "campaignBudget": {
                "amountMicros": "50000000",
            },
        },
    }


@pytest.fixture
def mock_ad_group_response():
    """Mock ad group API response."""
    return {
        "adGroup": {
            "id": "987654321",
            "name": "Test Ad Group",
            "campaign": "customers/1234567890/campaigns/12345678901",
            "status": "ENABLED",
            "cpcBidMicros": "1000000",
            "resourceName": "customers/1234567890/adGroups/987654321",
        }
    }


@pytest.fixture
def mock_ad_response():
    """Mock ad API response."""
    return {
        "adGroupAd": {
            "adGroup": "customers/1234567890/adGroups/987654321",
            "status": "ENABLED",
            "resourceName": "customers/1234567890/adGroupAds/987654321~111222333",
            "ad": {
                "id": "111222333",
                "type": "RESPONSIVE_SEARCH_AD",
                "finalUrls": ["https://example.com"],
                "displayUrl": "example.com",
                "responsiveSearchAd": {
                    "headlines": [
                        {"text": "Headline 1"},
                        {"text": "Headline 2"},
                        {"text": "Headline 3"},
                    ],
                    "descriptions": [
                        {"text": "Description 1"},
                        {"text": "Description 2"},
                    ],
                },
            },
        }
    }


@pytest.fixture
def mock_keyword_response():
    """Mock keyword API response."""
    return {
        "adGroupCriterion": {
            "criterionId": "555666777",
            "adGroup": "customers/1234567890/adGroups/987654321",
            "status": "ENABLED",
            "cpcBidMicros": "500000",
            "resourceName": "customers/1234567890/adGroupCriteria/987654321~555666777",
            "keyword": {
                "text": "test keyword",
                "matchType": "EXACT",
            },
            "qualityInfo": {
                "qualityScore": 8,
            },
        }
    }


# =============================================================================
# Client Initialization Tests
# =============================================================================


class TestGoogleAdsCredentials:
    """Tests for GoogleAdsCredentials dataclass."""

    def test_credentials_creation(self):
        """Test credentials creation with required fields."""
        from aragora.connectors.advertising.google_ads import GoogleAdsCredentials

        creds = GoogleAdsCredentials(
            developer_token="dev_token",
            client_id="client_id",
            client_secret="client_secret",
            refresh_token="refresh_token",
            customer_id="1234567890",
        )

        assert creds.developer_token == "dev_token"
        assert creds.client_id == "client_id"
        assert creds.client_secret == "client_secret"
        assert creds.refresh_token == "refresh_token"
        assert creds.customer_id == "1234567890"
        assert creds.login_customer_id is None
        assert creds.access_token is None
        assert creds.base_url == "https://googleads.googleapis.com/v15"

    def test_credentials_with_mcc(self):
        """Test credentials with MCC (manager) account."""
        from aragora.connectors.advertising.google_ads import GoogleAdsCredentials

        creds = GoogleAdsCredentials(
            developer_token="dev_token",
            client_id="client_id",
            client_secret="client_secret",
            refresh_token="refresh_token",
            customer_id="1234567890",
            login_customer_id="0987654321",
        )

        assert creds.login_customer_id == "0987654321"

    def test_credentials_with_custom_base_url(self):
        """Test credentials with custom API base URL."""
        from aragora.connectors.advertising.google_ads import GoogleAdsCredentials

        creds = GoogleAdsCredentials(
            developer_token="dev_token",
            client_id="client_id",
            client_secret="client_secret",
            refresh_token="refresh_token",
            customer_id="1234567890",
            base_url="https://googleads.googleapis.com/v16",
        )

        assert creds.base_url == "https://googleads.googleapis.com/v16"


class TestGoogleAdsConnectorInit:
    """Tests for GoogleAdsConnector initialization."""

    def test_connector_initialization(self, google_ads_credentials):
        """Test connector initializes correctly."""
        from aragora.connectors.advertising.google_ads import GoogleAdsConnector

        connector = GoogleAdsConnector(google_ads_credentials)

        assert connector.credentials == google_ads_credentials
        assert connector._client is None
        assert connector._access_token == google_ads_credentials.access_token

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self, google_ads_connector):
        """Test _get_client creates HTTP client."""
        client = await google_ads_connector._get_client()

        assert client is not None
        assert isinstance(client, httpx.AsyncClient)

        await google_ads_connector.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_client(self, google_ads_connector):
        """Test _get_client reuses existing client."""
        client1 = await google_ads_connector._get_client()
        client2 = await google_ads_connector._get_client()

        assert client1 is client2

        await google_ads_connector.close()

    def test_get_headers(self, google_ads_connector):
        """Test _get_headers returns correct headers."""
        headers = google_ads_connector._get_headers("test_token")

        assert headers["Authorization"] == "Bearer test_token"
        assert headers["developer-token"] == "test_dev_token"
        assert headers["Content-Type"] == "application/json"
        assert headers["login-customer-id"] == "0987654321"

    def test_get_headers_without_mcc(self, google_ads_credentials):
        """Test _get_headers without MCC account."""
        from aragora.connectors.advertising.google_ads import GoogleAdsConnector

        google_ads_credentials.login_customer_id = None
        connector = GoogleAdsConnector(google_ads_credentials)
        headers = connector._get_headers("test_token")

        assert "login-customer-id" not in headers


class TestTokenManagement:
    """Tests for OAuth token management."""

    @pytest.mark.asyncio
    async def test_ensure_token_returns_valid_token(self, google_ads_connector):
        """Test _ensure_token returns existing valid token."""
        google_ads_connector._token_expires_at = datetime.now() + timedelta(hours=1)

        token = await google_ads_connector._ensure_token()

        assert token == "test_access_token"

        await google_ads_connector.close()

    @pytest.mark.asyncio
    async def test_ensure_token_refreshes_expired_token(self, google_ads_connector):
        """Test _ensure_token refreshes expired token."""
        google_ads_connector._access_token = None
        google_ads_connector._token_expires_at = None

        with patch.object(google_ads_connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "access_token": "new_access_token",
                "expires_in": 3600,
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            token = await google_ads_connector._ensure_token()

            assert token == "new_access_token"
            assert google_ads_connector._access_token == "new_access_token"


# =============================================================================
# Campaign Operations Tests
# =============================================================================


class TestCampaignDataclass:
    """Tests for Campaign dataclass."""

    def test_campaign_from_api(self, mock_campaign_response):
        """Test Campaign.from_api parsing."""
        from aragora.connectors.advertising.google_ads import (
            Campaign,
            CampaignStatus,
            CampaignType,
            BiddingStrategyType,
        )

        campaign = Campaign.from_api(mock_campaign_response)

        assert campaign.id == "12345678901"
        assert campaign.name == "Test Campaign"
        assert campaign.status == CampaignStatus.ENABLED
        assert campaign.advertising_channel_type == CampaignType.SEARCH
        assert campaign.budget_amount_micros == 50000000
        assert campaign.start_date == date(2024, 1, 1)
        assert campaign.end_date == date(2024, 12, 31)
        assert campaign.bidding_strategy_type == BiddingStrategyType.TARGET_CPA

    def test_campaign_budget_amount_property(self):
        """Test Campaign.budget_amount property."""
        from aragora.connectors.advertising.google_ads import (
            Campaign,
            CampaignStatus,
            CampaignType,
        )

        campaign = Campaign(
            id="123",
            name="Test",
            status=CampaignStatus.ENABLED,
            advertising_channel_type=CampaignType.SEARCH,
            budget_amount_micros=50000000,
        )

        assert campaign.budget_amount == Decimal("50")

    def test_campaign_from_api_minimal(self):
        """Test Campaign.from_api with minimal data."""
        from aragora.connectors.advertising.google_ads import Campaign

        data = {
            "campaign": {
                "id": "123",
                "name": "Minimal Campaign",
            }
        }

        campaign = Campaign.from_api(data)

        assert campaign.id == "123"
        assert campaign.name == "Minimal Campaign"
        assert campaign.bidding_strategy_type is None


class TestCampaignOperations:
    """Tests for campaign CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_campaigns(self, google_ads_connector, mock_campaign_response):
        """Test get_campaigns returns list of campaigns."""
        with patch.object(google_ads_connector, "_search") as mock_search:
            mock_search.return_value = [mock_campaign_response]

            campaigns = await google_ads_connector.get_campaigns()

            assert len(campaigns) == 1
            assert campaigns[0].id == "12345678901"
            mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_campaigns_with_status_filter(self, google_ads_connector):
        """Test get_campaigns with status filter."""
        from aragora.connectors.advertising.google_ads import CampaignStatus

        with patch.object(google_ads_connector, "_search") as mock_search:
            mock_search.return_value = []

            await google_ads_connector.get_campaigns(status=CampaignStatus.ENABLED)

            call_args = mock_search.call_args[0][0]
            assert "campaign.status = 'ENABLED'" in call_args

    @pytest.mark.asyncio
    async def test_get_campaigns_with_type_filter(self, google_ads_connector):
        """Test get_campaigns with campaign type filter."""
        from aragora.connectors.advertising.google_ads import CampaignType

        with patch.object(google_ads_connector, "_search") as mock_search:
            mock_search.return_value = []

            await google_ads_connector.get_campaigns(campaign_type=CampaignType.DISPLAY)

            call_args = mock_search.call_args[0][0]
            assert "campaign.advertising_channel_type = 'DISPLAY'" in call_args

    @pytest.mark.asyncio
    async def test_get_campaign_by_id(self, google_ads_connector, mock_campaign_response):
        """Test get_campaign returns single campaign."""
        with patch.object(google_ads_connector, "_search") as mock_search:
            mock_search.return_value = [mock_campaign_response]

            campaign = await google_ads_connector.get_campaign("12345678901")

            assert campaign.id == "12345678901"

    @pytest.mark.asyncio
    async def test_get_campaign_not_found(self, google_ads_connector):
        """Test get_campaign raises error when not found."""
        from aragora.connectors.advertising.google_ads import GoogleAdsError

        with patch.object(google_ads_connector, "_search") as mock_search:
            mock_search.return_value = []

            with pytest.raises(GoogleAdsError, match="not found"):
                await google_ads_connector.get_campaign("nonexistent")

    @pytest.mark.asyncio
    async def test_update_campaign_status(self, google_ads_connector):
        """Test update_campaign_status."""
        from aragora.connectors.advertising.google_ads import CampaignStatus

        with patch.object(google_ads_connector, "_mutate") as mock_mutate:
            mock_mutate.return_value = []

            result = await google_ads_connector.update_campaign_status(
                "12345678901", CampaignStatus.PAUSED
            )

            assert result is True
            mock_mutate.assert_called_once()
            call_args = mock_mutate.call_args[0][0]
            assert call_args[0]["update"]["status"] == "PAUSED"

    @pytest.mark.asyncio
    async def test_update_campaign_budget(self, google_ads_connector):
        """Test update_campaign_budget."""
        with patch.object(google_ads_connector, "_search") as mock_search:
            mock_search.return_value = [
                {"campaign": {"campaignBudget": "customers/123/campaignBudgets/456"}}
            ]

            with patch.object(google_ads_connector, "_mutate") as mock_mutate:
                mock_mutate.return_value = []

                result = await google_ads_connector.update_campaign_budget("12345678901", 100000000)

                assert result is True


# =============================================================================
# Ad Group Operations Tests
# =============================================================================


class TestAdGroupDataclass:
    """Tests for AdGroup dataclass."""

    def test_ad_group_from_api(self, mock_ad_group_response):
        """Test AdGroup.from_api parsing."""
        from aragora.connectors.advertising.google_ads import AdGroup, AdGroupStatus

        ad_group = AdGroup.from_api(mock_ad_group_response)

        assert ad_group.id == "987654321"
        assert ad_group.name == "Test Ad Group"
        assert ad_group.campaign_id == "12345678901"
        assert ad_group.status == AdGroupStatus.ENABLED
        # Note: cpcBidMicros comes as string from API
        assert ad_group.cpc_bid_micros == "1000000"


class TestAdGroupOperations:
    """Tests for ad group operations."""

    @pytest.mark.asyncio
    async def test_get_ad_groups(self, google_ads_connector, mock_ad_group_response):
        """Test get_ad_groups returns list of ad groups."""
        with patch.object(google_ads_connector, "_search") as mock_search:
            mock_search.return_value = [mock_ad_group_response]

            ad_groups = await google_ads_connector.get_ad_groups()

            assert len(ad_groups) == 1
            assert ad_groups[0].id == "987654321"

    @pytest.mark.asyncio
    async def test_get_ad_groups_by_campaign(self, google_ads_connector):
        """Test get_ad_groups filtered by campaign."""
        with patch.object(google_ads_connector, "_search") as mock_search:
            mock_search.return_value = []

            await google_ads_connector.get_ad_groups(campaign_id="12345678901")

            call_args = mock_search.call_args[0][0]
            assert "campaigns/12345678901" in call_args

    @pytest.mark.asyncio
    async def test_update_ad_group_status(self, google_ads_connector):
        """Test update_ad_group_status."""
        from aragora.connectors.advertising.google_ads import AdGroupStatus

        with patch.object(google_ads_connector, "_mutate") as mock_mutate:
            mock_mutate.return_value = []

            result = await google_ads_connector.update_ad_group_status(
                "987654321", AdGroupStatus.PAUSED
            )

            assert result is True


# =============================================================================
# Ad Operations Tests
# =============================================================================


class TestAdDataclass:
    """Tests for Ad dataclass."""

    def test_ad_from_api(self, mock_ad_response):
        """Test Ad.from_api parsing."""
        from aragora.connectors.advertising.google_ads import Ad, AdStatus

        ad = Ad.from_api(mock_ad_response)

        assert ad.id == "111222333"
        assert ad.ad_group_id == "987654321"
        assert ad.status == AdStatus.ENABLED
        assert ad.type == "RESPONSIVE_SEARCH_AD"
        assert len(ad.headlines) == 3
        assert len(ad.descriptions) == 2
        assert ad.final_urls == ["https://example.com"]


class TestAdOperations:
    """Tests for ad operations."""

    @pytest.mark.asyncio
    async def test_get_ads(self, google_ads_connector, mock_ad_response):
        """Test get_ads returns list of ads."""
        with patch.object(google_ads_connector, "_search") as mock_search:
            mock_search.return_value = [mock_ad_response]

            ads = await google_ads_connector.get_ads()

            assert len(ads) == 1
            assert ads[0].id == "111222333"

    @pytest.mark.asyncio
    async def test_get_ads_by_ad_group(self, google_ads_connector):
        """Test get_ads filtered by ad group."""
        with patch.object(google_ads_connector, "_search") as mock_search:
            mock_search.return_value = []

            await google_ads_connector.get_ads(ad_group_id="987654321")

            call_args = mock_search.call_args[0][0]
            assert "adGroups/987654321" in call_args

    @pytest.mark.asyncio
    async def test_update_ad_status(self, google_ads_connector):
        """Test update_ad_status."""
        from aragora.connectors.advertising.google_ads import AdStatus

        with patch.object(google_ads_connector, "_mutate") as mock_mutate:
            mock_mutate.return_value = []

            result = await google_ads_connector.update_ad_status(
                "987654321", "111222333", AdStatus.PAUSED
            )

            assert result is True


# =============================================================================
# Keyword Operations Tests
# =============================================================================


class TestKeywordDataclass:
    """Tests for Keyword dataclass."""

    def test_keyword_from_api(self, mock_keyword_response):
        """Test Keyword.from_api parsing."""
        from aragora.connectors.advertising.google_ads import Keyword, KeywordMatchType

        keyword = Keyword.from_api(mock_keyword_response)

        assert keyword.id == "555666777"
        assert keyword.ad_group_id == "987654321"
        assert keyword.text == "test keyword"
        assert keyword.match_type == KeywordMatchType.EXACT
        assert keyword.quality_score == 8


class TestKeywordOperations:
    """Tests for keyword operations."""

    @pytest.mark.asyncio
    async def test_get_keywords(self, google_ads_connector, mock_keyword_response):
        """Test get_keywords returns list of keywords."""
        with patch.object(google_ads_connector, "_search") as mock_search:
            mock_search.return_value = [mock_keyword_response]

            keywords = await google_ads_connector.get_keywords("987654321")

            assert len(keywords) == 1
            assert keywords[0].text == "test keyword"

    @pytest.mark.asyncio
    async def test_add_keyword(self, google_ads_connector):
        """Test add_keyword creates a new keyword."""
        from aragora.connectors.advertising.google_ads import KeywordMatchType

        with patch.object(google_ads_connector, "_mutate") as mock_mutate:
            mock_mutate.return_value = [{"resourceName": "customers/123/adGroupCriteria/987~555"}]

            criterion_id = await google_ads_connector.add_keyword(
                ad_group_id="987654321",
                text="new keyword",
                match_type=KeywordMatchType.PHRASE,
                cpc_bid_micros=750000,
            )

            assert criterion_id == "555"

    @pytest.mark.asyncio
    async def test_update_keyword_bid(self, google_ads_connector):
        """Test update_keyword_bid."""
        with patch.object(google_ads_connector, "_mutate") as mock_mutate:
            mock_mutate.return_value = []

            result = await google_ads_connector.update_keyword_bid(
                "987654321", "555666777", 1000000
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_remove_keyword(self, google_ads_connector):
        """Test remove_keyword."""
        with patch.object(google_ads_connector, "_mutate") as mock_mutate:
            mock_mutate.return_value = []

            result = await google_ads_connector.remove_keyword("987654321", "555666777")

            assert result is True


# =============================================================================
# Reporting Tests
# =============================================================================


class TestCampaignMetricsDataclass:
    """Tests for CampaignMetrics dataclass."""

    def test_campaign_metrics_from_api(self):
        """Test CampaignMetrics.from_api parsing."""
        from aragora.connectors.advertising.google_ads import CampaignMetrics

        data = {
            "campaign": {"id": "123", "name": "Test Campaign"},
            "metrics": {
                "impressions": "10000",
                "clicks": "500",
                "costMicros": "25000000",
                "conversions": 50.0,
                "conversionsValue": 2500.0,
                "ctr": 0.05,
                "averageCpc": 50000,
                "averageCpm": 2500000,
                "conversionsFromInteractionsRate": 0.1,
                "costPerConversion": 500000,
            },
        }

        metrics = CampaignMetrics.from_api(data)

        assert metrics.impressions == 10000
        assert metrics.clicks == 500
        assert metrics.cost_micros == 25000000
        assert metrics.conversions == 50
        assert metrics.cost == Decimal("25")
        assert metrics.average_cpc == Decimal("0.05")

    def test_campaign_metrics_cost_property(self):
        """Test CampaignMetrics.cost property."""
        from aragora.connectors.advertising.google_ads import CampaignMetrics

        metrics = CampaignMetrics(
            campaign_id="123",
            campaign_name="Test",
            cost_micros=50000000,
        )

        assert metrics.cost == Decimal("50")


class TestSearchTermMetricsDataclass:
    """Tests for SearchTermMetrics dataclass."""

    def test_search_term_metrics_from_api(self):
        """Test SearchTermMetrics.from_api parsing."""
        from aragora.connectors.advertising.google_ads import SearchTermMetrics

        data = {
            "segments": {"searchTermView": {"searchTerm": "buy product"}},
            "campaign": {"id": "123"},
            "adGroup": {"id": "456"},
            "metrics": {
                "impressions": "1000",
                "clicks": "50",
                "costMicros": "5000000",
                "conversions": 5.0,
            },
        }

        metrics = SearchTermMetrics.from_api(data)

        assert metrics.search_term == "buy product"
        assert metrics.campaign_id == "123"
        assert metrics.impressions == 1000


class TestReportingOperations:
    """Tests for reporting operations."""

    @pytest.mark.asyncio
    async def test_get_campaign_performance(self, google_ads_connector):
        """Test get_campaign_performance."""
        with patch.object(google_ads_connector, "_search") as mock_search:
            mock_search.return_value = [
                {
                    "campaign": {"id": "123", "name": "Test"},
                    "metrics": {"impressions": "1000", "clicks": "50"},
                }
            ]

            metrics = await google_ads_connector.get_campaign_performance(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
            )

            assert len(metrics) == 1
            assert metrics[0].impressions == 1000

    @pytest.mark.asyncio
    async def test_get_campaign_performance_with_ids(self, google_ads_connector):
        """Test get_campaign_performance with campaign IDs filter."""
        with patch.object(google_ads_connector, "_search") as mock_search:
            mock_search.return_value = []

            await google_ads_connector.get_campaign_performance(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                campaign_ids=["123", "456"],
            )

            call_args = mock_search.call_args[0][0]
            assert "campaign.id IN (123, 456)" in call_args

    @pytest.mark.asyncio
    async def test_get_search_terms_report(self, google_ads_connector):
        """Test get_search_terms_report."""
        with patch.object(google_ads_connector, "_search") as mock_search:
            mock_search.return_value = []

            await google_ads_connector.get_search_terms_report(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                min_impressions=100,
            )

            call_args = mock_search.call_args[0][0]
            assert "metrics.impressions > 100" in call_args

    @pytest.mark.asyncio
    async def test_get_keyword_performance(self, google_ads_connector):
        """Test get_keyword_performance."""
        with patch.object(google_ads_connector, "_search") as mock_search:
            mock_search.return_value = []

            await google_ads_connector.get_keyword_performance(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                ad_group_id="987654321",
            )

            mock_search.assert_called_once()


# =============================================================================
# Account Info Tests
# =============================================================================


class TestAccountInfo:
    """Tests for account info operations."""

    @pytest.mark.asyncio
    async def test_get_account_info(self, google_ads_connector):
        """Test get_account_info."""
        with patch.object(google_ads_connector, "_search") as mock_search:
            mock_search.return_value = [
                {
                    "customer": {
                        "id": "1234567890",
                        "descriptiveName": "Test Account",
                        "currencyCode": "USD",
                        "timeZone": "America/New_York",
                        "manager": False,
                    }
                }
            ]

            info = await google_ads_connector.get_account_info()

            assert info["customer"]["id"] == "1234567890"

    @pytest.mark.asyncio
    async def test_get_account_info_empty(self, google_ads_connector):
        """Test get_account_info with empty result."""
        with patch.object(google_ads_connector, "_search") as mock_search:
            mock_search.return_value = []

            info = await google_ads_connector.get_account_info()

            assert info == {}


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestGoogleAdsError:
    """Tests for GoogleAdsError exception."""

    def test_error_creation(self):
        """Test GoogleAdsError creation."""
        from aragora.connectors.advertising.google_ads import GoogleAdsError

        error = GoogleAdsError(
            message="Authentication failed",
            errors=[{"errorCode": "AUTHENTICATION_ERROR"}],
            request_id="req_12345",
        )

        assert str(error) == "Authentication failed"
        assert len(error.errors) == 1
        assert error.request_id == "req_12345"

    def test_error_minimal(self):
        """Test GoogleAdsError with minimal info."""
        from aragora.connectors.advertising.google_ads import GoogleAdsError

        error = GoogleAdsError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.errors == []
        assert error.request_id is None


class TestAPIErrorHandling:
    """Tests for API error handling."""

    @pytest.mark.asyncio
    async def test_search_api_error(self, google_ads_connector):
        """Test _search handles API errors."""
        from aragora.connectors.advertising.google_ads import GoogleAdsError

        with patch.object(google_ads_connector, "_ensure_token") as mock_token:
            mock_token.return_value = "test_token"

            with patch.object(google_ads_connector, "_get_client") as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 400
                mock_response.json.return_value = {
                    "error": {
                        "message": "Invalid query",
                        "details": [],
                    }
                }
                mock_response.text = "Invalid query"
                mock_response.headers = {"request-id": "req_123"}
                mock_client.return_value.post = AsyncMock(return_value=mock_response)

                with pytest.raises(GoogleAdsError, match="Invalid query"):
                    await google_ads_connector._search("SELECT * FROM invalid")

    @pytest.mark.asyncio
    async def test_mutate_api_error(self, google_ads_connector):
        """Test _mutate handles API errors."""
        from aragora.connectors.advertising.google_ads import GoogleAdsError

        with patch.object(google_ads_connector, "_ensure_token") as mock_token:
            mock_token.return_value = "test_token"

            with patch.object(google_ads_connector, "_get_client") as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 403
                mock_response.json.return_value = {
                    "error": {
                        "message": "Permission denied",
                        "details": [],
                    }
                }
                mock_response.text = "Permission denied"
                mock_client.return_value.post = AsyncMock(return_value=mock_response)

                with pytest.raises(GoogleAdsError, match="Permission denied"):
                    await google_ads_connector._mutate([], "campaigns")


# =============================================================================
# Cleanup and Context Manager Tests
# =============================================================================


class TestCleanup:
    """Tests for cleanup and context manager."""

    @pytest.mark.asyncio
    async def test_close_closes_client(self, google_ads_connector):
        """Test close() closes HTTP client."""
        await google_ads_connector._get_client()
        assert google_ads_connector._client is not None

        await google_ads_connector.close()
        assert google_ads_connector._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self, google_ads_credentials):
        """Test async context manager."""
        from aragora.connectors.advertising.google_ads import GoogleAdsConnector

        async with GoogleAdsConnector(google_ads_credentials) as connector:
            await connector._get_client()
            assert connector._client is not None

        assert connector._client is None


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_campaign_status_values(self):
        """Test CampaignStatus enum values."""
        from aragora.connectors.advertising.google_ads import CampaignStatus

        assert CampaignStatus.ENABLED.value == "ENABLED"
        assert CampaignStatus.PAUSED.value == "PAUSED"
        assert CampaignStatus.REMOVED.value == "REMOVED"

    def test_campaign_type_values(self):
        """Test CampaignType enum values."""
        from aragora.connectors.advertising.google_ads import CampaignType

        assert CampaignType.SEARCH.value == "SEARCH"
        assert CampaignType.DISPLAY.value == "DISPLAY"
        assert CampaignType.PERFORMANCE_MAX.value == "PERFORMANCE_MAX"

    def test_bidding_strategy_type_values(self):
        """Test BiddingStrategyType enum values."""
        from aragora.connectors.advertising.google_ads import BiddingStrategyType

        assert BiddingStrategyType.MANUAL_CPC.value == "MANUAL_CPC"
        assert BiddingStrategyType.MAXIMIZE_CONVERSIONS.value == "MAXIMIZE_CONVERSIONS"
        assert BiddingStrategyType.TARGET_CPA.value == "TARGET_CPA"

    def test_keyword_match_type_values(self):
        """Test KeywordMatchType enum values."""
        from aragora.connectors.advertising.google_ads import KeywordMatchType

        assert KeywordMatchType.EXACT.value == "EXACT"
        assert KeywordMatchType.PHRASE.value == "PHRASE"
        assert KeywordMatchType.BROAD.value == "BROAD"


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_parse_date_valid(self):
        """Test _parse_date with valid date."""
        from aragora.connectors.advertising.google_ads import _parse_date

        result = _parse_date("2024-01-15")

        assert result == date(2024, 1, 15)

    def test_parse_date_none(self):
        """Test _parse_date with None."""
        from aragora.connectors.advertising.google_ads import _parse_date

        result = _parse_date(None)

        assert result is None

    def test_parse_date_invalid(self):
        """Test _parse_date with invalid date."""
        from aragora.connectors.advertising.google_ads import _parse_date

        result = _parse_date("invalid-date")

        assert result is None


# =============================================================================
# Mock Data Tests
# =============================================================================


class TestMockData:
    """Tests for mock data generation."""

    def test_get_mock_campaign(self):
        """Test get_mock_campaign."""
        from aragora.connectors.advertising.google_ads import get_mock_campaign

        campaign = get_mock_campaign()

        assert campaign.id == "12345678901"
        assert campaign.name == "Brand Campaign"
        assert campaign.budget_amount_micros == 50_000_000

    def test_get_mock_metrics(self):
        """Test get_mock_metrics."""
        from aragora.connectors.advertising.google_ads import get_mock_metrics

        metrics = get_mock_metrics()

        assert metrics.impressions == 10000
        assert metrics.clicks == 500
        assert metrics.conversions == 50
