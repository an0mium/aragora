"""
Comprehensive tests for Meta Ads Connector (Facebook/Instagram).

Tests for Meta Marketing API integration including:
- Client initialization
- Campaign operations (CRUD, status updates)
- Ad set operations
- Ad operations
- Insights and reporting
- Custom audiences
- Error handling
"""

import pytest
from datetime import date, datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def meta_ads_credentials():
    """Create test credentials."""
    from aragora.connectors.advertising.meta_ads import MetaAdsCredentials

    return MetaAdsCredentials(
        access_token="test_access_token",
        app_id="test_app_id",
        app_secret="test_app_secret",
        ad_account_id="act_123456789",
    )


@pytest.fixture
def meta_ads_connector(meta_ads_credentials):
    """Create test connector."""
    from aragora.connectors.advertising.meta_ads import MetaAdsConnector

    return MetaAdsConnector(meta_ads_credentials)


@pytest.fixture
def mock_campaign_response():
    """Mock campaign API response."""
    return {
        "id": "23456789012345678",
        "name": "Test Campaign",
        "status": "ACTIVE",
        "objective": "OUTCOME_TRAFFIC",
        "daily_budget": "5000",
        "lifetime_budget": "100000",
        "budget_remaining": "75000",
        "buying_type": "AUCTION",
        "special_ad_categories": [],
        "created_time": "2024-01-15T10:30:00+0000",
        "updated_time": "2024-01-20T15:45:00+0000",
    }


@pytest.fixture
def mock_ad_set_response():
    """Mock ad set API response."""
    return {
        "id": "34567890123456789",
        "name": "Test Ad Set",
        "campaign_id": "23456789012345678",
        "status": "ACTIVE",
        "daily_budget": "2000",
        "bid_amount": "500",
        "billing_event": "IMPRESSIONS",
        "optimization_goal": "LINK_CLICKS",
        "targeting": {"age_min": 18, "age_max": 65},
        "start_time": "2024-01-15T00:00:00+0000",
        "end_time": "2024-12-31T23:59:59+0000",
        "created_time": "2024-01-15T10:30:00+0000",
    }


@pytest.fixture
def mock_ad_response():
    """Mock ad API response."""
    return {
        "id": "45678901234567890",
        "name": "Test Ad",
        "adset_id": "34567890123456789",
        "campaign_id": "23456789012345678",
        "status": "ACTIVE",
        "creative": {"id": "56789012345678901"},
        "tracking_specs": [{"action.type": ["offsite_conversion"]}],
        "created_time": "2024-01-15T10:30:00+0000",
        "updated_time": "2024-01-20T15:45:00+0000",
    }


@pytest.fixture
def mock_insights_response():
    """Mock insights API response."""
    return {
        "campaign_id": "23456789012345678",
        "campaign_name": "Test Campaign",
        "impressions": "50000",
        "reach": "35000",
        "clicks": "2500",
        "spend": "250.50",
        "cpc": "0.10",
        "cpm": "5.01",
        "ctr": "5.0",
        "frequency": "1.43",
        "actions": [
            {"action_type": "link_click", "value": "2000"},
            {"action_type": "purchase", "value": "75"},
        ],
        "date_start": "2024-01-01",
        "date_stop": "2024-01-31",
    }


@pytest.fixture
def mock_audience_response():
    """Mock custom audience API response."""
    return {
        "id": "67890123456789012",
        "name": "Website Visitors",
        "description": "People who visited our website",
        "subtype": "WEBSITE",
        "approximate_count": 50000,
        "data_source": {"type": "PIXEL"},
    }


# =============================================================================
# Client Initialization Tests
# =============================================================================


class TestMetaAdsCredentials:
    """Tests for MetaAdsCredentials dataclass."""

    def test_credentials_creation(self):
        """Test credentials creation with required fields."""
        from aragora.connectors.advertising.meta_ads import MetaAdsCredentials

        creds = MetaAdsCredentials(
            access_token="test_token",
            ad_account_id="act_123456789",
        )

        assert creds.access_token == "test_token"
        assert creds.ad_account_id == "act_123456789"
        assert creds.app_id is None
        assert creds.app_secret is None
        assert creds.base_url == "https://graph.facebook.com/v18.0"

    def test_credentials_with_app_info(self):
        """Test credentials with app information."""
        from aragora.connectors.advertising.meta_ads import MetaAdsCredentials

        creds = MetaAdsCredentials(
            access_token="test_token",
            app_id="app_123",
            app_secret="secret_456",
            ad_account_id="act_789",
        )

        assert creds.app_id == "app_123"
        assert creds.app_secret == "secret_456"

    def test_credentials_with_custom_base_url(self):
        """Test credentials with custom API version."""
        from aragora.connectors.advertising.meta_ads import MetaAdsCredentials

        creds = MetaAdsCredentials(
            access_token="test_token",
            ad_account_id="act_123",
            base_url="https://graph.facebook.com/v19.0",
        )

        assert creds.base_url == "https://graph.facebook.com/v19.0"


class TestMetaAdsConnectorInit:
    """Tests for MetaAdsConnector initialization."""

    def test_connector_initialization(self, meta_ads_credentials):
        """Test connector initializes correctly."""
        from aragora.connectors.advertising.meta_ads import MetaAdsConnector

        connector = MetaAdsConnector(meta_ads_credentials)

        assert connector.credentials == meta_ads_credentials
        assert connector._client is None

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self, meta_ads_connector):
        """Test _get_client creates HTTP client."""
        client = await meta_ads_connector._get_client()

        assert client is not None
        assert isinstance(client, httpx.AsyncClient)

        await meta_ads_connector.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_client(self, meta_ads_connector):
        """Test _get_client reuses existing client."""
        client1 = await meta_ads_connector._get_client()
        client2 = await meta_ads_connector._get_client()

        assert client1 is client2

        await meta_ads_connector.close()


# =============================================================================
# Ad Account Tests
# =============================================================================


class TestAdAccountDataclass:
    """Tests for AdAccount dataclass."""

    def test_ad_account_from_api(self):
        """Test AdAccount.from_api parsing."""
        from aragora.connectors.advertising.meta_ads import AdAccount

        data = {
            "id": "act_123456789",
            "name": "Test Ad Account",
            "account_status": 1,
            "currency": "USD",
            "timezone_name": "America/New_York",
            "amount_spent": "50000",
            "balance": "10000",
            "spend_cap": "100000",
            "business": {"id": "biz_111"},
        }

        account = AdAccount.from_api(data)

        assert account.id == "act_123456789"
        assert account.name == "Test Ad Account"
        assert account.account_status == 1
        assert account.currency == "USD"
        assert account.amount_spent == 50000
        assert account.business_id == "biz_111"


class TestAdAccountOperations:
    """Tests for ad account operations."""

    @pytest.mark.asyncio
    async def test_get_ad_account(self, meta_ads_connector):
        """Test get_ad_account returns account details."""
        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.return_value = {
                "id": "act_123456789",
                "name": "Test Account",
                "account_status": 1,
                "currency": "USD",
                "timezone_name": "UTC",
            }

            account = await meta_ads_connector.get_ad_account()

            assert account.id == "act_123456789"
            assert account.name == "Test Account"

    @pytest.mark.asyncio
    async def test_get_ad_accounts(self, meta_ads_connector):
        """Test get_ad_accounts returns list of accounts."""
        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.return_value = {
                "data": [
                    {
                        "id": "act_1",
                        "name": "Account 1",
                        "account_status": 1,
                        "currency": "USD",
                        "timezone_name": "UTC",
                    },
                    {
                        "id": "act_2",
                        "name": "Account 2",
                        "account_status": 1,
                        "currency": "EUR",
                        "timezone_name": "UTC",
                    },
                ]
            }

            accounts = await meta_ads_connector.get_ad_accounts("biz_123")

            assert len(accounts) == 2
            assert accounts[0].id == "act_1"


# =============================================================================
# Campaign Operations Tests
# =============================================================================


class TestCampaignDataclass:
    """Tests for Campaign dataclass."""

    def test_campaign_from_api(self, mock_campaign_response):
        """Test Campaign.from_api parsing."""
        from aragora.connectors.advertising.meta_ads import (
            Campaign,
            CampaignStatus,
            CampaignObjective,
        )

        campaign = Campaign.from_api(mock_campaign_response)

        assert campaign.id == "23456789012345678"
        assert campaign.name == "Test Campaign"
        assert campaign.status == CampaignStatus.ACTIVE
        assert campaign.objective == CampaignObjective.OUTCOME_TRAFFIC
        assert campaign.daily_budget == 5000
        assert campaign.lifetime_budget == 100000
        assert campaign.budget_remaining == 75000

    def test_campaign_from_api_minimal(self):
        """Test Campaign.from_api with minimal data."""
        from aragora.connectors.advertising.meta_ads import Campaign

        data = {
            "id": "123",
            "name": "Minimal Campaign",
            "status": "PAUSED",
            "objective": "OUTCOME_AWARENESS",
        }

        campaign = Campaign.from_api(data)

        assert campaign.id == "123"
        assert campaign.name == "Minimal Campaign"
        assert campaign.daily_budget is None

    def test_campaign_from_api_with_datetime(self, mock_campaign_response):
        """Test Campaign.from_api parses datetime correctly."""
        from aragora.connectors.advertising.meta_ads import Campaign

        campaign = Campaign.from_api(mock_campaign_response)

        assert campaign.created_time is not None
        assert campaign.updated_time is not None


class TestCampaignOperations:
    """Tests for campaign CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_campaigns(self, meta_ads_connector, mock_campaign_response):
        """Test get_campaigns returns list of campaigns."""
        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": [mock_campaign_response]}

            campaigns = await meta_ads_connector.get_campaigns()

            assert len(campaigns) == 1
            assert campaigns[0].id == "23456789012345678"

    @pytest.mark.asyncio
    async def test_get_campaigns_with_status_filter(self, meta_ads_connector):
        """Test get_campaigns with status filter."""
        from aragora.connectors.advertising.meta_ads import CampaignStatus

        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": []}

            await meta_ads_connector.get_campaigns(status=CampaignStatus.ACTIVE)

            call_args = mock_request.call_args
            assert "filtering" in call_args[1]["params"]

    @pytest.mark.asyncio
    async def test_get_campaigns_with_limit(self, meta_ads_connector):
        """Test get_campaigns with custom limit."""
        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": []}

            await meta_ads_connector.get_campaigns(limit=50)

            call_args = mock_request.call_args
            assert call_args[1]["params"]["limit"] == 50

    @pytest.mark.asyncio
    async def test_get_campaign_by_id(self, meta_ads_connector, mock_campaign_response):
        """Test get_campaign returns single campaign."""
        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.return_value = mock_campaign_response

            campaign = await meta_ads_connector.get_campaign("23456789012345678")

            assert campaign.id == "23456789012345678"

    @pytest.mark.asyncio
    async def test_create_campaign(self, meta_ads_connector, mock_campaign_response):
        """Test create_campaign creates a new campaign."""
        from aragora.connectors.advertising.meta_ads import CampaignObjective

        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.side_effect = [
                {"id": "23456789012345678"},
                mock_campaign_response,
            ]

            campaign = await meta_ads_connector.create_campaign(
                name="New Campaign",
                objective=CampaignObjective.OUTCOME_TRAFFIC,
                daily_budget=5000,
            )

            assert campaign.name == "Test Campaign"

    @pytest.mark.asyncio
    async def test_create_campaign_with_special_categories(
        self, meta_ads_connector, mock_campaign_response
    ):
        """Test create_campaign with special ad categories."""
        from aragora.connectors.advertising.meta_ads import CampaignObjective

        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.side_effect = [
                {"id": "123"},
                mock_campaign_response,
            ]

            await meta_ads_connector.create_campaign(
                name="Housing Campaign",
                objective=CampaignObjective.OUTCOME_LEADS,
                special_ad_categories=["HOUSING"],
            )

            call_args = mock_request.call_args_list[0]
            assert "HOUSING" in call_args[1]["params"]["special_ad_categories"]

    @pytest.mark.asyncio
    async def test_update_campaign(self, meta_ads_connector, mock_campaign_response):
        """Test update_campaign updates campaign."""
        from aragora.connectors.advertising.meta_ads import CampaignStatus

        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.side_effect = [
                {"success": True},
                mock_campaign_response,
            ]

            campaign = await meta_ads_connector.update_campaign(
                "23456789012345678",
                name="Updated Campaign",
                status=CampaignStatus.PAUSED,
                daily_budget=10000,
            )

            assert campaign is not None

    @pytest.mark.asyncio
    async def test_delete_campaign(self, meta_ads_connector):
        """Test delete_campaign archives campaign."""
        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"success": True}

            result = await meta_ads_connector.delete_campaign("23456789012345678")

            assert result is True
            call_args = mock_request.call_args
            assert call_args[1]["params"]["status"] == "DELETED"


# =============================================================================
# Ad Set Operations Tests
# =============================================================================


class TestAdSetDataclass:
    """Tests for AdSet dataclass."""

    def test_ad_set_from_api(self, mock_ad_set_response):
        """Test AdSet.from_api parsing."""
        from aragora.connectors.advertising.meta_ads import (
            AdSet,
            AdSetStatus,
            BillingEvent,
            OptimizationGoal,
        )

        ad_set = AdSet.from_api(mock_ad_set_response)

        assert ad_set.id == "34567890123456789"
        assert ad_set.name == "Test Ad Set"
        assert ad_set.campaign_id == "23456789012345678"
        assert ad_set.status == AdSetStatus.ACTIVE
        assert ad_set.daily_budget == 2000
        assert ad_set.billing_event == BillingEvent.IMPRESSIONS
        assert ad_set.optimization_goal == OptimizationGoal.LINK_CLICKS

    def test_ad_set_from_api_with_targeting(self, mock_ad_set_response):
        """Test AdSet.from_api parses targeting correctly."""
        from aragora.connectors.advertising.meta_ads import AdSet

        ad_set = AdSet.from_api(mock_ad_set_response)

        assert ad_set.targeting == {"age_min": 18, "age_max": 65}


class TestAdSetOperations:
    """Tests for ad set operations."""

    @pytest.mark.asyncio
    async def test_get_ad_sets(self, meta_ads_connector, mock_ad_set_response):
        """Test get_ad_sets returns list of ad sets."""
        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": [mock_ad_set_response]}

            ad_sets = await meta_ads_connector.get_ad_sets()

            assert len(ad_sets) == 1
            assert ad_sets[0].id == "34567890123456789"

    @pytest.mark.asyncio
    async def test_get_ad_sets_by_campaign(self, meta_ads_connector):
        """Test get_ad_sets filtered by campaign."""
        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": []}

            await meta_ads_connector.get_ad_sets(campaign_id="23456789012345678")

            call_args = mock_request.call_args
            assert "/23456789012345678/adsets" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_get_ad_set_by_id(self, meta_ads_connector, mock_ad_set_response):
        """Test get_ad_set returns single ad set."""
        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.return_value = mock_ad_set_response

            ad_set = await meta_ads_connector.get_ad_set("34567890123456789")

            assert ad_set.id == "34567890123456789"

    @pytest.mark.asyncio
    async def test_create_ad_set(self, meta_ads_connector, mock_ad_set_response):
        """Test create_ad_set creates a new ad set."""
        from aragora.connectors.advertising.meta_ads import (
            BillingEvent,
            OptimizationGoal,
        )

        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.side_effect = [
                {"id": "34567890123456789"},
                mock_ad_set_response,
            ]

            ad_set = await meta_ads_connector.create_ad_set(
                name="New Ad Set",
                campaign_id="23456789012345678",
                daily_budget=2000,
                optimization_goal=OptimizationGoal.LINK_CLICKS,
                billing_event=BillingEvent.IMPRESSIONS,
                targeting={"age_min": 18, "age_max": 65},
            )

            assert ad_set.name == "Test Ad Set"

    @pytest.mark.asyncio
    async def test_create_ad_set_with_schedule(self, meta_ads_connector, mock_ad_set_response):
        """Test create_ad_set with start/end times."""
        from aragora.connectors.advertising.meta_ads import (
            BillingEvent,
            OptimizationGoal,
        )

        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.side_effect = [
                {"id": "123"},
                mock_ad_set_response,
            ]

            start = datetime(2024, 1, 15, tzinfo=timezone.utc)
            end = datetime(2024, 12, 31, tzinfo=timezone.utc)

            await meta_ads_connector.create_ad_set(
                name="Scheduled Ad Set",
                campaign_id="23456789012345678",
                daily_budget=2000,
                optimization_goal=OptimizationGoal.LINK_CLICKS,
                billing_event=BillingEvent.IMPRESSIONS,
                targeting={},
                start_time=start,
                end_time=end,
            )

            call_args = mock_request.call_args_list[0]
            assert "start_time" in call_args[1]["params"]
            assert "end_time" in call_args[1]["params"]

    @pytest.mark.asyncio
    async def test_update_ad_set_status(self, meta_ads_connector, mock_ad_set_response):
        """Test update_ad_set_status updates status."""
        from aragora.connectors.advertising.meta_ads import AdSetStatus

        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.side_effect = [
                {"success": True},
                mock_ad_set_response,
            ]

            ad_set = await meta_ads_connector.update_ad_set_status(
                "34567890123456789", AdSetStatus.PAUSED
            )

            assert ad_set is not None


# =============================================================================
# Ad Operations Tests
# =============================================================================


class TestAdDataclass:
    """Tests for Ad dataclass."""

    def test_ad_from_api(self, mock_ad_response):
        """Test Ad.from_api parsing."""
        from aragora.connectors.advertising.meta_ads import Ad, AdStatus

        ad = Ad.from_api(mock_ad_response)

        assert ad.id == "45678901234567890"
        assert ad.name == "Test Ad"
        assert ad.adset_id == "34567890123456789"
        assert ad.campaign_id == "23456789012345678"
        assert ad.status == AdStatus.ACTIVE
        assert ad.creative_id == "56789012345678901"


class TestAdOperations:
    """Tests for ad operations."""

    @pytest.mark.asyncio
    async def test_get_ads(self, meta_ads_connector, mock_ad_response):
        """Test get_ads returns list of ads."""
        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": [mock_ad_response]}

            ads = await meta_ads_connector.get_ads()

            assert len(ads) == 1
            assert ads[0].id == "45678901234567890"

    @pytest.mark.asyncio
    async def test_get_ads_by_adset(self, meta_ads_connector):
        """Test get_ads filtered by ad set."""
        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": []}

            await meta_ads_connector.get_ads(adset_id="34567890123456789")

            call_args = mock_request.call_args
            assert "/34567890123456789/ads" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_get_ad_by_id(self, meta_ads_connector, mock_ad_response):
        """Test get_ad returns single ad."""
        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.return_value = mock_ad_response

            ad = await meta_ads_connector.get_ad("45678901234567890")

            assert ad.id == "45678901234567890"

    @pytest.mark.asyncio
    async def test_update_ad_status(self, meta_ads_connector, mock_ad_response):
        """Test update_ad_status updates status."""
        from aragora.connectors.advertising.meta_ads import AdStatus

        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.side_effect = [
                {"success": True},
                mock_ad_response,
            ]

            ad = await meta_ads_connector.update_ad_status("45678901234567890", AdStatus.PAUSED)

            assert ad is not None


# =============================================================================
# Insights/Reporting Tests
# =============================================================================


class TestAdInsightsDataclass:
    """Tests for AdInsights dataclass."""

    def test_ad_insights_from_api(self, mock_insights_response):
        """Test AdInsights.from_api parsing."""
        from aragora.connectors.advertising.meta_ads import AdInsights

        insights = AdInsights.from_api(mock_insights_response)

        assert insights.campaign_id == "23456789012345678"
        assert insights.impressions == 50000
        assert insights.reach == 35000
        assert insights.clicks == 2500
        assert insights.spend == Decimal("250.50")
        assert insights.cpc == Decimal("0.10")
        assert insights.ctr == 5.0
        assert len(insights.actions) == 2

    def test_ad_insights_from_api_with_dates(self, mock_insights_response):
        """Test AdInsights.from_api parses dates correctly."""
        from aragora.connectors.advertising.meta_ads import AdInsights

        insights = AdInsights.from_api(mock_insights_response)

        assert insights.date_start == date(2024, 1, 1)
        assert insights.date_stop == date(2024, 1, 31)


class TestInsightsOperations:
    """Tests for insights operations."""

    @pytest.mark.asyncio
    async def test_get_insights(self, meta_ads_connector, mock_insights_response):
        """Test get_insights returns insights data."""
        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": [mock_insights_response]}

            insights = await meta_ads_connector.get_insights()

            assert len(insights) == 1
            assert insights[0].impressions == 50000

    @pytest.mark.asyncio
    async def test_get_insights_with_date_preset(self, meta_ads_connector):
        """Test get_insights with date preset."""
        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": []}

            await meta_ads_connector.get_insights(date_preset="last_30d")

            call_args = mock_request.call_args
            assert call_args[1]["params"]["date_preset"] == "last_30d"

    @pytest.mark.asyncio
    async def test_get_insights_with_time_range(self, meta_ads_connector):
        """Test get_insights with custom time range."""
        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": []}

            await meta_ads_connector.get_insights(
                time_range={"since": "2024-01-01", "until": "2024-01-31"}
            )

            call_args = mock_request.call_args
            assert "time_range" in call_args[1]["params"]

    @pytest.mark.asyncio
    async def test_get_insights_with_breakdowns(self, meta_ads_connector):
        """Test get_insights with breakdowns."""
        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": []}

            await meta_ads_connector.get_insights(breakdowns=["age", "gender"])

            call_args = mock_request.call_args
            assert call_args[1]["params"]["breakdowns"] == "age,gender"

    @pytest.mark.asyncio
    async def test_get_insights_with_level(self, meta_ads_connector):
        """Test get_insights with different level."""
        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": []}

            await meta_ads_connector.get_insights(level="adset")

            call_args = mock_request.call_args
            assert call_args[1]["params"]["level"] == "adset"

    @pytest.mark.asyncio
    async def test_get_campaign_insights(self, meta_ads_connector, mock_insights_response):
        """Test get_campaign_insights returns campaign-level insights."""
        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": [mock_insights_response]}

            insights = await meta_ads_connector.get_campaign_insights(
                "23456789012345678", date_preset="last_7d"
            )

            assert len(insights) == 1
            call_args = mock_request.call_args
            assert "/23456789012345678/insights" in call_args[0][1]


# =============================================================================
# Custom Audience Tests
# =============================================================================


class TestCustomAudienceDataclass:
    """Tests for CustomAudience dataclass."""

    def test_custom_audience_from_api(self, mock_audience_response):
        """Test CustomAudience.from_api parsing."""
        from aragora.connectors.advertising.meta_ads import CustomAudience

        audience = CustomAudience.from_api(mock_audience_response)

        assert audience.id == "67890123456789012"
        assert audience.name == "Website Visitors"
        assert audience.description == "People who visited our website"
        assert audience.subtype == "WEBSITE"
        assert audience.approximate_count == 50000


class TestCustomAudienceOperations:
    """Tests for custom audience operations."""

    @pytest.mark.asyncio
    async def test_get_custom_audiences(self, meta_ads_connector, mock_audience_response):
        """Test get_custom_audiences returns list of audiences."""
        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": [mock_audience_response]}

            audiences = await meta_ads_connector.get_custom_audiences()

            assert len(audiences) == 1
            assert audiences[0].name == "Website Visitors"

    @pytest.mark.asyncio
    async def test_create_custom_audience(self, meta_ads_connector):
        """Test create_custom_audience creates audience."""
        with patch.object(meta_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"id": "67890123456789012"}

            audience = await meta_ads_connector.create_custom_audience(
                name="New Audience",
                subtype="CUSTOM",
                description="Test audience",
            )

            assert audience.id == "67890123456789012"
            assert audience.name == "New Audience"


# =============================================================================
# Ad Creative Tests
# =============================================================================


class TestAdCreativeDataclass:
    """Tests for AdCreative dataclass."""

    def test_ad_creative_from_api(self):
        """Test AdCreative.from_api parsing."""
        from aragora.connectors.advertising.meta_ads import AdCreative

        data = {
            "id": "56789012345678901",
            "name": "Test Creative",
            "title": "Great Product",
            "body": "Buy now!",
            "image_url": "https://example.com/image.jpg",
            "call_to_action_type": "LEARN_MORE",
            "link_url": "https://example.com",
            "object_story_spec": {"page_id": "page_123"},
        }

        creative = AdCreative.from_api(data)

        assert creative.id == "56789012345678901"
        assert creative.name == "Test Creative"
        assert creative.title == "Great Product"
        assert creative.call_to_action_type == "LEARN_MORE"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestMetaAdsError:
    """Tests for MetaAdsError exception."""

    def test_error_creation(self):
        """Test MetaAdsError creation."""
        from aragora.connectors.advertising.meta_ads import MetaAdsError

        error = MetaAdsError(
            message="Invalid access token",
            code=190,
            error_subcode=463,
        )

        assert str(error) == "Invalid access token"
        assert error.code == 190
        assert error.error_subcode == 463

    def test_error_minimal(self):
        """Test MetaAdsError with minimal info."""
        from aragora.connectors.advertising.meta_ads import MetaAdsError

        error = MetaAdsError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.code is None
        assert error.error_subcode is None


class TestAPIErrorHandling:
    """Tests for API error handling."""

    @pytest.mark.asyncio
    async def test_request_handles_api_error(self, meta_ads_connector):
        """Test _request handles API errors correctly."""
        from aragora.connectors.advertising.meta_ads import MetaAdsError

        with patch.object(meta_ads_connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "error": {
                    "message": "Invalid OAuth access token",
                    "code": 190,
                    "error_subcode": 463,
                }
            }
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(MetaAdsError) as exc_info:
                await meta_ads_connector._request("GET", "/test")

            assert exc_info.value.code == 190
            assert exc_info.value.error_subcode == 463


# =============================================================================
# Cleanup and Context Manager Tests
# =============================================================================


class TestCleanup:
    """Tests for cleanup and context manager."""

    @pytest.mark.asyncio
    async def test_close_closes_client(self, meta_ads_connector):
        """Test close() closes HTTP client."""
        await meta_ads_connector._get_client()
        assert meta_ads_connector._client is not None

        await meta_ads_connector.close()
        assert meta_ads_connector._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self, meta_ads_credentials):
        """Test async context manager."""
        from aragora.connectors.advertising.meta_ads import MetaAdsConnector

        async with MetaAdsConnector(meta_ads_credentials) as connector:
            await connector._get_client()
            assert connector._client is not None

        assert connector._client is None


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_campaign_objective_values(self):
        """Test CampaignObjective enum values."""
        from aragora.connectors.advertising.meta_ads import CampaignObjective

        assert CampaignObjective.OUTCOME_AWARENESS.value == "OUTCOME_AWARENESS"
        assert CampaignObjective.OUTCOME_TRAFFIC.value == "OUTCOME_TRAFFIC"
        assert CampaignObjective.OUTCOME_SALES.value == "OUTCOME_SALES"

    def test_campaign_status_values(self):
        """Test CampaignStatus enum values."""
        from aragora.connectors.advertising.meta_ads import CampaignStatus

        assert CampaignStatus.ACTIVE.value == "ACTIVE"
        assert CampaignStatus.PAUSED.value == "PAUSED"
        assert CampaignStatus.DELETED.value == "DELETED"

    def test_billing_event_values(self):
        """Test BillingEvent enum values."""
        from aragora.connectors.advertising.meta_ads import BillingEvent

        assert BillingEvent.IMPRESSIONS.value == "IMPRESSIONS"
        assert BillingEvent.LINK_CLICKS.value == "LINK_CLICKS"
        assert BillingEvent.THRUPLAY.value == "THRUPLAY"

    def test_optimization_goal_values(self):
        """Test OptimizationGoal enum values."""
        from aragora.connectors.advertising.meta_ads import OptimizationGoal

        assert OptimizationGoal.CONVERSIONS.value == "CONVERSIONS"
        assert OptimizationGoal.LINK_CLICKS.value == "LINK_CLICKS"
        assert OptimizationGoal.REACH.value == "REACH"

    def test_placement_values(self):
        """Test Placement enum values."""
        from aragora.connectors.advertising.meta_ads import Placement

        assert Placement.FACEBOOK_FEED.value == "facebook_feed"
        assert Placement.INSTAGRAM_STORIES.value == "instagram_stories"
        assert Placement.INSTAGRAM_REELS.value == "instagram_reels"


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_parse_datetime_valid(self):
        """Test _parse_datetime with valid datetime."""
        from aragora.connectors.advertising.meta_ads import _parse_datetime

        result = _parse_datetime("2024-01-15T10:30:00+0000")

        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_datetime_with_z(self):
        """Test _parse_datetime with Z timezone."""
        from aragora.connectors.advertising.meta_ads import _parse_datetime

        result = _parse_datetime("2024-01-15T10:30:00Z")

        assert result is not None

    def test_parse_datetime_none(self):
        """Test _parse_datetime with None."""
        from aragora.connectors.advertising.meta_ads import _parse_datetime

        result = _parse_datetime(None)

        assert result is None

    def test_parse_datetime_invalid(self):
        """Test _parse_datetime with invalid datetime."""
        from aragora.connectors.advertising.meta_ads import _parse_datetime

        result = _parse_datetime("invalid-datetime")

        assert result is None

    def test_parse_date_valid(self):
        """Test _parse_date with valid date."""
        from aragora.connectors.advertising.meta_ads import _parse_date

        result = _parse_date("2024-01-15")

        assert result == date(2024, 1, 15)

    def test_parse_date_none(self):
        """Test _parse_date with None."""
        from aragora.connectors.advertising.meta_ads import _parse_date

        result = _parse_date(None)

        assert result is None

    def test_parse_date_invalid(self):
        """Test _parse_date with invalid date."""
        from aragora.connectors.advertising.meta_ads import _parse_date

        result = _parse_date("invalid")

        assert result is None


# =============================================================================
# Mock Data Tests
# =============================================================================


class TestMockData:
    """Tests for mock data generation."""

    def test_get_mock_campaign(self):
        """Test get_mock_campaign."""
        from aragora.connectors.advertising.meta_ads import get_mock_campaign

        campaign = get_mock_campaign()

        assert campaign.id == "23456789012345678"
        assert campaign.name == "Summer Sale Campaign"
        assert campaign.daily_budget == 5000

    def test_get_mock_insights(self):
        """Test get_mock_insights."""
        from aragora.connectors.advertising.meta_ads import get_mock_insights

        insights = get_mock_insights()

        assert insights.impressions == 50000
        assert insights.clicks == 2500
        assert insights.spend == Decimal("250.00")
