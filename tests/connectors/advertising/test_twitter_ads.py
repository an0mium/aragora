"""
Comprehensive tests for Twitter/X Ads Connector.

Tests for Twitter Ads API integration including:
- Client initialization and OAuth 1.0a authentication
- Campaign operations (CRUD, status updates)
- Line item (ad group) operations
- Promoted tweet operations
- Audience management
- Analytics and reporting
- Funding instrument management
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
def twitter_ads_credentials():
    """Create test credentials."""
    from aragora.connectors.advertising.twitter_ads import TwitterAdsCredentials

    return TwitterAdsCredentials(
        consumer_key="test_consumer_key",
        consumer_secret="test_consumer_secret",
        access_token="test_access_token",
        access_token_secret="test_access_token_secret",
        ads_account_id="acc_123456789",
    )


@pytest.fixture
def twitter_ads_connector(twitter_ads_credentials):
    """Create test connector."""
    from aragora.connectors.advertising.twitter_ads import TwitterAdsConnector

    return TwitterAdsConnector(twitter_ads_credentials)


@pytest.fixture
def mock_campaign_api_response():
    """Mock campaign API response."""
    return {
        "id": "abc123",
        "name": "Test Twitter Campaign",
        "account_id": "acc_123456789",
        "entity_status": "ACTIVE",
        "objective": "WEBSITE_CLICKS",
        "funding_instrument_id": "fi_987654",
        "daily_budget_amount_local_micro": 100000000,
        "total_budget_amount_local_micro": 3000000000,
        "start_time": "2024-01-01T00:00:00Z",
        "end_time": "2024-12-31T23:59:59Z",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-15T12:00:00Z",
    }


@pytest.fixture
def mock_line_item_api_response():
    """Mock line item API response."""
    return {
        "id": "li_456789",
        "campaign_id": "abc123",
        "name": "Test Line Item",
        "entity_status": "ACTIVE",
        "product_type": "PROMOTED_TWEETS",
        "placements": ["ALL_ON_TWITTER", "PUBLISHER_NETWORK"],
        "bid_amount_local_micro": 500000,
        "total_budget_amount_local_micro": 1000000000,
        "start_time": "2024-01-01T00:00:00Z",
        "end_time": "2024-06-30T23:59:59Z",
    }


@pytest.fixture
def mock_promoted_tweet_api_response():
    """Mock promoted tweet API response."""
    return {
        "id": "pt_789123",
        "line_item_id": "li_456789",
        "tweet_id": "tweet_111222333",
        "entity_status": "ACTIVE",
        "approval_status": "ACCEPTED",
        "created_at": "2024-01-05T10:00:00Z",
    }


@pytest.fixture
def mock_analytics_api_response():
    """Mock campaign analytics API response."""
    return {
        "id": "abc123",
        "metrics": {
            "impressions": [100000],
            "engagements": [5000],
            "clicks": [2500],
            "billed_charge_local_micro": [500000000],
            "retweets": [200],
            "replies": [100],
            "likes": [2000],
            "follows": [50],
            "video_views": [10000],
            "app_installs": [25],
        },
    }


@pytest.fixture
def mock_audience_api_response():
    """Mock tailored audience API response."""
    return {
        "id": "ta_123456",
        "name": "Test Audience",
        "audience_type": "CRM",
        "audience_size": 100000,
        "targetable": True,
        "created_at": "2024-01-10T08:00:00Z",
    }


@pytest.fixture
def mock_funding_instrument_api_response():
    """Mock funding instrument API response."""
    return {
        "id": "fi_987654",
        "account_id": "acc_123456789",
        "currency": "USD",
        "funded_amount_local_micro": 10000000000,
        "credit_remaining_local_micro": 7500000000,
        "start_time": "2024-01-01T00:00:00Z",
        "end_time": "2024-12-31T23:59:59Z",
    }


# =============================================================================
# Client Initialization Tests
# =============================================================================


class TestTwitterAdsCredentials:
    """Tests for TwitterAdsCredentials dataclass."""

    def test_credentials_creation(self):
        """Test credentials creation with required fields."""
        from aragora.connectors.advertising.twitter_ads import TwitterAdsCredentials

        creds = TwitterAdsCredentials(
            consumer_key="key123",
            consumer_secret="secret456",
            access_token="token789",
            access_token_secret="token_secret000",
            ads_account_id="acc_111",
        )

        assert creds.consumer_key == "key123"
        assert creds.consumer_secret == "secret456"
        assert creds.access_token == "token789"
        assert creds.access_token_secret == "token_secret000"
        assert creds.ads_account_id == "acc_111"


class TestTwitterAdsConnectorInit:
    """Tests for TwitterAdsConnector initialization."""

    def test_connector_initialization(self, twitter_ads_credentials):
        """Test connector initializes correctly."""
        from aragora.connectors.advertising.twitter_ads import TwitterAdsConnector

        connector = TwitterAdsConnector(twitter_ads_credentials)

        assert connector.credentials == twitter_ads_credentials
        assert connector._client is None
        assert connector.BASE_URL == "https://ads-api.twitter.com/12"

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self, twitter_ads_connector):
        """Test _get_client creates HTTP client."""
        client = await twitter_ads_connector._get_client()

        assert client is not None
        assert isinstance(client, httpx.AsyncClient)

        await twitter_ads_connector.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_client(self, twitter_ads_connector):
        """Test _get_client reuses existing client."""
        client1 = await twitter_ads_connector._get_client()
        client2 = await twitter_ads_connector._get_client()

        assert client1 is client2

        await twitter_ads_connector.close()

    @pytest.mark.asyncio
    async def test_close_closes_client(self, twitter_ads_connector):
        """Test close() closes HTTP client."""
        await twitter_ads_connector._get_client()
        assert twitter_ads_connector._client is not None

        await twitter_ads_connector.close()
        assert twitter_ads_connector._client is None


class TestOAuthHeader:
    """Tests for OAuth 1.0a header generation."""

    def test_oauth_header_generation(self, twitter_ads_connector):
        """Test _oauth_header generates valid OAuth header."""
        header = twitter_ads_connector._oauth_header(
            "GET", "https://ads-api.twitter.com/12/accounts/acc_123/campaigns"
        )

        assert header.startswith("OAuth ")
        assert "oauth_consumer_key" in header
        assert "oauth_token" in header
        assert "oauth_signature_method" in header
        assert "oauth_timestamp" in header
        assert "oauth_nonce" in header
        assert "oauth_signature" in header

    def test_oauth_header_contains_credentials(self, twitter_ads_connector):
        """Test _oauth_header contains credential values."""
        header = twitter_ads_connector._oauth_header(
            "POST", "https://ads-api.twitter.com/12/accounts/acc_123/campaigns"
        )

        assert "test_consumer_key" in header
        assert "test_access_token" in header


# =============================================================================
# Funding Instrument Dataclass Tests
# =============================================================================


class TestFundingInstrumentDataclass:
    """Tests for FundingInstrument dataclass."""

    def test_funding_instrument_from_api(self, mock_funding_instrument_api_response):
        """Test FundingInstrument.from_api parsing."""
        from aragora.connectors.advertising.twitter_ads import FundingInstrument

        fi = FundingInstrument.from_api(mock_funding_instrument_api_response)

        assert fi.id == "fi_987654"
        assert fi.account_id == "acc_123456789"
        assert fi.currency == "USD"
        assert fi.funded_amount == 10000.0
        assert fi.credit_remaining == 7500.0
        assert fi.start_time is not None
        assert fi.end_time is not None

    def test_funding_instrument_from_api_minimal(self):
        """Test FundingInstrument.from_api with minimal data."""
        from aragora.connectors.advertising.twitter_ads import FundingInstrument

        data = {
            "id": "fi_123",
            "account_id": "acc_456",
        }

        fi = FundingInstrument.from_api(data)

        assert fi.id == "fi_123"
        assert fi.currency == "USD"
        assert fi.funded_amount == 0.0
        assert fi.start_time is None


# =============================================================================
# Campaign Dataclass Tests
# =============================================================================


class TestCampaignDataclass:
    """Tests for Campaign dataclass."""

    def test_campaign_from_api(self, mock_campaign_api_response):
        """Test Campaign.from_api parsing."""
        from aragora.connectors.advertising.twitter_ads import (
            Campaign,
            CampaignStatus,
            CampaignObjective,
        )

        campaign = Campaign.from_api(mock_campaign_api_response)

        assert campaign.id == "abc123"
        assert campaign.name == "Test Twitter Campaign"
        assert campaign.account_id == "acc_123456789"
        assert campaign.status == CampaignStatus.ACTIVE
        assert campaign.objective == CampaignObjective.WEBSITE_CLICKS
        assert campaign.funding_instrument_id == "fi_987654"
        assert campaign.daily_budget == 100.0
        assert campaign.total_budget == 3000.0
        assert campaign.start_time is not None
        assert campaign.end_time is not None

    def test_campaign_from_api_minimal(self):
        """Test Campaign.from_api with minimal data."""
        from aragora.connectors.advertising.twitter_ads import Campaign

        data = {
            "id": "camp_123",
            "name": "Minimal Campaign",
        }

        campaign = Campaign.from_api(data)

        assert campaign.id == "camp_123"
        assert campaign.name == "Minimal Campaign"
        assert campaign.daily_budget is None
        assert campaign.total_budget is None
        assert campaign.start_time is None

    def test_campaign_from_api_no_budgets(self):
        """Test Campaign.from_api without budgets."""
        from aragora.connectors.advertising.twitter_ads import Campaign

        data = {
            "id": "camp_456",
            "name": "No Budget Campaign",
            "entity_status": "PAUSED",
            "objective": "FOLLOWERS",
            "funding_instrument_id": "fi_123",
        }

        campaign = Campaign.from_api(data)

        assert campaign.daily_budget is None
        assert campaign.total_budget is None


# =============================================================================
# Line Item Dataclass Tests
# =============================================================================


class TestLineItemDataclass:
    """Tests for LineItem dataclass."""

    def test_line_item_from_api(self, mock_line_item_api_response):
        """Test LineItem.from_api parsing."""
        from aragora.connectors.advertising.twitter_ads import (
            LineItem,
            LineItemType,
            PlacementType,
        )

        line_item = LineItem.from_api(mock_line_item_api_response)

        assert line_item.id == "li_456789"
        assert line_item.campaign_id == "abc123"
        assert line_item.name == "Test Line Item"
        assert line_item.status == "ACTIVE"
        assert line_item.line_item_type == LineItemType.PROMOTED_TWEETS
        assert len(line_item.placements) == 2
        assert PlacementType.ALL_ON_TWITTER in line_item.placements
        assert line_item.bid_amount == 0.5
        assert line_item.total_budget == 1000.0

    def test_line_item_from_api_minimal(self):
        """Test LineItem.from_api with minimal data."""
        from aragora.connectors.advertising.twitter_ads import LineItem

        data = {
            "id": "li_123",
            "campaign_id": "camp_456",
            "name": "Minimal Line Item",
        }

        line_item = LineItem.from_api(data)

        assert line_item.id == "li_123"
        assert line_item.bid_amount is None
        assert line_item.placements == []


# =============================================================================
# Promoted Tweet Dataclass Tests
# =============================================================================


class TestPromotedTweetDataclass:
    """Tests for PromotedTweet dataclass."""

    def test_promoted_tweet_from_api(self, mock_promoted_tweet_api_response):
        """Test PromotedTweet.from_api parsing."""
        from aragora.connectors.advertising.twitter_ads import PromotedTweet

        pt = PromotedTweet.from_api(mock_promoted_tweet_api_response)

        assert pt.id == "pt_789123"
        assert pt.line_item_id == "li_456789"
        assert pt.tweet_id == "tweet_111222333"
        assert pt.status == "ACTIVE"
        assert pt.approval_status == "ACCEPTED"
        assert pt.created_at is not None

    def test_promoted_tweet_from_api_minimal(self):
        """Test PromotedTweet.from_api with minimal data."""
        from aragora.connectors.advertising.twitter_ads import PromotedTweet

        data = {
            "id": "pt_123",
            "line_item_id": "li_456",
            "tweet_id": "tweet_789",
        }

        pt = PromotedTweet.from_api(data)

        assert pt.id == "pt_123"
        assert pt.approval_status == ""
        assert pt.created_at is None


# =============================================================================
# Campaign Analytics Dataclass Tests
# =============================================================================


class TestCampaignAnalyticsDataclass:
    """Tests for CampaignAnalytics dataclass."""

    def test_campaign_analytics_from_api(self, mock_analytics_api_response):
        """Test CampaignAnalytics.from_api parsing."""
        from aragora.connectors.advertising.twitter_ads import CampaignAnalytics

        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)

        analytics = CampaignAnalytics.from_api(
            mock_analytics_api_response,
            "abc123",
            start_date,
            end_date,
        )

        assert analytics.campaign_id == "abc123"
        assert analytics.impressions == 100000
        assert analytics.engagements == 5000
        assert analytics.clicks == 2500
        assert analytics.spend == 500.0
        assert analytics.retweets == 200
        assert analytics.replies == 100
        assert analytics.likes == 2000
        assert analytics.follows == 50
        assert analytics.video_views == 10000
        assert analytics.app_installs == 25

    def test_campaign_analytics_calculated_fields(self):
        """Test CampaignAnalytics calculated fields."""
        from aragora.connectors.advertising.twitter_ads import CampaignAnalytics

        data = {
            "metrics": {
                "impressions": [10000],
                "engagements": [500],
                "clicks": [250],
                "billed_charge_local_micro": [100000000],
            }
        }

        analytics = CampaignAnalytics.from_api(
            data, "camp_123", date(2024, 1, 1), date(2024, 1, 31)
        )

        # Engagement rate = engagements/impressions * 100
        assert analytics.engagement_rate == 5.0
        # CPE = spend/engagements
        assert analytics.cpe == 0.2
        # CPC = spend/clicks
        assert analytics.cpc == 0.4

    def test_campaign_analytics_zero_division(self):
        """Test CampaignAnalytics handles zero division."""
        from aragora.connectors.advertising.twitter_ads import CampaignAnalytics

        data = {
            "metrics": {
                "impressions": [0],
                "engagements": [0],
                "clicks": [0],
                "billed_charge_local_micro": [0],
            }
        }

        analytics = CampaignAnalytics.from_api(
            data, "camp_123", date(2024, 1, 1), date(2024, 1, 31)
        )

        assert analytics.engagement_rate == 0.0
        assert analytics.cpe == 0.0
        assert analytics.cpc == 0.0

    def test_campaign_analytics_empty_metrics(self):
        """Test CampaignAnalytics with empty metrics."""
        from aragora.connectors.advertising.twitter_ads import CampaignAnalytics

        data = {"metrics": {}}

        analytics = CampaignAnalytics.from_api(
            data, "camp_123", date(2024, 1, 1), date(2024, 1, 31)
        )

        assert analytics.impressions == 0
        assert analytics.clicks == 0
        assert analytics.spend == 0.0


# =============================================================================
# Tailored Audience Dataclass Tests
# =============================================================================


class TestTailoredAudienceDataclass:
    """Tests for TailoredAudience dataclass."""

    def test_tailored_audience_from_api(self, mock_audience_api_response):
        """Test TailoredAudience.from_api parsing."""
        from aragora.connectors.advertising.twitter_ads import TailoredAudience

        audience = TailoredAudience.from_api(mock_audience_api_response)

        assert audience.id == "ta_123456"
        assert audience.name == "Test Audience"
        assert audience.audience_type == "CRM"
        assert audience.audience_size == 100000
        assert audience.targetable is True
        assert audience.created_at is not None

    def test_tailored_audience_from_api_minimal(self):
        """Test TailoredAudience.from_api with minimal data."""
        from aragora.connectors.advertising.twitter_ads import TailoredAudience

        data = {
            "id": "ta_789",
            "name": "Minimal Audience",
            "audience_type": "WEB",
        }

        audience = TailoredAudience.from_api(data)

        assert audience.id == "ta_789"
        assert audience.audience_size is None
        assert audience.created_at is None


# =============================================================================
# Campaign Operations Tests
# =============================================================================


class TestCampaignOperations:
    """Tests for campaign CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_campaigns(self, twitter_ads_connector, mock_campaign_api_response):
        """Test get_campaigns returns list of campaigns."""
        with patch.object(twitter_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": [mock_campaign_api_response]}

            campaigns = await twitter_ads_connector.get_campaigns()

            assert len(campaigns) == 1
            assert campaigns[0].id == "abc123"
            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_campaigns_with_status_filter(self, twitter_ads_connector):
        """Test get_campaigns with status filter."""
        from aragora.connectors.advertising.twitter_ads import CampaignStatus

        with patch.object(twitter_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": []}

            await twitter_ads_connector.get_campaigns(status=CampaignStatus.ACTIVE)

            call_args = mock_request.call_args
            assert call_args[1]["params"]["entity_status"] == "ACTIVE"

    @pytest.mark.asyncio
    async def test_get_campaign_by_id(self, twitter_ads_connector, mock_campaign_api_response):
        """Test get_campaign returns single campaign."""
        with patch.object(twitter_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": mock_campaign_api_response}

            campaign = await twitter_ads_connector.get_campaign("abc123")

            assert campaign.id == "abc123"

    @pytest.mark.asyncio
    async def test_create_campaign(self, twitter_ads_connector, mock_campaign_api_response):
        """Test create_campaign creates a new campaign."""
        from aragora.connectors.advertising.twitter_ads import CampaignObjective

        with patch.object(twitter_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": mock_campaign_api_response}

            campaign = await twitter_ads_connector.create_campaign(
                name="New Campaign",
                funding_instrument_id="fi_987654",
                objective=CampaignObjective.WEBSITE_CLICKS,
                daily_budget=100.0,
            )

            assert campaign.id == "abc123"
            call_args = mock_request.call_args
            assert call_args[1]["json_data"]["daily_budget_amount_local_micro"] == 100000000

    @pytest.mark.asyncio
    async def test_create_campaign_without_budget(self, twitter_ads_connector):
        """Test create_campaign without daily budget."""
        from aragora.connectors.advertising.twitter_ads import CampaignObjective

        with patch.object(twitter_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": {"id": "camp_123"}}

            await twitter_ads_connector.create_campaign(
                name="No Budget Campaign",
                funding_instrument_id="fi_123",
                objective=CampaignObjective.FOLLOWERS,
            )

            call_args = mock_request.call_args
            assert "daily_budget_amount_local_micro" not in call_args[1]["json_data"]

    @pytest.mark.asyncio
    async def test_update_campaign_status(self, twitter_ads_connector, mock_campaign_api_response):
        """Test update_campaign_status."""
        from aragora.connectors.advertising.twitter_ads import CampaignStatus

        with patch.object(twitter_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": mock_campaign_api_response}

            campaign = await twitter_ads_connector.update_campaign_status(
                "abc123", CampaignStatus.PAUSED
            )

            assert campaign.id == "abc123"
            call_args = mock_request.call_args
            assert call_args[1]["json_data"]["entity_status"] == "PAUSED"


# =============================================================================
# Line Item Operations Tests
# =============================================================================


class TestLineItemOperations:
    """Tests for line item operations."""

    @pytest.mark.asyncio
    async def test_get_line_items(self, twitter_ads_connector, mock_line_item_api_response):
        """Test get_line_items returns list of line items."""
        with patch.object(twitter_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": [mock_line_item_api_response]}

            line_items = await twitter_ads_connector.get_line_items("abc123")

            assert len(line_items) == 1
            assert line_items[0].id == "li_456789"

    @pytest.mark.asyncio
    async def test_create_line_item(self, twitter_ads_connector, mock_line_item_api_response):
        """Test create_line_item creates a new line item."""
        from aragora.connectors.advertising.twitter_ads import (
            LineItemType,
            PlacementType,
        )

        with patch.object(twitter_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": mock_line_item_api_response}

            line_item = await twitter_ads_connector.create_line_item(
                campaign_id="abc123",
                name="New Line Item",
                line_item_type=LineItemType.PROMOTED_TWEETS,
                placements=[PlacementType.ALL_ON_TWITTER],
                bid_amount=0.5,
            )

            assert line_item.id == "li_456789"
            call_args = mock_request.call_args
            assert call_args[1]["json_data"]["bid_amount_local_micro"] == 500000


# =============================================================================
# Promoted Tweet Operations Tests
# =============================================================================


class TestPromotedTweetOperations:
    """Tests for promoted tweet operations."""

    @pytest.mark.asyncio
    async def test_get_promoted_tweets(
        self, twitter_ads_connector, mock_promoted_tweet_api_response
    ):
        """Test get_promoted_tweets returns list of promoted tweets."""
        with patch.object(twitter_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": [mock_promoted_tweet_api_response]}

            tweets = await twitter_ads_connector.get_promoted_tweets("li_456789")

            assert len(tweets) == 1
            assert tweets[0].id == "pt_789123"

    @pytest.mark.asyncio
    async def test_create_promoted_tweet(
        self, twitter_ads_connector, mock_promoted_tweet_api_response
    ):
        """Test create_promoted_tweet creates a new promoted tweet."""
        with patch.object(twitter_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": [mock_promoted_tweet_api_response]}

            tweet = await twitter_ads_connector.create_promoted_tweet(
                line_item_id="li_456789",
                tweet_id="tweet_111222333",
            )

            assert tweet.id == "pt_789123"


# =============================================================================
# Analytics Operations Tests
# =============================================================================


class TestAnalyticsOperations:
    """Tests for analytics operations."""

    @pytest.mark.asyncio
    async def test_get_campaign_analytics(self, twitter_ads_connector, mock_analytics_api_response):
        """Test get_campaign_analytics returns performance data."""
        with patch.object(twitter_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": [mock_analytics_api_response]}

            analytics = await twitter_ads_connector.get_campaign_analytics(
                campaign_ids=["abc123"],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
            )

            assert len(analytics) == 1
            assert analytics[0].impressions == 100000

    @pytest.mark.asyncio
    async def test_get_campaign_analytics_with_granularity(self, twitter_ads_connector):
        """Test get_campaign_analytics with custom granularity."""
        with patch.object(twitter_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": []}

            await twitter_ads_connector.get_campaign_analytics(
                campaign_ids=["abc123"],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                granularity="DAY",
            )

            call_args = mock_request.call_args
            assert call_args[1]["params"]["granularity"] == "DAY"


# =============================================================================
# Audience Operations Tests
# =============================================================================


class TestAudienceOperations:
    """Tests for audience operations."""

    @pytest.mark.asyncio
    async def test_get_tailored_audiences(self, twitter_ads_connector, mock_audience_api_response):
        """Test get_tailored_audiences returns list of audiences."""
        with patch.object(twitter_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": [mock_audience_api_response]}

            audiences = await twitter_ads_connector.get_tailored_audiences()

            assert len(audiences) == 1
            assert audiences[0].id == "ta_123456"

    @pytest.mark.asyncio
    async def test_create_tailored_audience(
        self, twitter_ads_connector, mock_audience_api_response
    ):
        """Test create_tailored_audience creates a new audience."""
        with patch.object(twitter_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": mock_audience_api_response}

            audience = await twitter_ads_connector.create_tailored_audience(
                name="New Audience",
                audience_type="CRM",
            )

            assert audience.id == "ta_123456"


# =============================================================================
# Funding Instrument Operations Tests
# =============================================================================


class TestFundingInstrumentOperations:
    """Tests for funding instrument operations."""

    @pytest.mark.asyncio
    async def test_get_funding_instruments(
        self, twitter_ads_connector, mock_funding_instrument_api_response
    ):
        """Test get_funding_instruments returns list of funding instruments."""
        with patch.object(twitter_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"data": [mock_funding_instrument_api_response]}

            instruments = await twitter_ads_connector.get_funding_instruments()

            assert len(instruments) == 1
            assert instruments[0].id == "fi_987654"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestTwitterAdsError:
    """Tests for TwitterAdsError exception."""

    def test_error_creation(self):
        """Test TwitterAdsError creation."""
        from aragora.connectors.advertising.twitter_ads import TwitterAdsError

        error = TwitterAdsError(
            message="Authentication failed",
            status_code=401,
            error_code="UNAUTHORIZED",
        )

        assert str(error) == "Authentication failed"
        assert error.status_code == 401
        assert error.error_code == "UNAUTHORIZED"

    def test_error_minimal(self):
        """Test TwitterAdsError with minimal info."""
        from aragora.connectors.advertising.twitter_ads import TwitterAdsError

        error = TwitterAdsError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.status_code is None
        assert error.error_code is None


class TestAPIErrorHandling:
    """Tests for API error handling."""

    @pytest.mark.asyncio
    async def test_request_api_error(self, twitter_ads_connector):
        """Test _request handles API errors."""
        from aragora.connectors.advertising.twitter_ads import TwitterAdsError

        with patch.object(twitter_ads_connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.content = (
                b'{"errors": [{"message": "Invalid credentials", "code": "UNAUTHORIZED"}]}'
            )
            mock_response.json.return_value = {
                "errors": [{"message": "Invalid credentials", "code": "UNAUTHORIZED"}]
            }
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(TwitterAdsError, match="Invalid credentials"):
                await twitter_ads_connector._request("GET", "campaigns")

    @pytest.mark.asyncio
    async def test_request_api_error_empty_response(self, twitter_ads_connector):
        """Test _request handles empty error response."""
        from aragora.connectors.advertising.twitter_ads import TwitterAdsError

        with patch.object(twitter_ads_connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.content = b""
            mock_response.json.return_value = {}
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(TwitterAdsError, match="API error: 500"):
                await twitter_ads_connector._request("GET", "campaigns")


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_campaign_status_values(self):
        """Test CampaignStatus enum values."""
        from aragora.connectors.advertising.twitter_ads import CampaignStatus

        assert CampaignStatus.ACTIVE.value == "ACTIVE"
        assert CampaignStatus.PAUSED.value == "PAUSED"
        assert CampaignStatus.DRAFT.value == "DRAFT"
        assert CampaignStatus.SCHEDULED.value == "SCHEDULED"

    def test_campaign_objective_values(self):
        """Test CampaignObjective enum values."""
        from aragora.connectors.advertising.twitter_ads import CampaignObjective

        assert CampaignObjective.AWARENESS.value == "AWARENESS"
        assert CampaignObjective.TWEET_ENGAGEMENTS.value == "TWEET_ENGAGEMENTS"
        assert CampaignObjective.VIDEO_VIEWS.value == "VIDEO_VIEWS"
        assert CampaignObjective.WEBSITE_CLICKS.value == "WEBSITE_CLICKS"
        assert CampaignObjective.FOLLOWERS.value == "FOLLOWERS"

    def test_line_item_type_values(self):
        """Test LineItemType enum values."""
        from aragora.connectors.advertising.twitter_ads import LineItemType

        assert LineItemType.PROMOTED_TWEETS.value == "PROMOTED_TWEETS"
        assert LineItemType.PROMOTED_ACCOUNT.value == "PROMOTED_ACCOUNT"
        assert LineItemType.PROMOTED_TREND.value == "PROMOTED_TREND"

    def test_placement_type_values(self):
        """Test PlacementType enum values."""
        from aragora.connectors.advertising.twitter_ads import PlacementType

        assert PlacementType.ALL_ON_TWITTER.value == "ALL_ON_TWITTER"
        assert PlacementType.TWITTER_PROFILE.value == "TWITTER_PROFILE"
        assert PlacementType.TWITTER_TIMELINE.value == "TWITTER_TIMELINE"
        assert PlacementType.TWITTER_SEARCH.value == "TWITTER_SEARCH"
        assert PlacementType.PUBLISHER_NETWORK.value == "PUBLISHER_NETWORK"


# =============================================================================
# Mock Data Tests
# =============================================================================


class TestMockData:
    """Tests for mock data generation."""

    def test_get_mock_campaign(self):
        """Test get_mock_campaign."""
        from aragora.connectors.advertising.twitter_ads import get_mock_campaign

        campaign = get_mock_campaign()

        assert campaign.id == "abc123"
        assert campaign.name == "Test Twitter Campaign"
        assert campaign.daily_budget == 100.0

    def test_get_mock_analytics(self):
        """Test get_mock_analytics."""
        from aragora.connectors.advertising.twitter_ads import get_mock_analytics

        analytics = get_mock_analytics()

        assert analytics.impressions == 100000
        assert analytics.engagements == 5000
        assert analytics.clicks == 2500
        assert analytics.spend == 500.0
        assert analytics.engagement_rate == 5.0
