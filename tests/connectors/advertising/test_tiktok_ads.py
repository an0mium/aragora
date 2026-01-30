"""
Comprehensive tests for TikTok Ads Connector.

Tests for TikTok Ads API integration including:
- Client initialization and authentication
- Campaign operations (CRUD, status updates)
- Ad group operations
- Ad creative operations
- Audience management
- Performance analytics
- Pixel tracking
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
def tiktok_ads_credentials():
    """Create test credentials."""
    from aragora.connectors.advertising.tiktok_ads import TikTokAdsCredentials

    return TikTokAdsCredentials(
        access_token="test_access_token",
        advertiser_id="987654321",
        app_id="test_app_id",
        secret="test_secret",
    )


@pytest.fixture
def tiktok_ads_connector(tiktok_ads_credentials):
    """Create test connector."""
    from aragora.connectors.advertising.tiktok_ads import TikTokAdsConnector

    return TikTokAdsConnector(tiktok_ads_credentials)


@pytest.fixture
def mock_campaign_api_response():
    """Mock campaign API response."""
    return {
        "campaign_id": "123456789",
        "campaign_name": "Test TikTok Campaign",
        "advertiser_id": "987654321",
        "status": "ENABLE",
        "objective_type": "TRAFFIC",
        "budget": 500.0,
        "budget_mode": "BUDGET_MODE_DAY",
        "create_time": "2024-01-01T00:00:00Z",
        "modify_time": "2024-01-15T12:00:00Z",
    }


@pytest.fixture
def mock_ad_group_api_response():
    """Mock ad group API response."""
    return {
        "adgroup_id": "456789",
        "campaign_id": "123456789",
        "adgroup_name": "Test Ad Group",
        "advertiser_id": "987654321",
        "status": "ENABLE",
        "optimization_goal": "CLICK",
        "placement_type": "PLACEMENT_TYPE_AUTOMATIC",
        "billing_event": "CPC",
        "budget": 100.0,
        "bid_price": 0.50,
        "schedule_start_time": "2024-01-01T00:00:00Z",
        "schedule_end_time": "2024-03-31T23:59:59Z",
    }


@pytest.fixture
def mock_ad_api_response():
    """Mock ad creative API response."""
    return {
        "ad_id": "789123456",
        "adgroup_id": "456789",
        "ad_name": "Test Ad Creative",
        "status": "ENABLE",
        "call_to_action": "LEARN_MORE",
        "landing_page_url": "https://example.com/landing",
        "video_id": "vid_123456",
        "image_ids": ["img_001", "img_002"],
        "create_time": "2024-01-05T10:00:00Z",
    }


@pytest.fixture
def mock_metrics_api_response():
    """Mock metrics API response."""
    return {
        "dimensions": {
            "adgroup_id": "456789",
            "adgroup_name": "Test Ad Group",
            "campaign_id": "123456789",
        },
        "metrics": {
            "impressions": 500000,
            "clicks": 15000,
            "spend": 3000.0,
            "conversion": 450,
            "video_views": 200000,
            "video_views_p25": 150000,
            "video_views_p50": 100000,
            "video_views_p75": 50000,
            "video_views_p100": 25000,
            "reach": 300000,
            "frequency": 1.67,
        },
    }


@pytest.fixture
def mock_audience_api_response():
    """Mock custom audience API response."""
    return {
        "audience_id": "aud_123456",
        "name": "Test Audience",
        "advertiser_id": "987654321",
        "audience_type": "CUSTOMER_FILE",
        "audience_sub_type": "EMAIL",
        "is_valid": True,
        "audience_size": 50000,
        "create_time": "2024-01-10T08:00:00Z",
    }


@pytest.fixture
def mock_pixel_api_response():
    """Mock pixel API response."""
    return {
        "pixel_id": "pix_987654",
        "pixel_name": "Test Pixel",
        "advertiser_id": "987654321",
        "pixel_code": "<script>/* pixel code */</script>",
        "create_time": "2024-01-01T00:00:00Z",
    }


# =============================================================================
# Client Initialization Tests
# =============================================================================


class TestTikTokAdsCredentials:
    """Tests for TikTokAdsCredentials dataclass."""

    def test_credentials_creation(self):
        """Test credentials creation with required fields."""
        from aragora.connectors.advertising.tiktok_ads import TikTokAdsCredentials

        creds = TikTokAdsCredentials(
            access_token="token123",
            advertiser_id="adv456",
        )

        assert creds.access_token == "token123"
        assert creds.advertiser_id == "adv456"
        assert creds.app_id is None
        assert creds.secret is None

    def test_credentials_with_optional_fields(self):
        """Test credentials with all optional fields."""
        from aragora.connectors.advertising.tiktok_ads import TikTokAdsCredentials

        creds = TikTokAdsCredentials(
            access_token="token123",
            advertiser_id="adv456",
            app_id="app789",
            secret="secret_key",
        )

        assert creds.app_id == "app789"
        assert creds.secret == "secret_key"


class TestTikTokAdsConnectorInit:
    """Tests for TikTokAdsConnector initialization."""

    def test_connector_initialization(self, tiktok_ads_credentials):
        """Test connector initializes correctly."""
        from aragora.connectors.advertising.tiktok_ads import TikTokAdsConnector

        connector = TikTokAdsConnector(tiktok_ads_credentials)

        assert connector.credentials == tiktok_ads_credentials
        assert connector._client is None
        assert connector.BASE_URL == "https://business-api.tiktok.com/open_api/v1.3"

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self, tiktok_ads_connector):
        """Test _get_client creates HTTP client."""
        client = await tiktok_ads_connector._get_client()

        assert client is not None
        assert isinstance(client, httpx.AsyncClient)

        await tiktok_ads_connector.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_client(self, tiktok_ads_connector):
        """Test _get_client reuses existing client."""
        client1 = await tiktok_ads_connector._get_client()
        client2 = await tiktok_ads_connector._get_client()

        assert client1 is client2

        await tiktok_ads_connector.close()

    @pytest.mark.asyncio
    async def test_close_closes_client(self, tiktok_ads_connector):
        """Test close() closes HTTP client."""
        await tiktok_ads_connector._get_client()
        assert tiktok_ads_connector._client is not None

        await tiktok_ads_connector.close()
        assert tiktok_ads_connector._client is None


# =============================================================================
# Campaign Dataclass Tests
# =============================================================================


class TestCampaignDataclass:
    """Tests for Campaign dataclass."""

    def test_campaign_from_api(self, mock_campaign_api_response):
        """Test Campaign.from_api parsing."""
        from aragora.connectors.advertising.tiktok_ads import (
            Campaign,
            CampaignStatus,
            CampaignObjective,
        )

        campaign = Campaign.from_api(mock_campaign_api_response)

        assert campaign.id == "123456789"
        assert campaign.name == "Test TikTok Campaign"
        assert campaign.advertiser_id == "987654321"
        assert campaign.status == CampaignStatus.ENABLE
        assert campaign.objective == CampaignObjective.TRAFFIC
        assert campaign.budget == 500.0
        assert campaign.budget_mode == "BUDGET_MODE_DAY"
        assert campaign.create_time is not None
        assert campaign.modify_time is not None

    def test_campaign_from_api_minimal(self):
        """Test Campaign.from_api with minimal data."""
        from aragora.connectors.advertising.tiktok_ads import Campaign

        data = {
            "campaign_id": "123",
            "campaign_name": "Minimal Campaign",
        }

        campaign = Campaign.from_api(data)

        assert campaign.id == "123"
        assert campaign.name == "Minimal Campaign"
        assert campaign.budget is None
        assert campaign.create_time is None

    def test_campaign_from_api_no_budget(self):
        """Test Campaign.from_api without budget."""
        from aragora.connectors.advertising.tiktok_ads import Campaign

        data = {
            "campaign_id": "456",
            "campaign_name": "No Budget Campaign",
            "status": "ENABLE",
            "objective_type": "REACH",
        }

        campaign = Campaign.from_api(data)

        assert campaign.budget is None


# =============================================================================
# Ad Group Dataclass Tests
# =============================================================================


class TestAdGroupDataclass:
    """Tests for AdGroup dataclass."""

    def test_ad_group_from_api(self, mock_ad_group_api_response):
        """Test AdGroup.from_api parsing."""
        from aragora.connectors.advertising.tiktok_ads import (
            AdGroup,
            AdGroupStatus,
            OptimizationGoal,
            PlacementType,
            BillingEvent,
        )

        ad_group = AdGroup.from_api(mock_ad_group_api_response)

        assert ad_group.id == "456789"
        assert ad_group.campaign_id == "123456789"
        assert ad_group.name == "Test Ad Group"
        assert ad_group.status == AdGroupStatus.ENABLE
        assert ad_group.optimization_goal == OptimizationGoal.CLICK
        assert ad_group.placement_type == PlacementType.PLACEMENT_TYPE_AUTOMATIC
        assert ad_group.billing_event == BillingEvent.CPC
        assert ad_group.budget == 100.0
        assert ad_group.bid_price == 0.50

    def test_ad_group_from_api_minimal(self):
        """Test AdGroup.from_api with minimal data."""
        from aragora.connectors.advertising.tiktok_ads import AdGroup

        data = {
            "adgroup_id": "789",
            "campaign_id": "123",
            "adgroup_name": "Minimal Ad Group",
        }

        ad_group = AdGroup.from_api(data)

        assert ad_group.id == "789"
        assert ad_group.campaign_id == "123"
        assert ad_group.budget is None
        assert ad_group.schedule_start_time is None


# =============================================================================
# Ad Creative Dataclass Tests
# =============================================================================


class TestAdDataclass:
    """Tests for Ad dataclass."""

    def test_ad_from_api(self, mock_ad_api_response):
        """Test Ad.from_api parsing."""
        from aragora.connectors.advertising.tiktok_ads import Ad

        ad = Ad.from_api(mock_ad_api_response)

        assert ad.id == "789123456"
        assert ad.adgroup_id == "456789"
        assert ad.name == "Test Ad Creative"
        assert ad.status == "ENABLE"
        assert ad.call_to_action == "LEARN_MORE"
        assert ad.landing_page_url == "https://example.com/landing"
        assert ad.video_id == "vid_123456"
        assert len(ad.image_ids) == 2
        assert ad.create_time is not None

    def test_ad_from_api_minimal(self):
        """Test Ad.from_api with minimal data."""
        from aragora.connectors.advertising.tiktok_ads import Ad

        data = {
            "ad_id": "123",
            "adgroup_id": "456",
            "ad_name": "Minimal Ad",
        }

        ad = Ad.from_api(data)

        assert ad.id == "123"
        assert ad.video_id is None
        assert ad.image_ids == []


# =============================================================================
# Ad Group Metrics Dataclass Tests
# =============================================================================


class TestAdGroupMetricsDataclass:
    """Tests for AdGroupMetrics dataclass."""

    def test_ad_group_metrics_from_api(self, mock_metrics_api_response):
        """Test AdGroupMetrics.from_api parsing."""
        from aragora.connectors.advertising.tiktok_ads import AdGroupMetrics

        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)

        metrics = AdGroupMetrics.from_api(
            mock_metrics_api_response,
            mock_metrics_api_response["dimensions"],
            start_date,
            end_date,
        )

        assert metrics.adgroup_id == "456789"
        assert metrics.campaign_id == "123456789"
        assert metrics.impressions == 500000
        assert metrics.clicks == 15000
        assert metrics.spend == 3000.0
        assert metrics.conversions == 450
        assert metrics.video_views == 200000
        assert metrics.reach == 300000

    def test_ad_group_metrics_calculated_fields(self):
        """Test AdGroupMetrics calculated fields."""
        from aragora.connectors.advertising.tiktok_ads import AdGroupMetrics

        data = {
            "metrics": {
                "impressions": 10000,
                "clicks": 500,
                "spend": 100.0,
                "conversion": 50,
            }
        }
        dimensions = {"adgroup_id": "123", "adgroup_name": "Test", "campaign_id": "456"}

        metrics = AdGroupMetrics.from_api(data, dimensions, date(2024, 1, 1), date(2024, 1, 31))

        # CTR = clicks/impressions * 100
        assert metrics.ctr == 5.0
        # CPC = spend/clicks
        assert metrics.cpc == 0.2
        # CPM = spend/impressions * 1000
        assert metrics.cpm == 10.0
        # Conversion rate = conversions/clicks * 100
        assert metrics.conversion_rate == 10.0
        # Cost per conversion = spend/conversions
        assert metrics.cost_per_conversion == 2.0

    def test_ad_group_metrics_zero_division(self):
        """Test AdGroupMetrics handles zero division."""
        from aragora.connectors.advertising.tiktok_ads import AdGroupMetrics

        data = {"metrics": {"impressions": 0, "clicks": 0, "spend": 0, "conversion": 0}}
        dimensions = {"adgroup_id": "123", "adgroup_name": "Test", "campaign_id": "456"}

        metrics = AdGroupMetrics.from_api(data, dimensions, date(2024, 1, 1), date(2024, 1, 31))

        assert metrics.ctr == 0.0
        assert metrics.cpc == 0.0
        assert metrics.cpm == 0.0
        assert metrics.conversion_rate == 0.0
        assert metrics.cost_per_conversion == 0.0


# =============================================================================
# Custom Audience Dataclass Tests
# =============================================================================


class TestCustomAudienceDataclass:
    """Tests for CustomAudience dataclass."""

    def test_custom_audience_from_api(self, mock_audience_api_response):
        """Test CustomAudience.from_api parsing."""
        from aragora.connectors.advertising.tiktok_ads import CustomAudience

        audience = CustomAudience.from_api(mock_audience_api_response)

        assert audience.id == "aud_123456"
        assert audience.name == "Test Audience"
        assert audience.advertiser_id == "987654321"
        assert audience.audience_type == "CUSTOMER_FILE"
        assert audience.audience_sub_type == "EMAIL"
        assert audience.is_valid is True
        assert audience.audience_size == 50000

    def test_custom_audience_from_api_minimal(self):
        """Test CustomAudience.from_api with minimal data."""
        from aragora.connectors.advertising.tiktok_ads import CustomAudience

        data = {
            "audience_id": "aud_789",
            "name": "Minimal Audience",
            "audience_type": "LOOKALIKE",
        }

        audience = CustomAudience.from_api(data)

        assert audience.id == "aud_789"
        assert audience.audience_sub_type is None
        assert audience.audience_size is None


# =============================================================================
# Pixel Dataclass Tests
# =============================================================================


class TestPixelDataclass:
    """Tests for Pixel dataclass."""

    def test_pixel_from_api(self, mock_pixel_api_response):
        """Test Pixel.from_api parsing."""
        from aragora.connectors.advertising.tiktok_ads import Pixel

        pixel = Pixel.from_api(mock_pixel_api_response)

        assert pixel.id == "pix_987654"
        assert pixel.name == "Test Pixel"
        assert pixel.advertiser_id == "987654321"
        assert pixel.code == "<script>/* pixel code */</script>"
        assert pixel.create_time is not None

    def test_pixel_from_api_minimal(self):
        """Test Pixel.from_api with minimal data."""
        from aragora.connectors.advertising.tiktok_ads import Pixel

        data = {
            "pixel_id": "pix_123",
            "pixel_name": "Minimal Pixel",
        }

        pixel = Pixel.from_api(data)

        assert pixel.id == "pix_123"
        assert pixel.code is None


# =============================================================================
# Campaign Operations Tests
# =============================================================================


class TestCampaignOperations:
    """Tests for campaign CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_campaigns(self, tiktok_ads_connector, mock_campaign_api_response):
        """Test get_campaigns returns list of campaigns."""
        with patch.object(tiktok_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"list": [mock_campaign_api_response]}

            campaigns = await tiktok_ads_connector.get_campaigns()

            assert len(campaigns) == 1
            assert campaigns[0].id == "123456789"
            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_campaigns_with_status_filter(self, tiktok_ads_connector):
        """Test get_campaigns with status filter."""
        from aragora.connectors.advertising.tiktok_ads import CampaignStatus

        with patch.object(tiktok_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"list": []}

            await tiktok_ads_connector.get_campaigns(status=CampaignStatus.ENABLE)

            call_args = mock_request.call_args
            assert call_args[1]["params"]["primary_status"] == "ENABLE"

    @pytest.mark.asyncio
    async def test_get_campaigns_pagination(self, tiktok_ads_connector):
        """Test get_campaigns with pagination."""
        with patch.object(tiktok_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"list": []}

            await tiktok_ads_connector.get_campaigns(page=2, page_size=50)

            call_args = mock_request.call_args
            assert call_args[1]["params"]["page"] == 2
            assert call_args[1]["params"]["page_size"] == 50

    @pytest.mark.asyncio
    async def test_get_campaign_by_id(self, tiktok_ads_connector, mock_campaign_api_response):
        """Test get_campaign returns single campaign."""
        with patch.object(tiktok_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"list": [mock_campaign_api_response]}

            campaign = await tiktok_ads_connector.get_campaign("123456789")

            assert campaign.id == "123456789"

    @pytest.mark.asyncio
    async def test_get_campaign_not_found(self, tiktok_ads_connector):
        """Test get_campaign raises error when not found."""
        from aragora.connectors.advertising.tiktok_ads import TikTokAdsError

        with patch.object(tiktok_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"list": []}

            with pytest.raises(TikTokAdsError, match="not found"):
                await tiktok_ads_connector.get_campaign("nonexistent")

    @pytest.mark.asyncio
    async def test_create_campaign(self, tiktok_ads_connector, mock_campaign_api_response):
        """Test create_campaign creates a new campaign."""
        from aragora.connectors.advertising.tiktok_ads import CampaignObjective

        with patch.object(tiktok_ads_connector, "_request") as mock_request:
            mock_request.side_effect = [
                {"campaign_id": "123456789"},
                {"list": [mock_campaign_api_response]},
            ]

            campaign = await tiktok_ads_connector.create_campaign(
                name="New Campaign",
                objective=CampaignObjective.TRAFFIC,
                budget=500.0,
            )

            assert campaign.id == "123456789"

    @pytest.mark.asyncio
    async def test_update_campaign_status(self, tiktok_ads_connector):
        """Test update_campaign_status."""
        from aragora.connectors.advertising.tiktok_ads import CampaignStatus

        with patch.object(tiktok_ads_connector, "_request") as mock_request:
            mock_request.return_value = {}

            await tiktok_ads_connector.update_campaign_status("123456789", CampaignStatus.DISABLE)

            call_args = mock_request.call_args
            assert call_args[1]["json_data"]["operation_status"] == "DISABLE"


# =============================================================================
# Ad Group Operations Tests
# =============================================================================


class TestAdGroupOperations:
    """Tests for ad group operations."""

    @pytest.mark.asyncio
    async def test_get_ad_groups(self, tiktok_ads_connector, mock_ad_group_api_response):
        """Test get_ad_groups returns list of ad groups."""
        with patch.object(tiktok_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"list": [mock_ad_group_api_response]}

            ad_groups = await tiktok_ads_connector.get_ad_groups()

            assert len(ad_groups) == 1
            assert ad_groups[0].id == "456789"

    @pytest.mark.asyncio
    async def test_get_ad_groups_by_campaign(self, tiktok_ads_connector):
        """Test get_ad_groups filtered by campaign."""
        with patch.object(tiktok_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"list": []}

            await tiktok_ads_connector.get_ad_groups(campaign_id="123456789")

            call_args = mock_request.call_args
            assert call_args[1]["params"]["campaign_ids"] == ["123456789"]

    @pytest.mark.asyncio
    async def test_create_ad_group(self, tiktok_ads_connector, mock_ad_group_api_response):
        """Test create_ad_group creates a new ad group."""
        from aragora.connectors.advertising.tiktok_ads import (
            OptimizationGoal,
            PlacementType,
            BillingEvent,
        )

        with patch.object(tiktok_ads_connector, "_request") as mock_request:
            mock_request.side_effect = [
                {"adgroup_id": "456789"},
                {"list": [mock_ad_group_api_response]},
            ]

            ad_group = await tiktok_ads_connector.create_ad_group(
                campaign_id="123456789",
                name="New Ad Group",
                optimization_goal=OptimizationGoal.CLICK,
                placement_type=PlacementType.PLACEMENT_TYPE_TIKTOK,
                billing_event=BillingEvent.CPC,
                budget=100.0,
            )

            assert ad_group.id == "456789"


# =============================================================================
# Ad Operations Tests
# =============================================================================


class TestAdOperations:
    """Tests for ad operations."""

    @pytest.mark.asyncio
    async def test_get_ads(self, tiktok_ads_connector, mock_ad_api_response):
        """Test get_ads returns list of ads."""
        with patch.object(tiktok_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"list": [mock_ad_api_response]}

            ads = await tiktok_ads_connector.get_ads()

            assert len(ads) == 1
            assert ads[0].id == "789123456"

    @pytest.mark.asyncio
    async def test_get_ads_by_adgroup(self, tiktok_ads_connector):
        """Test get_ads filtered by ad group."""
        with patch.object(tiktok_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"list": []}

            await tiktok_ads_connector.get_ads(adgroup_id="456789")

            call_args = mock_request.call_args
            assert call_args[1]["params"]["adgroup_ids"] == ["456789"]


# =============================================================================
# Analytics Tests
# =============================================================================


class TestAnalytics:
    """Tests for analytics operations."""

    @pytest.mark.asyncio
    async def test_get_ad_group_metrics(self, tiktok_ads_connector, mock_metrics_api_response):
        """Test get_ad_group_metrics returns performance data."""
        with patch.object(tiktok_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"list": [mock_metrics_api_response]}

            metrics = await tiktok_ads_connector.get_ad_group_metrics(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
            )

            assert len(metrics) == 1
            assert metrics[0].impressions == 500000

    @pytest.mark.asyncio
    async def test_get_ad_group_metrics_with_campaign_filter(self, tiktok_ads_connector):
        """Test get_ad_group_metrics with campaign filter."""
        with patch.object(tiktok_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"list": []}

            await tiktok_ads_connector.get_ad_group_metrics(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                campaign_id="123456789",
            )

            call_args = mock_request.call_args
            json_data = call_args[1]["json_data"]
            assert "filters" in json_data
            assert json_data["filters"][0]["field_name"] == "campaign_id"

    @pytest.mark.asyncio
    async def test_get_ad_group_metrics_with_adgroup_filter(self, tiktok_ads_connector):
        """Test get_ad_group_metrics with ad group filter."""
        with patch.object(tiktok_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"list": []}

            await tiktok_ads_connector.get_ad_group_metrics(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                adgroup_ids=["456789", "456790"],
            )

            call_args = mock_request.call_args
            json_data = call_args[1]["json_data"]
            assert "filters" in json_data
            assert json_data["filters"][0]["field_name"] == "adgroup_id"


# =============================================================================
# Audience Operations Tests
# =============================================================================


class TestAudienceOperations:
    """Tests for audience operations."""

    @pytest.mark.asyncio
    async def test_get_custom_audiences(self, tiktok_ads_connector, mock_audience_api_response):
        """Test get_custom_audiences returns list of audiences."""
        with patch.object(tiktok_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"list": [mock_audience_api_response]}

            audiences = await tiktok_ads_connector.get_custom_audiences()

            assert len(audiences) == 1
            assert audiences[0].id == "aud_123456"

    @pytest.mark.asyncio
    async def test_create_custom_audience(self, tiktok_ads_connector, mock_audience_api_response):
        """Test create_custom_audience creates a new audience."""
        with patch.object(tiktok_ads_connector, "_request") as mock_request:
            mock_request.side_effect = [
                {"audience_id": "aud_123456"},
                {"list": [mock_audience_api_response]},
            ]

            audience = await tiktok_ads_connector.create_custom_audience(
                name="New Audience",
                audience_type="CUSTOMER_FILE",
            )

            assert audience.id == "aud_123456"


# =============================================================================
# Pixel Operations Tests
# =============================================================================


class TestPixelOperations:
    """Tests for pixel operations."""

    @pytest.mark.asyncio
    async def test_get_pixels(self, tiktok_ads_connector, mock_pixel_api_response):
        """Test get_pixels returns list of pixels."""
        with patch.object(tiktok_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"pixels": [mock_pixel_api_response]}

            pixels = await tiktok_ads_connector.get_pixels()

            assert len(pixels) == 1
            assert pixels[0].id == "pix_987654"

    @pytest.mark.asyncio
    async def test_create_pixel(self, tiktok_ads_connector, mock_pixel_api_response):
        """Test create_pixel creates a new pixel."""
        with patch.object(tiktok_ads_connector, "_request") as mock_request:
            mock_request.side_effect = [
                {"pixel_id": "pix_987654"},
                {"pixels": [mock_pixel_api_response]},
            ]

            pixel = await tiktok_ads_connector.create_pixel(name="New Pixel")

            assert pixel.id == "pix_987654"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestTikTokAdsError:
    """Tests for TikTokAdsError exception."""

    def test_error_creation(self):
        """Test TikTokAdsError creation."""
        from aragora.connectors.advertising.tiktok_ads import TikTokAdsError

        error = TikTokAdsError(
            message="Authentication failed",
            status_code=401,
            error_code=40001,
        )

        assert str(error) == "Authentication failed"
        assert error.status_code == 401
        assert error.error_code == 40001

    def test_error_minimal(self):
        """Test TikTokAdsError with minimal info."""
        from aragora.connectors.advertising.tiktok_ads import TikTokAdsError

        error = TikTokAdsError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.status_code is None
        assert error.error_code is None


class TestAPIErrorHandling:
    """Tests for API error handling."""

    @pytest.mark.asyncio
    async def test_request_api_error(self, tiktok_ads_connector):
        """Test _request handles API errors."""
        from aragora.connectors.advertising.tiktok_ads import TikTokAdsError

        with patch.object(tiktok_ads_connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "code": 40001,
                "message": "Invalid access token",
            }
            mock_response.status_code = 401
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(TikTokAdsError, match="Invalid access token"):
                await tiktok_ads_connector._request("GET", "campaign/get/")

    @pytest.mark.asyncio
    async def test_request_adds_advertiser_id_to_params(self, tiktok_ads_connector):
        """Test _request adds advertiser_id to params."""
        with patch.object(tiktok_ads_connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"code": 0, "data": {}}
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await tiktok_ads_connector._request("GET", "test/endpoint", params={})

            call_args = mock_client.request.call_args
            assert call_args[1]["params"]["advertiser_id"] == "987654321"

    @pytest.mark.asyncio
    async def test_request_adds_advertiser_id_to_json_data(self, tiktok_ads_connector):
        """Test _request adds advertiser_id to json_data."""
        with patch.object(tiktok_ads_connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"code": 0, "data": {}}
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await tiktok_ads_connector._request("POST", "test/endpoint", json_data={})

            call_args = mock_client.request.call_args
            assert call_args[1]["json"]["advertiser_id"] == "987654321"


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_campaign_status_values(self):
        """Test CampaignStatus enum values."""
        from aragora.connectors.advertising.tiktok_ads import CampaignStatus

        assert CampaignStatus.ENABLE.value == "ENABLE"
        assert CampaignStatus.DISABLE.value == "DISABLE"
        assert CampaignStatus.DELETE.value == "DELETE"

    def test_campaign_objective_values(self):
        """Test CampaignObjective enum values."""
        from aragora.connectors.advertising.tiktok_ads import CampaignObjective

        assert CampaignObjective.TRAFFIC.value == "TRAFFIC"
        assert CampaignObjective.REACH.value == "REACH"
        assert CampaignObjective.VIDEO_VIEWS.value == "VIDEO_VIEWS"
        assert CampaignObjective.CONVERSIONS.value == "CONVERSIONS"
        assert CampaignObjective.APP_INSTALL.value == "APP_INSTALL"

    def test_optimization_goal_values(self):
        """Test OptimizationGoal enum values."""
        from aragora.connectors.advertising.tiktok_ads import OptimizationGoal

        assert OptimizationGoal.CLICK.value == "CLICK"
        assert OptimizationGoal.SHOW.value == "SHOW"
        assert OptimizationGoal.CONVERT.value == "CONVERT"
        assert OptimizationGoal.VIDEO_VIEW.value == "VIDEO_VIEW"

    def test_placement_type_values(self):
        """Test PlacementType enum values."""
        from aragora.connectors.advertising.tiktok_ads import PlacementType

        assert PlacementType.PLACEMENT_TYPE_TIKTOK.value == "PLACEMENT_TYPE_TIKTOK"
        assert PlacementType.PLACEMENT_TYPE_PANGLE.value == "PLACEMENT_TYPE_PANGLE"
        assert PlacementType.PLACEMENT_TYPE_AUTOMATIC.value == "PLACEMENT_TYPE_AUTOMATIC"

    def test_billing_event_values(self):
        """Test BillingEvent enum values."""
        from aragora.connectors.advertising.tiktok_ads import BillingEvent

        assert BillingEvent.CPM.value == "CPM"
        assert BillingEvent.CPC.value == "CPC"
        assert BillingEvent.OCPM.value == "OCPM"
        assert BillingEvent.CPV.value == "CPV"


# =============================================================================
# Mock Data Tests
# =============================================================================


class TestMockData:
    """Tests for mock data generation."""

    def test_get_mock_campaign(self):
        """Test get_mock_campaign."""
        from aragora.connectors.advertising.tiktok_ads import get_mock_campaign

        campaign = get_mock_campaign()

        assert campaign.id == "123456789"
        assert campaign.name == "Test TikTok Campaign"
        assert campaign.budget == 500.0

    def test_get_mock_metrics(self):
        """Test get_mock_metrics."""
        from aragora.connectors.advertising.tiktok_ads import get_mock_metrics

        metrics = get_mock_metrics()

        assert metrics.impressions == 500000
        assert metrics.clicks == 15000
        assert metrics.conversions == 450
        assert metrics.video_views == 200000
