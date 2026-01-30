"""
Comprehensive tests for LinkedIn Ads Connector.

Tests for LinkedIn Marketing API integration including:
- Client initialization and configuration
- Campaign group operations
- Campaign operations (CRUD, status updates)
- Creative operations
- Analytics and reporting
- Lead Gen Forms and lead retrieval
- Audience segment operations
- Conversion tracking
- Error handling
"""

import pytest
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def linkedin_ads_credentials():
    """Create test credentials."""
    from aragora.connectors.advertising.linkedin_ads import LinkedInAdsCredentials

    return LinkedInAdsCredentials(
        access_token="test_access_token",
        ad_account_id="123456789",
        client_id="test_client_id",
        client_secret="test_client_secret",
        refresh_token="test_refresh_token",
    )


@pytest.fixture
def linkedin_ads_connector(linkedin_ads_credentials):
    """Create test connector."""
    from aragora.connectors.advertising.linkedin_ads import LinkedInAdsConnector

    return LinkedInAdsConnector(linkedin_ads_credentials)


@pytest.fixture
def mock_campaign_group_response():
    """Mock campaign group API response."""
    return {
        "id": "987654321",
        "name": "Test Campaign Group",
        "status": "ACTIVE",
        "account": "urn:li:sponsoredAccount:123456789",
        "totalBudget": {"amount": "5000.00", "currencyCode": "USD"},
        "runSchedule": {
            "start": "2024-01-01T00:00:00",
            "end": "2024-12-31T23:59:59",
        },
        "createdAt": 1704067200000,
        "lastModifiedAt": 1704153600000,
    }


@pytest.fixture
def mock_campaign_response():
    """Mock campaign API response."""
    return {
        "id": "111222333",
        "name": "Test B2B Campaign",
        "status": "ACTIVE",
        "campaignGroup": "urn:li:sponsoredCampaignGroup:987654321",
        "account": "urn:li:sponsoredAccount:123456789",
        "type": "SPONSORED_UPDATES",
        "objectiveType": "LEAD_GENERATION",
        "dailyBudget": {"amount": "100.00", "currencyCode": "USD"},
        "totalBudget": {"amount": "3000.00", "currencyCode": "USD"},
        "optimizationTargetType": "MAXIMUM_DELIVERY",
        "unitCost": {"amount": "5.00", "currencyCode": "USD"},
        "targetingCriteria": {
            "include": {"and": [{"or": {"urn:li:adTargetingFacet:titles": ["12345"]}}]}
        },
        "runSchedule": {
            "start": "2024-01-01T00:00:00",
            "end": "2024-06-30T23:59:59",
        },
        "createdAt": 1704067200000,
        "lastModifiedAt": 1704153600000,
    }


@pytest.fixture
def mock_creative_response():
    """Mock creative API response."""
    return {
        "id": "444555666",
        "campaign": "urn:li:sponsoredCampaign:111222333",
        "status": "ACTIVE",
        "type": "SINGLE_IMAGE_AD",
        "reference": "urn:li:share:123456789",
        "callToAction": {
            "action": "LEARN_MORE",
            "destinationUrl": "https://example.com/landing",
        },
        "createdAt": 1704067200000,
    }


@pytest.fixture
def mock_analytics_response():
    """Mock analytics API response."""
    return {
        "pivotValue": "urn:li:sponsoredCampaign:111222333",
        "impressions": 50000,
        "clicks": 1250,
        "costInLocalCurrency": 2500.0,
        "externalWebsiteConversions": 75,
        "oneClickLeads": 40,
        "leadGenerationMailContactInfoShares": 5,
        "videoViews": 0,
        "videoCompletions": 0,
        "totalEngagements": 320,
        "follows": 15,
    }


@pytest.fixture
def mock_lead_gen_form_response():
    """Mock lead gen form API response."""
    return {
        "id": "777888999",
        "name": "Contact Request Form",
        "status": "ACTIVE",
        "account": "urn:li:sponsoredAccount:123456789",
        "headline": {"text": "Get in touch with us"},
        "description": {"text": "Fill out this form to learn more"},
        "privacyPolicy": {"url": "https://example.com/privacy"},
        "thankYouMessage": {"message": "Thank you for your interest!"},
        "questions": [
            {"questionType": "FIRST_NAME"},
            {"questionType": "LAST_NAME"},
            {"questionType": "EMAIL"},
        ],
        "createdAt": 1704067200000,
    }


@pytest.fixture
def mock_lead_response():
    """Mock lead API response."""
    return {
        "id": "lead_123456",
        "leadGenFormUrn": "urn:li:leadGenForm:777888999",
        "sponsoredCampaign": "urn:li:sponsoredCampaign:111222333",
        "submittedAt": 1704240000000,
        "formResponse": {
            "FIRST_NAME": "John",
            "LAST_NAME": "Doe",
            "EMAIL": "john.doe@example.com",
        },
        "owner": "urn:li:person:abcdef123",
    }


@pytest.fixture
def mock_audience_response():
    """Mock audience segment API response."""
    return {
        "id": "aud_123456",
        "name": "Website Visitors",
        "type": "REMARKETING",
        "account": "urn:li:sponsoredAccount:123456789",
        "matchedMemberCount": 15000,
        "status": "READY",
        "createdAt": 1704067200000,
    }


# =============================================================================
# Credentials Tests
# =============================================================================


class TestLinkedInAdsCredentials:
    """Tests for LinkedInAdsCredentials dataclass."""

    def test_credentials_creation(self):
        """Test credentials creation with required fields."""
        from aragora.connectors.advertising.linkedin_ads import LinkedInAdsCredentials

        creds = LinkedInAdsCredentials(
            access_token="access_token",
            ad_account_id="123456789",
        )

        assert creds.access_token == "access_token"
        assert creds.ad_account_id == "123456789"
        assert creds.client_id is None
        assert creds.client_secret is None
        assert creds.refresh_token is None

    def test_credentials_with_oauth(self):
        """Test credentials with OAuth fields."""
        from aragora.connectors.advertising.linkedin_ads import LinkedInAdsCredentials

        creds = LinkedInAdsCredentials(
            access_token="access_token",
            ad_account_id="123456789",
            client_id="client_id",
            client_secret="client_secret",
            refresh_token="refresh_token",
        )

        assert creds.client_id == "client_id"
        assert creds.client_secret == "client_secret"
        assert creds.refresh_token == "refresh_token"


# =============================================================================
# Connector Initialization Tests
# =============================================================================


class TestLinkedInAdsConnectorInit:
    """Tests for LinkedInAdsConnector initialization."""

    def test_connector_initialization(self, linkedin_ads_credentials):
        """Test connector initializes correctly."""
        from aragora.connectors.advertising.linkedin_ads import LinkedInAdsConnector

        connector = LinkedInAdsConnector(linkedin_ads_credentials)

        assert connector.credentials == linkedin_ads_credentials
        assert connector._client is None
        assert connector.BASE_URL == "https://api.linkedin.com/v2"
        assert connector.REST_URL == "https://api.linkedin.com/rest"

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self, linkedin_ads_connector):
        """Test _get_client creates HTTP client."""
        client = await linkedin_ads_connector._get_client()

        assert client is not None
        assert isinstance(client, httpx.AsyncClient)

        await linkedin_ads_connector.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_client(self, linkedin_ads_connector):
        """Test _get_client reuses existing client."""
        client1 = await linkedin_ads_connector._get_client()
        client2 = await linkedin_ads_connector._get_client()

        assert client1 is client2

        await linkedin_ads_connector.close()

    @pytest.mark.asyncio
    async def test_close_closes_client(self, linkedin_ads_connector):
        """Test close() closes HTTP client."""
        await linkedin_ads_connector._get_client()
        assert linkedin_ads_connector._client is not None

        await linkedin_ads_connector.close()
        assert linkedin_ads_connector._client is None


# =============================================================================
# Campaign Group Tests
# =============================================================================


class TestCampaignGroupDataclass:
    """Tests for CampaignGroup dataclass."""

    def test_campaign_group_from_api(self, mock_campaign_group_response):
        """Test CampaignGroup.from_api parsing."""
        from aragora.connectors.advertising.linkedin_ads import (
            CampaignGroup,
            CampaignStatus,
        )

        campaign_group = CampaignGroup.from_api(mock_campaign_group_response)

        assert campaign_group.id == "987654321"
        assert campaign_group.name == "Test Campaign Group"
        assert campaign_group.status == CampaignStatus.ACTIVE
        assert campaign_group.account_id == "123456789"
        assert campaign_group.total_budget == "5000.00"
        assert campaign_group.run_schedule_start is not None
        assert campaign_group.run_schedule_end is not None
        assert campaign_group.created_at is not None

    def test_campaign_group_from_api_minimal(self):
        """Test CampaignGroup.from_api with minimal data."""
        from aragora.connectors.advertising.linkedin_ads import CampaignGroup

        data = {
            "id": "123",
            "name": "Minimal Group",
            "status": "DRAFT",
            "account": "",
        }

        campaign_group = CampaignGroup.from_api(data)

        assert campaign_group.id == "123"
        assert campaign_group.name == "Minimal Group"
        assert campaign_group.total_budget is None


class TestCampaignGroupOperations:
    """Tests for campaign group operations."""

    @pytest.mark.asyncio
    async def test_get_campaign_groups(self, linkedin_ads_connector, mock_campaign_group_response):
        """Test get_campaign_groups returns list of campaign groups."""
        with patch.object(linkedin_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"elements": [mock_campaign_group_response]}

            campaign_groups = await linkedin_ads_connector.get_campaign_groups()

            assert len(campaign_groups) == 1
            assert campaign_groups[0].id == "987654321"
            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_campaign_groups_with_status_filter(self, linkedin_ads_connector):
        """Test get_campaign_groups with status filter."""
        from aragora.connectors.advertising.linkedin_ads import CampaignStatus

        with patch.object(linkedin_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"elements": []}

            await linkedin_ads_connector.get_campaign_groups(status=CampaignStatus.ACTIVE)

            call_args = mock_request.call_args
            assert "ACTIVE" in str(call_args)

    @pytest.mark.asyncio
    async def test_create_campaign_group(
        self, linkedin_ads_connector, mock_campaign_group_response
    ):
        """Test create_campaign_group."""
        with patch.object(linkedin_ads_connector, "_request") as mock_request:
            mock_request.return_value = mock_campaign_group_response

            campaign_group = await linkedin_ads_connector.create_campaign_group(
                name="New Campaign Group",
                total_budget=5000.0,
            )

            assert campaign_group.name == "Test Campaign Group"
            mock_request.assert_called_once()


# =============================================================================
# Campaign Tests
# =============================================================================


class TestCampaignDataclass:
    """Tests for Campaign dataclass."""

    def test_campaign_from_api(self, mock_campaign_response):
        """Test Campaign.from_api parsing."""
        from aragora.connectors.advertising.linkedin_ads import (
            Campaign,
            CampaignStatus,
            CampaignType,
            ObjectiveType,
            BidStrategy,
        )

        campaign = Campaign.from_api(mock_campaign_response)

        assert campaign.id == "111222333"
        assert campaign.name == "Test B2B Campaign"
        assert campaign.status == CampaignStatus.ACTIVE
        assert campaign.campaign_group_id == "987654321"
        assert campaign.account_id == "123456789"
        assert campaign.campaign_type == CampaignType.SPONSORED_UPDATES
        assert campaign.objective_type == ObjectiveType.LEAD_GENERATION
        assert campaign.daily_budget == "100.00"
        assert campaign.bid_strategy == BidStrategy.MAXIMUM_DELIVERY
        assert campaign.targeting_criteria is not None

    def test_campaign_from_api_minimal(self):
        """Test Campaign.from_api with minimal data."""
        from aragora.connectors.advertising.linkedin_ads import Campaign

        data = {
            "id": "123",
            "name": "Minimal Campaign",
            "status": "DRAFT",
            "campaignGroup": "",
            "account": "",
        }

        campaign = Campaign.from_api(data)

        assert campaign.id == "123"
        assert campaign.name == "Minimal Campaign"
        assert campaign.daily_budget is None


class TestCampaignOperations:
    """Tests for campaign operations."""

    @pytest.mark.asyncio
    async def test_get_campaigns(self, linkedin_ads_connector, mock_campaign_response):
        """Test get_campaigns returns list of campaigns."""
        with patch.object(linkedin_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"elements": [mock_campaign_response]}

            campaigns = await linkedin_ads_connector.get_campaigns()

            assert len(campaigns) == 1
            assert campaigns[0].id == "111222333"

    @pytest.mark.asyncio
    async def test_get_campaigns_with_status_filter(self, linkedin_ads_connector):
        """Test get_campaigns with status filter."""
        from aragora.connectors.advertising.linkedin_ads import CampaignStatus

        with patch.object(linkedin_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"elements": []}

            await linkedin_ads_connector.get_campaigns(status=CampaignStatus.ACTIVE)

            call_args = mock_request.call_args
            assert "ACTIVE" in str(call_args)

    @pytest.mark.asyncio
    async def test_get_campaigns_with_campaign_group_filter(self, linkedin_ads_connector):
        """Test get_campaigns with campaign group filter."""
        with patch.object(linkedin_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"elements": []}

            await linkedin_ads_connector.get_campaigns(campaign_group_id="987654321")

            call_args = mock_request.call_args
            assert "987654321" in str(call_args)

    @pytest.mark.asyncio
    async def test_get_campaign_by_id(self, linkedin_ads_connector, mock_campaign_response):
        """Test get_campaign returns single campaign."""
        with patch.object(linkedin_ads_connector, "_request") as mock_request:
            mock_request.return_value = mock_campaign_response

            campaign = await linkedin_ads_connector.get_campaign("111222333")

            assert campaign.id == "111222333"

    @pytest.mark.asyncio
    async def test_create_campaign(self, linkedin_ads_connector, mock_campaign_response):
        """Test create_campaign."""
        from aragora.connectors.advertising.linkedin_ads import (
            CampaignType,
            ObjectiveType,
        )

        with patch.object(linkedin_ads_connector, "_request") as mock_request:
            mock_request.return_value = mock_campaign_response

            campaign = await linkedin_ads_connector.create_campaign(
                name="New Campaign",
                campaign_group_id="987654321",
                campaign_type=CampaignType.SPONSORED_UPDATES,
                objective_type=ObjectiveType.LEAD_GENERATION,
                daily_budget=100.0,
            )

            assert campaign.id == "111222333"
            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_campaign_status(self, linkedin_ads_connector, mock_campaign_response):
        """Test update_campaign_status."""
        from aragora.connectors.advertising.linkedin_ads import CampaignStatus

        with patch.object(linkedin_ads_connector, "_request") as mock_request:
            mock_request.return_value = mock_campaign_response

            campaign = await linkedin_ads_connector.update_campaign_status(
                "111222333", CampaignStatus.PAUSED
            )

            assert campaign is not None
            mock_request.assert_called_once()


# =============================================================================
# Creative Tests
# =============================================================================


class TestCreativeDataclass:
    """Tests for Creative dataclass."""

    def test_creative_from_api(self, mock_creative_response):
        """Test Creative.from_api parsing."""
        from aragora.connectors.advertising.linkedin_ads import Creative, AdFormat

        creative = Creative.from_api(mock_creative_response)

        assert creative.id == "444555666"
        assert creative.campaign_id == "111222333"
        assert creative.status == "ACTIVE"
        assert creative.ad_format == AdFormat.SINGLE_IMAGE
        assert creative.reference == "urn:li:share:123456789"
        assert creative.call_to_action == "LEARN_MORE"
        assert creative.landing_page_url == "https://example.com/landing"


class TestCreativeOperations:
    """Tests for creative operations."""

    @pytest.mark.asyncio
    async def test_get_creatives(self, linkedin_ads_connector, mock_creative_response):
        """Test get_creatives returns list of creatives."""
        with patch.object(linkedin_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"elements": [mock_creative_response]}

            creatives = await linkedin_ads_connector.get_creatives("111222333")

            assert len(creatives) == 1
            assert creatives[0].id == "444555666"

    @pytest.mark.asyncio
    async def test_create_creative(self, linkedin_ads_connector, mock_creative_response):
        """Test create_creative."""
        from aragora.connectors.advertising.linkedin_ads import AdFormat

        with patch.object(linkedin_ads_connector, "_request") as mock_request:
            mock_request.return_value = mock_creative_response

            creative = await linkedin_ads_connector.create_creative(
                campaign_id="111222333",
                ad_format=AdFormat.SINGLE_IMAGE,
                reference="urn:li:share:123456789",
                call_to_action="LEARN_MORE",
                landing_page_url="https://example.com/landing",
            )

            assert creative.id == "444555666"
            mock_request.assert_called_once()


# =============================================================================
# Analytics Tests
# =============================================================================


class TestAdAnalyticsDataclass:
    """Tests for AdAnalytics dataclass."""

    def test_ad_analytics_from_api(self, mock_analytics_response):
        """Test AdAnalytics.from_api parsing."""
        from aragora.connectors.advertising.linkedin_ads import AdAnalytics

        start = date(2024, 1, 1)
        end = date(2024, 1, 31)

        analytics = AdAnalytics.from_api(mock_analytics_response, "111222333", start, end)

        assert analytics.campaign_id == "111222333"
        assert analytics.impressions == 50000
        assert analytics.clicks == 1250
        assert analytics.cost == 2500.0
        assert analytics.conversions == 75
        assert analytics.leads == 45  # 40 + 5
        assert analytics.engagement == 320
        assert analytics.follows == 15
        # Calculated metrics
        assert analytics.ctr == 2.5  # 1250/50000 * 100
        assert analytics.cpc == 2.0  # 2500/1250
        assert analytics.cpm == 50.0  # 2500/50000 * 1000

    def test_ad_analytics_from_api_zero_impressions(self):
        """Test AdAnalytics.from_api with zero impressions."""
        from aragora.connectors.advertising.linkedin_ads import AdAnalytics

        data = {"impressions": 0, "clicks": 0}
        start = date(2024, 1, 1)
        end = date(2024, 1, 31)

        analytics = AdAnalytics.from_api(data, "123", start, end)

        assert analytics.ctr == 0.0
        assert analytics.cpc == 0.0
        assert analytics.cpm == 0.0


class TestAnalyticsOperations:
    """Tests for analytics operations."""

    @pytest.mark.asyncio
    async def test_get_campaign_analytics(self, linkedin_ads_connector, mock_analytics_response):
        """Test get_campaign_analytics."""
        with patch.object(linkedin_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"elements": [mock_analytics_response]}

            analytics = await linkedin_ads_connector.get_campaign_analytics(
                campaign_ids=["111222333"],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
            )

            assert len(analytics) == 1
            assert analytics[0].impressions == 50000

    @pytest.mark.asyncio
    async def test_get_account_analytics(self, linkedin_ads_connector, mock_analytics_response):
        """Test get_account_analytics."""
        with patch.object(linkedin_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"elements": [mock_analytics_response]}

            analytics = await linkedin_ads_connector.get_account_analytics(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
            )

            assert analytics.impressions == 50000


# =============================================================================
# Lead Gen Tests
# =============================================================================


class TestLeadGenFormDataclass:
    """Tests for LeadGenForm dataclass."""

    def test_lead_gen_form_from_api(self, mock_lead_gen_form_response):
        """Test LeadGenForm.from_api parsing."""
        from aragora.connectors.advertising.linkedin_ads import LeadGenForm

        form = LeadGenForm.from_api(mock_lead_gen_form_response)

        assert form.id == "777888999"
        assert form.name == "Contact Request Form"
        assert form.status == "ACTIVE"
        assert form.account_id == "123456789"
        assert form.headline == "Get in touch with us"
        assert form.description == "Fill out this form to learn more"
        assert len(form.questions) == 3


class TestLeadDataclass:
    """Tests for Lead dataclass."""

    def test_lead_from_api(self, mock_lead_response):
        """Test Lead.from_api parsing."""
        from aragora.connectors.advertising.linkedin_ads import Lead

        lead = Lead.from_api(mock_lead_response)

        assert lead.id == "lead_123456"
        assert lead.form_id == "777888999"
        assert lead.campaign_id == "111222333"
        assert lead.form_response["FIRST_NAME"] == "John"
        assert lead.member_urn == "urn:li:person:abcdef123"


class TestLeadGenOperations:
    """Tests for lead gen operations."""

    @pytest.mark.asyncio
    async def test_get_lead_gen_forms(self, linkedin_ads_connector, mock_lead_gen_form_response):
        """Test get_lead_gen_forms."""
        with patch.object(linkedin_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"elements": [mock_lead_gen_form_response]}

            forms = await linkedin_ads_connector.get_lead_gen_forms()

            assert len(forms) == 1
            assert forms[0].name == "Contact Request Form"

    @pytest.mark.asyncio
    async def test_get_leads(self, linkedin_ads_connector, mock_lead_response):
        """Test get_leads."""
        with patch.object(linkedin_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"elements": [mock_lead_response]}

            leads = await linkedin_ads_connector.get_leads("777888999")

            assert len(leads) == 1
            assert leads[0].form_response["EMAIL"] == "john.doe@example.com"

    @pytest.mark.asyncio
    async def test_get_leads_with_time_range(self, linkedin_ads_connector):
        """Test get_leads with time range filter."""
        with patch.object(linkedin_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"elements": []}

            start_time = datetime(2024, 1, 1)
            end_time = datetime(2024, 1, 31)

            await linkedin_ads_connector.get_leads(
                "777888999",
                start_time=start_time,
                end_time=end_time,
            )

            call_args = mock_request.call_args
            assert "submittedAtRange" in str(call_args)


# =============================================================================
# Audience Tests
# =============================================================================


class TestAudienceSegmentDataclass:
    """Tests for AudienceSegment dataclass."""

    def test_audience_segment_from_api(self, mock_audience_response):
        """Test AudienceSegment.from_api parsing."""
        from aragora.connectors.advertising.linkedin_ads import AudienceSegment

        segment = AudienceSegment.from_api(mock_audience_response)

        assert segment.id == "aud_123456"
        assert segment.name == "Website Visitors"
        assert segment.segment_type == "REMARKETING"
        assert segment.account_id == "123456789"
        assert segment.matched_count == 15000
        assert segment.status == "READY"


class TestAudienceOperations:
    """Tests for audience operations."""

    @pytest.mark.asyncio
    async def test_get_audience_segments(self, linkedin_ads_connector, mock_audience_response):
        """Test get_audience_segments."""
        with patch.object(linkedin_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"elements": [mock_audience_response]}

            segments = await linkedin_ads_connector.get_audience_segments()

            assert len(segments) == 1
            assert segments[0].name == "Website Visitors"

    @pytest.mark.asyncio
    async def test_get_targeting_facets(self, linkedin_ads_connector):
        """Test get_targeting_facets."""
        with patch.object(linkedin_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"facets": ["titles", "industries", "companies"]}

            facets = await linkedin_ads_connector.get_targeting_facets()

            assert "facets" in facets


# =============================================================================
# Conversion Tracking Tests
# =============================================================================


class TestConversionOperations:
    """Tests for conversion tracking operations."""

    @pytest.mark.asyncio
    async def test_get_conversions(self, linkedin_ads_connector):
        """Test get_conversions."""
        with patch.object(linkedin_ads_connector, "_request") as mock_request:
            mock_request.return_value = {"elements": [{"id": "conv_123", "type": "PURCHASE"}]}

            conversions = await linkedin_ads_connector.get_conversions(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
            )

            assert len(conversions) == 1


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestLinkedInAdsError:
    """Tests for LinkedInAdsError exception."""

    def test_error_creation(self):
        """Test LinkedInAdsError creation."""
        from aragora.connectors.advertising.linkedin_ads import LinkedInAdsError

        error = LinkedInAdsError(
            message="Invalid token",
            status_code=401,
            error_code="UNAUTHORIZED",
        )

        assert str(error) == "Invalid token"
        assert error.status_code == 401
        assert error.error_code == "UNAUTHORIZED"

    def test_error_minimal(self):
        """Test LinkedInAdsError with minimal info."""
        from aragora.connectors.advertising.linkedin_ads import LinkedInAdsError

        error = LinkedInAdsError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.status_code is None
        assert error.error_code is None


class TestAPIErrorHandling:
    """Tests for API error handling."""

    @pytest.mark.asyncio
    async def test_request_handles_http_error(self, linkedin_ads_connector):
        """Test _request handles HTTP errors."""
        from aragora.connectors.advertising.linkedin_ads import LinkedInAdsError

        with patch.object(linkedin_ads_connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.content = b'{"message": "Unauthorized", "code": "UNAUTHORIZED"}'
            mock_response.json.return_value = {
                "message": "Unauthorized",
                "code": "UNAUTHORIZED",
            }
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(LinkedInAdsError) as exc_info:
                await linkedin_ads_connector._request("GET", "test")

            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_request_handles_empty_response(self, linkedin_ads_connector):
        """Test _request handles 204 No Content."""
        with patch.object(linkedin_ads_connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 204
            mock_response.content = b""
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await linkedin_ads_connector._request("DELETE", "test")

            assert result == {}


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_campaign_status_values(self):
        """Test CampaignStatus enum values."""
        from aragora.connectors.advertising.linkedin_ads import CampaignStatus

        assert CampaignStatus.ACTIVE.value == "ACTIVE"
        assert CampaignStatus.PAUSED.value == "PAUSED"
        assert CampaignStatus.ARCHIVED.value == "ARCHIVED"
        assert CampaignStatus.DRAFT.value == "DRAFT"

    def test_campaign_type_values(self):
        """Test CampaignType enum values."""
        from aragora.connectors.advertising.linkedin_ads import CampaignType

        assert CampaignType.TEXT_AD.value == "TEXT_AD"
        assert CampaignType.SPONSORED_UPDATES.value == "SPONSORED_UPDATES"
        assert CampaignType.SPONSORED_INMAILS.value == "SPONSORED_INMAILS"

    def test_objective_type_values(self):
        """Test ObjectiveType enum values."""
        from aragora.connectors.advertising.linkedin_ads import ObjectiveType

        assert ObjectiveType.BRAND_AWARENESS.value == "BRAND_AWARENESS"
        assert ObjectiveType.LEAD_GENERATION.value == "LEAD_GENERATION"
        assert ObjectiveType.WEBSITE_CONVERSIONS.value == "WEBSITE_CONVERSIONS"

    def test_ad_format_values(self):
        """Test AdFormat enum values."""
        from aragora.connectors.advertising.linkedin_ads import AdFormat

        assert AdFormat.SINGLE_IMAGE.value == "SINGLE_IMAGE_AD"
        assert AdFormat.CAROUSEL.value == "CAROUSEL_IMAGE_AD"
        assert AdFormat.VIDEO.value == "VIDEO_AD"
        assert AdFormat.MESSAGE.value == "MESSAGE_AD"

    def test_bid_strategy_values(self):
        """Test BidStrategy enum values."""
        from aragora.connectors.advertising.linkedin_ads import BidStrategy

        assert BidStrategy.MANUAL_CPC.value == "MANUAL_CPC"
        assert BidStrategy.MAXIMUM_DELIVERY.value == "MAXIMUM_DELIVERY"
        assert BidStrategy.COST_CAP.value == "COST_CAP"


# =============================================================================
# Mock Data Tests
# =============================================================================


class TestMockData:
    """Tests for mock data generation."""

    def test_get_mock_campaign(self):
        """Test get_mock_campaign."""
        from aragora.connectors.advertising.linkedin_ads import get_mock_campaign

        campaign = get_mock_campaign()

        assert campaign.id == "123456789"
        assert campaign.name == "Test B2B Campaign"
        assert campaign.daily_budget == 100.0
        assert campaign.targeting_criteria is not None

    def test_get_mock_analytics(self):
        """Test get_mock_analytics."""
        from aragora.connectors.advertising.linkedin_ads import get_mock_analytics

        analytics = get_mock_analytics()

        assert analytics.impressions == 50000
        assert analytics.clicks == 1250
        assert analytics.cost == 2500.0
        assert analytics.leads == 45
