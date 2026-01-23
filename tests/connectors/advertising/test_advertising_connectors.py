"""
Tests for Advertising Platform Connectors.

Tests for Google Ads, Meta Ads, LinkedIn Ads, Microsoft Ads, Twitter/X Ads, and TikTok Ads connectors.
"""

import pytest
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch


class TestGoogleAdsConnector:
    """Tests for Google Ads connector."""

    def test_google_ads_credentials(self):
        """Test GoogleAdsCredentials dataclass."""
        from aragora.connectors.advertising.google_ads import GoogleAdsCredentials

        creds = GoogleAdsCredentials(
            developer_token="dev_token",
            client_id="client_id",
            client_secret="client_secret",
            refresh_token="refresh_token",
            customer_id="123-456-7890",
        )

        assert creds.developer_token == "dev_token"
        assert creds.customer_id == "123-456-7890"

    def test_campaign_from_api(self):
        """Test Campaign.from_api parsing."""
        from aragora.connectors.advertising.google_ads import Campaign, CampaignStatus, CampaignType

        data = {
            "campaign": {
                "resourceName": "customers/123/campaigns/456",
                "id": "456",
                "name": "Test Campaign",
                "status": "ENABLED",
                "advertisingChannelType": "SEARCH",
            }
        }

        campaign = Campaign.from_api(data)

        assert campaign.id == "456"
        assert campaign.name == "Test Campaign"
        assert campaign.status == CampaignStatus.ENABLED
        assert campaign.advertising_channel_type == CampaignType.SEARCH

    def test_campaign_metrics_from_api(self):
        """Test CampaignMetrics.from_api parsing."""
        from aragora.connectors.advertising.google_ads import CampaignMetrics

        data = {
            "campaign": {"id": "456", "name": "Test Campaign"},
            "metrics": {
                "impressions": "10000",
                "clicks": "500",
                "costMicros": "25000000",
                "conversions": 50.0,
                "conversionsValue": 2500.0,
                "ctr": 0.05,
                "averageCpc": 50000,
            },
        }

        metrics = CampaignMetrics.from_api(data)

        assert metrics.impressions == 10000
        assert metrics.clicks == 500
        assert metrics.cost_micros == 25000000
        assert metrics.conversions == 50
        assert metrics.ctr == pytest.approx(0.05, rel=0.01)
        assert metrics.average_cpc_micros == 50000

    def test_mock_campaign(self):
        """Test mock campaign generation."""
        from aragora.connectors.advertising.google_ads import get_mock_campaign

        campaign = get_mock_campaign()

        assert campaign.id == "12345678901"
        assert campaign.name == "Brand Campaign"
        assert campaign.budget_amount_micros == 50_000_000

    def test_mock_metrics(self):
        """Test mock metrics generation."""
        from aragora.connectors.advertising.google_ads import get_mock_metrics

        metrics = get_mock_metrics()

        assert metrics.impressions == 10000
        assert metrics.clicks == 500
        assert metrics.conversions == 50


class TestMetaAdsConnector:
    """Tests for Meta/Facebook Ads connector."""

    def test_meta_ads_credentials(self):
        """Test MetaAdsCredentials dataclass."""
        from aragora.connectors.advertising.meta_ads import MetaAdsCredentials

        creds = MetaAdsCredentials(
            access_token="access_token",
            ad_account_id="act_123456789",
            app_id="app_123",
            app_secret="secret",
        )

        assert creds.access_token == "access_token"
        assert creds.ad_account_id == "act_123456789"

    def test_campaign_from_api(self):
        """Test Campaign.from_api parsing."""
        from aragora.connectors.advertising.meta_ads import (
            Campaign,
            CampaignStatus,
            CampaignObjective,
        )

        data = {
            "id": "123456789",
            "name": "Test Meta Campaign",
            "status": "ACTIVE",
            "objective": "OUTCOME_TRAFFIC",
            "daily_budget": "5000",
            "lifetime_budget": "150000",
        }

        campaign = Campaign.from_api(data)

        assert campaign.id == "123456789"
        assert campaign.name == "Test Meta Campaign"
        assert campaign.status == CampaignStatus.ACTIVE
        assert campaign.objective == CampaignObjective.OUTCOME_TRAFFIC
        assert campaign.daily_budget == 5000  # In cents
        assert campaign.lifetime_budget == 150000  # In cents

    def test_ad_insights_from_api(self):
        """Test AdInsights.from_api parsing."""
        from aragora.connectors.advertising.meta_ads import AdInsights
        from decimal import Decimal

        data = {
            "impressions": "50000",
            "clicks": "1500",
            "spend": "250.50",
            "actions": [
                {"action_type": "link_click", "value": "1200"},
                {"action_type": "purchase", "value": "75"},
            ],
            "reach": "35000",
            "frequency": "1.43",
        }

        insights = AdInsights.from_api(data)

        assert insights.impressions == 50000
        assert insights.clicks == 1500
        assert insights.spend == Decimal("250.50")
        assert insights.reach == 35000
        assert len(insights.actions) == 2

    def test_custom_audience_from_api(self):
        """Test CustomAudience.from_api parsing."""
        from aragora.connectors.advertising.meta_ads import CustomAudience

        data = {
            "id": "aud_123",
            "name": "Website Visitors",
            "subtype": "WEBSITE",
            "approximate_count": 50000,
            "data_source": {"type": "PIXEL"},
        }

        audience = CustomAudience.from_api(data)

        assert audience.id == "aud_123"
        assert audience.name == "Website Visitors"
        assert audience.subtype == "WEBSITE"
        assert audience.approximate_count == 50000

    def test_mock_campaign(self):
        """Test mock campaign generation."""
        from aragora.connectors.advertising.meta_ads import get_mock_campaign

        campaign = get_mock_campaign()

        assert campaign.id == "23456789012345678"
        assert campaign.name == "Summer Sale Campaign"

    def test_mock_insights(self):
        """Test mock insights generation."""
        from aragora.connectors.advertising.meta_ads import get_mock_insights

        insights = get_mock_insights()

        assert insights.impressions == 50000
        assert insights.clicks == 2500


class TestLinkedInAdsConnector:
    """Tests for LinkedIn Ads connector."""

    def test_linkedin_ads_credentials(self):
        """Test LinkedInAdsCredentials dataclass."""
        from aragora.connectors.advertising.linkedin_ads import LinkedInAdsCredentials

        creds = LinkedInAdsCredentials(
            access_token="access_token",
            ad_account_id="123456789",
            client_id="client_id",
            client_secret="client_secret",
        )

        assert creds.access_token == "access_token"
        assert creds.ad_account_id == "123456789"

    def test_campaign_from_api(self):
        """Test Campaign.from_api parsing."""
        from aragora.connectors.advertising.linkedin_ads import (
            Campaign,
            CampaignStatus,
            CampaignType,
            ObjectiveType,
        )

        data = {
            "id": "123456789",
            "name": "Test B2B Campaign",
            "status": "ACTIVE",
            "campaignGroup": "urn:li:sponsoredCampaignGroup:987654321",
            "account": "urn:li:sponsoredAccount:111222333",
            "type": "SPONSORED_UPDATES",
            "objectiveType": "LEAD_GENERATION",
            "dailyBudget": {"amount": "100.00"},
        }

        campaign = Campaign.from_api(data)

        assert campaign.id == "123456789"
        assert campaign.name == "Test B2B Campaign"
        assert campaign.status == CampaignStatus.ACTIVE
        assert campaign.campaign_type == CampaignType.SPONSORED_UPDATES
        assert campaign.objective_type == ObjectiveType.LEAD_GENERATION
        assert campaign.daily_budget == "100.00"  # Returns as string from API

    def test_ad_analytics_from_api(self):
        """Test AdAnalytics.from_api parsing."""
        from aragora.connectors.advertising.linkedin_ads import AdAnalytics

        data = {
            "impressions": 50000,
            "clicks": 1250,
            "costInLocalCurrency": 2500.0,
            "externalWebsiteConversions": 75,
            "oneClickLeads": 45,
            "leadGenerationMailContactInfoShares": 0,
            "videoViews": 0,
            "videoCompletions": 0,
            "totalEngagements": 320,
            "follows": 15,
        }

        analytics = AdAnalytics.from_api(data, "123", date(2024, 1, 1), date(2024, 1, 31))

        assert analytics.impressions == 50000
        assert analytics.clicks == 1250
        assert analytics.cost == 2500.0
        assert analytics.conversions == 75
        assert analytics.leads == 45
        assert analytics.ctr == pytest.approx(2.5, rel=0.01)

    def test_mock_campaign(self):
        """Test mock campaign generation."""
        from aragora.connectors.advertising.linkedin_ads import get_mock_campaign

        campaign = get_mock_campaign()

        assert campaign.id == "123456789"
        assert campaign.name == "Test B2B Campaign"

    def test_mock_analytics(self):
        """Test mock analytics generation."""
        from aragora.connectors.advertising.linkedin_ads import get_mock_analytics

        analytics = get_mock_analytics()

        assert analytics.impressions == 50000
        assert analytics.clicks == 1250
        assert analytics.leads == 45


class TestMicrosoftAdsConnector:
    """Tests for Microsoft Ads (Bing) connector."""

    def test_microsoft_ads_credentials(self):
        """Test MicrosoftAdsCredentials dataclass."""
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
        assert creds.account_id == "123456789"
        assert creds.customer_id == "987654321"

    def test_campaign_from_api(self):
        """Test Campaign.from_api parsing."""
        from aragora.connectors.advertising.microsoft_ads import (
            Campaign,
            CampaignStatus,
            CampaignType,
            BudgetType,
        )

        data = {
            "Id": "123456789",
            "Name": "Test Bing Campaign",
            "Status": "Active",
            "CampaignType": "Search",
            "BudgetType": "DailyBudgetStandard",
            "DailyBudget": 50.0,
            "BiddingScheme": {"Type": "EnhancedCpc"},
            "TimeZone": "PacificTimeUSCanadaTijuana",
        }

        campaign = Campaign.from_api(data)

        assert campaign.id == "123456789"
        assert campaign.name == "Test Bing Campaign"
        assert campaign.status == CampaignStatus.ACTIVE
        assert campaign.campaign_type == CampaignType.SEARCH
        assert campaign.budget_type == BudgetType.DAILY_BUDGET_STANDARD
        assert campaign.daily_budget == 50.0

    def test_campaign_performance_from_api(self):
        """Test CampaignPerformance.from_api parsing."""
        from aragora.connectors.advertising.microsoft_ads import CampaignPerformance

        data = {
            "CampaignId": "123456789",
            "CampaignName": "Test Campaign",
            "TimePeriod": "2024-01-15",
            "Impressions": "10000",
            "Clicks": "250",
            "Spend": "125.00",
            "Conversions": "15",
            "Revenue": "450.00",
            "AveragePosition": "2.3",
            "QualityScore": "7",
        }

        perf = CampaignPerformance.from_api(data)

        assert perf.campaign_id == "123456789"
        assert perf.impressions == 10000
        assert perf.clicks == 250
        assert perf.spend == 125.0
        assert perf.conversions == 15
        assert perf.ctr == pytest.approx(2.5, rel=0.01)
        assert perf.average_cpc == pytest.approx(0.5, rel=0.01)

    def test_mock_campaign(self):
        """Test mock campaign generation."""
        from aragora.connectors.advertising.microsoft_ads import get_mock_campaign

        campaign = get_mock_campaign()

        assert campaign.id == "123456789"
        assert campaign.name == "Test Bing Campaign"
        assert campaign.daily_budget == 50.0

    def test_mock_performance(self):
        """Test mock performance generation."""
        from aragora.connectors.advertising.microsoft_ads import get_mock_performance

        perf = get_mock_performance()

        assert perf.impressions == 10000
        assert perf.clicks == 250
        assert perf.quality_score == 7


class TestAdvertisingPackageImports:
    """Test that advertising package imports work correctly."""

    def test_google_ads_imports(self):
        """Test Google Ads can be imported from package."""
        from aragora.connectors.advertising import (
            GoogleAdsConnector,
            GoogleAdsCredentials,
            GoogleCampaign,
            CampaignMetrics,
            GoogleAdsError,
        )

        assert GoogleAdsConnector is not None
        assert GoogleAdsCredentials is not None

    def test_meta_ads_imports(self):
        """Test Meta Ads can be imported from package."""
        from aragora.connectors.advertising import (
            MetaAdsConnector,
            MetaAdsCredentials,
            MetaCampaign,
            AdInsights,
            CustomAudience,
            MetaAdsError,
        )

        assert MetaAdsConnector is not None
        assert MetaAdsCredentials is not None

    def test_linkedin_ads_imports(self):
        """Test LinkedIn Ads can be imported from package."""
        from aragora.connectors.advertising import (
            LinkedInAdsConnector,
            LinkedInAdsCredentials,
            LinkedInCampaign,
            AdAnalytics,
            LeadGenForm,
            LinkedInAdsError,
        )

        assert LinkedInAdsConnector is not None
        assert LinkedInAdsCredentials is not None

    def test_microsoft_ads_imports(self):
        """Test Microsoft Ads can be imported from package."""
        from aragora.connectors.advertising import (
            MicrosoftAdsConnector,
            MicrosoftAdsCredentials,
            MicrosoftCampaign,
            CampaignPerformance,
            MicrosoftAdsError,
        )

        assert MicrosoftAdsConnector is not None
        assert MicrosoftAdsCredentials is not None


class TestEnumValues:
    """Test enum values are correctly defined."""

    def test_google_ads_enums(self):
        """Test Google Ads enum values."""
        from aragora.connectors.advertising.google_ads import (
            CampaignStatus,
            CampaignType,
            BiddingStrategyType,
        )

        assert CampaignStatus.ENABLED.value == "ENABLED"
        assert CampaignStatus.PAUSED.value == "PAUSED"
        assert CampaignType.SEARCH.value == "SEARCH"
        assert BiddingStrategyType.MAXIMIZE_CONVERSIONS.value == "MAXIMIZE_CONVERSIONS"

    def test_meta_ads_enums(self):
        """Test Meta Ads enum values."""
        from aragora.connectors.advertising.meta_ads import (
            CampaignStatus,
            CampaignObjective,
            AdSetStatus,
        )

        assert CampaignStatus.ACTIVE.value == "ACTIVE"
        assert CampaignObjective.OUTCOME_TRAFFIC.value == "OUTCOME_TRAFFIC"
        assert AdSetStatus.ACTIVE.value == "ACTIVE"

    def test_linkedin_ads_enums(self):
        """Test LinkedIn Ads enum values."""
        from aragora.connectors.advertising.linkedin_ads import (
            CampaignStatus,
            ObjectiveType,
            AdFormat,
        )

        assert CampaignStatus.ACTIVE.value == "ACTIVE"
        assert ObjectiveType.LEAD_GENERATION.value == "LEAD_GENERATION"
        assert AdFormat.SINGLE_IMAGE.value == "SINGLE_IMAGE_AD"

    def test_microsoft_ads_enums(self):
        """Test Microsoft Ads enum values."""
        from aragora.connectors.advertising.microsoft_ads import (
            CampaignStatus,
            BiddingScheme,
            KeywordMatchType,
        )

        assert CampaignStatus.ACTIVE.value == "Active"
        assert BiddingScheme.ENHANCED_CPC.value == "EnhancedCpc"
        assert KeywordMatchType.EXACT.value == "Exact"

    def test_twitter_ads_enums(self):
        """Test Twitter Ads enum values."""
        from aragora.connectors.advertising.twitter_ads import (
            CampaignStatus,
            CampaignObjective,
            LineItemType,
            PlacementType,
        )

        assert CampaignStatus.ACTIVE.value == "ACTIVE"
        assert CampaignStatus.PAUSED.value == "PAUSED"
        assert CampaignObjective.WEBSITE_CLICKS.value == "WEBSITE_CLICKS"
        assert LineItemType.PROMOTED_TWEETS.value == "PROMOTED_TWEETS"
        assert PlacementType.ALL_ON_TWITTER.value == "ALL_ON_TWITTER"

    def test_tiktok_ads_enums(self):
        """Test TikTok Ads enum values."""
        from aragora.connectors.advertising.tiktok_ads import (
            CampaignStatus,
            CampaignObjective,
            OptimizationGoal,
            BillingEvent,
        )

        assert CampaignStatus.ENABLE.value == "ENABLE"
        assert CampaignStatus.DISABLE.value == "DISABLE"
        assert CampaignObjective.TRAFFIC.value == "TRAFFIC"
        assert OptimizationGoal.CLICK.value == "CLICK"
        assert BillingEvent.OCPM.value == "OCPM"


class TestTwitterAdsConnector:
    """Tests for Twitter/X Ads connector."""

    def test_twitter_ads_credentials(self):
        """Test TwitterAdsCredentials dataclass."""
        from aragora.connectors.advertising.twitter_ads import TwitterAdsCredentials

        creds = TwitterAdsCredentials(
            consumer_key="consumer_key",
            consumer_secret="consumer_secret",
            access_token="access_token",
            access_token_secret="access_token_secret",
            ads_account_id="123456789",
        )

        assert creds.consumer_key == "consumer_key"
        assert creds.ads_account_id == "123456789"

    def test_campaign_from_api(self):
        """Test Campaign.from_api parsing."""
        from aragora.connectors.advertising.twitter_ads import (
            Campaign,
            CampaignStatus,
            CampaignObjective,
        )

        data = {
            "id": "abc123",
            "name": "Test Twitter Campaign",
            "account_id": "123456789",
            "entity_status": "ACTIVE",
            "objective": "WEBSITE_CLICKS",
            "funding_instrument_id": "fi_123",
            "daily_budget_amount_local_micro": 100000000,
        }

        campaign = Campaign.from_api(data)

        assert campaign.id == "abc123"
        assert campaign.name == "Test Twitter Campaign"
        assert campaign.status == CampaignStatus.ACTIVE
        assert campaign.objective == CampaignObjective.WEBSITE_CLICKS
        assert campaign.daily_budget == 100.0

    def test_campaign_analytics_from_api(self):
        """Test CampaignAnalytics.from_api parsing."""
        from aragora.connectors.advertising.twitter_ads import CampaignAnalytics

        data = {
            "metrics": {
                "impressions": [100000],
                "engagements": [5000],
                "clicks": [2500],
                "billed_charge_local_micro": [500000000],
                "retweets": [200],
                "likes": [2000],
            }
        }

        analytics = CampaignAnalytics.from_api(data, "abc123", date(2024, 1, 1), date(2024, 1, 31))

        assert analytics.impressions == 100000
        assert analytics.engagements == 5000
        assert analytics.clicks == 2500
        assert analytics.spend == 500.0
        assert analytics.engagement_rate == pytest.approx(5.0, rel=0.01)

    def test_mock_campaign(self):
        """Test mock campaign generation."""
        from aragora.connectors.advertising.twitter_ads import get_mock_campaign

        campaign = get_mock_campaign()

        assert campaign.id == "abc123"
        assert campaign.name == "Test Twitter Campaign"
        assert campaign.daily_budget == 100.0

    def test_mock_analytics(self):
        """Test mock analytics generation."""
        from aragora.connectors.advertising.twitter_ads import get_mock_analytics

        analytics = get_mock_analytics()

        assert analytics.impressions == 100000
        assert analytics.engagements == 5000
        assert analytics.clicks == 2500


class TestTikTokAdsConnector:
    """Tests for TikTok Ads connector."""

    def test_tiktok_ads_credentials(self):
        """Test TikTokAdsCredentials dataclass."""
        from aragora.connectors.advertising.tiktok_ads import TikTokAdsCredentials

        creds = TikTokAdsCredentials(
            access_token="access_token",
            advertiser_id="123456789",
            app_id="app_123",
            secret="secret_key",
        )

        assert creds.access_token == "access_token"
        assert creds.advertiser_id == "123456789"

    def test_campaign_from_api(self):
        """Test Campaign.from_api parsing."""
        from aragora.connectors.advertising.tiktok_ads import (
            Campaign,
            CampaignStatus,
            CampaignObjective,
        )

        data = {
            "campaign_id": "123456789",
            "campaign_name": "Test TikTok Campaign",
            "advertiser_id": "987654321",
            "status": "ENABLE",
            "objective_type": "TRAFFIC",
            "budget": 500.0,
            "budget_mode": "BUDGET_MODE_DAY",
        }

        campaign = Campaign.from_api(data)

        assert campaign.id == "123456789"
        assert campaign.name == "Test TikTok Campaign"
        assert campaign.status == CampaignStatus.ENABLE
        assert campaign.objective == CampaignObjective.TRAFFIC
        assert campaign.budget == 500.0

    def test_ad_group_metrics_from_api(self):
        """Test AdGroupMetrics.from_api parsing."""
        from aragora.connectors.advertising.tiktok_ads import AdGroupMetrics

        data = {
            "metrics": {
                "impressions": 500000,
                "clicks": 15000,
                "spend": 3000.0,
                "conversion": 450,
                "video_views": 200000,
                "reach": 300000,
            }
        }
        dimensions = {
            "adgroup_id": "456789",
            "adgroup_name": "Test Ad Group",
            "campaign_id": "123456789",
        }

        metrics = AdGroupMetrics.from_api(data, dimensions, date(2024, 1, 1), date(2024, 1, 31))

        assert metrics.impressions == 500000
        assert metrics.clicks == 15000
        assert metrics.spend == 3000.0
        assert metrics.conversions == 450
        assert metrics.ctr == pytest.approx(3.0, rel=0.01)

    def test_mock_campaign(self):
        """Test mock campaign generation."""
        from aragora.connectors.advertising.tiktok_ads import get_mock_campaign

        campaign = get_mock_campaign()

        assert campaign.id == "123456789"
        assert campaign.name == "Test TikTok Campaign"
        assert campaign.budget == 500.0

    def test_mock_metrics(self):
        """Test mock metrics generation."""
        from aragora.connectors.advertising.tiktok_ads import get_mock_metrics

        metrics = get_mock_metrics()

        assert metrics.impressions == 500000
        assert metrics.clicks == 15000
        assert metrics.video_views == 200000


class TestTwitterTikTokPackageImports:
    """Test that Twitter and TikTok imports work correctly."""

    def test_twitter_ads_imports(self):
        """Test Twitter Ads can be imported from package."""
        from aragora.connectors.advertising import (
            TwitterAdsConnector,
            TwitterAdsCredentials,
            TwitterCampaign,
            TwitterCampaignAnalytics,
            TailoredAudience,
            TwitterAdsError,
        )

        assert TwitterAdsConnector is not None
        assert TwitterAdsCredentials is not None

    def test_tiktok_ads_imports(self):
        """Test TikTok Ads can be imported from package."""
        from aragora.connectors.advertising import (
            TikTokAdsConnector,
            TikTokAdsCredentials,
            TikTokCampaign,
            TikTokAdGroupMetrics,
            TikTokAdsError,
        )

        assert TikTokAdsConnector is not None
        assert TikTokAdsCredentials is not None
