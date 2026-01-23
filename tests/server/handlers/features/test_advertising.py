"""
Tests for Advertising Platform Handler.

Tests cover basic dataclass creation and platform configuration.
"""

import pytest
from datetime import date, datetime, timedelta

from aragora.server.handlers.features.advertising import (
    AdvertisingHandler,
    SUPPORTED_PLATFORMS,
    UnifiedCampaign,
    UnifiedPerformance,
)


class TestSupportedPlatforms:
    """Tests for platform configuration."""

    def test_all_platforms_defined(self):
        """Test that all supported platforms are configured."""
        expected = ["google_ads", "meta_ads", "linkedin_ads", "microsoft_ads"]
        for platform in expected:
            assert platform in SUPPORTED_PLATFORMS

    def test_platform_has_required_fields(self):
        """Test that all platforms have required configuration."""
        for platform_id, config in SUPPORTED_PLATFORMS.items():
            assert "name" in config, f"{platform_id} missing name"
            assert "description" in config, f"{platform_id} missing description"
            assert "features" in config, f"{platform_id} missing features"
            assert isinstance(config["features"], list)


class TestUnifiedCampaign:
    """Tests for UnifiedCampaign dataclass."""

    def test_campaign_creation(self):
        """Test creating a unified campaign."""
        campaign = UnifiedCampaign(
            id="camp_123",
            platform="google_ads",
            name="Test Campaign",
            status="active",
            objective="conversions",
            daily_budget=100.0,
            total_budget=3000.0,
            start_date=date.today(),
            end_date=date.today() + timedelta(days=30),
            created_at=datetime.now(),
        )

        assert campaign.id == "camp_123"
        assert campaign.platform == "google_ads"
        assert campaign.status == "active"
        assert campaign.daily_budget == 100.0

    def test_campaign_defaults(self):
        """Test campaign with minimal fields."""
        campaign = UnifiedCampaign(
            id="camp_456",
            platform="meta_ads",
            name="Meta Campaign",
            status="paused",
            objective=None,
            daily_budget=None,
            total_budget=None,
            start_date=None,
            end_date=None,
            created_at=None,
        )

        assert campaign.id == "camp_456"
        assert campaign.objective is None


class TestUnifiedPerformance:
    """Tests for UnifiedPerformance dataclass."""

    def test_performance_creation(self):
        """Test creating performance metrics."""
        perf = UnifiedPerformance(
            platform="google_ads",
            campaign_id="camp_123",
            campaign_name="Test Campaign",
            date_range=(date(2024, 1, 1), date(2024, 1, 31)),
            impressions=100000,
            clicks=2500,
            cost=5000.0,
            conversions=125,
            conversion_value=15000.0,
            ctr=0.025,
            cpc=2.0,
            cpm=50.0,
            roas=3.0,
        )

        assert perf.platform == "google_ads"
        assert perf.impressions == 100000
        assert perf.roas == 3.0


class TestAdvertisingHandler:
    """Tests for AdvertisingHandler class."""

    def test_handler_creation(self):
        """Test creating handler instance."""
        handler = AdvertisingHandler(server_context={})
        assert handler is not None

    def test_handler_has_routes(self):
        """Test that handler has route definitions."""
        handler = AdvertisingHandler(server_context={})
        # Handler should have methods for handling requests
        assert hasattr(handler, "handle_get") or hasattr(handler, "handle_request")
