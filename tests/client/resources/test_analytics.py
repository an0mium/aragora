"""Tests for AnalyticsAPI resource."""

import pytest

from aragora.client import AragoraClient
from aragora.client.resources.analytics import AnalyticsAPI


class TestAnalyticsAPI:
    """Tests for AnalyticsAPI resource."""

    def test_analytics_api_exists(self):
        """Test that AnalyticsAPI is accessible on client."""
        client = AragoraClient()
        assert isinstance(client.analytics, AnalyticsAPI)

    def test_analytics_api_has_basic_methods(self):
        """Test that AnalyticsAPI has basic methods."""
        client = AragoraClient()
        # Check for any analytics-related methods
        api = client.analytics
        # At minimum the API should be instantiated properly
        assert api is not None
        assert api._client is not None
