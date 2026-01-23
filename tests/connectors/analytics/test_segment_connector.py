"""
Tests for Segment CDP Connector.

Tests for Segment tracking, config, and profiles APIs.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


class TestSegmentCredentials:
    """Tests for Segment credentials."""

    def test_segment_credentials(self):
        """Test SegmentCredentials dataclass."""
        from aragora.connectors.analytics.segment import SegmentCredentials

        creds = SegmentCredentials(
            write_key="wk_abc123",
            workspace_slug="my-workspace",
            access_token="access_token_123",
        )

        assert creds.write_key == "wk_abc123"
        assert creds.workspace_slug == "my-workspace"
        assert creds.access_token == "access_token_123"


class TestSegmentDataclasses:
    """Tests for Segment dataclass parsing."""

    def test_source_from_api(self):
        """Test Source.from_api parsing."""
        from aragora.connectors.analytics.segment import Source

        data = {
            "id": "src_123",
            "name": "Website",
            "slug": "website",
            "sourceDefinitionId": "javascript",
            "workspaceId": "ws_456",
            "enabled": True,
            "createdAt": "2024-01-01T00:00:00Z",
            "writeKey": "wk_abc123",
        }

        source = Source.from_api(data)

        assert source.id == "src_123"
        assert source.name == "Website"
        assert source.slug == "website"
        assert source.source_type == "javascript"
        assert source.enabled is True
        assert source.write_key == "wk_abc123"

    def test_destination_from_api(self):
        """Test Destination.from_api parsing."""
        from aragora.connectors.analytics.segment import Destination

        data = {
            "id": "dst_123",
            "name": "Google Analytics",
            "sourceId": "src_123",
            "destinationDefinitionId": "google-analytics",
            "enabled": True,
            "settings": {"trackingId": "UA-123456"},
        }

        dest = Destination.from_api(data)

        assert dest.id == "dst_123"
        assert dest.name == "Google Analytics"
        assert dest.source_id == "src_123"
        assert dest.destination_type == "google-analytics"
        assert dest.enabled is True
        assert dest.settings["trackingId"] == "UA-123456"

    def test_profile_from_api(self):
        """Test Profile.from_api parsing."""
        from aragora.connectors.analytics.segment import Profile

        data = {
            "segment_id": "seg_123",
            "user_id": "user_456",
            "anonymous_id": "anon_789",
            "traits": {
                "email": "test@example.com",
                "name": "Test User",
            },
        }

        profile = Profile.from_api(data)

        assert profile.segment_id == "seg_123"
        assert profile.user_id == "user_456"
        assert profile.anonymous_id == "anon_789"
        assert profile.traits["email"] == "test@example.com"


class TestSegmentEvents:
    """Tests for Segment event models."""

    def test_track_event_to_api(self):
        """Test TrackEvent.to_api conversion."""
        from aragora.connectors.analytics.segment import TrackEvent

        event = TrackEvent(
            user_id="user_123",
            anonymous_id=None,
            event="Button Clicked",
            properties={"button_name": "signup"},
            context={"page": "/home"},
        )

        api_data = event.to_api()

        assert api_data["userId"] == "user_123"
        assert api_data["event"] == "Button Clicked"
        assert api_data["properties"]["button_name"] == "signup"
        assert api_data["context"]["page"] == "/home"
        assert "anonymousId" not in api_data

    def test_identify_call_to_api(self):
        """Test IdentifyCall.to_api conversion."""
        from aragora.connectors.analytics.segment import IdentifyCall

        identify = IdentifyCall(
            user_id="user_123",
            anonymous_id="anon_456",
            traits={"email": "test@example.com", "plan": "premium"},
        )

        api_data = identify.to_api()

        assert api_data["userId"] == "user_123"
        assert api_data["anonymousId"] == "anon_456"
        assert api_data["traits"]["email"] == "test@example.com"
        assert api_data["traits"]["plan"] == "premium"

    def test_page_event_to_api(self):
        """Test PageEvent.to_api conversion."""
        from aragora.connectors.analytics.segment import PageEvent

        page = PageEvent(
            user_id="user_123",
            anonymous_id=None,
            name="Home",
            category="Marketing",
            properties={"url": "/home"},
        )

        api_data = page.to_api()

        assert api_data["userId"] == "user_123"
        assert api_data["name"] == "Home"
        assert api_data["category"] == "Marketing"
        assert api_data["properties"]["url"] == "/home"

    def test_group_call_to_api(self):
        """Test GroupCall.to_api conversion."""
        from aragora.connectors.analytics.segment import GroupCall

        group = GroupCall(
            user_id="user_123",
            group_id="company_456",
            traits={"name": "Acme Inc", "industry": "Tech"},
        )

        api_data = group.to_api()

        assert api_data["userId"] == "user_123"
        assert api_data["groupId"] == "company_456"
        assert api_data["traits"]["name"] == "Acme Inc"


class TestSegmentEnums:
    """Tests for Segment enum values."""

    def test_source_type_enum(self):
        """Test SourceType enum values."""
        from aragora.connectors.analytics.segment import SourceType

        assert SourceType.JAVASCRIPT.value == "javascript"
        assert SourceType.PYTHON.value == "python"
        assert SourceType.HTTP.value == "http"

    def test_destination_type_enum(self):
        """Test DestinationType enum values."""
        from aragora.connectors.analytics.segment import DestinationType

        assert DestinationType.GOOGLE_ANALYTICS.value == "Google Analytics"
        assert DestinationType.MIXPANEL.value == "Mixpanel"
        assert DestinationType.BIGQUERY.value == "BigQuery"

    def test_event_type_enum(self):
        """Test EventType enum values."""
        from aragora.connectors.analytics.segment import EventType

        assert EventType.TRACK.value == "track"
        assert EventType.IDENTIFY.value == "identify"
        assert EventType.PAGE.value == "page"


class TestSegmentMocks:
    """Tests for Segment mock data."""

    def test_mock_source(self):
        """Test mock source generation."""
        from aragora.connectors.analytics.segment import get_mock_source

        source = get_mock_source()

        assert source.id == "src_123"
        assert source.name == "Website"
        assert source.write_key == "wk_abc123"

    def test_mock_profile(self):
        """Test mock profile generation."""
        from aragora.connectors.analytics.segment import get_mock_profile

        profile = get_mock_profile()

        assert profile.segment_id == "seg_123"
        assert profile.user_id == "user_456"
        assert profile.traits["email"] == "test@example.com"


class TestSegmentPackageImports:
    """Test that Segment imports work correctly from analytics package."""

    def test_segment_imports(self):
        """Test Segment can be imported from analytics package."""
        from aragora.connectors.analytics import (
            SegmentConnector,
            SegmentCredentials,
            Source,
            Destination,
            TrackEvent,
            IdentifyCall,
            SegmentError,
        )

        assert SegmentConnector is not None
        assert SegmentCredentials is not None
        assert Source is not None
