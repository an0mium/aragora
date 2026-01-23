"""
Tests for Marketing Platform Connectors.

Tests for Mailchimp and Klaviyo connectors.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


class TestMailchimpConnector:
    """Tests for Mailchimp connector."""

    def test_mailchimp_credentials(self):
        """Test MailchimpCredentials dataclass."""
        from aragora.connectors.marketing.mailchimp import MailchimpCredentials

        creds = MailchimpCredentials(
            api_key="abc123-us1",
            server_prefix="us1",
        )

        assert creds.api_key == "abc123-us1"
        assert creds.server_prefix == "us1"

    def test_audience_from_api(self):
        """Test Audience.from_api parsing."""
        from aragora.connectors.marketing.mailchimp import Audience

        data = {
            "id": "abc123def",
            "name": "Newsletter Subscribers",
            "contact": {"company": "Test Company"},
            "permission_reminder": "You signed up for our newsletter",
            "date_created": "2024-01-01T00:00:00Z",
            "stats": {
                "member_count": 5000,
                "unsubscribe_count": 150,
                "open_rate": 25.5,
            },
            "double_optin": True,
        }

        audience = Audience.from_api(data)

        assert audience.id == "abc123def"
        assert audience.name == "Newsletter Subscribers"
        assert audience.stats.member_count == 5000
        assert audience.double_optin is True

    def test_member_from_api(self):
        """Test Member.from_api parsing."""
        from aragora.connectors.marketing.mailchimp import Member, MemberStatus

        data = {
            "id": "member123",
            "email_address": "test@example.com",
            "status": "subscribed",
            "email_type": "html",
            "merge_fields": {"FNAME": "Test", "LNAME": "User"},
            "vip": True,
            "member_rating": 4,
            "tags_count": 3,
        }

        member = Member.from_api(data)

        assert member.id == "member123"
        assert member.email_address == "test@example.com"
        assert member.status == MemberStatus.SUBSCRIBED
        assert member.merge_fields["FNAME"] == "Test"
        assert member.vip is True

    def test_campaign_from_api(self):
        """Test Campaign.from_api parsing."""
        from aragora.connectors.marketing.mailchimp import Campaign, CampaignStatus, CampaignType

        data = {
            "id": "camp123",
            "web_id": 12345,
            "type": "regular",
            "status": "sent",
            "emails_sent": 4500,
            "create_time": "2024-01-01T00:00:00Z",
            "send_time": "2024-01-02T00:00:00Z",
            "settings": {
                "subject_line": "Test Newsletter",
                "from_name": "Test Company",
            },
        }

        campaign = Campaign.from_api(data)

        assert campaign.id == "camp123"
        assert campaign.type == CampaignType.REGULAR
        assert campaign.status == CampaignStatus.SENT
        assert campaign.emails_sent == 4500

    def test_campaign_report_from_api(self):
        """Test CampaignReport.from_api parsing."""
        from aragora.connectors.marketing.mailchimp import CampaignReport

        data = {
            "id": "camp123",
            "campaign_title": "Test Newsletter",
            "list_id": "abc123def",
            "list_name": "Newsletter Subscribers",
            "subject_line": "Test Newsletter",
            "emails_sent": 4500,
            "opens": {"opens_total": 1800, "unique_opens": 1500},
            "clicks": {"clicks_total": 450, "unique_clicks": 350},
            "bounces": {"hard_bounces": 25, "soft_bounces": 50},
        }

        report = CampaignReport.from_api(data)

        assert report.id == "camp123"
        assert report.emails_sent == 4500
        assert report.opens["opens_total"] == 1800
        assert report.open_rate == pytest.approx(40.0, rel=0.01)
        assert report.click_rate == pytest.approx(10.0, rel=0.01)

    def test_mock_audience(self):
        """Test mock audience generation."""
        from aragora.connectors.marketing.mailchimp import get_mock_audience

        audience = get_mock_audience()

        assert audience.id == "abc123def"
        assert audience.name == "Newsletter Subscribers"
        assert audience.stats.member_count == 5000

    def test_mock_campaign(self):
        """Test mock campaign generation."""
        from aragora.connectors.marketing.mailchimp import get_mock_campaign

        campaign = get_mock_campaign()

        assert campaign.id == "camp123"
        assert campaign.emails_sent == 4500

    def test_mock_report(self):
        """Test mock report generation."""
        from aragora.connectors.marketing.mailchimp import get_mock_report

        report = get_mock_report()

        assert report.id == "camp123"
        assert report.emails_sent == 4500


class TestKlaviyoConnector:
    """Tests for Klaviyo connector."""

    def test_klaviyo_credentials(self):
        """Test KlaviyoCredentials dataclass."""
        from aragora.connectors.marketing.klaviyo import KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_abc123")

        assert creds.api_key == "pk_abc123"

    def test_klaviyo_list_from_api(self):
        """Test KlaviyoList.from_api parsing."""
        from aragora.connectors.marketing.klaviyo import KlaviyoList

        data = {
            "id": "abc123",
            "attributes": {
                "name": "Newsletter Subscribers",
                "created": "2024-01-01T00:00:00Z",
                "profile_count": 15000,
                "opt_in_process": "double_opt_in",
            },
        }

        klist = KlaviyoList.from_api(data)

        assert klist.id == "abc123"
        assert klist.name == "Newsletter Subscribers"
        assert klist.profile_count == 15000
        assert klist.opt_in_process == "double_opt_in"

    def test_profile_from_api(self):
        """Test Profile.from_api parsing."""
        from aragora.connectors.marketing.klaviyo import Profile

        data = {
            "id": "prof_123",
            "attributes": {
                "email": "test@example.com",
                "first_name": "Test",
                "last_name": "User",
                "phone_number": "+15551234567",
                "properties": {"plan": "premium"},
                "created": "2024-01-01T00:00:00Z",
            },
        }

        profile = Profile.from_api(data)

        assert profile.id == "prof_123"
        assert profile.email == "test@example.com"
        assert profile.first_name == "Test"
        assert profile.phone_number == "+15551234567"
        assert profile.properties["plan"] == "premium"

    def test_campaign_from_api(self):
        """Test Campaign.from_api parsing."""
        from aragora.connectors.marketing.klaviyo import (
            Campaign,
            CampaignStatus,
            MessageChannel,
        )

        data = {
            "id": "camp_123",
            "attributes": {
                "name": "Summer Sale Announcement",
                "status": "sent",
                "channel": "email",
                "audiences": {
                    "included": [{"type": "list", "id": "abc123"}],
                },
                "created_at": "2024-01-01T00:00:00Z",
                "sent_at": "2024-01-02T00:00:00Z",
            },
        }

        campaign = Campaign.from_api(data)

        assert campaign.id == "camp_123"
        assert campaign.name == "Summer Sale Announcement"
        assert campaign.status == CampaignStatus.SENT
        assert campaign.channel == MessageChannel.EMAIL

    def test_flow_from_api(self):
        """Test Flow.from_api parsing."""
        from aragora.connectors.marketing.klaviyo import Flow, FlowStatus

        data = {
            "id": "flow_123",
            "attributes": {
                "name": "Welcome Series",
                "status": "live",
                "trigger_type": "List",
                "created": "2024-01-01T00:00:00Z",
                "archived": False,
            },
        }

        flow = Flow.from_api(data)

        assert flow.id == "flow_123"
        assert flow.name == "Welcome Series"
        assert flow.status == FlowStatus.LIVE
        assert flow.archived is False

    def test_segment_from_api(self):
        """Test Segment.from_api parsing."""
        from aragora.connectors.marketing.klaviyo import Segment, SegmentType

        data = {
            "id": "seg_123",
            "attributes": {
                "name": "VIP Customers",
                "segment_type": "dynamic",
                "created": "2024-01-01T00:00:00Z",
                "profile_count": 500,
            },
        }

        segment = Segment.from_api(data)

        assert segment.id == "seg_123"
        assert segment.name == "VIP Customers"
        assert segment.segment_type == SegmentType.DYNAMIC
        assert segment.profile_count == 500

    def test_mock_list(self):
        """Test mock list generation."""
        from aragora.connectors.marketing.klaviyo import get_mock_list

        klist = get_mock_list()

        assert klist.id == "abc123"
        assert klist.name == "Newsletter Subscribers"
        assert klist.profile_count == 15000

    def test_mock_profile(self):
        """Test mock profile generation."""
        from aragora.connectors.marketing.klaviyo import get_mock_profile

        profile = get_mock_profile()

        assert profile.id == "prof_123"
        assert profile.email == "test@example.com"

    def test_mock_campaign(self):
        """Test mock campaign generation."""
        from aragora.connectors.marketing.klaviyo import get_mock_campaign

        campaign = get_mock_campaign()

        assert campaign.id == "camp_123"
        assert campaign.name == "Summer Sale Announcement"


class TestMailchimpEnums:
    """Tests for Mailchimp enum values."""

    def test_member_status_enum(self):
        """Test MemberStatus enum values."""
        from aragora.connectors.marketing.mailchimp import MemberStatus

        assert MemberStatus.SUBSCRIBED.value == "subscribed"
        assert MemberStatus.UNSUBSCRIBED.value == "unsubscribed"
        assert MemberStatus.PENDING.value == "pending"

    def test_campaign_status_enum(self):
        """Test CampaignStatus enum values."""
        from aragora.connectors.marketing.mailchimp import CampaignStatus

        assert CampaignStatus.SAVE.value == "save"
        assert CampaignStatus.SENT.value == "sent"
        assert CampaignStatus.SENDING.value == "sending"

    def test_campaign_type_enum(self):
        """Test CampaignType enum values."""
        from aragora.connectors.marketing.mailchimp import CampaignType

        assert CampaignType.REGULAR.value == "regular"
        assert CampaignType.PLAINTEXT.value == "plaintext"
        assert CampaignType.ABSPLIT.value == "absplit"


class TestKlaviyoEnums:
    """Tests for Klaviyo enum values."""

    def test_campaign_status_enum(self):
        """Test CampaignStatus enum values."""
        from aragora.connectors.marketing.klaviyo import CampaignStatus

        assert CampaignStatus.DRAFT.value == "draft"
        assert CampaignStatus.SENT.value == "sent"
        assert CampaignStatus.SCHEDULED.value == "scheduled"

    def test_message_channel_enum(self):
        """Test MessageChannel enum values."""
        from aragora.connectors.marketing.klaviyo import MessageChannel

        assert MessageChannel.EMAIL.value == "email"
        assert MessageChannel.SMS.value == "sms"
        assert MessageChannel.PUSH.value == "push"

    def test_flow_status_enum(self):
        """Test FlowStatus enum values."""
        from aragora.connectors.marketing.klaviyo import FlowStatus

        assert FlowStatus.DRAFT.value == "draft"
        assert FlowStatus.LIVE.value == "live"
        assert FlowStatus.MANUAL.value == "manual"


class TestMarketingPackageImports:
    """Test that marketing imports work correctly."""

    def test_mailchimp_imports(self):
        """Test Mailchimp can be imported from package."""
        from aragora.connectors.marketing import (
            MailchimpConnector,
            MailchimpCredentials,
            MailchimpAudience,
            Member,
            MailchimpCampaign,
            CampaignReport,
            MailchimpError,
        )

        assert MailchimpConnector is not None
        assert MailchimpCredentials is not None

    def test_klaviyo_imports(self):
        """Test Klaviyo can be imported from package."""
        from aragora.connectors.marketing import (
            KlaviyoConnector,
            KlaviyoCredentials,
            KlaviyoList,
            KlaviyoProfile,
            KlaviyoCampaign,
            Flow,
            KlaviyoError,
        )

        assert KlaviyoConnector is not None
        assert KlaviyoCredentials is not None
