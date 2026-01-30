"""
Tests for Mailchimp Marketing Connector.

Comprehensive tests covering:
- Client initialization and configuration
- Audience (list) management
- Subscriber/member management
- Campaign creation and sending
- Template operations
- Automation workflows
- Reporting and analytics
- Rate limiting and error handling
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


class TestMailchimpCredentials:
    """Tests for Mailchimp credential configuration."""

    def test_credentials_creation(self):
        """Test MailchimpCredentials dataclass creation."""
        from aragora.connectors.marketing.mailchimp import MailchimpCredentials

        creds = MailchimpCredentials(
            api_key="abc123xyz-us1",
            server_prefix="us1",
        )

        assert creds.api_key == "abc123xyz-us1"
        assert creds.server_prefix == "us1"

    def test_credentials_different_datacenter(self):
        """Test credentials with different datacenter prefixes."""
        from aragora.connectors.marketing.mailchimp import MailchimpCredentials

        for prefix in ["us1", "us2", "us3", "us4", "us5", "us6", "us7", "us8", "us9", "us10"]:
            creds = MailchimpCredentials(api_key=f"key-{prefix}", server_prefix=prefix)
            assert creds.server_prefix == prefix


class TestMailchimpEnums:
    """Tests for Mailchimp enum values."""

    def test_member_status_values(self):
        """Test MemberStatus enum values."""
        from aragora.connectors.marketing.mailchimp import MemberStatus

        assert MemberStatus.SUBSCRIBED.value == "subscribed"
        assert MemberStatus.UNSUBSCRIBED.value == "unsubscribed"
        assert MemberStatus.CLEANED.value == "cleaned"
        assert MemberStatus.PENDING.value == "pending"
        assert MemberStatus.TRANSACTIONAL.value == "transactional"

    def test_campaign_status_values(self):
        """Test CampaignStatus enum values."""
        from aragora.connectors.marketing.mailchimp import CampaignStatus

        assert CampaignStatus.SAVE.value == "save"
        assert CampaignStatus.PAUSED.value == "paused"
        assert CampaignStatus.SCHEDULE.value == "schedule"
        assert CampaignStatus.SENDING.value == "sending"
        assert CampaignStatus.SENT.value == "sent"

    def test_campaign_type_values(self):
        """Test CampaignType enum values."""
        from aragora.connectors.marketing.mailchimp import CampaignType

        assert CampaignType.REGULAR.value == "regular"
        assert CampaignType.PLAINTEXT.value == "plaintext"
        assert CampaignType.ABSPLIT.value == "absplit"
        assert CampaignType.RSS.value == "rss"
        assert CampaignType.VARIATE.value == "variate"

    def test_automation_status_values(self):
        """Test AutomationStatus enum values."""
        from aragora.connectors.marketing.mailchimp import AutomationStatus

        assert AutomationStatus.SAVE.value == "save"
        assert AutomationStatus.PAUSED.value == "paused"
        assert AutomationStatus.SENDING.value == "sending"


class TestAudienceStatsDataclass:
    """Tests for AudienceStats dataclass."""

    def test_stats_creation(self):
        """Test AudienceStats dataclass creation."""
        from aragora.connectors.marketing.mailchimp import AudienceStats

        stats = AudienceStats(
            member_count=5000,
            unsubscribe_count=150,
            cleaned_count=50,
            campaign_count=25,
            open_rate=25.5,
            click_rate=3.2,
        )

        assert stats.member_count == 5000
        assert stats.unsubscribe_count == 150
        assert stats.open_rate == 25.5

    def test_stats_from_api(self):
        """Test AudienceStats.from_api parsing."""
        from aragora.connectors.marketing.mailchimp import AudienceStats

        data = {
            "member_count": 10000,
            "unsubscribe_count": 500,
            "cleaned_count": 100,
            "member_count_since_send": 200,
            "unsubscribe_count_since_send": 10,
            "cleaned_count_since_send": 5,
            "campaign_count": 50,
            "campaign_last_sent": "2024-01-15T10:30:00Z",
            "merge_field_count": 10,
            "avg_sub_rate": 2.5,
            "avg_unsub_rate": 0.5,
            "target_sub_rate": 3.0,
            "open_rate": 28.5,
            "click_rate": 4.2,
            "last_sub_date": "2024-01-20T08:00:00Z",
            "last_unsub_date": "2024-01-18T14:00:00Z",
        }

        stats = AudienceStats.from_api(data)

        assert stats.member_count == 10000
        assert stats.campaign_count == 50
        assert stats.open_rate == 28.5
        assert stats.campaign_last_sent is not None

    def test_stats_from_api_minimal(self):
        """Test AudienceStats.from_api with minimal data."""
        from aragora.connectors.marketing.mailchimp import AudienceStats

        data = {"member_count": 100}

        stats = AudienceStats.from_api(data)

        assert stats.member_count == 100
        assert stats.open_rate == 0.0
        assert stats.campaign_last_sent is None


class TestAudienceDataclass:
    """Tests for Audience dataclass."""

    def test_audience_creation(self):
        """Test Audience dataclass creation."""
        from aragora.connectors.marketing.mailchimp import Audience, AudienceStats

        audience = Audience(
            id="abc123def",
            name="Newsletter Subscribers",
            contact={"company": "Test Co"},
            permission_reminder="You signed up",
            double_optin=True,
        )

        assert audience.id == "abc123def"
        assert audience.name == "Newsletter Subscribers"
        assert audience.double_optin is True

    def test_audience_from_api(self):
        """Test Audience.from_api parsing."""
        from aragora.connectors.marketing.mailchimp import Audience

        data = {
            "id": "list_123",
            "name": "Marketing List",
            "contact": {
                "company": "ACME Corp",
                "address1": "123 Main St",
                "city": "New York",
                "state": "NY",
                "zip": "10001",
                "country": "US",
            },
            "permission_reminder": "You signed up for updates",
            "campaign_defaults": {
                "from_name": "Marketing Team",
                "from_email": "marketing@acme.com",
                "subject": "",
                "language": "en",
            },
            "notify_on_subscribe": "admin@acme.com",
            "notify_on_unsubscribe": "admin@acme.com",
            "date_created": "2023-06-01T00:00:00Z",
            "stats": {
                "member_count": 5000,
                "open_rate": 22.5,
            },
            "double_optin": True,
        }

        audience = Audience.from_api(data)

        assert audience.id == "list_123"
        assert audience.name == "Marketing List"
        assert audience.contact["company"] == "ACME Corp"
        assert audience.double_optin is True
        assert audience.stats is not None
        assert audience.stats.member_count == 5000

    def test_audience_from_api_no_stats(self):
        """Test Audience.from_api without stats."""
        from aragora.connectors.marketing.mailchimp import Audience

        data = {
            "id": "list_456",
            "name": "Basic List",
        }

        audience = Audience.from_api(data)

        assert audience.id == "list_456"
        assert audience.stats is None


class TestMemberDataclass:
    """Tests for Member dataclass."""

    def test_member_creation(self):
        """Test Member dataclass creation."""
        from aragora.connectors.marketing.mailchimp import Member, MemberStatus

        member = Member(
            id="member_123",
            email_address="john@example.com",
            status=MemberStatus.SUBSCRIBED,
            merge_fields={"FNAME": "John", "LNAME": "Doe"},
            vip=True,
            member_rating=4,
        )

        assert member.id == "member_123"
        assert member.email_address == "john@example.com"
        assert member.status == MemberStatus.SUBSCRIBED
        assert member.vip is True

    def test_member_from_api(self):
        """Test Member.from_api parsing."""
        from aragora.connectors.marketing.mailchimp import Member, MemberStatus

        data = {
            "id": "abc123hash",
            "email_address": "jane@example.com",
            "status": "subscribed",
            "email_type": "html",
            "merge_fields": {"FNAME": "Jane", "LNAME": "Smith", "COMPANY": "TechCo"},
            "interests": {"interest_1": True, "interest_2": False},
            "language": "en",
            "vip": True,
            "location": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "country_code": "US",
            },
            "marketing_permissions": [],
            "ip_signup": "192.168.1.1",
            "timestamp_signup": "2024-01-01T10:00:00Z",
            "ip_opt": "192.168.1.1",
            "timestamp_opt": "2024-01-01T10:05:00Z",
            "member_rating": 5,
            "last_changed": "2024-01-15T12:30:00Z",
            "email_client": "Gmail",
            "tags_count": 3,
            "tags": [{"id": 1, "name": "VIP"}],
            "list_id": "list_123",
        }

        member = Member.from_api(data)

        assert member.id == "abc123hash"
        assert member.email_address == "jane@example.com"
        assert member.status == MemberStatus.SUBSCRIBED
        assert member.merge_fields["FNAME"] == "Jane"
        assert member.vip is True
        assert member.member_rating == 5
        assert member.timestamp_signup is not None

    def test_member_from_api_unsubscribed(self):
        """Test Member.from_api with unsubscribed status."""
        from aragora.connectors.marketing.mailchimp import Member, MemberStatus

        data = {
            "id": "unsub_123",
            "email_address": "unsub@example.com",
            "status": "unsubscribed",
        }

        member = Member.from_api(data)

        assert member.status == MemberStatus.UNSUBSCRIBED


class TestCampaignDataclass:
    """Tests for Campaign dataclass."""

    def test_campaign_creation(self):
        """Test Campaign dataclass creation."""
        from aragora.connectors.marketing.mailchimp import Campaign, CampaignStatus, CampaignType

        campaign = Campaign(
            id="camp_123",
            web_id=12345,
            type=CampaignType.REGULAR,
            status=CampaignStatus.SENT,
            emails_sent=5000,
        )

        assert campaign.id == "camp_123"
        assert campaign.type == CampaignType.REGULAR
        assert campaign.status == CampaignStatus.SENT

    def test_campaign_from_api(self):
        """Test Campaign.from_api parsing."""
        from aragora.connectors.marketing.mailchimp import Campaign, CampaignStatus, CampaignType

        data = {
            "id": "camp_456",
            "web_id": 67890,
            "type": "regular",
            "create_time": "2024-01-01T00:00:00Z",
            "archive_url": "https://example.com/archive/camp_456",
            "long_archive_url": "https://example.com/archive/long/camp_456",
            "status": "sent",
            "emails_sent": 4500,
            "send_time": "2024-01-02T08:00:00Z",
            "content_type": "html",
            "recipients": {
                "list_id": "list_123",
                "list_name": "Newsletter",
                "segment_text": "All subscribers",
                "recipient_count": 4500,
            },
            "settings": {
                "subject_line": "January Newsletter",
                "from_name": "Marketing",
                "reply_to": "reply@example.com",
                "title": "January 2024 Newsletter",
            },
            "tracking": {
                "opens": True,
                "html_clicks": True,
                "text_clicks": True,
            },
        }

        campaign = Campaign.from_api(data)

        assert campaign.id == "camp_456"
        assert campaign.web_id == 67890
        assert campaign.type == CampaignType.REGULAR
        assert campaign.status == CampaignStatus.SENT
        assert campaign.emails_sent == 4500
        assert campaign.settings["subject_line"] == "January Newsletter"

    def test_campaign_from_api_absplit(self):
        """Test Campaign.from_api with A/B split type."""
        from aragora.connectors.marketing.mailchimp import Campaign, CampaignType

        data = {
            "id": "camp_ab",
            "web_id": 11111,
            "type": "absplit",
            "status": "save",
        }

        campaign = Campaign.from_api(data)

        assert campaign.type == CampaignType.ABSPLIT


class TestCampaignReportDataclass:
    """Tests for CampaignReport dataclass."""

    def test_report_creation(self):
        """Test CampaignReport dataclass creation."""
        from aragora.connectors.marketing.mailchimp import CampaignReport

        report = CampaignReport(
            id="camp_123",
            campaign_title="Test Campaign",
            list_id="list_456",
            list_name="Newsletter",
            subject_line="Test Subject",
            emails_sent=5000,
            opens={"opens_total": 2000, "unique_opens": 1500},
            clicks={"clicks_total": 500, "unique_clicks": 400},
            bounces={"hard_bounces": 25, "soft_bounces": 50},
        )

        assert report.id == "camp_123"
        assert report.emails_sent == 5000

    def test_report_open_rate_calculation(self):
        """Test CampaignReport.open_rate property."""
        from aragora.connectors.marketing.mailchimp import CampaignReport

        report = CampaignReport(
            id="camp_123",
            campaign_title="Test",
            list_id="list_1",
            list_name="List",
            subject_line="Subject",
            emails_sent=1000,
            opens={"opens_total": 400},
        )

        assert report.open_rate == 40.0

    def test_report_click_rate_calculation(self):
        """Test CampaignReport.click_rate property."""
        from aragora.connectors.marketing.mailchimp import CampaignReport

        report = CampaignReport(
            id="camp_123",
            campaign_title="Test",
            list_id="list_1",
            list_name="List",
            subject_line="Subject",
            emails_sent=1000,
            clicks={"clicks_total": 100},
        )

        assert report.click_rate == 10.0

    def test_report_bounce_rate_calculation(self):
        """Test CampaignReport.bounce_rate property."""
        from aragora.connectors.marketing.mailchimp import CampaignReport

        report = CampaignReport(
            id="camp_123",
            campaign_title="Test",
            list_id="list_1",
            list_name="List",
            subject_line="Subject",
            emails_sent=1000,
            bounces={"hard_bounces": 30, "soft_bounces": 70},
        )

        assert report.bounce_rate == 10.0

    def test_report_rates_with_zero_emails(self):
        """Test rate calculations with zero emails sent."""
        from aragora.connectors.marketing.mailchimp import CampaignReport

        report = CampaignReport(
            id="camp_123",
            campaign_title="Test",
            list_id="list_1",
            list_name="List",
            subject_line="Subject",
            emails_sent=0,
        )

        assert report.open_rate == 0.0
        assert report.click_rate == 0.0
        assert report.bounce_rate == 0.0

    def test_report_from_api(self):
        """Test CampaignReport.from_api parsing."""
        from aragora.connectors.marketing.mailchimp import CampaignReport

        data = {
            "id": "camp_report",
            "campaign_title": "Monthly Newsletter",
            "list_id": "list_main",
            "list_name": "Main List",
            "subject_line": "Your Monthly Update",
            "emails_sent": 10000,
            "abuse_reports": 2,
            "unsubscribed": 50,
            "send_time": "2024-01-15T08:00:00Z",
            "opens": {
                "opens_total": 4000,
                "unique_opens": 3500,
                "open_rate": 35.0,
            },
            "clicks": {
                "clicks_total": 1000,
                "unique_clicks": 800,
                "click_rate": 8.0,
            },
            "bounces": {
                "hard_bounces": 50,
                "soft_bounces": 100,
            },
            "forwards": {"forwards_count": 25},
            "industry_stats": {
                "type": "Technology",
                "open_rate": 22.0,
                "click_rate": 3.5,
            },
        }

        report = CampaignReport.from_api(data)

        assert report.id == "camp_report"
        assert report.emails_sent == 10000
        assert report.abuse_reports == 2
        assert report.unsubscribed == 50


class TestTemplateDataclass:
    """Tests for Template dataclass."""

    def test_template_creation(self):
        """Test Template dataclass creation."""
        from aragora.connectors.marketing.mailchimp import Template

        template = Template(
            id=12345,
            name="Welcome Email",
            type="user",
            category="newsletter",
            active=True,
        )

        assert template.id == 12345
        assert template.name == "Welcome Email"
        assert template.active is True

    def test_template_from_api(self):
        """Test Template.from_api parsing."""
        from aragora.connectors.marketing.mailchimp import Template

        data = {
            "id": 67890,
            "name": "Newsletter Template",
            "type": "user",
            "category": "newsletter",
            "created_by": "John Doe",
            "date_created": "2024-01-01T00:00:00Z",
            "date_edited": "2024-01-15T12:00:00Z",
            "active": True,
            "folder_id": "folder_123",
            "thumbnail": "https://example.com/thumb.png",
        }

        template = Template.from_api(data)

        assert template.id == 67890
        assert template.name == "Newsletter Template"
        assert template.created_by == "John Doe"
        assert template.date_created is not None


class TestAutomationDataclass:
    """Tests for Automation dataclass."""

    def test_automation_creation(self):
        """Test Automation dataclass creation."""
        from aragora.connectors.marketing.mailchimp import Automation, AutomationStatus

        automation = Automation(
            id="auto_123",
            name="Welcome Series",
            status=AutomationStatus.SENDING,
            emails_sent=1500,
        )

        assert automation.id == "auto_123"
        assert automation.name == "Welcome Series"
        assert automation.status == AutomationStatus.SENDING

    def test_automation_from_api(self):
        """Test Automation.from_api parsing."""
        from aragora.connectors.marketing.mailchimp import Automation, AutomationStatus

        data = {
            "id": "auto_456",
            "status": "sending",
            "emails_sent": 5000,
            "recipients": {
                "list_id": "list_123",
                "list_name": "Newsletter",
            },
            "settings": {
                "title": "Abandoned Cart",
                "from_name": "Sales Team",
                "reply_to": "sales@example.com",
            },
            "tracking": {
                "opens": True,
                "clicks": True,
            },
            "trigger_settings": {
                "workflow_type": "abandonedCart",
            },
            "create_time": "2024-01-01T00:00:00Z",
            "start_time": "2024-01-02T08:00:00Z",
        }

        automation = Automation.from_api(data)

        assert automation.id == "auto_456"
        assert automation.name == "Abandoned Cart"
        assert automation.status == AutomationStatus.SENDING
        assert automation.emails_sent == 5000

    def test_automation_from_api_paused(self):
        """Test Automation.from_api with paused status."""
        from aragora.connectors.marketing.mailchimp import Automation, AutomationStatus

        data = {
            "id": "auto_paused",
            "status": "paused",
            "settings": {"title": "Paused Workflow"},
        }

        automation = Automation.from_api(data)

        assert automation.status == AutomationStatus.PAUSED


class TestMailchimpError:
    """Tests for MailchimpError exception."""

    def test_error_creation(self):
        """Test MailchimpError exception creation."""
        from aragora.connectors.marketing.mailchimp import MailchimpError

        error = MailchimpError(
            message="Rate limit exceeded",
            status_code=429,
            error_type="rate_limit",
        )

        assert str(error) == "Rate limit exceeded"
        assert error.status_code == 429
        assert error.error_type == "rate_limit"

    def test_error_minimal(self):
        """Test MailchimpError with minimal info."""
        from aragora.connectors.marketing.mailchimp import MailchimpError

        error = MailchimpError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.status_code is None
        assert error.error_type is None


class TestMailchimpConnectorInit:
    """Tests for MailchimpConnector initialization."""

    def test_connector_creation(self):
        """Test MailchimpConnector initialization."""
        from aragora.connectors.marketing.mailchimp import MailchimpConnector, MailchimpCredentials

        creds = MailchimpCredentials(api_key="test-us1", server_prefix="us1")
        connector = MailchimpConnector(creds)

        assert connector.credentials == creds
        assert connector._client is None

    def test_connector_base_url(self):
        """Test MailchimpConnector generates correct base URL."""
        from aragora.connectors.marketing.mailchimp import MailchimpConnector, MailchimpCredentials

        creds = MailchimpCredentials(api_key="test-us5", server_prefix="us5")
        connector = MailchimpConnector(creds)

        assert connector.base_url == "https://us5.api.mailchimp.com/3.0"

    def test_connector_different_datacenters(self):
        """Test connector with different datacenter prefixes."""
        from aragora.connectors.marketing.mailchimp import MailchimpConnector, MailchimpCredentials

        for prefix in ["us1", "us10", "us20"]:
            creds = MailchimpCredentials(api_key=f"key-{prefix}", server_prefix=prefix)
            connector = MailchimpConnector(creds)
            assert f"{prefix}.api.mailchimp.com" in connector.base_url


class TestMailchimpConnectorClient:
    """Tests for MailchimpConnector HTTP client management."""

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self):
        """Test _get_client creates HTTP client."""
        from aragora.connectors.marketing.mailchimp import MailchimpConnector, MailchimpCredentials

        creds = MailchimpCredentials(api_key="test-us1", server_prefix="us1")
        connector = MailchimpConnector(creds)

        client = await connector._get_client()

        assert client is not None
        assert connector._client is client

        await connector.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_client(self):
        """Test _get_client reuses existing client."""
        from aragora.connectors.marketing.mailchimp import MailchimpConnector, MailchimpCredentials

        creds = MailchimpCredentials(api_key="test-us1", server_prefix="us1")
        connector = MailchimpConnector(creds)

        client1 = await connector._get_client()
        client2 = await connector._get_client()

        assert client1 is client2

        await connector.close()

    @pytest.mark.asyncio
    async def test_close_clears_client(self):
        """Test close() clears HTTP client."""
        from aragora.connectors.marketing.mailchimp import MailchimpConnector, MailchimpCredentials

        creds = MailchimpCredentials(api_key="test-us1", server_prefix="us1")
        connector = MailchimpConnector(creds)

        await connector._get_client()
        await connector.close()

        assert connector._client is None


class TestMailchimpConnectorAudiences:
    """Tests for MailchimpConnector audience operations."""

    @pytest.mark.asyncio
    async def test_get_audiences(self):
        """Test get_audiences returns audiences."""
        from aragora.connectors.marketing.mailchimp import MailchimpConnector, MailchimpCredentials

        creds = MailchimpCredentials(api_key="test-us1", server_prefix="us1")
        connector = MailchimpConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"lists": []}'
        mock_response.json.return_value = {
            "lists": [
                {"id": "list_1", "name": "List 1"},
                {"id": "list_2", "name": "List 2"},
            ],
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            audiences = await connector.get_audiences()

            assert len(audiences) == 2
            assert audiences[0].id == "list_1"
            assert audiences[1].name == "List 2"

    @pytest.mark.asyncio
    async def test_get_audience_by_id(self):
        """Test get_audience returns single audience."""
        from aragora.connectors.marketing.mailchimp import MailchimpConnector, MailchimpCredentials

        creds = MailchimpCredentials(api_key="test-us1", server_prefix="us1")
        connector = MailchimpConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"{}"
        mock_response.json.return_value = {
            "id": "list_123",
            "name": "My Audience",
            "stats": {"member_count": 5000},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            audience = await connector.get_audience("list_123")

            assert audience.id == "list_123"
            assert audience.name == "My Audience"

    @pytest.mark.asyncio
    async def test_create_audience(self):
        """Test create_audience creates new audience."""
        from aragora.connectors.marketing.mailchimp import MailchimpConnector, MailchimpCredentials

        creds = MailchimpCredentials(api_key="test-us1", server_prefix="us1")
        connector = MailchimpConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"{}"
        mock_response.json.return_value = {
            "id": "list_new",
            "name": "New Audience",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            audience = await connector.create_audience(
                name="New Audience",
                contact={"company": "Test Co"},
                permission_reminder="You signed up",
                campaign_defaults={
                    "from_name": "Test",
                    "from_email": "test@example.com",
                    "subject": "",
                    "language": "en",
                },
            )

            assert audience.id == "list_new"


class TestMailchimpConnectorMembers:
    """Tests for MailchimpConnector member operations."""

    @pytest.mark.asyncio
    async def test_get_members(self):
        """Test get_members returns members."""
        from aragora.connectors.marketing.mailchimp import MailchimpConnector, MailchimpCredentials

        creds = MailchimpCredentials(api_key="test-us1", server_prefix="us1")
        connector = MailchimpConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"members": []}'
        mock_response.json.return_value = {
            "members": [
                {"id": "m1", "email_address": "user1@test.com", "status": "subscribed"},
                {"id": "m2", "email_address": "user2@test.com", "status": "subscribed"},
            ],
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            members = await connector.get_members("list_123")

            assert len(members) == 2
            assert members[0].email_address == "user1@test.com"

    @pytest.mark.asyncio
    async def test_add_member(self):
        """Test add_member adds new member."""
        from aragora.connectors.marketing.mailchimp import (
            MailchimpConnector,
            MailchimpCredentials,
            MemberStatus,
        )

        creds = MailchimpCredentials(api_key="test-us1", server_prefix="us1")
        connector = MailchimpConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"{}"
        mock_response.json.return_value = {
            "id": "new_member",
            "email_address": "new@example.com",
            "status": "subscribed",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            member = await connector.add_member(
                list_id="list_123",
                email_address="new@example.com",
                status=MemberStatus.SUBSCRIBED,
                merge_fields={"FNAME": "New", "LNAME": "User"},
            )

            assert member.email_address == "new@example.com"

    @pytest.mark.asyncio
    async def test_update_member(self):
        """Test update_member updates existing member."""
        from aragora.connectors.marketing.mailchimp import (
            MailchimpConnector,
            MailchimpCredentials,
            MemberStatus,
        )

        creds = MailchimpCredentials(api_key="test-us1", server_prefix="us1")
        connector = MailchimpConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"{}"
        mock_response.json.return_value = {
            "id": "member_123",
            "email_address": "updated@example.com",
            "status": "unsubscribed",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            member = await connector.update_member(
                list_id="list_123",
                subscriber_hash="hash_123",
                status=MemberStatus.UNSUBSCRIBED,
            )

            assert member.status == MemberStatus.UNSUBSCRIBED

    @pytest.mark.asyncio
    async def test_add_member_tags(self):
        """Test add_member_tags adds tags to member."""
        from aragora.connectors.marketing.mailchimp import MailchimpConnector, MailchimpCredentials

        creds = MailchimpCredentials(api_key="test-us1", server_prefix="us1")
        connector = MailchimpConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.content = b""
        mock_response.json.return_value = {}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            # Should not raise
            await connector.add_member_tags(
                list_id="list_123",
                subscriber_hash="hash_123",
                tags=["VIP", "Premium"],
            )


class TestMailchimpConnectorCampaigns:
    """Tests for MailchimpConnector campaign operations."""

    @pytest.mark.asyncio
    async def test_get_campaigns(self):
        """Test get_campaigns returns campaigns."""
        from aragora.connectors.marketing.mailchimp import MailchimpConnector, MailchimpCredentials

        creds = MailchimpCredentials(api_key="test-us1", server_prefix="us1")
        connector = MailchimpConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"campaigns": []}'
        mock_response.json.return_value = {
            "campaigns": [
                {"id": "camp_1", "web_id": 1, "type": "regular", "status": "sent"},
            ],
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            campaigns = await connector.get_campaigns()

            assert len(campaigns) == 1
            assert campaigns[0].id == "camp_1"

    @pytest.mark.asyncio
    async def test_create_campaign(self):
        """Test create_campaign creates new campaign."""
        from aragora.connectors.marketing.mailchimp import (
            MailchimpConnector,
            MailchimpCredentials,
            CampaignType,
        )

        creds = MailchimpCredentials(api_key="test-us1", server_prefix="us1")
        connector = MailchimpConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"{}"
        mock_response.json.return_value = {
            "id": "camp_new",
            "web_id": 99999,
            "type": "regular",
            "status": "save",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            campaign = await connector.create_campaign(
                campaign_type=CampaignType.REGULAR,
                list_id="list_123",
                subject_line="Test Subject",
                from_name="Test Sender",
                reply_to="reply@example.com",
            )

            assert campaign.id == "camp_new"

    @pytest.mark.asyncio
    async def test_send_campaign(self):
        """Test send_campaign sends campaign."""
        from aragora.connectors.marketing.mailchimp import MailchimpConnector, MailchimpCredentials

        creds = MailchimpCredentials(api_key="test-us1", server_prefix="us1")
        connector = MailchimpConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.content = b""
        mock_response.json.return_value = {}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            # Should not raise
            await connector.send_campaign("camp_123")

    @pytest.mark.asyncio
    async def test_get_campaign_report(self):
        """Test get_campaign_report returns report."""
        from aragora.connectors.marketing.mailchimp import MailchimpConnector, MailchimpCredentials

        creds = MailchimpCredentials(api_key="test-us1", server_prefix="us1")
        connector = MailchimpConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"{}"
        mock_response.json.return_value = {
            "id": "camp_report",
            "campaign_title": "Test Campaign",
            "list_id": "list_123",
            "list_name": "Newsletter",
            "subject_line": "Test Subject",
            "emails_sent": 5000,
            "opens": {"opens_total": 2000},
            "clicks": {"clicks_total": 500},
            "bounces": {"hard_bounces": 25, "soft_bounces": 50},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            report = await connector.get_campaign_report("camp_123")

            assert report.id == "camp_report"
            assert report.emails_sent == 5000


class TestMailchimpConnectorAutomations:
    """Tests for MailchimpConnector automation operations."""

    @pytest.mark.asyncio
    async def test_get_automations(self):
        """Test get_automations returns automations."""
        from aragora.connectors.marketing.mailchimp import MailchimpConnector, MailchimpCredentials

        creds = MailchimpCredentials(api_key="test-us1", server_prefix="us1")
        connector = MailchimpConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"automations": []}'
        mock_response.json.return_value = {
            "automations": [
                {
                    "id": "auto_1",
                    "status": "sending",
                    "settings": {"title": "Welcome Flow"},
                },
            ],
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            automations = await connector.get_automations()

            assert len(automations) == 1
            assert automations[0].name == "Welcome Flow"

    @pytest.mark.asyncio
    async def test_start_automation(self):
        """Test start_automation starts automation."""
        from aragora.connectors.marketing.mailchimp import MailchimpConnector, MailchimpCredentials

        creds = MailchimpCredentials(api_key="test-us1", server_prefix="us1")
        connector = MailchimpConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.content = b""
        mock_response.json.return_value = {}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            # Should not raise
            await connector.start_automation("auto_123")

    @pytest.mark.asyncio
    async def test_pause_automation(self):
        """Test pause_automation pauses automation."""
        from aragora.connectors.marketing.mailchimp import MailchimpConnector, MailchimpCredentials

        creds = MailchimpCredentials(api_key="test-us1", server_prefix="us1")
        connector = MailchimpConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.content = b""
        mock_response.json.return_value = {}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            # Should not raise
            await connector.pause_automation("auto_123")


class TestMailchimpConnectorErrorHandling:
    """Tests for MailchimpConnector error handling."""

    @pytest.mark.asyncio
    async def test_api_error_400(self):
        """Test handling of 400 Bad Request errors."""
        from aragora.connectors.marketing.mailchimp import (
            MailchimpConnector,
            MailchimpCredentials,
            MailchimpError,
        )

        creds = MailchimpCredentials(api_key="test-us1", server_prefix="us1")
        connector = MailchimpConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.content = b'{"detail": "Invalid request"}'
        mock_response.json.return_value = {
            "type": "bad_request",
            "detail": "Invalid request parameters",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            with pytest.raises(MailchimpError) as exc_info:
                await connector.get_audiences()

            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_api_error_401_unauthorized(self):
        """Test handling of 401 Unauthorized errors."""
        from aragora.connectors.marketing.mailchimp import (
            MailchimpConnector,
            MailchimpCredentials,
            MailchimpError,
        )

        creds = MailchimpCredentials(api_key="invalid-us1", server_prefix="us1")
        connector = MailchimpConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.content = b'{"detail": "Invalid API key"}'
        mock_response.json.return_value = {
            "type": "authentication_error",
            "detail": "Invalid API key",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            with pytest.raises(MailchimpError) as exc_info:
                await connector.get_audiences()

            assert exc_info.value.status_code == 401


class TestMailchimpMockHelpers:
    """Tests for mock helper functions."""

    def test_get_mock_audience(self):
        """Test get_mock_audience helper."""
        from aragora.connectors.marketing.mailchimp import get_mock_audience

        audience = get_mock_audience()

        assert audience.id == "abc123def"
        assert audience.name == "Newsletter Subscribers"
        assert audience.stats.member_count == 5000

    def test_get_mock_campaign(self):
        """Test get_mock_campaign helper."""
        from aragora.connectors.marketing.mailchimp import get_mock_campaign

        campaign = get_mock_campaign()

        assert campaign.id == "camp123"
        assert campaign.emails_sent == 4500

    def test_get_mock_report(self):
        """Test get_mock_report helper."""
        from aragora.connectors.marketing.mailchimp import get_mock_report

        report = get_mock_report()

        assert report.id == "camp123"
        assert report.emails_sent == 4500
        assert report.opens["opens_total"] == 1800


class TestMailchimpPackageImports:
    """Tests for Mailchimp package imports."""

    def test_all_imports(self):
        """Test that all classes can be imported."""
        from aragora.connectors.marketing.mailchimp import (
            MailchimpConnector,
            MailchimpCredentials,
            MailchimpError,
            Audience,
            AudienceStats,
            Member,
            Campaign,
            CampaignReport,
            Template,
            Automation,
            MemberStatus,
            CampaignStatus,
            CampaignType,
            AutomationStatus,
        )

        assert MailchimpConnector is not None
        assert MailchimpCredentials is not None
        assert MailchimpError is not None
        assert Audience is not None
        assert Member is not None
        assert Campaign is not None
