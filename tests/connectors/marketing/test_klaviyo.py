"""
Tests for Klaviyo Marketing Connector.

Comprehensive tests covering:
- Client initialization and configuration
- Contact/profile management
- List operations
- Segment management
- Campaign operations
- Flow (automation) management
- Event tracking
- Template operations
- Webhook handling
- Rate limiting and error handling
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


class TestKlaviyoCredentials:
    """Tests for Klaviyo credential configuration."""

    def test_credentials_creation(self):
        """Test KlaviyoCredentials dataclass creation."""
        from aragora.connectors.marketing.klaviyo import KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_abc123xyz")

        assert creds.api_key == "pk_abc123xyz"

    def test_credentials_with_private_key(self):
        """Test credentials with private API key format."""
        from aragora.connectors.marketing.klaviyo import KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test_12345678901234567890123456789012")

        assert creds.api_key.startswith("pk_")


class TestKlaviyoEnums:
    """Tests for Klaviyo enum values."""

    def test_profile_subscription_status_values(self):
        """Test ProfileSubscriptionStatus enum values."""
        from aragora.connectors.marketing.klaviyo import ProfileSubscriptionStatus

        assert ProfileSubscriptionStatus.SUBSCRIBED.value == "SUBSCRIBED"
        assert ProfileSubscriptionStatus.UNSUBSCRIBED.value == "UNSUBSCRIBED"
        assert ProfileSubscriptionStatus.NEVER_SUBSCRIBED.value == "NEVER_SUBSCRIBED"

    def test_campaign_status_values(self):
        """Test CampaignStatus enum values."""
        from aragora.connectors.marketing.klaviyo import CampaignStatus

        assert CampaignStatus.DRAFT.value == "draft"
        assert CampaignStatus.SCHEDULED.value == "scheduled"
        assert CampaignStatus.SENT.value == "sent"
        assert CampaignStatus.CANCELLED.value == "cancelled"

    def test_message_channel_values(self):
        """Test MessageChannel enum values."""
        from aragora.connectors.marketing.klaviyo import MessageChannel

        assert MessageChannel.EMAIL.value == "email"
        assert MessageChannel.SMS.value == "sms"
        assert MessageChannel.PUSH.value == "push"

    def test_flow_status_values(self):
        """Test FlowStatus enum values."""
        from aragora.connectors.marketing.klaviyo import FlowStatus

        assert FlowStatus.DRAFT.value == "draft"
        assert FlowStatus.MANUAL.value == "manual"
        assert FlowStatus.LIVE.value == "live"

    def test_segment_type_values(self):
        """Test SegmentType enum values."""
        from aragora.connectors.marketing.klaviyo import SegmentType

        assert SegmentType.STATIC.value == "static"
        assert SegmentType.DYNAMIC.value == "dynamic"


class TestKlaviyoListDataclass:
    """Tests for KlaviyoList dataclass."""

    def test_list_creation(self):
        """Test KlaviyoList dataclass creation."""
        from aragora.connectors.marketing.klaviyo import KlaviyoList

        now = datetime.now(timezone.utc)
        klist = KlaviyoList(
            id="abc123",
            name="Newsletter Subscribers",
            created=now,
            updated=now,
            profile_count=5000,
            opt_in_process="double_opt_in",
        )

        assert klist.id == "abc123"
        assert klist.name == "Newsletter Subscribers"
        assert klist.profile_count == 5000
        assert klist.opt_in_process == "double_opt_in"

    def test_list_from_api(self):
        """Test KlaviyoList.from_api parsing."""
        from aragora.connectors.marketing.klaviyo import KlaviyoList

        data = {
            "id": "abc123",
            "attributes": {
                "name": "Newsletter Subscribers",
                "created": "2024-01-01T00:00:00Z",
                "updated": "2024-01-15T12:30:00Z",
                "profile_count": 15000,
                "opt_in_process": "double_opt_in",
            },
        }

        klist = KlaviyoList.from_api(data)

        assert klist.id == "abc123"
        assert klist.name == "Newsletter Subscribers"
        assert klist.profile_count == 15000
        assert klist.opt_in_process == "double_opt_in"
        assert klist.created is not None

    def test_list_from_api_minimal(self):
        """Test KlaviyoList.from_api with minimal data."""
        from aragora.connectors.marketing.klaviyo import KlaviyoList

        data = {
            "id": "list_123",
            "attributes": {
                "name": "Test List",
            },
        }

        klist = KlaviyoList.from_api(data)

        assert klist.id == "list_123"
        assert klist.name == "Test List"
        assert klist.created is None
        assert klist.profile_count is None
        assert klist.opt_in_process == "single_opt_in"


class TestSegmentDataclass:
    """Tests for Segment dataclass."""

    def test_segment_creation(self):
        """Test Segment dataclass creation."""
        from aragora.connectors.marketing.klaviyo import Segment, SegmentType

        segment = Segment(
            id="seg_123",
            name="VIP Customers",
            definition={"condition": "spent > 1000"},
            segment_type=SegmentType.DYNAMIC,
            profile_count=500,
        )

        assert segment.id == "seg_123"
        assert segment.name == "VIP Customers"
        assert segment.segment_type == SegmentType.DYNAMIC

    def test_segment_from_api(self):
        """Test Segment.from_api parsing."""
        from aragora.connectors.marketing.klaviyo import Segment, SegmentType

        data = {
            "id": "seg_456",
            "attributes": {
                "name": "High Value Customers",
                "segment_type": "dynamic",
                "definition": {"conditions": []},
                "created": "2024-01-01T00:00:00Z",
                "profile_count": 1000,
            },
        }

        segment = Segment.from_api(data)

        assert segment.id == "seg_456"
        assert segment.name == "High Value Customers"
        assert segment.segment_type == SegmentType.DYNAMIC
        assert segment.profile_count == 1000

    def test_segment_from_api_static(self):
        """Test Segment.from_api with static segment type."""
        from aragora.connectors.marketing.klaviyo import Segment, SegmentType

        data = {
            "id": "seg_789",
            "attributes": {
                "name": "Manual List",
                "segment_type": "static",
            },
        }

        segment = Segment.from_api(data)

        assert segment.segment_type == SegmentType.STATIC


class TestProfileDataclass:
    """Tests for Profile dataclass."""

    def test_profile_creation(self):
        """Test Profile dataclass creation."""
        from aragora.connectors.marketing.klaviyo import Profile

        profile = Profile(
            id="prof_123",
            email="john@example.com",
            first_name="John",
            last_name="Doe",
            phone_number="+15551234567",
            properties={"plan": "premium"},
        )

        assert profile.id == "prof_123"
        assert profile.email == "john@example.com"
        assert profile.first_name == "John"
        assert profile.properties["plan"] == "premium"

    def test_profile_from_api(self):
        """Test Profile.from_api parsing."""
        from aragora.connectors.marketing.klaviyo import Profile

        data = {
            "id": "prof_456",
            "attributes": {
                "email": "jane@example.com",
                "first_name": "Jane",
                "last_name": "Smith",
                "phone_number": "+15559876543",
                "external_id": "ext_123",
                "organization": "ACME Corp",
                "title": "CEO",
                "location": {"city": "New York", "country": "US"},
                "properties": {"lifetime_value": 5000},
                "created": "2024-01-01T00:00:00Z",
                "updated": "2024-01-20T15:30:00Z",
                "subscriptions": {"email": {"marketing": {"consent": "SUBSCRIBED"}}},
            },
        }

        profile = Profile.from_api(data)

        assert profile.id == "prof_456"
        assert profile.email == "jane@example.com"
        assert profile.first_name == "Jane"
        assert profile.organization == "ACME Corp"
        assert profile.location["city"] == "New York"
        assert profile.subscriptions["email"]["marketing"]["consent"] == "SUBSCRIBED"

    def test_profile_from_api_minimal(self):
        """Test Profile.from_api with minimal data."""
        from aragora.connectors.marketing.klaviyo import Profile

        data = {
            "id": "prof_min",
            "attributes": {
                "email": "minimal@example.com",
            },
        }

        profile = Profile.from_api(data)

        assert profile.id == "prof_min"
        assert profile.email == "minimal@example.com"
        assert profile.first_name is None
        assert profile.location == {}


class TestCampaignDataclass:
    """Tests for Campaign dataclass."""

    def test_campaign_creation(self):
        """Test Campaign dataclass creation."""
        from aragora.connectors.marketing.klaviyo import (
            Campaign,
            CampaignStatus,
            MessageChannel,
        )

        campaign = Campaign(
            id="camp_123",
            name="Summer Sale",
            status=CampaignStatus.SENT,
            channel=MessageChannel.EMAIL,
            audiences={"included": [{"type": "list", "id": "list_1"}]},
        )

        assert campaign.id == "camp_123"
        assert campaign.name == "Summer Sale"
        assert campaign.status == CampaignStatus.SENT
        assert campaign.channel == MessageChannel.EMAIL

    def test_campaign_from_api(self):
        """Test Campaign.from_api parsing."""
        from aragora.connectors.marketing.klaviyo import (
            Campaign,
            CampaignStatus,
            MessageChannel,
        )

        data = {
            "id": "camp_456",
            "attributes": {
                "name": "Black Friday Campaign",
                "status": "scheduled",
                "channel": "email",
                "audiences": {
                    "included": [{"type": "segment", "id": "seg_123"}],
                    "excluded": [],
                },
                "send_options": {"smart_send": True},
                "tracking_options": {"open_tracking": True},
                "created_at": "2024-01-01T00:00:00Z",
                "scheduled_at": "2024-11-29T08:00:00Z",
            },
        }

        campaign = Campaign.from_api(data)

        assert campaign.id == "camp_456"
        assert campaign.name == "Black Friday Campaign"
        assert campaign.status == CampaignStatus.SCHEDULED
        assert campaign.channel == MessageChannel.EMAIL
        assert campaign.scheduled_at is not None

    def test_campaign_sms_channel(self):
        """Test Campaign with SMS channel."""
        from aragora.connectors.marketing.klaviyo import Campaign, CampaignStatus, MessageChannel

        data = {
            "id": "camp_sms",
            "attributes": {
                "name": "SMS Alert",
                "status": "draft",
                "channel": "sms",
            },
        }

        campaign = Campaign.from_api(data)

        assert campaign.channel == MessageChannel.SMS


class TestCampaignMessageDataclass:
    """Tests for CampaignMessage dataclass."""

    def test_campaign_message_from_api(self):
        """Test CampaignMessage.from_api parsing."""
        from aragora.connectors.marketing.klaviyo import CampaignMessage, MessageChannel

        data = {
            "id": "msg_123",
            "attributes": {
                "campaign_id": "camp_456",
                "channel": "email",
                "label": "Primary Message",
                "content": {"subject": "Hello!", "body": "Welcome..."},
                "created_at": "2024-01-01T00:00:00Z",
            },
        }

        message = CampaignMessage.from_api(data)

        assert message.id == "msg_123"
        assert message.campaign_id == "camp_456"
        assert message.channel == MessageChannel.EMAIL
        assert message.content["subject"] == "Hello!"


class TestFlowDataclass:
    """Tests for Flow dataclass."""

    def test_flow_creation(self):
        """Test Flow dataclass creation."""
        from aragora.connectors.marketing.klaviyo import Flow, FlowStatus

        flow = Flow(
            id="flow_123",
            name="Welcome Series",
            status=FlowStatus.LIVE,
            trigger_type="List",
        )

        assert flow.id == "flow_123"
        assert flow.name == "Welcome Series"
        assert flow.status == FlowStatus.LIVE

    def test_flow_from_api(self):
        """Test Flow.from_api parsing."""
        from aragora.connectors.marketing.klaviyo import Flow, FlowStatus

        data = {
            "id": "flow_456",
            "attributes": {
                "name": "Abandoned Cart",
                "status": "live",
                "trigger_type": "Metric",
                "created": "2024-01-01T00:00:00Z",
                "archived": False,
            },
        }

        flow = Flow.from_api(data)

        assert flow.id == "flow_456"
        assert flow.name == "Abandoned Cart"
        assert flow.status == FlowStatus.LIVE
        assert flow.trigger_type == "Metric"
        assert flow.archived is False

    def test_flow_from_api_draft(self):
        """Test Flow.from_api with draft status."""
        from aragora.connectors.marketing.klaviyo import Flow, FlowStatus

        data = {
            "id": "flow_draft",
            "attributes": {
                "name": "Draft Flow",
                "status": "draft",
            },
        }

        flow = Flow.from_api(data)

        assert flow.status == FlowStatus.DRAFT


class TestFlowActionDataclass:
    """Tests for FlowAction dataclass."""

    def test_flow_action_from_api(self):
        """Test FlowAction.from_api parsing."""
        from aragora.connectors.marketing.klaviyo import FlowAction

        data = {
            "id": "action_123",
            "attributes": {
                "flow_id": "flow_456",
                "action_type": "EMAIL",
                "status": "live",
                "settings": {"template_id": "tmpl_789"},
            },
        }

        action = FlowAction.from_api(data)

        assert action.id == "action_123"
        assert action.flow_id == "flow_456"
        assert action.action_type == "EMAIL"
        assert action.settings["template_id"] == "tmpl_789"


class TestMetricDataclass:
    """Tests for Metric dataclass."""

    def test_metric_from_api(self):
        """Test Metric.from_api parsing."""
        from aragora.connectors.marketing.klaviyo import Metric

        data = {
            "id": "metric_123",
            "attributes": {
                "name": "Placed Order",
                "integration": {"name": "Shopify"},
                "created": "2024-01-01T00:00:00Z",
            },
        }

        metric = Metric.from_api(data)

        assert metric.id == "metric_123"
        assert metric.name == "Placed Order"
        assert metric.integration["name"] == "Shopify"


class TestEventDataclass:
    """Tests for Event dataclass."""

    def test_event_from_api(self):
        """Test Event.from_api parsing."""
        from aragora.connectors.marketing.klaviyo import Event

        data = {
            "id": "evt_123",
            "attributes": {
                "timestamp": "2024-01-15T10:30:00Z",
                "event_properties": {"value": 99.99, "items": ["SKU001"]},
                "datetime": "2024-01-15T10:30:00",
            },
            "relationships": {
                "metric": {"data": {"id": "metric_456"}},
                "profile": {"data": {"id": "prof_789"}},
            },
        }

        event = Event.from_api(data)

        assert event.id == "evt_123"
        assert event.metric_id == "metric_456"
        assert event.profile_id == "prof_789"
        assert event.event_properties["value"] == 99.99


class TestTemplateDataclass:
    """Tests for Template dataclass."""

    def test_template_creation(self):
        """Test Template dataclass creation."""
        from aragora.connectors.marketing.klaviyo import Template

        template = Template(
            id="tmpl_123",
            name="Welcome Email",
            editor_type="CODE",
            html="<h1>Welcome!</h1>",
            text="Welcome!",
        )

        assert template.id == "tmpl_123"
        assert template.name == "Welcome Email"
        assert template.html == "<h1>Welcome!</h1>"

    def test_template_from_api(self):
        """Test Template.from_api parsing."""
        from aragora.connectors.marketing.klaviyo import Template

        data = {
            "id": "tmpl_456",
            "attributes": {
                "name": "Newsletter Template",
                "editor_type": "DRAG_AND_DROP",
                "html": "<html><body>Content</body></html>",
                "text": "Text version",
                "created": "2024-01-01T00:00:00Z",
            },
        }

        template = Template.from_api(data)

        assert template.id == "tmpl_456"
        assert template.name == "Newsletter Template"
        assert template.editor_type == "DRAG_AND_DROP"


class TestCampaignRecipientEstimation:
    """Tests for CampaignRecipientEstimation dataclass."""

    def test_estimation_from_api(self):
        """Test CampaignRecipientEstimation.from_api parsing."""
        from aragora.connectors.marketing.klaviyo import CampaignRecipientEstimation

        data = {
            "id": "est_123",
            "attributes": {
                "estimated_recipient_count": 25000,
            },
        }

        estimation = CampaignRecipientEstimation.from_api(data)

        assert estimation.id == "est_123"
        assert estimation.estimated_recipient_count == 25000


class TestKlaviyoError:
    """Tests for KlaviyoError exception."""

    def test_error_creation(self):
        """Test KlaviyoError exception creation."""
        from aragora.connectors.marketing.klaviyo import KlaviyoError

        error = KlaviyoError(
            message="Rate limit exceeded",
            status_code=429,
            error_id="err_123",
        )

        assert str(error) == "Rate limit exceeded"
        assert error.status_code == 429
        assert error.error_id == "err_123"

    def test_error_minimal(self):
        """Test KlaviyoError with minimal info."""
        from aragora.connectors.marketing.klaviyo import KlaviyoError

        error = KlaviyoError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.status_code is None
        assert error.error_id is None


class TestKlaviyoConnectorInit:
    """Tests for KlaviyoConnector initialization."""

    def test_connector_creation(self):
        """Test KlaviyoConnector initialization."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        assert connector.credentials == creds
        assert connector._client is None

    def test_connector_base_url(self):
        """Test KlaviyoConnector has correct base URL."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        assert connector.BASE_URL == "https://a.klaviyo.com/api"

    def test_connector_api_revision(self):
        """Test KlaviyoConnector has API revision."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        assert connector.API_REVISION == "2024-10-15"


class TestKlaviyoConnectorClient:
    """Tests for KlaviyoConnector HTTP client management."""

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self):
        """Test _get_client creates HTTP client."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        client = await connector._get_client()

        assert client is not None
        assert connector._client is client

        await connector.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_client(self):
        """Test _get_client reuses existing client."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        client1 = await connector._get_client()
        client2 = await connector._get_client()

        assert client1 is client2

        await connector.close()

    @pytest.mark.asyncio
    async def test_close_clears_client(self):
        """Test close() clears HTTP client."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        await connector._get_client()
        await connector.close()

        assert connector._client is None


class TestKlaviyoConnectorLists:
    """Tests for KlaviyoConnector list operations."""

    @pytest.mark.asyncio
    async def test_get_lists_success(self):
        """Test get_lists returns list of KlaviyoList."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"data": []}'
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "list_1",
                    "attributes": {"name": "List 1"},
                },
                {
                    "id": "list_2",
                    "attributes": {"name": "List 2"},
                },
            ],
            "links": {},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            lists = await connector.get_lists()

            assert len(lists) == 2
            assert lists[0].id == "list_1"
            assert lists[1].name == "List 2"

    @pytest.mark.asyncio
    async def test_get_list_by_id(self):
        """Test get_list returns single list."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"data": {}}'
        mock_response.json.return_value = {
            "data": {
                "id": "list_123",
                "attributes": {"name": "My List", "profile_count": 5000},
            }
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            klist = await connector.get_list("list_123")

            assert klist.id == "list_123"
            assert klist.name == "My List"

    @pytest.mark.asyncio
    async def test_create_list(self):
        """Test create_list creates new list."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.content = b'{"data": {}}'
        mock_response.json.return_value = {
            "data": {
                "id": "list_new",
                "attributes": {"name": "New List"},
            }
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            klist = await connector.create_list("New List")

            assert klist.id == "list_new"
            assert klist.name == "New List"

    @pytest.mark.asyncio
    async def test_add_profiles_to_list(self):
        """Test add_profiles_to_list adds profiles."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.content = b""
        mock_response.json.return_value = {}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            # Should not raise
            await connector.add_profiles_to_list("list_123", ["prof_1", "prof_2"])


class TestKlaviyoConnectorProfiles:
    """Tests for KlaviyoConnector profile operations."""

    @pytest.mark.asyncio
    async def test_get_profiles(self):
        """Test get_profiles returns profiles."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"data": []}'
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "prof_1",
                    "attributes": {"email": "user1@example.com"},
                },
            ],
            "links": {},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            profiles = await connector.get_profiles()

            assert len(profiles) == 1
            assert profiles[0].email == "user1@example.com"

    @pytest.mark.asyncio
    async def test_get_profile_by_id(self):
        """Test get_profile returns single profile."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"data": {}}'
        mock_response.json.return_value = {
            "data": {
                "id": "prof_123",
                "attributes": {
                    "email": "john@example.com",
                    "first_name": "John",
                },
            }
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            profile = await connector.get_profile("prof_123")

            assert profile.id == "prof_123"
            assert profile.email == "john@example.com"

    @pytest.mark.asyncio
    async def test_get_profile_by_email(self):
        """Test get_profile_by_email returns profile."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"data": []}'
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "prof_found",
                    "attributes": {"email": "found@example.com"},
                },
            ],
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            profile = await connector.get_profile_by_email("found@example.com")

            assert profile is not None
            assert profile.id == "prof_found"

    @pytest.mark.asyncio
    async def test_get_profile_by_email_not_found(self):
        """Test get_profile_by_email returns None when not found."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"data": []}'
        mock_response.json.return_value = {"data": []}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            profile = await connector.get_profile_by_email("notfound@example.com")

            assert profile is None

    @pytest.mark.asyncio
    async def test_create_profile(self):
        """Test create_profile creates new profile."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.content = b'{"data": {}}'
        mock_response.json.return_value = {
            "data": {
                "id": "prof_new",
                "attributes": {
                    "email": "new@example.com",
                    "first_name": "New",
                    "last_name": "User",
                },
            }
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            profile = await connector.create_profile(
                email="new@example.com",
                first_name="New",
                last_name="User",
            )

            assert profile.id == "prof_new"
            assert profile.email == "new@example.com"

    @pytest.mark.asyncio
    async def test_update_profile(self):
        """Test update_profile updates existing profile."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"data": {}}'
        mock_response.json.return_value = {
            "data": {
                "id": "prof_123",
                "attributes": {
                    "email": "updated@example.com",
                    "first_name": "Updated",
                },
            }
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            profile = await connector.update_profile(
                profile_id="prof_123",
                first_name="Updated",
            )

            assert profile.first_name == "Updated"


class TestKlaviyoConnectorCampaigns:
    """Tests for KlaviyoConnector campaign operations."""

    @pytest.mark.asyncio
    async def test_get_campaigns(self):
        """Test get_campaigns returns campaigns."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"data": []}'
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "camp_1",
                    "attributes": {
                        "name": "Campaign 1",
                        "status": "sent",
                        "channel": "email",
                    },
                },
            ],
            "links": {},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            campaigns = await connector.get_campaigns()

            assert len(campaigns) == 1
            assert campaigns[0].name == "Campaign 1"

    @pytest.mark.asyncio
    async def test_create_campaign(self):
        """Test create_campaign creates new campaign."""
        from aragora.connectors.marketing.klaviyo import (
            KlaviyoConnector,
            KlaviyoCredentials,
            MessageChannel,
        )

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.content = b'{"data": {}}'
        mock_response.json.return_value = {
            "data": {
                "id": "camp_new",
                "attributes": {
                    "name": "New Campaign",
                    "status": "draft",
                    "channel": "email",
                },
            }
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            campaign = await connector.create_campaign(
                name="New Campaign",
                channel=MessageChannel.EMAIL,
                list_ids=["list_1"],
            )

            assert campaign.id == "camp_new"
            assert campaign.name == "New Campaign"

    @pytest.mark.asyncio
    async def test_send_campaign(self):
        """Test send_campaign sends campaign."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.content = b""
        mock_response.json.return_value = {}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            # Should not raise
            await connector.send_campaign("camp_123")


class TestKlaviyoConnectorFlows:
    """Tests for KlaviyoConnector flow operations."""

    @pytest.mark.asyncio
    async def test_get_flows(self):
        """Test get_flows returns flows."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"data": []}'
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "flow_1",
                    "attributes": {"name": "Welcome", "status": "live"},
                },
            ],
            "links": {},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            flows = await connector.get_flows()

            assert len(flows) == 1
            assert flows[0].name == "Welcome"

    @pytest.mark.asyncio
    async def test_update_flow_status(self):
        """Test update_flow_status updates flow."""
        from aragora.connectors.marketing.klaviyo import (
            KlaviyoConnector,
            KlaviyoCredentials,
            FlowStatus,
        )

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"data": {}}'
        mock_response.json.return_value = {
            "data": {
                "id": "flow_123",
                "attributes": {"name": "Test Flow", "status": "manual"},
            }
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            flow = await connector.update_flow_status("flow_123", FlowStatus.MANUAL)

            assert flow.status == FlowStatus.MANUAL


class TestKlaviyoConnectorEvents:
    """Tests for KlaviyoConnector event operations."""

    @pytest.mark.asyncio
    async def test_create_event(self):
        """Test create_event tracks event."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.content = b""
        mock_response.json.return_value = {}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            # Should not raise
            await connector.create_event(
                event_name="Placed Order",
                profile_email="customer@example.com",
                properties={"value": 99.99},
                value=99.99,
            )

    @pytest.mark.asyncio
    async def test_get_metrics(self):
        """Test get_metrics returns metrics."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"data": []}'
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "metric_1",
                    "attributes": {"name": "Placed Order"},
                },
            ],
            "links": {},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            metrics = await connector.get_metrics()

            assert len(metrics) == 1
            assert metrics[0].name == "Placed Order"


class TestKlaviyoConnectorTemplates:
    """Tests for KlaviyoConnector template operations."""

    @pytest.mark.asyncio
    async def test_get_templates(self):
        """Test get_templates returns templates."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"data": []}'
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "tmpl_1",
                    "attributes": {"name": "Welcome Email"},
                },
            ],
            "links": {},
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            templates = await connector.get_templates()

            assert len(templates) == 1
            assert templates[0].name == "Welcome Email"

    @pytest.mark.asyncio
    async def test_create_template(self):
        """Test create_template creates new template."""
        from aragora.connectors.marketing.klaviyo import KlaviyoConnector, KlaviyoCredentials

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.content = b'{"data": {}}'
        mock_response.json.return_value = {
            "data": {
                "id": "tmpl_new",
                "attributes": {
                    "name": "New Template",
                    "html": "<h1>Hello</h1>",
                },
            }
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            template = await connector.create_template(
                name="New Template",
                html="<h1>Hello</h1>",
            )

            assert template.id == "tmpl_new"


class TestKlaviyoConnectorErrorHandling:
    """Tests for KlaviyoConnector error handling."""

    @pytest.mark.asyncio
    async def test_api_error_400(self):
        """Test handling of 400 Bad Request errors."""
        from aragora.connectors.marketing.klaviyo import (
            KlaviyoConnector,
            KlaviyoCredentials,
            KlaviyoError,
        )

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.content = b'{"errors": []}'
        mock_response.json.return_value = {"errors": [{"id": "err_1", "detail": "Invalid request"}]}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            with pytest.raises(KlaviyoError) as exc_info:
                await connector.get_lists()

            assert exc_info.value.status_code == 400
            assert "Invalid request" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_api_error_401_unauthorized(self):
        """Test handling of 401 Unauthorized errors."""
        from aragora.connectors.marketing.klaviyo import (
            KlaviyoConnector,
            KlaviyoCredentials,
            KlaviyoError,
        )

        creds = KlaviyoCredentials(api_key="invalid_key")
        connector = KlaviyoConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.content = b'{"errors": []}'
        mock_response.json.return_value = {"errors": [{"detail": "Invalid API key"}]}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            with pytest.raises(KlaviyoError) as exc_info:
                await connector.get_lists()

            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_api_error_429_rate_limit(self):
        """Test handling of 429 Rate Limit errors."""
        from aragora.connectors.marketing.klaviyo import (
            KlaviyoConnector,
            KlaviyoCredentials,
            KlaviyoError,
        )

        creds = KlaviyoCredentials(api_key="pk_test")
        connector = KlaviyoConnector(creds)

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.content = b'{"errors": []}'
        mock_response.json.return_value = {"errors": [{"detail": "Rate limit exceeded"}]}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)
            connector._client = mock_instance

            with pytest.raises(KlaviyoError) as exc_info:
                await connector.get_lists()

            assert exc_info.value.status_code == 429


class TestKlaviyoMockHelpers:
    """Tests for mock helper functions."""

    def test_get_mock_list(self):
        """Test get_mock_list helper."""
        from aragora.connectors.marketing.klaviyo import get_mock_list

        klist = get_mock_list()

        assert klist.id == "abc123"
        assert klist.name == "Newsletter Subscribers"
        assert klist.profile_count == 15000

    def test_get_mock_profile(self):
        """Test get_mock_profile helper."""
        from aragora.connectors.marketing.klaviyo import get_mock_profile

        profile = get_mock_profile()

        assert profile.id == "prof_123"
        assert profile.email == "test@example.com"
        assert profile.first_name == "Test"

    def test_get_mock_campaign(self):
        """Test get_mock_campaign helper."""
        from aragora.connectors.marketing.klaviyo import get_mock_campaign

        campaign = get_mock_campaign()

        assert campaign.id == "camp_123"
        assert campaign.name == "Summer Sale Announcement"


class TestKlaviyoPackageImports:
    """Tests for Klaviyo package imports."""

    def test_all_imports(self):
        """Test that all classes can be imported."""
        from aragora.connectors.marketing.klaviyo import (
            KlaviyoConnector,
            KlaviyoCredentials,
            KlaviyoError,
            KlaviyoList,
            Profile,
            Campaign,
            CampaignMessage,
            Flow,
            FlowAction,
            Segment,
            Metric,
            Event,
            Template,
            CampaignRecipientEstimation,
            ProfileSubscriptionStatus,
            CampaignStatus,
            MessageChannel,
            FlowStatus,
            SegmentType,
        )

        assert KlaviyoConnector is not None
        assert KlaviyoCredentials is not None
        assert KlaviyoError is not None
        assert Profile is not None
        assert Campaign is not None
