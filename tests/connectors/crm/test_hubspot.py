"""
Tests for HubSpot CRM Connector.

Tests for HubSpot CRM integration including contacts, companies, deals,
engagements, pipelines, and owners.
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal


class TestHubSpotCredentials:
    """Tests for HubSpot credentials."""

    def test_credentials_creation(self):
        """Test HubSpotCredentials dataclass."""
        from aragora.connectors.crm.hubspot import HubSpotCredentials

        creds = HubSpotCredentials(
            access_token="pat-na1-test-token-1234",
        )

        assert creds.access_token == "pat-na1-test-token-1234"
        assert creds.base_url == "https://api.hubapi.com"

    def test_credentials_custom_base_url(self):
        """Test HubSpotCredentials with custom base URL."""
        from aragora.connectors.crm.hubspot import HubSpotCredentials

        creds = HubSpotCredentials(
            access_token="test-token",
            base_url="https://api.sandbox.hubapi.com",
        )

        assert creds.base_url == "https://api.sandbox.hubapi.com"


class TestHubSpotEnums:
    """Tests for HubSpot enums."""

    def test_deal_stage_values(self):
        """Test DealStage enum values."""
        from aragora.connectors.crm.hubspot import DealStage

        assert DealStage.APPOINTMENT_SCHEDULED.value == "appointmentscheduled"
        assert DealStage.QUALIFIED_TO_BUY.value == "qualifiedtobuy"
        assert DealStage.CLOSED_WON.value == "closedwon"
        assert DealStage.CLOSED_LOST.value == "closedlost"

    def test_engagement_type_values(self):
        """Test EngagementType enum values."""
        from aragora.connectors.crm.hubspot import EngagementType

        assert EngagementType.EMAIL.value == "EMAIL"
        assert EngagementType.CALL.value == "CALL"
        assert EngagementType.MEETING.value == "MEETING"
        assert EngagementType.NOTE.value == "NOTE"
        assert EngagementType.TASK.value == "TASK"

    def test_association_type_values(self):
        """Test AssociationType enum values."""
        from aragora.connectors.crm.hubspot import AssociationType

        assert AssociationType.CONTACT_TO_COMPANY.value == "contact_to_company"
        assert AssociationType.DEAL_TO_CONTACT.value == "deal_to_contact"
        assert AssociationType.DEAL_TO_COMPANY.value == "deal_to_company"


class TestContactDataclass:
    """Tests for Contact dataclass."""

    def test_contact_creation(self):
        """Test Contact dataclass creation."""
        from aragora.connectors.crm.hubspot import Contact

        now = datetime.now(timezone.utc)
        contact = Contact(
            id="123",
            email="john.doe@example.com",
            first_name="John",
            last_name="Doe",
            phone="+1234567890",
            company="ACME Inc",
            job_title="Software Engineer",
            created_at=now,
            updated_at=now,
        )

        assert contact.id == "123"
        assert contact.email == "john.doe@example.com"
        assert contact.full_name == "John Doe"

    def test_contact_full_name_with_missing_parts(self):
        """Test Contact.full_name with missing name parts."""
        from aragora.connectors.crm.hubspot import Contact

        contact_first_only = Contact(id="1", first_name="Jane")
        assert contact_first_only.full_name == "Jane"

        contact_last_only = Contact(id="2", last_name="Smith")
        assert contact_last_only.full_name == "Smith"

        contact_no_name = Contact(id="3")
        assert contact_no_name.full_name == ""

    def test_contact_from_api(self):
        """Test Contact.from_api() method."""
        from aragora.connectors.crm.hubspot import Contact

        api_data = {
            "id": "456",
            "properties": {
                "email": "jane@example.com",
                "firstname": "Jane",
                "lastname": "Smith",
                "phone": "+0987654321",
                "company": "Test Corp",
                "jobtitle": "CTO",
                "lifecyclestage": "opportunity",
                "hs_lead_status": "OPEN",
                "hubspot_owner_id": "owner_123",
                "createdate": "2024-01-15T10:30:00.000Z",
                "lastmodifieddate": "2024-01-20T15:45:00.000Z",
            },
            "archived": False,
        }

        contact = Contact.from_api(api_data)

        assert contact.id == "456"
        assert contact.email == "jane@example.com"
        assert contact.first_name == "Jane"
        assert contact.last_name == "Smith"
        assert contact.lifecycle_stage == "opportunity"
        assert contact.archived is False


class TestCompanyDataclass:
    """Tests for Company dataclass."""

    def test_company_creation(self):
        """Test Company dataclass creation."""
        from aragora.connectors.crm.hubspot import Company

        company = Company(
            id="789",
            name="ACME Inc",
            domain="acme.com",
            industry="Technology",
            phone="+1234567890",
            city="San Francisco",
            state="CA",
            country="US",
            num_employees=500,
            annual_revenue=Decimal("10000000"),
        )

        assert company.id == "789"
        assert company.name == "ACME Inc"
        assert company.annual_revenue == Decimal("10000000")

    def test_company_from_api(self):
        """Test Company.from_api() method."""
        from aragora.connectors.crm.hubspot import Company

        api_data = {
            "id": "company_123",
            "properties": {
                "name": "Test Company",
                "domain": "testcompany.com",
                "industry": "SAAS",
                "phone": "+1111111111",
                "city": "New York",
                "state": "NY",
                "country": "United States",
                "numberofemployees": "100",
                "annualrevenue": "5000000",
                "lifecyclestage": "customer",
                "createdate": "2023-06-01T00:00:00.000Z",
                "hs_lastmodifieddate": "2024-01-15T00:00:00.000Z",
            },
            "archived": False,
        }

        company = Company.from_api(api_data)

        assert company.id == "company_123"
        assert company.name == "Test Company"
        assert company.domain == "testcompany.com"
        assert company.num_employees == 100
        assert company.annual_revenue == Decimal("5000000")


class TestDealDataclass:
    """Tests for Deal dataclass."""

    def test_deal_creation(self):
        """Test Deal dataclass creation."""
        from aragora.connectors.crm.hubspot import Deal

        now = datetime.now(timezone.utc)
        deal = Deal(
            id="deal_456",
            name="Enterprise Deal",
            amount=Decimal("50000"),
            stage="qualifiedtobuy",
            pipeline="default",
            close_date=now,
        )

        assert deal.id == "deal_456"
        assert deal.name == "Enterprise Deal"
        assert deal.amount == Decimal("50000")

    def test_deal_from_api(self):
        """Test Deal.from_api() method."""
        from aragora.connectors.crm.hubspot import Deal

        api_data = {
            "id": "deal_789",
            "properties": {
                "dealname": "Big Sale",
                "amount": "25000.50",
                "dealstage": "presentationscheduled",
                "pipeline": "sales_pipeline",
                "closedate": "2024-03-01T00:00:00.000Z",
                "dealtype": "newbusiness",
                "hubspot_owner_id": "owner_456",
                "hs_priority": "high",
                "createdate": "2024-01-10T00:00:00.000Z",
            },
            "archived": False,
        }

        deal = Deal.from_api(api_data)

        assert deal.id == "deal_789"
        assert deal.name == "Big Sale"
        assert deal.amount == Decimal("25000.50")
        assert deal.stage == "presentationscheduled"
        assert deal.priority == "high"


class TestEngagementDataclass:
    """Tests for Engagement dataclass."""

    def test_engagement_creation(self):
        """Test Engagement dataclass creation."""
        from aragora.connectors.crm.hubspot import Engagement, EngagementType

        now = datetime.now(timezone.utc)
        engagement = Engagement(
            id="eng_123",
            type=EngagementType.EMAIL,
            owner_id="owner_1",
            timestamp=now,
            subject="Follow-up email",
            body="Thank you for your time...",
            direction="OUTBOUND",
            associated_contact_ids=["contact_1", "contact_2"],
        )

        assert engagement.id == "eng_123"
        assert engagement.type == EngagementType.EMAIL
        assert engagement.subject == "Follow-up email"
        assert len(engagement.associated_contact_ids) == 2

    def test_engagement_from_api(self):
        """Test Engagement.from_api() method."""
        from aragora.connectors.crm.hubspot import Engagement, EngagementType

        api_data = {
            "engagement": {
                "id": 12345,
                "type": "CALL",
                "ownerId": 789,
                "timestamp": 1705329000000,  # Unix timestamp in ms
                "createdAt": 1705329000000,
            },
            "metadata": {
                "subject": "Discovery call",
                "body": "Call notes here",
                "direction": "OUTBOUND",
                "durationMilliseconds": 1800000,  # 30 minutes
                "status": "COMPLETED",
            },
            "associations": {
                "contactIds": [111, 222],
                "companyIds": [333],
                "dealIds": [444],
            },
        }

        engagement = Engagement.from_api(api_data)

        assert engagement.id == "12345"
        assert engagement.type == EngagementType.CALL
        assert engagement.direction == "OUTBOUND"
        assert engagement.duration_ms == 1800000
        assert engagement.associated_contact_ids == ["111", "222"]


class TestPipelineDataclass:
    """Tests for Pipeline and PipelineStage dataclasses."""

    def test_pipeline_stage_creation(self):
        """Test PipelineStage dataclass creation."""
        from aragora.connectors.crm.hubspot import PipelineStage

        stage = PipelineStage(
            id="stage_1",
            label="Qualified to Buy",
            display_order=2,
            probability=0.25,
            closed_won=False,
        )

        assert stage.id == "stage_1"
        assert stage.label == "Qualified to Buy"
        assert stage.probability == 0.25

    def test_pipeline_stage_from_api(self):
        """Test PipelineStage.from_api() method."""
        from aragora.connectors.crm.hubspot import PipelineStage

        api_data = {
            "id": "stage_123",
            "label": "Contract Sent",
            "displayOrder": 5,
            "metadata": {
                "probability": "0.9",
                "isClosed": False,
            },
        }

        stage = PipelineStage.from_api(api_data)

        assert stage.id == "stage_123"
        assert stage.label == "Contract Sent"
        assert stage.display_order == 5
        assert stage.probability == 0.9

    def test_pipeline_creation(self):
        """Test Pipeline dataclass creation."""
        from aragora.connectors.crm.hubspot import Pipeline, PipelineStage

        pipeline = Pipeline(
            id="pipeline_1",
            label="Sales Pipeline",
            display_order=0,
            active=True,
            stages=[
                PipelineStage(id="s1", label="New", probability=0.0),
                PipelineStage(id="s2", label="Won", probability=1.0, closed_won=True),
            ],
        )

        assert pipeline.id == "pipeline_1"
        assert pipeline.label == "Sales Pipeline"
        assert len(pipeline.stages) == 2

    def test_pipeline_from_api(self):
        """Test Pipeline.from_api() method."""
        from aragora.connectors.crm.hubspot import Pipeline

        api_data = {
            "id": "default",
            "label": "Sales Pipeline",
            "displayOrder": 0,
            "archived": False,
            "stages": [
                {
                    "id": "new",
                    "label": "New",
                    "displayOrder": 0,
                    "metadata": {"probability": "0.2"},
                },
                {
                    "id": "won",
                    "label": "Closed Won",
                    "displayOrder": 6,
                    "metadata": {"probability": "1.0", "isClosed": True},
                },
            ],
        }

        pipeline = Pipeline.from_api(api_data)

        assert pipeline.id == "default"
        assert pipeline.active is True
        assert len(pipeline.stages) == 2
        assert pipeline.stages[0].label == "New"


class TestOwnerDataclass:
    """Tests for Owner dataclass."""

    def test_owner_creation(self):
        """Test Owner dataclass creation."""
        from aragora.connectors.crm.hubspot import Owner

        owner = Owner(
            id="owner_1",
            email="sales@example.com",
            first_name="Sales",
            last_name="Rep",
            user_id=12345,
        )

        assert owner.id == "owner_1"
        assert owner.email == "sales@example.com"
        assert owner.full_name == "Sales Rep"

    def test_owner_from_api(self):
        """Test Owner.from_api() method."""
        from aragora.connectors.crm.hubspot import Owner

        api_data = {
            "id": "owner_456",
            "email": "manager@example.com",
            "firstName": "Account",
            "lastName": "Manager",
            "userId": 67890,
            "teams": [{"id": "team_1", "name": "Sales"}],
            "archived": False,
        }

        owner = Owner.from_api(api_data)

        assert owner.id == "owner_456"
        assert owner.email == "manager@example.com"
        assert owner.full_name == "Account Manager"
        assert owner.user_id == 67890
        assert len(owner.teams) == 1


class TestHubSpotError:
    """Tests for HubSpot error handling."""

    def test_hubspot_error_creation(self):
        """Test HubSpotError exception."""
        from aragora.connectors.crm.hubspot import HubSpotError

        error = HubSpotError(
            message="Rate limit exceeded",
            status_code=429,
            details={"retryAfter": 60},
        )

        assert str(error) == "Rate limit exceeded"
        assert error.status_code == 429
        assert error.details["retryAfter"] == 60

    def test_hubspot_error_minimal(self):
        """Test HubSpotError with minimal info."""
        from aragora.connectors.crm.hubspot import HubSpotError

        error = HubSpotError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.status_code is None
        assert error.details == {}


class TestHubSpotConnector:
    """Tests for HubSpotConnector class."""

    def test_connector_creation(self):
        """Test HubSpotConnector initialization."""
        from aragora.connectors.crm.hubspot import HubSpotConnector, HubSpotCredentials

        credentials = HubSpotCredentials(access_token="test-token")
        connector = HubSpotConnector(credentials)

        assert connector.credentials == credentials
        assert connector._client is None

    @pytest.mark.asyncio
    async def test_connector_get_client(self):
        """Test HubSpotConnector._get_client() method."""
        from aragora.connectors.crm.hubspot import HubSpotConnector, HubSpotCredentials

        credentials = HubSpotCredentials(access_token="test-token")
        connector = HubSpotConnector(credentials)

        # First call should create client
        client = await connector._get_client()
        assert client is not None

        # Second call should return same client
        client2 = await connector._get_client()
        assert client2 is client

        # Cleanup
        await client.aclose()


class TestHubSpotPackageImports:
    """Test that HubSpot package imports work correctly."""

    def test_all_imports(self):
        """Test that main classes can be imported."""
        from aragora.connectors.crm.hubspot import (
            HubSpotConnector,
            HubSpotCredentials,
            HubSpotError,
            Contact,
            Company,
            Deal,
            Engagement,
            Pipeline,
            PipelineStage,
            Owner,
            DealStage,
            EngagementType,
            AssociationType,
        )

        assert HubSpotConnector is not None
        assert HubSpotCredentials is not None
        assert Contact is not None
        assert Company is not None
        assert Deal is not None
