"""
CRM Connectors.

Integrations for Customer Relationship Management platforms:
- HubSpot (contacts, companies, deals, marketing)
- Pipedrive (deals, persons, organizations, activities)
- Salesforce (via enterprise/crm module)
"""

from aragora.connectors.crm.hubspot import (
    HubSpotConnector,
    HubSpotCredentials,
    Contact,
    Company,
    Deal,
    Engagement,
    Pipeline,
    PipelineStage,
    Owner,
    HubSpotError,
    DealStage,
    EngagementType,
    AssociationType,
    get_mock_contact,
    get_mock_deal,
)
from aragora.connectors.crm.pipedrive import (
    PipedriveClient,
    PipedriveCredentials,
    PipedriveError,
    Person,
    Organization,
    Activity,
    Note,
    Product,
    User,
    DealStatus,
    ActivityType,
    get_mock_person,
    get_mock_organization,
    get_mock_activity,
)

# Alias to avoid name collision with HubSpot Deal
from aragora.connectors.crm.pipedrive import Deal as PipedriveDeal
from aragora.connectors.crm.pipedrive import Pipeline as PipedrivePipeline
from aragora.connectors.crm.pipedrive import Stage as PipedriveStage

__all__ = [
    # HubSpot
    "HubSpotConnector",
    "HubSpotCredentials",
    "Contact",
    "Company",
    "Deal",
    "Engagement",
    "Pipeline",
    "PipelineStage",
    "Owner",
    "HubSpotError",
    "DealStage",
    "EngagementType",
    "AssociationType",
    "get_mock_contact",
    "get_mock_deal",
    # Pipedrive
    "PipedriveClient",
    "PipedriveCredentials",
    "PipedriveError",
    "Person",
    "Organization",
    "Activity",
    "Note",
    "Product",
    "User",
    "DealStatus",
    "ActivityType",
    "PipedriveDeal",
    "PipedrivePipeline",
    "PipedriveStage",
    "get_mock_person",
    "get_mock_organization",
    "get_mock_activity",
]
