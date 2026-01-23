"""
CRM Connectors.

Integrations for Customer Relationship Management platforms:
- HubSpot (contacts, companies, deals, marketing)
- Salesforce - planned
- Pipedrive - planned
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

__all__ = [
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
]
