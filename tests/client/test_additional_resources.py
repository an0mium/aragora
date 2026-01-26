"""Tests for additional SDK API resources."""

from aragora.client import AragoraClient
from aragora.client.resources import (
    CodebaseAPI,
    CostManagementAPI,
    DecisionsAPI,
    ExplainabilityAPI,
    GmailAPI,
    NotificationsAPI,
    OnboardingAPI,
    OrganizationsAPI,
    PoliciesAPI,
    TenantsAPI,
)


def test_client_exposes_additional_resources() -> None:
    """Ensure new resources are available on the client."""
    client = AragoraClient()
    assert isinstance(client.organizations, OrganizationsAPI)
    assert isinstance(client.tenants, TenantsAPI)
    assert isinstance(client.codebase, CodebaseAPI)
    assert isinstance(client.policies, PoliciesAPI)
    assert isinstance(client.explainability, ExplainabilityAPI)
    assert isinstance(client.cost_management, CostManagementAPI)
    assert isinstance(client.notifications, NotificationsAPI)
    assert isinstance(client.decisions, DecisionsAPI)
    assert isinstance(client.onboarding, OnboardingAPI)
    assert isinstance(client.gmail, GmailAPI)


def test_resource_methods_present() -> None:
    """Spot-check common methods on new resource APIs."""
    client = AragoraClient()
    assert hasattr(client.organizations, "get")
    assert hasattr(client.organizations, "update")
    assert hasattr(client.tenants, "list")
    assert hasattr(client.tenants, "create")
    assert hasattr(client.policies, "list")
    assert hasattr(client.codebase, "list_dependencies")
    assert hasattr(client.cost_management, "get_summary")
    assert hasattr(client.decisions, "create")
    assert hasattr(client.decisions, "get_status")
    assert hasattr(client.notifications, "list")
    assert hasattr(client.onboarding, "get_flow")
    assert hasattr(client.gmail, "list_processed_emails")
