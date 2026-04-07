"""Tests for admin OpenAPI endpoint definitions."""

from __future__ import annotations


def test_mfa_compliance_endpoint_exists() -> None:
    from aragora.server.openapi.endpoints.admin import ADMIN_ENDPOINTS

    assert "/api/v1/admin/mfa-compliance" in ADMIN_ENDPOINTS


def test_feature_flag_detail_endpoint_supports_get_and_put() -> None:
    from aragora.server.openapi.endpoints.admin import ADMIN_ENDPOINTS

    endpoint = ADMIN_ENDPOINTS["/api/v1/admin/feature-flags/{name}"]

    assert "get" in endpoint
    assert "put" in endpoint
    assert endpoint["get"]["parameters"][0]["name"] == "name"
    assert endpoint["put"]["parameters"][0]["name"] == "name"
    assert {"bearerAuth": []} in endpoint["get"]["security"]
    assert {"bearerAuth": []} in endpoint["put"]["security"]


def test_system_health_endpoints_exist() -> None:
    from aragora.server.openapi.endpoints.admin import ADMIN_ENDPOINTS

    assert "/api/v1/admin/system-health" in ADMIN_ENDPOINTS
    assert "/api/v1/admin/system-health/{section}" in ADMIN_ENDPOINTS


def test_system_health_section_enum_matches_live_sections() -> None:
    from aragora.server.openapi.endpoints.admin import ADMIN_ENDPOINTS

    param = ADMIN_ENDPOINTS["/api/v1/admin/system-health/{section}"]["get"]["parameters"][0]

    assert param["schema"]["enum"] == [
        "circuit-breakers",
        "slos",
        "adapters",
        "agents",
        "budget",
    ]


def test_rotation_status_endpoint_exists() -> None:
    from aragora.server.openapi.endpoints.admin_security import ADMIN_SECURITY_ENDPOINTS

    endpoint = ADMIN_SECURITY_ENDPOINTS["/api/v1/admin/security/rotation-status"]
    schema = endpoint["get"]["responses"]["200"]["content"]["application/json"]["schema"]

    assert {"bearerAuth": []} in endpoint["get"]["security"]
    assert "data" in schema["properties"]
