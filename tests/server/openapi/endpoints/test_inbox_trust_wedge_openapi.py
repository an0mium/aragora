from __future__ import annotations

from aragora.server.openapi.endpoints.shared_inbox import INBOX_ENDPOINTS
from aragora.server.openapi_impl import generate_openapi_schema


REQUIRED_METHODS = {
    "/api/v1/inbox/wedge/receipts": {"get", "post"},
    "/api/v1/inbox/wedge/receipts/{receipt_id}": {"get"},
    "/api/v1/inbox/wedge/receipts/{receipt_id}/review": {"post"},
    "/api/v1/inbox/wedge/receipts/{receipt_id}/execute": {"post"},
}


def test_inbox_endpoint_registry_includes_trust_wedge_routes() -> None:
    for path, methods in REQUIRED_METHODS.items():
        assert path in INBOX_ENDPOINTS
        assert set(INBOX_ENDPOINTS[path]) == methods


def test_generated_openapi_schema_includes_trust_wedge_paths() -> None:
    schema = generate_openapi_schema()
    paths = schema["paths"]

    for path, methods in REQUIRED_METHODS.items():
        assert path in paths
        assert methods <= set(paths[path])


def test_generated_openapi_schema_registers_trust_wedge_schemas() -> None:
    schema = generate_openapi_schema()
    components = schema["components"]["schemas"]

    assert "InboxTrustWedgeActionResponse" in components
    assert "InboxTrustWedgeCreateReceiptRequest" in components
    assert "InboxTrustWedgeReviewRequest" in components

    create_ref = schema["paths"]["/api/v1/inbox/wedge/receipts"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["$ref"]
    assert create_ref == "#/components/schemas/InboxTrustWedgeCreateReceiptRequest"

    review_ref = schema["paths"]["/api/v1/inbox/wedge/receipts/{receipt_id}/review"]["post"][
        "requestBody"
    ]["content"]["application/json"]["schema"]["$ref"]
    assert review_ref == "#/components/schemas/InboxTrustWedgeReviewRequest"
