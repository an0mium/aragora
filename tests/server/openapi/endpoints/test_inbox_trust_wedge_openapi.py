from __future__ import annotations

from aragora.server.openapi.endpoints.shared_inbox import INBOX_ENDPOINTS
from aragora.server.openapi_impl import generate_openapi_schema


def test_inbox_endpoint_registry_includes_trust_wedge_routes() -> None:
    assert "/api/v1/inbox/wedge/receipts" in INBOX_ENDPOINTS
    assert set(INBOX_ENDPOINTS["/api/v1/inbox/wedge/receipts"]) == {"get", "post"}
    assert "/api/v1/inbox/wedge/receipts/{receipt_id}" in INBOX_ENDPOINTS
    assert "/api/v1/inbox/wedge/receipts/{receipt_id}/review" in INBOX_ENDPOINTS
    assert "/api/v1/inbox/wedge/receipts/{receipt_id}/execute" in INBOX_ENDPOINTS


def test_generated_openapi_schema_includes_trust_wedge_paths() -> None:
    schema = generate_openapi_schema()
    paths = schema["paths"]

    assert "/api/v1/inbox/wedge/receipts" in paths
    assert "get" in paths["/api/v1/inbox/wedge/receipts"]
    assert "post" in paths["/api/v1/inbox/wedge/receipts"]
    assert "/api/v1/inbox/wedge/receipts/{receipt_id}/review" in paths
    assert "post" in paths["/api/v1/inbox/wedge/receipts/{receipt_id}/review"]
    assert "/api/v1/inbox/wedge/receipts/{receipt_id}/execute" in paths
    assert "post" in paths["/api/v1/inbox/wedge/receipts/{receipt_id}/execute"]


def test_generated_openapi_schema_registers_trust_wedge_schemas() -> None:
    schema = generate_openapi_schema()
    components = schema["components"]["schemas"]

    assert "InboxTrustWedgeActionResponse" in components
    assert "InboxTrustWedgeCreateReceiptRequest" in components
    assert "InboxTrustWedgeReviewRequest" in components

    review_ref = schema["paths"]["/api/v1/inbox/wedge/receipts/{receipt_id}/review"]["post"][
        "requestBody"
    ]["content"]["application/json"]["schema"]["$ref"]
    assert review_ref == "#/components/schemas/InboxTrustWedgeReviewRequest"
