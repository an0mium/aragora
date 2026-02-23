"""SDK endpoint stubs for contract parity.

These endpoints are referenced by Python SDK namespaces but don't yet have
full handler implementations. Adding them to the OpenAPI spec ensures the
contract matrix test passes and documents the planned API surface.
"""

from aragora.server.openapi.helpers import _ok_response

_obj = {"type": "object"}
_str = {"type": "string"}
_arr_obj = {"type": "array", "items": {"type": "object"}}

SDK_MISSING_ENDPOINTS: dict = {
    # --- support ---
    "/api/support/connect": {
        "post": {
            "tags": ["Support"],
            "summary": "Connect support integration",
            "operationId": "createSupportConnect",
            "responses": {"200": _ok_response("Connected", _obj)},
        },
    },
    "/api/support/triage": {
        "post": {
            "tags": ["Support"],
            "summary": "Triage support request",
            "operationId": "createSupportTriage",
            "responses": {"200": _ok_response("Triage result", _obj)},
        },
    },
    "/api/support/auto-respond": {
        "post": {
            "tags": ["Support"],
            "summary": "Auto-respond to support ticket",
            "operationId": "createSupportAutoRespond",
            "responses": {"200": _ok_response("Response sent", _obj)},
        },
    },
    "/api/support/{support_id}": {
        "delete": {
            "tags": ["Support"],
            "summary": "Delete support integration",
            "operationId": "deleteSupportIntegration",
            "parameters": [{"name": "support_id", "in": "path", "required": True, "schema": _str}],
            "responses": {"200": _ok_response("Deleted")},
        },
    },
    "/api/support/{support_id}/tickets": {
        "post": {
            "tags": ["Support"],
            "summary": "Create support ticket",
            "operationId": "createSupportTicket",
            "parameters": [{"name": "support_id", "in": "path", "required": True, "schema": _str}],
            "responses": {"201": _ok_response("Ticket created", _obj)},
        },
    },
    "/api/support/{support_id}/tickets/{ticket_id}": {
        "put": {
            "tags": ["Support"],
            "summary": "Update support ticket",
            "operationId": "updateSupportTicket",
            "parameters": [
                {"name": "support_id", "in": "path", "required": True, "schema": _str},
                {"name": "ticket_id", "in": "path", "required": True, "schema": _str},
            ],
            "responses": {"200": _ok_response("Ticket updated", _obj)},
        },
    },
    "/api/support/{support_id}/tickets/{ticket_id}/reply": {
        "post": {
            "tags": ["Support"],
            "summary": "Reply to support ticket",
            "operationId": "createSupportTicketReply",
            "parameters": [
                {"name": "support_id", "in": "path", "required": True, "schema": _str},
                {"name": "ticket_id", "in": "path", "required": True, "schema": _str},
            ],
            "responses": {"200": _ok_response("Reply sent", _obj)},
        },
    },
    # --- verification (additional) ---
    "/api/verification/proofs": {
        "get": {
            "tags": ["Verification"],
            "summary": "List verification proofs",
            "operationId": "listVerificationProofs",
            "responses": {"200": _ok_response("Proofs list", _arr_obj)},
        },
    },
    "/api/verification/validate": {
        "post": {
            "tags": ["Verification"],
            "summary": "Validate claims",
            "operationId": "createVerificationValidate",
            "responses": {"200": _ok_response("Validation result", _obj)},
        },
    },
    # --- calibration ---
    "/api/calibration/curve": {
        "get": {
            "tags": ["Calibration"],
            "summary": "Get calibration curve",
            "operationId": "getCalibrationCurve",
            "responses": {"200": _ok_response("Calibration curve data", _obj)},
        },
    },
    "/api/calibration/history": {
        "get": {
            "tags": ["Calibration"],
            "summary": "Get calibration history",
            "operationId": "getCalibrationHistory",
            "responses": {"200": _ok_response("Calibration history", _arr_obj)},
        },
    },
    # --- services ---
    "/api/services/{service_id}/health": {
        "get": {
            "tags": ["Services"],
            "summary": "Get service health",
            "operationId": "getServiceHealth",
            "parameters": [{"name": "service_id", "in": "path", "required": True, "schema": _str}],
            "responses": {"200": _ok_response("Service health", _obj)},
        },
    },
    "/api/services/{service_id}/metrics": {
        "get": {
            "tags": ["Services"],
            "summary": "Get service metrics",
            "operationId": "getServiceMetrics",
            "parameters": [{"name": "service_id", "in": "path", "required": True, "schema": _str}],
            "responses": {"200": _ok_response("Service metrics", _obj)},
        },
    },
    # --- flips ---
    "/api/flips/{flip_id}": {
        "get": {
            "tags": ["Flips"],
            "summary": "Get flip details",
            "operationId": "getFlip",
            "parameters": [{"name": "flip_id", "in": "path", "required": True, "schema": _str}],
            "responses": {"200": _ok_response("Flip details", _obj)},
        },
    },
    # --- ecommerce ---
    "/api/ecommerce/connect": {
        "post": {
            "tags": ["Ecommerce"],
            "summary": "Connect ecommerce integration",
            "operationId": "createEcommerceConnect",
            "responses": {"200": _ok_response("Connected", _obj)},
        },
    },
    "/api/ecommerce/sync-inventory": {
        "post": {
            "tags": ["Ecommerce"],
            "summary": "Sync inventory",
            "operationId": "createEcommerceSyncInventory",
            "responses": {"200": _ok_response("Inventory synced", _obj)},
        },
    },
    "/api/ecommerce/ship": {
        "post": {
            "tags": ["Ecommerce"],
            "summary": "Ship order",
            "operationId": "createEcommerceShip",
            "responses": {"200": _ok_response("Shipment created", _obj)},
        },
    },
    "/api/ecommerce/{integration_id}": {
        "delete": {
            "tags": ["Ecommerce"],
            "summary": "Delete ecommerce integration",
            "operationId": "deleteEcommerceIntegration",
            "parameters": [
                {"name": "integration_id", "in": "path", "required": True, "schema": _str}
            ],
            "responses": {"200": _ok_response("Deleted")},
        },
    },
    # --- crm ---
    "/api/crm/connect": {
        "post": {
            "tags": ["CRM"],
            "summary": "Connect CRM integration",
            "operationId": "createCrmConnect",
            "responses": {"200": _ok_response("Connected", _obj)},
        },
    },
    "/api/crm/sync-lead": {
        "post": {
            "tags": ["CRM"],
            "summary": "Sync lead to CRM",
            "operationId": "createCrmSyncLead",
            "responses": {"200": _ok_response("Lead synced", _obj)},
        },
    },
    "/api/crm/enrich": {
        "post": {
            "tags": ["CRM"],
            "summary": "Enrich CRM contact",
            "operationId": "createCrmEnrich",
            "responses": {"200": _ok_response("Contact enriched", _obj)},
        },
    },
    "/api/crm/{integration_id}": {
        "delete": {
            "tags": ["CRM"],
            "summary": "Delete CRM integration",
            "operationId": "deleteCrmIntegration",
            "parameters": [
                {"name": "integration_id", "in": "path", "required": True, "schema": _str}
            ],
            "responses": {"200": _ok_response("Deleted")},
        },
    },
    # --- matches ---
    "/api/matches/stats": {
        "get": {
            "tags": ["Matches"],
            "summary": "Get match statistics",
            "operationId": "getMatchStats",
            "responses": {"200": _ok_response("Match statistics", _obj)},
        },
    },
    "/api/matches/{match_id}": {
        "get": {
            "tags": ["Matches"],
            "summary": "Get match details",
            "operationId": "getMatch",
            "parameters": [{"name": "match_id", "in": "path", "required": True, "schema": _str}],
            "responses": {"200": _ok_response("Match details", _obj)},
        },
    },
    # --- quotas ---
    "/api/quotas/request-increase": {
        "post": {
            "tags": ["Quotas"],
            "summary": "Request quota increase",
            "operationId": "createQuotaIncreaseRequest",
            "responses": {"200": _ok_response("Request submitted", _obj)},
        },
    },
    # --- reputation ---
    "/api/reputation/domain": {
        "get": {
            "tags": ["Reputation"],
            "summary": "Get domain reputation scores",
            "operationId": "getReputationDomain",
            "responses": {"200": _ok_response("Domain reputation", _obj)},
        },
    },
    "/api/reputation/history": {
        "get": {
            "tags": ["Reputation"],
            "summary": "Get reputation history",
            "operationId": "getReputationHistory",
            "responses": {"200": _ok_response("Reputation history", _arr_obj)},
        },
    },
    "/api/reputation/{agent_id}": {
        "get": {
            "tags": ["Reputation"],
            "summary": "Get agent reputation",
            "operationId": "getReputationByAgentId",
            "parameters": [{"name": "agent_id", "in": "path", "required": True, "schema": _str}],
            "responses": {"200": _ok_response("Agent reputation", _obj)},
        },
    },
}

__all__ = ["SDK_MISSING_ENDPOINTS"]
