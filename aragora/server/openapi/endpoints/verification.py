"""Verification and auditing endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

VERIFICATION_ENDPOINTS = {
    "/api/verification/status": {
        "get": {
            "tags": ["Verification"],
            "summary": "Verification status",
            "operationId": "listVerificationStatus",
            "description": "Get the current status of verification services and backend availability.",
            "responses": {"200": _ok_response("Verification status")},
        },
    },
    "/api/verification/formal-verify": {
        "post": {
            "tags": ["Verification"],
            "summary": "Formal verification",
            "operationId": "createVerificationFormalVerify",
            "description": "Run formal verification on claims",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {"200": _ok_response("Verification result")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/debates/capability-probe": {
        "post": {
            "tags": ["Auditing"],
            "summary": "Run capability probe",
            "operationId": "createDebatesCapabilityProbe",
            "description": "Run a capability probe to test agent abilities and identify potential limitations.",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {"200": _ok_response("Probe results")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/debates/deep-audit": {
        "post": {
            "tags": ["Auditing"],
            "summary": "Deep audit",
            "operationId": "createDebatesDeepAudit",
            "description": "Run a comprehensive deep audit on debate results to identify quality issues.",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {"200": _ok_response("Audit results")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/probes/capability": {
        "post": {
            "tags": ["Auditing"],
            "summary": "Capability probe",
            "operationId": "createProbesCapability",
            "description": "Execute a capability probe to assess agent performance across various dimensions.",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {"200": _ok_response("Probe results")},
            "security": [{"bearerAuth": []}],
        },
    },
}
