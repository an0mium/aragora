"""Verification and auditing endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

VERIFICATION_ENDPOINTS = {
    "/api/verification/status": {
        "get": {
            "tags": ["Verification"],
            "summary": "Verification status",
            "operationId": "listVerificationStatus",
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
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {"200": _ok_response("Probe results")},
            "security": [{"bearerAuth": []}],
        },
    },
}
