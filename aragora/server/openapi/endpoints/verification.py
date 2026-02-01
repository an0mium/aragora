"""Verification and auditing endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

VERIFICATION_ENDPOINTS = {
    "/api/verification/status": {
        "get": {
            "tags": ["Verification"],
            "summary": "Verification status",
            "operationId": "listVerificationStatus",
            "description": "Get the current status of verification services and backend availability.",
            "responses": {
                "200": _ok_response(
                    "Verification status",
                    {
                        "available": {"type": "boolean"},
                        "backends": {"type": "array", "items": {"type": "string"}},
                        "z3_version": {"type": "string"},
                    },
                )
            },
        },
    },
    "/api/verification/formal-verify": {
        "post": {
            "tags": ["Verification"],
            "summary": "Formal verification",
            "operationId": "createVerificationFormalVerify",
            "description": "Run formal verification on claims",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response(
                    "Verification result",
                    {
                        "verified": {"type": "boolean"},
                        "proofs": {"type": "array", "items": {"type": "object"}},
                        "counterexamples": {"type": "array", "items": {"type": "object"}},
                    },
                )
            },
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
            "responses": {
                "200": _ok_response(
                    "Probe results",
                    {
                        "probe_id": {"type": "string"},
                        "results": {"type": "array", "items": {"type": "object"}},
                        "limitations": {"type": "array", "items": {"type": "string"}},
                    },
                )
            },
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
            "responses": {
                "200": _ok_response(
                    "Audit results",
                    {
                        "audit_id": {"type": "string"},
                        "issues": {"type": "array", "items": {"type": "object"}},
                        "severity_counts": {"type": "object"},
                    },
                )
            },
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
