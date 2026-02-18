"""Reconciliation endpoint definitions.

Covers bank/book reconciliation, discrepancy management, and approval workflows.
"""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS, AUTH_REQUIREMENTS

RECONCILIATION_ENDPOINTS = {
    "/api/v1/reconciliation/run": {
        "post": {
            "tags": ["Reconciliation"],
            "summary": "Run reconciliation",
            "operationId": "createReconciliationRun",
            "description": "Run a new reconciliation between bank and book transactions.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": False,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "bank_account_id": {
                                    "type": "string",
                                    "description": "Bank account identifier",
                                },
                                "start_date": {
                                    "type": "string",
                                    "format": "date",
                                    "description": "Reconciliation start date (YYYY-MM-DD)",
                                },
                                "end_date": {
                                    "type": "string",
                                    "format": "date",
                                    "description": "Reconciliation end date (YYYY-MM-DD)",
                                },
                                "match_threshold": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "description": "Matching threshold (0.0-1.0)",
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Reconciliation job started", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/reconciliation/{reconciliation_id}/resolve": {
        "post": {
            "tags": ["Reconciliation"],
            "summary": "Resolve discrepancy",
            "operationId": "createReconciliationResolve",
            "description": "Resolve a discrepancy within a reconciliation.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "reconciliation_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Reconciliation job identifier",
                }
            ],
            "requestBody": {
                "required": False,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "discrepancy_id": {
                                    "type": "string",
                                    "description": "ID of the discrepancy to resolve",
                                },
                                "resolution": {
                                    "type": "string",
                                    "enum": ["match", "write_off", "adjust"],
                                    "description": "Resolution type",
                                },
                                "notes": {
                                    "type": "string",
                                    "description": "Resolution notes",
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Resolution confirmation", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/reconciliation/discrepancies/bulk-resolve": {
        "post": {
            "tags": ["Reconciliation"],
            "summary": "Bulk resolve discrepancies",
            "operationId": "createReconciliationDiscrepanciesBulkResolve",
            "description": "Bulk resolve multiple discrepancies at once.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": False,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "discrepancy_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of discrepancy IDs to resolve",
                                },
                                "resolution": {
                                    "type": "string",
                                    "enum": ["match", "write_off", "adjust"],
                                    "description": "Resolution type to apply to all",
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Bulk resolution results", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/reconciliation/{reconciliation_id}/approve": {
        "post": {
            "tags": ["Reconciliation"],
            "summary": "Approve reconciliation",
            "operationId": "createReconciliationApprove",
            "description": "Approve a completed reconciliation.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "reconciliation_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Reconciliation job identifier",
                }
            ],
            "requestBody": {
                "required": False,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "notes": {
                                    "type": "string",
                                    "description": "Optional approval notes",
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Approval confirmation", "StandardSuccessResponse"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
}


__all__ = ["RECONCILIATION_ENDPOINTS"]
