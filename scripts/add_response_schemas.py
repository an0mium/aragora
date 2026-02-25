#!/usr/bin/env python3
"""Add response schemas to non-GET endpoints missing them in the OpenAPI spec.

Classification rules:
- POST endpoints that create resources (heuristic: path has no action verb segment
  at the end, e.g. /api/v1/tenants, /api/v1/connectors, /api/v1/facts):
    201 response with ResourceCreatedResponse schema
- POST endpoints that trigger actions (path ends with action-like segment,
  e.g. /activate, /sync, /verify, /calibrate, /run):
    200 response with StandardSuccessResponse schema
- PUT/PATCH endpoints:
    200 response with StandardSuccessResponse schema
- DELETE endpoints:
    200 response with DeleteSuccessResponse schema

All responses include standard headers (X-Request-ID, X-Response-Time)
and follow the existing spec conventions.
"""

import json
import re
import sys

SPEC_PATH = "docs/api/openapi.json"

# Standard headers used across the spec
STANDARD_HEADERS = {
    "X-Request-ID": {
        "description": "Unique request identifier for tracing and debugging",
        "schema": {"type": "string", "format": "uuid"},
    },
    "X-Response-Time": {
        "description": "Server processing time in milliseconds",
        "schema": {"type": "integer"},
    },
}

# Action-like path segments that indicate a POST is an action, not a create
ACTION_SEGMENTS = {
    # State transitions
    "activate",
    "deactivate",
    "enable",
    "disable",
    "suspend",
    "reactivate",
    "pause",
    "resume",
    "start",
    "stop",
    "unlock",
    "impersonate",
    # Verification / validation
    "verify",
    "verify-comprehensive",
    "validate",
    "check",
    "test",
    "audit",
    "audit-verify",
    "classify",
    "scan",
    "security-scan",
    # Execution / triggers
    "run",
    "execute",
    "approve",
    "reject",
    "resolve",
    "reset",
    "retry",
    "sync",
    "trigger",
    "cleanup",
    "optimize",
    "rebuild",
    "restore-test",
    "calibrate",
    "prioritize",
    "reprioritize",
    "acknowledge",
    "revoke",
    "rotate-secret",
    "cancel",
    "disconnect",
    "fork",
    # Data operations (not creating new top-level resources)
    "connect",
    "query",
    "generate",
    "generate-bundle",
    "estimate",
    "categorize",
    "apply-label",
    "batch",
    "batch-payments",
    "boost",
    "rank-inbox",
    "prioritize",
    "synthesize",
    "convert",
    "search",
    "embed-batch",
    "diff",
    "review",
    "challenge",
    "nudge",
    "inject-evidence",
    "decision-integrity",
    "deidentify",
    "detect-phi",
    "merge",
    "deliver",
    "confirm",
    "notify",
    "heartbeat",
    "toggle",
    "instantiate",
    "install",
    "refresh",
    "right-to-be-forgotten",
    "breach-assessment",
    "bulk-actions",
    "bulk-action",
    "triage",
    # Specific endpoint names
    "snooze",
    "callback",
    "url",
    "publish",
    "subscribe",
    "intervention",
    "results",
    "status",
    "safe-harbor",
}

# Paths that are clearly resource-creation POSTs (create new entity)
CREATE_PATH_PATTERNS = [
    # Exact path endings that create resources
    r"/api/v1/connectors$",
    r"/api/v1/facts$",
    r"/api/v1/index$",
    r"/api/v1/plans$",
    r"/api/v1/policies$",
    r"/api/v1/tenants$",
    r"/api/v1/webhooks/[^/]+/events$",
    r"/api/v1/index/[^/]+/documents$",
    r"/api/v1/facts/relationships$",
    r"/api/v1/incidents/[^/]+/notes$",
    r"/api/v1/blockchain/agents$",
    r"/api/v1/media/audio$",
    r"/api/v1/partners/register$",
    r"/api/v1/payments/customer$",
    r"/api/v1/payments/subscription$",
    r"/api/v1/ap/invoices$",
    r"/api/costs/alerts$",
    r"/api/costs/budgets$",
    r"/api/personas$",
    r"/api/v1/voice/device$",
    r"/api/v1/privacy/preferences$",
    r"/api/v1/email/vip$",
    r"/api/v1/cross-pollination/subscribe$",
    r"/api/v1/inbox/shared/[^/]+/team$",
    r"/api/v1/devices/user/[^/]+/notify$",
    r"/api/v1/devices/[^/]+/notify$",
    r"/api/v1/gateway/devices/[^/]+/heartbeat$",
]


def is_create_endpoint(path: str) -> bool:
    """Determine if a POST endpoint creates a new resource."""
    for pattern in CREATE_PATH_PATTERNS:
        if re.search(pattern, path):
            return True
    return False


def is_action_endpoint(path: str) -> bool:
    """Determine if a POST endpoint triggers an action."""
    # Get the last path segment (strip parameter placeholders)
    segments = path.rstrip("/").split("/")
    last_seg = segments[-1] if segments else ""

    # If last segment is a path parameter like {id}, check the one before it
    if last_seg.startswith("{") and last_seg.endswith("}"):
        if len(segments) >= 2:
            last_seg = segments[-2]

    # Check if the last meaningful segment is an action word
    if last_seg in ACTION_SEGMENTS:
        return True

    # Check second-to-last for compound actions like /safe-harbor/verify
    if len(segments) >= 2:
        second_last = segments[-2]
        if second_last in ACTION_SEGMENTS:
            return True

    return False


def classify_post_endpoint(path: str) -> str:
    """Classify a POST endpoint as 'create' or 'action'."""
    if is_create_endpoint(path):
        return "create"
    if is_action_endpoint(path):
        return "action"

    # Default heuristic: if the path ends in a collection-like noun (no param),
    # treat as action (safer default since most missing are actions)
    return "action"


def build_success_response(description: str, schema_ref: str) -> dict:
    """Build a response object with standard headers and schema ref."""
    return {
        "description": description,
        "headers": STANDARD_HEADERS,
        "content": {
            "application/json": {
                "schema": {"$ref": f"#/components/schemas/{schema_ref}"}
            }
        },
    }


def build_created_response(description: str) -> dict:
    """Build a 201 response for resource creation."""
    return {
        "description": description,
        "headers": STANDARD_HEADERS,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "data": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "ID of the created resource",
                                }
                            },
                        },
                    },
                    "required": ["success"],
                }
            }
        },
    }


def build_delete_response() -> dict:
    """Build a response for DELETE endpoints."""
    return {
        "description": "Resource deleted successfully",
        "headers": STANDARD_HEADERS,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "message": {"type": "string"},
                    },
                    "required": ["success"],
                }
            }
        },
    }


def add_error_response(responses: dict, code: str, description: str) -> None:
    """Add an error response if not already present."""
    if code not in responses:
        responses[code] = {
            "description": description,
            "headers": STANDARD_HEADERS,
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/StandardErrorResponse"}
                }
            },
        }
    elif "content" not in responses[code]:
        # Existing code entry without schema -- add schema
        responses[code]["headers"] = STANDARD_HEADERS
        responses[code]["content"] = {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/StandardErrorResponse"}
            }
        }


def process_endpoint(method: str, path: str, op: dict) -> int:
    """Add response schemas to an endpoint. Returns count of schemas added."""
    responses = op.setdefault("responses", {})
    added = 0

    # Check if already has a schema on any response
    has_schema = any(
        r.get("content", {}).get("application/json", {}).get("schema")
        for r in responses.values()
    )
    if has_schema:
        return 0

    method_lower = method.lower()

    if method_lower == "delete":
        # DELETE endpoints: 200 with success response
        if "200" in responses:
            desc = responses["200"].get("description", "Resource deleted successfully")
            resp = build_delete_response()
            resp["description"] = desc
            responses["200"] = resp
        elif "204" in responses:
            # 204 No Content -- add a schema-bearing 200 alongside
            responses["200"] = build_delete_response()
        else:
            responses["200"] = build_delete_response()
        added += 1

        # Add 404 error response for DELETE
        add_error_response(
            responses, "404", "Not found - The requested resource does not exist"
        )

    elif method_lower in ("put", "patch"):
        # PUT/PATCH: 200 with StandardSuccessResponse
        desc = responses.get("200", {}).get("description", "Updated successfully")
        resp = build_success_response(desc, "StandardSuccessResponse")
        responses["200"] = resp
        added += 1

        # Add common error responses
        add_error_response(
            responses, "404", "Not found - The requested resource does not exist"
        )

    elif method_lower == "post":
        classification = classify_post_endpoint(path)

        if classification == "create":
            # Resource creation: use 201 if no existing 200, otherwise use existing code
            if "200" in responses:
                # Keep existing 200, add schema to it
                desc = responses["200"].get("description", "Created successfully")
                responses["200"] = build_success_response(
                    desc, "StandardSuccessResponse"
                )
            else:
                desc = "Resource created successfully"
                responses["201"] = build_created_response(desc)
            added += 1
        else:
            # Action endpoint: 200 with StandardSuccessResponse
            desc = responses.get("200", {}).get("description", "Operation successful")
            resp = build_success_response(desc, "StandardSuccessResponse")
            responses["200"] = resp
            added += 1

    # Add 500 error response if missing
    add_error_response(
        responses,
        "500",
        "Internal server error - Unexpected error occurred",
    )

    # Enhance existing error code entries that lack schemas
    for code in list(responses.keys()):
        code_int = int(code) if code.isdigit() else 0
        if 400 <= code_int < 600:
            if "content" not in responses[code]:
                responses[code]["headers"] = STANDARD_HEADERS
                responses[code]["content"] = {
                    "application/json": {
                        "schema": {
                            "$ref": "#/components/schemas/StandardErrorResponse"
                        }
                    }
                }

    return added


def main():
    with open(SPEC_PATH) as f:
        spec = json.load(f)

    total_fixed = 0
    details = {"post_action": 0, "post_create": 0, "put": 0, "patch": 0, "delete": 0}

    for path in sorted(spec.get("paths", {}).keys()):
        methods = spec["paths"][path]
        for method in ("post", "put", "patch", "delete"):
            if method not in methods:
                continue
            op = methods[method]

            # Check if missing
            has_schema = any(
                r.get("content", {}).get("application/json", {}).get("schema")
                for r in op.get("responses", {}).values()
            )
            if has_schema:
                continue

            count = process_endpoint(method, path, op)
            if count > 0:
                total_fixed += count
                if method == "post":
                    classification = classify_post_endpoint(path)
                    details[f"post_{classification}"] += 1
                else:
                    details[method] += 1

    # Write updated spec
    with open(SPEC_PATH, "w") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Total endpoints fixed: {total_fixed}")
    print(f"  POST (action):  {details['post_action']}")
    print(f"  POST (create):  {details['post_create']}")
    print(f"  PUT:            {details['put']}")
    print(f"  PATCH:          {details['patch']}")
    print(f"  DELETE:         {details['delete']}")

    # Verify coverage
    total_non_get = 0
    missing = 0
    for path_key, methods_val in spec.get("paths", {}).items():
        for m, op_val in methods_val.items():
            if m.lower() in ("post", "put", "patch", "delete"):
                total_non_get += 1
                has_s = any(
                    r.get("content", {}).get("application/json", {}).get("schema")
                    for r in op_val.get("responses", {}).values()
                )
                if not has_s:
                    missing += 1

    coverage = (total_non_get - missing) / total_non_get * 100 if total_non_get else 0
    print(f"\nCoverage: {total_non_get - missing}/{total_non_get} = {coverage:.1f}%")
    if missing > 0:
        print(f"Still missing: {missing}")
        for path_key, methods_val in spec.get("paths", {}).items():
            for m, op_val in methods_val.items():
                if m.lower() in ("post", "put", "patch", "delete"):
                    has_s = any(
                        r.get("content", {}).get("application/json", {}).get("schema")
                        for r in op_val.get("responses", {}).values()
                    )
                    if not has_s:
                        print(f"  {m.upper()} {path_key}")


if __name__ == "__main__":
    main()
