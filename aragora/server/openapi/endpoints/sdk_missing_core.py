"""Core utilities for SDK missing endpoints.

This module contains shared helper functions and utilities used across
the split sdk_missing modules.
"""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS


def _method_stub(
    tag: str,
    method: str,
    summary: str,
    *,
    op_id: str,
    has_path_param: bool = False,
    has_body: bool = False,
):
    """Build a minimal endpoint operation dict."""
    op: dict = {
        "tags": [tag],
        "summary": summary,
        "operationId": op_id,
        "responses": {
            "200": _ok_response("Success", {"success": {"type": "boolean"}}),
        },
    }
    if has_path_param:
        op["parameters"] = [
            {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
        ]
    if has_body:
        op["requestBody"] = {"content": {"application/json": {"schema": {"type": "object"}}}}
    return op


__all__ = ["_method_stub", "_ok_response", "STANDARD_ERRORS"]
