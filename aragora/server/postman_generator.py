"""
Postman Collection Generator for Aragora API.

Generates Postman Collection v2.1 from the OpenAPI schema for easy API testing.

Usage:
    from aragora.server.postman_generator import generate_postman_collection, save_postman_collection

    # Get collection as dict
    collection = generate_postman_collection()

    # Save to file
    path, count = save_postman_collection("docs/api/aragora.postman_collection.json")
"""

import json
from pathlib import Path
from typing import Any


def generate_postman_collection(base_url: str = "{{base_url}}") -> dict[str, Any]:
    """Generate Postman Collection v2.1 from OpenAPI schema.

    Args:
        base_url: Base URL variable for requests (default: {{base_url}})

    Returns:
        Postman Collection v2.1 format dictionary
    """
    # Import here to avoid circular import
    from aragora.server.openapi_impl import generate_openapi_schema

    schema = generate_openapi_schema()

    # Group endpoints by tag
    tag_items: dict[str, list[dict]] = {}
    for tag in schema.get("tags", []):
        tag_items[tag["name"]] = []

    # Convert each endpoint to Postman request
    for path, methods in schema.get("paths", {}).items():
        for method, details in methods.items():
            if method in ("parameters", "servers"):
                continue

            tags = details.get("tags", ["Other"])
            tag = tags[0] if tags else "Other"

            # Build request
            request_item = _openapi_to_postman_request(
                path=path,
                method=method.upper(),
                details=details,
                base_url=base_url,
            )

            if tag not in tag_items:
                tag_items[tag] = []
            tag_items[tag].append(request_item)

    # Build folder structure
    folders = []
    for tag_name, items in tag_items.items():
        if items:
            tag_info = next(
                (t for t in schema.get("tags", []) if t["name"] == tag_name),
                {"name": tag_name, "description": ""},
            )
            folders.append(
                {
                    "name": tag_name,
                    "description": tag_info.get("description", ""),
                    "item": items,
                }
            )

    return {
        "info": {
            "_postman_id": "aragora-api-collection",
            "name": schema["info"]["title"],
            "description": schema["info"]["description"],
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
            "version": schema["info"]["version"],
        },
        "variable": [
            {
                "key": "base_url",
                "value": "http://localhost:8080",
                "type": "string",
                "description": "API base URL",
            },
            {
                "key": "api_token",
                "value": "",
                "type": "string",
                "description": "API authentication token",
            },
        ],
        "auth": {
            "type": "bearer",
            "bearer": [{"key": "token", "value": "{{api_token}}", "type": "string"}],
        },
        "item": folders,
    }


def _openapi_to_postman_request(
    path: str,
    method: str,
    details: dict[str, Any],
    base_url: str,
) -> dict[str, Any]:
    """Convert OpenAPI endpoint to Postman request format."""
    # Convert path parameters: /api/debates/{id} -> /api/debates/:id
    postman_path = path.replace("{", ":").replace("}", "")

    # Build URL parts
    url_parts = postman_path.strip("/").split("/")

    # Extract path variables
    path_variables = []
    for part in url_parts:
        if part.startswith(":"):
            var_name = part[1:]
            path_variables.append(
                {
                    "key": var_name,
                    "value": "",
                    "description": f"Path parameter: {var_name}",
                }
            )

    # Extract query parameters
    query_params = []
    for param in details.get("parameters", []):
        if param.get("in") == "query":
            query_params.append(
                {
                    "key": param["name"],
                    "value": "",
                    "description": param.get("description", ""),
                    "disabled": not param.get("required", False),
                }
            )

    # Build request body if present
    body = None
    request_body = details.get("requestBody", {})
    if request_body:
        content = request_body.get("content", {})
        if "application/json" in content:
            body = {
                "mode": "raw",
                "raw": "{}",
                "options": {"raw": {"language": "json"}},
            }

    request = {
        "name": details.get("summary", details.get("operationId", f"{method} {path}")),
        "request": {
            "method": method,
            "header": [
                {"key": "Content-Type", "value": "application/json"},
                {"key": "Accept", "value": "application/json"},
            ],
            "url": {
                "raw": f"{base_url}{postman_path}",
                "host": [base_url],
                "path": url_parts,
            },
            "description": details.get("description", ""),
        },
        "response": [],
    }

    if path_variables:
        request["request"]["url"]["variable"] = path_variables

    if query_params:
        request["request"]["url"]["query"] = query_params

    if body:
        request["request"]["body"] = body

    return request


def get_postman_json() -> str:
    """Get Postman collection as JSON string."""
    return json.dumps(generate_postman_collection(), indent=2)


def save_postman_collection(
    output_path: str = "docs/api/aragora.postman_collection.json",
) -> tuple[str, int]:
    """Save Postman collection to file.

    Returns:
        Tuple of (file_path, request_count)
    """
    collection = generate_postman_collection()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        json.dump(collection, f, indent=2)

    # Count requests
    request_count = sum(len(folder.get("item", [])) for folder in collection.get("item", []))
    return str(output.absolute()), request_count


def handle_postman_request() -> tuple[str, str]:
    """Handle request for Postman collection.

    Returns:
        Tuple of (content, content_type)
    """
    return get_postman_json(), "application/json"
