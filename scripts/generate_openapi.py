#!/usr/bin/env python3
"""
Auto-generate OpenAPI spec from handler code.

This script extracts endpoint definitions from handler classes and generates
an OpenAPI 3.0 specification. It uses:
1. ROUTES attributes from handlers for path definitions
2. Method docstrings for descriptions
3. Handler class docstrings for tag descriptions

Usage:
    python scripts/generate_openapi.py
    python scripts/generate_openapi.py --output docs/api/openapi.json
    python scripts/generate_openapi.py --format yaml
"""

import argparse
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# API version
API_VERSION = "1.0.0"


def get_handlers() -> List[Tuple[str, Any]]:
    """Get all handler classes with their routes."""
    handlers = []

    try:
        # Import handlers module
        import aragora.server.handlers as handlers_module

        # Get all exported names
        all_exports = getattr(handlers_module, "__all__", dir(handlers_module))

        for name in all_exports:
            if name.endswith("Handler") and name != "BaseHandler":
                try:
                    cls = getattr(handlers_module, name, None)
                    if cls is not None and hasattr(cls, "ROUTES"):
                        handlers.append((name, cls))
                except Exception as e:
                    print(f"Warning: Could not load {name}: {e}", file=sys.stderr)

    except ImportError as e:
        print(f"Warning: Could not import handlers: {e}", file=sys.stderr)

    return handlers


def extract_tag_from_handler(handler_name: str) -> str:
    """Extract OpenAPI tag from handler class name."""
    # Remove 'Handler' suffix and convert to title case
    name = handler_name.replace("Handler", "")
    # Handle special cases
    tag_map = {
        "System": "System",
        "Debates": "Debates",
        "Agents": "Agents",
        "Pulse": "Pulse",
        "Analytics": "Analytics",
        "Metrics": "Monitoring",
        "Consensus": "Consensus",
        "Belief": "Belief",
        "Critique": "Critiques",
        "Genesis": "Genesis",
        "Replays": "Replays",
        "Tournament": "Tournaments",
        "Memory": "Memory",
        "LeaderboardView": "Agents",
        "Document": "Documents",
        "Verification": "Verification",
        "Auditing": "Auditing",
        "Relationship": "Relationships",
        "Moments": "Insights",
        "Persona": "Agents",
        "Dashboard": "Dashboard",
        "Introspection": "Introspection",
        "Calibration": "Agents",
        "Routing": "Routing",
        "Evolution": "Evolution",
        "EvolutionABTesting": "Evolution",
        "Plugins": "Plugins",
        "Broadcast": "Media",
        "Audio": "Media",
        "SocialMedia": "Social",
        "Laboratory": "Laboratory",
        "Probes": "Auditing",
        "Insights": "Insights",
        "Breakpoints": "Debugging",
        "Learning": "Learning",
        "Gallery": "Gallery",
        "Auth": "Authentication",
        "Billing": "Billing",
        "GraphDebates": "Debates",
        "MatrixDebates": "Debates",
        "Features": "Features",
        "MemoryAnalytics": "Analytics",
        "Gauntlet": "Gauntlet",
        "Slack": "Integrations",
        "Organizations": "Organizations",
        "OAuth": "Authentication",
        "Reviews": "Reviews",
        "FormalVerification": "Verification",
        "Sharing": "Social",
    }
    return tag_map.get(name, name)


def extract_path_params(path: str) -> List[Dict[str, Any]]:
    """Extract path parameters from a route path."""
    params = []
    # Match {param} or {param:type} patterns
    for match in re.finditer(r"\{(\w+)(?::(\w+))?\}", path):
        param_name = match.group(1)
        param_type = match.group(2) or "string"
        params.append(
            {
                "name": param_name,
                "in": "path",
                "required": True,
                "schema": {"type": param_type},
            }
        )
    return params


def convert_path_to_openapi(path: str) -> str:
    """Convert internal path format to OpenAPI format."""
    # Convert /api/agent/{name}/profile to /api/agent/{name}/profile
    # Most paths should already be in correct format
    return path


def extract_method_info(handler_cls: type, method_name: str) -> Optional[Dict[str, Any]]:
    """Extract endpoint info from handler method."""
    method = getattr(handler_cls, method_name, None)
    if method is None:
        return None

    doc = inspect.getdoc(method) or ""

    # Parse docstring for description
    lines = doc.split("\n")
    summary = lines[0] if lines else method_name
    description = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

    return {
        "summary": summary,
        "description": description,
    }


def generate_endpoint_spec(
    path: str,
    handler_name: str,
    handler_cls: type,
    methods: Set[str],
) -> Dict[str, Any]:
    """Generate OpenAPI spec for a single endpoint."""
    tag = extract_tag_from_handler(handler_name)
    path_params = extract_path_params(path)
    openapi_path = convert_path_to_openapi(path)

    spec: Dict[str, Any] = {}

    for method in methods:
        method_lower = method.lower()

        # Generate operation spec
        operation: Dict[str, Any] = {
            "tags": [tag],
            "summary": f"{method} {path}",
            "responses": {
                "200": {"description": "Success"},
                "400": {"description": "Bad request"},
                "401": {"description": "Unauthorized"},
                "404": {"description": "Not found"},
                "500": {"description": "Server error"},
            },
        }

        # Add path parameters
        if path_params:
            operation["parameters"] = path_params.copy()

        # Add request body for POST/PUT/PATCH
        if method_lower in ("post", "put", "patch"):
            operation["requestBody"] = {
                "content": {"application/json": {"schema": {"type": "object"}}}
            }

        # Check if method requires auth (based on naming convention)
        if any(auth_hint in path for auth_hint in ["/auth/", "/billing/", "/organizations/"]):
            operation["security"] = [{"bearerAuth": []}]

        spec[method_lower] = operation

    return spec


def extract_routes_from_handler(handler_cls: type) -> List[Tuple[str, Set[str]]]:
    """Extract routes and supported methods from a handler class."""
    routes: List[Tuple[str, Set[str]]] = []

    # Get ROUTES attribute
    handler_routes = getattr(handler_cls, "ROUTES", [])

    # Handle dict ROUTES (features handler uses dict)
    if isinstance(handler_routes, dict):
        handler_routes = list(handler_routes.keys())

    # Determine supported methods
    has_get = hasattr(handler_cls, "handle")
    has_post = hasattr(handler_cls, "handle_post")
    has_delete = hasattr(handler_cls, "handle_delete")
    has_patch = hasattr(handler_cls, "handle_patch")
    has_put = hasattr(handler_cls, "handle_put")

    default_methods = set()
    if has_get:
        default_methods.add("GET")
    if has_post:
        default_methods.add("POST")
    if has_delete:
        default_methods.add("DELETE")
    if has_patch:
        default_methods.add("PATCH")
    if has_put:
        default_methods.add("PUT")

    for route in handler_routes:
        routes.append((route, default_methods.copy()))

    return routes


def generate_openapi_schema() -> Dict[str, Any]:
    """Generate complete OpenAPI 3.0 schema from handlers."""
    handlers = get_handlers()

    paths: Dict[str, Any] = {}
    tags: Dict[str, str] = {}  # tag -> description

    for handler_name, handler_cls in handlers:
        tag = extract_tag_from_handler(handler_name)

        # Extract tag description from handler docstring
        handler_doc = inspect.getdoc(handler_cls)
        if handler_doc and tag not in tags:
            tags[tag] = handler_doc.split("\n")[0]

        # Extract routes
        routes = extract_routes_from_handler(handler_cls)

        for path, methods in routes:
            openapi_path = convert_path_to_openapi(path)
            if openapi_path not in paths:
                paths[openapi_path] = {}

            endpoint_spec = generate_endpoint_spec(path, handler_name, handler_cls, methods)
            paths[openapi_path].update(endpoint_spec)

    # Sort paths alphabetically
    sorted_paths = dict(sorted(paths.items()))

    # Build tag list
    tag_list = [
        {"name": tag, "description": desc or f"{tag} operations"}
        for tag, desc in sorted(tags.items())
    ]

    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Aragora API",
            "description": "AI multi-agent debate framework API. Auto-generated from handler code.",
            "version": API_VERSION,
            "contact": {"name": "Aragora Team"},
            "license": {"name": "MIT"},
        },
        "servers": [
            {"url": "http://localhost:8080", "description": "Development server"},
            {"url": "https://api.aragora.ai", "description": "Production server"},
        ],
        "tags": tag_list,
        "paths": sorted_paths,
        "components": {
            "schemas": {
                "Error": {
                    "type": "object",
                    "properties": {
                        "error": {"type": "string"},
                        "code": {"type": "string"},
                        "trace_id": {"type": "string"},
                    },
                    "required": ["error"],
                },
            },
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "description": "API token authentication",
                },
            },
        },
    }


def save_schema(schema: Dict[str, Any], output_path: str, fmt: str = "json") -> int:
    """Save schema to file. Returns endpoint count."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "yaml":
        try:
            import yaml

            content = yaml.dump(schema, default_flow_style=False, sort_keys=False)
        except ImportError:
            print("Warning: PyYAML not installed, using JSON", file=sys.stderr)
            content = json.dumps(schema, indent=2)
    else:
        content = json.dumps(schema, indent=2)

    path.write_text(content)

    # Count endpoints
    endpoint_count = sum(len(methods) for methods in schema["paths"].values())
    return endpoint_count


def main():
    parser = argparse.ArgumentParser(description="Generate OpenAPI spec from handler code")
    parser.add_argument(
        "--output",
        "-o",
        default="docs/api/openapi.json",
        help="Output file path (default: docs/api/openapi.json)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "yaml"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument("--stdout", action="store_true", help="Print to stdout instead of file")
    args = parser.parse_args()

    print("Generating OpenAPI spec from handlers...", file=sys.stderr)

    schema = generate_openapi_schema()

    if args.stdout:
        if args.format == "yaml":
            try:
                import yaml

                print(yaml.dump(schema, default_flow_style=False, sort_keys=False))
            except ImportError:
                print(json.dumps(schema, indent=2))
        else:
            print(json.dumps(schema, indent=2))
    else:
        endpoint_count = save_schema(schema, args.output, args.format)
        print(f"Generated OpenAPI spec with {endpoint_count} endpoints", file=sys.stderr)
        print(f"Output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
