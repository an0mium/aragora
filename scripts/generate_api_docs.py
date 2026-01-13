#!/usr/bin/env python3
"""
Generate API documentation from handler classes.

Scans all handler modules and extracts endpoint definitions and docstrings
to generate comprehensive API documentation.

Usage:
    python scripts/generate_api_docs.py > docs/API_ENDPOINTS.md
    python scripts/generate_api_docs.py --output docs/API_ENDPOINTS.md
    python scripts/generate_api_docs.py --format json
"""

import argparse
import importlib
import inspect
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class Endpoint:
    """Represents an API endpoint."""

    path: str
    method: str = "GET"
    description: str = ""
    handler_class: str = ""
    handler_method: str = ""
    parameters: list[dict] = field(default_factory=list)
    response_example: Optional[str] = None
    auth_required: bool = False


@dataclass
class EndpointGroup:
    """Group of related endpoints."""

    name: str
    description: str
    endpoints: list[Endpoint] = field(default_factory=list)


# Handler modules to scan - dynamically discovered
def discover_handler_modules() -> list[str]:
    """Discover all handler modules in aragora.server.handlers."""
    handlers_dir = Path(__file__).parent.parent / "aragora" / "server" / "handlers"
    modules = []
    for py_file in sorted(handlers_dir.glob("*.py")):
        if py_file.stem.startswith("_") or py_file.stem in ("base", "routing"):
            continue
        modules.append(f"aragora.server.handlers.{py_file.stem}")
    return modules


HANDLER_MODULES = discover_handler_modules()


def extract_endpoints_from_module_docstring(module) -> list[tuple[str, str, str]]:
    """Extract endpoints from module docstring.

    Module docstrings follow the format:
        Endpoints:
        - GET /api/path - Description
        - POST /api/path/{param} - Description

    Returns:
        List of (method, path, description) tuples
    """
    docstring = module.__doc__
    if not docstring:
        return []

    endpoints = []
    match = re.search(r"Endpoints:\s*\n((?:- [A-Z]+ /\S+.*\n?)+)", docstring)
    if not match:
        return []

    endpoint_block = match.group(1)
    for line in endpoint_block.strip().split("\n"):
        endpoint_match = re.match(r"^-\s*([A-Z]+)\s+(/\S+)\s*(?:-\s*(.*))?$", line.strip())
        if endpoint_match:
            method = endpoint_match.group(1)
            path = endpoint_match.group(2)
            description = (endpoint_match.group(3) or "").strip()
            endpoints.append((method, path, description))

    return endpoints


def extract_routes_from_handler(handler_class) -> list[str]:
    """Extract route definitions from a handler class."""
    routes = []

    # Check for ROUTES class attribute
    if hasattr(handler_class, "ROUTES"):
        routes.extend(handler_class.ROUTES)

    # Check for POST_ROUTES class attribute
    if hasattr(handler_class, "POST_ROUTES"):
        routes.extend(handler_class.POST_ROUTES)

    return routes


def extract_docstring_info(docstring: str) -> tuple[str, list[dict]]:
    """Extract description and parameters from docstring.

    Returns:
        Tuple of (description, list of parameter dicts)
    """
    if not docstring:
        return "", []

    lines = docstring.strip().split("\n")
    description_lines = []
    parameters = []

    in_params = False
    in_returns = False
    current_param = None

    for line in lines:
        line = line.strip()

        if line.lower().startswith("args:") or line.lower().startswith("parameters:"):
            in_params = True
            in_returns = False
            continue
        elif line.lower().startswith("returns:"):
            in_params = False
            in_returns = True
            continue
        elif line.lower().startswith("raises:"):
            in_params = False
            in_returns = False
            continue

        if in_params:
            # Match parameter definitions like "param_name: description" or "param_name (type): description"
            param_match = re.match(r"^(\w+)\s*(?:\(([^)]+)\))?\s*:\s*(.+)?$", line)
            if param_match:
                current_param = {
                    "name": param_match.group(1),
                    "type": param_match.group(2) or "string",
                    "description": param_match.group(3) or "",
                }
                parameters.append(current_param)
            elif current_param and line:
                # Continuation of previous parameter description
                current_param["description"] += " " + line
        elif not in_returns:
            description_lines.append(line)

    description = " ".join(description_lines).strip()
    # Clean up description - take first sentence or first 200 chars
    if description:
        first_sentence = description.split(".")[0]
        description = first_sentence[:200] + ("..." if len(first_sentence) > 200 else "")

    return description, parameters


def check_auth_required(handler_class, method_name: str) -> bool:
    """Check if a handler method requires authentication."""
    if hasattr(handler_class, method_name):
        method = getattr(handler_class, method_name)
        # Check for @require_auth decorator
        if hasattr(method, "__wrapped__"):
            return True
        # Check method source for require_auth
        try:
            source = inspect.getsource(method)
            if "@require_auth" in source:
                return True
        except (OSError, TypeError):
            pass
    return False


def discover_endpoints() -> list[EndpointGroup]:
    """Discover all API endpoints from handler modules."""
    groups = []

    for module_name in HANDLER_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            print(f"Warning: Could not import {module_name}: {e}", file=sys.stderr)
            continue

        # First try to extract endpoints from module docstring
        module_endpoints = extract_endpoints_from_module_docstring(module)

        # Get module description from first paragraph before "Endpoints:"
        module_doc = module.__doc__ or ""
        module_desc = ""
        if module_doc:
            desc_match = re.match(r"^(.+?)(?=\n\s*Endpoints:|$)", module_doc, re.DOTALL)
            if desc_match:
                module_desc = desc_match.group(1).strip().split("\n")[0]

        # Find handler classes in the module
        handler_class_name = None
        auth_endpoints = []
        post_routes = []

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if name.endswith("Handler") and hasattr(obj, "ROUTES"):
                handler_class_name = name
                auth_endpoints = getattr(obj, "AUTH_REQUIRED_ENDPOINTS", [])
                post_routes = getattr(obj, "POST_ROUTES", [])
                break

        # If we have module docstring endpoints, use those (preferred)
        if module_endpoints:
            group_name = module_name.split(".")[-1].title().replace("_", " ")
            group = EndpointGroup(name=group_name, description=module_desc)

            for method, path, description in module_endpoints:
                auth_required = any(auth_pattern in path for auth_pattern in auth_endpoints)
                endpoint = Endpoint(
                    path=path,
                    method=method,
                    description=description,
                    handler_class=handler_class_name or "",
                    handler_method="",
                    parameters=[],
                    auth_required=auth_required,
                )
                group.endpoints.append(endpoint)

            if group.endpoints:
                groups.append(group)
            continue

        # Fall back to ROUTES attribute
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if name.endswith("Handler") and hasattr(obj, "ROUTES"):
                group_name = name.replace("Handler", "")
                group_desc = obj.__doc__ or f"{group_name} endpoints"

                # Extract first line of docstring as description
                if group_desc:
                    group_desc = group_desc.strip().split("\n")[0]

                group = EndpointGroup(name=group_name, description=group_desc)

                routes = extract_routes_from_handler(obj)
                post_routes = getattr(obj, "POST_ROUTES", [])

                for route in routes:
                    method = "POST" if route in post_routes else "GET"

                    # Try to find the corresponding handler method
                    handler_method = None
                    method_doc = ""
                    params = []

                    # Look for method that handles this route
                    for attr_name in dir(obj):
                        if attr_name.startswith("_") and not attr_name.startswith("__"):
                            attr = getattr(obj, attr_name)
                            if callable(attr):
                                attr_doc = attr.__doc__ or ""
                                # Check if method handles this route
                                if route in attr_doc or route.split("/")[-1] in attr_name:
                                    handler_method = attr_name
                                    method_doc, params = extract_docstring_info(attr_doc)
                                    break

                    auth_required = (
                        check_auth_required(obj, handler_method) if handler_method else False
                    )

                    endpoint = Endpoint(
                        path=route,
                        method=method,
                        description=method_doc or f"{method} {route}",
                        handler_class=name,
                        handler_method=handler_method or "",
                        parameters=params,
                        auth_required=auth_required,
                    )
                    group.endpoints.append(endpoint)

                if group.endpoints:
                    groups.append(group)

    return groups


def generate_markdown(groups: list[EndpointGroup]) -> str:
    """Generate markdown documentation from endpoint groups."""
    lines = [
        "# Aragora API Documentation",
        "",
        "This document describes the HTTP API endpoints provided by the Aragora server.",
        "",
        "## Table of Contents",
        "",
    ]

    # Generate TOC
    for group in groups:
        anchor = group.name.lower().replace(" ", "-")
        lines.append(f"- [{group.name}](#{anchor})")

    lines.extend(["", "---", ""])

    # Generate endpoint documentation
    for group in groups:
        lines.extend(
            [
                f"## {group.name}",
                "",
                group.description,
                "",
            ]
        )

        for endpoint in group.endpoints:
            method_badge = f"`{endpoint.method}`"
            auth_badge = " 🔒" if endpoint.auth_required else ""

            lines.extend(
                [
                    f"### {method_badge} `{endpoint.path}`{auth_badge}",
                    "",
                    endpoint.description or "No description available.",
                    "",
                ]
            )

            if endpoint.parameters:
                lines.append("**Parameters:**")
                lines.append("")
                lines.append("| Name | Type | Description |")
                lines.append("|------|------|-------------|")
                for param in endpoint.parameters:
                    lines.append(
                        f"| `{param['name']}` | {param['type']} | {param['description']} |"
                    )
                lines.append("")

            if endpoint.response_example:
                lines.extend(
                    [
                        "**Response Example:**",
                        "",
                        "```json",
                        endpoint.response_example,
                        "```",
                        "",
                    ]
                )

        lines.append("---")
        lines.append("")

    # Add footer
    lines.extend(
        [
            "## Authentication",
            "",
            "Endpoints marked with 🔒 require authentication.",
            "",
            "Include an `Authorization` header with your API token:",
            "",
            "```",
            "Authorization: Bearer <your-api-token>",
            "```",
            "",
            "Set `ARAGORA_API_TOKEN` environment variable to configure the token.",
            "",
            "---",
            "",
            "*Generated automatically by `scripts/generate_api_docs.py`*",
        ]
    )

    return "\n".join(lines)


def generate_json(groups: list[EndpointGroup]) -> str:
    """Generate JSON documentation from endpoint groups."""
    data = {"title": "Aragora API", "version": "1.0.0", "groups": []}

    for group in groups:
        group_data = {"name": group.name, "description": group.description, "endpoints": []}

        for endpoint in group.endpoints:
            endpoint_data = {
                "path": endpoint.path,
                "method": endpoint.method,
                "description": endpoint.description,
                "auth_required": endpoint.auth_required,
                "parameters": endpoint.parameters,
            }
            group_data["endpoints"].append(endpoint_data)

        data["groups"].append(group_data)

    return json.dumps(data, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate API documentation from handler classes")
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    args = parser.parse_args()

    print("Discovering endpoints...", file=sys.stderr)
    groups = discover_endpoints()

    total_endpoints = sum(len(g.endpoints) for g in groups)
    print(f"Found {total_endpoints} endpoints in {len(groups)} groups", file=sys.stderr)

    if args.format == "json":
        output = generate_json(groups)
    else:
        output = generate_markdown(groups)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Documentation written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
