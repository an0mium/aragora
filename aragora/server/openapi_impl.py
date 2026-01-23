"""
OpenAPI Schema Generator for Aragora API.

Generates OpenAPI 3.0 specification for all API endpoints.
Endpoints are organized by tag/category for clear documentation.

Usage:
    from aragora.server.openapi import generate_openapi_schema, save_openapi_schema

    # Get schema as dict
    schema = generate_openapi_schema()

    # Save to file
    path, count = save_openapi_schema("docs/api/openapi.json")
"""

import json
from pathlib import Path
from typing import Any

# Import schemas and helpers from submodules
from aragora.server.openapi.schemas import COMMON_SCHEMAS

# Import all endpoint definitions from endpoints subpackage
from aragora.server.openapi.endpoints import ALL_ENDPOINTS

# API version
API_VERSION = "1.0.0"


def generate_openapi_schema() -> dict[str, Any]:
    """Generate complete OpenAPI 3.0 schema."""
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Aragora API",
            "description": "Control plane for multi-agent deliberation across org knowledge and channels. "
            "Orchestrate 15+ AI models to debate your organization's knowledge and deliver "
            "defensible decisions with full audit trails.",
            "version": API_VERSION,
            "contact": {"name": "Aragora Team"},
            "license": {"name": "MIT"},
        },
        "servers": [
            {"url": "http://localhost:8080", "description": "Development server"},
            {"url": "https://api.aragora.ai", "description": "Production server"},
        ],
        "tags": [
            {"name": "System", "description": "Health checks and system status"},
            {"name": "Agents", "description": "Agent management, profiles, and rankings"},
            {"name": "Debates", "description": "Debate operations, history, and export"},
            {"name": "Analytics", "description": "Analysis and aggregated statistics"},
            {"name": "Insights", "description": "Position flips, moments, and patterns"},
            {"name": "Consensus", "description": "Consensus memory and settled questions"},
            {"name": "Relationships", "description": "Agent relationship tracking"},
            {"name": "Memory", "description": "Continuum memory management"},
            {"name": "Belief", "description": "Belief networks and claim analysis"},
            {"name": "Pulse", "description": "Trending topics and suggestions"},
            {"name": "Monitoring", "description": "Metrics and observability"},
            {"name": "Verification", "description": "Formal verification and proofs"},
            {"name": "Auditing", "description": "Capability probes and red teaming"},
            {"name": "Documents", "description": "Document upload and export"},
            {"name": "Codebase", "description": "Codebase security scans and metrics"},
            {"name": "GitHub", "description": "GitHub PR review automation"},
            {"name": "Inbox", "description": "Shared inbox and routing rules"},
            {"name": "Costs", "description": "Cost visibility and budgeting"},
            {"name": "Media", "description": "Audio/video and podcast"},
            {"name": "Social", "description": "Social media publishing"},
            {"name": "Control Plane", "description": "Agent orchestration and task routing"},
            {"name": "Decisions", "description": "Unified decision routing results"},
            {"name": "Plugins", "description": "Plugin management and execution"},
            {"name": "Laboratory", "description": "Emergent trait analysis"},
            {"name": "Tournaments", "description": "Tournament management"},
            {"name": "Genesis", "description": "Agent genesis and lineage"},
            {"name": "Evolution", "description": "Agent evolution tracking"},
            {"name": "Replays", "description": "Debate replay management"},
            {"name": "Learning", "description": "Meta-learning statistics"},
            {"name": "Critiques", "description": "Critique patterns and reputation"},
            {"name": "Routing", "description": "Agent selection and team routing"},
            {"name": "Introspection", "description": "Agent self-awareness queries"},
            {"name": "Workflows", "description": "Workflow management and execution"},
            {"name": "Classification", "description": "Question and content classification"},
            {"name": "Retention", "description": "Data retention policies"},
            {"name": "OAuth", "description": "OAuth authentication flows"},
            {"name": "Audit", "description": "Audit logging and compliance"},
            {"name": "Workspace", "description": "Workspace management"},
            {"name": "Workflow Templates", "description": "Pre-built workflow templates"},
            {"name": "Patterns", "description": "Workflow pattern management"},
            {"name": "Gauntlet", "description": "Decision receipts and risk heatmaps"},
            {"name": "Explainability", "description": "Decision explanations and provenance"},
            {"name": "Cross-Pollination", "description": "Cross-debate knowledge sharing"},
            {"name": "Knowledge Mound", "description": "Knowledge extraction and retrieval"},
            {"name": "Checkpoints", "description": "Debate checkpoint management"},
        ],
        "paths": ALL_ENDPOINTS,
        "components": {
            "schemas": COMMON_SCHEMAS,
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "description": "API token authentication. Set via ARAGORA_API_TOKEN environment variable.",
                },
            },
        },
        "security": [],  # Global security is optional, per-endpoint security defined above
    }


def get_openapi_json() -> str:
    """Get OpenAPI schema as JSON string."""
    return json.dumps(generate_openapi_schema(), indent=2)


def get_openapi_yaml() -> str:
    """Get OpenAPI schema as YAML string."""
    try:
        import yaml

        result: str = yaml.dump(
            generate_openapi_schema(), default_flow_style=False, sort_keys=False
        )
        return result
    except ImportError:
        # Fallback to JSON if PyYAML not installed
        return get_openapi_json()


def handle_openapi_request(format: str = "json") -> tuple[str, str]:
    """Handle request for OpenAPI spec.

    Returns:
        Tuple of (content, content_type)
    """
    if format == "yaml":
        return get_openapi_yaml(), "application/yaml"
    return get_openapi_json(), "application/json"


def save_openapi_schema(output_path: str = "docs/api/openapi.json") -> tuple[str, int]:
    """Save complete OpenAPI schema to file.

    Returns:
        Tuple of (file_path, endpoint_count)
    """
    schema = generate_openapi_schema()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        json.dump(schema, f, indent=2)

    endpoint_count = sum(len(methods) for methods in schema["paths"].values())
    return str(output.absolute()), endpoint_count


def get_endpoint_count() -> int:
    """Get total number of documented endpoints."""
    schema = generate_openapi_schema()
    return sum(len(methods) for methods in schema["paths"].values())


# =============================================================================
# Postman Collection Export (moved to postman_generator.py)
# =============================================================================

# Re-export for backwards compatibility
from aragora.server.postman_generator import (  # noqa: F401
    generate_postman_collection,
    get_postman_json,
    handle_postman_request,
    save_postman_collection,
)
