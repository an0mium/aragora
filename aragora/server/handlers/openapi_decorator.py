"""
OpenAPI Auto-Generation Decorator for Aragora API Handlers.

Provides decorators that automatically register endpoint metadata for OpenAPI
schema generation. This reduces duplication between handler code and manual
endpoint definitions.

Usage:
    from aragora.server.handlers.openapi_decorator import api_endpoint

    class MyHandler(BaseHandler):
        @api_endpoint(
            path="/api/v1/resource",
            method="GET",
            summary="Get resource",
            tags=["Resources"],
            parameters=[
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
        )
        async def handle_get_resource(self, ...):
            ...

    # To get all registered endpoints:
    from aragora.server.handlers.openapi_decorator import get_registered_endpoints
    endpoints = get_registered_endpoints()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

# Type for decorated functions
F = TypeVar("F", bound=Callable[..., Any])

# Global registry for decorated endpoints
_endpoint_registry: List["OpenAPIEndpoint"] = []


@dataclass
class OpenAPIEndpoint:
    """Metadata for an API endpoint."""

    path: str
    method: str
    summary: str
    tags: List[str]
    description: str = ""
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    security: List[Dict[str, List[str]]] = field(default_factory=list)
    operation_id: Optional[str] = None
    deprecated: bool = False

    def to_openapi_spec(self) -> Dict[str, Any]:
        """Convert to OpenAPI specification format."""
        spec: Dict[str, Any] = {
            "summary": self.summary,
            "tags": self.tags,
        }

        if self.description:
            spec["description"] = self.description

        if self.operation_id:
            spec["operationId"] = self.operation_id

        if self.parameters:
            spec["parameters"] = self.parameters

        if self.request_body:
            spec["requestBody"] = self.request_body

        if self.responses:
            spec["responses"] = self.responses
        else:
            # Default response
            spec["responses"] = {
                "200": {
                    "description": "Success",
                    "content": {
                        "application/json": {
                            "schema": {"type": "object"},
                        }
                    },
                }
            }

        if self.security:
            spec["security"] = self.security

        if self.deprecated:
            spec["deprecated"] = True

        return spec


def api_endpoint(
    path: str,
    method: str = "GET",
    summary: str = "",
    tags: Optional[List[str]] = None,
    description: str = "",
    parameters: Optional[List[Dict[str, Any]]] = None,
    request_body: Optional[Dict[str, Any]] = None,
    responses: Optional[Dict[str, Dict[str, Any]]] = None,
    auth_required: bool = True,
    deprecated: bool = False,
    operation_id: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator to register an endpoint for OpenAPI documentation.

    Args:
        path: The API endpoint path (e.g., "/api/v1/debates")
        method: HTTP method (GET, POST, PUT, DELETE, PATCH)
        summary: Short description for the endpoint
        tags: List of tags for grouping in docs
        description: Detailed description
        parameters: List of parameter definitions
        request_body: Request body schema
        responses: Response schemas by status code
        auth_required: Whether authentication is required
        deprecated: Mark endpoint as deprecated
        operation_id: Optional custom operation ID (defaults to function name)

    Returns:
        Decorated function with _openapi attribute

    Example:
        @api_endpoint(
            path="/api/consensus/similar",
            method="GET",
            summary="Find debates similar to a topic",
            tags=["Consensus"],
            parameters=[
                {"name": "topic", "in": "query", "required": True, "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 5}},
            ],
            auth_required=False,
        )
        async def handle_similar_debates(self, topic: str, limit: int = 5):
            ...
    """

    def decorator(func: F) -> F:
        # Use function name as operation_id if not provided
        op_id = operation_id or func.__name__

        # Use docstring as description if not provided
        desc = description or (func.__doc__.strip() if func.__doc__ else "")

        # Build security requirement
        security: List[Dict[str, List[str]]] = []
        if auth_required:
            security = [{"bearerAuth": []}]

        # Create endpoint metadata
        endpoint = OpenAPIEndpoint(
            path=path,
            method=method.upper(),
            summary=summary or func.__name__.replace("_", " ").title(),
            tags=tags or [],
            description=desc,
            parameters=parameters or [],
            request_body=request_body,
            responses=responses or {},
            security=security,
            operation_id=op_id,
            deprecated=deprecated,
        )

        # Register in global registry
        _endpoint_registry.append(endpoint)

        # Attach metadata to function for introspection
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        wrapper._openapi = endpoint  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


def get_registered_endpoints() -> List[OpenAPIEndpoint]:
    """Get all registered endpoint metadata.

    Returns:
        List of OpenAPIEndpoint objects
    """
    return _endpoint_registry.copy()


def get_registered_endpoints_dict() -> Dict[str, Dict[str, Any]]:
    """Get registered endpoints as OpenAPI paths dictionary.

    This format can be merged directly with ALL_ENDPOINTS.

    Returns:
        Dictionary in OpenAPI paths format
    """
    paths: Dict[str, Dict[str, Any]] = {}

    for endpoint in _endpoint_registry:
        if endpoint.path not in paths:
            paths[endpoint.path] = {}

        paths[endpoint.path][endpoint.method.lower()] = endpoint.to_openapi_spec()

    return paths


def clear_registry() -> None:
    """Clear the endpoint registry. Useful for testing."""
    _endpoint_registry.clear()


def register_endpoint(endpoint: OpenAPIEndpoint) -> None:
    """Manually register an endpoint.

    Args:
        endpoint: OpenAPIEndpoint to register
    """
    _endpoint_registry.append(endpoint)


# Helper functions for common parameter patterns
def path_param(name: str, description: str = "", schema_type: str = "string") -> Dict[str, Any]:
    """Create a path parameter definition.

    Args:
        name: Parameter name
        description: Parameter description
        schema_type: JSON Schema type (string, integer, etc.)

    Returns:
        Parameter definition dict
    """
    return {
        "name": name,
        "in": "path",
        "required": True,
        "description": description,
        "schema": {"type": schema_type},
    }


def query_param(
    name: str,
    description: str = "",
    schema_type: str = "string",
    required: bool = False,
    default: Any = None,
    enum: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a query parameter definition.

    Args:
        name: Parameter name
        description: Parameter description
        schema_type: JSON Schema type
        required: Whether parameter is required
        default: Default value
        enum: List of allowed values

    Returns:
        Parameter definition dict
    """
    param: Dict[str, Any] = {
        "name": name,
        "in": "query",
        "description": description,
        "schema": {"type": schema_type},
    }

    if required:
        param["required"] = True

    if default is not None:
        param["schema"]["default"] = default

    if enum:
        param["schema"]["enum"] = enum

    return param


def json_body(
    schema: Dict[str, Any],
    description: str = "",
    required: bool = True,
) -> Dict[str, Any]:
    """Create a JSON request body definition.

    Args:
        schema: JSON Schema for the body
        description: Body description
        required: Whether body is required

    Returns:
        Request body definition dict
    """
    return {
        "description": description,
        "required": required,
        "content": {
            "application/json": {
                "schema": schema,
            }
        },
    }


def ok_response(
    description: str = "Success",
    schema: Optional[Dict[str, Any]] = None,
    status_code: str = "200",
) -> Dict[str, Dict[str, Any]]:
    """Create an OK response definition.

    Args:
        description: Response description
        schema: Optional JSON Schema for response
        status_code: HTTP status code (default: "200")

    Returns:
        Response definition dict keyed by status code
    """
    return {
        status_code: {
            "description": description,
            "content": {
                "application/json": {
                    "schema": schema or {"type": "object"},
                }
            },
        }
    }


def error_response(status_code: str, description: str) -> Dict[str, Dict[str, Any]]:
    """Create an error response definition.

    Args:
        status_code: HTTP status code as string
        description: Error description

    Returns:
        Response definition dict keyed by status code
    """
    return {
        status_code: {
            "description": description,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "error": {"type": "string"},
                            "details": {"type": "object"},
                        },
                    },
                }
            },
        }
    }


__all__ = [
    "OpenAPIEndpoint",
    "api_endpoint",
    "get_registered_endpoints",
    "get_registered_endpoints_dict",
    "clear_registry",
    "register_endpoint",
    "path_param",
    "query_param",
    "json_body",
    "ok_response",
    "error_response",
]
