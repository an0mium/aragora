"""
OpenAPI Specification Parser.

Parses OpenAPI 3.x specs and extracts endpoints, schemas, and metadata
for SDK code generation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class Parameter:
    """API endpoint parameter."""

    name: str
    location: Literal["path", "query", "header", "cookie"]
    required: bool
    schema_type: str
    description: str = ""
    default: Any = None


@dataclass
class Schema:
    """API schema definition."""

    name: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class Endpoint:
    """API endpoint definition."""

    path: str
    method: str
    operation_id: str
    summary: str
    description: str
    tags: list[str]
    parameters: list[Parameter]
    request_body: dict[str, Any] | None
    responses: dict[str, Any]

    @property
    def namespace(self) -> str:
        """Extract namespace from path or tags."""
        if self.tags:
            return self.tags[0].lower().replace(" ", "_").replace("-", "_")
        # Extract from path: /api/v1/debates/{id} -> debates
        parts = self.path.strip("/").split("/")
        for i, part in enumerate(parts):
            if part in ("api", "v1", "v2"):
                continue
            if not part.startswith("{"):
                return part.replace("-", "_")
        return "misc"

    @property
    def method_name(self) -> str:
        """Generate Python method name from operation_id or path."""
        if self.operation_id:
            # Convert camelCase to snake_case
            name = re.sub(r"(?<!^)(?=[A-Z])", "_", self.operation_id).lower()
            return name.replace("-", "_")
        # Generate from path and method
        parts = self.path.strip("/").split("/")
        name_parts = [p for p in parts if not p.startswith("{") and p not in ("api", "v1", "v2")]
        if self.method == "GET":
            prefix = "get" if "{" in self.path else "list"
        elif self.method == "POST":
            prefix = "create"
        elif self.method == "PUT":
            prefix = "update"
        elif self.method == "PATCH":
            prefix = "patch"
        elif self.method == "DELETE":
            prefix = "delete"
        else:
            prefix = self.method.lower()
        return f"{prefix}_{'_'.join(name_parts)}"


class OpenAPIParser:
    """Parser for OpenAPI 3.x specifications."""

    def __init__(self, spec_path: str | Path):
        self.spec_path = Path(spec_path)
        self._spec: dict[str, Any] = {}
        self._endpoints: list[Endpoint] = []
        self._schemas: dict[str, Schema] = {}

    def parse(self) -> None:
        """Parse the OpenAPI specification."""
        with open(self.spec_path) as f:
            if self.spec_path.suffix == ".json":
                self._spec = json.load(f)
            else:
                import yaml

                self._spec = yaml.safe_load(f)

        self._parse_schemas()
        self._parse_endpoints()

    def _parse_schemas(self) -> None:
        """Parse component schemas."""
        schemas = self._spec.get("components", {}).get("schemas", {})
        for name, schema_def in schemas.items():
            self._schemas[name] = Schema(
                name=name,
                type=schema_def.get("type", "object"),
                properties=schema_def.get("properties", {}),
                required=schema_def.get("required", []),
                description=schema_def.get("description", ""),
            )

    def _parse_endpoints(self) -> None:
        """Parse path endpoints."""
        for path, path_item in self._spec.get("paths", {}).items():
            for method in ("get", "post", "put", "patch", "delete"):
                if method not in path_item:
                    continue

                operation = path_item[method]
                parameters = self._parse_parameters(
                    path_item.get("parameters", []) + operation.get("parameters", [])
                )

                endpoint = Endpoint(
                    path=path,
                    method=method.upper(),
                    operation_id=operation.get("operationId", ""),
                    summary=operation.get("summary", ""),
                    description=operation.get("description", ""),
                    tags=operation.get("tags", []),
                    parameters=parameters,
                    request_body=operation.get("requestBody"),
                    responses=operation.get("responses", {}),
                )
                self._endpoints.append(endpoint)

    def _parse_parameters(self, params: list[dict]) -> list[Parameter]:
        """Parse endpoint parameters."""
        result = []
        for p in params:
            schema = p.get("schema", {})
            result.append(
                Parameter(
                    name=p.get("name", ""),
                    location=p.get("in", "query"),
                    required=p.get("required", False),
                    schema_type=schema.get("type", "string"),
                    description=p.get("description", ""),
                    default=schema.get("default"),
                )
            )
        return result

    @property
    def endpoints(self) -> list[Endpoint]:
        """Get all parsed endpoints."""
        return self._endpoints

    @property
    def schemas(self) -> dict[str, Schema]:
        """Get all parsed schemas."""
        return self._schemas

    def get_endpoints_by_namespace(self) -> dict[str, list[Endpoint]]:
        """Group endpoints by namespace."""
        by_namespace: dict[str, list[Endpoint]] = {}
        for endpoint in self._endpoints:
            ns = endpoint.namespace
            if ns not in by_namespace:
                by_namespace[ns] = []
            by_namespace[ns].append(endpoint)
        return by_namespace

    @property
    def info(self) -> dict[str, Any]:
        """Get API info from spec."""
        return self._spec.get("info", {})
