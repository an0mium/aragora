"""
Python SDK Generator.

Generates Python namespace classes from parsed OpenAPI endpoints.
"""

from __future__ import annotations

import re
from pathlib import Path
from textwrap import dedent, indent
from typing import Any

from .openapi_parser import Endpoint, OpenAPIParser, Parameter


class PythonGenerator:
    """Generates Python SDK namespace files."""

    def __init__(self, parser: OpenAPIParser, output_dir: str | Path):
        self.parser = parser
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self) -> dict[str, str]:
        """Generate all namespace files."""
        results = {}
        for namespace, endpoints in self.parser.get_endpoints_by_namespace().items():
            content = self.generate_namespace(namespace, endpoints)
            results[namespace] = content
            self._write_namespace(namespace, content)
        return results

    def generate_namespace(self, name: str, endpoints: list[Endpoint]) -> str:
        """Generate a single namespace file."""
        class_name = self._to_class_name(name)

        # Generate method strings
        sync_methods = []
        async_methods = []

        for endpoint in endpoints:
            sync_method = self._generate_method(endpoint, is_async=False)
            async_method = self._generate_method(endpoint, is_async=True)
            if sync_method:
                sync_methods.append(sync_method)
            if async_method:
                async_methods.append(async_method)

        return self._render_namespace(
            name=name,
            class_name=class_name,
            sync_methods=sync_methods,
            async_methods=async_methods,
            endpoints=endpoints,
        )

    def _generate_method(self, endpoint: Endpoint, is_async: bool) -> str:
        """Generate a single method."""
        method_name = endpoint.method_name

        # Build parameters
        params = self._build_params(endpoint)
        param_str = ", ".join(params) if params else ""
        if param_str:
            param_str = f",\n        {param_str}"

        # Build request call
        http_method = endpoint.method
        path = self._format_path(endpoint)

        # Determine if we need json, params, or data
        call_kwargs = []

        query_params = [p for p in endpoint.parameters if p.location == "query"]
        if query_params:
            call_kwargs.append("params=params")

        if endpoint.request_body:
            call_kwargs.append("json=data")

        kwargs_str = ", ".join(call_kwargs)
        if kwargs_str:
            kwargs_str = f", {kwargs_str}"

        # Build method body
        body_lines = []

        # Build query params dict
        if query_params:
            body_lines.append("params: dict[str, Any] = {}")
            for p in query_params:
                if p.required:
                    body_lines.append(f'params["{p.name}"] = {self._snake_case(p.name)}')
                else:
                    body_lines.append(f"if {self._snake_case(p.name)} is not None:")
                    body_lines.append(f'    params["{p.name}"] = {self._snake_case(p.name)}')

        # Build request body dict
        if endpoint.request_body:
            body_lines.append("data: dict[str, Any] = {}")
            # Add required body params
            body_lines.append("# TODO: Populate data from parameters")

        # Add the request call
        prefix = "await " if is_async else ""
        body_lines.append(
            f'return {prefix}self._client.request("{http_method}", {path}{kwargs_str})'
        )

        body = "\n        ".join(body_lines)

        # Generate docstring
        docstring = endpoint.summary or f"{http_method} {endpoint.path}"

        async_prefix = "async " if is_async else ""

        return f'''
    {async_prefix}def {method_name}(
        self{param_str},
    ) -> dict[str, Any]:
        """{docstring}"""
        {body}'''

    def _build_params(self, endpoint: Endpoint) -> list[str]:
        """Build method parameters."""
        params = []

        # Path parameters first (required)
        for p in endpoint.parameters:
            if p.location == "path":
                type_hint = self._get_type_hint(p.schema_type)
                params.append(f"{self._snake_case(p.name)}: {type_hint}")

        # Query parameters (optional with defaults)
        for p in endpoint.parameters:
            if p.location == "query":
                type_hint = self._get_type_hint(p.schema_type)
                if not p.required:
                    type_hint = f"{type_hint} | None"
                default = (
                    f" = {repr(p.default)}"
                    if p.default is not None
                    else " = None"
                    if not p.required
                    else ""
                )
                params.append(f"{self._snake_case(p.name)}: {type_hint}{default}")

        return params

    def _format_path(self, endpoint: Endpoint) -> str:
        """Format path with f-string interpolation."""
        path = endpoint.path
        # Replace {param} with {snake_case_param}
        for p in endpoint.parameters:
            if p.location == "path":
                snake_name = self._snake_case(p.name)
                path = path.replace(f"{{{p.name}}}", f"{{quote({snake_name}, safe='')}}")

        if "{" in path:
            return f'f"{path}"'
        return f'"{path}"'

    def _render_namespace(
        self,
        name: str,
        class_name: str,
        sync_methods: list[str],
        async_methods: list[str],
        endpoints: list[Endpoint],
    ) -> str:
        """Render the complete namespace file."""
        # Collect unique tags for docstring
        tags = set()
        for e in endpoints:
            tags.update(e.tags)

        return f'''"""
{class_name} Namespace API.

Auto-generated from OpenAPI specification.
Tags: {", ".join(sorted(tags)) or "general"}
Endpoints: {len(endpoints)}
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import quote

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class {class_name}API:
    """
    Synchronous {class_name} API.

    Auto-generated from OpenAPI specification.
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client
{"".join(sync_methods)}


class Async{class_name}API:
    """Asynchronous {class_name} API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client
{"".join(async_methods)}
'''

    def _write_namespace(self, name: str, content: str) -> None:
        """Write namespace to file."""
        filename = f"{name}.py"
        filepath = self.output_dir / filename
        filepath.write_text(content)

    @staticmethod
    def _to_class_name(name: str) -> str:
        """Convert namespace name to class name."""
        # debates -> Debates, memory_store -> MemoryStore
        return "".join(word.capitalize() for word in name.split("_"))

    @staticmethod
    def _snake_case(name: str) -> str:
        """Convert to snake_case."""
        # Convert camelCase to snake_case
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower().replace("-", "_")

    @staticmethod
    def _get_type_hint(schema_type: str) -> str:
        """Map JSON schema type to Python type hint."""
        type_map = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "list",
            "object": "dict[str, Any]",
        }
        return type_map.get(schema_type, "Any")
