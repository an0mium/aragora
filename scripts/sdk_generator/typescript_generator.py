"""
TypeScript SDK Generator.

Generates TypeScript namespace classes from parsed OpenAPI endpoints.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .openapi_parser import Endpoint, OpenAPIParser, Parameter


class TypeScriptGenerator:
    """Generates TypeScript SDK namespace files."""

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
        methods = []

        for endpoint in endpoints:
            method = self._generate_method(endpoint)
            if method:
                methods.append(method)

        return self._render_namespace(
            name=name,
            class_name=class_name,
            methods=methods,
            endpoints=endpoints,
        )

    def _generate_method(self, endpoint: Endpoint) -> str:
        """Generate a single method."""
        method_name = self._to_camel_case(endpoint.method_name)

        # Build parameters
        params = self._build_params(endpoint)
        param_str = ", ".join(params) if params else ""

        # Build request call
        http_method = endpoint.method
        path = self._format_path(endpoint)

        # Determine if we need query params or body
        call_args = [f'"{http_method}"', path]

        query_params = [p for p in endpoint.parameters if p.location == "query"]
        has_body = endpoint.request_body is not None

        # Build method body
        body_lines = []

        # Build query params object
        if query_params:
            body_lines.append("const params: Record<string, unknown> = {};")
            for p in query_params:
                camel_name = self._to_camel_case(p.name)
                if p.required:
                    body_lines.append(f'params["{p.name}"] = {camel_name};')
                else:
                    body_lines.append(f"if ({camel_name} !== undefined) {{")
                    body_lines.append(f'  params["{p.name}"] = {camel_name};')
                    body_lines.append("}")

        # Build request options
        options_parts = []
        if query_params:
            options_parts.append("params")
        if has_body:
            options_parts.append("body: data")

        if options_parts:
            call_args.append("{ " + ", ".join(options_parts) + " }")

        # Add the request call
        call_str = ", ".join(call_args)
        body_lines.append(f"return this.client.request({call_str});")

        body = "\n    ".join(body_lines)

        # Generate JSDoc
        docstring = endpoint.summary or f"{http_method} {endpoint.path}"
        jsdoc_lines = ["/**", f" * {docstring}"]

        # Add parameter docs
        for p in endpoint.parameters:
            camel_name = self._to_camel_case(p.name)
            desc = p.description or f"{p.location} parameter"
            jsdoc_lines.append(f" * @param {camel_name} - {desc}")

        if has_body:
            jsdoc_lines.append(" * @param data - Request body")

        jsdoc_lines.append(" */")
        jsdoc = "\n  ".join(jsdoc_lines)

        # Add data parameter if needed
        if has_body:
            if param_str:
                param_str += ", data: Record<string, unknown>"
            else:
                param_str = "data: Record<string, unknown>"

        return f"""
  {jsdoc}
  async {method_name}({param_str}): Promise<unknown> {{
    {body}
  }}"""

    def _build_params(self, endpoint: Endpoint) -> list[str]:
        """Build method parameters."""
        params = []

        # Path parameters first (required)
        for p in endpoint.parameters:
            if p.location == "path":
                type_hint = self._get_type_hint(p.schema_type)
                params.append(f"{self._to_camel_case(p.name)}: {type_hint}")

        # Query parameters (optional with defaults)
        for p in endpoint.parameters:
            if p.location == "query":
                type_hint = self._get_type_hint(p.schema_type)
                camel_name = self._to_camel_case(p.name)
                if not p.required:
                    params.append(f"{camel_name}?: {type_hint}")
                else:
                    params.append(f"{camel_name}: {type_hint}")

        return params

    def _format_path(self, endpoint: Endpoint) -> str:
        """Format path with template literal interpolation."""
        path = endpoint.path
        has_params = False

        # Replace {param} with ${camelCaseParam}
        for p in endpoint.parameters:
            if p.location == "path":
                camel_name = self._to_camel_case(p.name)
                path = path.replace(
                    f"{{{p.name}}}", f"${{encodeURIComponent(String({camel_name}))}}"
                )
                has_params = True

        if has_params:
            return f"`{path}`"
        return f'"{path}"'

    def _render_namespace(
        self,
        name: str,
        class_name: str,
        methods: list[str],
        endpoints: list[Endpoint],
    ) -> str:
        """Render the complete namespace file."""
        # Collect unique tags for docstring
        tags = set()
        for e in endpoints:
            tags.update(e.tags)

        return f"""/**
 * {class_name} Namespace API
 *
 * Auto-generated from OpenAPI specification.
 * Tags: {", ".join(sorted(tags)) or "general"}
 * Endpoints: {len(endpoints)}
 */

import type {{ AragoraClient }} from "../client";

/**
 * {class_name} API namespace.
 *
 * Provides methods for interacting with {name} endpoints.
 */
export class {class_name}API {{
  private client: AragoraClient;

  constructor(client: AragoraClient) {{
    this.client = client;
  }}
{"".join(methods)}
}}

export default {class_name}API;
"""

    def _write_namespace(self, name: str, content: str) -> None:
        """Write namespace to file."""
        filename = f"{name}.ts"
        filepath = self.output_dir / filename
        filepath.write_text(content)

    @staticmethod
    def _to_class_name(name: str) -> str:
        """Convert namespace name to class name (PascalCase)."""
        # debates -> Debates, memory_store -> MemoryStore
        return "".join(word.capitalize() for word in name.replace("-", "_").split("_"))

    @staticmethod
    def _to_camel_case(name: str) -> str:
        """Convert to camelCase."""
        # Convert snake_case or kebab-case to camelCase
        name = name.replace("-", "_")
        parts = name.split("_")
        if not parts:
            return name
        return parts[0].lower() + "".join(word.capitalize() for word in parts[1:])

    @staticmethod
    def _get_type_hint(schema_type: str) -> str:
        """Map JSON schema type to TypeScript type."""
        type_map = {
            "string": "string",
            "integer": "number",
            "number": "number",
            "boolean": "boolean",
            "array": "unknown[]",
            "object": "Record<string, unknown>",
        }
        return type_map.get(schema_type, "unknown")
