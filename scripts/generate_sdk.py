#!/usr/bin/env python3
"""
SDK Client Generator for Aragora API.

Generates TypeScript and Python SDK clients from the OpenAPI specification.

Usage:
    python scripts/generate_sdk.py --output sdk/ --lang typescript
    python scripts/generate_sdk.py --output sdk/ --lang python
    python scripts/generate_sdk.py --output sdk/ --lang all

Requirements:
    - openapi-python-client (for Python)
    - Node.js with openapi-typescript (for TypeScript)

    pip install openapi-python-client
    npm install -g openapi-typescript
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_openapi_spec(spec_path: str) -> dict[str, Any]:
    """Load the OpenAPI specification."""
    with open(spec_path) as f:
        return json.load(f)


def generate_typescript_types(spec_path: str, output_dir: str) -> bool:
    """Generate TypeScript types from OpenAPI spec.

    Uses openapi-typescript to generate type definitions.
    """
    output_file = os.path.join(output_dir, "typescript", "api-types.ts")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        # Try using npx if openapi-typescript is not globally installed
        result = subprocess.run(
            ["npx", "openapi-typescript", spec_path, "-o", output_file],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Warning: openapi-typescript failed: {result.stderr}")
            # Generate minimal types manually
            return generate_typescript_types_manual(spec_path, output_dir)

        print(f"Generated TypeScript types: {output_file}")
        return True
    except FileNotFoundError:
        print("openapi-typescript not found, generating types manually")
        return generate_typescript_types_manual(spec_path, output_dir)


def generate_typescript_types_manual(spec_path: str, output_dir: str) -> bool:
    """Generate TypeScript types manually from OpenAPI spec."""
    spec = load_openapi_spec(spec_path)
    output_file = os.path.join(output_dir, "typescript", "api-types.ts")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    lines = [
        "// Auto-generated TypeScript types from OpenAPI spec",
        "// Do not edit manually",
        "",
        "// API Configuration",
        "export interface ApiConfig {",
        "  baseUrl: string;",
        "  apiKey?: string;",
        "  headers?: Record<string, string>;",
        "}",
        "",
    ]

    # Generate types from schemas
    schemas = spec.get("components", {}).get("schemas", {})
    for name, schema in schemas.items():
        type_def = convert_schema_to_typescript(name, schema)
        lines.append(type_def)
        lines.append("")

    # Generate endpoint types
    lines.append("// API Endpoints")
    lines.append("export type ApiEndpoints = {")
    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            if method in ["get", "post", "put", "delete", "patch"]:
                operation_id = details.get("operationId", f"{method}_{path.replace('/', '_')}")
                lines.append(f"  '{operation_id}': {{")
                lines.append(f"    path: '{path}';")
                lines.append(f"    method: '{method.upper()}';")
                lines.append("  };")
    lines.append("};")

    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated TypeScript types (manual): {output_file}")
    return True


def convert_schema_to_typescript(name: str, schema: dict) -> str:
    """Convert an OpenAPI schema to TypeScript interface."""
    schema_type = schema.get("type", "object")

    if schema_type == "object":
        lines = [f"export interface {name} {{"]
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for prop_name, prop_schema in properties.items():
            ts_type = get_typescript_type(prop_schema)
            optional = "?" if prop_name not in required else ""
            lines.append(f"  {prop_name}{optional}: {ts_type};")

        lines.append("}")
        return "\n".join(lines)

    elif schema_type == "array":
        item_type = get_typescript_type(schema.get("items", {}))
        return f"export type {name} = {item_type}[];"

    else:
        ts_type = get_typescript_type(schema)
        return f"export type {name} = {ts_type};"


def get_typescript_type(schema: dict) -> str:
    """Convert OpenAPI type to TypeScript type."""
    if "$ref" in schema:
        return schema["$ref"].split("/")[-1]

    schema_type = schema.get("type", "any")

    if schema_type == "string":
        if "enum" in schema:
            return " | ".join(f'"{v}"' for v in schema["enum"])
        return "string"
    elif schema_type == "integer" or schema_type == "number":
        return "number"
    elif schema_type == "boolean":
        return "boolean"
    elif schema_type == "array":
        item_type = get_typescript_type(schema.get("items", {}))
        return f"{item_type}[]"
    elif schema_type == "object":
        if "additionalProperties" in schema:
            value_type = get_typescript_type(schema["additionalProperties"])
            return f"Record<string, {value_type}>"
        return "Record<string, unknown>"

    return "unknown"


def generate_typescript_client(spec_path: str, output_dir: str) -> bool:
    """Generate a TypeScript API client."""
    spec = load_openapi_spec(spec_path)
    output_file = os.path.join(output_dir, "typescript", "client.ts")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    client_code = '''// Auto-generated TypeScript client from OpenAPI spec
// Do not edit manually

import type { ApiConfig } from './api-types';

export class AragoraClient {
  private config: ApiConfig;

  constructor(config: ApiConfig) {
    this.config = config;
  }

  private async request<T>(
    method: string,
    path: string,
    options: {
      body?: unknown;
      params?: Record<string, string | number | boolean>;
      headers?: Record<string, string>;
    } = {}
  ): Promise<T> {
    const url = new URL(path, this.config.baseUrl);

    if (options.params) {
      Object.entries(options.params).forEach(([key, value]) => {
        url.searchParams.append(key, String(value));
      });
    }

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...this.config.headers,
      ...options.headers,
    };

    if (this.config.apiKey) {
      headers['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    const response = await fetch(url.toString(), {
      method,
      headers,
      body: options.body ? JSON.stringify(options.body) : undefined,
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  // Debates
  async listDebates(params?: { limit?: number; offset?: number }) {
    return this.request<{ debates: unknown[] }>('GET', '/api/debates', { params });
  }

  async getDebate(debateId: string) {
    return this.request<unknown>('GET', `/api/debates/${debateId}`);
  }

  async createDebate(body: { question: string; agents: string[]; rounds?: number }) {
    return this.request<{ debate_id: string; success: boolean }>('POST', '/api/debate', { body });
  }

  // Explainability
  async getExplanation(debateId: string, options?: {
    include_factors?: boolean;
    include_counterfactuals?: boolean;
    include_provenance?: boolean;
  }) {
    return this.request<unknown>('GET', `/api/debates/${debateId}/explainability`, { params: options });
  }

  async getFactors(debateId: string, options?: { min_contribution?: number }) {
    return this.request<unknown>('GET', `/api/debates/${debateId}/explainability/factors`, { params: options });
  }

  async getCounterfactuals(debateId: string, options?: { max_scenarios?: number }) {
    return this.request<unknown>('GET', `/api/debates/${debateId}/explainability/counterfactual`, { params: options });
  }

  async generateCounterfactual(debateId: string, body: { hypothesis: string; affected_agents?: string[] }) {
    return this.request<unknown>('POST', `/api/debates/${debateId}/explainability/counterfactual`, { body });
  }

  async getProvenance(debateId: string) {
    return this.request<unknown>('GET', `/api/debates/${debateId}/explainability/provenance`);
  }

  async getNarrative(debateId: string, options?: { format?: 'brief' | 'detailed' | 'executive_summary' }) {
    return this.request<unknown>('GET', `/api/debates/${debateId}/explainability/narrative`, { params: options });
  }

  // Workflow Templates
  async listWorkflowTemplates(params?: {
    category?: string;
    pattern?: string;
    search?: string;
    limit?: number;
    offset?: number;
  }) {
    return this.request<{ templates: unknown[]; total: number }>('GET', '/api/workflow/templates', { params });
  }

  async getWorkflowTemplate(templateId: string) {
    return this.request<unknown>('GET', `/api/workflow/templates/${templateId}`);
  }

  async getWorkflowTemplatePackage(templateId: string, includeExamples?: boolean) {
    return this.request<unknown>('GET', `/api/workflow/templates/${templateId}/package`, {
      params: { include_examples: includeExamples },
    });
  }

  async runWorkflowTemplate(templateId: string, body: {
    inputs?: Record<string, unknown>;
    config?: { timeout?: number; priority?: string; async?: boolean };
    workspace_id?: string;
  }) {
    return this.request<unknown>('POST', `/api/workflow/templates/${templateId}/run`, { body });
  }

  async listWorkflowCategories() {
    return this.request<{ categories: unknown[] }>('GET', '/api/workflow/categories');
  }

  async listWorkflowPatterns() {
    return this.request<{ patterns: unknown[] }>('GET', '/api/workflow/patterns');
  }

  // Gauntlet
  async listGauntletReceipts(params?: { verdict?: string; limit?: number; offset?: number }) {
    return this.request<{ receipts: unknown[]; total: number }>('GET', '/api/gauntlet/receipts', { params });
  }

  async getGauntletReceipt(receiptId: string) {
    return this.request<unknown>('GET', `/api/gauntlet/receipts/${receiptId}`);
  }

  async verifyGauntletReceipt(receiptId: string) {
    return this.request<{ valid: boolean; hash: string }>('GET', `/api/gauntlet/receipts/${receiptId}/verify`);
  }

  async exportGauntletReceipt(receiptId: string, format: 'json' | 'html' | 'markdown' | 'sarif') {
    return this.request<unknown>('GET', `/api/gauntlet/receipts/${receiptId}/export`, { params: { format } });
  }

  // Agents
  async listAgents() {
    return this.request<{ agents: unknown[] }>('GET', '/api/agents');
  }

  async getAgent(agentName: string) {
    return this.request<unknown>('GET', `/api/agents/${agentName}`);
  }

  // Health
  async getHealth() {
    return this.request<{ status: string }>('GET', '/api/health');
  }
}

export function createClient(config: ApiConfig): AragoraClient {
  return new AragoraClient(config);
}
'''

    with open(output_file, "w") as f:
        f.write(client_code)

    # Also create an index file
    index_file = os.path.join(output_dir, "typescript", "index.ts")
    with open(index_file, "w") as f:
        f.write("export * from './api-types';\n")
        f.write("export * from './client';\n")

    print(f"Generated TypeScript client: {output_file}")
    return True


def generate_python_client(spec_path: str, output_dir: str) -> bool:
    """Generate a Python API client."""
    spec = load_openapi_spec(spec_path)
    output_file = os.path.join(output_dir, "python", "aragora_client.py")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    client_code = '''"""
Auto-generated Python client for Aragora API.

Do not edit manually.

Usage:
    from aragora_client import AragoraClient

    client = AragoraClient(base_url="http://localhost:8080", api_key="your-key")
    debates = await client.list_debates()
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore


@dataclass
class ApiConfig:
    """Configuration for the API client."""
    base_url: str
    api_key: Optional[str] = None
    timeout: float = 30.0
    headers: Optional[Dict[str, str]] = None


class AragoraClient:
    """Async client for the Aragora API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        if httpx is None:
            raise ImportError("httpx is required for AragoraClient. Install with: pip install httpx")

        self.config = ApiConfig(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=timeout,
            headers=headers or {},
        )
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "AragoraClient":
        await self._ensure_client()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json", **self.config.headers}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=self.config.timeout,
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Any:
        client = await self._ensure_client()
        response = await client.request(method, path, params=params, json=json)
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Debates
    # =========================================================================

    async def list_debates(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List debates."""
        return await self._request("GET", "/api/debates", params={"limit": limit, "offset": offset})

    async def get_debate(self, debate_id: str) -> Dict[str, Any]:
        """Get a specific debate."""
        return await self._request("GET", f"/api/debates/{debate_id}")

    async def create_debate(
        self,
        question: str,
        agents: List[str],
        rounds: int = 3,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a new debate."""
        return await self._request(
            "POST",
            "/api/debate",
            json={"question": question, "agents": agents, "rounds": rounds, **kwargs},
        )

    # =========================================================================
    # Explainability
    # =========================================================================

    async def get_explanation(
        self,
        debate_id: str,
        include_factors: bool = True,
        include_counterfactuals: bool = True,
        include_provenance: bool = True,
    ) -> Dict[str, Any]:
        """Get full explanation for a debate decision."""
        return await self._request(
            "GET",
            f"/api/debates/{debate_id}/explainability",
            params={
                "include_factors": include_factors,
                "include_counterfactuals": include_counterfactuals,
                "include_provenance": include_provenance,
            },
        )

    async def get_factors(
        self,
        debate_id: str,
        min_contribution: Optional[float] = None,
        sort_by: str = "contribution",
    ) -> Dict[str, Any]:
        """Get contributing factors for a debate decision."""
        params = {"sort_by": sort_by}
        if min_contribution is not None:
            params["min_contribution"] = min_contribution
        return await self._request("GET", f"/api/debates/{debate_id}/explainability/factors", params=params)

    async def get_counterfactuals(
        self,
        debate_id: str,
        max_scenarios: int = 5,
        min_probability: float = 0.3,
    ) -> Dict[str, Any]:
        """Get counterfactual scenarios for a debate."""
        return await self._request(
            "GET",
            f"/api/debates/{debate_id}/explainability/counterfactual",
            params={"max_scenarios": max_scenarios, "min_probability": min_probability},
        )

    async def generate_counterfactual(
        self,
        debate_id: str,
        hypothesis: str,
        affected_agents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate a custom counterfactual scenario."""
        body = {"hypothesis": hypothesis}
        if affected_agents:
            body["affected_agents"] = affected_agents
        return await self._request("POST", f"/api/debates/{debate_id}/explainability/counterfactual", json=body)

    async def get_provenance(
        self,
        debate_id: str,
        include_timestamps: bool = True,
        include_agents: bool = True,
        include_confidence: bool = True,
    ) -> Dict[str, Any]:
        """Get decision provenance chain."""
        return await self._request(
            "GET",
            f"/api/debates/{debate_id}/explainability/provenance",
            params={
                "include_timestamps": include_timestamps,
                "include_agents": include_agents,
                "include_confidence": include_confidence,
            },
        )

    async def get_narrative(
        self,
        debate_id: str,
        format: str = "detailed",
        language: str = "en",
    ) -> Dict[str, Any]:
        """Get natural language narrative explanation."""
        return await self._request(
            "GET",
            f"/api/debates/{debate_id}/explainability/narrative",
            params={"format": format, "language": language},
        )

    # =========================================================================
    # Workflow Templates
    # =========================================================================

    async def list_workflow_templates(
        self,
        category: Optional[str] = None,
        pattern: Optional[str] = None,
        search: Optional[str] = None,
        tags: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List workflow templates."""
        params = {"limit": limit, "offset": offset}
        if category:
            params["category"] = category
        if pattern:
            params["pattern"] = pattern
        if search:
            params["search"] = search
        if tags:
            params["tags"] = tags
        return await self._request("GET", "/api/workflow/templates", params=params)

    async def get_workflow_template(self, template_id: str) -> Dict[str, Any]:
        """Get workflow template details."""
        return await self._request("GET", f"/api/workflow/templates/{template_id}")

    async def get_workflow_template_package(
        self,
        template_id: str,
        include_examples: bool = True,
    ) -> Dict[str, Any]:
        """Get full workflow template package."""
        return await self._request(
            "GET",
            f"/api/workflow/templates/{template_id}/package",
            params={"include_examples": include_examples},
        )

    async def run_workflow_template(
        self,
        template_id: str,
        inputs: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a workflow template."""
        body: Dict[str, Any] = {}
        if inputs:
            body["inputs"] = inputs
        if config:
            body["config"] = config
        if workspace_id:
            body["workspace_id"] = workspace_id
        return await self._request("POST", f"/api/workflow/templates/{template_id}/run", json=body)

    async def list_workflow_categories(self) -> Dict[str, Any]:
        """List workflow template categories."""
        return await self._request("GET", "/api/workflow/categories")

    async def list_workflow_patterns(self) -> Dict[str, Any]:
        """List workflow patterns."""
        return await self._request("GET", "/api/workflow/patterns")

    async def instantiate_pattern(
        self,
        pattern_id: str,
        name: str,
        description: str,
        category: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        agents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a template from a workflow pattern."""
        body = {"name": name, "description": description}
        if category:
            body["category"] = category
        if config:
            body["config"] = config
        if agents:
            body["agents"] = agents
        return await self._request("POST", f"/api/workflow/patterns/{pattern_id}/instantiate", json=body)

    # =========================================================================
    # Gauntlet
    # =========================================================================

    async def list_gauntlet_receipts(
        self,
        verdict: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List gauntlet receipts."""
        params = {"limit": limit, "offset": offset}
        if verdict:
            params["verdict"] = verdict
        return await self._request("GET", "/api/gauntlet/receipts", params=params)

    async def get_gauntlet_receipt(self, receipt_id: str) -> Dict[str, Any]:
        """Get a specific gauntlet receipt."""
        return await self._request("GET", f"/api/gauntlet/receipts/{receipt_id}")

    async def verify_gauntlet_receipt(self, receipt_id: str) -> Dict[str, Any]:
        """Verify receipt integrity."""
        return await self._request("GET", f"/api/gauntlet/receipts/{receipt_id}/verify")

    async def export_gauntlet_receipt(
        self,
        receipt_id: str,
        format: str = "json",
    ) -> Dict[str, Any]:
        """Export receipt in specified format."""
        return await self._request(
            "GET",
            f"/api/gauntlet/receipts/{receipt_id}/export",
            params={"format": format},
        )

    # =========================================================================
    # Agents
    # =========================================================================

    async def list_agents(self) -> Dict[str, Any]:
        """List available agents."""
        return await self._request("GET", "/api/agents")

    async def get_agent(self, agent_name: str) -> Dict[str, Any]:
        """Get agent details."""
        return await self._request("GET", f"/api/agents/{agent_name}")

    # =========================================================================
    # Health
    # =========================================================================

    async def health(self) -> Dict[str, Any]:
        """Check API health."""
        return await self._request("GET", "/api/health")


# Sync wrapper for convenience
class AragoraClientSync:
    """Synchronous wrapper for AragoraClient."""

    def __init__(self, *args: Any, **kwargs: Any):
        self._async_client = AragoraClient(*args, **kwargs)

    def _run(self, coro: Any) -> Any:
        return asyncio.get_event_loop().run_until_complete(coro)

    def close(self) -> None:
        self._run(self._async_client.close())

    def list_debates(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.list_debates(**kwargs))

    def get_debate(self, debate_id: str) -> Dict[str, Any]:
        return self._run(self._async_client.get_debate(debate_id))

    def create_debate(self, question: str, agents: List[str], **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.create_debate(question, agents, **kwargs))

    def get_explanation(self, debate_id: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.get_explanation(debate_id, **kwargs))

    def list_workflow_templates(self, **kwargs: Any) -> Dict[str, Any]:
        return self._run(self._async_client.list_workflow_templates(**kwargs))

    def health(self) -> Dict[str, Any]:
        return self._run(self._async_client.health())
'''

    with open(output_file, "w") as f:
        f.write(client_code)

    # Create __init__.py
    init_file = os.path.join(output_dir, "python", "__init__.py")
    with open(init_file, "w") as f:
        f.write('"""Aragora Python SDK."""\n\n')
        f.write("from .aragora_client import AragoraClient, AragoraClientSync, ApiConfig\n\n")
        f.write("__all__ = ['AragoraClient', 'AragoraClientSync', 'ApiConfig']\n")
        f.write("__version__ = '1.0.0'\n")

    print(f"Generated Python client: {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate SDK clients from OpenAPI spec")
    parser.add_argument(
        "--spec",
        default="docs/api/openapi.json",
        help="Path to OpenAPI spec file (default: docs/api/openapi.json)",
    )
    parser.add_argument(
        "--output",
        default="sdk",
        help="Output directory (default: sdk)",
    )
    parser.add_argument(
        "--lang",
        choices=["typescript", "python", "all"],
        default="all",
        help="Target language (default: all)",
    )

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.parent
    spec_path = script_dir / args.spec
    output_dir = script_dir / args.output

    if not spec_path.exists():
        print(f"Error: OpenAPI spec not found at {spec_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    success = True

    if args.lang in ("typescript", "all"):
        success = generate_typescript_types(str(spec_path), str(output_dir)) and success
        success = generate_typescript_client(str(spec_path), str(output_dir)) and success

    if args.lang in ("python", "all"):
        success = generate_python_client(str(spec_path), str(output_dir)) and success

    if success:
        print(f"\nSDK generation complete. Output: {output_dir}")
    else:
        print("\nSDK generation completed with warnings")
        sys.exit(1)


if __name__ == "__main__":
    main()
