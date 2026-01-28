"""
SDK Code Generator for Aragora.

Generates Python and TypeScript SDK namespaces from OpenAPI specification.

Usage:
    python -m scripts.sdk_generator --openapi docs/api/openapi.json --output sdk/python/aragora/namespaces/

Example:
    # Generate all namespaces
    python -m scripts.sdk_generator --openapi docs/api/openapi.json --output /tmp/sdk_test

    # Generate specific namespace
    python -m scripts.sdk_generator --openapi docs/api/openapi.json --namespace debates
"""

from .openapi_parser import OpenAPIParser, Endpoint, Schema
from .python_generator import PythonGenerator

__all__ = ["OpenAPIParser", "PythonGenerator", "Endpoint", "Schema"]
