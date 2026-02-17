from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}
REQUEST_RE = re.compile(
    r"this\.client\.request\(\s*['\"](?P<method>[A-Z]+)['\"]\s*,\s*(?P<path>`[^`]+`|'[^']+'|\"[^\"]+\")"
)
DIRECT_RE = re.compile(
    r"this\.client\.(?P<method>get|post|put|delete|patch)\(\s*(?P<path>`[^`]+`|'[^']+'|\"[^\"]+\")"
)

NAMESPACE_CONTRACTS = {
    "decisions": {
        "file": "sdk/typescript/src/namespaces/decisions.ts",
        "endpoints": [
            ("get", "/api/v1/decisions"),
            ("post", "/api/v1/decisions"),
            ("get", "/api/v1/decisions/{param}"),
            ("get", "/api/v1/decisions/{param}/status"),
            ("get", "/api/v1/decisions/{param}/explain"),
        ],
    },
    "integrations": {
        "file": "sdk/typescript/src/namespaces/integrations.ts",
        "endpoints": [
            ("get", "/api/v1/integrations/teams/status"),
            ("post", "/api/v1/integrations/teams/notify"),
        ],
    },
    "webhooks": {
        "file": "sdk/typescript/src/namespaces/webhooks.ts",
        "endpoints": [
            ("get", "/api/v1/webhooks"),
            ("post", "/api/v1/webhooks"),
            ("get", "/api/v1/webhooks/{param}"),
            ("patch", "/api/v1/webhooks/{param}"),
            ("delete", "/api/v1/webhooks/{param}"),
            ("post", "/api/v1/webhooks/{param}/test"),
            ("get", "/api/v1/webhooks/events"),
            ("get", "/api/v1/webhooks/slo/status"),
            ("post", "/api/v1/webhooks/slo/test"),
            ("get", "/api/v1/webhooks/dead-letter"),
            ("get", "/api/v1/webhooks/dead-letter/{param}"),
            ("post", "/api/v1/webhooks/dead-letter/{param}/retry"),
            ("get", "/api/v1/webhooks/queue/stats"),
        ],
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize_path(path: str) -> str:
    path = path.split("?", 1)[0]
    path = re.sub(r"\$\{[^}]+\}", "{param}", path)
    path = re.sub(r"\{[^}]+\}", "{param}", path)
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return path


def _extract_sdk_endpoints(content: str) -> set[tuple[str, str]]:
    endpoints: set[tuple[str, str]] = set()

    for match in REQUEST_RE.finditer(content):
        method = match.group("method").lower()
        raw_path = match.group("path")[1:-1]
        endpoints.add((method, _normalize_path(raw_path)))

    for match in DIRECT_RE.finditer(content):
        method = match.group("method").lower()
        raw_path = match.group("path")[1:-1]
        endpoints.add((method, _normalize_path(raw_path)))

    return endpoints


@pytest.fixture(scope="module")
def openapi_endpoints() -> set[tuple[str, str]]:
    spec_path = _repo_root() / "docs/api/openapi.json"
    spec = json.loads(spec_path.read_text())
    endpoints: set[tuple[str, str]] = set()
    for path, operations in spec.get("paths", {}).items():
        for method in operations:
            method_lower = method.lower()
            if method_lower in HTTP_METHODS:
                endpoints.add((method_lower, _normalize_path(path)))
    return endpoints


@pytest.mark.parametrize("namespace", sorted(NAMESPACE_CONTRACTS.keys()))
def test_sdk_namespace_contract(namespace: str, openapi_endpoints: set[tuple[str, str]]) -> None:
    config = NAMESPACE_CONTRACTS[namespace]
    file_path = _repo_root() / config["file"]
    assert file_path.exists(), f"SDK namespace file missing: {config['file']}"

    content = file_path.read_text()
    sdk_endpoints = _extract_sdk_endpoints(content)

    expected = {(method.lower(), _normalize_path(path)) for method, path in config["endpoints"]}

    missing_in_sdk = sorted(expected - sdk_endpoints)
    assert not missing_in_sdk, f"{namespace} missing endpoints in SDK: {missing_in_sdk}"

    missing_in_spec = sorted(expected - openapi_endpoints)
    assert not missing_in_spec, f"{namespace} missing endpoints in OpenAPI spec: {missing_in_spec}"
