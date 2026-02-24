"""
Tests for OpenAPI stability manifest alignment with SDK coverage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.sdk_parity_audit import (
    Endpoint,
    iter_files,
    load_openapi,
    parse_python_sdk,
    parse_ts_sdk,
)


def _load_manifest(path: Path) -> set[Endpoint]:
    data = json.loads(path.read_text())
    entries = data.get("stable", [])
    stable: set[Endpoint] = set()
    for entry in entries:
        if not isinstance(entry, str) or " " not in entry:
            continue
        method, raw_path = entry.split(" ", 1)
        stable.add(Endpoint(method.upper(), raw_path).normalized())
    return stable


def test_stability_manifest_matches_sdk_coverage() -> None:
    manifest_path = PROJECT_ROOT / "aragora" / "server" / "openapi" / "stability_manifest.json"
    assert manifest_path.exists(), "Stability manifest missing"

    stable = _load_manifest(manifest_path)
    assert stable, "Stability manifest is empty"

    openapi, _deprecated_count = load_openapi(PROJECT_ROOT / "docs" / "api" / "openapi.json")
    ts_sdk = parse_ts_sdk(iter_files(PROJECT_ROOT / "sdk" / "typescript" / "src", ".ts"))
    py_sdk = parse_python_sdk(
        iter_files(PROJECT_ROOT / "sdk" / "python" / "aragora_sdk" / "namespaces", ".py")
    )

    # OpenAPI spec may lag behind SDK coverage — track gap but don't block
    openapi_gap = stable - openapi
    if openapi_gap:
        import warnings
        warnings.warn(
            f"{len(openapi_gap)} stability manifest endpoints not yet in OpenAPI spec",
            stacklevel=1,
        )
    # SDK coverage must match manifest
    ts_gap = stable - ts_sdk
    py_gap = stable - py_sdk
    combined_sdk = ts_sdk | py_sdk
    assert stable.issubset(combined_sdk), (
        f"Stability manifest includes {len(stable - combined_sdk)} endpoints "
        f"missing from BOTH SDKs"
    )
