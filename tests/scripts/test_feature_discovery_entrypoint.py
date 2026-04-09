from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DISCOVERY = REPO_ROOT / "docs" / "FEATURE_DISCOVERY.md"
OPENAPI_SPEC = REPO_ROOT / "docs" / "api" / "openapi.json"


def _current_snapshot() -> tuple[int, int, int, int]:
    module_count = sum(1 for _ in (REPO_ROOT / "aragora").rglob("*.py"))
    test_file_count = sum(1 for _ in (REPO_ROOT / "tests").rglob("*.py"))

    openapi = json.loads(OPENAPI_SPEC.read_text(encoding="utf-8"))
    paths = openapi.get("paths", {})
    operations = sum(
        1
        for methods in paths.values()
        for method, definition in methods.items()
        if isinstance(definition, dict)
        and method.lower() in {"get", "post", "put", "patch", "delete", "head", "options"}
    )
    return module_count, test_file_count, operations, len(paths)


def test_root_feature_discovery_is_a_truthful_entrypoint() -> None:
    content = FEATURE_DISCOVERY.read_text(encoding="utf-8")
    module_count, test_file_count, operations, path_count = _current_snapshot()

    assert "Compatibility entrypoint for older links" in content
    assert "[status/FEATURE_DISCOVERY.md](status/FEATURE_DISCOVERY.md)" in content
    assert "Verified on " in content
    assert "against the checked-out repo" in content
    assert f"- `aragora/` Python modules: `{module_count:,}`" in content
    assert f"- `tests/` Python test files: `{test_file_count:,}`" in content
    assert (
        f"- `docs/api/openapi.json` API surface: `{operations:,}` operations across `{path_count:,}` paths"
        in content
    )

    stale_claims = [
        "- `aragora/` Python modules: `3,897`",
        "- `tests/` Python test files: `5,272`",
        "- `docs/api/openapi.json` API surface: `3,206` operations across `2,724` paths",
        "Debate spectating includes live SSE on `/api/v1/spectate/stream`",
    ]
    for claim in stale_claims:
        assert claim not in content


def test_root_feature_discovery_preserves_legacy_section_anchors() -> None:
    content = FEATURE_DISCOVERY.read_text(encoding="utf-8")

    expected_sections = [
        "## 1. Core Debate Features",
        "## 2. Agent System",
        "## 3. Memory & Learning",
        "## 4. Knowledge Management",
        "## 5. Enterprise Features",
        "## 6. Integrations & Connectors",
        "## 7. Observability & Monitoring",
        "## 8. Developer Tools",
        "## 9. Self-Improvement / Nomic Loop",
    ]
    for section in expected_sections:
        assert section in content

    expected_links = [
        "(status/FEATURE_DISCOVERY.md#1-core-debate-features)",
        "(status/FEATURE_DISCOVERY.md#2-agent-system)",
        "(status/FEATURE_DISCOVERY.md#3-memory--learning)",
        "(status/FEATURE_DISCOVERY.md#4-knowledge-management)",
        "(status/FEATURE_DISCOVERY.md#5-enterprise-features)",
        "(status/FEATURE_DISCOVERY.md#6-integrations--connectors)",
        "(status/FEATURE_DISCOVERY.md#7-observability--monitoring)",
        "(status/FEATURE_DISCOVERY.md#8-developer-tools)",
        "(status/FEATURE_DISCOVERY.md#9-self-improvement--nomic-loop)",
    ]
    for link in expected_links:
        assert link in content
