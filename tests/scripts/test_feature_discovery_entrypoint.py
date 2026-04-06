from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DISCOVERY = REPO_ROOT / "docs" / "FEATURE_DISCOVERY.md"


def test_root_feature_discovery_is_a_truthful_entrypoint() -> None:
    content = FEATURE_DISCOVERY.read_text(encoding="utf-8")

    assert "Compatibility entrypoint for older links" in content
    assert "[status/FEATURE_DISCOVERY.md](status/FEATURE_DISCOVERY.md)" in content
    assert "Verified on April 6, 2026 against the checked-out repo" in content

    stale_claims = [
        "3,000+ Python modules",
        "153,000+ tests",
        "3,000+ API operations across 2,600+ paths",
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
