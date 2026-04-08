"""Tests for decision-package assembly and markdown export."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from aragora.server.handlers.debates.decision_package import (
    DecisionPackageHandler,
    _build_markdown,
)


def _make_handler(result: dict) -> DecisionPackageHandler:
    storage = MagicMock()
    storage.get_debate.return_value = {
        "status": "completed",
        "question": "Should provider routing be visible in the package?",
        "messages": [],
        "result": result,
    }
    return DecisionPackageHandler({"storage": storage})


@patch("aragora.storage.receipt_store.get_receipt_store")
def test_assemble_package_preserves_provider_routing_metadata(mock_get_receipt_store):
    mock_get_receipt_store.return_value = MagicMock(get_by_gauntlet=MagicMock(return_value=None))
    handler = _make_handler(
        {
            "confidence": 0.92,
            "consensus_reached": True,
            "participants": ["claude", "gpt"],
            "metadata": {
                "provider_names": ["anthropic", "openai"],
                "provider_hints": ["claude-sonnet-4", "gpt-4o"],
                "provider_routing": {
                    "routing_applied": True,
                    "routing_strategy": "provider_router_selection",
                    "routed_agent_names": ["claude", "gpt"],
                    "provider_matches": {"claude": "anthropic", "gpt": "openai"},
                    "provider_hint_scores": {"anthropic": 0.91, "openai": 0.73},
                },
            },
        }
    )

    package, err = handler._assemble_package("debate-123")

    assert err is None
    assert package is not None
    assert package["provider_names"] == ["anthropic", "openai"]
    assert package["provider_hints"] == ["claude-sonnet-4", "gpt-4o"]
    assert package["provider_routing"] == {
        "routing_applied": True,
        "routing_strategy": "provider_router_selection",
        "routed_agent_names": ["claude", "gpt"],
        "provider_matches": {"claude": "anthropic", "gpt": "openai"},
        "provider_hint_scores": {"anthropic": 0.91, "openai": 0.73},
    }


def test_build_markdown_includes_provider_routing_section():
    markdown = _build_markdown(
        {
            "debate_id": "debate-123",
            "question": "Should provider routing be visible in the package?",
            "verdict": "APPROVED",
            "confidence": 0.92,
            "consensus_reached": True,
            "status": "completed",
            "provider_names": ["anthropic", "openai"],
            "provider_routing": {
                "routing_applied": True,
                "routing_strategy": "provider_router_selection",
                "routed_agent_names": ["claude", "gpt"],
                "provider_matches": {"claude": "anthropic", "gpt": "openai"},
                "provider_hint_scores": {"anthropic": 0.91, "openai": 0.73},
            },
            "export_formats": ["json"],
            "assembled_at": "2026-04-05T17:00:00Z",
        }
    )

    assert "## Provider Routing" in markdown
    assert "- **Providers:** anthropic, openai" in markdown
    assert "- **Strategy:** provider_router_selection" in markdown
    assert "- **Routed Agents:** claude, gpt" in markdown
    assert "  - claude: anthropic" in markdown
    assert "  - gpt: openai" in markdown
    assert "  - anthropic: 0.91" in markdown
    assert "  - openai: 0.73" in markdown
