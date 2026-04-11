from __future__ import annotations

from pathlib import Path

from aragora.swarm.strategic_issue_bridge import StrategicIssueBridge, StrategicIssueBridgeConfig

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "strategic_issue_bridge"


def test_parse_roadmap_items_priority() -> None:
    bridge = StrategicIssueBridge(repo_root=FIXTURE_ROOT)
    active_text = (FIXTURE_ROOT / "docs" / "status" / "ACTIVE_EXECUTION_ISSUES.md").read_text(
        encoding="utf-8"
    )
    items, priority_order = bridge.parse_roadmap_items(active_text)

    assert priority_order[:2] == ["Reliability Substrate", "Bounded Autonomy Control Plane"]
    assert any(item.code == "RS-01" for item in items)
    rs_item = next(item for item in items if item.code == "RS-01")
    assert rs_item.epic == "Reliability Substrate"
    assert rs_item.priority_rank == 1
    assert rs_item.milestone


def test_generate_candidates_heuristic_only() -> None:
    config = StrategicIssueBridgeConfig(
        max_issues=5,
        heuristic_only=True,
        enable_scanner=False,
    )
    bridge = StrategicIssueBridge(repo_root=FIXTURE_ROOT, config=config)
    candidates = bridge.generate_candidates()

    assert len(candidates) >= 3
    assert any(candidate.title.startswith("RS-01") for candidate in candidates)
    for candidate in candidates:
        assert candidate.title
        assert candidate.description
        assert candidate.file_scope
        assert candidate.acceptance_criteria
        assert candidate.validation_command
        assert 0.0 <= candidate.success_estimate <= 1.0


def test_llm_fallback_without_keys(monkeypatch) -> None:
    for key in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        "XAI_API_KEY",
        "GROK_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    config = StrategicIssueBridgeConfig(
        max_issues=4,
        enable_llm=True,
        heuristic_only=False,
        enable_scanner=False,
    )
    bridge = StrategicIssueBridge(repo_root=FIXTURE_ROOT, config=config)
    candidates = bridge.generate_candidates()
    assert len(candidates) >= 3
