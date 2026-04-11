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
        assert candidate.mission_id
        assert candidate.stage_id
        assert candidate.assertion_ids
        assert candidate.evidence_expectations
        assert candidate.gate_expectations["draft_ready"]["verdict"] == "pass"
        assert set(candidate.mission_context_policies) == {"worker", "validator"}
        assert 0.0 <= candidate.success_estimate <= 1.0


def test_generate_candidates_scope_roadmap_items_by_signal() -> None:
    config = StrategicIssueBridgeConfig(
        max_issues=6,
        heuristic_only=True,
        enable_scanner=False,
    )
    bridge = StrategicIssueBridge(repo_root=FIXTURE_ROOT, config=config)
    candidates = bridge.generate_candidates()
    by_code = {
        candidate.roadmap_refs[0]: candidate for candidate in candidates if candidate.roadmap_refs
    }

    assert "RS-01" in by_code
    assert "RS-02" in by_code
    assert "RS-03" in by_code
    assert "RS-04" in by_code
    assert "BC-04" in by_code
    assert "TW-07" in by_code

    assert by_code["RS-01"].file_scope == [
        "aragora/swarm/worker_launcher.py",
        "aragora/swarm/tranche_integrate.py",
        "aragora/swarm/boss_validation.py",
    ]
    assert by_code["RS-02"].file_scope == [
        "scripts/run_dogfood_benchmark.py",
        "scripts/dogfood_score.py",
        "aragora/swarm/worker_launcher.py",
    ]
    assert by_code["RS-03"].file_scope == [
        "github/workflows/benchmarks.yml",
        "scripts/check_benchmark_regression.py",
        "scripts/dogfood_score.py",
    ]
    assert by_code["RS-04"].file_scope == [
        "aragora/swarm/worker_contract.py",
        "aragora/swarm/worker_launcher.py",
        "aragora/swarm/runner_registry.py",
    ]
    assert by_code["BC-04"].file_scope == [
        "aragora/swarm/spec.py",
        "aragora/swarm/boss_validation.py",
        "aragora/swarm/prompt_refiner.py",
    ]
    assert by_code["TW-07"].file_scope == [
        "aragora/swarm/spec.py",
        "aragora/cli/commands/spec.py",
        "aragora/prompt_engine/spec_builder.py",
    ]

    assert (
        by_code["RS-03"].validation_command
        == "python3 -m pytest tests/scripts/test_run_dogfood_benchmark.py "
        "tests/scripts/test_phase0b_role_benchmark.py -q"
    )
    assert (
        by_code["BC-04"].validation_command
        == "python3 -m pytest tests/swarm/test_spec.py tests/swarm/test_boss_validation.py -q"
    )
    assert by_code["RS-02"].acceptance_criteria[1].startswith("Keep the diff focused to:")
    assert (
        len({tuple(by_code[code].file_scope) for code in ("RS-01", "RS-02", "RS-03", "RS-04")}) == 4
    )


def test_candidate_formats_boss_ready_mission_body() -> None:
    bridge = StrategicIssueBridge(repo_root=FIXTURE_ROOT)
    candidate = bridge.generate_candidates()[0]

    body = candidate.boss_ready_issue_body()

    assert "### Mission" in body
    assert candidate.mission_id in body
    assert candidate.stage_id in body
    assert "### Gate Expectations" in body
    assert "### Context Policies" in body
    assert "<!-- fingerprint:" in body


def test_generate_candidates_can_filter_by_theme() -> None:
    config = StrategicIssueBridgeConfig(
        max_issues=6,
        heuristic_only=True,
        enable_scanner=False,
        categories=["BC"],
    )
    bridge = StrategicIssueBridge(repo_root=FIXTURE_ROOT, config=config)

    candidates = bridge.generate_candidates()

    assert candidates
    assert all(candidate.metadata["theme"] == "BC" for candidate in candidates)


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
