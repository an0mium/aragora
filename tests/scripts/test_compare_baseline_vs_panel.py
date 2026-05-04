from __future__ import annotations

import json
from pathlib import Path

import pytest

from aragora.heterogeneity.receipt import (
    HETEROGENEITY_RECEIPT_SCHEMA_VERSION,
    compute_receipt_id,
)
from scripts.compare_baseline_vs_panel import (
    ComparisonError,
    build_comparison_receipt,
    load_probe_receipt,
    main,
)


PANEL_AGENTS = (
    "claude-haiku-4-5",
    "gpt-4.1",
    "gemini-3.1-pro",
    "mistral-large",
    "grok-4",
    "openrouter/qwen",
)


def _breakdown(agents: tuple[str, ...] = PANEL_AGENTS) -> list[dict[str, object]]:
    return _breakdown_with_metadata(agents)


def _breakdown_with_metadata(
    agents: tuple[str, ...] = PANEL_AGENTS,
    *,
    metadata_by_agent: dict[str, dict[str, object]] | None = None,
    dispatch_failed_agents: set[str] | None = None,
) -> list[dict[str, object]]:
    metadata_by_agent = metadata_by_agent or {}
    dispatch_failed_agents = dispatch_failed_agents or set()
    return [
        {
            "prompt_class": prompt_class,
            "prompt_id": prompt_id,
            "panelist_classifications": [
                {
                    "agent": agent,
                    "verdict": "dispatch_failed"
                    if agent in dispatch_failed_agents
                    else "flagged_correctly",
                    "rationale": "",
                    **metadata_by_agent.get(agent, {}),
                }
                for agent in agents
            ],
        }
        for prompt_class, prompt_id in (
            ("single_seeded_error", "single"),
            ("multi_seeded_error", "multi"),
            ("red_team_paraphrase", "red-team"),
        )
    ]


def _receipt(
    *,
    ci: tuple[float, float],
    rate: float,
    successes: int,
    trials: int = 18,
    n_per_class: dict[str, int] | None = None,
    agents: tuple[str, ...] = PANEL_AGENTS,
    null_negative_fpr: float = 1 / 6,
    metadata_by_agent: dict[str, dict[str, object]] | None = None,
    dispatch_failed_agents: set[str] | None = None,
    metrics_n_per_class: bool = True,
) -> dict[str, object]:
    counts = {
        "clean_neutral": 1,
        "correlated_priming": 0,
        "multi_seeded_error": 1,
        "null_negative": 1,
        "red_team_paraphrase": 1,
        "single_seeded_error": 1,
    }
    if n_per_class:
        counts.update(n_per_class)
    receipt: dict[str, object] = {
        "schema_version": HETEROGENEITY_RECEIPT_SCHEMA_VERSION,
        "produced_at": "2026-05-04T00:00:00Z",
        "run_id": "synthetic",
        "judge_model": "synthetic-judge",
        "n_panelists": len(agents),
        "n_prompts": sum(counts.values()),
        "n_per_class": counts,
        "panel_models": list(agents),
        "per_prompt_breakdown": _breakdown_with_metadata(
            agents,
            metadata_by_agent=metadata_by_agent,
            dispatch_failed_agents=dispatch_failed_agents,
        ),
        "pilot_token_spend_usd_estimate": {"actual_usd": 0.0, "estimate_usd": 0.0},
        "scope_caveats": ["synthetic test fixture"],
        "verdict": "synthetic",
        "verdict_rationale": "synthetic",
        "metrics": {
            "n_panelists": len(agents),
            "n_prompts": sum(counts.values()),
            "independent_flag_successes": successes,
            "independent_flag_trials": trials,
            "independent_flag_rate": rate,
            "independent_flag_rate_ci_95_wilson": list(ci),
            "catastrophic_correlation_failures": 0,
            "catastrophic_correlation_trials": 0,
            "catastrophic_correlation_rate": 0.0,
            "catastrophic_correlation_rate_ci_95_wilson": [0.0, 1.0],
            "clean_neutral_false_positives": 0,
            "clean_neutral_false_positive_trials": 6,
            "false_positive_rate_on_clean_neutral": 0.0,
            "null_negative_false_positives": round(null_negative_fpr * 6),
            "null_negative_false_positive_trials": 6,
            "false_positive_rate_on_null_negative": null_negative_fpr,
        },
    }
    if metrics_n_per_class:
        metrics = receipt["metrics"]
        assert isinstance(metrics, dict)
        metrics["n_per_class"] = counts
    receipt["receipt_id"] = compute_receipt_id(receipt)
    return receipt


def _write(path: Path, receipt: dict[str, object]) -> Path:
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_composition_match_required() -> None:
    baseline = _receipt(ci=(0.0, 0.2), rate=0.0, successes=0)
    panel = _receipt(
        ci=(0.3, 0.7),
        rate=0.5,
        successes=9,
        n_per_class={"red_team_paraphrase": 0},
    )

    comparison = build_comparison_receipt(
        baseline,
        panel,
        baseline_receipt_path="baseline.json",
        panel_receipt_path="panel.json",
        produced_at="2026-05-04T00:00:00Z",
    )

    assert comparison["composition_match"] is False
    assert comparison["verdict"] == "INSUFFICIENT"
    assert comparison["verdict_reason"] == "composition_mismatch"


def test_ci_separation_go_when_provider_rule_passes() -> None:
    baseline = _receipt(ci=(0.0, 0.2), rate=0.0, successes=0)
    panel = _receipt(ci=(0.31, 0.7), rate=0.5, successes=9)

    comparison = build_comparison_receipt(
        baseline,
        panel,
        baseline_receipt_path="baseline.json",
        panel_receipt_path="panel.json",
        produced_at="2026-05-04T00:00:00Z",
    )

    assert comparison["verdict"] == "GO"
    provider_rule = comparison["comparison"]["provider_rule"]
    assert provider_rule["satisfied"] is True
    assert provider_rule["provider_distribution"]["anthropic"] == 3
    assert provider_rule["provider_distribution"]["openai"] == 3


def test_ci_no_go_when_panel_ci_within_baseline_ci() -> None:
    baseline = _receipt(ci=(0.0, 0.3), rate=0.0, successes=0)
    panel = _receipt(ci=(0.05, 0.2), rate=0.1, successes=2)

    comparison = build_comparison_receipt(
        baseline,
        panel,
        baseline_receipt_path="baseline.json",
        panel_receipt_path="panel.json",
        produced_at="2026-05-04T00:00:00Z",
    )

    assert comparison["verdict"] == "NO-GO"
    assert comparison["verdict_reason"] == "panel_ci_within_or_below_baseline_ci"


def test_ci_overlap_is_insufficient() -> None:
    baseline = _receipt(ci=(0.1, 0.5), rate=0.3, successes=5)
    panel = _receipt(ci=(0.4, 0.8), rate=0.6, successes=11)

    comparison = build_comparison_receipt(
        baseline,
        panel,
        baseline_receipt_path="baseline.json",
        panel_receipt_path="panel.json",
        produced_at="2026-05-04T00:00:00Z",
    )

    assert comparison["verdict"] == "INSUFFICIENT"
    assert str(comparison["verdict_reason"]).startswith("ci_overlap")


def test_insufficient_n_is_insufficient() -> None:
    baseline = _receipt(ci=(0.0, 0.2), rate=0.0, successes=0, trials=12)
    panel = _receipt(ci=(0.31, 0.7), rate=0.5, successes=9, trials=18)

    comparison = build_comparison_receipt(
        baseline,
        panel,
        baseline_receipt_path="baseline.json",
        panel_receipt_path="panel.json",
        produced_at="2026-05-04T00:00:00Z",
    )

    assert comparison["verdict"] == "INSUFFICIENT"
    assert str(comparison["verdict_reason"]).startswith("insufficient_n:")


def test_successful_seeded_n_uses_dispatch_failures_not_attempted_only() -> None:
    baseline = _receipt(ci=(0.0, 0.2), rate=0.0, successes=0, trials=18)
    panel = _receipt(
        ci=(0.31, 0.7),
        rate=0.5,
        successes=9,
        trials=18,
        dispatch_failed_agents={"gpt-4.1"},
    )

    comparison = build_comparison_receipt(
        baseline,
        panel,
        baseline_receipt_path="baseline.json",
        panel_receipt_path="panel.json",
        produced_at="2026-05-04T00:00:00Z",
    )

    assert comparison["verdict"] == "INSUFFICIENT"
    assert "panel_successful=15" in str(comparison["verdict_reason"])


def test_top_level_n_per_class_is_used_when_metrics_omits_it() -> None:
    baseline = _receipt(
        ci=(0.0, 0.2),
        rate=0.0,
        successes=0,
        metrics_n_per_class=False,
    )
    panel = _receipt(
        ci=(0.31, 0.7),
        rate=0.5,
        successes=9,
        metrics_n_per_class=False,
    )

    comparison = build_comparison_receipt(
        baseline,
        panel,
        baseline_receipt_path="baseline.json",
        panel_receipt_path="panel.json",
        produced_at="2026-05-04T00:00:00Z",
    )

    assert comparison["composition_match"] is True
    assert comparison["comparison"]["composition"]["baseline_seeded_class_counts"] == {
        "multi_seeded_error": 1,
        "red_team_paraphrase": 1,
        "single_seeded_error": 1,
    }
    assert comparison["baseline_metrics"]["n_per_class"]["single_seeded_error"] == 1


def test_openrouter_fallback_counts_as_openrouter_transport_not_direct_provider() -> None:
    agents = (
        "claude-haiku-4-5",
        "openrouter:openai/gpt-5.5",
        "gemini-3.1-pro",
        "mistral-large",
        "grok-4",
        "openrouter/qwen",
    )
    baseline = _receipt(ci=(0.0, 0.2), rate=0.0, successes=0, agents=agents)
    panel = _receipt(
        ci=(0.31, 0.7),
        rate=0.5,
        successes=9,
        agents=agents,
        metadata_by_agent={
            "openrouter:openai/gpt-5.5": {
                "requested_provider": "openai",
                "requested_model": "gpt-5.5",
                "transport_provider": "openrouter",
                "actual_model_id": "openai/gpt-5.5",
                "fallback_used": True,
                "fallback_reason": "direct_openai_insufficient_quota",
            }
        },
    )

    comparison = build_comparison_receipt(
        baseline,
        panel,
        baseline_receipt_path="baseline.json",
        panel_receipt_path="panel.json",
        produced_at="2026-05-04T00:00:00Z",
    )

    provider_rule = comparison["comparison"]["provider_rule"]
    summary = comparison["comparison"]["provider_summary"]
    assert provider_rule["provider_distribution"]["openrouter"] == 6
    assert "openai" not in provider_rule["provider_distribution"]
    assert summary["fallback_augmented"] is True
    assert comparison["fallback_augmented"] is True
    assert summary["fallback_successes"] == 3
    assert summary["model_family_distribution"]["openai"] == 3


def test_openrouter_only_panel_is_insufficient() -> None:
    agents = tuple(f"openrouter:model-{index}" for index in range(6))
    baseline = _receipt(ci=(0.0, 0.2), rate=0.0, successes=0, agents=agents)
    panel = _receipt(ci=(0.31, 0.7), rate=0.5, successes=9, agents=agents)

    comparison = build_comparison_receipt(
        baseline,
        panel,
        baseline_receipt_path="baseline.json",
        panel_receipt_path="panel.json",
        produced_at="2026-05-04T00:00:00Z",
    )

    assert comparison["verdict"] == "INSUFFICIENT"
    assert "openrouter_only_panel" in str(comparison["verdict_reason"])


def test_openrouter_dominance_above_half_is_insufficient() -> None:
    agents = (
        "openrouter:model-a",
        "openrouter:model-b",
        "openrouter:model-c",
        "openrouter:model-d",
        "claude-haiku-4-5",
        "grok-4",
    )
    baseline = _receipt(ci=(0.0, 0.2), rate=0.0, successes=0, agents=agents)
    panel = _receipt(ci=(0.31, 0.7), rate=0.5, successes=9, agents=agents)

    comparison = build_comparison_receipt(
        baseline,
        panel,
        baseline_receipt_path="baseline.json",
        panel_receipt_path="panel.json",
        produced_at="2026-05-04T00:00:00Z",
    )

    assert comparison["verdict"] == "INSUFFICIENT"
    assert "provider_dominance:openrouter:0.667" in str(comparison["verdict_reason"])


def test_baseline_saturation_is_insufficient_with_data() -> None:
    baseline = _receipt(ci=(0.9, 1.0), rate=1.0, successes=18)
    panel = _receipt(ci=(0.95, 1.0), rate=1.0, successes=18)

    comparison = build_comparison_receipt(
        baseline,
        panel,
        baseline_receipt_path="baseline.json",
        panel_receipt_path="panel.json",
        produced_at="2026-05-04T00:00:00Z",
    )

    assert comparison["verdict"] == "INSUFFICIENT-WITH-DATA"
    assert comparison["verdict_reason"] == "baseline_saturation"


def test_false_positive_flags_are_false_when_rates_are_below_gates() -> None:
    baseline = _receipt(ci=(0.0, 0.2), rate=0.0, successes=0, null_negative_fpr=0.0)
    panel = _receipt(ci=(0.31, 0.7), rate=0.5, successes=9, null_negative_fpr=0.0)

    comparison = build_comparison_receipt(
        baseline,
        panel,
        baseline_receipt_path="baseline.json",
        panel_receipt_path="panel.json",
        produced_at="2026-05-04T00:00:00Z",
    )

    flags = comparison["comparison"]["false_positive_flags"]
    assert flags["baseline_clean_neutral_exceeds_gate"] is False
    assert flags["baseline_null_negative_exceeds_gate"] is False
    assert flags["panel_clean_neutral_exceeds_baseline"] is False
    assert flags["panel_clean_neutral_exceeds_gate"] is False
    assert flags["panel_null_negative_exceeds_baseline"] is False
    assert flags["panel_null_negative_exceeds_gate"] is False


def test_baseline_null_negative_fpr_gate_warning_is_explicit() -> None:
    baseline = _receipt(
        ci=(0.0, 0.2),
        rate=0.0,
        successes=0,
        null_negative_fpr=0.3333333333333333,
    )
    panel = _receipt(ci=(0.31, 0.7), rate=0.5, successes=9, null_negative_fpr=0.0)

    comparison = build_comparison_receipt(
        baseline,
        panel,
        baseline_receipt_path="baseline.json",
        panel_receipt_path="panel.json",
        produced_at="2026-05-04T00:00:00Z",
    )

    flags = comparison["comparison"]["false_positive_flags"]
    assert flags["baseline_null_negative_exceeds_gate"] is True
    assert flags["baseline_clean_neutral_exceeds_gate"] is False
    assert flags["panel_null_negative_exceeds_gate"] is False


def test_schema_validation_rejects_non_v1_receipt(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "bad.json",
        {
            "schema_version": "not-the-probe-schema",
            "metrics": {"n_per_class": {}},
        },
    )

    with pytest.raises(ComparisonError, match="expected schema_version"):
        load_probe_receipt(path)


def test_output_receipt_id_is_deterministic() -> None:
    baseline = _receipt(ci=(0.0, 0.2), rate=0.0, successes=0)
    panel = _receipt(ci=(0.31, 0.7), rate=0.5, successes=9)

    first = build_comparison_receipt(
        baseline,
        panel,
        baseline_receipt_path="baseline.json",
        panel_receipt_path="panel.json",
        produced_at="2026-05-04T00:00:00Z",
    )
    second = build_comparison_receipt(
        baseline,
        panel,
        baseline_receipt_path="baseline.json",
        panel_receipt_path="panel.json",
        produced_at="2026-05-04T00:00:00Z",
    )

    assert first["receipt_id"] == second["receipt_id"]
    assert first == second


def test_cli_writes_receipt_and_prints_json_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline_path = _write(
        tmp_path / "baseline.json", _receipt(ci=(0.0, 0.2), rate=0.0, successes=0)
    )
    panel_path = _write(tmp_path / "panel.json", _receipt(ci=(0.31, 0.7), rate=0.5, successes=9))
    output_path = tmp_path / "comparison.json"

    rc = main(
        [
            "--baseline-receipt",
            str(baseline_path),
            "--panel-receipt",
            str(panel_path),
            "--output-receipt",
            str(output_path),
            "--json",
        ]
    )

    assert rc == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["verdict"] == "GO"
    assert output_path.exists()
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["schema_version"] == "heterogeneity_comparison_receipt.v1"
