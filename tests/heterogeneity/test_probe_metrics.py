from __future__ import annotations

from aragora.heterogeneity.probe import (
    ACCEPTANCE_GATES,
    PanelistClassification,
    PromptProbeResult,
    compute_metrics,
    decide_verdict,
    wilson_interval,
)


PANEL = ("a", "b", "c", "d", "e", "f")


def _result(prompt_id: str, prompt_class: str, verdicts: list[str]) -> PromptProbeResult:
    return PromptProbeResult(
        prompt_id=prompt_id,
        prompt_class=prompt_class,
        seeded_error="seed" if prompt_class != "clean_neutral" else None,
        classifications=tuple(
            PanelistClassification(agent=agent, verdict=verdict)  # type: ignore[arg-type]
            for agent, verdict in zip(PANEL, verdicts)
        ),
    )


def test_wilson_interval_bounds_proportion() -> None:
    low, high = wilson_interval(8, 10)
    assert 0 <= low < 0.8 < high <= 1


def test_correlated_priming_perfect_pilot_can_clear_ci_gate() -> None:
    _, high = wilson_interval(0, 6)
    assert high < ACCEPTANCE_GATES["catastrophic_correlation_rate_ci_high_max"]


def test_compute_metrics_counts_seeded_and_correlation_classes() -> None:
    results = [
        _result("s1", "single_seeded_error", ["flagged_correctly"] * 4 + ["missed"] * 2),
        _result("m1", "multi_seeded_error", ["flagged_correctly"] * 6),
        _result("r1", "red_team_paraphrase", ["flagged_correctly"] * 3 + ["missed"] * 3),
        _result("c1", "correlated_priming", ["flagged_correctly"] * 2 + ["missed"] * 4),
        _result("n1", "clean_neutral", ["missed"] * 5 + ["flagged_wrongly"]),
        _result("z1", "null_negative", ["missed"] * 6),
    ]
    metrics = compute_metrics(results, n_panelists=6)
    assert metrics["independent_flag_successes"] == 13
    assert metrics["independent_flag_trials"] == 18
    assert metrics["independent_flag_rate"] == 13 / 18
    assert metrics["partial_multi_seeded_successes"] == 0
    assert metrics["partial_multi_seeded_trials"] == 6
    assert metrics["catastrophic_correlation_failures"] == 1
    assert metrics["catastrophic_correlation_trials"] == 1
    assert metrics["catastrophic_correlation_rate"] == 1.0
    assert metrics["clean_neutral_false_positives"] == 1
    assert metrics["clean_neutral_false_positive_trials"] == 6
    assert metrics["false_positive_rate_on_clean_neutral"] == 1 / 6
    assert metrics["null_negative_false_positives"] == 0
    assert metrics["null_negative_false_positive_trials"] == 6
    assert metrics["false_positive_rate_on_null_negative"] == 0


def test_compute_metrics_tracks_partial_multi_seeded_separately() -> None:
    results = [
        _result(
            "m1",
            "multi_seeded_error",
            ["partial_multi_seeded"] * 4 + ["flagged_correctly", "missed"],
        ),
        _result("s1", "single_seeded_error", ["partial_multi_seeded"] + ["missed"] * 5),
    ]
    metrics = compute_metrics(results, n_panelists=6)
    assert metrics["independent_flag_successes"] == 1
    assert metrics["independent_flag_trials"] == 12
    assert metrics["independent_flag_rate"] == 1 / 12
    assert metrics["partial_multi_seeded_successes"] == 4
    assert metrics["partial_multi_seeded_trials"] == 6


def test_decide_verdict_reports_insufficient_prompt_classes() -> None:
    results = [
        _result("s1", "single_seeded_error", ["flagged_correctly"] * 6),
    ]
    metrics = compute_metrics(results, n_panelists=6)
    verdict, rationale = decide_verdict(metrics, results)
    assert verdict == "insufficient_pilot"
    assert "below minimum prompts per class" in rationale


def test_decide_verdict_reports_metric_failures_after_minimums() -> None:
    results = []
    for cls in (
        "clean_neutral",
        "single_seeded_error",
        "multi_seeded_error",
        "correlated_priming",
        "red_team_paraphrase",
        "null_negative",
    ):
        for index in range(2):
            verdicts = ["missed"] * 6
            if cls in {"single_seeded_error", "multi_seeded_error", "red_team_paraphrase"}:
                verdicts = ["flagged_correctly"] + ["missed"] * 5
            results.append(_result(f"{cls}-{index}", cls, verdicts))
    metrics = compute_metrics(results, n_panelists=6)
    verdict, rationale = decide_verdict(metrics, results)
    assert verdict == "fail"
    assert "independent_flag_rate" in rationale
