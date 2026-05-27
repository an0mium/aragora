"""Unit tests for `scripts/aft_family_bench_scoring.py` + harness primitives.

Tests cover the pure, deterministic, side-effect-free pieces of the
model-family bench harness scaffold (PR-B):

- `pareto_frontier`: cost/quality non-domination logic
- `jaccard_distance`: finding-set distance for H4 redundancy check
- `cost_quality_table`: sortable per-family aggregate builder
- `small_n_warning`: under-powered McNemar pair detection
- `scripts.aft_family_bench.load_corpus`: bundled-fixture loader
- `scripts.aft_family_bench.stub_predict`: deterministic family stubs
- `scripts.aft_family_bench.family_summary`: per-family aggregation
- `scripts.aft_family_bench.correct_of`: prediction match helper

No model loading, no network, no subprocess invocations. All inputs
are synthetic. This is the Tier-1 coverage floor for the bench harness.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.aft_family_bench import (
    COST_PER_CALL_ESTIMATE,
    correct_of,
    family_summary,
    load_corpus,
    pairwise_significance,
    predict_for_family,
    run_bench,
    stub_predict,
)
from scripts.aft_family_bench_scoring import (
    cost_quality_table,
    jaccard_distance,
    pareto_frontier,
    small_n_warning,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "family_bench"


# ----- pareto_frontier ------------------------------------------------


class TestParetoFrontier:
    def test_single_point_is_on_frontier(self) -> None:
        result = pareto_frontier([{"family": "a", "cost_usd": 1.0, "accuracy": 0.5, "brier": 0.5}])
        assert len(result) == 1
        assert result[0]["family"] == "a"

    def test_dominated_point_excluded(self) -> None:
        # b dominates a: same cost, higher accuracy
        result = pareto_frontier(
            [
                {"family": "a", "cost_usd": 1.0, "accuracy": 0.5, "brier": 0.5},
                {"family": "b", "cost_usd": 1.0, "accuracy": 0.8, "brier": 0.3},
            ]
        )
        families = [p["family"] for p in result]
        assert "b" in families
        assert "a" not in families

    def test_cheap_low_quality_and_expensive_high_quality_both_on_frontier(self) -> None:
        # Classic Pareto: neither dominates the other
        result = pareto_frontier(
            [
                {"family": "cheap", "cost_usd": 0.001, "accuracy": 0.5, "brier": 0.5},
                {"family": "expensive", "cost_usd": 1.000, "accuracy": 0.9, "brier": 0.1},
            ]
        )
        families = {p["family"] for p in result}
        assert families == {"cheap", "expensive"}

    def test_exact_tie_keeps_both(self) -> None:
        # Same cost AND same accuracy → neither dominates → both on frontier
        result = pareto_frontier(
            [
                {"family": "a", "cost_usd": 1.0, "accuracy": 0.7, "brier": 0.3},
                {"family": "b", "cost_usd": 1.0, "accuracy": 0.7, "brier": 0.2},
            ]
        )
        assert len(result) == 2

    def test_empty_input_returns_empty(self) -> None:
        assert pareto_frontier([]) == []

    def test_frontier_sorted_by_cost_then_accuracy_desc(self) -> None:
        result = pareto_frontier(
            [
                {"family": "expensive_great", "cost_usd": 10.0, "accuracy": 0.95, "brier": 0.05},
                {"family": "cheap_ok", "cost_usd": 0.10, "accuracy": 0.60, "brier": 0.40},
                {"family": "mid", "cost_usd": 1.00, "accuracy": 0.80, "brier": 0.20},
            ]
        )
        # All three are non-dominated; expect cost-ascending
        costs = [p["cost_usd"] for p in result]
        assert costs == sorted(costs)


# ----- jaccard_distance -----------------------------------------------


class TestJaccardDistance:
    def test_identical_sets_distance_zero(self) -> None:
        assert jaccard_distance(["a", "b", "c"], ["a", "b", "c"]) == 0.0

    def test_disjoint_sets_distance_one(self) -> None:
        assert jaccard_distance(["a", "b"], ["c", "d"]) == 1.0

    def test_partial_overlap(self) -> None:
        # {a,b,c} ∩ {b,c,d} = {b,c}; ∪ = {a,b,c,d}; J = 1 - 2/4 = 0.5
        assert jaccard_distance(["a", "b", "c"], ["b", "c", "d"]) == 0.5

    def test_both_empty_treated_as_identical(self) -> None:
        assert jaccard_distance([], []) == 0.0

    def test_duplicates_collapsed(self) -> None:
        # Sets dedupe inputs
        assert jaccard_distance(["a", "a", "b"], ["a", "b"]) == 0.0


# ----- cost_quality_table ----------------------------------------------


class TestCostQualityTable:
    def _per_family(self) -> dict:
        return {
            "claude": {"family": "claude", "cost_usd": 0.30, "accuracy": 0.90, "brier": 0.10},
            "deepseek": {"family": "deepseek", "cost_usd": 0.01, "accuracy": 0.70, "brier": 0.25},
            "gemini": {"family": "gemini", "cost_usd": 0.12, "accuracy": 0.85, "brier": 0.15},
        }

    def test_default_sort_by_cost_per_correct_ascending(self) -> None:
        table = cost_quality_table(self._per_family())
        # cost_per_correct = cost_usd / accuracy
        # deepseek: 0.01 / 0.70 ≈ 0.0143  (cheapest)
        # gemini:   0.12 / 0.85 ≈ 0.1412
        # claude:   0.30 / 0.90 ≈ 0.3333  (most expensive)
        assert [r["family"] for r in table] == ["deepseek", "gemini", "claude"]

    def test_sort_by_accuracy_descending(self) -> None:
        table = cost_quality_table(self._per_family(), sort_by="accuracy")
        assert [r["family"] for r in table] == ["claude", "gemini", "deepseek"]

    def test_invalid_sort_key_raises(self) -> None:
        with pytest.raises(ValueError):
            cost_quality_table(self._per_family(), sort_by="not_a_column")

    def test_zero_accuracy_produces_infinite_cost_per_correct(self) -> None:
        table = cost_quality_table(
            {"broken": {"family": "broken", "cost_usd": 1.0, "accuracy": 0.0, "brier": 1.0}},
            sort_by="cost_per_correct",
        )
        assert table[0]["cost_per_correct"] == float("inf")


# ----- small_n_warning ------------------------------------------------


class TestSmallNWarning:
    def test_below_threshold_warns(self) -> None:
        warnings = small_n_warning({"a__vs__b": 5, "c__vs__d": 20})
        assert len(warnings) == 1
        assert "a__vs__b" in warnings[0]
        assert "5" in warnings[0]

    def test_above_threshold_silent(self) -> None:
        assert small_n_warning({"a__vs__b": 20, "c__vs__d": 99}, threshold=15) == []

    def test_custom_threshold_respected(self) -> None:
        warnings = small_n_warning({"a__vs__b": 12}, threshold=10)
        assert warnings == []  # 12 >= 10
        warnings = small_n_warning({"a__vs__b": 12}, threshold=20)
        assert len(warnings) == 1


# ----- load_corpus -----------------------------------------------------


class TestLoadCorpus:
    def test_loads_all_bundled_tasks(self) -> None:
        tasks = load_corpus(CORPUS_DIR)
        # Spec promises 3 PR + 3 debate + 3 inbox = 9 in this PR's scaffold
        assert len(tasks) == 9
        categories = {t["category"] for t in tasks}
        assert categories == {"pr_review", "debate_critique", "inbox_triage"}

    def test_each_task_has_required_fields(self) -> None:
        tasks = load_corpus(CORPUS_DIR)
        for t in tasks:
            assert "task_id" in t
            assert "category" in t
            assert "ground_truth" in t

    def test_missing_corpus_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_corpus(Path("/nonexistent/family_bench"))


# ----- stub_predict + predict_for_family ------------------------------


class TestStubPredict:
    def test_stub_returns_valid_shape(self) -> None:
        task = {"task_id": "pr-1", "category": "pr_review", "ground_truth": "merge_recommend"}
        result = stub_predict("claude", task)
        assert "label" in result
        assert "confidence" in result
        assert result["stub"] is True

    def test_stub_is_deterministic_per_family_task_pair(self) -> None:
        task = {"task_id": "pr-1", "category": "pr_review", "ground_truth": "merge_recommend"}
        a = stub_predict("claude", task)
        b = stub_predict("claude", task)
        assert a == b

    def test_unknown_category_returns_unknown_label(self) -> None:
        result = stub_predict("claude", {"task_id": "x", "category": "weird", "ground_truth": "y"})
        assert result["label"] == "unknown"

    def test_allow_live_flag_propagated_but_path_not_yet_wired(self) -> None:
        task = {"task_id": "pr-1", "category": "pr_review", "ground_truth": "merge_recommend"}
        result = predict_for_family("claude", task, allow_live=True, max_cost_usd=1.0)
        assert result["live_path"] == "not_yet_wired_in_pr_b"
        assert result["max_cost_usd"] == 1.0

    def test_cost_estimate_attached(self) -> None:
        task = {"task_id": "pr-1", "category": "pr_review", "ground_truth": "merge_recommend"}
        result = predict_for_family("deepseek", task, allow_live=False, max_cost_usd=1.0)
        assert result["cost_usd_estimate"] == COST_PER_CALL_ESTIMATE["deepseek"]

    def test_codex_family_is_supported(self) -> None:
        task = {"task_id": "pr-1", "category": "pr_review", "ground_truth": "merge_recommend"}
        result = predict_for_family("codex", task, allow_live=False, max_cost_usd=1.0)
        assert COST_PER_CALL_ESTIMATE["codex"] > 0
        assert result["cost_usd_estimate"] == COST_PER_CALL_ESTIMATE["codex"]

    def test_stub_predictions_are_stable_across_hash_seeds(self) -> None:
        script = """
import json
from scripts.aft_family_bench import load_corpus, stub_predict
from pathlib import Path

tasks = load_corpus(Path("tests/fixtures/family_bench"))
families = ["claude", "openai", "gemini", "grok", "deepseek", "qwen", "kimi", "codex"]
labels = [
    stub_predict(family, task)["label"]
    for family in families
    for task in tasks
]
print(json.dumps(labels, sort_keys=True))
"""
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "1"
        seed_1 = subprocess.check_output([sys.executable, "-c", script], cwd=REPO_ROOT, env=env)
        env["PYTHONHASHSEED"] = "2"
        seed_2 = subprocess.check_output([sys.executable, "-c", script], cwd=REPO_ROOT, env=env)
        assert json.loads(seed_1) == json.loads(seed_2)


# ----- family_summary --------------------------------------------------


class TestFamilySummary:
    def _records(self, correct_flags: list[bool]) -> list[dict]:
        return [
            {
                "task_id": f"t{i}",
                "correct": flag,
                "cost_usd_estimate": 0.01,
                "confidence": 0.8 if flag else 0.4,
            }
            for i, flag in enumerate(correct_flags)
        ]

    def test_all_correct_accuracy_one(self) -> None:
        s = family_summary("claude", self._records([True, True, True]))
        assert s["accuracy"] == 1.0
        assert s["n_correct"] == 3

    def test_all_wrong_accuracy_zero(self) -> None:
        s = family_summary("claude", self._records([False, False]))
        assert s["accuracy"] == 0.0

    def test_cost_sums_per_record(self) -> None:
        s = family_summary("claude", self._records([True, False, True]))
        assert s["cost_usd"] == pytest.approx(0.03)


# ----- correct_of ------------------------------------------------------


def test_correct_of_exact_match() -> None:
    assert correct_of("merge_recommend", "merge_recommend") is True


def test_correct_of_mismatch() -> None:
    assert correct_of("merge_recommend", "request_changes") is False


# ----- pairwise_significance ------------------------------------------


class TestPairwiseSignificance:
    def test_no_pairs_when_single_family(self) -> None:
        sig = pairwise_significance({"only": [{"task_id": "t1", "correct": True}]})
        assert sig == {}

    def test_pair_emitted_for_two_families(self) -> None:
        sig = pairwise_significance(
            {
                "a": [{"task_id": "t1", "correct": True}, {"task_id": "t2", "correct": False}],
                "b": [{"task_id": "t1", "correct": False}, {"task_id": "t2", "correct": True}],
            }
        )
        assert "a__vs__b" in sig
        assert sig["a__vs__b"]["bonferroni_factor"] == 1
        assert sig["a__vs__b"]["n_disagreements"] == 2
        # The new task_id-join contract: both families saw both tasks.
        assert sig["a__vs__b"]["n_paired_tasks"] == 2

    def test_inner_joins_on_task_id_not_sorted_position(self) -> None:
        """Defensive: differing per-family task sets MUST inner-join on task_id.

        A previous implementation sorted and truncated to ``n_min``, which
        would silently compare different tasks if the per-family record
        sets ever diverged (live runs with one family failing or skipped
        tasks). This test pins the inner-join behavior: only the
        intersection of task_ids is compared, never positionally aligned
        records.
        """
        # Family "a" has t1, t2. Family "b" has t2, t3. Common: {t2}.
        # Family a: t1=True, t2=False. Family b: t2=False, t3=True.
        # The only paired task is t2 (both False → no disagreement).
        sig = pairwise_significance(
            {
                "a": [
                    {"task_id": "t1", "correct": True},
                    {"task_id": "t2", "correct": False},
                ],
                "b": [
                    {"task_id": "t2", "correct": False},
                    {"task_id": "t3", "correct": True},
                ],
            }
        )
        assert sig["a__vs__b"]["n_paired_tasks"] == 1
        assert sig["a__vs__b"]["n_disagreements"] == 0

    def test_disjoint_task_sets_produce_zero_paired_tasks(self) -> None:
        """Inner-join correctness under fully disjoint task sets."""
        sig = pairwise_significance(
            {
                "a": [{"task_id": "t1", "correct": True}],
                "b": [{"task_id": "t2", "correct": True}],
            }
        )
        assert sig["a__vs__b"]["n_paired_tasks"] == 0
        assert sig["a__vs__b"]["n_disagreements"] == 0


# ----- CLI main() behavior --------------------------------------------


class TestMainFailClosedOnAllowLive:
    """PR-B contract: ``--allow-live`` must fail closed.

    Letting --allow-live silently fall back to the stub would let a
    future caller mistake stub-only output for live evidence. The flag
    survives in the CLI surface for forward-compat (a follow-on PR
    will wire the actual live provider paths), but invoking it in
    PR-B must refuse with a clear error.
    """

    def test_allow_live_returns_nonzero_exit(self) -> None:
        from scripts.aft_family_bench import main

        rc = main(
            [
                "--allow-live",
                "--corpus",
                str(CORPUS_DIR),
                "--families",
                "claude",
            ]
        )
        assert rc == 3, (
            f"--allow-live must fail closed with exit 3, got {rc}. "
            "Silently running the stub when --allow-live is requested "
            "would let stub-only output be misread as live evidence."
        )

    def test_stub_only_invocation_still_succeeds(self, tmp_path) -> None:
        from scripts.aft_family_bench import main

        out_dir = tmp_path / "bench-out"
        rc = main(
            [
                "--corpus",
                str(CORPUS_DIR),
                "--families",
                "claude",
                "deepseek",
                "--out",
                str(out_dir),
            ]
        )
        assert rc == 0
        # One summary file written
        produced = list(out_dir.glob("bench_summary_*.json"))
        assert len(produced) == 1


# ----- end-to-end smoke ------------------------------------------------


def test_run_bench_e2e_stub_only_produces_valid_summary() -> None:
    summary = run_bench(
        ["claude", "deepseek"],
        CORPUS_DIR,
        allow_live=False,
        max_cost_usd=1.0,
    )
    assert summary["n_tasks"] == 9
    assert set(summary["families"]) == {"claude", "deepseek"}
    assert summary["stub_only_run"] is True
    assert "claude" in summary["per_family"]
    assert "deepseek" in summary["per_family"]
    assert "claude__vs__deepseek" in summary["pairwise_significance"]
    # Pareto frontier non-empty
    assert len(summary["pareto_frontier"]) >= 1
