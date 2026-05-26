"""Unit tests for pure functions in the AFT scripts.

These tests cover the deterministic, side-effect-free pieces of the
Advocate Feasibility Test scaffold:

- `scripts/aft_extract_training_data.py`: classify_decision, tier_hint,
  rationale_seeds (no `gh` calls — we hand-build `PRDecision` instances)
- `scripts/aft_harness.py`: mcnemar_p, brier_score, accuracy
- `scripts/aft_to_mlx_chat.py`: convert_row
- `scripts/aft_repeated_eval.py`: aggregate

No model loading, no network, no subprocess. All inputs are synthetic and
deterministic. This is the safe coverage floor under the AFT spec's
"Tier 1 additive, no live caller" governance posture.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from scripts.aft_extract_training_data import (
    PRDecision,
    classify_decision,
    rationale_seeds,
    tier_hint,
)
from scripts.aft_harness import (
    CLASSES,
    Prediction,
    accuracy,
    brier_score,
    mcnemar_p,
)
from scripts.aft_repeated_eval import aggregate
from scripts.aft_to_mlx_chat import convert_row

# bin/aft-advocate has no .py extension, so import via importlib.util's
# spec-from-file-location with an explicit SourceFileLoader.
import importlib.machinery
import importlib.util
import pathlib as _pl

_AFT_ADV_PATH = _pl.Path(__file__).resolve().parents[2] / "bin" / "aft-advocate"
_AFT_ADV_LOADER = importlib.machinery.SourceFileLoader("aft_advocate_shim", str(_AFT_ADV_PATH))
_AFT_ADV_SPEC = importlib.util.spec_from_loader("aft_advocate_shim", _AFT_ADV_LOADER)
if _AFT_ADV_SPEC is None:  # pragma: no cover - defensive
    raise ImportError(f"Could not load spec for {_AFT_ADV_PATH}")
aft_advocate = importlib.util.module_from_spec(_AFT_ADV_SPEC)  # noqa: N816
_AFT_ADV_LOADER.exec_module(aft_advocate)


def _pr(
    *,
    pr_number: int = 1,
    title: str = "feat: do thing",
    head_branch: str = "claude/feat-thing",
    files_changed: tuple[str, ...] = ("aragora/feature.py",),
    additions: int = 50,
    deletions: int = 10,
    labels: tuple[str, ...] = (),
    state: str = "open",
    created_at: str = "2026-05-01T00:00:00Z",
    closed_at: str | None = None,
    merged_at: str | None = None,
    is_merged: bool = False,
    comment_count: int = 0,
    review_count: int = 0,
) -> PRDecision:
    """Build a minimal `PRDecision` for tests."""
    return PRDecision(
        pr_number=pr_number,
        title=title,
        head_branch=head_branch,
        head_sha="abcd1234",
        files_changed_count=len(files_changed),
        additions=additions,
        deletions=deletions,
        labels=labels,
        author_login="tester",
        state=state,
        created_at=created_at,
        closed_at=closed_at,
        merged_at=merged_at,
        is_merged=is_merged,
        files_changed=files_changed,
        comment_count=comment_count,
        review_count=review_count,
    )


# ----- classify_decision -----------------------------------------------------


class TestClassifyDecision:
    NOW = datetime(2026, 5, 30, 0, 0, tzinfo=timezone.utc)

    def test_merged_within_14d_is_merged_fast(self) -> None:
        pr = _pr(
            state="closed",
            is_merged=True,
            created_at="2026-05-20T00:00:00Z",
            merged_at="2026-05-22T00:00:00Z",
        )
        assert classify_decision(pr, self.NOW) == "merged_fast"

    def test_merged_after_14d_is_skipped_empty_string(self) -> None:
        pr = _pr(
            state="closed",
            is_merged=True,
            created_at="2026-04-01T00:00:00Z",
            merged_at="2026-05-22T00:00:00Z",
        )
        assert classify_decision(pr, self.NOW) == ""

    def test_closed_unmerged_is_closed_no_merge(self) -> None:
        pr = _pr(state="closed", is_merged=False, merged_at=None)
        assert classify_decision(pr, self.NOW) == "closed_no_merge"

    def test_open_more_than_14d_is_open_aged(self) -> None:
        pr = _pr(state="open", created_at="2026-05-01T00:00:00Z")
        assert classify_decision(pr, self.NOW) == "open_aged"

    def test_open_within_14d_is_skipped(self) -> None:
        pr = _pr(state="open", created_at="2026-05-25T00:00:00Z")
        assert classify_decision(pr, self.NOW) == ""

    def test_merged_boundary_exactly_14d_is_merged_fast(self) -> None:
        # 2026-05-01 -> 2026-05-15 is 14 days, must classify as merged_fast.
        pr = _pr(
            state="closed",
            is_merged=True,
            created_at="2026-05-01T00:00:00Z",
            merged_at="2026-05-15T00:00:00Z",
        )
        assert classify_decision(pr, self.NOW) == "merged_fast"


# ----- tier_hint -------------------------------------------------------------


class TestTierHint:
    def test_workflow_path_is_tier_3_or_4(self) -> None:
        pr = _pr(files_changed=(".github/workflows/ci.yml",))
        assert tier_hint(pr) == "tier_3_or_4"

    def test_review_queue_path_is_tier_3_or_4(self) -> None:
        pr = _pr(files_changed=("aragora/cli/commands/review_queue.py",))
        assert tier_hint(pr) == "tier_3_or_4"

    def test_branch_governance_token_is_tier_3_or_4(self) -> None:
        pr = _pr(head_branch="codex/governance-fix", files_changed=("aragora/x.py",))
        assert tier_hint(pr) == "tier_3_or_4"

    def test_branch_security_token_is_tier_3_or_4(self) -> None:
        pr = _pr(head_branch="claude/security-hardening", files_changed=("aragora/x.py",))
        assert tier_hint(pr) == "tier_3_or_4"

    def test_pure_docs_change_is_tier_0(self) -> None:
        pr = _pr(files_changed=("docs/README.md", "docs/guides/foo.md"))
        assert tier_hint(pr) == "tier_0"

    def test_default_is_tier_1_or_2(self) -> None:
        pr = _pr(files_changed=("aragora/x.py",))
        assert tier_hint(pr) == "tier_1_or_2"

    def test_no_files_falls_through_to_tier_1_or_2(self) -> None:
        pr = _pr(files_changed=())
        assert tier_hint(pr) == "tier_1_or_2"


# ----- rationale_seeds (privacy invariant: low-info features only) ---------


class TestRationaleSeeds:
    def test_branch_namespace_extracted(self) -> None:
        pr = _pr(head_branch="codex/short-name")
        seeds = rationale_seeds(pr)
        assert "branch_namespace=codex" in seeds

    def test_branch_without_slash_emits_no_namespace(self) -> None:
        pr = _pr(head_branch="topbranch")
        seeds = rationale_seeds(pr)
        assert not any(s.startswith("branch_namespace=") for s in seeds)

    def test_title_token_first_match_wins(self) -> None:
        pr = _pr(title="feat(security): tighten governance")
        seeds = rationale_seeds(pr)
        title_tokens = [s for s in seeds if s.startswith("title_token=")]
        assert len(title_tokens) == 1
        # 'governance' appears first in the iteration order in the source
        assert title_tokens[0] == "title_token=governance"

    def test_label_count_emitted_when_present(self) -> None:
        pr = _pr(labels=("bug", "p1"))
        seeds = rationale_seeds(pr)
        assert "label_count=2" in seeds

    def test_label_count_omitted_when_empty(self) -> None:
        pr = _pr(labels=())
        seeds = rationale_seeds(pr)
        assert not any(s.startswith("label_count=") for s in seeds)

    def test_diff_size_always_present(self) -> None:
        pr = _pr(additions=100, deletions=50)
        seeds = rationale_seeds(pr)
        assert "diff_size=150" in seeds

    def test_no_diff_content_in_seeds(self) -> None:
        # Privacy invariant: rationale_seeds must never contain the diff itself.
        # Hand a PR with a long title token to make sure we only emit the
        # canonicalized token, not arbitrary substrings.
        pr = _pr(title="feat(do-thing): private internal company sauce details", additions=999)
        seeds = rationale_seeds(pr)
        # Joined seed string must not contain any of the private title content.
        joined = " ".join(seeds)
        assert "company sauce" not in joined
        assert "private internal" not in joined

    def test_review_count_only_emitted_when_positive(self) -> None:
        pr = _pr(review_count=0)
        seeds = rationale_seeds(pr)
        assert not any(s.startswith("has_reviews=") for s in seeds)
        pr2 = _pr(review_count=3)
        seeds2 = rationale_seeds(pr2)
        assert "has_reviews=3" in seeds2


# ----- mcnemar_p -------------------------------------------------------------


class TestMcnemarP:
    def test_no_disagreement_returns_one(self) -> None:
        # Same outcomes everywhere → no signal → p=1.
        a = [True, True, False, False]
        b = [True, True, False, False]
        assert mcnemar_p(a, b) == 1.0

    def test_one_sided_disagreement_returns_low_p_when_strong(self) -> None:
        # b is right 10 times where a is wrong; a is right 0 times where b is wrong.
        # Strongly asymmetric -> small p.
        a = [False] * 10
        b = [True] * 10
        p = mcnemar_p(a, b)
        assert p < 0.01

    def test_symmetric_disagreement_returns_one(self) -> None:
        # Equal off-diagonal counts → most-conservative case → p=1.
        a = [True, False, True, False]
        b = [False, True, False, True]
        assert mcnemar_p(a, b) == 1.0

    def test_empty_inputs_return_one(self) -> None:
        assert mcnemar_p([], []) == 1.0

    def test_p_is_in_unit_interval(self) -> None:
        a = [True, False, True, False, True]
        b = [True, True, False, False, True]
        p = mcnemar_p(a, b)
        assert 0.0 <= p <= 1.0


# ----- accuracy --------------------------------------------------------------


def _pred(n: int, label: str, conf: float = 0.7, probs: dict | None = None) -> Prediction:
    return Prediction(
        pr_number=n,
        condition="t",
        prediction=label,
        confidence=conf,
        latency_ms=1.0,
        cost_usd_estimate=0.0,
        raw_probabilities=probs or {label: conf},
    )


class TestAccuracy:
    def test_all_correct(self) -> None:
        preds = [_pred(1, "merged_fast"), _pred(2, "closed_no_merge")]
        labels = {1: "merged_fast", 2: "closed_no_merge"}
        assert accuracy(preds, labels) == 1.0

    def test_all_wrong(self) -> None:
        preds = [_pred(1, "open_aged"), _pred(2, "open_aged")]
        labels = {1: "merged_fast", 2: "closed_no_merge"}
        assert accuracy(preds, labels) == 0.0

    def test_missing_truth_skipped(self) -> None:
        preds = [_pred(1, "merged_fast"), _pred(2, "merged_fast")]
        labels = {1: "merged_fast"}  # no truth for pr 2
        assert accuracy(preds, labels) == 1.0

    def test_empty_returns_nan(self) -> None:
        from math import isnan

        assert isnan(accuracy([], {}))


# ----- brier_score -----------------------------------------------------------


class TestBrierScore:
    def test_perfect_calibration_is_zero(self) -> None:
        probs = {cls: (1.0 if cls == "merged_fast" else 0.0) for cls in CLASSES}
        preds = [_pred(1, "merged_fast", conf=1.0, probs=probs)]
        labels = {1: "merged_fast"}
        assert brier_score(preds, labels) == pytest.approx(0.0, abs=1e-9)

    def test_uniform_three_class_brier_is_known_value(self) -> None:
        # Predict (1/3, 1/3, 1/3) on truth=merged_fast.
        # Brier = (1/3-1)^2 + (1/3)^2 + (1/3)^2 = 4/9 + 1/9 + 1/9 = 6/9 = 2/3
        probs = dict.fromkeys(CLASSES, 1 / 3)
        preds = [_pred(1, "merged_fast", conf=1 / 3, probs=probs)]
        labels = {1: "merged_fast"}
        assert brier_score(preds, labels) == pytest.approx(2 / 3, abs=1e-9)

    def test_brier_skipped_when_no_truth(self) -> None:
        # Empty labels => no scored predictions => nan
        from math import isnan

        preds = [_pred(1, "merged_fast")]
        assert isnan(brier_score(preds, {}))


# ----- convert_row (MLX chat-format) ---------------------------------------


class TestConvertRow:
    def test_extractor_schema_decision_field_maps_to_label(self) -> None:
        row = {
            "pr_number": 7,
            "title": "feat: x",
            "tier_hint": "tier_1_or_2",
            "rationale_seeds": ["branch_namespace=codex", "diff_size=10"],
            "decision": "merged_fast",
        }
        out = convert_row(row)
        assert out is not None
        msgs = out["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert '"label": "merged_fast"' in msgs[2]["content"]
        # Privacy: assistant message must contain *only* the JSON tuple,
        # not the full PR title or any prose.
        assert "feat: x" not in msgs[2]["content"]

    def test_invalid_label_returns_none(self) -> None:
        row = {"pr_number": 1, "decision": "no_such_class"}
        assert convert_row(row) is None

    def test_missing_label_returns_none(self) -> None:
        assert convert_row({"pr_number": 1}) is None

    def test_label_field_also_accepted(self) -> None:
        row = {"pr_number": 9, "label": "closed_no_merge"}
        out = convert_row(row)
        assert out is not None
        assert '"label": "closed_no_merge"' in out["messages"][2]["content"]


# ----- aggregate (repeated-seed summary) -----------------------------------


class TestAggregate:
    def _summary(self, name: str, accs: list[float]) -> list[dict]:
        # Build a minimal per-seed summary covering one condition with given accuracies.
        return [
            {
                "conditions": {
                    name: {
                        "accuracy": a,
                        "brier": 1.0 - a,
                        "cost_usd_total": 0.0,
                        "latency_ms_mean": 1.0,
                    }
                },
                "pairwise_significance": {},
            }
            for a in accs
        ]

    def test_aggregate_mean_and_stddev(self) -> None:
        summaries = self._summary("local_advocate", [0.7, 0.8, 0.9])
        agg = aggregate(summaries)
        acc = agg["conditions"]["local_advocate"]["accuracy"]
        assert acc["mean"] == pytest.approx(0.8, abs=1e-9)
        assert acc["min"] == pytest.approx(0.7, abs=1e-9)
        assert acc["max"] == pytest.approx(0.9, abs=1e-9)
        # population stddev of [0.7, 0.8, 0.9] is sqrt(2/3 * 0.01) ~= 0.0816497
        assert acc["stddev"] == pytest.approx(0.0816497, abs=1e-6)
        assert acc["n"] == 3

    def test_aggregate_single_run_stddev_is_zero(self) -> None:
        summaries = self._summary("baseline_random", [0.5])
        agg = aggregate(summaries)
        acc = agg["conditions"]["baseline_random"]["accuracy"]
        assert acc["stddev"] == 0.0
        assert acc["mean"] == 0.5

    def test_aggregate_significance_counts(self) -> None:
        summaries = [
            {
                "conditions": {},
                "pairwise_significance": {
                    "a_vs_b": {"p_value_bonferroni": 0.01},
                    "a_vs_c": {"p_value_bonferroni": 0.10},
                },
            },
            {
                "conditions": {},
                "pairwise_significance": {
                    "a_vs_b": {"p_value_bonferroni": 0.06},
                    "a_vs_c": {"p_value_bonferroni": 0.50},
                },
            },
        ]
        agg = aggregate(summaries)
        a_vs_b = agg["pairwise_significance"]["a_vs_b"]
        assert a_vs_b["n"] == 2
        assert a_vs_b["significant_at_0.05"] == 1  # only the 0.01 one
        assert a_vs_b["p_bonferroni_mean"] == pytest.approx(0.035, abs=1e-9)
        a_vs_c = agg["pairwise_significance"]["a_vs_c"]
        assert a_vs_c["significant_at_0.05"] == 0

    def test_aggregate_empty_returns_skeleton(self) -> None:
        agg = aggregate([])
        assert agg["n_runs"] == 0
        assert agg["conditions"] == {}


# ----- bin/aft-advocate::_parse_model_reply --------------------------------
#
# These cover the fix landed in response to the PR #7438 Tier-2 logic
# reviewer's defer dissent: the previous keyword-scan fallback used a
# substring match over the full reply text and silently translated prose
# mentions of a class token into a positive prediction. The fix requires
# either valid JSON or an exact bare-label reply; everything else falls
# through to DEFAULT_LABEL / DEFAULT_CONFIDENCE.


class TestParseModelReply:
    def test_valid_json_round_trips(self) -> None:
        out = aft_advocate._parse_model_reply('{"label": "closed_no_merge", "confidence": 0.85}')
        assert out == {"label": "closed_no_merge", "confidence": 0.85}

    def test_last_valid_json_line_wins(self) -> None:
        # Reverse-iteration: later valid JSON overrides earlier prose
        reply = 'Some prose mentioning merged_fast.\n{"label": "open_aged", "confidence": 0.6}\n'
        out = aft_advocate._parse_model_reply(reply)
        assert out["label"] == "open_aged"

    def test_invalid_label_in_json_falls_through(self) -> None:
        # JSON label outside CLASSES doesn't count as parsed; no other
        # signal present → falls through to DEFAULT_LABEL.
        out = aft_advocate._parse_model_reply('{"label": "MERGED", "confidence": 0.9}')
        assert out["label"] == aft_advocate.DEFAULT_LABEL
        assert out["confidence"] == aft_advocate.DEFAULT_CONFIDENCE

    def test_prose_substring_does_not_spoof_prediction(self) -> None:
        # DEFECT REGRESSION: prior implementation substring-matched class
        # tokens in arbitrary prose. Now this must fall through to default.
        reply = "Looks like a merged_fast PR (small diff, has reviews)."
        out = aft_advocate._parse_model_reply(reply)
        assert out["label"] == aft_advocate.DEFAULT_LABEL
        assert out["confidence"] == aft_advocate.DEFAULT_CONFIDENCE

    def test_defect_repro_invalid_json_plus_prose_token(self) -> None:
        # Combination case from the dissent: invalid-JSON label PLUS prose
        # containing a CLASSES token. Must NOT translate prose into a
        # prediction; must fall through.
        reply = (
            "Looking at this PR: matches the pattern of a merged_fast change.\n"
            '{"label": "MERGED", "confidence": 0.7}\n'
        )
        out = aft_advocate._parse_model_reply(reply)
        assert out["label"] == aft_advocate.DEFAULT_LABEL

    def test_bare_label_reply_is_accepted_with_default_confidence(self) -> None:
        # Bare-token reply (e.g., model trained to emit just the class)
        # remains acceptable but only at DEFAULT_CONFIDENCE so a downstream
        # threshold can escalate.
        out = aft_advocate._parse_model_reply("merged_fast")
        assert out["label"] == "merged_fast"
        assert out["confidence"] == aft_advocate.DEFAULT_CONFIDENCE

    def test_bare_label_uppercase_normalized(self) -> None:
        out = aft_advocate._parse_model_reply("OPEN_AGED")
        assert out["label"] == "open_aged"
        assert out["confidence"] == aft_advocate.DEFAULT_CONFIDENCE

    def test_empty_reply_falls_through(self) -> None:
        out = aft_advocate._parse_model_reply("")
        assert out["label"] == aft_advocate.DEFAULT_LABEL
        assert out["confidence"] == aft_advocate.DEFAULT_CONFIDENCE

    def test_label_inside_token_does_not_match(self) -> None:
        # 'merged_fast_typo' must not match 'merged_fast' as a bare label.
        out = aft_advocate._parse_model_reply("merged_fast_typo")
        assert out["label"] == aft_advocate.DEFAULT_LABEL

    def test_json_with_extra_keys_still_parses(self) -> None:
        out = aft_advocate._parse_model_reply(
            '{"label": "closed_no_merge", "confidence": 0.9, "rationale": "ignored"}'
        )
        assert out["label"] == "closed_no_merge"

    def test_confidence_clamped_to_unit_interval(self) -> None:
        high = aft_advocate._parse_model_reply('{"label": "merged_fast", "confidence": 99}')
        assert high["confidence"] == 1.0
        low = aft_advocate._parse_model_reply('{"label": "merged_fast", "confidence": -3}')
        assert low["confidence"] == 0.0
