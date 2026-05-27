#!/usr/bin/env python3
"""Model-family bench orchestrator (PR-B scaffolding, Tier 1).

Runs a tiny synthetic task corpus through one or more named model
families and produces a Pareto cost/quality report. Reuses AFT's
statistical primitives (brier_score, accuracy, mcnemar_p) and shim
contract (JSONL stdin/stdout per family) so the bench's plumbing is
the same as the AFT v0.2 harness operators already understand.

This file only runs the **stub backend** in this PR. Live provider
runs require `--allow-live` AND a `--max-cost-usd` cost cap AND
explicit per-family API credentials in env. Those guardrails are
wired here but the live code paths are stubbed in this scaffold;
turning them on is operator-attended follow-on work, not a PR-B
landing condition.

Privacy posture (same as AFT): the bundled corpus is synthetic and
contains no PR diffs, comment bodies, or PII. Live runs may only
submit payloads that satisfy
`docs/REVIEW_AUTHORITY_PRINCIPLES.md::Payload-jurisdiction routing rule`.

Usage (stub backend):
    python3 scripts/aft_family_bench.py \\
        --families claude codex openai gemini grok deepseek qwen kimi \\
        --corpus tests/fixtures/family_bench \\
        --out data/family_bench/results

Output: a single summary JSON per run, plus per-task JSONL records,
under the chosen results dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# AFT statistical primitives — reused, not duplicated.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aft_family_bench_scoring import (  # noqa: E402
    cost_quality_table,
    pareto_frontier,
    small_n_warning,
)
from aft_harness import accuracy, mcnemar_p  # noqa: E402

LOG = logging.getLogger("aft.family_bench")
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = REPO_ROOT / "tests" / "fixtures" / "family_bench"
DEFAULT_OUT = REPO_ROOT / "data" / "family_bench" / "results"
SCHEMA_VERSION = "aft-family-bench/0.1"

# Cost estimate per family in USD per call. Order-of-magnitude; the bench
# is meant to surface 10× / 100× / 1000× differences, not 5% deltas.
COST_PER_CALL_ESTIMATE: dict[str, float] = {
    "claude": 0.030,
    "codex": 0.025,
    "openai": 0.025,
    "gemini": 0.012,
    "gemini_flash": 0.003,
    "grok": 0.020,
    "grok_4_3": 0.006,
    "mistral": 0.012,
    "deepseek": 0.0008,
    "qwen": 0.0009,
    "kimi": 0.002,
    "yi": 0.001,
    "glm": 0.0012,
    "minimax": 0.0011,
    "hermes": 0.0015,
}


# ---- Corpus loader -------------------------------------------------------


def _stable_bucket(*parts: object, modulo: int = 10) -> int:
    """Return a process-stable bucket for deterministic stub predictions."""
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big") % modulo


def load_corpus(corpus_dir: Path) -> list[dict]:
    """Load all .jsonl files under `corpus_dir` into a flat task list."""
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus dir not found: {corpus_dir}")
    tasks: list[dict] = []
    for path in sorted(corpus_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    tasks.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    LOG.warning("skipping malformed task in %s: %s", path, exc)
    return tasks


# ---- Family backends -----------------------------------------------------


def stub_predict(family: str, task: dict) -> dict:
    """Deterministic family-specific stub for plumbing tests.

    Each family stub picks a slightly different rule so the harness
    produces non-degenerate Pareto + McNemar output even without live
    providers. The stubs are NOT a real evaluation — they exist so the
    end-to-end run produces a valid summary JSON the operator can
    inspect before turning on `--allow-live`.
    """
    cat = task.get("category", "")
    gt = task.get("ground_truth", "")
    # Family priors: each family has a different prior label preference.
    family_priors: dict[str, str] = {
        "claude": "merge_recommend",
        "codex": "merge_recommend",
        "openai": "merge_recommend",
        "gemini": "request_changes",
        "gemini_flash": "merge_recommend",
        "grok": "defer",
        "grok_4_3": "request_changes",
        "mistral": "merge_recommend",
        "deepseek": "merge_recommend",
        "qwen": "request_changes",
        "kimi": "merge_recommend",
        "yi": "merge_recommend",
        "glm": "request_changes",
        "minimax": "merge_recommend",
        "hermes": "defer",
    }
    if cat == "pr_review":
        # 70% of the time return the family's prior; 30% return ground truth.
        # The "30% return ground truth" is keyed on task_id hash so it's
        # deterministic but family-varying.
        if _stable_bucket(family, task.get("task_id", "")) < 3:
            return {"label": gt, "confidence": 0.65, "stub": True}
        return {"label": family_priors.get(family, "defer"), "confidence": 0.55, "stub": True}
    if cat == "debate_critique":
        # claude/openai/deepseek often catch flaws; grok/hermes often don't
        catchers = {"claude", "codex", "openai", "deepseek", "qwen", "gemini"}
        if family in catchers and _stable_bucket(family, task["task_id"]) < 7:
            return {"label": "names_specific_flaw", "confidence": 0.75, "stub": True}
        return {"label": "generic_critique", "confidence": 0.5, "stub": True}
    if cat == "inbox_triage":
        # Stub: predict ground truth 60% of the time, family prior 40%
        if _stable_bucket(family, task["task_id"]) < 6:
            return {"label": gt, "confidence": 0.7, "stub": True}
        return {"label": "archive", "confidence": 0.45, "stub": True}
    return {"label": "unknown", "confidence": 0.0, "stub": True}


def predict_for_family(family: str, task: dict, *, allow_live: bool, max_cost_usd: float) -> dict:
    """Route to the appropriate backend.

    PR-B only wires the stub backend. The `allow_live` flag is parsed
    and propagated so the wiring is testable, but no live API call
    paths are activated in this PR — a follow-on PR will wire them
    behind the same cost-cap guardrail.
    """
    result = stub_predict(family, task)
    result["family"] = family
    result["cost_usd_estimate"] = COST_PER_CALL_ESTIMATE.get(family, 0.0)
    if allow_live:
        # PR-B placeholder: when this branch becomes a live call, it must
        # also check that running estimated cost stays below max_cost_usd.
        # For now, we just note the flag in the output so the operator
        # can verify it was honored.
        result["live_path"] = "not_yet_wired_in_pr_b"
        result["max_cost_usd"] = max_cost_usd
    return result


# ---- Per-task scoring ----------------------------------------------------


def correct_of(prediction: str, ground_truth: str) -> bool:
    return prediction == ground_truth


def family_summary(family: str, per_task_records: list[dict]) -> dict[str, float | int | str]:
    """Aggregate per-task records into a per-family summary."""
    n = len(per_task_records)
    correct = sum(1 for r in per_task_records if r["correct"])
    cost = sum(float(r.get("cost_usd_estimate", 0.0)) for r in per_task_records)
    accuracies = [1.0 if r["correct"] else 0.0 for r in per_task_records]
    confidences = [float(r.get("confidence", 0.5)) for r in per_task_records]
    return {
        "family": family,
        "n": n,
        "n_correct": correct,
        "accuracy": (correct / n) if n else float("nan"),
        "cost_usd": cost,
        "confidence_mean": statistics.fmean(confidences) if confidences else 0.0,
        # Brier shorthand for multi-class: per-call (1 - confidence-on-true-class)^2
        # is the closest approximation we can compute from prediction + confidence
        # alone (without per-class probability mass). Treat as approximate.
        "brier_approx": (
            statistics.fmean([(c - a) ** 2 for c, a in zip(confidences, accuracies)])
            if confidences
            else float("nan")
        ),
    }


def pairwise_significance(
    family_records: dict[str, list[dict]],
) -> dict[str, dict[str, float]]:
    """Paired McNemar across all family pairs, Bonferroni-corrected.

    Pairs are joined by ``task_id`` (the inner-join intersection of the
    two families' task sets). Under the current ``run_bench`` every
    family runs every task, so the intersection equals the full task
    set; the explicit join keeps the comparison correct under live or
    failure-tolerant runs where one family may emit fewer records than
    another. A previous implementation sorted each family's records and
    truncated to ``n_min``, which would silently compare different
    tasks if the per-family record sets ever diverged.
    """
    families = sorted(family_records.keys())
    pairs = [(a, b) for i, a in enumerate(families) for b in families[i + 1 :]]
    bonf = max(1, len(pairs))
    out: dict[str, dict[str, float]] = {}
    for a, b in pairs:
        # Inner-join on task_id so each pair compares the *same* task,
        # not just the same-indexed sorted position. Defensive against
        # duplicate task_ids within a single family by keeping the last
        # record (predictable; matches dict-construction semantics).
        a_by_task = {r["task_id"]: r["correct"] for r in family_records[a]}
        b_by_task = {r["task_id"]: r["correct"] for r in family_records[b]}
        common_task_ids = sorted(a_by_task.keys() & b_by_task.keys())
        a_correct = [a_by_task[t] for t in common_task_ids]
        b_correct = [b_by_task[t] for t in common_task_ids]
        p = mcnemar_p(a_correct, b_correct)
        # Disagreement count for the small-n warning
        disagreements = sum(1 for x, y in zip(a_correct, b_correct) if x != y)
        out[f"{a}__vs__{b}"] = {
            "p_value": p,
            "p_value_bonferroni": min(1.0, p * bonf),
            "n_disagreements": disagreements,
            "bonferroni_factor": bonf,
            "n_paired_tasks": len(common_task_ids),
        }
    return out


# ---- Orchestrator --------------------------------------------------------


def run_bench(
    families: list[str],
    corpus_dir: Path,
    *,
    allow_live: bool,
    max_cost_usd: float,
) -> dict[str, Any]:
    tasks = load_corpus(corpus_dir)
    if not tasks:
        raise RuntimeError(f"No tasks loaded from {corpus_dir}")
    LOG.info("Loaded %d tasks across %d families", len(tasks), len(families))

    family_records: dict[str, list[dict]] = {f: [] for f in families}
    for task in tasks:
        for family in families:
            pred = predict_for_family(
                family, task, allow_live=allow_live, max_cost_usd=max_cost_usd
            )
            record = {
                "task_id": task["task_id"],
                "category": task.get("category", ""),
                "family": family,
                "prediction": pred["label"],
                "confidence": pred["confidence"],
                "ground_truth": task.get("ground_truth", ""),
                "correct": correct_of(pred["label"], task.get("ground_truth", "")),
                "cost_usd_estimate": pred["cost_usd_estimate"],
                "stub": bool(pred.get("stub", False)),
            }
            family_records[family].append(record)

    per_family = {f: family_summary(f, recs) for f, recs in family_records.items()}
    pareto = pareto_frontier(
        [
            {
                "family": f,
                "cost_usd": float(s["cost_usd"]),
                "accuracy": float(s["accuracy"]) if s["accuracy"] == s["accuracy"] else 0.0,
                "brier": float(s["brier_approx"])
                if s["brier_approx"] == s["brier_approx"]
                else 1.0,
            }
            for f, s in per_family.items()
        ]
    )
    table = cost_quality_table(
        {
            f: {
                "family": f,
                "cost_usd": float(s["cost_usd"]),
                "accuracy": float(s["accuracy"]) if s["accuracy"] == s["accuracy"] else 0.0,
                "brier": float(s["brier_approx"])
                if s["brier_approx"] == s["brier_approx"]
                else 1.0,
            }
            for f, s in per_family.items()
        }
    )
    pair_sig = pairwise_significance(family_records)
    disagreements = {pair: stats["n_disagreements"] for pair, stats in pair_sig.items()}
    warnings = small_n_warning(disagreements)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_dir": str(corpus_dir),
        "n_tasks": len(tasks),
        "families": families,
        "allow_live": allow_live,
        "max_cost_usd": max_cost_usd,
        "stub_only_run": all(
            all(r.get("stub", False) for r in recs) for recs in family_records.values()
        ),
        "per_family": per_family,
        "pareto_frontier": pareto,
        "cost_quality_table": table,
        "pairwise_significance": pair_sig,
        "small_n_warnings": warnings,
        "per_task_records": [r for recs in family_records.values() for r in recs],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Model-family bench (stub-only in PR-B)")
    parser.add_argument(
        "--families",
        nargs="+",
        default=["claude", "openai", "gemini", "grok", "deepseek", "qwen", "kimi"],
        help="Family names to bench (must be in COST_PER_CALL_ESTIMATE)",
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help=(
            "Reserved for a future PR that wires live provider backends. "
            "In PR-B this flag is parsed for forward-compat only; passing "
            "it FAILS CLOSED with a non-zero exit instead of silently "
            "running the stub backend."
        ),
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=5.00,
        help=(
            "Cost cap reserved for live runs. Parsed for forward-compat "
            "only in PR-B; has no effect on the stub backend."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.allow_live:
        # Fail closed: PR-B does not wire any live backend. Silently running
        # the stub while accepting --allow-live would let a future caller
        # mistake a stub-only run for live evidence. The flag survives in
        # the CLI surface for forward-compat (a follow-on PR will wire the
        # actual live provider paths with credentials + cost-cap guardrail),
        # but invoking it here must refuse with a clear error.
        LOG.error(
            "--allow-live is reserved for a future PR. No live provider "
            "backend is wired in PR-B; refusing to run to avoid letting "
            "stub-only output be misread as live evidence. Re-run without "
            "--allow-live for a stub-only bench."
        )
        return 3

    unknown_families = [f for f in args.families if f not in COST_PER_CALL_ESTIMATE]
    if unknown_families:
        LOG.error(
            "Unknown families %s; add to COST_PER_CALL_ESTIMATE first.",
            unknown_families,
        )
        return 2

    summary = run_bench(
        args.families,
        args.corpus,
        allow_live=args.allow_live,
        max_cost_usd=args.max_cost_usd,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out / f"bench_summary_{ts}.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    LOG.info("Wrote summary: %s", out_path)

    print()
    print(f"Bench v{SCHEMA_VERSION} — {summary['n_tasks']} tasks, {len(args.families)} families")
    if summary["stub_only_run"]:
        print("STUB-ONLY RUN: results are plumbing-validation, not empirical evidence.")
    print()
    print(f"{'family':18s} {'acc':>6s} {'brier':>7s} {'cost_usd':>10s}")
    for f, s in summary["per_family"].items():
        acc = s["accuracy"] if s["accuracy"] == s["accuracy"] else 0.0
        brier = s["brier_approx"] if s["brier_approx"] == s["brier_approx"] else 1.0
        cost = s["cost_usd"]
        if (
            isinstance(acc, (int, float))
            and isinstance(brier, (int, float))
            and isinstance(cost, (int, float))
        ):
            print(f"  {f:16s} {acc:6.3f} {brier:7.3f} {cost:10.4f}")
    print()
    print("Pareto frontier (cost asc; ties keep all):")
    for p in summary["pareto_frontier"]:
        print(f"  {p['family']:18s} cost=${p['cost_usd']:.4f}  acc={p['accuracy']:.3f}")
    if summary["small_n_warnings"]:
        print()
        print(f"⚠ {len(summary['small_n_warnings'])} small-n warning(s):")
        for w in summary["small_n_warnings"][:5]:
            print(f"  - {w}")
        if len(summary["small_n_warnings"]) > 5:
            print(f"  (+ {len(summary['small_n_warnings']) - 5} more in summary JSON)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
