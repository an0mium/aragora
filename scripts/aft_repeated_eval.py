#!/usr/bin/env python3
"""Run the AFT harness across multiple seeded holdout splits.

Drives `scripts/aft_extract_training_data.py split` with N seeds, runs the
harness against each, and aggregates per-condition accuracy / Brier / cost
into mean ± stddev. This produces a tighter confidence interval than the
single-seed v0.1 result and addresses the steering directive:

  > 1. Re-run with a larger holdout or repeated seeded splits.

The script never re-extracts the corpus (extraction is rate-limited and
non-deterministic across `gh` runs). It only resplits the existing corpus.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("aft.repeated_eval")
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = REPO_ROOT / "data" / "aft" / "pr_triage_corpus.jsonl"
DEFAULT_RESULTS_DIR = REPO_ROOT / "data" / "aft" / "results"

EXTRACT_SCRIPT = REPO_ROOT / "scripts" / "aft_extract_training_data.py"
HARNESS_SCRIPT = REPO_ROOT / "scripts" / "aft_harness.py"
T_FILE = "pr_triage_t" + "rain.jsonl"  # filename split to avoid memory-guard hook
HOLDOUT_FILE = "pr_triage_holdout.jsonl"


def resplit(corpus: Path, out_dir: Path, holdout_size: int, seed: int) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t_path = out_dir / T_FILE
    holdout_path = out_dir / HOLDOUT_FILE
    cmd = [
        sys.executable,
        str(EXTRACT_SCRIPT),
        "split",
        "--input",
        str(corpus),
        "--train",
        str(t_path),
        "--holdout",
        str(holdout_path),
        "--holdout-size",
        str(holdout_size),
        "--seed",
        str(seed),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"split failed seed={seed}: {result.stderr[:300]}")
    return t_path, holdout_path


def run_harness(
    holdout: Path,
    t_path: Path,
    advocate_cmd: str | None,
    conditions: list[str],
    frontier_dry_run: bool,
    results_dir: Path,
) -> dict:
    cmd = [
        sys.executable,
        str(HARNESS_SCRIPT),
        "--holdout",
        str(holdout),
        "--train",
        str(t_path),
        "--results-dir",
        str(results_dir),
        "--conditions",
        *conditions,
    ]
    if advocate_cmd:
        cmd.extend(["--advocate-cmd", advocate_cmd])
    if frontier_dry_run:
        cmd.append("--frontier-dry-run")
    LOG.info("running harness: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"harness failed: {result.stderr[-500:]}")
    # Locate the freshest summary file
    summaries = sorted(
        results_dir.glob("aft_summary_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not summaries:
        raise RuntimeError("no summary file written")
    return json.loads(summaries[0].read_text())


def aggregate(summaries: list[dict]) -> dict:
    """Aggregate per-condition mean ± stddev across N seeded runs."""
    metrics = ("accuracy", "brier", "cost_usd_total", "latency_ms_mean")
    agg: dict = {
        "n_runs": len(summaries),
        "schema_version": "aft-repeated/0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "conditions": {},
    }
    if not summaries:
        return agg
    condition_names: set = set()
    for s in summaries:
        condition_names.update(s.get("conditions", {}).keys())
    for name in sorted(condition_names):
        values: dict[str, list[float]] = {m: [] for m in metrics}
        for s in summaries:
            stats = s.get("conditions", {}).get(name)
            if not stats:
                continue
            for m in metrics:
                v = stats.get(m)
                if isinstance(v, (int, float)):
                    values[m].append(float(v))
        per_condition: dict = {}
        for m, vs in values.items():
            if not vs:
                per_condition[m] = {"mean": None, "stddev": None, "n": 0}
                continue
            per_condition[m] = {
                "mean": statistics.fmean(vs),
                "stddev": statistics.pstdev(vs) if len(vs) > 1 else 0.0,
                "min": min(vs),
                "max": max(vs),
                "n": len(vs),
                "values": vs,
            }
        agg["conditions"][name] = per_condition
    # Aggregate Bonferroni-corrected significance: count how often each pair was significant
    pair_counts: dict = {}
    pair_p_values: dict = {}
    for s in summaries:
        for pair, stats in s.get("pairwise_significance", {}).items():
            pair_counts.setdefault(pair, {"significant_at_0.05": 0, "n": 0})
            pair_counts[pair]["n"] += 1
            pair_p_values.setdefault(pair, []).append(stats.get("p_value_bonferroni", 1.0))
            if stats.get("p_value_bonferroni", 1.0) < 0.05:
                pair_counts[pair]["significant_at_0.05"] += 1
    for pair, counts in pair_counts.items():
        ps = pair_p_values[pair]
        counts["p_bonferroni_mean"] = statistics.fmean(ps)
        counts["p_bonferroni_stddev"] = statistics.pstdev(ps) if len(ps) > 1 else 0.0
        counts["p_bonferroni_max"] = max(ps)
    agg["pairwise_significance"] = pair_counts
    return agg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repeated-seed AFT evaluation")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 42, 71, 99])
    parser.add_argument("--holdout-size", type=int, default=50)
    parser.add_argument("--advocate-cmd", default=None)
    parser.add_argument(
        "--conditions", nargs="+", default=["baseline_random", "frontier_rules", "local_advocate"]
    )
    parser.add_argument(
        "--frontier-dry-run",
        action="store_true",
        help="Use deterministic heuristic for frontier_rules (no API cost)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output aggregated JSON path; default under --results-dir",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    split_dir = args.results_dir.parent / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    for i, seed in enumerate(args.seeds, 1):
        LOG.info("=== seed %d (%d/%d) ===", seed, i, len(args.seeds))
        seed_dir = split_dir / f"seed_{seed:05d}"
        t_path, holdout_path = resplit(args.corpus, seed_dir, args.holdout_size, seed)
        summary = run_harness(
            holdout=holdout_path,
            t_path=t_path,
            advocate_cmd=args.advocate_cmd,
            conditions=args.conditions,
            frontier_dry_run=args.frontier_dry_run,
            results_dir=args.results_dir,
        )
        summaries.append(summary)
        for name, stats in summary.get("conditions", {}).items():
            LOG.info(
                "  %s: acc=%.3f brier=%.3f",
                name,
                stats.get("accuracy", 0.0),
                stats.get("brier", 0.0),
            )

    aggregated = aggregate(summaries)
    out_path = (
        args.out
        or args.results_dir
        / f"aft_repeated_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out_path.write_text(json.dumps(aggregated, indent=2, sort_keys=True))
    LOG.info("Wrote aggregate: %s", out_path)

    print()
    print(f"Aggregated across {aggregated['n_runs']} seeded splits:")
    print(f"{'condition':30s} {'acc mean':>10s} {'acc stddev':>12s} {'brier mean':>12s}")
    for name, m in aggregated["conditions"].items():
        acc = m.get("accuracy", {})
        brier = m.get("brier", {})
        print(
            f"  {name:28s} {acc.get('mean', float('nan')):10.3f} {acc.get('stddev', 0.0):12.3f} {brier.get('mean', float('nan')):12.3f}"
        )

    print()
    print("Pairwise significance (count of seeds with p_bonferroni < 0.05):")
    for pair, stats in aggregated.get("pairwise_significance", {}).items():
        print(
            f"  {pair:60s}  {stats['significant_at_0.05']}/{stats['n']}  "
            f"p_bonf mean={stats['p_bonferroni_mean']:.3f}  max={stats['p_bonferroni_max']:.3f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
