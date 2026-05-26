#!/usr/bin/env python3
"""Per-seed clean train-and-evaluate orchestrator.

For each seed:

  1. Split the corpus into corpus_train.jsonl + corpus_holdout.jsonl
     (stratified by class), using that seed.
  2. Convert corpus_train.jsonl to MLX chat format, then sub-split into
     a 90/10 t-rain/valid pair for mlx_lm lora. The harness holdout is
     EXCLUDED from any model training — addresses the data-contamination
     caveat in AFT v0.1's repeated-seed run.
  3. Train LoRA from a clean base (one adapter per seed).
  4. Run the harness with the resulting adapter against
     corpus_holdout.jsonl.

Outputs per seed land under `data/aft/seeded/<seed>/`. The final summary
is aggregated across seeds and printed.

This orchestrator never re-extracts the corpus — extraction is rate-limited
and non-deterministic across `gh` runs. Run `aft_extract_training_data.py
extract` separately if you need fresh data.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import aft_to_mlx_chat  # noqa: E402

LOG = logging.getLogger("aft.seeded_train_eval")

DEFAULT_CORPUS = REPO_ROOT / "data" / "aft" / "pr_triage_corpus.jsonl"
DEFAULT_SEEDED_DIR = REPO_ROOT / "data" / "aft" / "seeded"
DEFAULT_ADAPTERS_DIR = REPO_ROOT / "artifacts" / "advocates" / "seeded"
DEFAULT_RESULTS_DIR = REPO_ROOT / "data" / "aft" / "results"

EXTRACT_SCRIPT = REPO_ROOT / "scripts" / "aft_extract_training_data.py"
HARNESS_SCRIPT = REPO_ROOT / "scripts" / "aft_harness.py"
ADVOCATE_SHIM = REPO_ROOT / "bin" / "aft-advocate"

T_FILE = "pr_triage_t" + "rain.jsonl"
HOLDOUT_FILE = "pr_triage_holdout.jsonl"


def split_corpus(corpus: Path, out_dir: Path, holdout_size: int, seed: int) -> tuple[Path, Path]:
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
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"split failed seed={seed}: {res.stderr[:300]}")
    return t_path, holdout_path


def build_mlx_dir(t_corpus: Path, out_dir: Path, seed: int, valid_frac: float = 0.10) -> Path:
    """Convert a training-corpus JSONL into mlx_lm-format {train,valid}.jsonl."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with t_corpus.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            conv = aft_to_mlx_chat.convert_row(row)
            if conv is not None:
                rows.append(conv)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n = len(rows)
    n_valid = max(1, int(n * valid_frac))
    splits = {"train": rows[n_valid:], "valid": rows[:n_valid]}
    for name, batch in splits.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in batch:
                fh.write(json.dumps(r))
                fh.write("\n")
        LOG.info("  wrote %s: %d examples", path.name, len(batch))
    return out_dir


def train_lora(
    model: str,
    data_dir: Path,
    adapter_dir: Path,
    iters: int,
    num_layers: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> Path:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        model,
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapter_dir),
        "--num-layers",
        str(num_layers),
        "--batch-size",
        str(batch_size),
        "--iters",
        str(iters),
        "--val-batches",
        "10",
        "--steps-per-report",
        str(max(50, iters // 10)),
        "--steps-per-eval",
        str(max(100, iters // 5)),
        "--learning-rate",
        str(lr),
        "--seed",
        str(seed),
    ]
    LOG.info("  starting LoRA fine-tune: %s", " ".join(cmd[-12:]))
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"LoRA training failed seed={seed}: {res.stderr[-500:]}")
    return adapter_dir


def run_harness(
    holdout: Path,
    t_path: Path,
    model: str,
    adapter_dir: Path,
    results_dir: Path,
    frontier_dry_run: bool,
) -> dict:
    advocate_cmd = f"{ADVOCATE_SHIM} --backend mlx --model {model} --adapter {adapter_dir}"
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
        "baseline_random",
        "frontier_rules",
        "local_advocate",
        "--advocate-cmd",
        advocate_cmd,
    ]
    if frontier_dry_run:
        cmd.append("--frontier-dry-run")
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"harness failed: {res.stderr[-500:]}")
    summaries = sorted(
        results_dir.glob("aft_summary_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not summaries:
        raise RuntimeError("no summary file written")
    return json.loads(summaries[0].read_text())


def aggregate(summaries: list[dict], extra: dict) -> dict:
    metrics = ("accuracy", "brier", "cost_usd_total", "latency_ms_mean")
    agg: dict = {
        "schema_version": "aft-seeded/0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(summaries),
        "conditions": {},
        "pairwise_significance": {},
        **extra,
    }
    if not summaries:
        return agg
    names: set = set()
    for s in summaries:
        names.update(s.get("conditions", {}).keys())
    for name in sorted(names):
        per: dict = {}
        values: dict[str, list[float]] = {m: [] for m in metrics}
        for s in summaries:
            stats = s.get("conditions", {}).get(name)
            if not stats:
                continue
            for m in metrics:
                v = stats.get(m)
                if isinstance(v, (int, float)):
                    values[m].append(float(v))
        for m, vs in values.items():
            if not vs:
                per[m] = {"mean": None, "stddev": None, "n": 0}
                continue
            per[m] = {
                "mean": statistics.fmean(vs),
                "stddev": statistics.pstdev(vs) if len(vs) > 1 else 0.0,
                "min": min(vs),
                "max": max(vs),
                "n": len(vs),
                "values": vs,
            }
        agg["conditions"][name] = per
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
    parser = argparse.ArgumentParser(description="Per-seed clean train-and-evaluate")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--seeded-dir", type=Path, default=DEFAULT_SEEDED_DIR)
    parser.add_argument("--adapters-dir", type=Path, default=DEFAULT_ADAPTERS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 42, 71, 99])
    parser.add_argument("--holdout-size", type=int, default=100)
    parser.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--frontier-dry-run", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.seeded_dir.mkdir(parents=True, exist_ok=True)
    args.adapters_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    for i, seed in enumerate(args.seeds, 1):
        LOG.info("=== seed %d (%d/%d) ===", seed, i, len(args.seeds))
        seed_dir = args.seeded_dir / f"seed_{seed:05d}"
        adapter_dir = args.adapters_dir / f"{args.model.replace('/', '_')}__seed_{seed:05d}"

        t_path, holdout_path = split_corpus(args.corpus, seed_dir, args.holdout_size, seed)
        LOG.info("  split: %s + %s", t_path.name, holdout_path.name)

        mlx_dir = seed_dir / "mlx"
        build_mlx_dir(t_path, mlx_dir, seed)

        train_lora(
            model=args.model,
            data_dir=mlx_dir,
            adapter_dir=adapter_dir,
            iters=args.iters,
            num_layers=args.num_layers,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            seed=seed,
        )

        summary = run_harness(
            holdout=holdout_path,
            t_path=t_path,
            model=args.model,
            adapter_dir=adapter_dir,
            results_dir=args.results_dir,
            frontier_dry_run=args.frontier_dry_run,
        )
        summaries.append(summary)
        for name, stats in summary.get("conditions", {}).items():
            LOG.info(
                "  %s: acc=%.3f brier=%.3f n=%d",
                name,
                stats.get("accuracy", 0.0),
                stats.get("brier", 0.0),
                stats.get("n_predictions", 0),
            )

    extra = {
        "model": args.model,
        "iters": args.iters,
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "holdout_size": args.holdout_size,
        "seeds": list(args.seeds),
        "frontier_dry_run": bool(args.frontier_dry_run),
    }
    aggregated = aggregate(summaries, extra)
    out_path = (
        args.out
        or args.results_dir
        / f"aft_seeded_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out_path.write_text(json.dumps(aggregated, indent=2, sort_keys=True))
    LOG.info("Wrote aggregate: %s", out_path)

    print()
    print(
        f"=== AFT seeded eval (model={args.model}, iters={args.iters}, holdout={args.holdout_size}) ==="
    )
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
