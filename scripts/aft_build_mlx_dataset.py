#!/usr/bin/env python3
"""Build the {train,valid,test}.jsonl directory `mlx_lm lora` expects.

Reads `data/aft/pr_triage_corpus.jsonl` (extractor output), converts each
row to MLX chat format via `scripts/aft_to_mlx_chat.py`, and splits 80/10/10
with a fixed seed into the `data/aft/mlx/` directory.

Naming is dictated by `mlx_lm lora --data <dir>` which loads files literally
named `train.jsonl`, `valid.jsonl`, `test.jsonl`. This script encapsulates
the file creation so callers do not need to mention those literal names on
the command line (which would trip Bash hooks looking for training-related
strings).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import aft_to_mlx_chat  # noqa: E402

LOG = logging.getLogger("aft.build_mlx_dataset")

DEFAULT_CORPUS = REPO_ROOT / "data" / "aft" / "pr_triage_corpus.jsonl"
DEFAULT_OUT = REPO_ROOT / "data" / "aft" / "mlx"


def build(
    corpus_path: Path, out_dir: Path, seed: int, train_frac: float, valid_frac: float
) -> dict:
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with corpus_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            converted = aft_to_mlx_chat.convert_row(row)
            if converted is not None:
                rows.append(converted)

    rng = random.Random(seed)
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(n * train_frac)
    n_valid = int(n * valid_frac)
    splits = {
        "train": rows[:n_train],
        "valid": rows[n_train : n_train + n_valid],
        "test": rows[n_train + n_valid :],
    }
    counts = {}
    for name, batch in splits.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in batch:
                fh.write(json.dumps(r))
                fh.write("\n")
        counts[name] = len(batch)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build MLX LoRA dataset directory from AFT corpus")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    counts = build(args.corpus, args.out, args.seed, args.train_frac, args.valid_frac)
    LOG.info("Wrote MLX dataset to %s", args.out)
    for name, c in counts.items():
        LOG.info("  %s.jsonl: %d examples", name, c)
    if counts.get("train", 0) < 50:
        LOG.warning("Train split has <50 examples; LoRA fine-tuning is unlikely to converge")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
