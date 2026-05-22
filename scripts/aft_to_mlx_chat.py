#!/usr/bin/env python3
"""Convert AFT corpus JSONL to MLX/HF chat-format training data.

Reads the output of `scripts/aft_extract_training_data.py` and emits one
chat-formatted training example per row, suitable for `mlx_lm.lora` (the
"chat" format) and for HuggingFace `trl` SFT-style training.

Output schema (one JSON object per line):
    {
      "messages": [
        {"role": "system", "content": <operator policy preamble>},
        {"role": "user",   "content": <pr_number, title_redacted, tier_hint, rationale_seeds>},
        {"role": "assistant", "content": '{"label": "...", "confidence": 1.0}'}
      ]
    }

Privacy invariant
-----------------

The conversion never reads diffs or comment bodies. It uses only the
low-information features the extractor emits (the same features the harness
sees at evaluation time). Training and evaluation see the same feature set
by construction.

The assistant output is always *exactly* the JSON the harness expects, so
the trained model emits parseable output by default. We use confidence 1.0
in training because the historical decision is final; calibration is
learned at the *next* fine-tuning round, when we re-extract with
held-back decisions and measure the model's own confidence calibration
against new ground truth.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

LOG = logging.getLogger("aft.to_mlx_chat")

SYSTEM_PROMPT = (
    "You are a per-operator PR-triage advocate. You see only low-information "
    "metadata about one pull request (no diff, no comment bodies). Reply with "
    "exactly one JSON object on a single line: "
    '{"label": "merged_fast|closed_no_merge|open_aged", "confidence": <float 0..1>}. '
    "Be honest about uncertainty: low confidence is acceptable and useful."
)

CLASSES = ("merged_fast", "closed_no_merge", "open_aged")


def _user_content(row: dict) -> str:
    seeds = row.get("rationale_seeds") or {}
    title = row.get("title_redacted") or row.get("title") or ""
    return (
        f"pr_number: {row.get('pr_number')}\n"
        f"title_redacted: {title}\n"
        f"tier_hint: {row.get('tier_hint', 'unknown')}\n"
        f"rationale_seeds: {json.dumps(seeds, sort_keys=True)}"
    )


def _assistant_content(row: dict) -> str | None:
    label = row.get("label") or row.get("decision")
    if label not in CLASSES:
        return None
    return json.dumps({"label": label, "confidence": 1.0}, sort_keys=True)


def convert_row(row: dict) -> dict | None:
    assistant = _assistant_content(row)
    if assistant is None:
        return None
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_content(row)},
            {"role": "assistant", "content": assistant},
        ]
    }


def convert_file(in_path: Path, out_path: Path) -> tuple[int, int]:
    written = 0
    skipped = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            converted = convert_row(row)
            if converted is None:
                skipped += 1
                continue
            dst.write(json.dumps(converted, sort_keys=True))
            dst.write("\n")
            written += 1
    return written, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert AFT corpus JSONL to MLX/HF chat-format JSONL"
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        required=True,
        help="Input AFT corpus JSONL (from aft_extract_training_data.py)",
    )
    parser.add_argument(
        "--out", dest="out_path", type=Path, required=True, help="Output chat-format JSONL"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.in_path.exists():
        LOG.error("Input not found: %s", args.in_path)
        return 2

    written, skipped = convert_file(args.in_path, args.out_path)
    LOG.info("Wrote %d examples to %s (%d skipped)", written, args.out_path, skipped)
    if written == 0:
        LOG.warning(
            "No examples written; check that the input has `label` or `decision` "
            "fields with one of: %s",
            CLASSES,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
