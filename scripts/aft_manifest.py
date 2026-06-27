#!/usr/bin/env python3
"""Build a reproducibility manifest for an AFT v0.1 evaluation.

Records:
  - Dataset hash (SHA256 of the corpus JSONL)
  - Split seed(s) used
  - Model name and revision (HuggingFace repo id; revision pinned if known)
  - Adapter path(s) and SHA256 of adapter files
  - Training args (rank/num-layers, lr, batch, iters, seed)
  - Eval command (the exact harness invocation)
  - Summary artifact hashes (per-run summaries)
  - Tool versions (mlx-lm, transformers, python)

The manifest is intentionally simple JSON; it is the single source of truth
that lets a reviewer reproduce the evaluation. Adapter artifacts themselves
remain gitignored under `artifacts/advocates/`; the manifest just records
their hashes so a re-trained adapter can be compared bit-for-bit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("aft.manifest")
REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def adapter_hashes(adapter_dir: Path) -> dict:
    out: dict = {}
    if not adapter_dir.exists():
        return out
    for child in sorted(adapter_dir.iterdir()):
        if child.is_file():
            out[child.name] = {"sha256": sha256(child), "bytes": child.stat().st_size}
    return out


def git_state() -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=str(REPO_ROOT),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return ""

    return {
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head_sha": git("rev-parse", "HEAD"),
        "head_short": git("rev-parse", "--short", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def tool_versions() -> dict:
    out: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for mod in ("mlx_lm", "mlx", "transformers", "huggingface_hub"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            out[mod] = "not-installed"
    return out


def build_manifest(
    corpus: Path,
    summary_paths: list[Path],
    adapter_dirs: list[Path],
    seeds: list[int],
    model: str,
    iters: int,
    num_layers: int,
    batch_size: int,
    learning_rate: float,
    holdout_size: int,
    eval_command: list[str],
    notes: str,
) -> dict:
    summaries: list[dict] = []
    for p in summary_paths:
        if p.exists():
            summaries.append(
                {
                    "path": str(p.relative_to(REPO_ROOT))
                    if p.is_relative_to(REPO_ROOT)
                    else str(p),
                    "sha256": sha256(p),
                    "bytes": p.stat().st_size,
                }
            )
    adapters: list[dict] = []
    for d in adapter_dirs:
        adapters.append(
            {
                "path": str(d.relative_to(REPO_ROOT)) if d.is_relative_to(REPO_ROOT) else str(d),
                "files": adapter_hashes(d),
            }
        )
    return {
        "schema_version": "aft-manifest/0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "path": str(corpus.relative_to(REPO_ROOT))
            if corpus.is_relative_to(REPO_ROOT)
            else str(corpus),
            "sha256": sha256(corpus) if corpus.exists() else None,
            "bytes": corpus.stat().st_size if corpus.exists() else None,
        },
        "seeds": list(seeds),
        "holdout_size": holdout_size,
        "model": {
            "name": model,
            "revision": "unpinned",  # HF repo @ default branch; pin via env if needed
            "huggingface_url": f"https://huggingface.co/{model}",
        },
        "training_args": {
            "fine_tune_type": "lora",
            "num_layers": num_layers,
            "batch_size": batch_size,
            "iters": iters,
            "learning_rate": learning_rate,
        },
        "adapter_artifacts": adapters,
        "evaluation": {
            "command": eval_command,
            "summaries": summaries,
        },
        "tool_versions": tool_versions(),
        "git": git_state(),
        "environment": {
            "user": os.environ.get("USER", "unknown"),
            "hostname": platform.node(),
            "cwd": str(REPO_ROOT),
        },
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an AFT reproducibility manifest")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--summary",
        type=Path,
        nargs="+",
        required=True,
        help="One or more harness summary JSON files",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        nargs="+",
        default=[],
        help="One or more adapter directories whose files will be hashed",
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--iters", type=int, required=True)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--holdout-size", type=int, required=True)
    parser.add_argument(
        "--eval-command",
        nargs=argparse.REMAINDER,
        default=[],
        help="Use after `--`: the exact eval command",
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    manifest = build_manifest(
        corpus=args.corpus,
        summary_paths=args.summary,
        adapter_dirs=args.adapter_dir,
        seeds=args.seeds,
        model=args.model,
        iters=args.iters,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        holdout_size=args.holdout_size,
        eval_command=args.eval_command,
        notes=args.notes,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    LOG.info("Wrote manifest: %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
