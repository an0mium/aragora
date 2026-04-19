"""CLI entry point for the Tier-1 benchmark.

Usage::

    # Full pilot (N=10 per domain, all 4 domains, ~40 items)
    python -m benchmarks.bench_readiness.tier1.cli

    # Smoke test (N=1 per domain, 4 items)
    python -m benchmarks.bench_readiness.tier1.cli --limit 1

    # One domain only
    python -m benchmarks.bench_readiness.tier1.cli --domains legal --limit 3
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

ALL_DOMAINS = ("legal", "aragora_custom", "mmlu_pro", "swebench_lite")
DEFAULT_OUT = pathlib.Path("benchmarks/bench_readiness/tier1/results")


def _resolve_api_key() -> str:
    """Fetch the Anthropic key. Prefers AWS-first path per the hardening posture.

    If ``ARAGORA_USE_SECRETS_MANAGER=true`` and AWS is reachable, we pull
    from the ``aragora/production`` bundle. Otherwise fall back to the env
    var. Strict mode callers should set the env before invocation.
    """
    try:
        from aragora.config.secrets import get_secret

        v = get_secret("ANTHROPIC_API_KEY")
        if v:
            return v
    except Exception:  # noqa: BLE001 - any secrets backend failure should fall back to env
        pass

    v = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not v:
        sys.stderr.write(
            "ERROR: ANTHROPIC_API_KEY not available via AWS Secrets Manager "
            "or environment. Set ARAGORA_USE_SECRETS_MANAGER=true with AWS "
            "credentials, or export ANTHROPIC_API_KEY directly.\n",
        )
        sys.exit(2)
    return v


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tier-1 benchmark: solo Opus 4.7 vs aragora-debate (3x Opus 4.7)."
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=ALL_DOMAINS,
        default=list(ALL_DOMAINS),
        help="Task domains to include. Defaults to all four.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max items per domain. Use 1 for a smoke test.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=DEFAULT_OUT,
        help="Output directory for CSV + summary.",
    )
    parser.add_argument(
        "--debate-rounds",
        type=int,
        default=2,
        help="Number of debate rounds (default 2).",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-7",
        help="Model ID for both solo and debate agents.",
    )
    args = parser.parse_args()

    api_key = _resolve_api_key()

    from benchmarks.bench_readiness.tier1.runner import run

    manifest = run(
        api_key=api_key,
        domains=list(args.domains),
        limit=args.limit,
        seed=args.seed,
        out_dir=args.out,
        debate_rounds=args.debate_rounds,
        model=args.model,
    )

    sys.stdout.write(
        "\n"
        + "=" * 60
        + "\n"
        + f"Items : {manifest['items']}\n"
        + f"CSV   : {manifest['csv']}\n"
        + f"Sum   : {manifest['summary']}\n"
        + "=" * 60
        + "\n"
    )


if __name__ == "__main__":
    main()
