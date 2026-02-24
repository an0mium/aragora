#!/usr/bin/env python3
"""Measure quickstart onboarding SLO: time from invocation to first debate result.

Target: <= 10 minutes (KPI #14 in EXECUTION_PROGRAM_2026Q2_Q4.md)

Usage:
    python scripts/measure_quickstart_time.py              # Demo mode (no API keys)
    python scripts/measure_quickstart_time.py --live       # Live agents (requires API keys)
    python scripts/measure_quickstart_time.py --rounds 3   # Custom round count
    python scripts/measure_quickstart_time.py --json       # Machine-readable output
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time


SLO_TARGET_SECONDS = 600  # 10 minutes


def measure_quickstart(
    *,
    demo: bool = True,
    rounds: int = 2,
    question: str = "Should we adopt microservices or keep our monolith?",
) -> dict:
    """Run aragora quickstart and measure wall-clock time.

    Returns a dict with timing breakdown and SLO pass/fail.
    """
    cmd = [
        sys.executable, "-m", "aragora.cli.main",
        "quickstart",
        "--question", question,
        "--rounds", str(rounds),
        "--no-browser",
    ]
    if demo:
        cmd.append("--demo")

    t_start = time.monotonic()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=SLO_TARGET_SECONDS + 60,  # grace period
    )

    t_end = time.monotonic()
    elapsed = t_end - t_start

    # Parse elapsed from output if available
    debate_elapsed = None
    for line in result.stdout.splitlines():
        if "Elapsed:" in line:
            try:
                debate_elapsed = float(line.split("Elapsed:")[1].strip().rstrip("s"))
            except (ValueError, IndexError):
                pass

    # Parse verdict
    verdict = None
    for line in result.stdout.splitlines():
        if "Verdict:" in line:
            verdict = line.split("Verdict:")[1].strip()

    return {
        "slo_target_seconds": SLO_TARGET_SECONDS,
        "total_wall_seconds": round(elapsed, 2),
        "debate_elapsed_seconds": debate_elapsed,
        "overhead_seconds": round(elapsed - (debate_elapsed or elapsed), 2),
        "slo_pass": elapsed <= SLO_TARGET_SECONDS,
        "exit_code": result.returncode,
        "verdict": verdict,
        "mode": "demo" if demo else "live",
        "rounds": rounds,
        "question": question,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure quickstart onboarding SLO")
    parser.add_argument("--live", action="store_true", help="Use live agents")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--question", default="Should we adopt microservices or keep our monolith?")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--iterations", type=int, default=1, help="Run N times for averaging")
    args = parser.parse_args()

    results = []
    for i in range(args.iterations):
        if not args.json and args.iterations > 1:
            print(f"Run {i + 1}/{args.iterations}...")

        r = measure_quickstart(
            demo=not args.live,
            rounds=args.rounds,
            question=args.question,
        )
        results.append(r)

    if args.json:
        if len(results) == 1:
            print(json.dumps(results[0], indent=2))
        else:
            avg = sum(r["total_wall_seconds"] for r in results) / len(results)
            print(json.dumps({
                "runs": results,
                "average_wall_seconds": round(avg, 2),
                "slo_pass": avg <= SLO_TARGET_SECONDS,
            }, indent=2))
        return

    # Human-readable output
    print("\n" + "=" * 50)
    print("  QUICKSTART ONBOARDING SLO MEASUREMENT")
    print("=" * 50)

    for i, r in enumerate(results):
        if len(results) > 1:
            print(f"\n--- Run {i + 1} ---")
        print(f"  Mode:           {r['mode']}")
        print(f"  Rounds:         {r['rounds']}")
        print(f"  Total wall:     {r['total_wall_seconds']:.1f}s")
        if r["debate_elapsed_seconds"] is not None:
            print(f"  Debate time:    {r['debate_elapsed_seconds']:.1f}s")
            print(f"  Overhead:       {r['overhead_seconds']:.1f}s")
        print(f"  Verdict:        {r['verdict'] or 'N/A'}")
        print(f"  Exit code:      {r['exit_code']}")
        status = "PASS" if r["slo_pass"] else "FAIL"
        print(f"  SLO (≤{SLO_TARGET_SECONDS}s):   {status}")

    if len(results) > 1:
        avg = sum(r["total_wall_seconds"] for r in results) / len(results)
        status = "PASS" if avg <= SLO_TARGET_SECONDS else "FAIL"
        print(f"\n  Average:        {avg:.1f}s — {status}")

    print("=" * 50)


if __name__ == "__main__":
    main()
