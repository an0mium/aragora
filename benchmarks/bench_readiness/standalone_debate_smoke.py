#!/usr/bin/env python3
# ruff: noqa: BLE001, T201
"""
Bench-readiness: does the standalone `aragora-debate` package run cleanly?

The full `aragora` Arena pulls in research, pipeline, gates, and background
services that make deterministic offline smoke tests infeasible (see
flag_ablation_smoke.json -> 10/10 timeouts at 30s per flag). This probes the
standalone package that the playground handler actually uses for demos.

If this completes in <5s with a usable result, the benchmark path can build on
aragora-debate (the thin core) rather than the full orchestrator.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "aragora-debate" / "src"))


async def _run_debate(
    *, enable_trickster: bool = False, enable_convergence: bool = False
) -> dict[str, object]:
    from aragora_debate import Debate
    from aragora_debate.styled_mock import StyledMockAgent

    debate = Debate(
        topic="Should we adopt caching?",
        rounds=2,
        consensus="majority",
        enable_trickster=enable_trickster,
        enable_convergence=enable_convergence,
    )
    debate.add_agent(StyledMockAgent("analyst", style="supportive"))
    debate.add_agent(StyledMockAgent("critic", style="critical"))
    debate.add_agent(StyledMockAgent("moderator", style="balanced"))

    started = time.perf_counter()
    result = await asyncio.wait_for(debate.run(), timeout=30)
    duration = time.perf_counter() - started

    receipt = getattr(result, "receipt", None)
    return {
        "duration_s": round(duration, 3),
        "consensus_reached": bool(getattr(result, "consensus_reached", False)),
        "rounds_completed": int(getattr(result, "rounds_completed", 0) or 0),
        "final_answer_len": len(str(getattr(result, "final_answer", "") or "")),
        "num_messages": len(getattr(result, "messages", []) or []),
        "has_receipt": receipt is not None,
        "receipt_md_len": len(receipt.to_markdown()) if receipt is not None else 0,
    }


async def _run() -> dict[str, object]:
    # Baseline: no flags. Then: trickster on. Then: convergence on.
    baseline = await _run_debate()
    trickster = await _run_debate(enable_trickster=True)
    convergence = await _run_debate(enable_convergence=True)

    return {
        "baseline": baseline,
        "enable_trickster": trickster,
        "enable_convergence": convergence,
        "diffs": {
            "trickster_vs_baseline": {
                k: (baseline[k], trickster[k]) for k in baseline if baseline[k] != trickster[k]
            },
            "convergence_vs_baseline": {
                k: (baseline[k], convergence[k]) for k in baseline if baseline[k] != convergence[k]
            },
        },
    }


def _main() -> None:
    out_path = HERE / "standalone_debate_smoke.json"
    try:
        payload = asyncio.run(_run())
        payload["outcome"] = "ok"
    except asyncio.TimeoutError:
        payload = {"outcome": "timeout", "error": "run exceeded 30s"}
    except Exception as e:
        payload = {
            "outcome": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
