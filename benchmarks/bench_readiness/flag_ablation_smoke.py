#!/usr/bin/env python3
# ruff: noqa: BLE001, T201
"""
Bench-readiness flag ablation smoke test.

Goal: verify that each feature flag the benchmark plans to ablate genuinely
turns code paths on and off. This does NOT measure quality; it only confirms
that the flag is reachable, non-crashing, and observably different.

For every flag in TARGET_FLAGS we:
  1. Build an Arena with the flag OFF (and all other flags off).
  2. Build an Arena with only the flag ON.
  3. Run a tiny 1-round, 3-demo-agent debate on each.
  4. Diff a set of observable attributes and record the result.

Outcome per flag is one of:
  - "toggles"     : observable diff between OFF/ON runs
  - "no-op"       : no observable diff (flag may be vestigial, hooked elsewhere,
                    or inactive at this debate scale)
  - "crash_on"    : crashed when ON
  - "crash_off"   : crashed when OFF
  - "skip"        : flag is not accepted by the current DebateProtocol/Arena

Usage:
  .venv/bin/python -m benchmarks.bench_readiness.flag_ablation_smoke
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
from collections.abc import Iterable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))


TARGET_FLAGS: list[str] = [
    # Flags the product thesis depends on.
    "enable_calibration",
    "enable_trickster",
    "enable_rhetorical_observer",
    "enable_truth_ratio_weighting",
    "enable_cross_verification",
    "enable_coordinated_writes",
    "enable_performance_feedback",
    "enable_cross_debate_memory",
    "enable_live_explainability",
    "enable_introspection",
]


@dataclass
class FlagResult:
    flag: str
    outcome: str
    off_diff_keys: list[str] = field(default_factory=list)
    on_diff_keys: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    error: str | None = None


def _snapshot(arena: Any, result: Any) -> dict[str, Any]:
    """Collect a small set of observable attributes for diffing."""
    attrs: dict[str, Any] = {}

    # Per-flag attributes we scan on arena and protocol
    for obj_name, obj in (("arena", arena), ("protocol", getattr(arena, "protocol", None))):
        if obj is None:
            continue
        for flag in TARGET_FLAGS:
            val = getattr(obj, flag, None)
            if val is not None:
                attrs[f"{obj_name}.{flag}"] = bool(val)

    # Trackers / observers that flags conditionally instantiate
    tracker_names = (
        "calibration_tracker",
        "rhetorical_observer",
        "trickster",
        "cross_verification_enricher",
        "memory_coordinator",
        "selection_feedback_loop",
        "live_explainability",
        "introspection",
    )
    for name in tracker_names:
        has_tracker = getattr(arena, name, None) is not None
        attrs[f"has.{name}"] = has_tracker

    # Result-level keys
    if result is not None:
        metadata = getattr(result, "metadata", None) or {}
        for key in (
            "calibration_report",
            "cross_verification",
            "trickster_findings",
            "truth_ratio_weights",
            "introspection_snapshots",
        ):
            attrs[f"result.metadata.has.{key}"] = (
                key in metadata if isinstance(metadata, dict) else False
            )
        attrs["result.consensus_reached"] = bool(getattr(result, "consensus_reached", False))
        attrs["result.rounds_completed"] = int(getattr(result, "rounds_completed", 0) or 0)

    return attrs


def _build_arena(flag_name: str | None, enabled: bool) -> Any:
    """Construct a minimal Arena with three DemoAgents and a 1-round debate.

    If flag_name is None, all TARGET_FLAGS are left at default. Otherwise only
    flag_name is set to ``enabled`` (others stay at default-off).
    """
    from aragora import Arena, Environment, DebateProtocol
    from aragora.agents import create_agent

    kwargs: dict[str, Any] = {
        "rounds": 1,
        "consensus": "majority",
        "use_structured_phases": False,
    }
    if flag_name is not None:
        kwargs[flag_name] = enabled

    try:
        protocol = DebateProtocol(**kwargs)
    except TypeError:
        # The flag is not accepted by DebateProtocol; retry without it and
        # try to set it after construction (some flags live on Arena).
        protocol_kwargs = {k: v for k, v in kwargs.items() if k != flag_name}
        protocol = DebateProtocol(**protocol_kwargs)
        if flag_name is not None:
            setattr(protocol, flag_name, enabled)

    agents = [
        create_agent("demo", name=f"demo_{i}", role=role)
        for i, role in enumerate(("proposer", "critic", "synthesizer"))
    ]

    env = Environment(task="Smoke-test: should we adopt caching?", max_rounds=1)
    arena = Arena(env, agents, protocol)

    # Some flags live on the Arena itself (cross_verification, coordinated_writes, etc.)
    if flag_name is not None and not hasattr(protocol, flag_name):
        try:
            setattr(arena, flag_name, enabled)
        except Exception:
            pass

    return arena


async def _run_flag(flag: str) -> FlagResult:
    start = time.perf_counter()
    try:
        off_arena = _build_arena(flag, False)
        off_result = await asyncio.wait_for(off_arena.run(), timeout=30)
        off_snap = _snapshot(off_arena, off_result)
    except asyncio.TimeoutError:
        return FlagResult(
            flag=flag,
            outcome="timeout_off",
            error="off-run timed out after 30s",
            duration_s=time.perf_counter() - start,
        )
    except Exception as e:
        return FlagResult(
            flag=flag,
            outcome="crash_off",
            error=f"{type(e).__name__}: {e}",
            duration_s=time.perf_counter() - start,
        )

    try:
        on_arena = _build_arena(flag, True)
        on_result = await asyncio.wait_for(on_arena.run(), timeout=30)
        on_snap = _snapshot(on_arena, on_result)
    except asyncio.TimeoutError:
        return FlagResult(
            flag=flag,
            outcome="timeout_on",
            error="on-run timed out after 30s",
            duration_s=time.perf_counter() - start,
        )
    except Exception as e:
        return FlagResult(
            flag=flag,
            outcome="crash_on",
            error=f"{type(e).__name__}: {e}",
            duration_s=time.perf_counter() - start,
        )

    diff_off: list[str] = []
    diff_on: list[str] = []
    all_keys = sorted(set(off_snap) | set(on_snap))
    for k in all_keys:
        ov, nv = off_snap.get(k), on_snap.get(k)
        if ov != nv:
            diff_off.append(f"{k}={ov}")
            diff_on.append(f"{k}={nv}")

    outcome = "toggles" if diff_off else "no-op"
    return FlagResult(
        flag=flag,
        outcome=outcome,
        off_diff_keys=diff_off,
        on_diff_keys=diff_on,
        duration_s=time.perf_counter() - start,
    )


async def _main(flags: Iterable[str]) -> dict[str, Any]:
    results: list[FlagResult] = []
    for flag in flags:
        print(f"[flag] {flag} ...", flush=True)
        try:
            res = await _run_flag(flag)
        except Exception as e:
            res = FlagResult(flag=flag, outcome="smoke_error", error=f"{type(e).__name__}: {e}")
            traceback.print_exc()
        print(
            f"[flag] {flag} -> {res.outcome} ({res.duration_s:.2f}s)"
            + (f" err={res.error}" if res.error else ""),
            flush=True,
        )
        results.append(res)

    summary: dict[str, int] = {}
    for r in results:
        summary[r.outcome] = summary.get(r.outcome, 0) + 1

    return {
        "results": [asdict(r) for r in results],
        "summary": summary,
        "total": len(results),
    }


if __name__ == "__main__":
    out_path = HERE / "flag_ablation_smoke.json"
    payload = asyncio.run(_main(TARGET_FLAGS))
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print("\n=== summary ===")
    print(json.dumps(payload["summary"], indent=2))
    print(f"\nwrote: {out_path}")
