#!/usr/bin/env python3
"""Proof that many simple workers self-partition work with NO dispatcher.

Three independent worker processes scan the same mission queue and claim units
through the file-locked stigmergic Ledger. The claims are atomic, so the 9 units
partition across the 3 workers with **zero collisions** and **zero double-work** —
the "social insect" coordination, made real.

    python scripts/mission_swarm_demo.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aragora.missions import Feature, Handoff, MissionState, run_worker  # noqa: E402

N_UNITS = 9
N_WORKERS = 3
WORK_SECONDS = 0.08


def _worker(worker_id: str, state_path: Path, ledger_path: Path, log_path: Path) -> None:
    def dispatch(_: Feature) -> Handoff:
        time.sleep(WORK_SECONDS)  # 'work' the unit
        return Handoff(success=True)

    res = run_worker(state_path, ledger_path, worker_id, dispatch)
    log_path.write_text("\n".join(res.done), encoding="utf-8")


def main() -> int:
    import subprocess
    import tempfile

    work = Path(tempfile.mkdtemp(prefix="mission-swarm-"))
    state_path = work / "state.json"
    ledger_path = work / "ledger.json"
    MissionState(
        mission_id="swarm-demo",
        goal="prove dispatcher-free self-partition",
        milestones=["m1"],
        features=[
            Feature(id=f"f{i}", description=f"unit {i}", milestone="m1")
            for i in range(1, N_UNITS + 1)
        ],
    ).save(state_path)

    print(f"{N_UNITS} units, {N_WORKERS} independent worker processes, no dispatcher.\n")

    procs, logs = [], []
    for w in range(1, N_WORKERS + 1):
        wid = f"w{w}"
        log = work / f"{wid}.log"
        logs.append((wid, log))
        procs.append(
            subprocess.Popen(  # noqa: S603
                [
                    sys.executable,
                    __file__,
                    "--worker",
                    wid,
                    "--state",
                    str(state_path),
                    "--ledger",
                    str(ledger_path),
                    "--log",
                    str(log),
                ]
            )
        )
    for p in procs:
        p.wait()

    partition = {
        wid: (log.read_text(encoding="utf-8").split() if log.exists() else []) for wid, log in logs
    }
    all_done = [u for units in partition.values() for u in units]

    print("self-organized partition (who did what, decided by NO one):")
    for wid, units in partition.items():
        print(f"  {wid}: {units}")
    print(f"\n  total units done: {len(all_done)}")
    print(f"  distinct units:   {len(set(all_done))}")

    expected = {f"f{i}" for i in range(1, N_UNITS + 1)}
    ok = (
        len(all_done) == N_UNITS  # every unit done...
        and set(all_done) == expected  # ...exactly the right ones...
        and len(all_done) == len(set(all_done))  # ...each exactly once (no collision)
        and all(units for units in partition.values())  # ...and every worker contributed
    )
    print(
        "\n"
        + (
            "✅ PASS — 3 workers self-partitioned 9 units, zero collisions, no dispatcher"
            if ok
            else "❌ FAIL — collision or lost work"
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--state")
    ap.add_argument("--ledger")
    ap.add_argument("--log")
    args = ap.parse_args()
    if args.worker:
        _worker(args.worker, Path(args.state), Path(args.ledger), Path(args.log))
        sys.exit(0)
    sys.exit(main())
