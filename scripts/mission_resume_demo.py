#!/usr/bin/env python3
"""End-to-end proof that a mission survives a real ``kill -9``.

Run with no args to drive the whole demo::

    python scripts/mission_resume_demo.py

It seeds a 6-feature mission, runs the orchestrator as a child process, SIGKILLs
it mid-feature, then relaunches it on the same on-disk state and shows the mission
finishing with every feature completed exactly once, in order.

(The ``--worker`` mode is the child the parent spawns and kills.)
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

# Allow running from a source checkout without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aragora.missions import Feature, Handoff, MissionOrchestrator, MissionState, Status  # noqa: E402

PER_FEATURE_SECONDS = 0.6


def _seed(state_path: Path, log_path: Path) -> None:
    log_path.unlink(missing_ok=True)
    MissionState(
        mission_id="resume-demo",
        goal="prove kill -9 survivability",
        milestones=["m1"],
        features=[
            Feature(id=f"f{i}", description=f"step {i}", milestone="m1") for i in range(1, 7)
        ],
    ).save(state_path)


def _worker(state_path: Path, log_path: Path) -> None:
    """Child: each feature 'works' for PER_FEATURE_SECONDS then logs completion."""

    def dispatch(feat: Feature) -> Handoff:
        time.sleep(PER_FEATURE_SECONDS)  # the window the parent kills us in
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(feat.id + "\n")
        return Handoff(success=True, session_id=f"pid-{os.getpid()}")

    MissionOrchestrator(state_path).run(dispatch)


def _completed(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    return [ln.strip() for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> int:
    import subprocess
    import tempfile

    work = Path(tempfile.mkdtemp(prefix="mission-resume-"))
    state_path = work / "state.json"
    log_path = work / "completed.log"
    _seed(state_path, log_path)

    print(f"seeded 6-feature mission at {state_path}")
    print(f"each feature takes ~{PER_FEATURE_SECONDS}s of 'work'\n")

    # ---- launch worker, let it finish ~2 features, then kill -9 --------------
    child = subprocess.Popen(  # noqa: S603
        [sys.executable, __file__, "--worker", str(state_path), "--log", str(log_path)]
    )
    time.sleep(PER_FEATURE_SECONDS * 2 + 0.3)  # ~2 features done, mid-way through f3
    os.kill(child.pid, signal.SIGKILL)
    child.wait()
    print(f"*** SENT kill -9 to worker pid {child.pid} (mid-feature) ***\n")

    mid = MissionState.load(state_path)
    done_mid = [f.id for f in mid.features if f.status == Status.COMPLETED]
    inflight = [f.id for f in mid.features if f.status == Status.IN_PROGRESS]
    print("on-disk state immediately after kill:")
    print(f"  completed (durable): {done_mid}")
    print(f"  in_progress (checkpointed, will be reclaimed): {inflight}")
    print(f"  completion log:      {_completed(log_path)}\n")

    # ---- relaunch on the SAME state; it resumes and finishes ----------------
    print("relaunching orchestrator on the same on-disk state...")
    rc = subprocess.run(  # noqa: S603
        [sys.executable, __file__, "--worker", str(state_path), "--log", str(log_path)],
        check=False,
    ).returncode

    final = MissionState.load(state_path)
    done_final = [f.id for f in final.features if f.status == Status.COMPLETED]
    log = _completed(log_path)
    print("\nafter resume:")
    print(f"  completed: {done_final}")
    print(f"  completion log (each feature done exactly once, in order): {log}")

    ok = (
        rc == 0
        and len(done_final) == 6
        and log == [f"f{i}" for i in range(1, 7)]  # no loss, no dupes, in order
    )
    print(
        "\n"
        + (
            "✅ PASS — mission survived kill -9 with zero lost or double-done work"
            if ok
            else "❌ FAIL — resume did not converge cleanly"
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--log")
    args = ap.parse_args()
    if args.worker:
        _worker(Path(args.worker), Path(args.log))
        sys.exit(0)
    sys.exit(main())
