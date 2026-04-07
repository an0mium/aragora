#!/usr/bin/env python3
"""
Measure developer onboarding quickstart time for Aragora.

Times each step from project clone to first debate receipt, producing a
structured report with pass/fail thresholds.  Designed to run in CI or
locally so regressions in developer experience are caught early.

Steps measured:
  1. Clone/Download  -- repo .git size as proxy (informational)
  2. Dependencies     -- pip install --dry-run duration
  3. Import aragora   -- import aragora + key submodules
  4. First debate     -- Arena with 2 mock agents, 1-round debate
  5. Receipt          -- DecisionReceipt.from_debate_result()
  6. Server startup   -- aragora serve --demo to /readyz 200

Usage:
    python scripts/measure_quickstart_time.py          # human-readable table
    python scripts/measure_quickstart_time.py --json   # machine-readable JSON

Exit codes:
    0  all steps within thresholds
    1  one or more steps exceeded thresholds
"""

from __future__ import annotations

import argparse
import asyncio
import json as _json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Threshold configuration (seconds)
# ---------------------------------------------------------------------------

THRESHOLDS: dict[str, float] = {
    "clone": 0.0,  # informational only -- measured separately
    "install": 60.0,
    "import": 5.0,
    "first_debate": 10.0,
    "receipt": 1.0,
    "server_startup": 5.0,
    "total": 120.0,
}


_MEASUREMENT_TIMEOUT_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Result of a single timing step."""

    name: str
    duration: float  # seconds, -1.0 means skipped / not measured
    threshold: float  # seconds, 0.0 means informational (always passes)
    passed: bool
    error: str = ""
    detail: str = ""

    @property
    def status(self) -> str:
        if self.duration < 0:
            return "SKIP"
        if self.threshold <= 0:
            return "INFO"
        return "PASS" if self.passed else "FAIL"


@dataclass
class QuickstartReport:
    """Full quickstart timing report."""

    steps: list[StepResult] = field(default_factory=list)
    total_seconds: float = 0.0
    all_passed: bool = True
    repo_size_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [asdict(s) for s in self.steps],
            "total_seconds": round(self.total_seconds, 3),
            "all_passed": self.all_passed,
            "repo_size_mb": round(self.repo_size_mb, 1),
            "thresholds": THRESHOLDS,
        }


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def evaluate_step(
    name: str,
    duration: float,
    threshold: float,
    error: str = "",
    detail: str = "",
) -> StepResult:
    """Build a StepResult with pass/fail evaluation."""
    if duration < 0:
        return StepResult(
            name=name,
            duration=-1.0,
            threshold=threshold,
            passed=True,
            error=error,
            detail=detail,
        )
    passed = (threshold <= 0) or ((duration <= threshold) and not error)
    return StepResult(
        name=name,
        duration=round(duration, 3),
        threshold=threshold,
        passed=passed,
        error=error,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Individual measurement functions
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """Return the repository root for this script."""
    return Path(__file__).resolve().parent.parent


def _resolve_git_storage_root(repo_root: Path) -> Path | None:
    """Resolve the backing git storage directory, including worktree pointers."""
    git_path = repo_root / ".git"
    if not git_path.exists():
        return None
    if git_path.is_dir():
        return git_path
    if not git_path.is_file():
        return None

    try:
        first_line = git_path.read_text().splitlines()[0].strip()
    except (IndexError, OSError):
        return None

    prefix = "gitdir:"
    if not first_line.lower().startswith(prefix):
        return None

    gitdir = first_line[len(prefix) :].strip()
    if not gitdir:
        return None

    worktree_gitdir = (git_path.parent / gitdir).resolve()
    commondir_file = worktree_gitdir / "commondir"
    if commondir_file.is_file():
        try:
            commondir = commondir_file.read_text().strip()
        except OSError:
            commondir = ""
        if commondir:
            return (worktree_gitdir / commondir).resolve()
    return worktree_gitdir if worktree_gitdir.exists() else None


def _measurement_env() -> dict[str, str]:
    """Return a stable env for quickstart measurements."""
    env = os.environ.copy()
    repo_root = str(_repo_root())
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = (
        repo_root if not existing_pythonpath else os.pathsep.join([repo_root, existing_pythonpath])
    )
    env.setdefault("ARAGORA_USE_SECRETS_MANAGER", "false")
    return env


def _summarize_output(text: str, *, limit: int = 240) -> str:
    """Compact noisy subprocess output into a single short line."""
    compact = " | ".join(line.strip() for line in text.splitlines() if line.strip())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _summarize_process_output(result: subprocess.CompletedProcess[str]) -> str:
    """Summarize both stdout and stderr so warnings do not hide real failures."""
    return _summarize_output("\n".join(part for part in (result.stdout, result.stderr) if part))


def _parse_json_line(stdout: str) -> dict[str, Any]:
    """Parse the last non-empty stdout line as JSON."""
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        return _json.loads(candidate)
    raise ValueError("step produced no JSON output")


def _run_repo_python_step(
    *,
    name: str,
    threshold: float,
    code: str,
    detail_from_payload: Callable[[dict[str, Any]], str],
    timeout: float = _MEASUREMENT_TIMEOUT_SECONDS,
) -> StepResult:
    """Execute a measurement snippet in an isolated repo-root subprocess."""
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(_repo_root()),
            env=_measurement_env(),
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        return evaluate_step(name, elapsed, threshold, error="timeout")

    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        combined = _summarize_process_output(result)
        error = f"exit={result.returncode}"
        if combined:
            error = f"{error}: {combined}"
        return evaluate_step(name, elapsed, threshold, error=error)

    try:
        payload = _parse_json_line(result.stdout)
    except (ValueError, _json.JSONDecodeError) as exc:
        return evaluate_step(name, elapsed, threshold, error=str(exc))

    detail = detail_from_payload(payload)
    error = str(payload.get("error", "")).strip()
    return evaluate_step(name, elapsed, threshold, error=error, detail=detail)


def measure_repo_size() -> tuple[float, float]:
    """Return (duration_seconds, size_mb) for the .git directory.

    Used as an informational proxy for clone time.
    """
    repo_root = _repo_root()
    git_dir = _resolve_git_storage_root(repo_root)
    if git_dir is None or not git_dir.exists():
        return -1.0, 0.0

    if git_dir.is_file():
        try:
            gitdir_text = git_dir.read_text(encoding="utf-8").strip()
        except OSError:
            return -1.0, 0.0
        if not gitdir_text.startswith("gitdir:"):
            return -1.0, 0.0
        worktree_git_dir = (git_dir.parent / gitdir_text.split(":", 1)[1].strip()).resolve()
        if not worktree_git_dir.exists():
            return -1.0, 0.0
        commondir_file = worktree_git_dir / "commondir"
        if commondir_file.is_file():
            try:
                commondir = commondir_file.read_text(encoding="utf-8").strip()
            except OSError:
                commondir = ""
            if commondir:
                common_git_dir = (worktree_git_dir / commondir).resolve()
                if common_git_dir.exists():
                    git_dir = common_git_dir
                else:
                    git_dir = worktree_git_dir
            else:
                git_dir = worktree_git_dir
        else:
            git_dir = worktree_git_dir

    t0 = time.perf_counter()
    total_bytes = 0
    git_paths = [git_dir] if git_dir.is_file() else git_dir.rglob("*")
    for p in git_paths:
        if p.is_file():
            try:
                total_bytes += p.stat().st_size
            except OSError:
                pass
    elapsed = time.perf_counter() - t0
    return elapsed, total_bytes / (1024 * 1024)


def measure_install_time() -> StepResult:
    """Measure pip install --dry-run duration.

    Uses ``pip install --dry-run`` to exercise dependency resolution without
    modifying the environment.  Falls back to ``pip check`` when the
    dry-run flag is unavailable.
    """
    repo_root = _repo_root()
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--dry-run",
                "-e",
                f"{repo_root}[dev]",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        elapsed = time.perf_counter() - t0
        detail = f"pip install --dry-run exit={result.returncode}"
        error = ""

        if result.returncode != 0:
            # dry-run may not be supported; fall back to pip check
            t0b = time.perf_counter()
            result = subprocess.run(
                [sys.executable, "-m", "pip", "check"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            elapsed = time.perf_counter() - t0b
            detail = f"pip check exit={result.returncode} (dry-run unavailable)"
            if result.returncode != 0:
                error = _summarize_process_output(result) or detail

        return evaluate_step(
            "Dependencies Install",
            elapsed,
            THRESHOLDS["install"],
            error=error,
            detail=detail,
        )

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        return evaluate_step(
            "Dependencies Install", elapsed, THRESHOLDS["install"], error="timeout"
        )
    except FileNotFoundError:
        return evaluate_step(
            "Dependencies Install", -1.0, THRESHOLDS["install"], error="pip not found"
        )


def measure_import_time() -> StepResult:
    """Measure ``import aragora`` and key submodule import times."""
    code = r"""
import json

target_modules = [
    "aragora",
    "aragora.core",
    "aragora.debate.orchestrator",
    "aragora.debate.protocol",
    "aragora.gauntlet.receipt_models",
]
errors = []
imported = []
for mod in target_modules:
    try:
        __import__(mod)
        imported.append(mod)
    except Exception as exc:
        errors.append(f"{mod}: {exc}")

print(json.dumps({"imported": imported, "total": len(target_modules), "error": "; ".join(errors)}))
"""
    return _run_repo_python_step(
        name="Import aragora",
        threshold=THRESHOLDS["import"],
        code=code,
        detail_from_payload=lambda payload: (
            f"imported {len(payload.get('imported', []))}/{payload.get('total', 0)} modules"
        ),
    )


def measure_first_debate() -> StepResult:
    """Create an Arena with 2 mock agents and run a 1-round debate."""
    code = r"""
import asyncio
import json

from aragora.core import Agent, Critique, Environment, Message, Vote
from aragora.debate.orchestrator import Arena
from aragora.debate.protocol import DebateProtocol


class _QuickstartAgent(Agent):
    def __init__(self, name: str, response: str = "test response"):
        super().__init__(name=name, model="mock-model", role="proposer")
        self.agent_type = "mock"
        self._response = response

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        return self._response

    async def generate_stream(self, prompt: str, context: list[Message] | None = None):
        yield self._response

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list[Message] | None = None,
        target_agent: str | None = None,
    ) -> Critique:
        return Critique(
            agent=self.name,
            target_agent=target_agent or "proposer",
            target_content=proposal[:100],
            issues=["Minor issue"],
            suggestions=["Consider refactoring"],
            severity=0.3,
            reasoning="Test reasoning",
        )

    async def vote(self, proposals: dict[str, str], task: str) -> Vote:
        choice = self.name if self.name in proposals else next(iter(proposals), "unknown")
        return Vote(agent=self.name, choice=choice, reasoning="Test vote", confidence=0.8)


async def _main() -> None:
    agents = [
        _QuickstartAgent("agent-alpha", "Alpha: Use a token bucket algorithm with Redis."),
        _QuickstartAgent("agent-beta", "Beta: Agreed, token bucket with Redis backend."),
    ]
    env = Environment(task="Design a rate limiter for our API")
    protocol = DebateProtocol(rounds=1, consensus="majority")
    arena = Arena(environment=env, agents=agents, protocol=protocol)
    result = await arena.run()
    print(
        json.dumps(
            {
                "consensus": bool(result.consensus_reached),
                "rounds": int(result.rounds_used),
                "confidence": float(result.confidence),
            }
        )
    )


asyncio.run(_main())
"""
    return _run_repo_python_step(
        name="First debate",
        threshold=THRESHOLDS["first_debate"],
        code=code,
        detail_from_payload=lambda payload: (
            "consensus="
            f"{payload.get('consensus')}, rounds={payload.get('rounds')}, "
            f"confidence={float(payload.get('confidence', 0.0)):.2f}"
        ),
    )


def measure_receipt_generation() -> StepResult:
    """Create a DecisionReceipt from a synthetic DebateResult."""
    code = r"""
import json

from aragora.core_types import DebateResult
from aragora.gauntlet.receipt_models import DecisionReceipt

result = DebateResult(
    debate_id="bench-001",
    task="Design a rate limiter",
    final_answer="Use token bucket with Redis backend",
    confidence=0.85,
    consensus_reached=True,
    rounds_used=1,
    participants=["agent-alpha", "agent-beta"],
    proposals={"agent-alpha": "Token bucket", "agent-beta": "Sliding window"},
    messages=[],
    votes=[],
    winner="agent-alpha",
    duration_seconds=1.0,
)

receipt = DecisionReceipt.from_debate_result(result)
print(json.dumps({"receipt_id": receipt.receipt_id, "verdict": receipt.verdict}))
"""
    return _run_repo_python_step(
        name="Receipt generation",
        threshold=THRESHOLDS["receipt"],
        code=code,
        detail_from_payload=lambda payload: (
            f"receipt_id={str(payload.get('receipt_id', ''))[:8]}..., "
            f"verdict={payload.get('verdict', '')}"
        ),
    )


def measure_server_startup() -> StepResult:
    """Start the server in demo mode and measure time to /readyz 200.

    Uses ephemeral high ports to avoid conflicts with other services.
    """
    repo_root = _repo_root()
    env = os.environ.copy()
    env["ARAGORA_OFFLINE"] = "true"
    env["ARAGORA_DEMO_MODE"] = "true"
    env["ARAGORA_DB_BACKEND"] = "sqlite"
    env["ARAGORA_ENV"] = "development"

    http_port = 18_932
    ws_port = 18_933

    proc = None
    t0 = time.perf_counter()
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "aragora.cli.main",
                "serve",
                "--demo",
                "--api-port",
                str(http_port),
                "--ws-port",
                str(ws_port),
                "--host",
                "127.0.0.1",
            ],
            env=env,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        import urllib.error
        import urllib.request

        deadline = t0 + THRESHOLDS["server_startup"]
        ready = False
        while time.perf_counter() < deadline:
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{http_port}/readyz")
                with urllib.request.urlopen(req, timeout=1) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except (urllib.error.URLError, OSError, ConnectionError):
                pass
            time.sleep(0.15)

        elapsed = time.perf_counter() - t0
        if ready:
            return evaluate_step(
                "Server startup", elapsed, THRESHOLDS["server_startup"], detail="readyz=200"
            )
        return evaluate_step(
            "Server startup",
            elapsed,
            THRESHOLDS["server_startup"],
            error="server did not reach ready state within threshold",
        )

    except FileNotFoundError:
        elapsed = time.perf_counter() - t0
        return evaluate_step(
            "Server startup", elapsed, THRESHOLDS["server_startup"], error="could not start process"
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return evaluate_step(
            "Server startup", elapsed, THRESHOLDS["server_startup"], error=str(exc)
        )
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_report(report: QuickstartReport) -> str:
    """Render a human-readable timing report."""
    lines: list[str] = [
        "",
        "Aragora Quickstart Timing Report",
        "=" * 56,
        f"{'Step':<24}{'Duration':>10}    {'Status'}",
        "-" * 56,
    ]

    for step in report.steps:
        dur_str = "N/A" if step.duration < 0 else f"{step.duration:.1f}s"
        threshold_note = f" (< {step.threshold:.0f}s)" if step.threshold > 0 else ""
        status_str = f"{step.status}{threshold_note}"
        lines.append(f"{step.name:<24}{dur_str:>10}    {status_str}")
        if step.error:
            lines.append(f"  {'':24}  ERROR: {step.error}")

    lines.append("-" * 56)
    total_dur = f"{report.total_seconds:.1f}s"
    total_status = "PASS" if report.all_passed else "FAIL"
    lines.append(
        f"{'Total quickstart time':<24}{total_dur:>10}    "
        f"{total_status} (< {THRESHOLDS['total']:.0f}s)"
    )
    if report.repo_size_mb > 0:
        lines.append(f"{'Repo size (.git)':<24}{report.repo_size_mb:>9.1f}M")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_all_measurements() -> QuickstartReport:
    """Execute all quickstart timing measurements and return a report."""
    report = QuickstartReport()

    # 1. Clone / repo size (informational)
    _clone_dur, repo_mb = measure_repo_size()
    report.repo_size_mb = repo_mb
    report.steps.append(
        evaluate_step(
            "Clone/Download",
            -1.0,
            THRESHOLDS["clone"],
            detail=f"repo .git = {repo_mb:.1f} MB",
        )
    )

    # 2. Dependencies install
    report.steps.append(measure_install_time())

    # 3. Import time
    report.steps.append(measure_import_time())

    # 4. First debate
    report.steps.append(measure_first_debate())

    # 5. Receipt generation
    report.steps.append(measure_receipt_generation())

    # 6. Server startup
    report.steps.append(measure_server_startup())

    # Totals
    measured = [s for s in report.steps if s.duration >= 0]
    report.total_seconds = round(sum(s.duration for s in measured), 3)
    report.all_passed = all(s.passed for s in report.steps)
    if report.total_seconds > THRESHOLDS["total"]:
        report.all_passed = False

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure Aragora developer onboarding quickstart times.",
    )
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()

    report = run_all_measurements()

    if args.json:
        print(_json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
