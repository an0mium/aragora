#!/usr/bin/env python3
"""
Production smoke test for Aragora.

Validates that core subsystems are functional:
1. Server starts in offline mode and binds to a port
2. Health endpoint responds with 200
3. A debate can be created and run with mock agents
4. Receipt generation works with SHA-256 integrity verification
5. Frontend (next build) compiles successfully

Usage:
    python scripts/smoke_test.py           # Run all checks
    python scripts/smoke_test.py --quick   # Skip frontend build (faster)
    python scripts/smoke_test.py --verbose # Show detailed output

Exit codes:
    0 - All checks passed
    1 - One or more checks failed
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []
_verbose = False


def _log(msg: str) -> None:
    print(msg, flush=True)


def _check(name: str, passed: bool, detail: str = "") -> bool:
    _results.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    marker = "[+]" if passed else "[-]"
    line = f"  {marker} {name}: {status}"
    if detail and (_verbose or not passed):
        line += f"  ({detail})"
    _log(line)
    return passed


def _find_free_port() -> int:
    """Find an available TCP port on localhost."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Check 1: Server startup in offline mode
# ---------------------------------------------------------------------------


def check_server_startup() -> bool:
    """Start the server with --offline flag and verify it binds."""
    api_port = _find_free_port()
    ws_port = _find_free_port()

    env = {
        **os.environ,
        "ARAGORA_OFFLINE": "true",
        "ARAGORA_DEMO_MODE": "true",
        "ARAGORA_DB_BACKEND": "sqlite",
        "ARAGORA_ENV": "development",
    }

    cmd = [
        sys.executable, "-m", "aragora",
        "serve",
        "--api-port", str(api_port),
        "--ws-port", str(ws_port),
        "--host", "127.0.0.1",
    ]

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Wait up to 15 seconds for the port to become available
        deadline = time.monotonic() + 15
        bound = False
        while time.monotonic() < deadline:
            # Check process is still alive
            if proc.poll() is not None:
                output = proc.stdout.read() if proc.stdout else ""
                return _check(
                    "server_startup",
                    False,
                    f"Server exited with code {proc.returncode}: {output[:300]}",
                )
            try:
                with closing(socket.create_connection(("127.0.0.1", api_port), timeout=1)):
                    bound = True
                    break
            except (ConnectionRefusedError, OSError):
                time.sleep(0.5)

        return _check(
            "server_startup",
            bound,
            f"port {api_port}" if bound else "server did not bind within 15s",
        )

    except FileNotFoundError:
        return _check("server_startup", False, "python executable not found")
    except Exception as exc:
        return _check("server_startup", False, str(exc))
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)


# ---------------------------------------------------------------------------
# Check 2: Health endpoint responds 200
# ---------------------------------------------------------------------------


def check_health_endpoint() -> bool:
    """Start a server and verify /health returns 200."""
    api_port = _find_free_port()
    ws_port = _find_free_port()

    env = {
        **os.environ,
        "ARAGORA_OFFLINE": "true",
        "ARAGORA_DEMO_MODE": "true",
        "ARAGORA_DB_BACKEND": "sqlite",
        "ARAGORA_ENV": "development",
    }

    cmd = [
        sys.executable, "-m", "aragora",
        "serve",
        "--api-port", str(api_port),
        "--ws-port", str(ws_port),
        "--host", "127.0.0.1",
    ]

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Wait for port to be ready
        deadline = time.monotonic() + 15
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                output = proc.stdout.read() if proc.stdout else ""
                return _check(
                    "health_endpoint",
                    False,
                    f"Server exited early: {output[:200]}",
                )
            try:
                with closing(socket.create_connection(("127.0.0.1", api_port), timeout=1)):
                    ready = True
                    break
            except (ConnectionRefusedError, OSError):
                time.sleep(0.5)

        if not ready:
            return _check("health_endpoint", False, "server did not become ready")

        # Hit the health endpoint using urllib (no external deps)
        import urllib.request
        import urllib.error

        url = f"http://127.0.0.1:{api_port}/health"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                status = resp.status
                body = resp.read().decode("utf-8", errors="replace")
                data = json.loads(body) if body else {}
                ok = status == 200 and data.get("status") == "ok"
                return _check(
                    "health_endpoint",
                    ok,
                    f"status={status} body={body[:100]}",
                )
        except urllib.error.URLError as exc:
            return _check("health_endpoint", False, f"URLError: {exc}")

    except Exception as exc:
        return _check("health_endpoint", False, str(exc))
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)


# ---------------------------------------------------------------------------
# Check 3: Debate creation and execution with mock agents
# ---------------------------------------------------------------------------


def check_debate_run() -> bool:
    """Create and run a debate with mock agents, verify result structure."""
    try:
        from aragora.core import Agent, Critique, Vote, Environment, DebateResult
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol
    except ImportError as exc:
        return _check("debate_run", False, f"import failed: {exc}")

    class SmokeAgent(Agent):
        """Minimal mock agent for smoke testing."""

        def __init__(self, name: str):
            super().__init__(name, "smoke-model", "proposer")
            self.agent_type = "smoke"

        async def generate(self, prompt: str, context: list | None = None) -> str:
            return f"Proposal from {self.name}: use token bucket rate limiting"

        async def critique(
            self,
            proposal: str,
            task: str,
            context: list | None = None,
            target_agent: str | None = None,
        ) -> Critique:
            return Critique(
                agent=self.name,
                target_agent=target_agent or "unknown",
                target_content=proposal[:80],
                issues=["Minor: add error handling"],
                suggestions=["Add retry logic"],
                severity=2.0,
                reasoning=f"Critique from {self.name}",
            )

        async def vote(self, proposals: dict[str, str], task: str) -> Vote:
            choice = list(proposals.keys())[0] if proposals else self.name
            return Vote(
                agent=self.name,
                choice=choice,
                reasoning=f"{self.name} votes for {choice}",
                confidence=0.9,
                continue_debate=False,
            )

    async def _run() -> DebateResult:
        agents = [SmokeAgent("alpha"), SmokeAgent("beta"), SmokeAgent("gamma")]
        env = Environment(task="Design a rate limiter API", max_rounds=1)
        protocol = DebateProtocol(rounds=1, consensus="any")
        arena = Arena(env, agents, protocol)
        return await asyncio.wait_for(arena.run(), timeout=30)

    try:
        result = asyncio.run(_run())
        checks = [
            isinstance(result, DebateResult),
            result.task == "Design a rate limiter API",
            result.rounds_used >= 1 or result.rounds_completed >= 1,
            bool(result.debate_id),
            isinstance(result.messages, list),
        ]
        if all(checks):
            return _check(
                "debate_run",
                True,
                f"rounds={result.rounds_used} participants={len(result.participants)}",
            )
        else:
            failed = [
                desc
                for desc, ok in zip(
                    ["isinstance", "task", "rounds", "debate_id", "messages"],
                    checks,
                )
                if not ok
            ]
            return _check("debate_run", False, f"failed assertions: {', '.join(failed)}")

    except Exception as exc:
        return _check("debate_run", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Check 4: Receipt generation with SHA-256 integrity
# ---------------------------------------------------------------------------


def check_receipt_generation() -> bool:
    """Create a DecisionReceipt and verify SHA-256 hash integrity."""
    try:
        from aragora.gauntlet.receipt_models import (
            DecisionReceipt,
            ConsensusProof,
            ProvenanceRecord,
        )
    except ImportError as exc:
        return _check("receipt_generation", False, f"import failed: {exc}")

    try:
        now = datetime.now(timezone.utc).isoformat()
        input_text = "Should we adopt microservices architecture?"
        input_hash = hashlib.sha256(input_text.encode()).hexdigest()

        receipt = DecisionReceipt(
            receipt_id=str(uuid.uuid4()),
            gauntlet_id=str(uuid.uuid4()),
            timestamp=now,
            input_summary=input_text,
            input_hash=input_hash,
            risk_summary={"critical": 0, "high": 1, "medium": 2, "low": 3},
            attacks_attempted=10,
            attacks_successful=1,
            probes_run=5,
            vulnerabilities_found=3,
            verdict="CONDITIONAL",
            confidence=0.82,
            robustness_score=0.75,
            verdict_reasoning="Generally sound with minor concerns",
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.85,
                supporting_agents=["agent_a", "agent_b"],
                dissenting_agents=["agent_c"],
                method="majority",
            ),
            provenance_chain=[
                ProvenanceRecord(
                    timestamp=now,
                    event_type="verdict",
                    description="Smoke test verdict",
                ),
            ],
        )

        # Verify the hash was computed
        assert receipt.artifact_hash, "artifact_hash is empty"
        assert len(receipt.artifact_hash) == 64, "artifact_hash is not SHA-256 hex"

        # Verify integrity check passes
        assert receipt.verify_integrity(), "verify_integrity() returned False"

        # Verify to_dict round-trip
        d = receipt.to_dict()
        assert d["receipt_id"] == receipt.receipt_id
        assert d["verdict"] == "CONDITIONAL"
        assert d["artifact_hash"] == receipt.artifact_hash

        # Verify tampering detection
        original_hash = receipt.artifact_hash
        receipt.verdict = "PASS"
        assert not receipt.verify_integrity(), "tampering was not detected"
        receipt.verdict = "CONDITIONAL"
        receipt.artifact_hash = original_hash

        return _check(
            "receipt_generation",
            True,
            f"hash={receipt.artifact_hash[:16]}... integrity verified",
        )

    except AssertionError as exc:
        return _check("receipt_generation", False, f"assertion failed: {exc}")
    except Exception as exc:
        return _check("receipt_generation", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Check 5: Frontend build
# ---------------------------------------------------------------------------


def check_frontend_build() -> bool:
    """Run `next build` in aragora/live/ and verify it compiles."""
    live_dir = PROJECT_ROOT / "aragora" / "live"
    package_json = live_dir / "package.json"

    if not package_json.exists():
        return _check("frontend_build", False, "aragora/live/package.json not found")

    # Check node_modules exist
    node_modules = live_dir / "node_modules"
    if not node_modules.exists():
        _log("    Installing frontend dependencies...")
        install = subprocess.run(
            ["npm", "install"],
            cwd=str(live_dir),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if install.returncode != 0:
            return _check(
                "frontend_build",
                False,
                f"npm install failed: {install.stderr[:200]}",
            )

    # Run the build
    _log("    Running next build (this may take a minute)...")
    try:
        build = subprocess.run(
            ["npm", "run", "build:local"],
            cwd=str(live_dir),
            capture_output=True,
            text=True,
            timeout=300,
            env={
                **os.environ,
                "NODE_ENV": "production",
                "NEXT_TELEMETRY_DISABLED": "1",
            },
        )
        if build.returncode == 0:
            return _check("frontend_build", True, "next build succeeded")
        else:
            # Extract useful error info from output
            stderr = build.stderr or ""
            stdout = build.stdout or ""
            error_lines = [
                line for line in (stderr + stdout).splitlines()
                if "error" in line.lower() or "Error" in line
            ]
            summary = "; ".join(error_lines[:3]) if error_lines else stderr[:200]
            return _check("frontend_build", False, f"build failed: {summary}")

    except subprocess.TimeoutExpired:
        return _check("frontend_build", False, "build timed out after 300s")
    except FileNotFoundError:
        return _check("frontend_build", False, "npm not found on PATH")
    except Exception as exc:
        return _check("frontend_build", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    global _verbose

    parser = argparse.ArgumentParser(
        description="Aragora production smoke test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip frontend build check (faster)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed output for passing checks",
    )
    parser.add_argument(
        "--skip-server", action="store_true",
        help="Skip server startup and health checks (CI without port access)",
    )
    args = parser.parse_args()
    _verbose = args.verbose

    _log("=" * 60)
    _log("ARAGORA SMOKE TEST")
    _log("=" * 60)
    _log("")

    # --- Server checks ---
    if not args.skip_server:
        _log("[1/5] Server startup (offline mode)...")
        check_server_startup()
        _log("")

        _log("[2/5] Health endpoint...")
        check_health_endpoint()
        _log("")
    else:
        _log("[1/5] Server startup: SKIPPED (--skip-server)")
        _results.append(("server_startup", True, "skipped"))
        _log("[2/5] Health endpoint: SKIPPED (--skip-server)")
        _results.append(("health_endpoint", True, "skipped"))
        _log("")

    # --- In-process checks ---
    _log("[3/5] Debate run (mock agents)...")
    check_debate_run()
    _log("")

    _log("[4/5] Receipt generation (SHA-256 integrity)...")
    check_receipt_generation()
    _log("")

    # --- Frontend build ---
    if not args.quick:
        _log("[5/5] Frontend build (next build)...")
        check_frontend_build()
    else:
        _log("[5/5] Frontend build: SKIPPED (--quick)")
        _results.append(("frontend_build", True, "skipped"))
    _log("")

    # --- Summary ---
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    failed = total - passed

    _log("=" * 60)
    if failed == 0:
        _log(f"ALL {total} CHECKS PASSED")
    else:
        _log(f"{failed}/{total} CHECKS FAILED:")
        for name, ok, detail in _results:
            if not ok:
                _log(f"  - {name}: {detail}")
    _log("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
