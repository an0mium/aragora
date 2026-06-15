"""Tests for ``aragora.swarm.lane_conductor`` -- pure pass planning + safe run.

Every case injects fake fetch/claim/dispatch callables, so the conductor's
decision and its execute-mode side effects are exercised without GitHub, the
lane registry, a worktree, or a spawned worker.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aragora.swarm import lane_conductor as lc
from aragora.swarm import lane_supervisor as ls
from aragora.swarm.lane_dispatcher import LaneAssignment


def _load_cli() -> Any:
    script = Path(__file__).resolve().parents[2] / "scripts" / "lane_conductor.py"
    spec = importlib.util.spec_from_file_location("lane_conductor_cli_under_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


def _cand(number: int, branch: str = "") -> dict[str, Any]:
    return {"number": number, "branch": branch or f"codex/pr{number}"}


def _seq_session(pr: int) -> str:
    return f"sess-{pr}"


def _fixed_now() -> str:
    return "2026-06-15T00:00:00Z"


# ---------------------------------------------------------------------------
# build_work_orders / plan_pass (pure)
# ---------------------------------------------------------------------------


def test_build_work_orders_one_per_assignment_with_prompt() -> None:
    orders = lc.build_work_orders(
        [LaneAssignment(pr=8405, branch="codex/a", owner_session="sess-8405")],
        repo="synaptent/aragora",
        target_agent="codex",
        now=_fixed_now,
    )
    (wo,) = orders
    assert wo.work_order_id == "lane-8405-sess-8405"
    assert wo.pr == 8405
    assert wo.target_agent == "codex"
    assert "CLAIM-OR-YIELD" in wo.prompt
    assert "#8405" in wo.prompt
    assert wo.created_at == "2026-06-15T00:00:00Z"


def test_work_order_to_dict_mirrors_launcher_key() -> None:
    wo = lc.build_work_orders(
        [LaneAssignment(pr=1, branch="b", owner_session="s")],
        repo="r",
        now=_fixed_now,
    )[0]
    data = wo.to_dict()
    # worker_launcher.WorkerLauncher.launch reads owner_session_id.
    assert data["owner_session_id"] == "s"
    assert data["owner_session"] == "s"


def test_plan_pass_assigns_and_carries_owned_deferred() -> None:
    result = lc.plan_pass(
        candidates=[_cand(1), _cand(2), _cand(3)],
        live_claims_by_pr={2: "other"},
        repo="r",
        max_workers=1,
        session_id_for=_seq_session,
        now=_fixed_now,
    )
    assert [wo.pr for wo in result.work_orders] == [1]
    assert result.owned == {2: "other"}
    assert result.deferred == [3]
    assert result.executed is False


# ---------------------------------------------------------------------------
# run_pass: dry-run does no side effects
# ---------------------------------------------------------------------------


def test_run_pass_dry_run_does_not_claim_or_dispatch() -> None:
    claims: list[int] = []
    dispatched: list[int] = []

    result = lc.run_pass(
        repo="r",
        fetch_candidates=lambda repo: [_cand(1), _cand(2)],
        fetch_live_claims=lambda repo, cands: {},
        max_workers=5,
        execute=False,
        claim_fn=lambda wo: claims.append(wo.pr),
        dispatch_fn=lambda wo: (dispatched.append(wo.pr), "x")[1],
        session_id_for=_seq_session,
        now=_fixed_now,
    )
    assert [wo.pr for wo in result.work_orders] == [1, 2]
    assert claims == []
    assert dispatched == []
    assert result.executed is False
    assert "dry-run" in result.reason


def test_run_pass_execute_claims_then_dispatches_each() -> None:
    order: list[str] = []

    def claim(wo: lc.WorkOrderSpec) -> bool:
        order.append(f"claim:{wo.pr}")
        return True

    def dispatch(wo: lc.WorkOrderSpec) -> str:
        order.append(f"dispatch:{wo.pr}")
        return f"/pending/{wo.work_order_id}.json"

    result = lc.run_pass(
        repo="r",
        fetch_candidates=lambda repo: [_cand(1), _cand(2)],
        fetch_live_claims=lambda repo, cands: {},
        max_workers=5,
        execute=True,
        claim_fn=claim,
        dispatch_fn=dispatch,
        session_id_for=_seq_session,
        now=_fixed_now,
    )
    # Claim must precede dispatch for each lane.
    assert order == ["claim:1", "dispatch:1", "claim:2", "dispatch:2"]
    assert result.executed is True
    assert len(result.dispatched) == 2
    assert result.dispatched[0].endswith("lane-1-sess-1.json")


def test_run_pass_execute_requires_explicit_claim_success() -> None:
    dispatched: list[int] = []

    result = lc.run_pass(
        repo="r",
        fetch_candidates=lambda repo: [_cand(1)],
        fetch_live_claims=lambda repo, cands: {},
        max_workers=5,
        execute=True,
        claim_fn=lambda wo: None,
        dispatch_fn=lambda wo: (dispatched.append(wo.pr), "p")[1],
        session_id_for=_seq_session,
        now=_fixed_now,
    )

    assert dispatched == []
    assert result.claim_failed == [1]
    assert result.dispatched == []


def test_run_pass_execute_skips_dispatch_when_claim_fails() -> None:
    dispatched: list[int] = []

    def claim(wo: lc.WorkOrderSpec) -> bool:
        return wo.pr != 1

    result = lc.run_pass(
        repo="r",
        fetch_candidates=lambda repo: [_cand(1), _cand(2)],
        fetch_live_claims=lambda repo, cands: {},
        max_workers=5,
        execute=True,
        claim_fn=claim,
        dispatch_fn=lambda wo: (dispatched.append(wo.pr), "p")[1],
        session_id_for=_seq_session,
        now=_fixed_now,
    )
    assert dispatched == [2]
    assert result.claim_failed == [1]
    assert result.dispatched == ["p"]


def test_run_pass_execute_skips_live_owned_lanes() -> None:
    claimed: list[int] = []
    result = lc.run_pass(
        repo="r",
        fetch_candidates=lambda repo: [_cand(1), _cand(2)],
        fetch_live_claims=lambda repo, cands: {1: "owner-x"},
        max_workers=5,
        execute=True,
        claim_fn=lambda wo: (claimed.append(wo.pr), True)[1],
        dispatch_fn=lambda wo: "p",
        session_id_for=_seq_session,
        now=_fixed_now,
    )
    # PR 1 is live-owned: never claimed/dispatched; only PR 2 is worked.
    assert claimed == [2]
    assert result.owned == {1: "owner-x"}


# ---------------------------------------------------------------------------
# default_dispatch writes an atomic, well-formed work-order file
# ---------------------------------------------------------------------------


def test_default_dispatch_writes_atomic_json(tmp_path: Path) -> None:
    wo = lc.build_work_orders(
        [LaneAssignment(pr=42, branch="codex/x", owner_session="sess-42")],
        repo="synaptent/aragora",
        now=_fixed_now,
    )[0]
    written = lc.default_dispatch(wo, repo_root=tmp_path)
    path = Path(written)
    assert path.exists()
    assert path.parent == tmp_path / lc.DISPATCH_PENDING_DIR
    # No leftover tmp file.
    assert not path.with_suffix(".json.tmp").exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["pr"] == 42
    assert payload["owner_session_id"] == "sess-42"
    assert "CLAIM-OR-YIELD" in payload["prompt"]


def test_default_dispatch_uses_explicit_repo_root(tmp_path: Path) -> None:
    wo = lc.build_work_orders(
        [LaneAssignment(pr=42, branch="codex/x", owner_session="sess-42")],
        repo="synaptent/aragora",
        now=_fixed_now,
    )[0]

    written = Path(lc.default_dispatch(wo, repo_root=tmp_path))

    assert written.parent == tmp_path / ".aragora" / "lane_dispatch" / "pending"
    assert written.exists()


def test_conductor_and_supervisor_share_dispatch_root_constant() -> None:
    assert lc.DISPATCH_PENDING_DIR == ls.DISPATCH_ROOT / ls.PENDING


def test_cli_execute_uses_explicit_root_for_claim_and_dispatch(
    tmp_path: Path, monkeypatch: Any
) -> None:
    roots: list[Path] = []

    def fake_claim(wo: lc.WorkOrderSpec, *, repo_root: Path | None = None) -> bool:
        assert repo_root is not None
        roots.append(repo_root)
        return True

    def fake_dispatch(wo: lc.WorkOrderSpec, *, repo_root: Path | None = None) -> str:
        assert repo_root is not None
        roots.append(repo_root)
        return str(
            repo_root / ".aragora" / "lane_dispatch" / "pending" / f"{wo.work_order_id}.json"
        )

    monkeypatch.setattr(cli, "fetch_candidates", lambda repo: [_cand(42, "codex/x")])
    monkeypatch.setattr(cli, "fetch_live_claims", lambda repo, candidates: {})
    monkeypatch.setattr(cli, "default_claim", fake_claim)
    monkeypatch.setattr(cli, "default_dispatch", fake_dispatch)

    code = cli.main(["--execute", "--root", str(tmp_path), "--json"])

    assert code == 0
    assert roots == [tmp_path.resolve(), tmp_path.resolve()]


# ---------------------------------------------------------------------------
# CLI live-owner liveness parsing
# ---------------------------------------------------------------------------


def _proc(stdout: str = "", *, returncode: int = 0, stderr: str = "") -> Any:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_fetch_live_claims_reads_canonical_assessed_field(monkeypatch: Any) -> None:
    payload = {"owner_session": "owner-a", "owner_liveness": {"assessed": "live"}}
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: _proc(json.dumps(payload)))

    assert cli.fetch_live_claims("synaptent/aragora", [_cand(1)]) == {1: "owner-a"}


def test_fetch_live_claims_treats_empty_liveness_as_live(monkeypatch: Any) -> None:
    payload = {"owner_session": "owner-a", "owner_liveness": {}}
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: _proc(json.dumps(payload)))

    assert cli.fetch_live_claims("synaptent/aragora", [_cand(1)]) == {1: "owner-a"}


def test_fetch_live_claims_reclaims_explicit_stale_owner(monkeypatch: Any) -> None:
    payload = {"owner_session": "owner-a", "owner_liveness": {"assessed": "stale"}}
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: _proc(json.dumps(payload)))

    assert cli.fetch_live_claims("synaptent/aragora", [_cand(1)]) == {}


def test_fetch_live_claims_allows_explicit_absent_owner(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: _proc(
            returncode=1, stderr="ERROR: no lane matched criteria {'pr': 1}"
        ),
    )

    assert cli.fetch_live_claims("synaptent/aragora", [_cand(1)]) == {}


def test_fetch_live_claims_timeout_fails_closed_with_reason(monkeypatch: Any) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd=["identify_lane_owner.py"], timeout=60)

    monkeypatch.setattr(cli.subprocess, "run", boom)

    claims = cli.fetch_live_claims("synaptent/aragora", [_cand(1)])

    assert claims[1].startswith("owner-liveness-unavailable:")
    assert "timed out" in claims[1]


def test_fetch_live_claims_oserror_fails_closed_with_reason(monkeypatch: Any) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError("python missing")

    monkeypatch.setattr(cli.subprocess, "run", boom)

    claims = cli.fetch_live_claims("synaptent/aragora", [_cand(1)])

    assert claims[1].startswith("owner-liveness-unavailable:")
    assert "python missing" in claims[1]


def test_fetch_live_claims_broken_probe_fails_closed_with_stderr(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: _proc(returncode=2, stderr="ImportError: schema drift"),
    )

    claims = cli.fetch_live_claims("synaptent/aragora", [_cand(1)])

    assert claims[1].startswith("owner-liveness-unavailable:")
    assert "ImportError" in claims[1]
