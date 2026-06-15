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


def _wo_42() -> lc.WorkOrderSpec:
    return lc.build_work_orders(
        [LaneAssignment(pr=42, branch="codex/x", owner_session="sess-42")],
        repo="synaptent/aragora",
        now=_fixed_now,
    )[0]


def test_default_claim_is_plain_no_force(tmp_path: Path, monkeypatch: Any) -> None:
    # A clean claim succeeds on the first try with NO --force/--allow-resource-
    # conflicts: claim_lane's conflict check then provides mutual exclusion, so a
    # competing live claim correctly fails instead of being clobbered.
    wo = _wo_42()
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lc.subprocess, "run", fake_run)

    assert lc.default_claim(wo, repo_root=tmp_path) is True
    assert len(calls) == 1  # one plain claim, no probe/release/retry
    cmd = calls[0]
    assert cmd[cmd.index("--lane-id") + 1] == wo.work_order_id
    assert cmd[cmd.index("--owner-session") + 1] == "sess-42"
    assert "--release-stale" not in cmd
    assert "--force" not in cmd
    assert "--allow-resource-conflicts" not in cmd


def test_default_claim_releases_stale_then_retries(tmp_path: Path, monkeypatch: Any) -> None:
    # Claim refused (stale-but-active row holds the resource). The probe assesses
    # the owner stale, so we release it and retry -- without --force.
    wo = _wo_42()
    seq: list[str] = []
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        calls.append(cmd)
        script = next((c for c in cmd if c.endswith(".py")), "")
        if script.endswith("identify_lane_owner.py"):
            seq.append("probe")
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"owner_session": "stale-owner", "owner_liveness": {"assessed": "stale"}}
                ),
                stderr="",
            )
        if "--release-stale" in cmd:
            seq.append("release")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        # plain claim: fail first, succeed on retry
        seq.append("claim")
        return SimpleNamespace(returncode=0 if seq.count("claim") > 1 else 2, stdout="", stderr="x")

    monkeypatch.setattr(lc.subprocess, "run", fake_run)
    assert lc.default_claim(wo, repo_root=tmp_path) is True
    assert seq == ["claim", "probe", "release", "claim"]
    release_cmd = next(cmd for cmd in calls if "--release-stale" in cmd)
    assert release_cmd[release_cmd.index("--pr-number") + 1] == str(wo.pr)


def test_default_claim_does_not_release_live_owner(tmp_path: Path, monkeypatch: Any) -> None:
    # Claim refused and the owner is LIVE -> never release/displace; claim fails.
    wo = _wo_42()
    released: list[str] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        script = next((c for c in cmd if c.endswith(".py")), "")
        if script.endswith("identify_lane_owner.py"):
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"owner_session": "live-owner", "owner_liveness": {"assessed": "live"}}
                ),
                stderr="",
            )
        if "--release-stale" in cmd:
            released.append("released")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=2, stdout="", stderr="conflict")  # claim always refused

    monkeypatch.setattr(lc.subprocess, "run", fake_run)
    assert lc.default_claim(wo, repo_root=tmp_path) is False
    assert released == []  # a live owner is never released


def test_run_pass_releases_lane_when_dispatch_fails() -> None:
    # Claimed, then dispatch raises: the lane must be released so it isn't wedged
    # (claimed with no pending order) until TTL/manual cleanup.
    released: list[int] = []

    def boom_dispatch(wo: lc.WorkOrderSpec) -> str:
        raise RuntimeError("disk full")

    result = lc.run_pass(
        repo="r",
        fetch_candidates=lambda repo: [_cand(1)],
        fetch_live_claims=lambda repo, cands: {},
        max_workers=5,
        execute=True,
        claim_fn=lambda wo: True,
        dispatch_fn=boom_dispatch,
        release_fn=lambda wo: released.append(wo.pr),
        session_id_for=_seq_session,
        now=_fixed_now,
    )
    assert released == [1]
    assert result.claim_failed == [1]
    assert result.dispatched == []


def test_default_release_scopes_to_work_order_lane_and_pr(tmp_path: Path, monkeypatch: Any) -> None:
    wo = _wo_42()
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lc.subprocess, "run", fake_run)

    lc.default_release(wo, repo_root=tmp_path)

    assert len(calls) == 1
    cmd = calls[0]
    assert "--release-stale" in cmd
    assert cmd[cmd.index("--lane-id") + 1] == wo.work_order_id
    assert cmd[cmd.index("--owner-session") + 1] == wo.owner_session
    assert cmd[cmd.index("--pr-number") + 1] == str(wo.pr)


def test_fetch_candidates_skips_cross_repo_drafts_and_unblocked(monkeypatch: Any) -> None:
    rows = [
        {
            "number": 1,
            "headRefName": "b1",
            "isDraft": False,
            "mergeStateStatus": "BLOCKED",
            "isCrossRepository": False,
        },
        # fork PR: head branch is not an origin ref -> a lane can't service it.
        {
            "number": 2,
            "headRefName": "b2",
            "isDraft": False,
            "mergeStateStatus": "BLOCKED",
            "isCrossRepository": True,
        },
        {
            "number": 3,
            "headRefName": "b3",
            "isDraft": True,
            "mergeStateStatus": "BLOCKED",
            "isCrossRepository": False,
        },
        {
            "number": 4,
            "headRefName": "b4",
            "isDraft": False,
            "mergeStateStatus": "CLEAN",
            "isCrossRepository": False,
        },
        {
            "number": 5,
            "headRefName": "",
            "isDraft": False,
            "mergeStateStatus": "BLOCKED",
            "isCrossRepository": False,
        },
    ]
    monkeypatch.setattr(cli, "_gh_json", lambda args: rows)
    cands = cli.fetch_candidates("synaptent/aragora")
    assert [c["number"] for c in cands] == [1]  # fork(2), draft(3), clean(4), empty(5) excluded


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
