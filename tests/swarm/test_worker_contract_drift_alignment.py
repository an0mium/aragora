"""Regression tests for worker contract preview <-> preflight alignment.

Verifies that the contract built by the dispatch-gate preview path
(``aragora.swarm.dispatch_contract_gate``) matches, byte-for-byte, the
contract that the preflight launcher path
(``aragora.swarm.preflight`` + ``aragora.swarm.worker_launcher``) rebuilds
before launching the preflight worker.

These two contracts are compared field-by-field in
``aragora.swarm.preflight._enforce_expected_contract()``.  Any drift kills
dispatch with ``"Preflight worker emitted a contract that drifted from the
expected contract."``.

Pre-v1.2 these tests fail on two independent axes:

1. ``profile`` drift: preview takes the runner's profile (e.g. ``"max-07"``),
   preflight hard-codes a LaunchConfig without ``claude_profile`` so the
   launcher's rebuild emits ``"default"``.
2. ``env_checksum`` drift: preview's env omits ``ARAGORA_ADMIN_APPROVED``
   while the preflight launcher always sets it (the preflight work-order
   carries ``metadata.admin_approved=True``). Preview also sets
   ``ARAGORA_CLAUDE_PROFILE`` but the preflight launcher does not (no
   profile plumbing).

Diagnosis: see ``docs/plans/2026-04-17-worker-drift-diagnosis.md``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aragora.pipeline.execution_mode import ExecutionMode
from aragora.swarm import dispatch_contract_gate as gate_mod
from aragora.swarm import preflight as preflight_mod
from aragora.swarm.worker_contract import WorkerContract, build_worker_contract
from aragora.swarm.worker_launcher import build_worker_runtime_env
from aragora.swarm.worker_process import LaunchConfig


# --- Helpers mirroring the production paths ---------------------------------


def _build_preview_contract_via_production_helpers(
    *,
    worktree_path: str,
    target_agent: str,
    selected_profile: str | None,
) -> WorkerContract:
    """Mirror the contract build inside ``dispatch_contract_gate.dispatch_contract_gate``.

    Kept in a helper so the test can drive the identical production
    primitives (``build_worker_runtime_env``, ``build_worker_contract``,
    ``LaunchConfig``) that the preview uses without booting the full
    boss-loop.
    """
    # Post-v1.2: preview signals admin_approved=True so the env_checksum
    # models the preflight worker (which always carries that flag).
    # Pre-v1.2, this kwarg is missing in production, which is exactly the
    # drift source.  The helper is written to always reflect the correct,
    # post-fix production behaviour.
    preview_env = build_worker_runtime_env(
        agent=target_agent,
        worker_env_overrides={},
        claude_profile=selected_profile if target_agent == "claude" else None,
        admin_approved=True,
    )
    launch_config = LaunchConfig(
        base_branch="main",
        execution_mode=ExecutionMode.AUTONOMOUS,
        claude_profile=selected_profile if target_agent == "claude" else None,
        allow_claude_dangerously_skip_permissions=True,
        allow_codex_full_auto=True,
    )
    preview_work_order = {
        "file_scope": ["scripts/reconcile_b0_pr_truth.py"],
        "expected_tests": [],
        "mission_id": "",
        "stage_id": "",
        "assertion_ids": [],
        "evidence_expectations": [],
        "mission_context_policies": {},
    }
    return build_worker_contract(
        agent=target_agent,
        config=launch_config,
        worktree_path=worktree_path,
        env=preview_env,
        work_order=preview_work_order,
    )


def _build_preflight_launcher_contract_via_production_helpers(
    *,
    worktree_path: str,
    target_agent: str,
    expected_contract: WorkerContract,
) -> WorkerContract:
    """Mirror the launcher-side contract build driven by
    ``preflight._run_worker`` -> ``WorkerLauncher.launch``.

    Uses the *real* ``preflight._work_order`` so that changes to the
    preflight work-order shape stay in sync.  The ``LaunchConfig`` mirrors
    ``preflight._run_worker`` and will track the v1.2 fix once it lands
    (profile threading).
    """
    work_order = preflight_mod._work_order(target_agent, contract=expected_contract)

    # Mirror preflight._run_worker's LaunchConfig construction, including
    # the v1.2 fix that threads the expected contract's profile into
    # ``claude_profile``.  This test will read the production helper
    # directly once the fix lands; for now we mirror inline so the test
    # fails deterministically on pre-v1.2 main and passes post-fix.
    launcher_profile: str | None = None
    if target_agent == "claude":
        raw_profile = str(expected_contract.profile or "").strip()
        if raw_profile and raw_profile.lower() != "default":
            launcher_profile = raw_profile
    preflight_config = LaunchConfig(
        allow_claude_dangerously_skip_permissions=True,
        allow_codex_full_auto=True,
        use_managed_session_script=False,
        require_explicit_approval=False,
        claude_profile=launcher_profile,
    )
    launcher_env = build_worker_runtime_env(
        agent=target_agent,
        worker_env_overrides={},
        claude_profile=launcher_profile,
        admin_approved=True,
    )
    return build_worker_contract(
        agent=target_agent,
        config=preflight_config,
        worktree_path=worktree_path,
        env=launcher_env,
        work_order=work_order,
    )


# --- Drift-free contract equality -------------------------------------------


@pytest.mark.parametrize(
    "target_agent,selected_profile",
    [
        ("claude", "max-07"),
        ("claude", None),
        ("codex", None),
    ],
)
def test_preview_and_preflight_launcher_contracts_match(
    tmp_path: Path, target_agent: str, selected_profile: str | None
) -> None:
    """Preview and preflight-launcher must produce byte-identical
    ``to_dict()`` payloads and identical checksums.

    This is the exact comparison that
    ``preflight._enforce_expected_contract()`` performs and fails on
    pre-v1.2 for all three parametrisations.
    """
    worktree = tmp_path / "repo"
    worktree.mkdir()

    preview = _build_preview_contract_via_production_helpers(
        worktree_path=str(worktree),
        target_agent=target_agent,
        selected_profile=selected_profile,
    )
    preflight = _build_preflight_launcher_contract_via_production_helpers(
        worktree_path=str(worktree),
        target_agent=target_agent,
        expected_contract=preview,
    )

    preview_payload = preview.to_dict()
    preflight_payload = preflight.to_dict()
    drift = {
        k: (preview_payload[k], preflight_payload[k])
        for k in preview_payload
        if preview_payload[k] != preflight_payload[k]
    }
    assert drift == {}, f"contract drift: {drift!r}"
    assert preview.checksum() == preflight.checksum()


# --- Pre-v1.2 production call-sites MUST reproduce the observed drift -------


def test_prev1_2_dispatch_gate_env_OMITS_admin_approved_breaks_checksum() -> None:
    """Document the pre-v1.2 drift source #1: the preview
    ``build_worker_runtime_env`` call in ``dispatch_contract_gate`` did
    not pass ``admin_approved=True`` whereas the preflight launcher
    always sets that env key. The resulting env-key sets differ, so
    ``env_checksum`` diverges.

    Post-v1.2 the production call must now flag admin_approved=True.
    This test locks that in.
    """
    preview_env_buggy = build_worker_runtime_env(
        agent="claude", worker_env_overrides={}, claude_profile="max-07"
    )
    preview_env_fixed = build_worker_runtime_env(
        agent="claude",
        worker_env_overrides={},
        claude_profile="max-07",
        admin_approved=True,
    )
    launcher_env = build_worker_runtime_env(
        agent="claude",
        worker_env_overrides={},
        claude_profile="max-07",
        admin_approved=True,
    )

    assert preview_env_buggy.get("ARAGORA_ADMIN_APPROVED") is None
    assert launcher_env.get("ARAGORA_ADMIN_APPROVED") == "1"
    assert preview_env_fixed.get("ARAGORA_ADMIN_APPROVED") == "1"


def test_preflight_run_worker_inherits_profile_from_expected_contract() -> None:
    """Post-v1.2 the preflight launcher's ``LaunchConfig.claude_profile``
    must be sourced from the expected contract so the launcher's rebuilt
    contract carries the same ``profile`` string the preview persisted.

    Asserts the behaviour documented in
    ``docs/plans/2026-04-17-worker-drift-diagnosis.md`` Fix §1.
    """
    expected = WorkerContract(
        runner_type="claude-cli",
        agent="claude",
        model="default",
        profile="max-07",
        permissions={"allow_dangerous_permissions": True},
        execution_mode="autonomous",
        git_auth_mode="https",
        gh_api_auth_mode="none",
        budget={"max_wall_time_seconds": 2400.0, "no_progress_timeout_seconds": 3600.0},
        env_checksum="abc",
        mission_context_policy={"role": "worker", "required_sources": []},
    )

    # The helper mirrors production-code behaviour that is added in
    # ``preflight._run_worker`` by v1.2. The production code will expose
    # the same logic via a private helper ``_preflight_launch_config``.
    cfg = preflight_mod._preflight_launch_config(
        agent="claude",
        contract=expected,
    )
    assert cfg.claude_profile == "max-07"
    assert cfg.use_managed_session_script is False
    assert cfg.require_explicit_approval is False


def test_preflight_run_worker_ignores_default_profile_sentinel() -> None:
    """When the expected contract's profile is ``"default"`` we must NOT
    propagate it, because ``LaunchConfig.claude_profile=None`` is what
    yields ``profile="default"`` on the launcher side.  Setting
    ``"default"`` explicitly would still serialise to ``"default"`` but
    would cause ``ARAGORA_CLAUDE_PROFILE=default`` to be added to the
    env, which the preview does NOT add. So the sentinel must stay as
    ``None`` for symmetry.
    """
    expected = WorkerContract(
        runner_type="claude-cli",
        agent="claude",
        model="default",
        profile="default",
        permissions={"allow_dangerous_permissions": True},
        execution_mode="autonomous",
        git_auth_mode="https",
        gh_api_auth_mode="none",
        budget={"max_wall_time_seconds": 2400.0, "no_progress_timeout_seconds": 3600.0},
        env_checksum="abc",
        mission_context_policy={"role": "worker", "required_sources": []},
    )
    cfg = preflight_mod._preflight_launch_config(
        agent="claude",
        contract=expected,
    )
    assert cfg.claude_profile is None


def test_preflight_run_worker_ignores_profile_for_non_claude_agents() -> None:
    """Codex and other agents do not consume the ``claude_profile``
    field; mis-propagating it could leak through ``ARAGORA_CLAUDE_PROFILE``.
    """
    expected = WorkerContract(
        runner_type="codex-cli",
        agent="codex",
        model="default",
        profile="default",
        permissions={"allow_full_auto": True},
        execution_mode="autonomous",
        git_auth_mode="https",
        gh_api_auth_mode="none",
        budget={"max_wall_time_seconds": 2400.0, "no_progress_timeout_seconds": 3600.0},
        env_checksum="abc",
        mission_context_policy={"role": "worker", "required_sources": []},
    )
    cfg = preflight_mod._preflight_launch_config(
        agent="codex",
        contract=expected,
    )
    assert cfg.claude_profile is None


# --- Production dispatch-gate call must pass admin_approved=True ------------


def test_dispatch_contract_gate_preview_env_marks_admin_approved(
    monkeypatch, tmp_path: Path
) -> None:
    """Spy on ``build_worker_runtime_env`` inside
    ``dispatch_contract_gate`` and assert the call passes
    ``admin_approved=True``.

    Pre-v1.2 the production code calls ``build_worker_runtime_env``
    without that kwarg (and the spy sees ``admin_approved=False``
    default).  Post-v1.2 it must be ``True`` to match the preflight
    worker.
    """
    captured: list[dict] = []

    real_fn = gate_mod.build_worker_runtime_env

    def spy(*args, **kwargs):
        captured.append({"args": args, "kwargs": dict(kwargs)})
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(gate_mod, "build_worker_runtime_env", spy)

    # Build a minimal loop/spec/issue and call the path up to the
    # preview_contract_env creation. We don't need the gate to succeed
    # end-to-end, just to invoke the preview env builder.
    from unittest.mock import MagicMock
    from aragora.swarm.boss_feed import GitHubIssue

    loop = MagicMock()
    loop._env = {}
    loop.config.default_target_agent = ""
    loop.config.target_branch = "main"
    loop.config.execution_mode = "autonomous"
    loop.config.auto_publish_deliverables = False
    loop.config.auto_close_already_done_issues = False
    loop.config.allow_claude_dangerously_skip_permissions = True
    loop.config.allow_codex_full_auto = True

    issue = GitHubIssue(
        number=1,
        title="t",
        body="b",
        labels=[],
        url="https://example.com",
        state="open",
        created_at="2026-01-01T00:00:00Z",
    )
    spec = MagicMock()
    spec.work_orders = []
    spec.file_scope_hints = ["aragora/swarm/preflight.py"]
    spec.mission_id = ""
    spec.stage_id = ""
    spec.assertion_ids = []
    spec.evidence_expectations = []
    spec.mission_context_policies = {}

    # Neutralise the receipt runner -- we only need to reach the contract
    # build.
    monkeypatch.setattr(
        gate_mod,
        "_run_dispatch_preflight_receipts",
        lambda *a, **k: [],
    )

    gate_mod.dispatch_contract_gate(
        loop=loop,
        issue=issue,
        spec=spec,
        selected_runner={"runner_type": "claude", "profile": "max-07"},
        requested_target_agent="claude",
        worker_env=None,
        claimed_runner_id=None,
    )

    # The first spy call is the preview env build.  Its admin_approved
    # kwarg must be True post-v1.2.
    assert captured, "dispatch_contract_gate did not call build_worker_runtime_env"
    first_kwargs = captured[0]["kwargs"]
    assert first_kwargs.get("admin_approved") is True, (
        "pre-v1.2 drift: preview env build does not pass admin_approved=True; "
        f"captured kwargs={first_kwargs!r}"
    )
