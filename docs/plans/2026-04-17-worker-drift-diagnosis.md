# Worker Contract Drift — Phase 1 Diagnosis (2026-04-17)

**Status:** Diagnosis complete. **Go** for Phase 2 fix (scope is tight).

**Operator context:** Boss loop v1.1 (with Seam A + Seam B spec-upgrader wiring)
fails to dispatch every issue because every preflight run trips
`_enforce_expected_contract()` with
`contract_preflight: Preflight worker emitted a contract that drifted from the
expected contract.` Telemetry (`boss_metrics.jsonl`) shows this for
`#5844`, `#5887`, `#5893`, `#5894`, `#5895`, `#5897`, `#5899`, `#5962` — i.e. it
is **universal and deterministic**, not noise.

---

## 1. What "drift" actually means here

The term "drift" in
`aragora/swarm/preflight.py::_enforce_expected_contract()` is misleading.
The **"actual" contract is not emitted by the worker process**. It is
rebuilt by `WorkerLauncher.launch()` before the child process is even
spawned — see
`aragora/swarm/worker_launcher.py:190-204` — and then stored on
`WorkerProcess.worker_contract`.

So the check compares two contracts that are both built by Aragora itself:

| side | code path | `build_worker_contract()` inputs |
|------|-----------|-----------------------------------|
| **expected** (persisted to `.aragora/dispatch_contracts/...`) | `dispatch_contract_gate.dispatch_contract_gate()` | preview `LaunchConfig` + preview work-order (spec-derived) + `preview_contract_env` |
| **actual** (set on the preflight `WorkerProcess`) | `WorkerLauncher.launch()` via `preflight._run_worker()` | preflight-owned `LaunchConfig` + preflight scratch work-order + launcher-built `worker_env` |

Both paths call the same `build_worker_contract()` helper, so any
divergence is entirely driven by **input divergence** between preview and
launcher.

## 2. Reproduction (deterministic, no live boss loop required)

Harness: `build_worker_contract()` called twice with the exact inputs each
side uses in production. Running it locally against `main` @
`bfd6aec2d`:

```
DRIFTED FIELDS:
  profile:
    preview:   'max-07'
    preflight: 'default'
  env_checksum:
    preview:   '4a634c78c3ceda6722a2f7d698388649d42c3455261716b47f79ab716f1ce94e'
    preflight: '1370c21d9bdee78f516ac31c706be2a7c04acdeaef572e64ec3f47c0847157c9'
```

The checksums at the top level diverge because `to_dict()` diverges on
two fields: `profile` and `env_checksum`. The exact same result can be
read out of the persisted contract for `#5899`
(`.aragora/dispatch_contracts/issue-5899-f7cb747c14af.json`):

- `profile: "max-07"` — set by the preview from `selected_runner.profile`.
- `env_checksum: 8a33d708...` — computed from the preview env, which
  includes `ARAGORA_CLAUDE_PROFILE=max-07` and **does not** include
  `ARAGORA_ADMIN_APPROVED`.

Both drifts are consistent across runs (no LLM variance), consistent
across issues (universal), and consistent between Claude and Codex
dispatch attempts (same drift shape for both agents in the logs).

## 3. Mechanism — where each field diverges

### 3.1 `profile` field

In `dispatch_contract_gate.py` the preview `LaunchConfig` carries the
selected runner's profile:

```python
launch_config = LaunchConfig(
    ...
    claude_profile=(
        str((selected_runner or {}).get("profile", "")).strip() or None
        if target_agent == "claude" else None
    ),
    ...
)
```

`build_worker_contract()` then serialises that as
`profile = config.claude_profile or "default"` → `"max-07"`.

In `preflight._run_worker()` the preflight-owned `LaunchConfig` is built
from scratch and **never receives** `contract.profile`:

```python
config = LaunchConfig(
    allow_claude_dangerously_skip_permissions=True,
    allow_codex_full_auto=True,
    use_managed_session_script=False,
    require_explicit_approval=False,
)
launcher = WorkerLauncher(config=config)
```

So `self.config.claude_profile` is `None` → `profile = "default"` in the
launcher-built contract. This is the first drift source.

### 3.2 `env_checksum` field

`env_checksum` is a SHA256 over all env keys starting with `ARAGORA_`,
`CLAUDE_`, `CODEX_`, `GH_`, `GITHUB_` (`worker_contract._env_checksum`).

Two independent inputs to that env differ between preview and launcher:

1. **`ARAGORA_CLAUDE_PROFILE`** — the preview sets this (via
   `build_worker_runtime_env(..., claude_profile="max-07")`), but the
   launcher does **not** (its `self.config.claude_profile` is `None`
   because the preflight `LaunchConfig` was built without it).
2. **`ARAGORA_ADMIN_APPROVED`** — the preview does **not** set this
   (`build_worker_runtime_env(...)` is called without `admin_approved`
   kwarg), but the preflight work-order sets
   `metadata["admin_approved"] = True`, so the launcher's
   `build_worker_runtime_env(..., admin_approved=True)` writes
   `ARAGORA_ADMIN_APPROVED=1` into the worker env.

Either of these alone is enough to make `env_checksum` diverge.

## 4. Root cause classification

**Derivation bug** — category 2 from the mission brief.

The "expected" contract is derived with a
`LaunchConfig`/`work_order`/`env` triple that the launcher provably will
not reproduce. No worker prompt is involved; no protocol format
mismatch; both sides use the same `to_dict()`. The bug is that two
different call-sites of `build_worker_contract()` are fed structurally
different inputs for the same logical dispatch.

Evidence rule-out:

- Not **prompt**: the worker process never emits the contract; the
  launcher builds it deterministically before spawning the process.
- Not **protocol**: both sides produce `WorkerContract.to_dict()` using
  the same class, sorting, and stringification. Fields that match
  (runner_type, agent, model, permissions, execution_mode,
  git_auth_mode, gh_api_auth_mode, budget, mission_context_policy,
  lineage, contract_version) prove the format is identical.
- **Derivation**: the inputs to the second call differ in two
  identifiable places (§3.1, §3.2).

## 5. Fix strategy (Phase 2, minimum scope)

Minimum fix to make `#5899` / `#5895` dispatches succeed (or fail for a
non-drift reason): align the preview and launcher inputs in exactly the
two places where they currently differ.

1. **Propagate `claude_profile` into the preflight launcher.** In
   `preflight._run_worker()`, when an `expected_contract` is supplied and
   the agent is `claude`, thread
   `contract.profile` → `LaunchConfig.claude_profile`. This single change
   fixes both the `profile` drift and the `ARAGORA_CLAUDE_PROFILE`
   component of the `env_checksum` drift, because the launcher's
   `build_worker_runtime_env()` call already consumes
   `self.config.claude_profile`.

2. **Mark the preview env as admin-approved.** In
   `dispatch_contract_gate.dispatch_contract_gate()` where
   `preview_contract_env` is built, pass `admin_approved=True` to
   `build_worker_runtime_env()`. This is semantically correct: the
   preview is specifically modelling the contract that the admin-owned
   preflight will produce, and the preflight work-order always carries
   `metadata.admin_approved=True`. This fixes the remaining
   `env_checksum` drift.

Net diff: ~6 lines of production code across two files, plus tests.

### Tests to add

- **Unit**: two `WorkerContract` instances built from the preview path and
  preflight-launcher path with the same inputs must have
  `to_dict()` equal and identical checksums. Parametrise over `claude` /
  `codex` and over `None` / `"max-07"` profiles.
- **Regression**: load
  `.aragora/dispatch_contracts/issue-5899-f7cb747c14af.json`
  (frozen as a fixture), build the matching preflight-side contract
  using the new code paths, and assert drift-free.

Both tests must **fail** on `main` @ `bfd6aec2d` and pass after the fix.

## 6. Scope-boundedness for one PR

**Go.** The fix is two small alignments in two files, plus focused unit
tests. It does not require changes to:

- `WorkerContract` shape, checksum algorithm, or serialisation.
- `_enforce_expected_contract()` itself.
- Spec upgrader (Seam A / Seam B).
- Worker prompt building, worker launcher command lines, or any
  subprocess behaviour.

## 7. Residual open questions (out of scope for v1.2)

- Should `admin_approved` participate in the contract checksum at all?
  It's a trust-boundary flag, not a runtime-determinism knob. Candidate
  follow-up: carve it out of `env_checksum` so that future
  admin/non-admin divergence can't reintroduce this drift.
- Should `_enforce_expected_contract()` diff two contracts field-by-field
  and surface the drifting fields in the error message? The current
  monolithic error string is what made this investigation slow. A
  follow-up that writes a drift diff to
  `.aragora/overnight/contract_drift_diagnostics.jsonl` would have
  saved days of boss-loop churn.

## 8. Decision

Proceed to Phase 2 on branch `fix/worker-contract-drift-v1-2`.
