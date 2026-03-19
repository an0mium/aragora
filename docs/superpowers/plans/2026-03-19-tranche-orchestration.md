# Tranche Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable prompt-driven multi-lane execution through the Tranche orchestration spine — from vague human intake through bounded pre-implementation design review, adaptive review, and PR integration with durable watch/reattach.

**Architecture:** Tranche is the orchestration spine and lifecycle state machine. Campaign provides decomposition and cross-model review. Boss-loop stays the bounded dispatch engine. DevCoordinationStore provides durable state for leases, receipts, and integration decisions.

**Tech Stack:** Python 3.11+, asyncio, PyYAML, `gh` CLI for GitHub integration, existing aragora.swarm and aragora.nomic primitives.

**Spec:** `docs/superpowers/specs/2026-03-19-tranche-orchestration-design.md`

---

## File Structure

### New files
- `aragora/swarm/tranche_submit.py` — Submit pipeline: validate, enrich, decompose, normalize, compile, inspect, persist
- `aragora/swarm/tranche_design_review.py` — Bounded proposer/critic/synthesizer loop over normalized bundle + inspected manifest
- `aragora/swarm/tranche_review.py` — Adaptive review: tier selection, multi-reviewer consensus, bounded retry
- `aragora/swarm/tranche_integrate.py` — PR discovery, check classification, merge recommendation/execution
- `aragora/swarm/tranche_state.py` — TrancheRunState, LaneRunState dataclasses + persistence
- `aragora/swarm/tranche_watch.py` — Watch loop: observer/driver modes, state refresh, autonomous advancement
- `tests/swarm/test_tranche_submit.py`
- `tests/swarm/test_tranche_design_review.py`
- `tests/swarm/test_tranche_review.py`
- `tests/swarm/test_tranche_integrate.py`
- `tests/swarm/test_tranche_state.py`
- `tests/swarm/test_tranche_watch.py`

### Modified files
- `aragora/swarm/tranche.py` — Add receipt_id/lease_id persistence in run artifacts, import new modules
- `aragora/swarm/__init__.py` — Export new public names
- `aragora/cli/commands/swarm.py` — Wire submit/review/integrate/watch/list CLI actions
- `aragora/cli/parser.py` — Add new parser arguments
- `tests/swarm/test_tranche.py` — Extend with receipt_id/lease_id round-trip tests

### Reference files (read-only)
- `aragora/swarm/campaign.py` — CampaignPlanner, CampaignReviewer, CampaignReviewGate, CampaignProject
- `aragora/swarm/boss_loop.py` — dispatch_bounded_spec
- `aragora/nomic/dev_coordination.py` — DevCoordinationStore, WorkLease, CompletionReceipt, IntegrationDecision
- `aragora/swarm/pr_registry.py` — PullRequestRegistry
- `aragora/ralph/github_control.py` — Check partitioning logic

---

## Phase 0: Prerequisites

### Task 0.1: Land the tranche-plan-prepare-run branch

**Files:**
- Review: `aragora/swarm/tranche.py` (on branch `codex/tranche-plan-prepare-run`)

This is external to the plan — it must be merged before any task below begins. Verify by:

- [ ] **Step 1: Confirm branch is merged to main**

Run: `git log --oneline origin/main -5 | grep tranche`
Expected: A commit containing plan/prepare/run tranche work

- [ ] **Step 2: Verify plan/prepare/run CLI works**

Run: `python3 -m aragora.cli.main swarm tranche plan --help`
Expected: Help text showing `--bundle` or `--from-prompts` argument

### Task 0.2: Persist receipt_id and lease_id in run artifacts

**Files:**
- Modify: `aragora/swarm/tranche.py` (the `_artifact_from_run_result` method)
- Test: `tests/swarm/test_tranche.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/swarm/test_tranche.py

@pytest.mark.asyncio
async def test_run_artifact_persists_receipt_and_lease_ids():
    """run must write receipt_id and lease_id into artifact metadata."""
    manifest = _make_manifest(lane_ids=["lane_a"])
    executor = TrancheExecutor(repo_root=Path("/tmp/test"))

    mock_result = {
        "status": "completed",
        "outcome": "deliverable_created",
        "run_id": "run-123",
        "run": {
            "run_id": "run-123",
            "status": "completed",
            "work_orders": [{
                "work_order_id": "wo-1",
                "lease_id": "lease-abc",
                "receipt_id": "receipt-xyz",
                "status": "completed",
                "worktree_path": "/tmp/wt",
            }],
        },
        "deliverable": {"type": "branch"},
    }

    with patch.object(executor, "_prepare_lane_workspace") as mock_prep:
        mock_prep.return_value = TrancheLaneArtifact(
            lane_id="lane_a", source_ref="", status="prepared",
        )
        artifact = await executor._artifact_from_run_result(
            manifest, manifest.lane("lane_a"),
            prepared=mock_prep.return_value,
            result=mock_result,
            review_model="claude",
            skip_review=True,
        )

    assert artifact.metadata.get("receipt_id") == "receipt-xyz"
    assert artifact.metadata.get("lease_id") == "lease-abc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/swarm/test_tranche.py::test_run_artifact_persists_receipt_and_lease_ids -v`
Expected: FAIL — receipt_id/lease_id not in metadata

- [ ] **Step 3: Implement in _artifact_from_run_result**

In `aragora/swarm/tranche.py`, inside `_artifact_from_run_result`, after `run_dict` extraction, add:

```python
# Extract receipt_id and lease_id from first work order
work_orders = run_dict.get("work_orders", [])
if work_orders and isinstance(work_orders[0], dict):
    first_wo = work_orders[0]
    if first_wo.get("receipt_id"):
        metadata["receipt_id"] = str(first_wo["receipt_id"])
    if first_wo.get("lease_id"):
        metadata["lease_id"] = str(first_wo["lease_id"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/swarm/test_tranche.py::test_run_artifact_persists_receipt_and_lease_ids -v`
Expected: PASS

- [ ] **Step 5: Run full tranche tests**

Run: `python3 -m pytest tests/swarm/test_tranche.py -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add aragora/swarm/tranche.py tests/swarm/test_tranche.py
git commit -m "fix(tranche): persist receipt_id and lease_id in run artifacts"
```

---

## Phase 1: Submit

### Task 1.1: TrancheRunState and LaneRunState dataclasses

**Files:**
- Create: `aragora/swarm/tranche_state.py`
- Test: `tests/swarm/test_tranche_state.py`

- [ ] **Step 1: Write failing tests for state round-trip**

```python
# tests/swarm/test_tranche_state.py
from aragora.swarm.tranche_state import TrancheRunState, LaneRunState

def test_tranche_run_state_round_trip():
    state = TrancheRunState(
        manifest_id="test-manifest",
        status="planned",
        autonomy_mode="adaptive",
    )
    state.lane_states["lane_a"] = LaneRunState(lane_id="lane_a", status="pending")
    d = state.to_dict()
    restored = TrancheRunState.from_dict(d)
    assert restored.manifest_id == "test-manifest"
    assert restored.status == "planned"
    assert restored.lane_states["lane_a"].status == "pending"

def test_lane_run_state_defaults():
    lane = LaneRunState(lane_id="x", status="pending")
    assert lane.run_id is None
    assert lane.receipt_id is None
    assert lane.lease_id is None
    assert lane.retry_count == 0

def test_tranche_run_state_persistence(tmp_path):
    state = TrancheRunState(
        manifest_id="persist-test",
        status="running",
        autonomy_mode="fire_and_forget",
    )
    path = tmp_path / "run_state.yaml"
    state.save(path)
    loaded = TrancheRunState.load(path)
    assert loaded.manifest_id == "persist-test"
    assert loaded.status == "running"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/swarm/test_tranche_state.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement TrancheRunState and LaneRunState**

Create `aragora/swarm/tranche_state.py` with:
- `LaneRunState` dataclass: lane_id, status, run_id, receipt_id, lease_id, worktree_path, pr_url, retry_count, last_updated
- `TrancheRunState` dataclass: manifest_id, status, autonomy_mode, created_at, updated_at, lane_states dict, driver_session, driver_heartbeat, session_history
- `to_dict()` / `from_dict()` / `save(path)` / `load(path)` methods
- Status constants as module-level strings (not an Enum — keep it simple)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/swarm/test_tranche_state.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/tranche_state.py tests/swarm/test_tranche_state.py
git commit -m "feat(tranche): add TrancheRunState and LaneRunState dataclasses"
```

### Task 1.2: Source ref classification and enrichment

**Files:**
- Create: `aragora/swarm/tranche_submit.py`
- Test: `tests/swarm/test_tranche_submit.py`

- [ ] **Step 1: Write failing tests for ref classification**

```python
# tests/swarm/test_tranche_submit.py
from aragora.swarm.tranche_submit import classify_source_ref, enrich_github_refs

def test_classify_github_issue_ref():
    result = classify_source_ref("https://github.com/synaptent/aragora/issues/1064")
    assert result["kind"] == "github"
    assert result["github_kind"] == "issue"
    assert result["number"] == 1064

def test_classify_github_pr_ref():
    result = classify_source_ref("https://github.com/synaptent/aragora/pull/1065")
    assert result["kind"] == "github"
    assert result["github_kind"] == "pull_request"

def test_classify_local_file_ref():
    result = classify_source_ref("/path/to/local/file.md")
    assert result["kind"] == "context"
    assert result["gated"] is False

def test_classify_doc_url_ref():
    result = classify_source_ref("https://docs.example.com/guide")
    assert result["kind"] == "context"
    assert result["gated"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/swarm/test_tranche_submit.py -v`
Expected: FAIL

- [ ] **Step 3: Implement classify_source_ref and enrich_github_refs**

In `aragora/swarm/tranche_submit.py`:
- `classify_source_ref(url: str) -> dict` — parse GitHub URLs via `parse_github_reference_url()` from `aragora.swarm.tranche` (available after Task 0.1 merges), classify others as context-only
- `enrich_github_refs(refs: list[dict], client: GhReferenceClient) -> list[dict]` — resolve GitHub refs via `GhReferenceClient` from `aragora.swarm.tranche`, mark stale (closed issues, merged PRs with no open work), preserve context refs unchanged

**Important:** This task depends on Task 0.1. If the branch is not yet merged, these imports will fail. Confirm 0.1 is complete before starting.

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 -m pytest tests/swarm/test_tranche_submit.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/tranche_submit.py tests/swarm/test_tranche_submit.py
git commit -m "feat(tranche): add source ref classification and enrichment"
```

### Task 1.3: Decomposition triggers and lane normalization

**Files:**
- Modify: `aragora/swarm/tranche_submit.py`
- Test: `tests/swarm/test_tranche_submit.py`

- [ ] **Step 1: Write failing tests for decomposition triggers**

```python
def test_no_lanes_triggers_full_decomposition():
    bundle = {"objective": "Fix the user journey", "candidate_lanes": []}
    result = determine_decomposition_action(bundle)
    assert result == "full_decomposition"

def test_lane_missing_prompt_triggers_rebuild():
    bundle = {
        "objective": "Fix things",
        "candidate_lanes": [{"lane_id": "a", "title": "Fix auth"}],
    }
    result = determine_decomposition_action(bundle)
    assert result == "augment"

def test_lane_missing_scope_triggers_inference_only():
    bundle = {
        "objective": "Fix things",
        "candidate_lanes": [
            {"lane_id": "a", "title": "Fix auth", "prompt": "Fix the auth flow", "owner_role": "engineer"},
        ],
    }
    result = determine_decomposition_action(bundle)
    assert result == "inference_only"

def test_complete_lanes_skip_decomposition():
    bundle = {
        "objective": "Fix things",
        "candidate_lanes": [{
            "lane_id": "a", "title": "Fix auth", "prompt": "Fix auth",
            "owner_role": "engineer", "allowed_write_scope": ["aragora/auth/**"],
            "verification_commands": ["pytest tests/auth/"],
        }],
    }
    result = determine_decomposition_action(bundle)
    assert result == "none"
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/swarm/test_tranche_submit.py -v -k decomposition`
Expected: FAIL

- [ ] **Step 3: Implement determine_decomposition_action and normalize_lanes**

In `tranche_submit.py`:
- `determine_decomposition_action(bundle: dict) -> str` — returns "full_decomposition", "augment", "inference_only", or "none"
- `normalize_lanes(bundle: dict, planner: CampaignPlanner | None) -> list[dict]` — applies the decomposition action, generates lane_ids where missing, infers write scope from file-scope hints

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/swarm/test_tranche_submit.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/tranche_submit.py tests/swarm/test_tranche_submit.py
git commit -m "feat(tranche): add decomposition triggers and lane normalization"
```

### Task 1.4: Full submit pipeline

**Files:**
- Modify: `aragora/swarm/tranche_submit.py`
- Test: `tests/swarm/test_tranche_submit.py`

- [ ] **Step 1: Write failing test for end-to-end submit**

```python
def test_submit_returns_dual_status(tmp_path):
    bundle = {
        "objective": "Bump supabase in aragora/live",
        "candidate_lanes": [{
            "lane_id": "bump",
            "title": "Bump supabase",
            "prompt": "Bump @supabase/supabase-js to latest",
            "owner_role": "engineer",
            "allowed_write_scope": ["aragora/live/**"],
            "verification_commands": ["cd aragora/live && npm run lint"],
        }],
        "autonomy_mode": "adaptive",
    }

    result = submit_intake_bundle(
        bundle,
        repo_root=tmp_path,
        skip_github_resolution=True,
    )

    assert "inspection_status" in result
    assert "submission_status" in result
    assert "recommended_action" in result
    assert result["inspection_status"] in ("ok", "blocked")
    assert result["submission_status"] in ("ready_to_prepare", "awaiting_confirmation", "blocked")
    assert "manifest_id" in result

def test_submit_persists_three_layers(tmp_path):
    bundle = {"objective": "Test persistence", "candidate_lanes": []}

    result = submit_intake_bundle(
        bundle,
        repo_root=tmp_path,
        skip_github_resolution=True,
    )

    manifest_id = result["manifest_id"]
    tranche_dir = tmp_path / ".aragora" / "tranches" / manifest_id
    assert (tranche_dir / "intake_bundle.yaml").exists()
    assert (tranche_dir / "normalized_bundle.yaml").exists()
    assert (tranche_dir / "tranche.yaml").exists()
    assert (tranche_dir / "run_state.yaml").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/swarm/test_tranche_submit.py -v -k submit`
Expected: FAIL

- [ ] **Step 3: Implement submit_intake_bundle**

**Important:** This task depends on Task 0.1. `TranchePlanner` and `TrancheInspector` are on the branch, not main. Confirm 0.1 is complete before starting.

In `tranche_submit.py`, implement the full pipeline:
1. Validate (objective required)
2. Classify + enrich source refs
3. Determine decomposition action + normalize lanes
4. Compile via `TranchePlanner.plan_from_prompt_bundle()` (import from `aragora.swarm.tranche`)
5. Inspect via `TrancheInspector.inspect()` (import from `aragora.swarm.tranche`)
6. Create `TrancheRunState` with status derived from autonomy mode:
   - `checkpoint` or `spectator` → `submission_status = "awaiting_confirmation"`
   - `fire_and_forget` with `inspection_status == "ok"` → `submission_status = "ready_to_prepare"`
   - `adaptive` → derive from inspection result and risk tier (high confidence + ok → `ready_to_prepare`, otherwise `awaiting_confirmation`)
7. Persist all three layers + run state to `<repo_root>/.aragora/tranches/<manifest_id>/`
8. Return dual-status dict, with `recommended_action = "design-review"` for
   adaptive/checkpoint tranches that have writable lanes and passed inspection

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/swarm/test_tranche_submit.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/tranche_submit.py tests/swarm/test_tranche_submit.py
git commit -m "feat(tranche): implement full submit pipeline with three-layer persistence"
```

### Task 1.5: Wire submit CLI

**Files:**
- Modify: `aragora/cli/commands/swarm.py`
- Modify: `aragora/cli/parser.py`
- Test: `tests/cli/test_swarm_command.py`

- [ ] **Step 1: Write failing test for CLI wiring**

```python
def test_swarm_tranche_submit_action(capsys):
    with patch("aragora.swarm.tranche_submit.submit_intake_bundle") as mock_submit:
        mock_submit.return_value = {
            "inspection_status": "ok",
            "submission_status": "ready_to_prepare",
            "recommended_action": "Run swarm tranche run",
            "manifest_id": "test-123",
        }
        result = cmd_swarm({
            "action": "tranche",
            "tranche_action": "submit",
            "intake": "/tmp/bundle.yaml",
            "autonomy": "adaptive",
            "json": True,
        })
    assert '"submission_status"' in capsys.readouterr().out or result is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/cli/test_swarm_command.py -v -k submit`
Expected: FAIL

- [ ] **Step 3: Wire the CLI**

In `aragora/cli/parser.py`: add `--intake` and `--autonomy` arguments for the tranche subparser.
In `aragora/cli/commands/swarm.py`: add `submit` case in the tranche action handler that calls `submit_intake_bundle()`.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/cli/test_swarm_command.py -v -k submit`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aragora/cli/commands/swarm.py aragora/cli/parser.py tests/cli/test_swarm_command.py
git commit -m "feat(cli): wire swarm tranche submit command"
```

---

## Phase 1A: Design Review

### Task 1A.1: Bounded proposer/critic/synthesizer design review

**Files:**
- Create: `aragora/swarm/tranche_design_review.py`
- Test: `tests/swarm/test_tranche_design_review.py`

- [ ] **Step 1: Write failing tests for the bounded challenge loop**

```python
# tests/swarm/test_tranche_design_review.py
@pytest.mark.asyncio
async def test_design_review_runs_proposer_critic_and_synthesizer():
    proposer = AsyncMock(return_value={"proposal": {"objective": "ship it"}})
    critic = AsyncMock(return_value={"findings": ["Scope is too broad"]})
    synthesizer = AsyncMock(return_value={
        "recommendation": "awaiting_confirmation",
        "revised_manifest": {"manifest_id": "m1"},
        "unresolved_assumptions": ["Need narrower write scope"],
    })

    result = await run_design_review(
        manifest=_make_manifest(),
        normalized_bundle={"objective": "ship it"},
        inspection={"preflight_status": "ok"},
        proposer_fn=proposer,
        critic_fn=critic,
        synthesizer_fn=synthesizer,
        max_rounds=2,
    )

    assert result["recommendation"] == "awaiting_confirmation"
    proposer.assert_awaited_once()
    critic.assert_awaited_once()
    synthesizer.assert_awaited_once()

@pytest.mark.asyncio
async def test_design_review_stops_after_two_rounds():
    proposer = AsyncMock(side_effect=[
        {"proposal": {"round": 1}},
        {"proposal": {"round": 2}},
    ])
    critic = AsyncMock(side_effect=[
        {"findings": ["Issue 1"]},
        {"findings": ["Issue 2"]},
    ])
    synthesizer = AsyncMock(side_effect=[
        {"recommendation": "revise", "revised_manifest": {"round": 1}, "unresolved_assumptions": []},
        {"recommendation": "needs_human", "revised_manifest": {"round": 2}, "unresolved_assumptions": ["Still disputed"]},
    ])

    result = await run_design_review(
        manifest=_make_manifest(),
        normalized_bundle={"objective": "ship it"},
        inspection={"preflight_status": "ok"},
        proposer_fn=proposer,
        critic_fn=critic,
        synthesizer_fn=synthesizer,
        max_rounds=2,
    )

    assert result["rounds_completed"] == 2
    assert result["recommendation"] == "needs_human"
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/swarm/test_tranche_design_review.py -v`
Expected: FAIL

- [ ] **Step 3: Implement run_design_review and persisted review record**

Create `aragora/swarm/tranche_design_review.py` with:
- `DesignReviewRecord` dataclass: manifest_id, status, rounds, proposed_manifest, critique_findings, revised_manifest, unresolved_assumptions, created_at, updated_at
- `run_design_review(...) -> dict` — bounded max-2-round proposer/critic/synthesizer loop
- `save_design_review(path)` / `load_design_review(path)` helpers storing
  `design_review.yaml` beside the tranche manifest/run state
- hard rule: critique findings must be grounded in manifest/ref/repo state passed into the adapter

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/swarm/test_tranche_design_review.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/tranche_design_review.py tests/swarm/test_tranche_design_review.py
git commit -m "feat(tranche): add bounded pre-implementation design review"
```

### Task 1A.2: Wire design-review CLI and submit recommendation

**Files:**
- Modify: `aragora/swarm/tranche_submit.py`
- Modify: `aragora/cli/commands/swarm.py`
- Modify: `aragora/cli/parser.py`
- Test: `tests/swarm/test_tranche_submit.py`
- Test: `tests/cli/test_swarm_command.py`

- [ ] **Step 1: Write failing tests**

```python
def test_submit_recommends_design_review_for_adaptive_writable_tranche():
    result = submit_intake_bundle(
        {
            "objective": "Ship feature",
            "candidate_lanes": [{
                "lane_id": "lane_a",
                "title": "Build it",
                "owner_role": "engineer",
                "prompt": "Implement the feature",
                "allowed_write_scope": ["aragora/server/**"],
            }],
            "autonomy_mode": "adaptive",
        },
        repo_root=Path("/tmp/repo"),
        skip_github_resolution=True,
    )
    assert result["recommended_action"] == "design-review"

def test_swarm_tranche_design_review_action(capsys):
    with patch("aragora.swarm.tranche_design_review.run_design_review") as mock_run:
        mock_run.return_value = {"recommendation": "approved", "rounds_completed": 1}
        cmd_swarm({
            "action": "tranche",
            "tranche_action": "design-review",
            "manifest": "/tmp/tranche.yaml",
            "json": True,
        })
    assert '"recommendation"' in capsys.readouterr().out
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/swarm/test_tranche_submit.py tests/cli/test_swarm_command.py -v -k design_review`
Expected: FAIL

- [ ] **Step 3: Implement**

- add `design-review` tranche subcommand
- add `--rounds` parser argument
- make `submit` return `recommended_action = "design-review"` for adaptive/checkpoint
  tranches with writable lanes and `inspection_status == "ok"`

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/swarm/test_tranche_submit.py tests/cli/test_swarm_command.py -v -k design_review`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/tranche_submit.py aragora/cli/commands/swarm.py aragora/cli/parser.py tests/swarm/test_tranche_submit.py tests/cli/test_swarm_command.py
git commit -m "feat(cli): wire tranche design-review command and submit recommendation"
```

---

## Phase 2: Adaptive Review

### Task 2.1: Risk tier selector

**Files:**
- Create: `aragora/swarm/tranche_review.py`
- Test: `tests/swarm/test_tranche_review.py`

- [ ] **Step 1: Write failing tests for tier selection**

```python
# tests/swarm/test_tranche_review.py
from aragora.swarm.tranche_review import select_review_tier

def test_narrow_scope_gets_tier_1():
    tier = select_review_tier(
        write_scope=["aragora/live/package.json"],
        diff_lines=5,
        verification_passed=True,
        risk_tolerance=None,
    )
    assert tier == 1

def test_medium_scope_gets_tier_2():
    tier = select_review_tier(
        write_scope=["aragora/server/**", "aragora/api/**", "aragora/auth/**"],
        diff_lines=150,
        verification_passed=True,
        risk_tolerance=None,
    )
    assert tier == 2

def test_broad_scope_gets_tier_3():
    tier = select_review_tier(
        write_scope=["aragora/server/**", "aragora/api/**", "aragora/auth/**", "aragora/swarm/**"],
        diff_lines=500,
        verification_passed=False,
        risk_tolerance=None,
    )
    assert tier == 3

def test_explicit_risk_override():
    tier = select_review_tier(
        write_scope=["aragora/live/package.json"],
        diff_lines=2,
        verification_passed=True,
        risk_tolerance="high",
    )
    assert tier == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/swarm/test_tranche_review.py -v`
Expected: FAIL

- [ ] **Step 3: Implement select_review_tier**

Logic: scope breadth → lane type → verification result → explicit override → diff size as escalation.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/swarm/test_tranche_review.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/tranche_review.py tests/swarm/test_tranche_review.py
git commit -m "feat(tranche): add risk-based review tier selector"
```

### Task 2.2: Tier 1 review (single reviewer) as first-class command

**Files:**
- Modify: `aragora/swarm/tranche_review.py`
- Test: `tests/swarm/test_tranche_review.py`

- [ ] **Step 1: Write failing test**

```python
@pytest.mark.asyncio
async def test_tier_1_review_delegates_to_campaign_reviewer():
    lane = _make_lane(lane_id="a", write_scope=["aragora/live/**"])
    manifest = _make_manifest(lanes=[lane])
    artifact = _make_artifact(lane_id="a", status="completed", run_id="run-1")

    mock_reviewer = AsyncMock()
    mock_reviewer.review.return_value = CampaignReviewGate(
        status="passed", findings=[], review_model="claude",
    )

    result = await review_lane(
        manifest=manifest,
        lane_id="a",
        artifact=artifact,
        run_dict={"run_id": "run-1", "status": "completed", "work_orders": []},
        reviewer=mock_reviewer,
        tier=1,
    )

    assert result["status"] == "passed"
    mock_reviewer.review.assert_awaited_once()
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/swarm/test_tranche_review.py -v -k tier_1`
Expected: FAIL

- [ ] **Step 3: Implement review_lane with TrancheLane → CampaignProject adapter**

In `tranche_review.py`:
- `_adapt_lane_to_campaign_project(manifest, lane, artifact) -> CampaignProject` — thin projection
- `review_lane(manifest, lane_id, artifact, run_dict, reviewer, tier) -> dict` — dispatches to appropriate tier

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/swarm/test_tranche_review.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/tranche_review.py tests/swarm/test_tranche_review.py
git commit -m "feat(tranche): add first-class Tier 1 review with CampaignProject adapter"
```

### Task 2.3: Tier 2 multi-reviewer consensus and Tier 3 bounded retry

**Files:**
- Modify: `aragora/swarm/tranche_review.py`
- Test: `tests/swarm/test_tranche_review.py`

- [ ] **Step 1: Write failing tests for Tier 2 and Tier 3**

```python
@pytest.mark.asyncio
async def test_tier_2_runs_two_reviewers_and_synthesizes():
    lane = _make_lane(lane_id="a", write_scope=["aragora/server/**"])
    manifest = _make_manifest(lanes=[lane])
    artifact = _make_artifact(lane_id="a", status="completed", run_id="run-1")
    run_dict = {"run_id": "run-1", "status": "completed", "work_orders": []}

    reviewer_1 = AsyncMock()
    reviewer_1.review.return_value = CampaignReviewGate(
        status="passed", findings=[], review_model="claude",
    )
    reviewer_2 = AsyncMock()
    reviewer_2.review.return_value = CampaignReviewGate(
        status="passed", findings=[], review_model="gpt-4",
    )

    result = await review_lane(
        manifest=manifest, lane_id="a", artifact=artifact,
        run_dict=run_dict, reviewers=[reviewer_1, reviewer_2], tier=2,
    )
    assert result["status"] == "passed"
    assert result["tier"] == 2
    reviewer_1.review.assert_awaited_once()
    reviewer_2.review.assert_awaited_once()

@pytest.mark.asyncio
async def test_tier_3_retries_with_findings_as_constraints():
    lane = _make_lane(lane_id="a", write_scope=["aragora/server/**"])
    manifest = _make_manifest(lanes=[lane])
    artifact = _make_artifact(lane_id="a", status="completed", run_id="run-1")
    run_dict = {"run_id": "run-1", "status": "completed", "work_orders": []}

    # First review fails with findings
    reviewer = AsyncMock()
    reviewer.review.return_value = CampaignReviewGate(
        status="changes_requested",
        findings=["Missing error handling in endpoint"],
        review_model="claude",
    )

    mock_dispatch = AsyncMock(return_value={
        "status": "completed", "outcome": "deliverable_created",
        "run_id": "retry-run-1", "run": {"run_id": "retry-run-1", "status": "completed"},
    })

    result = await review_lane(
        manifest=manifest, lane_id="a", artifact=artifact,
        run_dict=run_dict, reviewer=reviewer, tier=3,
        dispatch_fn=mock_dispatch, max_retries=2,
    )

    # Verify dispatch was called with findings appended as constraints
    dispatch_call = mock_dispatch.call_args
    spec_arg = dispatch_call.args[0] if dispatch_call.args else dispatch_call.kwargs.get("spec")
    assert "Missing error handling in endpoint" in str(spec_arg.constraints)
    assert result["retry_count"] <= 2
    assert result["status"] in ("passed", "needs_human")

@pytest.mark.asyncio
async def test_tier_3_stops_after_max_retries():
    lane = _make_lane(lane_id="a", write_scope=["aragora/server/**"])
    manifest = _make_manifest(lanes=[lane])
    artifact = _make_artifact(lane_id="a", status="completed", run_id="run-1")
    run_dict = {"run_id": "run-1", "status": "completed", "work_orders": []}

    reviewer = AsyncMock()
    reviewer.review.return_value = CampaignReviewGate(
        status="changes_requested", findings=["Still broken"], review_model="claude",
    )
    mock_dispatch = AsyncMock(return_value={
        "status": "completed", "outcome": "deliverable_created",
        "run_id": "retry-run", "run": {"run_id": "retry-run", "status": "completed"},
    })

    result = await review_lane(
        manifest=manifest, lane_id="a", artifact=artifact,
        run_dict=run_dict, reviewer=reviewer, tier=3,
        dispatch_fn=mock_dispatch, max_retries=2,
    )

    assert result["status"] == "needs_human"
    assert result["retry_count"] == 2
```

- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Implement Tier 2 (two reviewers + synthesizer) and Tier 3 (campaign-style bounded retry)**
- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/tranche_review.py tests/swarm/test_tranche_review.py
git commit -m "feat(tranche): add Tier 2 multi-reviewer and Tier 3 bounded retry"
```

### Task 2.4: Wire review CLI

**Files:**
- Modify: `aragora/cli/commands/swarm.py`
- Test: `tests/cli/test_swarm_command.py`

- [ ] **Step 1: Write failing test for review CLI**
- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Wire `swarm tranche review` with --lane, --all-completed, --tier, --json**
- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit**

```bash
git add aragora/cli/commands/swarm.py tests/cli/test_swarm_command.py
git commit -m "feat(cli): wire swarm tranche review command"
```

---

## Phase 3: Integrate

### Task 3.1: PR discovery and check classification

**Files:**
- Create: `aragora/swarm/tranche_integrate.py`
- Test: `tests/swarm/test_tranche_integrate.py`

- [ ] **Step 1: Write failing tests**

```python
def test_discover_pr_from_artifact_metadata():
    artifact = _make_artifact(metadata={"deliverable": {"pr_url": "https://github.com/org/repo/pull/42"}})
    pr = discover_lane_pr(artifact)
    assert pr == "https://github.com/org/repo/pull/42"

def test_classify_checks_all_green():
    checks = [
        {"name": "lint", "conclusion": "SUCCESS", "required": True},
        {"name": "typecheck", "conclusion": "SUCCESS", "required": True},
    ]
    result = classify_check_results(checks)
    assert result == "checks_passed"

def test_classify_checks_advisory_noise():
    checks = [
        {"name": "lint", "conclusion": "SUCCESS", "required": True},
        {"name": "Self-Host Compose Smoke", "conclusion": "FAILURE", "required": False},
    ]
    result = classify_check_results(checks)
    assert result == "checks_passed"  # non-required failure is noise
```

- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Implement discover_lane_pr, register_pr, and classify_check_results**

- `discover_lane_pr(artifact) -> str | None` — extract PR URL from artifact metadata or search via `gh pr list`
- `register_pr(pr_url, branch, registry: PullRequestRegistry)` — register discovered PR in `PullRequestRegistry` if not already tracked. Import from `aragora.swarm.pr_registry`.
- `classify_check_results(checks) -> str` — reuse check partitioning logic from `aragora/ralph/github_control.py` where applicable.

Add assertion to test:
```python
def test_discover_and_register_pr():
    registry = MagicMock(spec=PullRequestRegistry)
    artifact = _make_artifact(metadata={"deliverable": {"pr_url": "https://github.com/org/repo/pull/42"}})
    pr = discover_lane_pr(artifact)
    register_pr(pr, "feat-branch", registry)
    registry.register.assert_called_once()
```

- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/tranche_integrate.py tests/swarm/test_tranche_integrate.py
git commit -m "feat(tranche): add PR discovery and check classification"
```

### Task 3.2: Integration assessment and merge execution

**Files:**
- Modify: `aragora/swarm/tranche_integrate.py`
- Test: `tests/swarm/test_tranche_integrate.py`

- [ ] **Step 1: Write failing tests for assess and merge**

```python
def test_assess_returns_recommendation_without_merging():
    result = assess_lane_integration(artifact=a, checks="checks_passed",
                                      review_status="passed", merge_policy="confirm")
    assert result["recommendation"] == "merge"
    assert result["executed"] is False  # confirm policy → no auto-merge

def test_merge_executes_with_approve_flag():
    result = assess_lane_integration(artifact=a, checks="checks_passed",
                                      review_status="passed", merge_policy="auto",
                                      approve=True)
    assert result["executed"] is True

@pytest.mark.asyncio
async def test_integrate_records_decision_in_coordination_store():
    mock_store = MagicMock(spec=DevCoordinationStore)
    artifact = _make_artifact(
        lane_id="a", status="completed",
        metadata={"receipt_id": "receipt-xyz", "lease_id": "lease-abc"},
    )

    await record_lane_integration(
        artifact=artifact,
        decision="merge",
        rationale="checks passed, review approved",
        decided_by="tranche-integrate",
        store=mock_store,
        target_branch="main",
    )

    mock_store.record_integration_decision.assert_called_once()
    call_kwargs = mock_store.record_integration_decision.call_args.kwargs
    assert call_kwargs["receipt_id"] == "receipt-xyz"
    assert call_kwargs["lease_id"] == "lease-abc"
    assert call_kwargs["decision"] == "merge"

def test_assess_checkpoint_mode_always_awaits_confirmation():
    result = assess_lane_integration(
        artifact=_make_artifact(status="completed"),
        checks="checks_passed",
        review_status="passed",
        merge_policy="auto",
        autonomy_mode="checkpoint",
    )
    assert result["recommendation"] == "merge"
    assert result["executed"] is False  # checkpoint always waits

def test_assess_fire_and_forget_auto_merges():
    result = assess_lane_integration(
        artifact=_make_artifact(status="completed"),
        checks="checks_passed",
        review_status="passed",
        merge_policy="auto",
        autonomy_mode="fire_and_forget",
        approve=True,
    )
    assert result["recommendation"] == "merge"
    assert result["executed"] is True
```

- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Implement assess_lane_integration and execute_merge**
- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/tranche_integrate.py tests/swarm/test_tranche_integrate.py
git commit -m "feat(tranche): add integration assessment and merge execution"
```

### Task 3.3: Wire integrate CLI

**Files:**
- Modify: `aragora/cli/commands/swarm.py`
- Test: `tests/cli/test_swarm_command.py`

- [ ] **Step 1: Write failing test**
- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Wire `swarm tranche integrate` with --lane, --all-mergeable, --approve, --json**
- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit**

```bash
git add aragora/cli/commands/swarm.py tests/cli/test_swarm_command.py
git commit -m "feat(cli): wire swarm tranche integrate command"
```

---

## Phase 4: Watch & Durable State

### Task 4.1: State refresh from authoritative stores

**Files:**
- Create: `aragora/swarm/tranche_watch.py`
- Test: `tests/swarm/test_tranche_watch.py`

- [ ] **Step 1: Write failing tests for state refresh**

```python
def test_refresh_updates_lane_status_from_artifact_store():
    state = TrancheRunState(manifest_id="m1", status="running", autonomy_mode="adaptive")
    state.lane_states["a"] = LaneRunState(lane_id="a", status="dispatched")

    artifact = _make_artifact(lane_id="a", status="completed", run_id="run-1")
    refreshed = refresh_tranche_state(state, artifacts={"a": artifact})

    assert refreshed.lane_states["a"].status == "completed"
    assert refreshed.lane_states["a"].run_id == "run-1"
```

- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Implement refresh_tranche_state**

Refresh from: artifact store, supervisor runs, leases, receipts. Update LaneRunState fields. Update TrancheRunState.status based on aggregate lane states.

- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/tranche_watch.py tests/swarm/test_tranche_watch.py
git commit -m "feat(tranche): add state refresh from authoritative stores"
```

### Task 4.2: Observer and driver session modes

**Files:**
- Modify: `aragora/swarm/tranche_watch.py`
- Test: `tests/swarm/test_tranche_watch.py`

- [ ] **Step 1: Write failing tests**

```python
def test_driver_claim_succeeds_when_no_active_driver():
    state = TrancheRunState(manifest_id="m1", status="running", autonomy_mode="adaptive")
    updated = claim_driver(state, session_id="sess-1")
    assert updated.driver_session == "sess-1"
    assert updated.driver_heartbeat is not None

def test_driver_claim_fails_when_active_driver_with_heartbeat():
    state = TrancheRunState(manifest_id="m1", status="running", autonomy_mode="adaptive",
                             driver_session="sess-1", driver_heartbeat=_utcnow())
    with pytest.raises(DriverAlreadyClaimedError):
        claim_driver(state, session_id="sess-2")

def test_stale_driver_can_be_taken_over():
    state = TrancheRunState(manifest_id="m1", status="running", autonomy_mode="adaptive",
                             driver_session="sess-1",
                             driver_heartbeat=_utcnow() - timedelta(minutes=10))
    updated = claim_driver(state, session_id="sess-2", takeover_timeout_seconds=300)
    assert updated.driver_session == "sess-2"
```

- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Implement claim_driver, release_driver, heartbeat_driver**
- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/tranche_watch.py tests/swarm/test_tranche_watch.py
git commit -m "feat(tranche): add observer/driver session modes with heartbeat"
```

### Task 4.3: Watch loop and autonomous advancement

**Files:**
- Modify: `aragora/swarm/tranche_watch.py`
- Test: `tests/swarm/test_tranche_watch.py`

- [ ] **Step 1: Write failing tests for watch tick behavior**

```python
@pytest.mark.asyncio
async def test_watch_tick_triggers_review_when_lane_completes():
    state = _make_state(lane_statuses={"a": "completed"})
    mock_rev = AsyncMock()
    mock_rev.return_value = {"status": "passed", "tier": 1}
    new_state = await watch_tick(state, manifest=m, autonomy_mode="adaptive",
                                  review_fn=mock_rev, artifact_store=store)
    assert new_state.lane_states["a"].status in ("reviewing", "review_passed")
    mock_rev.assert_awaited_once()  # Verify review was actually triggered

@pytest.mark.asyncio
async def test_watch_tick_marks_tranche_completed_when_all_lanes_done():
    state = _make_state(lane_statuses={"a": "completed", "b": "completed"})
    # All lanes completed → tranche completed
    new_state = await watch_tick(state, manifest=m, autonomy_mode="adaptive",
                                  artifact_store=store)
    assert new_state.status == "completed"

@pytest.mark.asyncio
async def test_watch_tick_fire_and_forget_auto_advances():
    state = _make_state(lane_statuses={"a": "review_passed"})
    mock_integrate = AsyncMock(return_value={"recommendation": "merge", "executed": True})
    new_state = await watch_tick(state, manifest=m, autonomy_mode="fire_and_forget",
                                  integrate_fn=mock_integrate, artifact_store=store)
    mock_integrate.assert_awaited_once()  # Verify auto-advance
```

- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Implement watch_tick and watch_loop**
- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit**

```bash
git add aragora/swarm/tranche_watch.py tests/swarm/test_tranche_watch.py
git commit -m "feat(tranche): add watch loop with autonomous advancement"
```

### Task 4.4: Wire watch and list CLI

**Files:**
- Modify: `aragora/cli/commands/swarm.py`
- Modify: `aragora/cli/parser.py`
- Test: `tests/cli/test_swarm_command.py`

- [ ] **Step 1: Write failing tests for watch/list CLI**
- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Wire `swarm tranche watch` with --driver, --interval, --json and `swarm tranche list` with --json**
- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit**

```bash
git add aragora/cli/commands/swarm.py aragora/cli/parser.py tests/cli/test_swarm_command.py
git commit -m "feat(cli): wire swarm tranche watch and list commands"
```

---

## Phase 5: Integration Testing & CLI Reference

### Task 5.1: End-to-end submit → design-review → run → review → integrate test

**Files:**
- Create: `tests/swarm/test_tranche_e2e.py`

- [ ] **Step 1: Write end-to-end test**

```python
@pytest.mark.asyncio
async def test_tranche_lifecycle_e2e(tmp_path):
    """Submit a bundle, design-review it, mock-dispatch a lane, review it, and assess integration."""
    bundle = {
        "objective": "Test e2e flow",
        "candidate_lanes": [{
            "lane_id": "e2e_lane",
            "title": "E2E test lane",
            "prompt": "Do the thing",
            "owner_role": "engineer",
            "allowed_write_scope": ["tests/**"],
            "verification_commands": ["echo ok"],
        }],
        "autonomy_mode": "checkpoint",
    }

    # Submit
    submit_result = submit_intake_bundle(bundle, repo_root=tmp_path,
                                          skip_github_resolution=True)
    assert submit_result["submission_status"] == "awaiting_confirmation"
    assert submit_result["recommended_action"] == "design-review"

    # Load state
    state = TrancheRunState.load(
        tmp_path / ".aragora" / "tranches" / submit_result["manifest_id"] / "run_state.yaml"
    )
    assert state.lane_states["e2e_lane"].status == "pending"
```

- [ ] **Step 2: Run to verify it passes (or debug)**

Run: `python3 -m pytest tests/swarm/test_tranche_e2e.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/swarm/test_tranche_e2e.py
git commit -m "test(tranche): add end-to-end lifecycle integration test"
```

### Task 5.2: Update CLI reference and exports

**Files:**
- Modify: `docs/CLI_REFERENCE.md`
- Modify: `aragora/swarm/__init__.py`

- [ ] **Step 1: Add submit/review/integrate/watch/list to CLI_REFERENCE.md**

Add entries for each new tranche subcommand with usage examples.

- [ ] **Step 2: Update __init__.py exports**

Export new public names: `submit_intake_bundle`, `review_lane`, `assess_lane_integration`, `TrancheRunState`, `LaneRunState`.

- [ ] **Step 3: Run full test suite**

Run: `python3 -m pytest tests/swarm/test_tranche*.py tests/cli/test_swarm_command.py -q`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add docs/CLI_REFERENCE.md aragora/swarm/__init__.py
git commit -m "docs: update CLI reference and exports for tranche orchestration"
```
