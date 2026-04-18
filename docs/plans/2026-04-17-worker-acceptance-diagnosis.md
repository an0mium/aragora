# Worker Acceptance-Criteria Binding — Phase 1 Diagnosis

**Date:** 2026-04-17
**Mission:** SpecUpgrader v1.3 — bind worker deliverables to issue acceptance criteria

## TL;DR

After the full dispatch repair loop shipped (v1–v1.2), three Cycle 1 PRs
(#6165, #6171, #6175) were mechanically valid — they committed, pushed, and
opened PRs — but **none fulfilled their source issue's "Add tests"
acceptance criteria**. The worker interpreted "Add tests for X.py" as
"improve X.py". Acceptance criteria survive to the prompt, but only as
soft bullets, and no post-delivery gate checks whether the deliverable
actually satisfies them.

This document traces the spec-to-prompt translation path end-to-end and
identifies the concrete places where acceptance binding breaks down.

---

## 1. Evidence: what Cycle 1 actually produced

| Issue  | Requested deliverable                                               | PR     | What the PR actually changed                                      |
|--------|---------------------------------------------------------------------|--------|-------------------------------------------------------------------|
| #5904  | `tests/swarm/test_boss_worker_lifecycle.py` (new or expanded)       | #6165  | `aragora/swarm/boss_worker_lifecycle.py` — +1/-0 line              |
| #5899  | `tests/scripts/test_reconcile_b0_pr_truth.py` (new)                 | #6171  | `scripts/reconcile_b0_pr_truth.py` — +14/-6 lines                  |
| #5895  | `tests/benchmarks/test_rescue_productization.py` (new)              | #6175  | `docs/status/TW03_RESCUE_PRODUCTIZATION_STATUS.md` — doc timestamp |

In every case:

- The worker modified the **subject file** of the issue instead of **creating
  the test file specified in the issue body**.
- The PR body was auto-filled from the commit message, with no `Closes #N`.
- GitHub's `closedByPullRequestsReferences` edge recorded zero genuine
  closures. The corpus honesty gate registers `truth_success = 0.0%` as a
  result.

---

## 2. Spec-to-prompt path — the flow

### 2.1 Issue → SwarmSpec

`aragora.swarm.boss_worker_lifecycle.dispatch_issue` builds a
[`SwarmSpec`](../aragora/swarm/spec.py) from a sanitized issue body:

```python
# aragora/swarm/boss_worker_lifecycle.py (excerpted)
spec = SwarmSpec(
    raw_goal=goal,                        # composed from [Issue #N] title + body
    refined_goal=goal,
    constraints=constraints,              # inferred from "don't"/"must not" markers
    budget_limit_usd=loop.config.budget_limit_usd,
    file_scope_hints=scope_hints,         # regex-extracted path-like strings
    requires_approval=True,
    ...
)
# Later — validation contract filled from an Acceptance Criteria section
validation_contract = extract_issue_validation_contract(sanitized_issue_body)
if validation_contract:
    spec.acceptance_criteria = list(validation_contract)
```

`extract_issue_validation_contract` looks specifically for **"Acceptance
Criteria" / "Validation" / "Definition of Done"** sections. Issues like
#5904 use a shorter `## Acceptance` heading (not in the known prefix list),
so the contract may not be extracted, leaving `acceptance_criteria` empty.

### 2.2 SwarmSpec → work orders

`SwarmSupervisor._build_supervised_work_orders(spec)` (supervisor.py:1420):

```python
work_orders = self.bridge.build_work_orders(subtasks)
for item in work_orders:
    _ensure_work_order_scope(item, spec)
    item.expected_tests = self._default_tests(item, spec)
    item.metadata = {
        **dict(item.metadata),
        "acceptance_criteria": list(spec.acceptance_criteria),
        "constraints": list(spec.constraints),
    }
```

So the work order metadata carries `acceptance_criteria` and
`file_scope` — good. The problem is what happens next.

### 2.3 Work order → worker prompt

`WorkerLauncher._build_prompt(work_order)` (worker_launcher.py:1075) is
the **last mile**. Here is the relevant block (annotated):

```python
# --- Section 4: File scope (guidance, not hard boundary) ---
if file_scope:
    scope_list = "\n".join(f"  - {f}" for f in file_scope)
    parts.append(
        f"FILE SCOPE GUIDANCE:\n"
        f"The planner expects you to work in these paths:\n{scope_list}\n"
        "IMPORTANT: Before starting, verify these paths exist. If they do not, "
        "search the codebase for the actual files that match the intent ..."
        "Treat the resolved scope as a hard boundary ..."
    )

# --- Section 5: Validation (concise) ---
acceptance = metadata.get("acceptance_criteria", [])
if acceptance:
    criteria_text = "\n".join(f"  - {c}" for c in acceptance)
    parts.append(f"Acceptance criteria:\n{criteria_text}")
```

### 2.4 Observed prompt content (Issue #5904 worked example)

When the issue body contains `## Acceptance\n- pytest tests/swarm/test_boss_worker_lifecycle.py -v passes`, the
`extract_issue_validation_contract` function does NOT recognize the
heading `Acceptance` — only the prefixes in `_VALIDATION_SECTION_PREFIXES`
(which require `Acceptance Criteria`, `Validation`, `Definition of
Done`, etc.). So `spec.acceptance_criteria` ends up EMPTY for this issue.

The only surviving signal for the worker was:
- `raw_goal` = `[Issue #5904] [TW-02] Add boss worker lifecycle module tests for benchmark truth\n\n## Scope\nAdd unit tests for aragora/swarm/boss_worker_lifecycle.py — the recently extracted worker execution boundary module.\n...`
- `file_scope_hints` = `["tests/swarm/test_boss_worker_lifecycle.py", "aragora/swarm/boss_worker_lifecycle.py"]` (regex-extracted)

But in the prompt:
- Section 4 ("FILE SCOPE GUIDANCE") lists **both** the test file and the
  source file. The worker sees two options and picks the "easier" one —
  editing the source file by one line.
- Section 5 ("Acceptance criteria:") renders **empty** and is omitted,
  removing the only strong constraint that would have said "tests must
  be added".
- The CRITICAL section is about committing — it does not say anything
  about what the deliverable MUST contain.

The result: a mechanically valid commit that does not satisfy the issue.

---

## 3. Root-cause diagnosis — what actually fails

### 3.1 Acceptance criteria are soft at best, absent at worst

- **Absent** when the issue uses `## Acceptance` (a common short form) —
  `extract_issue_validation_contract` only honors `Acceptance Criteria`.
- **Soft** even when present — rendered as `"Acceptance criteria:\n  - ..."`
  with no binding language. An LLM worker is free to decide the criteria
  are "guidance" and commit anything the file-scope "guidance" allows.

### 3.2 File scope is guidance, and includes both subject and test file

The regex `_FILE_SCOPE_HINT_RE` greedily extracts every path-like string
from the issue body. For a "tests for X.py" issue, both `X.py` and
`tests/.../test_X.py` end up in scope. The worker is told to pick the
one that exists. `X.py` always exists; the test file usually does not.

### 3.3 No post-delivery gate verifies the deliverable satisfies the intent

After the worker exits with a commit, the existing scope-violation gate
fires on **out-of-scope edits**. But a worker that edits `X.py` is by
definition in-scope (since `X.py` was in the hints). There is no gate
that checks:

- the deliverable must include **at least one new test file** when the
  acceptance said "add tests"
- the deliverable must include **the specific file the issue named as
  the expected new file**

### 3.4 No `Closes #N` is injected into the PR body

`publish_lane_deliverable` calls `github.create_pr_for_branch(branch, target_branch)`,
which in turn runs `gh pr create --fill --head <branch> --base <target>`.
`gh pr create --fill` auto-fills title and body from the latest commit.
Commit messages come from the worker (`fix: update X.py`), so `Closes #N`
is only present if the worker deliberately added it. In Cycle 1, none
of the three PRs included it.

As a result, issues stay open, and GitHub's
`closedByPullRequestsReferences` edge (which the corpus honesty gate
consults) records zero closures — truth_success stays at 0.0% even as
the boss loop produces mechanically valid PRs.

---

## 4. Fix shape for Phases 2–4

Rather than restructuring the prompt path (risky, cross-cuts many
components), v1.3 adds a **conservative post-delivery acceptance gate**
that runs between worker-completion and PR-publish:

1. **File-scope adherence (explicit)** — changed paths must be in scope
   or obvious companion test files. Current scope enforcement is
   *within* the worker-result apply step but it is permissive; v1.3
   adds an explicit, structured check before `publish_lane_deliverable`.

2. **Test-presence check (new)** — if any acceptance criterion mentions
   "test"/"tests", require the deliverable to include at least one new
   or substantially edited file under `tests/`. A one-line production
   change with no test file fails the gate.

3. **File-creation check (new)** — if the issue's `## Files` or similar
   section names a **new** test file path, require the deliverable to
   create that file (or a sibling under the same directory matching the
   pattern `test_<subject>.py`).

4. **`Closes #N` auto-injection (new)** — if the gate passes AND the
   deliverable originated from an issue number, `gh pr edit` the PR
   body to prepend `Closes #<issue_number>` (unless already present).

Failure mode: structured `needs_human` with a reason code, emit a
`spec_upgrade` telemetry row so the Stage-Gate Conductor sees the class.

**Conservative tilt:** false negatives (rejecting a valid deliverable)
are cheaper than false positives (auto-closing an unsatisfying one).
When uncertain, the gate prefers `needs_human`.

---

## 5. What changes at the prompt level? (Nothing in v1.3)

A stronger prompt path — binding acceptance criteria to file creation,
explicitly listing required test files, dropping the "pick an existing
file" instruction — would reduce gate rejections. But this cross-cuts
`worker_launcher._build_prompt`, `SwarmSupervisor`, `SwarmSpec`, and
the issue parser. v1.3 intentionally does NOT touch the prompt path.

Phase 6 scoping note: if after v1.3 the acceptance gate still rejects
>30 % of deliverables, v1.4 should tighten the prompt (explicit
"you must create this file, not modify it").

---

## 6. Files involved

| File | Role in the gap |
|------|-----------------|
| `aragora/swarm/boss_worker_lifecycle.py:dispatch_issue` | Builds the SwarmSpec from the issue |
| `aragora/swarm/boss_validation.py:extract_issue_validation_contract` | Parses acceptance criteria out of the body; strict about heading names |
| `aragora/swarm/supervisor.py:_build_supervised_work_orders` | Passes `acceptance_criteria` into `work_order.metadata` |
| `aragora/swarm/worker_launcher.py:_build_prompt` | Renders the criteria as a soft bullet list |
| `aragora/swarm/supervisor_probes.py:_check_file_scope_violations` | Current scope gate — only flags truly out-of-scope edits |
| `aragora/swarm/boss_loop.py:_maybe_publish_deliverable` | Publish path; no acceptance gate before PR create |
| `aragora/ralph/github_control.py:create_pr_for_branch` | Uses `gh pr create --fill` — no `Closes #N` injection |

## 7. Expected truth_success lift

With the gate in place:

- Deliverables that satisfy the acceptance → PR body includes `Closes #N`
  → merging the PR auto-closes the issue → GitHub exposes the
  `closedByPullRequestsReferences` edge → corpus honesty gate records a
  genuine closure.
- Tangential deliverables are rejected, go to `needs_human`, and do
  NOT consume a benchmark "truth_success" slot.

Forecast: **truth_success moves from 0.0 % to a non-zero floor** that
reflects the rate at which workers actually fulfill acceptance criteria.
If that rate is low (e.g. <20 %), v1.4 must strengthen the prompt. But
even a low number is honest — whereas 0.0 % under a steady stream of
PRs was noise masquerading as throughput.
