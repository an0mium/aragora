# CI Baselines — Governance Contract

Files in this directory are **enforcement baselines**: committed snapshots of
accepted debt that CI gates compare against. Growing a baseline is a
governance event, not a routine edit.

## Rules

1. **Never grow a baseline to silence a failing gate.** Fix the drift, or get
   explicit human sign-off for accepting the new debt in the PR that grows it.
2. **Shrinking a baseline is always welcome** — when drift items are actually
   fixed, refresh the corresponding baseline downward in the same PR.
3. **Baseline changes are Tier 3+** under `docs/AGENT_OPERATING_CONTRACT.md`:
   they change what CI enforces on main, so they require operator
   human-settlement before merge. Use a `chore(governance): ...` commit title
   (precedent: #1669, #2270, #6054).

## Contract drift ratchet (`contract_drift_program.json`)

`scripts/check_contract_drift_ratchet.py` (workflow: `contract-drift-governance.yml`)
sums item counts from three sibling baselines —
`verify_sdk_contracts.json` (py + ts SDK drift),
`validate_openapi_routes.json` (missing + orphaned routes), and
`check_sdk_parity.json` (missing from both SDKs) —
and fails `--strict` when the total exceeds the program target.

- `weekly_reduction: 0.0` — **hold-the-line mode** (current): target stays at
  `start_total_items`; the gate fails only if the aggregate grows. This is a
  no-regression gate and stays green without dedicated burn-down staffing.
- `weekly_reduction > 0` — **burn-down mode**: the target decays
  compounding per week from `start_date`. Only set this when drift-reduction
  work is actually staffed at the implied weekly rate (10%/week from ~500
  items requires closing ~35–50 items *every week*). History: four programs
  (Feb 13 / Mar 29 / Apr 5+9 / Apr 17 2026) all went red within ~3 weeks of
  their reset because the decay outpaced actual work, and the permanently red
  check then masked real regressions.

### Re-baselining the program

Allowed when (a) drift items were genuinely fixed (lower the
`start_total_items` to the new aggregate), or (b) a deliberate, human-approved
debt acceptance occurred (raise it, documenting why in the PR). When
re-baselining, set `start_total_items` to the current sum reported by
`python scripts/check_contract_drift_ratchet.py` ("Current total") and
`start_date` to the merge date.
