# 2-Week Hardening Sprint Skills Runbook

Purpose: execute all installed Codex skills in the best order for Aragora hardening, with CI stabilization first and lower-priority skills staged later.

Source checklist:
- `docs/status/HARDENING_SPRINT_2W_CHECKLIST.md`

## Ground Rules

- Keep feature freeze active during hardening.
- Run CI burn-down continuously while applying skill-driven workflows.
- Prioritize correctness, determinism, and rollback safety over new capability work.
- Use deploy/media skills only after hardening gates are green.

## Week 1: Stabilize Core Lanes

### Day 1: Planning + Baseline Failures

Skills:
- `linear`
- `gh-fix-ci`

Actions:
- Build a 2-week board with the same 8 lanes from the hardening checklist.
- Open CI checks and capture first failing test per lane.
- Assign owners for Debate, Knowledge, Nomic, QA, Ops, Security.

Exit criteria:
- Baseline failure list exists with owner + ETA for each blocker.

### Day 2: Debate/Knowledge/Nomic Burn-Down

Skills:
- `gh-fix-ci`
- `gh-address-comments`

Actions:
- Fix first fail in `tests/debate`, then rerun first-fail mode.
- Fix first fail in `tests/knowledge` + `tests/nomic`, then rerun first-fail mode.
- Address active review comments immediately after each fix merge.

Exit criteria:
- No unresolved P0/P1 failures in targeted debate/knowledge/nomic lanes.

### Day 3: Runtime/Flag Consistency

Skills:
- `gh-fix-ci`
- `openai-docs`

Actions:
- Align runtime and feature-flag behavior between `scripts/nomic_loop.py` and `aragora/nomic/handlers.py`.
- Validate external model/API assumptions for currently configured agent providers.

Exit criteria:
- Same flag values produce same behavior across legacy + handler paths.

### Day 4: Security Pass 1 (Design-Level)

Skills:
- `security-threat-model`
- `security-best-practices`

Actions:
- Threat-model self-modification and approval boundaries (SICA paths).
- Review touched Python/TypeScript code for secure defaults and unsafe patterns.

Exit criteria:
- Documented threats, mitigations, and required follow-up issues.

### Day 5: Ownership + Risk Concentration

Skills:
- `security-ownership-map`
- `linear`

Actions:
- Compute sensitive-code ownership and bus-factor hotspots.
- File ownership-risk tasks and assign backups for high-risk files.

Exit criteria:
- Critical sensitive areas have named owners and backup maintainers.

## Week 2: Validate, Observe, and Roll Out

### Day 6: UI/Operator Flow Validation

Skills:
- `playwright`
- `screenshot`

Actions:
- Run deterministic UI smoke checks for `aragora/live`.
- Capture failing states and attach artifacts to issues/PRs.

Exit criteria:
- Operator-critical flows have reproducible smoke coverage and visual evidence.

### Day 7: Production Signal Triage

Skills:
- `sentry`
- `gh-fix-ci`

Actions:
- Pull highest-impact recent errors and map them to code owners.
- Prioritize fixes that overlap active hardening lanes.

Exit criteria:
- Error backlog is triaged by severity and owner.

### Day 8: Benchmarks + Acceptance Gates

Skills:
- `linear`
- `spreadsheet`
- `jupyter-notebook`

Actions:
- Record benchmark outputs and define pass/fail thresholds.
- Publish acceptance gate tables and baseline deltas.

Exit criteria:
- Bench and quality thresholds are documented and reviewed.

### Day 9: Review Closure + Final Security Pass

Skills:
- `gh-address-comments`
- `security-best-practices`

Actions:
- Close remaining PR feedback and rerun impacted tests.
- Re-check security-sensitive diffs before rollout approval.

Exit criteria:
- No open blocking review threads on hardening PRs.

### Day 10: Documentation and Operator Handoff

Skills:
- `notion-spec-to-implementation`
- `notion-knowledge-capture`
- `notion-research-documentation`

Actions:
- Convert outcomes into operator runbooks and decision logs.
- Publish rollback triggers and feature-flag staging guidance.

Exit criteria:
- Operator docs are complete, linked, and reviewable.

### Day 11-12: Meeting + Communication Pack

Skills:
- `notion-meeting-intelligence`
- `doc`
- `pdf`

Actions:
- Prepare rollout-readiness review materials.
- Export final internal handoff docs for audit/review.

Exit criteria:
- Readout package is ready for go/no-go meeting.

### Day 13-14: Controlled Deploy + Optional Demo Assets

Skills:
- `render-deploy`
- `vercel-deploy`
- `netlify-deploy`
- `cloudflare-deploy`
- `figma`
- `figma-implement-design`
- `imagegen`
- `speech`
- `sora`
- `transcribe`
- `yeet`

Actions:
- Use only the deploy skill matching target environment(s).
- Keep rollout staged: telemetry, staging flags, canary, then full release.
- Use media/design skills only if launch/demo assets are needed.
- Use `yeet` only when you explicitly want stage+commit+push+PR in one flow.

Exit criteria:
- Rollout completed with stable metrics and documented rollback path.

## Skill-to-Checklist Mapping

- Checklist Step 1 (Freeze/Baseline): `linear`, `gh-fix-ci`
- Checklist Step 2 (CI Stabilization): `gh-fix-ci`, `gh-address-comments`
- Checklist Step 3 (Runtime Consolidation): `gh-fix-ci`, `openai-docs`
- Checklist Step 4 (Feature Gates): `gh-fix-ci`, `playwright`, `screenshot`
- Checklist Step 5 (Benchmarks): `spreadsheet`, `jupyter-notebook`, `linear`
- Checklist Step 6 (Observability): `sentry`, `linear`
- Checklist Step 7 (Safety/Security): `security-threat-model`, `security-best-practices`, `security-ownership-map`
- Checklist Step 8 (Rollout): deploy skills + Notion/documentation skills

## Daily Operating Command Set

Use these each day to keep execution deterministic:

```bash
python -m pytest tests/reasoning/test_claim_check.py -q -p no:randomly
python -m pytest tests/verification/test_hilbert.py -q -p no:randomly
python -m pytest tests/debate/test_stability_detector.py -q -p no:randomly
python -m pytest tests/knowledge/test_lara_router.py -q -p no:randomly
python -m pytest tests/scripts/test_nomic_sica.py tests/nomic/test_handlers.py -q -p no:randomly
python -m pytest tests/debate tests/knowledge tests/nomic -q -p no:randomly -x --tb=short
```

## Definition of Done

- Hardening checklist exit criteria met for all 8 checklist steps.
- Debate/knowledge/nomic lanes are stable under repeated runs.
- Security and ownership risks are tracked with named owners.
- Rollout artifacts and rollback playbook are complete.
