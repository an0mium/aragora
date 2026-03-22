# Brain Dump to Execution Map (2026 Q2)

Last updated: 2026-03-22

Related:
- `docs/status/EXECUTION_PROGRAM_2026Q2_Q4.md`
- `docs/status/NEXT_STEPS_CANONICAL.md`
- `docs/status/ACTIVE_EXECUTION_ISSUES.md`
- `docs/plans/ARAGORA_IDEA_TO_EXECUTION_STRATEGY.md`

The February map was too broad for the repo state that exists on `main` now. The near-term execution map is reset around the proof surfaces that are actually merged and the wedge they imply.

## Best Order (Priority Sequence)

### P0: Close the truthful proof surfaces (Weeks 1-2)

1. Default product loop: credentials/provider routing -> live debate -> KM-enriched receipt -> visible result
2. Truthful public and operator state: `/demo`, integrations status/edit, receipts page, pipeline live state
3. Bounded execution operator contract: authoritative integrator view, per-lane provenance, remote-head PR review, completed-lane publish and truthful terminalization

### P1: Harvest PMF on the merged wedges (Weeks 3-6)

4. Inbox trust wedge as the first recurring partner workflow
5. Swarm/OpenClaw bounded execution as the second recurring proof surface
6. Five functional frontend paths instead of page-count theater
7. Shareable public proof surface and receipt-sharing loop

### P2: Close the learning loop and workbench (Weeks 6-10)

8. Default KM retrieval + writeback + settlement hooks as part of the canonical debate loop
9. Live stage-transition review slices in the workbench
10. Execution outcomes revising upstream specs, goals, and ideas

### P3: Widen only after the wedge is repeatable (continuous)

11. Multi-agent scale-out beyond the current proof lanes
12. Broader enterprise/FinOps/compliance productization after the wedge repeats weekly with design partners

## Translation of the 12 Ideas

## 1) Close the default product loop

Outcome:
- One truthful path from setup to visible result exists on `main` without hidden operator rescue.

Code anchors:
- `aragora/cli/commands/api_keys.py`
- `aragora/debate/`
- `aragora/live/src/app/(app)/`
- `docs/status/NEXT_STEPS_CANONICAL.md`

Owner:
- `@team-core` + `@team-growth`

Primary KPIs:
- Time to first truthful result <= 15 minutes
- Guided setup-to-receipt success >= 90%

## 2) Make public and core result views truthful

Outcome:
- The public demo, receipts page, integrations status/edit flows, and pipeline views truthfully reflect live or partial state.

Code anchors:
- `aragora/live/src/`
- `aragora/server/handlers/`
- `docs/status/ACTIVE_EXECUTION_ISSUES.md`

Owner:
- `@team-growth` + `@team-platform`

Primary KPIs:
- Demo/live-state misrepresentation incidents = 0
- Core surface error rate <= 1%

## 3) Lock the inbox trust wedge

Outcome:
- Gmail triage remains the first narrow workflow that proves receipt-before-action under real usage.

Code anchors:
- `aragora/inbox/`
- `scripts/gmail_oauth_setup.py`
- `docs/plans/2026-03-06-openrouter-inbox-dogfood-plan.md`

Owner:
- `@team-integrations`

Primary KPIs:
- Receipt-before-action validation = 100%
- Override rate trends down week-over-week on pilot inboxes

## 4) Make integrator/operator views authoritative

Outcome:
- Operators can see the real lane state, claims, evidence, and next actions without reconstructing runs manually.

Code anchors:
- `aragora/swarm/`
- `aragora/live/src/components/pipeline-canvas/`
- `docs/status/ACTIVE_EXECUTION_ISSUES.md`

Owner:
- `@team-platform`

Primary KPIs:
- Lane state disagreements between UI/CLI/artifacts = 0
- Operator diagnosis time for blocked lanes <= 5 minutes

## 5) Turn bounded repo execution into a repeatable proof surface

Outcome:
- Ralph/tranche/swarm runs end in a deliverable or a preserved blocker reason with a canonical operator handoff.

Code anchors:
- `aragora/swarm/supervisor.py`
- `aragora/swarm/reconciler.py`
- `aragora/swarm/integrator.py`

Owner:
- `@team-platform` + `@team-core`

Primary KPIs:
- Truthful terminalization for bounded lanes >= 95%
- Completed-lane publish success >= 95%

## 6) Make OpenClaw real on a narrow path

Outcome:
- At least one narrow action-dispatch path uses real OpenClaw execution and receipt linkage in production-like conditions.

Code anchors:
- `aragora/openclaw/`
- `aragora/computer_use/`
- `aragora/server/handlers/computer_use_handler.py`

Owner:
- `@team-core`

Primary KPIs:
- Narrow-path action dispatch success >= 90%
- Every executed action has linked receipt/provenance

## 7) Make default debates learn from KM

Outcome:
- The canonical debate loop retrieves relevant prior receipts, writes back useful outcomes, and fires settlement hooks automatically.

Code anchors:
- `aragora/debate/knowledge_injection.py`
- `aragora/debate/post_debate/`
- `aragora/knowledge/`

Owner:
- `@team-core`

Primary KPIs:
- Default debate KM enrichment coverage >= 80%
- Outcome writeback success >= 95%

## 8) Ship five functional frontend paths

Outcome:
- Aragora stops claiming breadth via page count and instead ships a handful of complete, trustworthy user journeys.

Code anchors:
- `aragora/live/src/app/(app)/`
- `docs/FEATURE_GAP_LIST.md`
- `docs/status/ACTIVE_EXECUTION_ISSUES.md`

Owner:
- `@team-growth`

Primary KPIs:
- Five core paths complete and dogfooded weekly
- Shell-page count on critical routes trends down every sprint

## 9) Surface live pipeline and stage-transition state

Outcome:
- The workbench shows live orchestration state and one or more truthful transition-review slices.

Code anchors:
- `aragora/live/src/components/pipeline-canvas/`
- `aragora/pipeline/`
- `docs/plans/IDEA_TO_EXECUTION_PIPELINE.md`

Owner:
- `@team-growth` + `@team-core`

Primary KPIs:
- Pipeline live state accuracy >= 95%
- At least one stage transition is reviewable and editable in the UI

## 10) Emit canonical run receipts and review evidence

Outcome:
- Every lane can carry forward its provenance, verification evidence, and remote review target without lossy operator reconstruction.

Code anchors:
- `aragora/swarm/receipts.py`
- `aragora/swarm/integrator.py`
- `aragora/cli/commands/review.py`

Owner:
- `@team-platform`

Primary KPIs:
- Lanes with canonical receipt/provenance artifact >= 95%
- Review decisions grounded in remote head for 100% of publishable PRs

## 11) Turn merged proof surfaces into PMF proof

Outcome:
- Merged autonomy proofs become repeatable partner workflows with scorecards, not just internal engineering wins.

Code anchors:
- `docs/status/DESIGN_PARTNER_PROGRAM.md`
- `docs/status/PMF_SCORECARD.md`
- `docs/status/COMMERCIAL_POSITIONING.md`

Owner:
- `@team-growth` + `@team-analytics`

Primary KPIs:
- 3-5 partners running one bounded workflow weekly
- 2 publishable case studies with hard metrics

## 12) Then widen to the unified idea-to-execution DAG

Outcome:
- The long-range moat work continues, but only after the current wedge is truthful and repeatable.

Code anchors:
- `aragora/pipeline/`
- `aragora/live/src/components/pipeline-canvas/`
- `docs/plans/ARAGORA_EVOLUTION_ROADMAP.md`

Owner:
- `@team-core` + `@team-growth`

Primary KPIs:
- Idea -> goal -> action -> execution transitions visible in one shell
- Execution outcomes can revise upstream planning artifacts

## Dependencies and Constraints

1. P0 reliability must be completed before major DAG/automation UX rollout.
2. Frontend parity depends on stable auth/session and deployment paths.
3. Autonomous self-improvement must remain policy-gated and audit-logged.
4. Worktree orchestration depends on branch hygiene and merge gate reliability.
5. Model lineup updates require parity checks across SDK, server, and docs.

## Immediate Sprint Slice (next 2 weeks)

1. Auth + routing E2E fixes for onboarding, OAuth callbacks, and post-login debate redirect.
2. Release pipeline hardening for frontend/backend deploy verification against expected SHA.
3. Oracle stream/TTS first-token-first-audio prototype behind feature flag.
4. Worktree coordinator default path audit and stalled-session watchdog.
5. Security rotation automation dry run with Secrets Manager and rotation telemetry.
