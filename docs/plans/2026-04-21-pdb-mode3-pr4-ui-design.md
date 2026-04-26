# PDB Mode 3 PR 4 UI Design

Last updated: 2026-04-21
Status: design-stage spec for `#6306` PR4 (follows PR 3 endpoints)

Extends:
- [docs/plans/2026-04-20-pdb-brief-generation-mode3-design.md](2026-04-20-pdb-brief-generation-mode3-design.md)
- [docs/plans/2026-04-21-pdb-mode3-pr2-spec.md](2026-04-21-pdb-mode3-pr2-spec.md)

Depends on: PR 3 endpoints (`POST/GET/DELETE /api/v1/review-queue/prs/{n}/brief/generate|state`) shipping first.

## Purpose

Put the Mode 3 brief generation surface in front of a human. When the user opens the review queue and wants more signal than CI + diff, one click triggers a real heterogeneous panel debate; the UI reflects state transitions live; the BriefPanel below the queue renders the ready artifact.

This PR owns exactly the UI wiring — no backend logic, no new endpoints, no schema changes.

## Boundary

PR 4 owns exactly these files (new and modified):

| File | Responsibility |
|---|---|
| `aragora/live/src/hooks/useReviewQueue.ts` | Add `generateBrief(prNumber)`, `getBriefState(prNumber)`, `cancelBriefGeneration(prNumber)` hooks + a `useBriefState` polling hook |
| `aragora/live/src/components/review-queue/ApproveDecisionModal.tsx` | **NEW** — replaces the current `window.confirm` path in `ReviewQueueCard.handleApprove` with a 3-option modal (Approve / Generate brief / Cancel) |
| `aragora/live/src/components/review-queue/BriefPanel.tsx` | State-aware rendering: absent → CTA, queued/running → progress, ready → existing layout, failed → error + retry, stale → warning + regenerate CTA |
| `aragora/live/src/components/review-queue/ReviewQueueCard.tsx` | Pulse animation on the PR number badge while `state == 'running'`; wire approve-path through the new modal |
| `aragora/live/__tests__/ApproveDecisionModal.test.tsx` | **NEW** tests for 3-way modal |
| `aragora/live/__tests__/BriefPanel.test.tsx` | Update or create tests for all 6 state renderings |
| `aragora/live/__tests__/useReviewQueue.test.ts` | Update to cover new hook methods with mocked `/brief/generate` flow |

## State-aware BriefPanel rendering

The BriefPanel must handle all six lifecycle states from PR 1. Current implementation only handles two (present/absent). New behavior:

| State | UI |
|---|---|
| `absent` | Empty slate: "No brief yet. Click Generate brief below to start a panel debate." + inline button that triggers `generateBrief(prNumber)` |
| `queued` | Skeleton with spinner + "Queued — starting soon" |
| `running` | Skeleton with spinner + phase indicator: "Findings (3/8 roles done) · ~45s elapsed" |
| `ready` | Current rendering (logic/security/maintainability/skeptic sections + verdict + confidence) |
| `failed` | Red-tinted panel: "Brief generation failed at <phase>. Reason: <reason>. Cost so far: $<n>." + "Retry" button that calls `generateBrief` with `force=true` |
| `stale` | Amber-tinted panel: "Brief is for a previous commit (<old-sha> ≠ current <new-sha>). Regenerate?" + "Regenerate" button |

Polling cadence: while state is `queued` or `running`, poll `GET /brief/state` every 3s. Stop polling once state stabilizes (`ready`, `failed`, or `absent` after cancel).

## ApproveDecisionModal — 3-way decision

Current UX uses `window.confirm` with "No brief on file. Approve without PDB brief?" which is untrustworthy because 100% of PRs lack briefs today.

New modal appears when user clicks Approve and state is not `ready`:

```
┌─────────────────────────────────────────────────────────────────┐
│  Approve PR #6389?                                              │
│                                                                 │
│  No brief exists for this PR yet. Your options:                │
│                                                                 │
│  [  Generate brief first (~2 min)  ]   ← primary, accent color │
│  [  Approve anyway                 ]   ← secondary, muted      │
│  [  Cancel                         ]   ← tertiary, text link   │
│                                                                 │
│  Tip: Press "a" twice to bypass this check and approve.        │
└─────────────────────────────────────────────────────────────────┘
```

Keyboard handling:
- `g` → Generate brief first
- `a` → Approve anyway
- `Esc` → Cancel
- `a` twice within 500ms → bypass modal entirely (for trusted PRs)

When brief state IS `ready`:
- If verdict `approve_candidate` → skip modal, approve directly (current behavior)
- If verdict differs from user's intent → show modal with the verdict context: "Brief says needs_human_attention — approve anyway?"

## Card pulse animation

While `state === 'running'` for a selected card:
- PR number badge gets a subtle pulse animation (opacity 0.6 → 1.0 over 1.2s, infinite)
- A small spinner icon appears next to the badge
- Tooltip on hover: "Brief generation in progress — Findings phase (3/8)"

Use the existing `--accent` color for the pulse; no new color tokens.

## Hook surface

New methods on `useReviewQueue`:

```typescript
// Trigger generation (matches POST /brief/generate)
generateBrief(prNumber: number, options?: { force?: boolean }): Promise<void>;

// One-shot state fetch (matches GET /brief/state)
getBriefState(prNumber: number): Promise<BriefLifecycleState>;

// Cancel (matches DELETE /brief/generate)
cancelBriefGeneration(prNumber: number): Promise<void>;
```

And a separate polling hook:

```typescript
// React hook — polls every 3s while state is 'queued' or 'running'
useBriefState(prNumber: number): {
  state: BriefLifecycleState;
  phase?: string;              // e.g., "findings" | "critique" | "synthesis"
  rolesComplete?: number;      // for progress indicator
  rolesTotal?: number;
  elapsedSeconds?: number;
  costUsdSoFar?: number;
  reason?: string;             // for failed state
};
```

## Feature-flag behavior

`ARAGORA_PDB_BRIEF_GENERATION_ENABLED=1` is read from the backend; frontend doesn't need its own flag. When the backend returns 503 to `POST /brief/generate`, the UI should:

- Not show the "Generate brief" button at all (hide it)
- Fall back to the current `window.confirm` "Approve without PDB brief?" UX (preserving existing behavior for users without the flag enabled)
- Log the 503 once to console at INFO level, not as an error

The feature-flag detection should happen once on mount (via a `GET /review-queue/config` or similar lightweight probe) — not on every click.

## Acceptance criteria

PR 4 is done when:

- All 6 lifecycle states render correctly in BriefPanel (mock backend responses for tests)
- Clicking Generate brief triggers `POST /brief/generate` and the panel immediately shows `queued` state
- Polling updates the UI as state transitions through `queued → running → ready` (tested with fake timers)
- ApproveDecisionModal replaces the current `window.confirm` for the no-brief case
- Keyboard shortcuts work: `g` / `a` / `Esc` / `a×2` bypass
- Running-state pulse animation works (visual regression test via Playwright snapshot acceptable)
- Feature flag OFF → existing behavior unchanged; no 503 error dialogs ever reach the user
- Existing tests (`ReviewQueueStatsHeader.test.tsx`, `ThemeToggle.test.tsx`, etc.) still pass

## Non-goals for PR 4

- Settings UI for configuring the panel — that's a future PR
- Brief editing / manual verdict override — future PR
- Background pre-generation on page load — Mode 2 territory
- Webhook-triggered generation — Mode 1 territory
- Multi-PR bulk generate — future UX iteration
- Mobile-specific layouts — the current review-queue page is desktop-first; no change
- Accessibility audit beyond keyboard shortcuts + aria labels — ship and iterate

## Follow-on

After PR 4 lands, the Mode 3 rollout is complete. Then:

- Dogfood on founder's own queue for a week; measure time-to-decision
- Promote feature flag to default-on
- Start Mode 2 design (batched/scheduled generation on issue-open)
- Then Mode 1 (webhook on PR-open)

## Dependencies

- **Requires**: PR 3 endpoints (`POST/GET/DELETE /brief/generate`) must land first
- **Builds on**: the PR 1 state machine (`BriefLifecycleState` enum from `aragora.pdb.brief_state`)
- **Uses**: the existing `BriefPanel` / `ReviewQueueCard` / `useReviewQueue` hooks, extending rather than rewriting

## Risk notes

- **Hooks rewrite risk**: current `useReviewQueue` is used by the main page. Any change to its return type breaks callers. Keep existing methods untouched; only ADD new methods.
- **Race conditions**: user clicks Generate brief twice quickly. Backend dedupes via PR 3's `(repo, pr_number, head_sha)` key; UI should also dedupe via local `useRef` to avoid visual flicker.
- **Polling waste**: polling every 3s for 2 minutes = 40 HTTP calls per generation. Acceptable for single-user founder-facing. Reconsider when scaling to multi-user.
- **Stale brief surprise**: if user generated a brief yesterday and approves today, the `stale` path must always invalidate before the Approve action completes — otherwise old verdict gets used for today's commit.
