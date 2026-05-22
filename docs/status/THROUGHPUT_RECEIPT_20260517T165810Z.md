# Throughput receipt — /review-queue/packets keyboard sign-off

**Generated:** 2026-05-17T16:58:10Z
**Branch:** `worktree-packets-keyboard-throughput-20260517`
**Base:** PR #7277 (stacked; depends on it landing first)
**Surface:** `/review-queue/packets/[receiptId]` settlement-packet sign-off
**Schema change:** `aragora-operator-decisions/1.0` payload now carries
`first_focused_at_utc`, `decided_at_utc`, `decision_seconds` per entry
(additive — `payload_sha256` covers them).

## What changed

1. Added keyboard navigation to `PacketsClient`:
   - `j` / `↓` next card, `k` / `↑` prev card
   - `1`..`5` pick `PACKET_DECISION_OPTIONS` in order
     (1=approve_tier · 2=approve_downgrade · 3=request_changes ·
      4=reject · 5=hold_operator)
   - `Tab` focus comment textarea on selected card
   - `?` / `Shift+/` toggle help overlay
   - `Esc` close help
   - Editable-element detection suppresses key handling inside textareas
2. Added `selected?: boolean` + `onSelect?: () => void` props to
   `PacketDecisionCard` with visual highlight matching the live
   `ReviewQueueCard` selected style (accent border + glow).
3. Added per-decision timing: capture `first_focused_at_utc` when a
   card becomes the keyboard focus target for the first time, capture
   `decided_at_utc` on each `setDecisionFor` call. Both persist in the
   downloaded JSON's `decisions[]` entries, along with derived
   `decision_seconds`.
4. Surfaced live median seconds-per-decision in the receipt summary
   (`packets-median-decision-seconds` testid).

## Before / after operator action count (5-PR sign-off pass)

### Before (mouse-only)

For each PR:
- 1 click on a radio button (target ~150px wide, requires aim)
- 1 mouse repositioning to comment textarea
- 1 click on the textarea to focus
- (optional) ~20 char comment typing
- 1 mouse repositioning to next card / scroll

**Per-decision actions:**
- Without comment: **2 clicks + 2 mouse repositionings**
- With comment: **2 clicks + 2 mouse repositionings + 20 keystrokes**

**5-PR pass without comments:** 10 clicks + 10 repositionings + 0 keystrokes.

### After (keyboard-only)

For each PR:
- 1 digit key (decision)
- 1 `j` key (next card)
- (optional) 1 `Tab` key + ~20 char comment + nothing extra

**Per-decision actions:**
- Without comment: **2 keystrokes**
- With comment: **2 keystrokes + Tab + 20 keystrokes = 23 keystrokes**

**5-PR pass without comments:** 10 keystrokes total, 0 clicks, 0 mouse repositionings.

### Estimated seconds saved per decision

Baseline assumption: **0.5s per click + mouse repositioning event**
(Fitts'-Law lower-bound for a ~150px on-screen target at moderate viewport
distance; a real measurement would be ~0.8–1.2s with cognitive load).
Keystrokes for single-character shortcuts: ~0.15s each (no Fitts-Law cost —
fingers are already on the home row).

| Scenario | Before (action time) | After (action time) | Saved per decision |
|---|---|---|---|
| Approve without comment | 2 × 0.5 + 2 × 0.5 = **2.0s** | 2 × 0.15 = **0.3s** | **1.7s** |
| Approve with 20-char comment | 2.0s + (20 × 0.1 keystroke) = **4.0s** | 0.3s + Tab + 20 × 0.1 = **2.45s** | **1.55s** |

Conservative net: **~1.5s saved per decision** for the dominant approve-with-comment path.
On a 15-PR settlement packet (this morning's open-queue settlement size):
**~22s saved per pass**, every pass.

The new `first_focused_at_utc` + `decided_at_utc` fields make this empirical
in future sessions — operators can compute the actual median by reading any
downloaded `operator-decisions-*.json`.

## Files touched (additive only)

- `aragora/live/src/app/(app)/review-queue/packets/[receiptId]/PacketsClient.tsx`
- `aragora/live/src/components/review-queue/PacketDecisionCard.tsx`
  (new `selected?: boolean`, `onSelect?: () => void` props; otherwise unchanged behaviour)
- `aragora/live/__tests__/PacketDecisionCard.test.tsx` (+ 4 tests)
- `aragora/live/__tests__/ReviewQueuePacketsPage.test.tsx` (+ 6 tests)
- `docs/status/THROUGHPUT_RECEIPT_20260517T165810Z.md` (this file)

## Validation

```
$ cd aragora/live && npx jest __tests__/PacketDecisionCard \
    __tests__/ReviewQueuePacketsPage __tests__/ReviewQueue --no-coverage
Test Suites: 8 passed, 8 total
Tests:       70 passed, 70 total
$ npx tsc --noEmit         # clean
$ npx eslint <changed>     # clean
$ bash scripts/automation_pr_preflight.sh origin/main HEAD
preflight: ok
```

## Holds respected

- No PR mutation, no labels, draft only.
- Zero AI-key consumption.
- No `automation.toml` edit, no launchd install.
- No held PR advancement (#7173, #7215, #7240, #7243, #7245, #7249,
  #7252, #4990, BC-12 soak, #7209 lane).
- All output is local-only — the download path is the operator's browser;
  nothing is sent.

## Reproduction

```
cd aragora/live
npm install --prefer-offline --no-audit --no-fund --silent
npx jest __tests__/PacketDecisionCard __tests__/ReviewQueuePacketsPage \
   __tests__/ReviewQueue --no-coverage
```

Manual end-to-end (after `npm run dev` and routing to
`/review-queue/packets/test`):

1. Pick any `aragora-open-queue-settlement/1.0` receipt JSON via the
   file picker (e.g. `docs/receipts/open-queue-settlement-*.json`).
2. Hit `?` to confirm help overlay renders.
3. Press `1`, then `j`, then `4`, then `j`, then `5` — confirm three
   decisions land on the first three PRs with the selection highlight
   advancing each time.
4. Click Download decisions JSON; open the downloaded file and verify
   the `decisions[]` entries carry `first_focused_at_utc`,
   `decided_at_utc`, `decision_seconds`, and the payload-level
   `payload_sha256` is a 64-char hex string.

## Receipt self-binding

The SHA-256 of this file is computed after content is finalized via:

```
sha256sum docs/status/THROUGHPUT_RECEIPT_20260517T165810Z.md
```

The PR description and final response print the resulting hex.
