import PacketsClient from './PacketsClient';

/**
 * `/review-queue/packets/[receiptId]` — operator settlement-packet
 * sign-off surface.
 *
 * Loads a settlement-packet receipt (shape
 * `aragora-open-queue-settlement/1.0`) and lets the operator record a
 * per-PR decision (5-tier sign-off) + comment, then download a
 * SHA-256-bound decision JSON for archival or commit. No mutation of
 * any PR; download-only output.
 *
 * The `receiptId` route param is a hint for the receipt-file basename
 * (e.g. `open-queue-settlement-20260517T142811Z`). The actual receipt
 * is loaded client-side via a file picker so the page works
 * offline-first and doesn't depend on Next.js public-asset hosting.
 *
 * Part 3 of 3 in the operator-chosen refactor of #7266 (standalone
 * HTML operator UI) into enhancements to the existing /review-queue
 * UI; depends on PR #7274's `useReviewQueueFromPacket` hook + PR
 * #7273's `tier?` field on `ReviewQueuePR`. After this lands, #7266
 * will be closed as superseded.
 */

// Allow runtime receipt IDs while still providing a fallback static
// export path. Real receipt IDs are resolved client-side via
// `useParams()` (mirrors the spectate/[debateId] route pattern).
export const dynamicParams = true;

export async function generateStaticParams() {
  // Return a placeholder so the static export has at least one path.
  return [{ receiptId: '_' }];
}

export default function ReviewQueuePacketsPage() {
  return <PacketsClient />;
}
