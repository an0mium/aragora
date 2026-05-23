# Settlement Packet Sign-off UI

This directory contains static, non-mutating operator worksheets for settlement
packets. The first worksheet is:

- `2026-05-17-open-queue-settlement-ui.html`

It is intentionally separate from the live `/review-queue` app. That app can
approve, request changes, and defer PRs through the review-queue backend. This
static page only helps the operator record decisions against an already pinned
settlement receipt.

## Use

From the repository root:

```bash
python3 -m http.server 8765 --directory docs
```

Then open:

```text
http://127.0.0.1:8765/status/settlement-packets/2026-05-17-open-queue-settlement-ui.html
```

The page loads:

```text
docs/receipts/open-queue-settlement-20260517T142811Z.json
docs/status/settlement-packets/2026-05-17-open-queue-settlement-context.json
```

For each pinned PR, read the "What this does", red flags, risk statement, and
safe default before choosing a decision:

- approve the captured tier
- approve with a downgraded tier
- request changes
- reject
- hold

Known duplicate/conflict groups are shown above the PR cards. Clustered PRs
intentionally make approval awkward until a cluster-level choice is selected.
This is meant to prevent blind approval of mutually exclusive or merge-order
sensitive PRs.

The page downloads an `operator-decisions-*.json` file. The downloaded payload
includes:

- a statement that the artifact does not mutate GitHub
- the source receipt hash
- the source context hash
- cluster choices
- per-PR decisions, red flags seen, and notes
- a SHA-256 binding over the selected decisions and cluster choices

The download is only evidence. It does not call GitHub, mutate PRs, install
anything, edit `automation.toml`, label, mark ready, close, merge, approve, or
request changes.

Use the downloaded file by attaching it to a follow-up operator prompt, or by
committing it later under a future `docs/receipts/operator-decisions/` path as
explicit operator evidence for a separate queue-drain lane.

If a browser blocks local file fetches, serve the directory with the command
above or use the page's manual receipt-file picker.

## Deployment note

This worksheet is intentionally separate from `/review-queue`, which is backed
by mutating review-queue endpoints. A redacted static copy can be exposed on
`docs.aragora.ai` or `aragora.ai` later, but the raw packet/context should not be
published publicly until it is reviewed for private PR-comment content. If this
ever moves into the live app, it should be a separate non-mutating settlement
mode, not the existing approval/request-changes/defer workflow.
