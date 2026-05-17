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
```

For each pinned PR, choose one decision:

- approve the captured tier
- approve with a downgraded tier
- request changes
- reject
- hold

The page downloads an `operator-decisions-*.json` file. The downloaded payload
includes the source receipt hash and a SHA-256 binding over the selected
decisions. It does not call GitHub, mutate PRs, install anything, or write to
the repository.

If a browser blocks local file fetches, serve the directory with the command
above or use the page's manual receipt-file picker.
