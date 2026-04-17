# Inbox Trust Wedge Proof Pack

Use this template for the compact artifact that closes the internal-dogfood
gate and supports a bounded design-partner conversation.

## 1. Run Window

- Window start:
- Window end:
- Operators:
- Inbox/account scope:
- Workflow version / branch / commit:
- Related issue or tranche IDs:

## 2. Workflow Statement

Describe the exact workflow that was proven in one sentence:

`real inbox -> debate-backed triage -> persisted receipt -> approval/review -> provider action`

Allowed actions during this window:

- `ARCHIVE`
- `STAR`
- `LABEL`
- `IGNORE`

Explicitly excluded actions:

- reply
- send
- forward

## 3. Activation Scoreboard

| Gate | Target | Actual | Pass/Fail | Evidence |
|---|---|---|---|---|
| Consecutive live runs | `10` over at least `5` business days |  |  |  |
| Non-builder operators | `2` operators |  |  |  |
| Time to first useful result | `<=10m` |  |  |  |
| Accepted action rate | `>=70%` over `2` weeks |  |  |  |
| Truthful-stop coverage | `100%` of non-accepted runs |  |  |  |
| False-success incidents | `0` |  |  |  |
| Receipt-before-action | `100%` |  |  |  |

## 4. Exact Commands

List the exact commands used during the proof window.

```bash
python3 -m aragora.cli.main triage status
python3 -m aragora.cli.main triage auth
python3 -m aragora.cli.main triage run --batch 5 --dry-run
python3 -m aragora.cli.main triage run --batch 5
python3 -m aragora.cli.main inbox-wedge list --limit 20
python3 -m aragora.cli.main inbox-wedge report --limit 200
python3 -m aragora.cli.main triage calibrate --json
```

## 5. Receipt Bundle

List representative receipts and why they matter.

| Receipt ID / path | Outcome | Why included |
|---|---|---|
|  | accepted action |  |
|  | truthful stop |  |
|  | override/review example |  |

## 6. Outcome Summary

| Metric | Value | Notes |
|---|---|---|
| Total runs |  |  |
| Total emails processed |  |  |
| Accepted actions |  |  |
| Truthful stops |  |  |
| Overrides |  |  |
| Important-email misses |  |  |
| Average latency per email |  |  |
| Average cost per email |  |  |

## 7. Truthful Stops

For every non-accepted run, record the blocker and next action.

| Run / receipt | Blocker class | Next action | Was it truthful? |
|---|---|---|---|
|  |  |  |  |

## 8. Operator Transferability

Summarize the non-builder operator results.

| Operator | Time to auth | Time to first useful result | Blockers hit | Outcome |
|---|---|---|---|---|
|  |  |  |  |  |
|  |  |  |  |  |

## 9. Before / After Evidence

Capture the smallest honest delta versus the incumbent path.

- What was slower, riskier, or less reviewable before:
- What is better now:
- What still requires human attention:

## 10. Remaining Risks

- Risk 1:
- Risk 2:
- Risk 3:

## 11. Gate Decision

Choose exactly one:

- `internal_ready`
- `partner_dry_run_ready`
- `not_ready`

Decision rationale:

## 12. Next Bounded Actions

- Action 1:
- Action 2:
- Action 3:
