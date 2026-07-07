# Prompt-Handoff Publication State Machine

Status: structural settlement plan for PR #8948. This document is a design
artifact only; it does not implement the publisher change.

Target PR: #8948, head `f1771ccd480256dbac4393021165782f72831ca5`.

## Problem

PR #8948 adds prompt handoffs to the automation outbox. The current-head
dry-run was adjudicated as `adjudicated_escalate` because the publisher can
produce a repo-visible issue that is not actually eligible for dispatch:

- `scripts/publish_automation_handoffs.py` creates prompt-handoff issues
  without consumer-facing labels, then posts required prompt artifact comments,
  then calls `_add_issue_labels`.
- `_add_issue_labels` can silently return when `gh issue edit` fails.
- `publish_handoffs` can still write a terminal `published` receipt after that
  label failure, which suppresses retries while the issue lacks `boss-ready`.

The missing abstraction is a publication state machine. Issue creation, artifact
comment publication, label application, and terminal receipts must be one
transactional publication flow rather than independent best-effort steps.

## States

`prepared`
: The outbox JSON is eligible for publication. Prompt artifact paths, prompt
  SHA-256 values, idempotency key, target branch/head fields, and issue body
  bytes have passed local validation. No GitHub issue has been created.

`created`
: The GitHub issue exists, but it is not yet consumable. For prompt handoffs
  this state must not expose dispatch labels such as `boss-ready` unless no
  post-create artifact step is required and labels were applied at create time.

`artifacts_posted`
: Every required prompt artifact comment has been posted and verified against
  the validated artifact bytes and hash. The issue is still not terminal unless
  the required labels are already confirmed.

`labeled`
: All consumer-facing labels required for dispatch are confirmed present on the
  issue. This is the first state where the issue may enter the intended queue.

`published`
: The publication is terminal. A `published` receipt is written only after the
  issue is `labeled` and the created issue URL is known.

`failed_artifact`
: A required prompt artifact comment failed after issue creation. The issue must
  be closed as incomplete or left in an explicit non-terminal state that cannot
  be consumed. No `published` receipt is legal.

`failed_labeling`
: Label application failed after issue creation and any required artifact
  comments. The issue may be visible, but it is not dispatch-ready. No
  `published` receipt is legal.

`parked`
: A retry cannot safely proceed automatically, for example because the issue URL
  cannot be parsed, the issue state cannot be verified, or cleanup failed. The
  handoff remains non-terminal and requires an explicit blocker receipt or
  operator action.

## Transitions

For prompt handoffs with artifact comments:

```text
prepared -> created -> artifacts_posted -> labeled -> published
```

For prompt handoffs that do not require post-create artifact comments:

```text
prepared -> created+labeled -> published
```

The shortcut is allowed only when labels are passed at `gh issue create` time
or the publisher verifies labels before the receipt. A prompt handoff must not
use `created -> published` if label application happens as a silent best-effort
side effect.

Failure transitions:

```text
created -> failed_artifact -> retryable_or_parked
artifacts_posted -> failed_labeling -> retryable_or_parked
failed_artifact -> prepared        # only after incomplete issue cleanup is verified
failed_labeling -> artifacts_posted # only if the existing issue is verified and relabeling is safe
```

## Receipt Semantics

Receipt status is the contract consumed by reconciliation, so it must encode the
publication state precisely:

| State | Receipt status | Terminal | Meaning |
| --- | --- | --- | --- |
| `prepared` | none | no | Outbox record still needs publication. |
| `created` | none or `in_progress` | no | Issue exists but is not consumable. |
| `artifacts_posted` | none or `in_progress` | no | Artifacts are visible; labels still unconfirmed. |
| `labeled` | none or `in_progress` | no | The next write may be terminal. |
| `published` | `published` | yes | Issue is dispatch-ready and labels are confirmed. |
| `failed_artifact` | `blocked` or none | no | Retry is allowed after cleanup proof. |
| `failed_labeling` | `blocked` or none | no | Retry is allowed after label proof. |
| `parked` | `blocked` | no | Operator or owner action is needed. |

Rules:

- A `published` receipt is legal only after label confirmation; publishing
  without confirmed labels is a bug, not a successful handoff.
- A receipt with `status=published` must include `created_issue_url`,
  `idempotency_key`, source file, repo, and enough state to prove the issue was
  labeled.
- `existing_issue` is terminal only when the existing issue is verified open,
  dispatch-ready, and represents the same prompt identity. A closed incomplete
  issue or unlabeled issue never satisfies the handoff.
- A failed label operation must not be collapsed into `published`. It is either
  a raised publish failure with no terminal receipt, or a non-terminal blocker
  receipt that reconciliation treats as retryable or parked.

## Idempotency And Ownership

Prompt handoff identity remains keyed by `idempotency_key` and prompt SHA, not
by issue title alone. Retry rules:

- The idempotency key owns exactly one successful terminal publication.
- A retry may reuse an existing issue only after verifying it is open,
  artifact-complete, and label-complete for the same prompt SHA.
- A retry must ignore closed incomplete issues unless the retry explicitly
  records them as cleanup history.
- If two outbox records share a title but have different prompt SHA values,
  they are distinct handoffs.
- If the publisher cannot prove issue state or label state through GitHub, it
  must fail closed and keep the outbox record active.
- Reconciliation may archive an outbox record only when the terminal receipt
  proves `published`, `existing_pr`, `target_open_pr`, or another terminal
  state appropriate for that handoff class. A prompt handoff with an unlabeled
  issue is not terminal.

## Failure Table

| Failure | Required publisher behavior | Required reconciler behavior |
| --- | --- | --- |
| Artifact path outside `_prompt-artifacts` | Do not create an issue. Emit local blocker. | Keep outbox active. |
| Artifact missing, hash missing, or hash mismatch | Do not create an issue. Emit local blocker. | Keep outbox active. |
| Issue create fails | Raise publish failure. No receipt. | Keep outbox active. |
| Artifact comment fails after issue create | Close incomplete issue if possible, raise publish failure, and write no terminal receipt. | Ignore closed incomplete issue as delivery proof. |
| Incomplete issue close fails | Raise a parked blocker with issue URL and cleanup error. | Keep outbox active until operator or owner resolves cleanup. |
| Issue URL cannot be parsed | Raise parked blocker if artifact posting or cleanup depends on the URL. | Keep outbox active. |
| Label application fails after artifacts | Raise publish failure or write a non-terminal `failed_labeling` receipt. Never write `published`. | Keep outbox active or retry label application against the verified issue. |
| Existing issue found by prompt identity but missing labels | Treat as `failed_labeling`, not `existing_issue`. | Keep outbox active until labels are confirmed. |
| Existing issue found by prompt identity but closed incomplete | Ignore as delivery proof; retry or park with cleanup history. | Keep outbox active. |
| Receipt write fails after successful label confirmation | Do not republish blindly; next run must detect the verified labeled issue by prompt identity and write a terminal `existing_issue` or `published` receipt. | Keep outbox active until receipt exists. |

## Adoption For PR #8948

The minimal conforming code change for PR #8948 is one of:

1. Treat post-create label failure as fatal and retryable for prompt handoffs.
   `_add_issue_labels` must report failure to `_create_issue`; `_create_issue`
   must raise or return a non-terminal state; `publish_handoffs` must not call
   `_write_receipt` with reason `published` unless labels are confirmed.
2. When no artifact comments are required, apply dispatch labels at issue-create
   time and verify the created issue is labeled before writing `published`.
   Artifact-backed prompt handoffs still defer labels until all required
   artifact comments land.

Either path must add focused tests for:

- label command failure after artifact comments does not write a `published`
  receipt;
- label command failure leaves the outbox handoff retryable or explicitly
  parked;
- a labeled, artifact-complete existing prompt issue can satisfy retry;
- an unlabeled or closed incomplete prompt issue cannot satisfy retry.

After the conforming code change lands, #8948 needs a fresh exact-head dry-run.
The existing artifact `/tmp/ev_8948_f1771cc_dry_20260707T1251Z.json` remains a
blocked crux record and must not be applied as countable evidence.
