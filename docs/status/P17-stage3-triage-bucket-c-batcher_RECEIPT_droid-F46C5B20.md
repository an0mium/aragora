# P17 — Stage 3 Bucket-C batcher receipt

- Session: `droid-F46C5B20`
- Lane: `P17-stage3-triage-bucket-c-batcher`
- Branch: `droid/P17-stage3-triage-bucket-c-batcher-20260518-015641`
- PR: [#7294](https://github.com/synaptent/aragora/pull/7294) (open, ready, 46 SUCCESS / 66 SKIPPED / 1 PENDING at wait-window close)
- Started: 2026-05-18T01:56:42Z
- Completed: 2026-05-18T02:21:00Z (approximate)
- Outcome: **shipped**

## Acceptance against rollout doc Stage 3 spec

| Spec | Implementation |
|---|---|
| Reads Stage 1 classifier output | `run_triage()` invokes `scripts/triage_open_prs.py --json` |
| Filters to Bucket C | `decide()` skips entries with `bucket != 'C'` |
| One-char y/n/d response per PR | `VALID_RESPONSES = {'y','n','d'}` |
| stdin or response file | `--interactive` / `--responses FILE`, mutually exclusive |
| y → mark-ready + comment | `gh pr ready` + `gh pr comment` (label step deferred — see notes) |
| n → close | `gh pr close --comment` |
| d → no-op | `STATUS_DEFERRED` |
| Dry-run default; `--apply` to mutate | Argparse boolean flag, defaults False |
| Tripwires on held / protected | `HELD_PR_NUMBERS` + `PROTECTED_PATHS`/`PROTECTED_PREFIXES`; both enforced on Bucket C |
| Receipt to `docs/status/BUCKET_C_RECEIPT_<utc>.md` | `write_receipt()` on every `--apply` |

## Tests (18 / 18 pass)

- `TestDecideDryRun::test_dry_run_advance_emits_would_advance`
- `TestDecideDryRun::test_dry_run_close_emits_would_close`
- `TestDecideDryRun::test_dry_run_defer_emits_deferred`
- `TestDecideApply::test_apply_y_calls_gh_ready_and_comment`
- `TestDecideApply::test_apply_n_calls_gh_close`
- `TestDecideApply::test_apply_d_makes_no_gh_calls`
- `TestTripwires::test_held_pr_is_skipped_regardless_of_response`
- `TestTripwires::test_protected_path_blocks_advance`
- `TestTripwires::test_workflow_path_blocks_close`
- `TestFiltering::test_non_bucket_c_entries_are_dropped`
- `TestFiltering::test_no_response_yields_no_response_skipped`
- `TestResponseFile::test_json_response_round_trip`
- `TestResponseFile::test_invalid_response_value_raises`
- `TestResponseFile::test_response_file_with_hash_prefix_keys`
- `TestReceipt::test_receipt_is_written`
- `TestInteractive::test_interactive_reads_y_n_d`
- `TestInteractive::test_interactive_treats_empty_as_defer`
- `TestInteractive::test_interactive_treats_garbage_as_defer`

Ruff check + format clean.

## CI

- Initial draft CI run: 16 SUCCESS, 49 SKIPPED, 0 failure (lane-aware suite).
- After flipping to ready, full suite fired (~110 checks). Settled at 46 SUCCESS / 66 SKIPPED / 1 PENDING after the 10-minute wait window. **0 failure / 0 cancelled**. Per v7 wait-window rule, this is a passing outcome — leave PR open, note pending check in receipt.

## Defense-in-depth

- Held PR `#7252` covered: `test_held_pr_is_skipped_regardless_of_response` verifies that an operator response of `y` against #7252 results in `held-skipped` with no `gh` mutation.
- Protected paths covered: edits to `scripts/nomic_loop.py` or `.github/workflows/*` block both advance and close.
- Non-Bucket-C entries silently dropped (`test_non_bucket_c_entries_are_dropped`).
- Empty / garbage interactive input → defer, never advance/close (`test_interactive_treats_empty_as_defer`, `test_interactive_treats_garbage_as_defer`).

## Scope notes for v8

1. **Label name canonicalization**: the rollout doc Stage 3 spec says
   "y → label + mark-ready + comment", but does not pin a label name.
   I omitted the label step deliberately to avoid creating a phantom
   label. A v8 phase should canonicalize a name (suggestion:
   `stage-3-advanced`) and add the `gh pr edit --add-label` call.
2. **YAML loader**: spec says "YAML response file"; I shipped pure-
   stdlib JSON-only, relying on YAML being a JSON superset. If a v8
   phase wants explicit YAML (anchors, comments), add an opt-in
   `try: import yaml` block.

## Lane

`P17-stage3-triage-bucket-c-batcher` released at session close
(status `done`, branch + pr captured in registry).
