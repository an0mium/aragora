---
prompt_id: cn_07_dic14_claim_runner
class: clean_neutral
seeded_error: null
expected_flags: 0
verification_refs:
  - "PR #6885 (DIC-14 executable claim runner)"
  - "aragora/reasoning/claim_runner.py"
---

DIC-14 introduced an executable claim runner at `aragora/reasoning/claim_runner.py`. The runner takes a structured claim with optional verification scripts and executes them in a bounded subprocess, capturing stdout/stderr/return code into a `ClaimRunReceipt`. It composes against the existing belief-and-provenance subsystem rather than defining a parallel claim type.

PR #6885 landed the skeleton (~200 LOC plus 18-22 tests) as a Tier 2 PR. The runner is read-only against repo state by default, with explicit opt-in for any mutation.

Review for accuracy. Flag errors if present.
