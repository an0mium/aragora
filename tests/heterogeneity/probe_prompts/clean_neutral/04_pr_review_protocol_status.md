---
prompt_id: cn_04_pr_review_protocol_status
class: clean_neutral
seeded_error: null
expected_flags: 0
verification_refs:
  - "aragora/swarm/pr_review_protocol.py (PROTOCOL_STATUS)"
  - "docs/status/STATUS.md"
---

The Aragora `aragora/swarm/pr_review_protocol.py` module declares `PROTOCOL_STATUS = "metadata_heuristic"`. This is the schema-only fallback default for PR review packets constructed without execution. The active heterogeneous-ensemble execution path now lives in `aragora/pdb/` and `aragora/brief_engine/`. That path invokes heterogeneous providers, preserves real dissent, and emits execution statuses distinct from `metadata_heuristic`.

Issue #6374 ("upgrade pr_review_protocol.py from metadata_heuristic to heterogeneous_ensemble_v1") was closed as of 2026-04-25, with the resolution being source-of-truth alignment: docs and any schema-only callers now treat the PDB path as canonical and the metadata-only packet as an explicit fallback.

Review for accuracy. Flag errors if present.
