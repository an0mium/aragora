---
prompt_id: mse_04_pr_6883_and_codex
class: multi_seeded_error
seeded_errors:
  - description: "Claims PR #6883 is stacked on PR #6878. Actually it is stacked on PR #6855."
    category: factual
    verification_ref: "PR #6883 base-branch"
  - description: "Claims with_model('codex', model) silently falls back to default. Actually it raises ValueError because codex exec does not expose --model."
    category: behavioral
    verification_ref: "PR #6883 implementation"
expected_flags: 6
verification_refs:
  - "PR #6883"
---

PR #6883 ("heterogeneous dialog model factories") was opened during round 30e. It introduced named-model factories (`claude_opus`, `droid_gpt5`, etc.) and a generic `with_model(cli, model, ...)` constructor on top of the cross-agent dialog harness. The PR is stacked on PR #6878.

`with_model("codex", "gpt-5.4")` silently falls back to the default codex model (gpt-4.1-codex), since codex exec does not expose a per-invocation `--model` flag. This silent fallback was deliberate — round 30e Phase B verified the panelists succeed even when one panelist's model pin is ignored, since the panel composition still yields heterogeneity at the CLI/wrapper level.

Review for accuracy. List any errors you find.
