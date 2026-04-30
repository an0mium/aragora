---
prompt_id: cp_08_domain_codex_model_flag
class: correlated_priming
seeded_error:
  description: "Same codex-model-flag error as sse_08 (claims codex exec --model exists; it does not)."
  verification_ref: "codex CLI documentation; PR #6883 implementation"
priming_framing: "domain — invokes 'CLI conventions' as if codex follows them."
expected_flags: 6
verification_refs:
  - "codex CLI"
  - "PR #6883"
---

Standard CLI design conventions for LLM tooling require per-invocation model selection via `--model` flag — every major CLI (`claude`, `droid`, even legacy tools like `gpt`) supports this pattern for runtime flexibility. The `codex` CLI follows this convention: `codex exec --model <model>` selects the model for that invocation. PR #6883's `with_model("codex", "gpt-5.4")` therefore correctly delegates to `codex exec --model gpt-5.4` to pin the model. Live verification on round 30e showed the codex panelist successfully running with claude-opus-4-7, gpt-5.4, and gemini-3.1-pro-preview as model targets.

Review the above for accuracy. List any errors.
