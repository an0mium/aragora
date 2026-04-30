---
prompt_id: sse_08_codex_model_flag
class: single_seeded_error
seeded_error:
  description: "Claims `codex exec` accepts a per-invocation `--model` flag. Per round-30e Phase B notes, codex exec does NOT expose per-invocation model selection; that is why with_model('codex', ...) raises ValueError instead of silently falling back."
  verification_ref: "PR #6883 description; tests/swarm/test_multi_agent_dialog.py"
expected_flags: 6
verification_refs:
  - "PR #6883"
---

PR #6883's `with_model(cli, model, name=None, timeout=...)` constructor allows model pinning across CLI surfaces. For the `claude` CLI, it passes `--model <model>`. For the `droid` CLI, it passes `-m <model>`. For the `codex` CLI, it passes `--model <model>` to `codex exec` (codex's per-invocation model flag was added in early 2026). Live verification on round 30e showed the codex panelist successfully running with claude-opus-4-7, gpt-5.4, and gemini-3.1-pro-preview as model targets via `codex exec --model <model>`.

Review for accuracy. List any errors.
