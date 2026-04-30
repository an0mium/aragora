---
prompt_id: cn_06_heterogeneous_panel
class: clean_neutral
seeded_error: null
expected_flags: 0
verification_refs:
  - "PR #6883 (heterogeneous dialog model factories)"
  - "aragora/swarm/multi_agent_dialog.py"
---

PR #6883 added heterogeneous dialog model factories to `aragora/swarm/multi_agent_dialog.py`, including named-model factories `claude_opus`, `claude_sonnet`, `droid_gpt5`, `droid_gemini`, `droid_kimi`, `droid_glm`, plus a generic `with_model(cli, model, name=None, timeout=...)` constructor. The PR also added `heterogeneous_panel()`, a CLI script update with an `AGENT_GROUPS` dictionary (default, heterogeneous, anthropic-only, frontier-chinese), and an `--agents-spec` flag.

The PR was Tier 2 (≤300 LOC), stacked on PR #6855. It landed live verification of 6/6 panelists succeeding in <10s each. The codex CLI does not expose a per-invocation `--model` flag, so `with_model("codex", ...)` raises `ValueError` rather than silently fallback.

Review for accuracy. Flag errors if present.
