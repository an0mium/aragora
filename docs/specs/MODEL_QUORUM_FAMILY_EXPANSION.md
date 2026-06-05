# Model Quorum Family Expansion (Pre-Approval Design, Tier 4 implementation)

**Status:** operator preapproval recorded; PR-A2 implementation pending
**Owner:** Armand
**Date:** 2026-05-24
**Related:** `docs/REVIEW_AUTHORITY_PRINCIPLES.md`,
`docs/briefs/automation-merge-contract.md`,
`aragora/cli/commands/review_queue.py` (Tier 4 surface),
`aragora/swarm/pr_review_protocol.py`,
PR #7438 (the gap that motivated this)

## Why this exists

When PR #7438 attempted to clear its model-quorum gate, the reviewer
executor produced one partial signal (Claude logic reviewer, returned
`defer`) and zero countable signals from the other configured slots
(codex CLI stdin EAGAIN, gemini-cli auth missing, grok-cli API key
missing, mistral provider absent). Even with all four CLIs technically
"available on PATH", the quorum count was 0/2.

A separate but adjacent gap surfaced during that attempt: even when
heterogeneous Chinese open-weight families like DeepSeek, Qwen, and Kimi
*are* available via OpenRouter (and aragora's `api_agents/openrouter.py`
already wires them), they cannot be *counted* as reviewer signals because
`aragora/cli/commands/review_queue.py::_infer_model_reviewer_from_text`
only recognizes the marker substrings `claude / codex / tesla / harvey /
factory / grok / gemini`. The broader normalizer knows about OpenAI,
Mistral, DeepSeek, Qwen, and Kimi, but that code path never runs for PR
comments once the recognizer returns `unknown_model_reviewer`. The new
May-2026 Western variants (Gemini 3.5 Flash, Grok 4.3) also need explicit
family mapping.

This document proposes the narrowest viable expansion of the reviewer
pool and the governance contract it must satisfy. **It is the pre-
approval artifact, not the implementation.** The actual code change to
`review_queue.py` is Tier 4 per `docs/REVIEW_AUTHORITY_PRINCIPLES.md`
("merge-authority self-modifications") and requires human preapproval
before implementation AND before merge.

## Scope

### In scope (Tier 1, no operator preapproval needed)
1. This spec doc
2. A new privacy-tier-by-jurisdiction subsection in
   `docs/REVIEW_AUTHORITY_PRINCIPLES.md` codifying which model families
   are eligible for which Tier
3. Current-state governance tests in `tests/governance/` demonstrating
   the current recognizer gap on the families enumerated below

### Out of scope for this PR (Tier 4, requires operator nod)
1. The actual `_infer_model_reviewer_from_text` patch
2. The new "open-weight skeptic" slot in `default_pr_review_protocol`
3. Any change to `aragora/swarm/pr_review_protocol.py` slot definitions

### Explicitly out of scope (separate PRs)
- Provider bootstrap fixes (gemini-cli auth, codex CLI stdin EAGAIN) —
  environmental, not gate-modification
- Bench harness for cost/quality comparison (PR-B, follow-on)
- Any wiring of these reviewers into operator-facing surfaces

## Current state inventory

### The recognizer gap is *bigger* than initially documented

While drafting this spec the governance tests in
`tests/governance/test_model_quorum_recognizer_gaps.py` surfaced an
inconvenient truth: `_infer_model_reviewer_from_text` in
`aragora/cli/commands/review_queue.py` only matches **seven** markers
today:

```python
for name in ("claude", "codex", "tesla", "harvey", "factory", "grok", "gemini"):
    if name in lower:
        return name
return "unknown_model_reviewer"
```

So even families that aragora *already pays for* (OpenAI, Mistral,
DeepSeek, Qwen, Kimi via OpenRouter) cannot post a counted reviewer
signal today. A second downstream function
(`_normalize_model_reviewer_id` in the same file) has the broader list
with `anthropic / openai / gpt / deepseek / qwen / kimi / mistral`
markers, but it is only consulted *after* the recognizer already
returned `"unknown_model_reviewer"` — so it is dead code for those
families. This is the root of PR #7438's 0/2 model-quorum count even
when CLI providers were technically available.

### Family wiring + recognizer coverage table

| Family | Wired in `api_agents/` | Recognized today? | Proposed for quorum |
|---|---|---|---|
| Anthropic / Claude | ✓ `anthropic/claude-opus-4.7` | ✓ `claude` | ✓ keep |
| OpenAI | ✓ `openai/gpt-5.5` | **✗ NOT recognized** | ✓ add |
| Google Gemini | ✓ `google/gemini-3.1-pro-preview` | ✓ `gemini` | ✓ keep |
| Gemini 3.5 Flash | not yet pinned | ✓ via `gemini` | ✓ wire as routing alias |
| xAI Grok 4 | ✓ `x-ai/grok-4` | ✓ `grok` | ✓ keep |
| xAI Grok 4.3 | not yet pinned | ✓ via `grok` | ✓ wire as routing alias |
| Mistral | ✓ `mistralai/mistral-large-*` | **✗ NOT recognized** | ✓ add (EU-jurisdiction routing only) |
| DeepSeek | ✓ `deepseek/deepseek-v3.2 / r1 / v4-pro` | **✗ NOT recognized** | ✓ add |
| Qwen | ✓ `qwen/qwen3-max / 235b / 3.5-plus` | **✗ NOT recognized** | ✓ add |
| Kimi (Moonshot) | ✓ `moonshotai/kimi-k2.6 / k2.5 / thinking` | **✗ NOT recognized** | ✓ add |
| Yi (01.AI) | ✓ `01-ai/yi-large` | **✗ NOT recognized** | ✓ add |
| Codex (vendor name) | n/a (CLI) | ✓ `codex` | ✓ keep |
| Tesla / Harvey / Factory | n/a (vendor names) | ✓ pinned | ✓ keep |

### NOT YET wired anywhere in `aragora/agents/api_agents/` (new additions)

| Family | OpenRouter id | Why add |
|---|---|---|
| **GLM-4.6** (Zhipu) | `z-ai/glm-4.6` (or `zhipuai/glm-4.6`) | Strong reasoning + agentic; distinct family from DeepSeek; sane prices |
| **MiniMax M2** | `minimax/minimax-m2` | Distinct training lineage; multimodal capable |
| **Nous Hermes 4** (Western open-weight) | `nousresearch/hermes-4-405b` | Western-jurisdiction balance to the Chinese cluster |
| **Gemini 3.5 Flash** (Western, agentic-tier) | `google/gemini-3.5-flash` | 4× faster, $1.50/$9, wins on agentic/coding benchmarks (76% Terminal-Bench 2.1, 84% MCP Atlas), loses on pure reasoning vs 3.1 Pro — both have a place |
| **Grok 4.3** (Western, May 2026) | `x-ai/grok-4.3` | Current xAI/OpenRouter listing with 1M context and low token price; role-restricted to policy/governance reviewer lenses until local evals prove broader coding value |

## Family-by-Tier-by-Jurisdiction privacy contract

This is the "be careful" codification. Adding more reviewer families
without an explicit tier/jurisdiction policy would silently weaken the
gate for high-Tier work.

### Tier-eligibility matrix

| Tier | Western families (Anthropic, OpenAI, Google, xAI, Mistral, Nous) | Chinese-routed families (DeepSeek, Qwen, Kimi, GLM, MiniMax, Yi) |
|---|---|---|
| 0 (docs, status) | ✓ counted | ✓ counted |
| 1 (additive internal, no live caller) | ✓ counted | ✓ counted |
| 2 (live automation, CLI, observability, retry, cache) | ✓ counted | ✓ counted; **at least 1 of the 2 required signals must be Western** |
| 3 (semantic correctness, persistence, security, public API, migrations) | ✓ counted | **Advisory-only — not counted toward quorum** |
| 4 (secrets, deployment, workflow policy, destructive ops, merge-authority self-mod) | ✓ counted — **Western-only counted quorum required** | **Advisory-only — not counted** |

### Privacy-jurisdiction-routing rule

Independent of Tier, the **payload sent to the reviewer** determines
jurisdiction eligibility:

| Payload type | Western families | Chinese-routed families |
|---|---|---|
| PR title + diff (public OSS repo) | ✓ | ✓ |
| PR title + diff (private repo, no PII) | ✓ | ✓ if repo policy permits |
| Inbox triage features (low-information, no body) | ✓ | ✓ |
| Inbox triage *raw email body* | ✓ if AWS Secrets Manager loaded | **✗ never** |
| Customer PII / financials / credentials | ✓ if data-residency policy permits | **✗ never** |
| Secrets, encryption keys, OAuth tokens | ✓ if data-residency policy permits | **✗ never** |
| Private legal material (contracts, settlements, NDAs) | ✓ if data-residency policy permits | **✗ never** |
| Healthcare / regulated data | Vertical-specific allowlist only | **✗ never** |

### Rationale

- **Tier 2 mixed-Western-Chinese quorum:** lets us realize the cost
  advantage of DeepSeek/Qwen/Kimi/GLM on routine automation work while
  preserving one Western alignment-trained reviewer for behavioral
  sanity. Concretely: a 2/2 quorum of `claude` + `deepseek` counts;
  `deepseek` + `qwen` alone does not for Tier 2.
- **Tier 3+ Western-only:** the calibration concerns for high-stakes
  decisions outweigh the cost savings. Chinese models can still post
  advisory comments (and the executor can still query them), but they
  don't gate the merge.
- **PII never routed to Chinese providers:** treats jurisdiction as a
  hard boundary, not a soft preference. This is the same discipline the
  AFT extractor enforces (low-information features only; no diffs, no
  comment bodies, no PII).

## Recognizer changes (Tier 4 — needs preapproval before I implement)

Note: the patch needs to fix `_infer_model_reviewer_from_text` first
(the narrow 7-marker tuple is the actual gate-blocker), then ensure
`_normalize_model_reviewer_id` stays in sync with the recognizer's
output. The two functions diverged at some point and the recognizer is
the bottleneck.

### Patch sketch — `_infer_model_reviewer_from_text` (the gate-blocker)

Replace today's 7-marker tuple:

```python
# CURRENT (gate-blocker):
for name in ("claude", "codex", "tesla", "harvey", "factory", "grok", "gemini"):
    if name in lower:
        return name
return "unknown_model_reviewer"
```

with a table that mirrors `_normalize_model_reviewer_id`'s known-markers
list, so the two stay in sync by construction:

```python
# PROPOSED:
_REVIEWER_MARKERS = (
    ("claude",   ("claude", "anthropic")),
    ("openai",   ("openai", "gpt")),
    ("gemini",   ("gemini", "google")),
    ("grok",     ("grok", "xai", "x-ai")),
    ("mistral",  ("mistral", "codestral")),
    ("codex",    ("codex",)),
    ("deepseek", ("deepseek",)),
    ("qwen",     ("qwen",)),
    ("kimi",     ("kimi", "moonshot", "moonshotai")),
    ("yi",       ("yi-large", "01-ai")),
    ("glm",      ("glm", "zhipu", "z-ai", "zhipuai")),
    ("minimax",  ("minimax",)),
    ("hermes",   ("hermes", "nous", "nousresearch")),
    ("tesla",    ("tesla",)),
    ("harvey",   ("harvey",)),
    ("factory",  ("factory",)),
)
for family, aliases in _REVIEWER_MARKERS:
    if any(a in lower for a in aliases):
        return family
return "unknown_model_reviewer"
```

The same table feeds `_normalize_model_reviewer_id` (refactor: pull
`_REVIEWER_MARKERS` to module-scope, both functions iterate it).

### Patch sketch — `_normalize_model_reviewer_id` (kept in sync)

Same `known_markers` table — see below. Today's version omits
yi/glm/minimax/hermes; the proposed addition extends them:

```python
known_markers = (
    ("claude",   ("claude", "anthropic")),
    ("openai",   ("openai", "gpt")),
    ("gemini",   ("gemini", "google")),
    ("grok",     ("grok", "xai", "x-ai")),
    ("mistral",  ("mistral", "codestral")),
    ("codex",    ("codex",)),
    # --- already-wired Chinese families recognized in recognizer:
    ("deepseek", ("deepseek",)),
    ("qwen",     ("qwen",)),
    ("kimi",     ("kimi", "moonshot", "moonshotai")),
    # --- proposed additions:
    ("yi",       ("yi-large", "01-ai")),
    ("glm",      ("glm", "zhipu", "z-ai", "zhipuai")),
    ("minimax",  ("minimax",)),
    ("hermes",   ("hermes", "nous", "nousresearch")),
    # --- existing markers unrelated to this PR:
    ("tesla",    ("tesla",)),
    ("harvey",   ("harvey",)),
    ("factory",  ("factory",)),
)
```

Plus the family-jurisdiction mapping (new helper):

```python
_WESTERN_FAMILIES = frozenset({
    "claude", "openai", "gemini", "grok", "mistral", "codex", "hermes",
})
_CHINESE_FAMILIES = frozenset({
    "deepseek", "qwen", "kimi", "yi", "glm", "minimax",
})

def _is_western_family(family: str) -> bool:
    return family in _WESTERN_FAMILIES

def _quorum_counts_family(family: str, tier: int) -> bool:
    """Return whether `family` counts toward the quorum at `tier`."""
    if not family:
        return False
    if tier <= 2:
        return True  # all recognized families count
    return _is_western_family(family)  # Tier 3+ Western-only
```

Plus the Tier-2 "at least one Western" rule (one branch in
`_build_model_review_quorum` near the `signal_count >= required` check).

The diff is small (~40 lines including the helpers + one branch). The
Tier 4 escalation is about the *governance significance*, not the line
count.

## Current-state governance tests landed in this PR (Tier 1)

`tests/governance/test_model_quorum_recognizer_gaps.py` — proves the
current state of the gate so the eventual Tier 4 patch has a
machine-checkable gap record to invert:

1. `test_existing_recognizers_still_resolve` — sanity that the seven
   families recognized today (`claude`, `codex`, `gemini`, `grok`,
   `tesla`, `harvey`, `factory`) stay recognized. This is the regression
   floor — the Tier 4 patch must keep these passing.
2. `test_proposed_family_markers_currently_unrecognized` — parameterized
   gap-of-record for already-routed-but-uncounted families (OpenAI,
   Anthropic, Mistral/Codestral, DeepSeek, Qwen, Kimi/Moonshot) plus new
   proposed additions (GLM/Zhipu/Z-AI, MiniMax, Yi, Nous/Hermes).
   After the Tier 4 patch lands this test will need to be inverted (or
   replaced with a positive-recognition suite).
3. `test_unknown_garbage_stays_unknown` — input that mentions no model
   family at all returns `unknown_model_reviewer`. Guards against
   over-eager recognizers that match arbitrary substrings.
4. `test_recognizer_is_case_insensitive` — pins current lowercase
   matching behavior for existing recognized families.

These tests are *governance tests* and live under `tests/governance/` so
they can be required CI for any future change to the recognizer.

## Open questions for operator preapproval

1. **Tier 4 nod:** do you accept the proposed recognizer expansion +
   helper additions in `aragora/cli/commands/review_queue.py` as
   appropriately scoped, with the family/tier rules above?
2. **Family additions:** should `yi`, `glm`, `minimax`, `hermes`,
   `gemini-3.5-flash`, `grok-4.3` all be wired? Or some subset?
3. **Tier 3 "advisory-only" framing:** is Chinese-family advisory
   posting OK at Tier 3 (just not counted toward quorum), or should
   Tier 3 paths suppress Chinese reviewers entirely?
4. **PR author posting recognized markers:** if `an0mium` (PR author)
   posts a comment that mentions a model family, should that count?
   Today's recognizer would catch it; the four-factor independence rule
   says no. Recommend adding an explicit author-exclusion rule in the
   same patch.

## Operator preapproval — answers received 2026-05-24

**Directionally approved. Tighten before marking ready.** Operator
answers (transcribed verbatim from the preapproval message):

1. **Yes**, preapprove a follow-on Tier 4 recognizer patch that
   **replaces the hard-coded reviewer-name loop with a table-driven
   `_REVIEWER_MARKERS`, kept in sync with `_normalize_model_reviewer_id`**.

2. **Include markers for** `anthropic/claude`, `openai/gpt`,
   `gemini/google`, `grok/xai`, `deepseek`, `qwen`, `kimi/moonshot`,
   `glm/z-ai`, `minimax`, `hermes/nous`, `yi`, and `mistral/codestral`.
   **Mistral is retained for EU/regulatory diversity and backward
   compatibility, not because it is a preferred capability reviewer.**

3. **Tier policy:** Tier 0-2 may count Chinese/open-weight families
   if the payload is non-sensitive and exact-head grounded. Tier 3+
   requires Western-family counted quorum; Chinese-routed families
   may post advisory comments but must not satisfy the gate. **Tier 4
   governance changes require Western-only counted quorum.** Raw PII,
   inbox bodies, financials, healthcare, secrets, customer data, and
   private legal material must not be routed to Chinese-routed /
   OpenRouter providers.

4. **Yes**, add explicit PR-author / comment-author exclusion in the
   same patch. A PR author must not be able to satisfy model quorum
   by posting `# Claude review` or equivalent.

These answers are the binding scope for the Tier 4 PR-A2 patch (to be
opened separately from a fresh worktree off origin/main). The
principles-doc tier matrix below has been tightened to reflect
answer #3's stricter Tier 4 wording ("Western-only counted quorum"
replaces my draft "at least one Anthropic-or-OpenAI signal").

## Model ID source-of-truth (verified against provider docs 2026-05-24)

Per the operator's correction "Do not hard-code 'latest' from memory.
Verify against provider docs/API listings and document any mismatch":

| Family | Repo pin (default) | Provider-official current | Status | Action for PR-A2 |
|---|---|---|---|---|
| Anthropic | `anthropic/claude-opus-4.7` (in `api_agents/anthropic.py`) | `claude-opus-4-7` (Opus 4.7 GA per Anthropic blog 2026-04-16) | ✓ **aligned** | none — repo pin matches provider |
| OpenAI | `openai/gpt-5.5` (alias destination in `api_agents/openrouter.py`) | `gpt-5.5` (flagship per OpenAI API docs) | ✓ **aligned** | none |
| Google Gemini | `google/gemini-3.1-pro-preview` (default in `api_agents/gemini.py`) | `gemini-3-pro` (still GA) and `gemini-3.5-flash` (GA 2026-05-20, agentic-tier) | ⚠ **repo does not yet pin 3.5 Flash** | PR-A2 routing alias addition: wire `gemini-3.5-flash` as the agentic-task routing target; keep `gemini-3.1-pro-preview` for reasoning/long-context |
| xAI Grok | `x-ai/grok-4` (default in `api_agents/grok.py`; `grok-4.2` noted as "not yet on OpenRouter") | `grok-4.3` (launched 2026-04-30, ~40% cheaper than 4.20, on OpenRouter) | ⚠ **repo does not yet pin 4.3** | PR-A2 routing alias addition: wire `grok-4.3` for policy/governance reviewer slot; keep `grok-4` for back-compat alias |
| Mistral | `mistralai/mistral-large-*` | (operator's posture: retained for EU/regulatory diversity; not a preferred capability reviewer) | aligned | none on pin; tier-policy demotion captured in principles doc |
| Open-weight (DeepSeek/Qwen/Kimi/Yi) | various (`deepseek/v3.2`, `qwen/qwen3-max`, `moonshotai/kimi-k2.6`, `01-ai/yi-large`) | match provider model pages | aligned | recognizer additions only |
| GLM / MiniMax / Hermes | not yet wired | `z-ai/glm-4.6`, `minimax/minimax-m2`, `nousresearch/hermes-4-405b` | new wirings needed | wire in `api_agents/openrouter.py` alongside recognizer additions |

Sources checked: Anthropic Claude Opus page
(`https://www.anthropic.com/claude/opus`), OpenAI models page
(`https://developers.openai.com/api/docs/models`), Google DeepMind Gemini
3.5 Flash model card
(`https://deepmind.google/models/model-cards/gemini-3-5-flash/`), xAI Grok
4.3 docs (`https://docs.x.ai/developers/models/grok-4.3`), and OpenRouter's
Grok 4.3 listing (`https://openrouter.ai/x-ai/grok-4.3`). The OpenAI
`gpt-5.5` row was rechecked against official OpenAI docs on 2026-05-26
after one stale search snippet briefly suggested an older `gpt-5.2` latest
link; the canonical model index and model detail page both confirmed
`gpt-5.5`.

**Mismatch tally:** **0 hard mismatches** between repo pins and provider-
official IDs. **2 "lagging on newest variant" cases**: Gemini 3.5 Flash
and Grok 4.3 are real and current per provider docs but not yet pinned
in `api_agents/`. These are routing-alias additions, not pin corrections.

I am **not silently blessing the repo pins.** The above table records
my verification against provider docs as of 2026-05-24 and should be
re-verified before PR-A2 is implemented in case provider docs have
shifted (model IDs can change weekly in this space). The
verification command pattern is: `WebFetch` against each provider's
canonical models page, compare to repo `api_agents/<family>.py`
default model and alias map.

## Note on PR-B

The bench harness scaffold PR-B (PR **#7451** on branch
`codex/claude-20260524-042836-bb3096b9`) is already open as a draft
on a separate worktree off origin/main. It contains no gate-policy
changes and no live provider calls in this iteration — only the
stub-backed orchestrator, scoring helpers, synthetic corpus, and
34 unit tests. The operator's "Phase 3" directive ("Use the AFT
pattern: pre-registered hypotheses, fixed eval set, cost/latency
capture, head-grounded outputs") matches PR-B's design; no
additional work is needed to fulfill that directive beyond what is
already in #7451.

## Net assessment

The expansion is technically small (~40 lines + tests) and
substantively meaningful (it unblocks 5+ heterogeneous reviewer
families that aragora *already pays for* via OpenRouter but cannot
*count*). The governance discipline lives in the family/tier matrix:
mixed quorums for routine work, Western-only quorums for high-stakes
work, jurisdictional payload boundaries always.

This PR lands the spec, the contract, and passing current-state
governance tests that document the recognizer gap. The implementation
patch itself waits for operator preapproval per the Tier 4 rule.
