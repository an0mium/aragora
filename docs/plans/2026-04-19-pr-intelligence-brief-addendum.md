# PR Intelligence Brief — Implementation Addendum

Last updated: 2026-04-19
Status: implementation-ready design addendum

**Extends:** the primary PR Intelligence Brief plan proposed in PR #6309 (`docs/plans/2026-04-19-pr-intelligence-brief.md` once landed)
**Tracking:** #6303 (epic) · #6304 (UI) · #6305 (cost) · #6306 (protocol) · #6307 (receipts)

## Purpose

This addendum captures decisions from the 2026-04-19 brainstorm session that operationalize the primary plan's "heterogeneous review debate" concept with concrete models, phases, CLI, UI, cache, receipts, and budget controls.

The primary plan remains authoritative for product framing, category, non-goals, and GTM positioning. This addendum is authoritative for the technical sections below.

## Decisions Locked In

### 1. Model panel (provider-slot pattern)

Sharpens "3–5 heterogeneous" in the primary plan. Phase A panel = 8 already-wired models, organized by lens:

**Core lens** (default synthesis + primary review voice):
- `pdb.core.claude` → latest most capable Claude family
- `pdb.core.gpt` → latest most capable OpenAI GPT family

**Heterodox lens** (different architectures, different training corpora, reveals correlated-error blind spots):
- `pdb.heterodox.gemini`
- `pdb.heterodox.grok`
- `pdb.heterodox.deepseek`
- `pdb.heterodox.kimi`
- `pdb.heterodox.qwen`

**Regulatory lens** (EU/regulatory framing, explicit not implicit):
- `pdb.regulatory.mistral`

**Phase B additions** (need adapters):
- `pdb.heterodox.glm` (Zhipu)
- `pdb.heterodox.ernie` (Baidu)

Models are resolved at run time via `aragora.agents.registry.resolve_slot("pdb.core.claude")` which reads a central slot config at `aragora/config/pdb_panel.yaml`. Each slot maps to a current provider-specific model id. Operators bump the config when new frontier models land. **No hardcoded model names in the debate protocol code.**

### 2. Debate protocol shape

#### Protocol B (default, every PR)

1. **Findings round** — all 8 panel models run in parallel. Each reviews the PR packet + diff + CI state, produces a free-form findings list across logic / security / maintainability / risk / evidence dimensions. Role-structured output is NOT requested here — let each model surface what it independently sees.
2. **Critique round** — all 8 panel models receive the round-1 findings from the other 7 and update their own findings in light of cross-model critique. Findings may be strengthened, retracted, or flagged as contested.
3. **Synthesis** — one synthesizer model (default slot `pdb.core.claude`) ingests all 16 findings (8 × 2) and produces the final brief with role-structured sections: logic / security / maintainability / skeptic. Dissent is partitioned explicitly by lens (core / heterodox / regulatory).

**Cost estimate:** 16 findings calls + 1 synthesis ≈ $5–8/PR. **Latency:** ~2–3 min parallelized.

#### Protocol C (adaptive escalation)

Steps 1–3 identical to B. Then:

4. **Dual-synthesis check** — run a second synthesis with `pdb.core.gpt` on the same 16 findings. If the two synthesized briefs materially diverge, surface both summaries and a "synthesis disagreement" panel. Material divergence = verdict changes OR confidence gap ≥ 2 OR dissent-map lens partitions differ.

**Cost estimate:** B cost + 1 additional synthesis ≈ $8–12/PR.

#### Escalation triggers (any one → C instead of B)

1. Diff touches high-consequence paths: `.github/workflows/`, `aragora/auth/`, `aragora/rbac/`, `aragora/security/`, `aragora/privacy/`, `aragora/compliance/`, `aragora/swarm/merge_arbiter.py`, `aragora/cli/commands/review_pr.py`, `aragora/cli/commands/review_queue.py`.
2. Broad blast radius: `scripts/export_openapi.py`, SDK generation paths, repo-wide generated artifacts, policy/config files under `aragora/config/`.
3. Large diff (>500 lines) combined with multi-subsystem scope (≥2 top-level `aragora/` subdirectories). Size alone is **not** enough.
4. Manual label: `escalate-pdb`. Applying this label is an operator assertion that at least one principle-level triage trigger from `docs/THESIS.md` § "When does the human actually need to weigh in?" applies to this PR. The asserting operator is recorded in the receipt; if later review finds no principle-level trigger applied, that signals either a mis-applied label or an incompleteness in the thesis's principle-level list.
5. Low confidence from B synthesis: final confidence < 3/5, or dissent spans ≥ 2 lenses.
6. Stale brief: cached brief exists but `head_sha` has changed.
7. Repeated flaky CI: more than 2 retried/cancelled required-check cycles on the same head SHA.

### 3. Timing modes (shared cache)

All three modes write to the same `(pr_number, head_sha)`-keyed cache. The mode only determines **when** the cache is filled.

**Mode 2 (batched morning build)** — default. Enumerates unsettled PRs, checks cache, generates missing briefs in batch. Progress rendered at CLI. Invoked by `review-queue build`.

**Mode 1 (webhook/PR-open)** — opt-in via env flag `ARAGORA_PDB_AUTO_BRIEF=1`. Listener on `pull_request` events (opened, synchronize, ready_for_review); cache miss enqueues background brief generation. Webhook returns fast. Intended for Phase B/C where always-fresh briefs matter.

**Mode 3 (on-demand)** — always available via `review-queue brief <pr>`. Serves from cache if fresh; regenerates otherwise. Useful for first-use trial, out-of-band review, or manual escalation.

### 4. Brief format (progressive disclosure hybrid)

**Default view (A level)** — one line per PR:
- PR number + title
- Verdict glyph (✓ ⚠ ✗) + verdict enum
- Confidence (X/5), agreement count (X/8)
- Risk summary (N yellow, N red)
- Cost used

**Click-to-expand (B level)** — ~150-word PDB prose:
- Verdict sentence
- Key risks (top 3)
- Dissent summary by lens
- Recommended action

**"Transcript" link (C level)** — full dashboard:
- Per-model findings table (8 rows × 2 rounds)
- Synthesis transcript
- Dissent-map rendered as table
- Full packet + diff + CI references
- Receipt signature + artifact URI

All three levels share one Brief artifact; the renderer decides density.

### 5. Brief schema

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Mapping, Sequence

@dataclass(frozen=True)
class Finding:
    model_id: str
    round: int  # 1 or 2
    lens: Literal["core", "heterodox", "regulatory"]
    dimension: Literal["logic", "security", "maintainability", "risk", "evidence", "other"]
    statement: str
    severity: Literal["info", "yellow", "red"]
    confidence: int  # 1-5

@dataclass(frozen=True)
class Brief:
    # Identity
    pr_number: int
    head_sha: str
    repo: str
    created_at: datetime

    # Protocol
    protocol: Literal["B", "C"]
    protocol_version: str  # e.g., "pdb-b.1.0"

    # Panel
    panel_models: tuple[str, ...]
    panel_lenses: Mapping[str, Literal["core", "heterodox", "regulatory"]]
    synthesizer_slot: str
    dual_synthesis: bool

    # Verdict
    verdict: Literal["approve_candidate", "needs_human_attention", "repair_first"]
    confidence: int  # 1-5
    agreement_count: int  # of panel_size

    # Findings
    findings: tuple[Finding, ...]  # flattened; round+model_id are fields

    # Role-structured synthesis sections
    logic: str
    security: str
    maintainability: str
    skeptic: str

    # Dissent map: lens → list of disagreement statements
    dissent_map: Mapping[Literal["core", "heterodox", "regulatory"], tuple[str, ...]]

    # Dual synthesis (C only)
    secondary_verdict: str | None
    secondary_confidence: int | None
    synthesis_divergence: str | None  # narrative if materially divergent

    # Evidence & context
    evidence_refs: tuple[str, ...]
    touched_subsystems: tuple[str, ...]
    diff_lines_changed: int
    ci_state: Mapping[str, str]  # check_name → conclusion at brief time

    # Cost
    cost_usd: float
    budget_remaining_usd: float

    # Signature
    packet_sha: str
    signature: str  # ed25519
```

### 6. Cache / storage contract

**Artifact location:** `.aragora/review-queue/briefs/pr-{number}-{head_sha_short}.json` where `head_sha_short = head_sha[:12]`. Full SHA inside the file.

**Index:** `.aragora/review-queue/briefs/index.jsonl` — append-only for O(1) listing.

**Invalidation:** `.aragora/review-queue/briefs/invalidated/` — stale briefs moved here (not deleted) with index entry noting reason.

**Freshness rules:**
- Brief is fresh iff `head_sha` matches current GitHub PR head
- Force-push or merge-of-base invalidates
- Reads: `get_brief(pr, head_sha) → Brief | None`; None triggers regen

**Ledger linkage:** Brief generation appends `pdb_brief_generated` event to `shift_ledger.jsonl`.

### 7. CLI flags

```
# Mode 2 (default) — batched
aragora review-queue build                   # fills brief cache
aragora review-queue build --no-briefs       # advisory packet only
aragora review-queue build --protocol=C      # force C for all PRs
aragora review-queue build --budget=200      # USD cap for this invocation

# Mode 3 — on-demand
aragora review-queue brief <pr>
aragora review-queue brief <pr> --force      # ignore cache
aragora review-queue brief <pr> --protocol=C

# Shared
--panel=slot1,slot2,...                      # override default panel
--synthesizer=slot                           # override synthesizer
--json                                       # raw Brief JSON to stdout

# Mode 1 (webhook, opt-in)
export ARAGORA_PDB_AUTO_BRIEF=1
aragora webhook register-pdb
aragora webhook status-pdb
```

### 8. Web route / surface

**New page:** `aragora/live/src/app/(app)/review-queue/page.tsx`

Single-column priority-ranked cards (CRT theme matching existing live pages). Each row:
- Left: priority indicator + PR number + title
- Middle: verdict glyph + confidence + agreement + risk dots
- Right: cost + timestamp + one-click actions (approve / request-changes / defer / open-diff)

Click row → in-place expansion to B level (150-word synthesis sections + dissent map).

"Transcript" link → slide-over to C level (per-model findings table + synthesis transcript + evidence links).

**New API endpoints** (under `/api/v1/review-queue/`):
- `GET /briefs` — list
- `GET /briefs/{pr}/{head_sha}` — single
- `POST /briefs/{pr}/settle` — settle action with head-SHA binding
- `GET /briefs/health` — cache + panel + budget status

**Settlement integration:** Web UI settlement writes the same local receipt file (`.aragora/review-queue/settlements/`) that `merge_arbiter` reads. Web UI is equivalent to CLI for settlement purposes; no bypass.

### 9. Receipt extensions

Two new event types extend `shift_ledger.jsonl`:

**`pdb_brief_generated`:**

```json
{
  "event": "pdb_brief_generated",
  "timestamp": "2026-04-19T12:34:56Z",
  "pr_number": 6297,
  "head_sha": "2272f79cc7ae...",
  "repo": "synaptent/aragora",
  "protocol": "B",
  "protocol_version": "pdb-b.1.0",
  "synthesizer_slot": "pdb.core.claude",
  "synthesizer_model_id": "claude-opus-4.7",
  "dual_synthesis": false,
  "verdict": "approve_candidate",
  "confidence": 4,
  "agreement_count": 7,
  "panel_size": 8,
  "cost_usd": 5.82,
  "budget_remaining_usd": 194.18,
  "artifact_path": ".aragora/review-queue/briefs/pr-6297-2272f79cc7ae.json",
  "packet_sha": "sha256:...",
  "signature": "ed25519:..."
}
```

**`pdb_brief_settled`:**

```json
{
  "event": "pdb_brief_settled",
  "timestamp": "2026-04-19T13:45:00Z",
  "pr_number": 6297,
  "head_sha": "2272f79cc7ae...",
  "brief_signature": "ed25519:...",
  "reviewer": "armand",
  "action": "approve",
  "reason": null,
  "channel": "web-ui"
}
```

`merge_arbiter` consumes `pdb_brief_settled` as equivalent to the existing local settlement receipt. Signatures are verified before acceptance. Existing gauntlet receipts are retained for the debate itself; brief receipts extend but do not replace them.

### 10. Budget controls

**Per-invocation:** `--budget=<usd>` flag on `build` and `brief`. Exceeded mid-batch → stop generating new briefs, preserve already-generated. Emits `pdb_budget_exceeded` ledger event.

**Per-day:** `pdb_daily_budget_usd` in `aragora/config/pdb_panel.yaml`. Counter at `.aragora/review-queue/budget_usage.json`. UTC midnight reset.

**Per-PR:** hard cap `pdb_per_pr_max_usd` (default $20) prevents runaway single-PR cost. Exceeded → brief generation fails with `budget_exceeded` verdict; falls back to advisory packet only.

**Dashboard visibility:** Each brief displays cost used. Morning build summary shows total spent + remaining daily budget. Web UI includes a budget badge in the page header.

### 11. Phase ordering

| Phase | Consumer | Panel | Protocol | Default mode | UI | Budget | Success signal |
|-------|----------|-------|----------|--------------|----|--------|----------------|
| **A** Aragora dogfood | Founder | 8 wired | B default, C escalation | 2 (batched) | CLI first; web parallel | $200/day start | 20–50 PRs settled in 10–15 min; dissent correlates with real issues |
| **B** External maintainer (Stenberg profile) | cURL/HAProxy-style design partner | 8 + GLM + Ernie | same | 1 (webhook) | Web polish required | Per-repo cap; SCIM/RBAC | Reduced triage time; brief catches ≥1 real vulnerability/week |
| **C** Regulated SMB (EU AI Act) | Design partner with compliance mandate | Possibly region-locked models | same | 1 required | Tenant-isolated | Multi-tenant billing | Audit trail satisfies Article 14 human-oversight for code-change decisions |

### 12. Implementation order across #6303–#6307

Sequential steps; each produces mergeable increments.

**Step 1 — #6306 `PRReviewProtocol` + slot resolver** (~1 week)
- New module `aragora/pdb/protocol.py`
- `ProviderSlotResolver` reading `aragora/config/pdb_panel.yaml`
- `PRReviewProtocol` implementing 2-round findings + synthesis
- `Brief` + `Finding` dataclasses
- Flag `ARAGORA_PDB_ENABLED` default off
- Unit tests: slot resolution, findings schema, synthesis with mocked models
- Not user-facing yet

**Step 2 — #6307 receipt extensions** (~3 days)
- Extend `aragora/swarm/shift_ledger.py` schema
- Extend `merge_arbiter` to accept `pdb_brief_settled` as settlement
- Extend `DurableFileSigner` to sign Brief artifacts
- Tests: signature roundtrip, ledger I/O, merge_arbiter integration

**Step 3 — #6305 cost + escalation** (~4 days)
- `aragora/pdb/budget.py`: per-invocation / per-day / per-PR logic
- Budget usage file + UTC rollover
- Escalation-rule engine (the 7 rules in §2)
- Tests: budget exhaustion mid-batch, each escalation trigger

**Step 4 — CLI integration** (~3 days)
- Extend `aragora/cli/commands/review_queue.py` with brief generation in `build`
- Add `brief` subcommand
- Add `--no-briefs`, `--protocol`, `--budget`, `--panel`, `--synthesizer`, `--json` flags
- Tests: CLI integration, cache hit/miss

**Step 5 — #6304 Web UI** (~1–2 weeks)
- New page, new API endpoints
- Progressive disclosure card layout (A → B → C)
- Dissent map component
- Budget badge
- Settlement actions with head-SHA binding

**Step 6 — Mode 1 webhook** (~1 week, opt-in, ships last)
- Webhook handler, background worker
- `aragora webhook register-pdb` / `status-pdb`
- Phase B prerequisite

**Step 7 — GLM + Ernie adapters** (~2–3 days each, Phase B prep)
- Standard adapter pattern
- Add slots to resolver
- Tests: agent roundtrip

## Open Questions

- **Round-2 convergence shortcut:** if round 2 shows strong convergence (all 8 models agree, high confidence), should we skip synthesis and return a lightweight brief? Saves cost; breaks uniform brief format. Defer until Phase A data.
- **Priority ranking for morning queue:** oldest-first / most-dissent-first / highest-risk-first / operator-configurable? Proposal: default most-dissent-first — you see contested PRs early while alert. Needs a product call.
- **Model version update cadence:** when a panel model version bumps, invalidate cached briefs? Proposal: no — briefs are timestamped snapshots; provide `--force-refresh-panel` flag for ad-hoc re-brief. UX discussion needed.
- **Dual-synthesis divergence threshold:** current definition (verdict change OR confidence gap ≥ 2 OR lens partition differs) is a guess. Tune from Phase A data.
- **Mistral regulatory prompting:** explicit "approach from EU regulatory/compliance perspective" system prompt, or let natural training bias surface? Proposal: explicit — makes the lens legible to brief consumers.
- **Multi-tenant cost attribution (Phase C):** whose API keys fund design-partner briefs? Proposal: Aragora holds provider keys, passes through as unit cost to partner. Needs billing decision.

## Summary

This addendum operationalizes the primary plan without contradicting it. The settlement gate is preserved throughout. All flags default off or safe. No existing workflow is broken by merging this design doc.

Implementation proceeds via the 7 ordered steps above, landing through #6306 → #6307 → #6305 → CLI → #6304 → webhook → adapters. Each step is independently mergeable and testable.
