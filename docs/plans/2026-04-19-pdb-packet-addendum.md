# PDB-shaped review packet — Phase 2b addendum

**Parent design:** [2026-04-19-batched-pr-review-triage.md](./2026-04-19-batched-pr-review-triage.md)
**Status:** spec / implementation order
**Labels:** capability, hygiene (split per step below)
**Written:** 2026-04-19 (morning)

---

## Why this addendum exists

Two external signals arrived the morning after the batched-triage design landed:

1. Anthropic is selling a PR-review agent at ~$25 per PR using multi-model consensus. That is market validation: someone pays $25 per PR for machine-assisted review. The *service* exists; the *adversarial* version does not.

2. The operator-persona framing sharpened. A non-coder founder running 30 PRs in 10 minutes needs the same cognitive shape that presidents get in a Presidential Daily Brief (PDB): a top-line summary, ranked items, explicit source reliability, dissent noted, recommended action.

The existing `review-queue packet` schema produces roughly the right raw data but not the right *shape*. This addendum specifies the delta that turns the existing packet into a PDB-shaped artifact, and the bounded implementation order to get there.

## What is already on `main`

| Capability | Landed via | Surface |
|------------|-----------|---------|
| Advisory-only machine review | #6280 | `aragora review-pr` with `advisory_only=True` default, `COMMENT` not `APPROVE` |
| Prioritized queue, read-only | #6288 | `aragora review-queue build` with `ready_now / needs_attention / repairable / parked` lanes |
| Advisory packet per PR | #6288 | `aragora review-queue packet <pr>` with touched subsystems, risk flags, machine recommendation |
| Human-settlement-gated merge | #6286 | `merge_arbiter` requires explicit human approval review tied to current head SHA |
| Founder settlement loop | #6297 (pending merge) | `aragora review-queue run / act` with `approve / request-changes / defer` actions |

These give ~80% of the substrate. The remaining 20% is the PDB-shape wrapper and the commercial-positioning surfaces (cost, latency, tier labels) needed to compare against Anthropic's $25/PR.

## What the PDB framing requires that today's packet does not have

### 1. Explicit source reliability

Today the packet has `machine_recommendation` and `machine_recommendation_reason`. It does not say which models contributed, with what confidence, or at what cost. The PDB pattern expects sources enumerated (with reliability tiers) so the reader can weight them.

**Gap:** no `sources` field in `ReviewPacket`.

### 2. Dissent surfacing

Today the packet collapses multi-model output into one recommendation. An adversarial-debate substrate *should* have dissent — that is the product. Hiding it defeats the differentiator.

**Gap:** no `dissent` field in `ReviewPacket`.

### 3. Top-line executive summary

Today the packet opens with PR number + title + URL, then a data dump. A PDB opens with the 1–3 sentence top-line that a reader can skim and act on without reading anything else.

**Gap:** no `top_line` field in `ReviewPacket`.

### 4. Cost and latency discipline

Today the packet has no cost or latency. Anthropic's $25/PR puts a hard external anchor on pricing. Without our own number visible in the packet, there is no comparable offer.

**Gap:** no `cost_usd` or `wall_clock_ms` in `ReviewPacket`.

## Implementation order (bounded, small-reversible, no human-gate changes)

Each step is a separate draft PR, reviewable in under 10 minutes, with its own validation block.

### Step 1 — `sources` field

**Type:** capability
**Scope:** `ReviewPacket` dataclass + populator + renderer + tests
**New shape:**

```python
@dataclass(frozen=True)
class PacketSource:
    agent: str           # e.g. "claude-opus-4-7"
    model: str           # pinned model ID
    confidence: float    # 0..1 from critique phase
    latency_ms: int
    cost_usd: float

# ReviewPacket gains:
sources: list[PacketSource]
```

Populate from the existing debate trace (review_runs persisted in `.aragora/review-pr/run.json`). No upstream change needed; `packet` command reads the trace path.

**Test coverage:** schema, JSON roundtrip, text renderer, empty-sources fallback.

### Step 2 — `dissent` field

**Type:** capability
**Scope:** same files as step 1
**New shape:**

```python
@dataclass(frozen=True)
class DissentingView:
    agent: str
    position: Literal["approve", "request_changes", "defer"]
    reason: str  # 1-2 sentence summary from that agent's critique
```

`ReviewPacket.dissent: list[DissentingView]` listing any non-majority positions from the debate. Empty list = unanimous. Populate from critique phase output.

**Test coverage:** unanimous case, single dissent, multi-dissent, JSON round-trip.

### Step 3 — `top_line` field

**Type:** hygiene
**Scope:** new templated renderer
**New shape:**

```python
# ReviewPacket gains:
top_line: str  # 1-3 sentences, templated fill
```

Template fill (deterministic, cheap, fast). LLM-generated summarization optional behind a flag, out of scope for this step.

Template candidate:

```
{machine_recommendation_title} — {top_risk_flag_or_approval_note}.
Consensus from {n_agents} agents; {dissent_count} dissent.
{one_line_fact_from_packet}
```

**Test coverage:** renders sensibly on 5 representative packets (ready/needs-attention/repairable/dissent-heavy/high-risk).

### Step 4 — cost + latency aggregation

**Type:** hygiene
**Scope:** cost rollup from sources list + renderer
**New shape:**

```python
# ReviewPacket gains:
total_cost_usd: float    # sum of sources[].cost_usd
total_latency_ms: int    # max of sources[].latency_ms for parallel; sum for sequential
```

Displayed in the rendered packet under a `packet cost:` line. Makes the Anthropic $25/PR comparison visible in the artifact itself.

**Test coverage:** cost aggregation correctness, empty sources = 0 USD.

### Step 5 — `pr_truth_snapshot` CLI

**Type:** capability + tool-that-reduces-theater
**Scope:** new `aragora/cli/commands/pr_truth_snapshot.py` + parser wiring + tests
**Surface:**

```bash
aragora pr-truth-snapshot <pr> [--json]
```

Prints the canonical, literal status summary that anyone claiming "PR is green" must paste first:

```
PR #6297 | head=2272f79cc | mergeable=MERGEABLE/BLOCKED

CONCLUSIONS:
  SUCCESS:   53  (list: ...)
  FAILURE:    0
  CANCELLED:  5  (list: Zero Coverage Check, test-fast (core), ...)
  SKIPPED:   22
  PENDING:    0

REQUIRED CHECKS (5):  5/5 SUCCESS
REPO-CONTRACT GATES:  Version Alignment=SUCCESS, Status Doc Reconciliation=SUCCESS

BLOCKED-BY: review_required
```

This becomes the canonical receipt format. Any agent or human claiming a PR is merge-ready must paste this output first, unedited. The gate against the status-overstatement pattern is the tool, not a prose commitment.

**Test coverage:** schema output, literal fixture comparison.

### Step 6 — CI rerun ledger

**Type:** capability + tool-that-reduces-theater
**Scope:** new module recording retry attempts per PR
**Shape:**

```
~/.aragora/ci_rerun_ledger.jsonl:
{"ts": "...", "pr": 6297, "workflow": "Tests", "attempt": 4, "action": "classify_infra_noise"}
```

Retries per `(pr, workflow)` pair are counted. After 2 retries on the same (pr, workflow) without a code/config change between them, the CLI refuses further retries and prints "classified as infra-noise, not engineering fix lane."

**Test coverage:** attempt counting, code-change detection (git head SHA delta), refusal path.

## Non-goals

- No change to merge semantics. `merge_arbiter` still requires explicit human settlement.
- No change to GitHub review-posting semantics. Machine review stays `COMMENT`, never `APPROVE`.
- No Phase 3 `act --approve` auto-wiring beyond what #6297 already builds.
- No commercial tier packaging. Pricing strategy is a separate GTM thread; this addendum only ensures cost is **visible** in the packet.
- No LLM-generated top-line (deterministic template only; LLM version is future work).
- No change to the debate engine or agent orchestration.

## Measurement plan

After step 4 lands, run on a real PR:

```bash
aragora review-queue packet <pr> --json > packet.json
```

Compare against a PDB-style checklist:

- [ ] Has top-line at first 3 lines?
- [ ] Sources listed with model + confidence + cost + latency?
- [ ] Dissent visible if any?
- [ ] Total cost visible and under $25?
- [ ] Recommended action visible and explicit?
- [ ] Settlement note present saying "advisory only, human required"?

If 6/6, the packet is PDB-shaped. If fewer, iterate on the failing dimensions.

## Open questions (not blocking — recorded, not asked)

1. **Debate trace format.** Does `run.json` today include per-agent confidence scores? If not, step 1 needs a small upstream write-side change to emit them. *Diagnosis needed before step 1 starts, not now.*

2. **Cost-tracker coupling.** Is per-debate cost already tagged with the PR number? If yes, step 4 is pure renderer. If no, step 4 needs a tracker plumbing delta. *Diagnosis needed before step 4 starts.*

3. **Top-line via LLM vs template.** This addendum defaults to template. LLM-generated is future-work; deferred.

4. **Commercial tiering.** Whether to offer cheap/standard/premium tiers in the product (with different agent mixes at different cost points). GTM question, separate thread.

5. **Cross-debate memory for reviewer calibration.** Should the reviewer track "model X has been wrong Y times on this repo, downweight accordingly"? Touches `aragora/ranking/` + knowledge mound. Out of scope for this addendum, but the PacketSource schema does not preclude it — a future `reviewer_calibration` field can slot in without breaking existing consumers.

6. **Disk/ledger format for step 6.** JSONL vs SQLite. Default JSONL; SQLite only if query volume justifies it.

## Relationship to other artifacts

- **#6279 batched-PR-review-triage design** — parent. This addendum is additive.
- **#6300 REVIEW_AUTHORITY_PRINCIPLES** — complementary. This addendum is the *how*; #6300 is the *why human settlement stays*.
- **EU AI Act Article 14 oversight framing** — preserved. The PDB framing is compatible because settlement remains human; the packet only helps the human read faster.
- **Aragora's five pillars (CANONICAL_GOALS.md)** — PDB packet sits at the intersection of pillar 4 (multi-agent robustness via heterogeneous consensus) and pillar 1 (SMB-ready with enterprise-grade security). Makes both legible to non-coder operators.

## Exit criteria

This addendum's implementation is done when:

- Steps 1–6 have landed as 6 small PRs.
- `aragora review-queue packet <pr>` produces a PDB-shaped artifact that passes the measurement checklist on 10 consecutive real PRs.
- `pr_truth_snapshot` is the standard receipt in any "PR is merge-ready" claim by any agent (human or machine) in this repo.
- CI rerun ledger has capped at least one retry loop automatically.

Only after all four is the PDB reframe considered delivered. Before then, it is in-progress.
