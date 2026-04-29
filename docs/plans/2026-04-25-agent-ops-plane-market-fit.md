# Agent-Ops Plane: market-fit, broader utility, and roadmap leverage

> **Provenance.** Originally drafted by Factory Droid on 2026-04-25 as a refinement of the in-flight agent-bridge plan after Codex's PR-2 scoping. Ported into the canonical doc chain on 2026-04-28 so the market-fit framing and the five extraction-readiness disciplines are discoverable from `docs/PACKAGING.md` and `docs/CANONICAL_GOALS.md`.
>
> **Status.** Planning truth. The bridge code itself lives in `aragora/swarm/agent_bridge/` (in-flight); the implementation plan tranche is `docs/plans/2026-04-21-agent-bridge-pr2-scoping.md` → `docs/plans/2026-04-21-pdb-mode3-pr4-ui-design.md` → `docs/plans/2026-04-21-agent-bridge-design-refresh.md`. This doc is the *positioning* layer, not an implementation plan.

---

## 1. What is actually being built

Strip away naming and the capability is one sentence:

> An operator-local control plane that lets independent CLI/runtime agent harnesses (Claude Code, Codex CLI, Factory Droid, Aider, Gemini CLI, etc.) hold a structured ongoing dialog and cross-check each other while each retains its own native session state, tools, and context.

The non-trivial property is **state-preserving heterogeneous harness interop**. Every existing framework either:

- absorbs the agent (AutoGen Proxy, LangGraph nodes, CrewAI roles, OpenHands runtime), or
- shares tools/resources but not turns (MCP), or
- assumes agents are HTTP services with cards (A2A), not local CLIs with rich session state.

None of them lets a Claude Code session and a Codex CLI session pass a turn back and forth, each keeping its own context window, MCP servers, file tools, and project state.

## 2. Is this an unmet market need? Yes — and growing.

**Trend that creates the demand:**

- 2025-2026: explosion of high-quality coding/agent harnesses (Claude Code, Codex CLI, Aider, Factory Droid, Kilocode, Gemini CLI, Cursor Agent, Windsurf Cascade, Goose, Cline, …).
- Each ships its own session model, MCP integration, tool authorization, project memory.
- Users already run 2-4 harnesses concurrently for different tasks; the cost is copy-paste, no shared transcript, no cross-check, no merge of strengths.
- Vendors have zero incentive to build interop — each wants you in their walled garden.

**Where each existing solution falls short:**

| Solution | Strength | Why it doesn't solve this |
|---|---|---|
| MCP | Tool/resource sharing | No turn-dispatch primitive; client-server, not agent↔agent |
| A2A (Google) | Agent message envelopes | Assumes HTTP-service agents with cards, not CLI harnesses |
| AutoGen | Conversable agent debate | Requires surrendering CLI state to the orchestrator |
| LangGraph | Cyclic graphs over LLMs | Treats LLM as node; no harness session preservation |
| CrewAI | Role hierarchy | Internal LLM abstraction |
| OpenHands | Engineering runtime | Subsumes the agent into its runtime |
| claude-flow / claude-squad | Multi-Claude spawn | Single-vendor; no Codex/Droid bridge |
| Aider swarm | Multi-process aider | Single-model |

The intersection — **heterogeneous + state-preserving + dialog/cross-check + permissive license** — is empty. That is the unmet need.

## 3. Broader utility (beyond Aragora)

Audiences that benefit if this exists as a usable capability:

1. **Solo developers** running multi-harness setups (the universal "I copy-paste between Claude Code and Codex" pain).
2. **Engineering teams** doing high-stakes PR review where two-harness disagreement is signal.
3. **AI evaluation / red-team groups** who need adversarial cross-checking with authentic per-harness behavior, not toy proxies.
4. **Researchers** studying multi-agent collaboration where each agent's state is real.
5. **Companies running internal bake-offs** to compare harness performance on identical tasks with shared context.
6. **Compliance/audit functions** wanting cryptographically receipted multi-agent decision trails.

This is general-purpose infrastructure. It is **not** intrinsically tied to Aragora's debate-orchestration use case.

## 4. How to maximize generalized value (extraction-readiness)

Build for Aragora first, but with five disciplines so extraction stays cheap:

1. **Package boundary discipline**: keep `aragora/swarm/agent_bridge/` import-clean of Aragora-specific code (no `debate/`, no `review/`, no `thesis/`). Aragora-specific UI and policies live elsewhere. This enables a future `pip install agent-bridge` extraction with zero rewrite.
2. **Footer contract as a versioned wire spec**: publish an "Agent Bridge Footer Protocol v1" markdown doc with semver. Treat it like Language Server Protocol. Other tools can adopt it independently.
3. **Adopt-where-useful, depend-where-stable**: speak A2A envelope shape on the wire so future A2A-compliant agents can join; expose MCP-style tool-sharing as an optional transport between turns. Do not depend on either being mature.
4. **Pluggable transports**: subprocess (current), HTTP, WebSocket, future gRPC. Same broker contract, different adapters.
5. **Reference adapters for ≥4 harnesses**: Claude Code, Codex CLI, Droid, Aider (and one of: Gemini CLI / Cursor Agent / Goose). Demonstrates cross-harness scope, makes the protocol concrete for outside adopters.
6. **License**: Apache-2.0 (preferred over MIT for patent-grant clarity in a multi-vendor space).

## 5. How this accelerates the Aragora roadmap, thesis, maximalist vision

The bridge isn't a side project — it directly unlocks every pillar Aragora claims.

**Pillar 4 (multi-agent robustness via heterogeneous model consensus):**
Today, "consensus" runs inside one process with API agents. The bridge makes consensus run across *real* harnesses with their own state. This is the difference between "claimed multi-agent" and "actual multi-agent."

**Pillar 5 (self-healing / self-extending via the Nomic Loop):**
Nomic Loop is one-process today. With the bridge, Phase 3 (Implement) can be Codex while Phase 4 (Verify) is Claude, with footer-receipted cross-check at the handoff boundary. Self-improvement gets honest cross-checking instead of single-harness blind spots.

**Empirical threshold (#6375) loop closure:**
Once threshold scheduler (step B) lands, every threshold update can be dispatched through the bridge for a "second opinion" before applying. Every threshold change becomes a multi-harness signed receipt — the H1 thesis claim about decision integrity becomes self-validating.

**Review-queue (#6608, #6614, this PR family):**
The advisory packet is monologue today. With the bridge, the packet is authored by 2-3 harnesses in dialog. That is exactly the "dialectic / Hegelian sublation" the thesis claims to embody.

**Argonaut ledger / ERC-8004 identity:**
Per-agent on-chain identity is meaningful only when provenance attaches to native runtime state, not just LLM weights. The bridge's per-actor session ID + footer chain is the missing provenance substrate.

**Direct copy-paste elimination today:**
Right now codex, droid, and Claude are coordinating via human copy-paste. The write API turns that into programmatic dispatch. Roadmap velocity goes up by a constant multiplier — every multi-agent task gets faster.

**Commercial differentiation:**
Most platforms position as "LLM orchestration." Aragora becomes "harness orchestration" — likely the only commercial-grade platform that respects native CLI agent autonomy. That is a defensible positioning niche as harnesses proliferate.

## 6. Recommended refinements to the in-flight plan

The codex plan is sound. Five tightenings worth applying:

1. **Default-deny on write surface**: separate `agent_bridge_read` and `agent_bridge_write` feature gates. Enabling read-only UI must not silently enable subprocess spawning. (Codex captured this; reinforce in tests.)
2. **One-step auto-baton, not daemon**: ship `POST /runs/{id}/auto-step` first; defer the looping daemon until one-step is proven safe. Termination conditions are explicit: `needs_human=true`, repair-loop exhausted, footer malformed twice, completed/failed, missing next actor, lease conflict.
3. **Active-work snapshot is read-only and union-of-truths**: never invent leases, only project from `coordination/fleet_status` + `swarm/status` + `dev_coordination` + active bridge runs. When sources disagree, surface the disagreement, do not paper over it.
4. **First real bridge run targets a real roadmap item**: not a synthetic smoke. Suggested: dispatch `#6375` review or H2 thesis-gap scoping through the bridge with droid/claude/codex as actors. Tier-0 success is functional, but the *first non-smoke* should generate a real receipt in the live review-queue.
5. **Extraction-readiness disciplines** (section 4) folded in from the start, not retrofitted.

## 7. Recommended sequencing (refined from codex's plan)

| # | Step | Outcome | Estimated effort |
|---|---|---|---|
| 0 | Live-smoke on the worktree (already passing) | Confirms transport layer real | done |
| 1 | Write API: `POST /runs`, `POST /runs/{id}/dispatch` + RBAC + write gate | Programmatic dispatch — kills copy-paste for sequential debate | 1 PR, ~4-6h |
| 2 | Active-work snapshot: `GET /coordination/active-work` | Parallel-pattern coordination unblock | 1 PR, ~2-3h |
| 3 | One-step auto-baton: `POST /runs/{id}/auto-step` | `next_actor` becomes real signal; humans only on real decisions | 1 PR, ~3-4h |
| 4 | Footer Protocol v1 spec doc + semver tests | Extraction-readiness, public adopter contract | small |
| 5 | First real bridge-run on a roadmap item (e.g., #6375 step B review) | Validates the loop on real work; first receipt | small |
| 6 | Operator-action UI on existing `/autonomous/bridge` | Reduce CLI for non-power-users | optional |
| 7 | Extract `agent_bridge` to standalone package | Broader-value capture | future |

## 8. Bottom line

This is **not** plumbing-about-plumbing. It is the substrate without which the rest of the Aragora maximalist vision is aspirational rather than operational.

- The capability addresses a real, growing, unmet market need (heterogeneous state-preserving harness interop).
- It is genuinely general-purpose; with five extraction-readiness disciplines it can become a standalone OSS capability without slowing Aragora.
- It directly unlocks Pillars 4 and 5 of the thesis, makes the empirical-threshold loop self-validating, gives the Argonaut ledger its missing provenance substrate, and turns the dialectical-debate claim from rhetoric into mechanism.
- It eliminates the copy-paste tax on every multi-agent roadmap item from this point forward — which compounds.

**Recommendation: ship Tier 1-3 in three bounded PRs, fold in section-4 disciplines, run #6375 step B's review through the first real bridge run, then resume the H1/H2 thesis-gap drain at higher velocity.**

The work codex started today is the right work. The refinements above are surface-area choices, not direction changes.

---

## See also

- [`docs/CANONICAL_GOALS.md`](../CANONICAL_GOALS.md) — eight doctrinal pillars (Pillar 1: Adversarial Heterogeneous Consensus; Pillar 2: Reliable Autonomous Execution)
- [`docs/PACKAGING.md`](../PACKAGING.md) — modular delivery; the agent-ops plane is a future extractable substrate (see "Future Extractions")
- [`docs/THESIS.md`](../THESIS.md) — premise 3 (no safe single-agent delegation); convergence-as-evidence requires *different priors, different evidence, active incentive to dissent* (= heterogeneity that is not theatrical)
- [`docs/plans/2026-04-21-agent-bridge-pr2-scoping.md`](2026-04-21-agent-bridge-pr2-scoping.md) — in-flight implementation tranche
- [`docs/plans/2026-04-21-pdb-mode3-pr4-ui-design.md`](2026-04-21-pdb-mode3-pr4-ui-design.md) — Mode-3 PDB heterogeneous panel UI
