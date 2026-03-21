# Aragora Idea-to-Execution Strategy

*Converted from `Aragora Idea-to-Execution Strategy.docx` and updated for the March 20, 2026 repo state.*

## Executive Summary

Aragora is evolving from a multi-agent debate engine into a local-first idea-to-execution system: a person starts with rough ideas, Aragora turns them into goals and executable specs, and heterogeneous agent lanes carry the work through review, receipts, PR creation, and merge gates.

The strategic thesis from the original document still holds:

- the four-stage idea-to-execution pipeline is differentiated
- the codebase already contains most of the backend substrate
- the missing moat layer is a unified interactive canvas and stage-transition UX
- heterogeneous model diversity plus audit receipts remain the product advantage

What changed during the March 19-20 tranche/overnight work is the backend maturity. The repo is no longer only proving a single Ralph lane. It now has a real tranche queue, design review, watch/review/integrate flow, and a growing body of evidence about what unattended execution can and cannot yet do truthfully.

## Five Strategic Goals

### 1. Reliable Spec-to-Merge Engine

Make Aragora a real autonomous engineering loop, not a demo. The loop is:

`plan -> dispatch -> implement -> review -> classify blockers -> repair -> PR creation -> merge gate`

Every handoff in that loop must behave truthfully under unattended execution.

### 2. Idea-to-Execution for Non-Engineers

Turn vague human intent into executable plans without forcing the user to become a project manager or spec writer.

### 3. Unified DAG Visual Language

Ideas, goals, tasks, and agent orchestration are all DAGs with provenance. They should share one visual language and one canvas instead of fragmenting across tools.

### 4. Heterogeneous Model Diversity as Product Advantage

Claude plans. Codex implements. Cross-model review catches real issues. Aragora's moat is not "many models talking," but routing the right model strengths to the right stage and preserving the dissent trail.

### 5. Benchmarked Evidence, Not Anecdotes

Each autonomy cycle should end with an explicit contract: what the system can do unattended, where it truthfully stops, and what evidence supports that claim.

## The Four-Stage Pipeline

| Stage | Name | Input | Output | AI Role |
|---|---|---|---|---|
| 1 | Ideas | Freeform notes, thoughts, sketches | Structured idea graph | Cluster, link, surface gaps |
| 2 | Goals | Idea graph plus clarifying Q&A | Goal and principle DAG | Synthesize goals, extract constraints, score confidence |
| 3 | Actions | Goal DAG plus domain context | Task and spec DAG | Decompose goals, assign effort, generate specs |
| 4 | Orchestration | Task DAG plus agent pool | Executed, reviewed, merged code with receipts | Assign agents, run debate review, coordinate repair |

### Why DAGs, Not Trees

DAGs model the real structure of work better than trees. One idea can support multiple goals, one goal can influence multiple tasks, and one task can depend on several upstream conditions.

### Provenance and Stage Gates

Every stage transition should preserve provenance. Aragora's debate receipts and pipeline provenance model already establish the foundation. The missing piece is a high-quality interface that makes those transitions legible and editable.

## Market Landscape

The original research remains directionally correct:

- Stage 1 idea organization is crowded
- Stage 2 goal derivation is still the emptiest part of the market
- Stage 3 project management is mature but mostly proprietary
- Stage 4 agent orchestration is rich in open source, but starts too late in the pipeline

No serious competitor spans all four stages with one visual abstraction, auditable stage transitions, heterogeneous review, and provenance from idea to merged artifact.

## What Is Genuinely Novel

The core differentiators are still the same six gaps identified in the source document:

1. AI-driven goal derivation from structured idea graphs remains mostly unserved.
2. Existing tools still do not close the loop bidirectionally from execution back to planning.
3. No existing product exposes one DAG abstraction across ideas, goals, tasks, and orchestration.
4. No existing product offers one unified visual canvas across all four stages.
5. Open-source tooling is still rich at stages 1 and 4 and weak in the middle.
6. "Idea-rich, execution-poor" remains a distinct and underserved user profile.

## Capability Audit

The codebase is still best understood as "audit, unify, and finish," not "build from scratch."

| Stage | Current Assets | Repo Reality | Main Gap |
|---|---|---|---|
| Ideas | `aragora/canvas/*`, `idea_store.py`, `idea_canvas.py`, `/ideas` page | Strong local-first graph model and CRUD surface | Better user onboarding and graph generation UX |
| Goals | `aragora/goals/extractor.py`, goal store, goal canvas types | Good backend substrate and structural extraction | Higher-quality interactive transition UX |
| Actions | Workflow engine, pipeline planning, action graph models | Strong decomposition substrate | Better task-quality heuristics and richer review loops |
| Orchestration | Ralph, swarm supervisor, tranche submit/review/integrate/watch, queue compiler/runner | Now materially stronger after March 19-20 hardening | Truthful unattended completion and publish edge cases |
| Cross-stage | provenance links, receipts, pipeline abstractions | Strong backend pattern exists | Unified canvas and stage transition product surface |

## Recommended Open Source Stack

The recommended stack from the original document is still reasonable:

| Layer | Recommendation | Why |
|---|---|---|
| Canvas UI | React Flow / Xyflow | Strong node-and-edge interaction model |
| Whiteboard / freeform capture | Excalidraw | Useful early-stage ideation affordance |
| Layout / graph helpers | dagre, Mermaid | Good DAG layout and export utilities |
| Orchestration patterns | LangGraph, Temporal, Hatchet | Durable execution and workflow references |
| Multi-agent coordination | Aragora native plus targeted external patterns | Aragora's differentiation is the debate/receipt layer, not generic workflow wiring |

Avoid introducing new dependencies that weaken embedding or self-hosting economics when the current stack already covers the core need.

## March 20, 2026 Addendum

The source document's "Immediate - Finish the Clean Ralph Rerun" recommendation is now historically stale. The backend frontier has moved forward during the March 19-20 tranche/overnight autonomy sprint.

### What Landed On `main`

Recent merged work materially expanded the autonomous execution substrate:

- sequential tranche queue execution and curated queue compilation
- low-risk auto-merge policy for suitable single-lane work
- dead-worker reconciliation and deliverable recovery
- single-lane queue behavior for broad issue sources
- verification-command propagation through queue compile
- explicit single work-order emission for single-lane tranche specs
- supervisor-to-artifact deliverable sync
- stale fleet-claim reaping before conflict checks
- truthful queue-state persistence before long watch loops

### What The Overnight Runs Proved

The overnight dogfood runs produced real evidence, not just theory:

- `#1108` merged as a real output from the queue system
- `#1110` and `#1111` are the canonical candidate PMF outputs from later runs
- `#1113` and `#1114` are alternate variants from the same issues
- repeated runs exposed real control-plane bugs, which were then fixed on `main`

The key bugs uncovered through dogfooding were:

1. multi-lane planner overlap blocking queue preflight
2. dead workers not terminalizing truthfully
3. single-lane specs being re-decomposed inside the supervisor
4. verification commands being dropped before the merge gate
5. stale fleet claims permanently blocking fresh work
6. queue resume losing `manifest_path` and resubmitting duplicate tranches

### Current Frontier

As of the latest durable March 20 state:

- `queue-v4b` is the active reduced proof lane for `#1047` and `#819`
- `#1047` has already reached a truthful `needs_human` / review-blocked state in the queue
- `#819` is still pending behind it

The main remaining autonomy questions are now narrower:

- controller-side publish and worker-environment edge cases
- truthful terminalization of every blocker type
- making the active proof lane produce either a real deliverable or a preserved blocker reason without operator guesswork

That is a much better place to be than the original March 2026 strategy assumed.

## Updated Strategy And Best Next Steps

### Immediate

1. Finish the unattended tranche proof truthfully.
   The active queue proof should end in either a real deliverable or a preserved blocker, not another ambiguous stuck state.
2. Harvest the real PMF outputs already generated.
   Review and merge the strongest implementations instead of letting autonomous output rot in open PRs.
3. Keep the autonomy contract explicit.
   Every proof run should end with a short, evidence-backed statement of the first manual step.

### Near Term

1. Build the interactive stage-transition UX.
   This remains the highest-value product gap.
2. Unify the four stages on one canvas.
   The backend already behaves like one system; the product still feels like multiple systems.
3. Make execution results revise upstream plans.
   Provenance should not stop at auditability; it should power feedback into goals and ideas.

### Medium Term

1. Turn the pipeline into the default product shell for non-engineers.
2. Preserve local-first speed and durability as a feature, not a compromise.
3. Use benchmarked autonomy evidence as sales proof, not just engineering validation.

## MVP Vertical Slice

The original MVP slice is still right:

| Step | Human Action | AI Action |
|---|---|---|
| Idea capture | Add and connect ideas on a local canvas | Cluster and structure them |
| Stage 1 -> 2 | Answer clarifying questions | Derive a goal and principle DAG |
| Goal review | Edit, split, or remove goals | Re-derive dependent structure |
| Stage 2 -> 3 | Approve task breakdown | Produce fully specified tasks with acceptance criteria |
| Stage 3 -> 4 | Approve orchestration plan | Assign agents and launch execution |
| Execution | Watch or intervene | Run the spec-to-merge loop |
| Merge gate | Approve or request changes | Hold, surface findings, and prepare repair work |

### Success Criteria

- A non-engineer can move from 10 rough ideas to a merged PR in under 30 minutes.
- Every merged action can be traced back to goals and source ideas.
- The human makes minimal edits to machine-generated content.
- The orchestration layer requires zero hidden manual rescue between dispatch and the merge gate.

## Competitive Positioning

If Aragora finishes this system, its distinct position is:

- one pipeline spanning ideas, goals, actions, and orchestration
- adversarial debate and review, not just "AI suggestions"
- provenance from originating idea to merged artifact
- heterogeneous model diversity as a reliability mechanism
- local-first execution for people who have many ideas but weak execution bandwidth

## Summary

The original Word document was directionally right: Aragora's biggest opportunity is not another backend substrate, but finishing the product surface that lets people move from ideas to execution with confidence.

What changed after the March 19-20 work is that the orchestration backend is substantially more real than it was when the document was drafted. The strategic center is now:

- finish truthful unattended execution,
- harvest the outputs already produced,
- and then package the full system behind a unified idea-to-execution interface.
