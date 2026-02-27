# Aragora Loop-First Evolution Design

**Date:** 2026-02-26
**Status:** Approved
**Approach:** Loop-First (Aragora eats its own dogfood)

## Vision

Aragora accepts a vague, broad prompt — breaks it down, researches, asks clarifying questions, explains what the user needs to know, extends the inputs, crystallizes a structured spec, and executes it with a team of agents. The first user of this pipeline is Aragora itself.

## Core Insight

The self-improvement loop and the user-facing pipeline are the **same system**. The only difference is the target: when improving itself, the execution target is the Aragora codebase. When serving a user, the target is their project. This means every self-improvement cycle validates the entire product.

## Architecture

```
PROMPT INTAKE ("Make X better")
        │
INTERROGATION ENGINE
  ├── Decomposer: vague → concrete dimensions
  ├── Researcher: context from KM + Obsidian + web + codebase
  ├── Questioner: questions with explanations (debate-prioritized)
  └── Crystallizer: answers + extensions → structured MoSCoW spec
        │
ADVERSARIAL DEBATE (5+ agents, multi-provider)
        │
SPEC CRYSTALLIZATION (human approval gate, configurable)
        │
AGENT TEAM EXECUTION (skill-based selection, worktree isolation)
        │
VERIFICATION + LEARNING (tests, gauntlet, KM feedback)
```

## Component 1: Interrogation Engine

New module: `aragora/interrogation/`

**decomposer.py** — Takes vague prompt, identifies concrete dimensions. Extends existing `TaskDecomposer` in `aragora/nomic/task_decomposer.py`.

**researcher.py** — For each dimension, gathers context from:
- Current codebase state (file analysis)
- Prior decisions (KnowledgeMound query)
- User's notes (Obsidian vault sync)
- External landscape (web search)

**questioner.py** — Generates prioritized questions with:
- WHY each question matters
- Context so user can answer well
- Multiple-choice options where possible
- Hidden assumption flagging
- Uses mini-debate among agents to prioritize which questions matter most

**crystallizer.py** — Combines user answers with extended inputs to produce:
- Problem statement
- Requirements (must/should/could — MoSCoW)
- Non-requirements (explicitly out of scope)
- Success criteria (measurable)
- Risk register
- Implications user didn't state
- Constraints they'd want if they knew
- Prior art and alternatives

**Autonomy levels at this stage:**
- Fully autonomous: skip questions, infer answers from research
- Propose-and-approve: ask questions, present spec for approval
- Human-guided: iterate on spec until user satisfied
- Metrics-driven: use historical question effectiveness to decide what to ask

## Component 2: Agent Team Orchestration Evolution

**Skill-Aware Team Selection:** Pipeline Goal stage queries ELO rankings via `team_selector.py` to pick optimal agents per subtask. Code-generation tasks get high-code-ELO agents. Risk assessments get high-analytical-ELO agents. 80% built — needs pipeline to call team_selector at workflow generation time.

**Diversity Constraint:** Debate teams must include 2+ different model providers. `heterogeneous_consensus` flag promoted to default-on for Interrogation debate step.

**Learning Feedback:** After each pipeline run, record agent contributions via influence tracking (exists in `aragora/introspection/`). Feed back to ELO with phase-specific tags. Agents that consistently produce good specs get higher selection priority for spec tasks.

## Component 3: Obsidian Bidirectional Sync

**Vault Watcher** (`aragora/connectors/knowledge/obsidian_watcher.py`): Event-driven file watcher or HTTP webhook listener. Pushes `#aragora`-tagged notes to KnowledgeMound.

**HTTP Handlers** (6 endpoints):
- `POST /api/v1/obsidian/sync` — trigger full vault sync
- `GET /api/v1/obsidian/status` — sync health and last-sync time
- `POST /api/v1/obsidian/export/{receipt_id}` — push receipt to vault
- `GET /api/v1/obsidian/notes` — search synced notes
- `PUT /api/v1/obsidian/config` — configure vault path, tags, frequency
- `POST /api/v1/obsidian/connect` — initial vault connection handshake

**Pipeline Integration:** Interrogation Engine researcher queries Obsidian-synced notes. Pipeline completion exports decision receipt as Obsidian note with backlinks.

**Conflict resolution:** Obsidian is source of truth for user notes (read-only in Aragora). Aragora is source of truth for receipts/specs (tagged `aragora-generated`, not re-imported).

## Component 4: GUI — Prompt-to-Execution View

**Single-page progressive disclosure.** All stages visible but locked until previous completes. Each stage expands when active, collapses when done.

**Stages rendered inline:**
1. **Interrogation** — dimensions, research findings, Obsidian references, interactive questions
2. **Debate** — agents arguing in real-time, user can intervene
3. **Spec** — editable structured spec, approve/reject/modify
4. **Execution** — agent progress per worktree, pause/redirect controls

**Bottom bar controls:** Autonomy level toggle and execution target selector, always visible. Mid-run switching allowed.

**Obsidian references** surface as linked cards with title + date. Click opens via `obsidian://` URI.

**Coexists with** existing `/pipeline` canvas view for power users who want the node-editor.

**NOT in V1:** Multi-user collaboration, mobile layout, voice input.

## Component 5: Self-Improvement & Learning Flywheel

### Channel 1: Outcome Feedback → KnowledgeMound
Structured outcome record after each pipeline run:
- Prompt, spec quality score, execution success
- Agent contributions (per-agent influence)
- Time to completion, human intervention count
- Failure points, interrogation quality score

### Channel 2: Interrogation Calibration
Track which questions users change answers on vs. confirm. Learn to skip obvious questions, ask more in novel domains. Simple frequency table in KnowledgeMound.

### Channel 3: Agent ELO Evolution
Extend ELO updates from debate-level to pipeline-level with phase tags. Agent gets separate ELO for spec-writing, code-generation, review, etc. Team selector uses domain-specific ELO.

### Autonomy Modes

| Mode | Auto | Needs Approval |
|------|------|----------------|
| Fully autonomous | Everything including PR merge | Nothing (kill switch available) |
| Propose & approve (default) | Identify weaknesses, draft specs | Spec approval, PR merge |
| Human-guided | Execute on user-set goals | Spec approval, PR merge |
| Metrics-driven | Auto-fix regressions | New features need approval |

### Meta-Loop
Every N cycles (default 10, configurable), Aragora targets its own improvement system. Manual-triggered only in V1.

## Roadmap

### Wave 1 (Weeks 1-2): First Loop Closes
Type a prompt → Aragora breaks it down, asks questions, debates, specs, executes, creates PR.

- Interrogation Engine core (`aragora/interrogation/` — decomposer + questioner)
- Pipeline execution handoff (wire Execute → HardenedOrchestrator)
- GUI prompt-to-execution view (new page, progressive disclosure)
- WebSocket progress streaming (pipeline events → frontend)
- Worktree PR output (pipeline run → worktree → PR)

### Wave 2 (Weeks 3-4): Loop Gets Smarter
System learns from Wave 1 runs and picks better teams.

- Outcome feedback → KnowledgeMound (pipeline-level metrics)
- Dynamic team selection (Goal stage queries ELO)
- Diversity constraint (multi-provider default-on)
- Interrogation calibration (question effectiveness tracking)
- Obsidian HTTP surface (6 endpoints + vault watcher)

### Wave 3 (Weeks 5-6): Loop Eats Itself
Aragora runs self-improvement through its own GUI.

- Agent ELO per-phase (phase-tagged ELO updates)
- Researcher component (unified KM + Obsidian + web + codebase)
- Autonomy level controls (bottom bar, mid-run switching)
- Crystallizer (debate output → structured MoSCoW spec)
- Meta-loop trigger (manual "improve yourself" button)

### Wave 4 (Weeks 7-8): Polish & External Targets
Pipeline works for non-Aragora codebases.

- External repo targeting (clone + work on any git repo)
- User type presets (Founder/CTO/Team/Non-technical)
- Obsidian bidirectional (receipt export, backlinks)
- Input extension (implications, constraints, prior art)
- Spec editing in GUI (modify before approving execution)

## Milestones

- **End of Wave 1:** Use Aragora to improve Aragora through the GUI
- **End of Wave 2:** Each run makes the next run measurably better
- **End of Wave 3:** Aragora proposes its own improvements and implements them
- **End of Wave 4:** Anyone can point Aragora at a repo and go from idea to PR

## What's Unique (Not Solved by Others in 6-12 Months)

1. **Interrogation before execution.** No other tool runs an adversarial debate to figure out what questions to ask you before writing code.
2. **Multi-model truth-seeking.** Single-model copilots can't challenge their own assumptions. Aragora uses 5+ models from different providers to surface blind spots.
3. **Decision receipts.** Every decision has an auditable trail — who argued what, what evidence was cited, how consensus was reached.
4. **Self-improvement flywheel.** The tool that builds software also builds itself, and each cycle makes it measurably better at both.
5. **Obsidian as knowledge layer.** Your personal knowledge vault directly informs AI decisions — no other tool bridges personal PKM with multi-agent execution.
