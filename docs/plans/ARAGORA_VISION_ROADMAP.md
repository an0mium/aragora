# Aragora Vision Roadmap

> **Mission:** The Decision Integrity Platform that takes vague ideas through adversarial refinement to verified execution — orchestrating heterogeneous AI agents with cryptographic accountability, self-improving through its own debate infrastructure.

**Created:** Feb 26, 2026
**User Profile:** Non-technical decision maker who needs configurable autonomy (fully autonomous to propose-and-approve)
**Primary GUI:** Web app + bidirectional Obsidian sync
**Execution Target:** Start with self-improvement, generalize to any codebase/project

---

## Four Reinforcing Pillars

| Pillar | Core Value | Defensible Moat |
|--------|-----------|-----------------|
| **AI Truth-Seeking** | Heterogeneous multi-model debate reduces hallucinations 53%→23% (A-HMAD research) | Calibration data flywheel — every debate improves agent selection |
| **Idea-to-Execution** | Vague prompt → structured spec → verified implementation | Process-level verification of multi-agent reasoning (ThinkPRM) |
| **Agent Orchestration** | 43 agent types with ELO ranking, circuit breakers, fallback chains | Institutional memory compounds — impossible to replicate |
| **Decision Accountability** | Cryptographic receipts, provenance chains, EU AI Act compliance | Regulatory moat deepens with every decision recorded |

Each pillar feeds the others: truth-seeking improves execution quality, execution generates accountability data, accountability builds trust for more autonomous orchestration, and orchestration enables better truth-seeking.

---

## What Already Exists (Inventory)

**Backend (production-ready):**
- `IdeaToExecutionPipeline` — 4-stage orchestrator (Ideas → Goals → Actions → Orchestration), 1,173 LOC
- `MetaPlanner` — debate-driven goal prioritization with cross-cycle learning
- `TaskDecomposer` — hierarchical goal decomposition with oracle validation
- `AutonomousOrchestrator` — multi-agent coordination with minimal human intervention
- `HardenedOrchestrator` — canary tokens, output validation, review gates, sandbox
- `WorkflowEngine` — DAG-based automation with checkpointing and parallelism
- `GoalExtractor` — structural + AI modes for idea→goal transitions
- `ObsidianAdapter` — KM adapter #25, already registered
- `BrainDumpParser` — unstructured text → structured idea extraction
- 34 KM adapters, 43 agent types, 2,000+ API operations

**Frontend (implemented, needs polish):**
- 4 React Flow canvases (Idea, Goal, Action, Orchestration) with drag-and-drop
- `PipelineCanvas` + `StageNavigator` for cross-stage views
- WebSocket streaming (190+ event types)
- Landing page, Arena, Oracle, Spectate pages

**Research-validated techniques not yet integrated:**
- A-HMAD (Adaptive Heterogeneous Multi-Agent Debate) — 4-6% accuracy gains
- ThinkPRM (Process Reward Models) — verify reasoning steps, not just final answers
- AlphaEvolve patterns — self-improving code through evolutionary search
- DSPy-style programmatic prompt optimization
- MCP (Model Context Protocol) — de facto standard with 10,000+ tool servers

---

## Phase 0: Foundation Wiring (Now — Sprint 1)

**Goal:** Make existing infrastructure accessible end-to-end. No new systems — just connect what's built.

### 0.1 Prompt-to-Pipeline Gateway
Wire the "accept a vague prompt" entry point.

| Task | Files | What |
|------|-------|------|
| Natural language intake | `aragora/pipeline/intake.py` (new) | Accept free-text prompt, call `BrainDumpParser`, feed to `IdeaToExecutionPipeline.from_ideas()` |
| Interrogation loop | `aragora/pipeline/interrogator.py` (new) | Use `SwarmInterrogator` to ask clarifying questions before spec generation |
| Frontend intake UI | `aragora/live/src/app/(app)/pipeline/page.tsx` | Text input → interrogation chat → pipeline visualization |
| API endpoint | `aragora/server/handlers/pipeline/intake.py` | `POST /api/v1/pipeline/start` with `{ prompt: string, autonomy_level: 1-5 }` |

**Autonomy levels:**
1. **Propose-and-explain** — Show every decision, require approval for each stage
2. **Propose-and-approve** — Show summary, require approval at stage boundaries (default)
3. **Execute-and-report** — Run autonomously, report results with ability to rollback
4. **Fully autonomous** — Run end-to-end, notify on completion
5. **Continuous** — Monitor for new inputs, execute autonomously on triggers

### 0.2 Canvas Golden Path Buttons
Wire the existing canvases to the pipeline API.

| Task | Files | What |
|------|-------|------|
| "Generate Goals" button | `IdeaCanvas` → `POST /api/v1/canvas/pipeline/extract-goals` | One-click idea→goal transition |
| "Generate Actions" button | `GoalCanvas` → pipeline advance endpoint | Goal→action decomposition |
| "Run Pipeline" button | `PipelineCanvas` → `IdeaToExecutionPipeline.run()` | Full pipeline execution |
| Real-time canvas updates | `pipeline_stream.py` → WebSocket → canvas state | Watch agents work in real-time |
| Stage transition animations | Canvas UI | Visual provenance links between stages |

### 0.3 Obsidian Bidirectional Sync
Connect the existing `ObsidianAdapter` to a sync service.

| Task | Files | What |
|------|-------|------|
| Vault watcher | `aragora/connectors/obsidian/watcher.py` (new) | Watch Obsidian vault for changes, extract ideas from markdown |
| Sync service | `aragora/connectors/obsidian/sync.py` (new) | Bidirectional: notes→ideas, results→notes |
| Obsidian plugin spec | `docs/integrations/OBSIDIAN_PLUGIN.md` (new) | Community plugin design for Obsidian sidebar panel |
| KM bridge | Wire `ObsidianAdapter` through `KnowledgeBridgeHub` | Obsidian notes as first-class knowledge source |

**Sync pattern:** File watcher detects `.md` changes → parse frontmatter for `aragora-id` tags → upsert into KnowledgeMound → pipeline results write back as new notes with provenance links in frontmatter.

### 0.4 Verification: Phase 0 Complete When
- [ ] User can type a vague prompt in the web UI and see it decomposed into ideas
- [ ] Ideas flow through all 4 canvas stages with one-click transitions
- [ ] Results appear as annotated notes in an Obsidian vault
- [ ] Autonomy level selector controls how much approval is required
- [ ] All existing tests continue passing

---

## Phase 1: Intelligence Layer (Sprint 2-3)

**Goal:** Make the AI transitions genuinely useful, not generic decomposition.

### 1.1 Debate-Driven Stage Transitions
Use the debate engine for every stage transition (not just final decisions).

| Task | What |
|------|------|
| Idea→Goal debate | Run mini-debate about which goals are most impactful from idea cluster |
| Goal→Action debate | Agents argue about best implementation approach per goal |
| Action→Orchestration debate | Debate agent assignment, parallelism strategy, risk assessment |
| Transition receipts | Every stage transition generates a Decision Receipt with debate summary |

### 1.2 Process-Level Verification (ThinkPRM Integration)
Verify reasoning at each step, not just the final output.

| Task | What |
|------|------|
| Step verifier | `aragora/reasoning/step_verifier.py` (new) — validate each reasoning step in agent proposals |
| Confidence calibration | Use calibration data from past debates to score confidence at each step |
| Reasoning trace UI | Show verified/unverified reasoning steps in the canvas with color coding |
| Auto-flag weak reasoning | If step verification fails, auto-trigger re-debate on that specific step |

### 1.3 Programmatic Prompt Optimization (DSPy Patterns)
Optimize prompts systematically, not through manual iteration.

| Task | What |
|------|------|
| Prompt registry | `aragora/prompts/registry.py` — version-controlled prompt templates with A/B metrics |
| Auto-optimizer | `aragora/prompts/optimizer.py` — optimize prompts based on outcome data |
| Feedback integration | User ratings on generated goals/actions feed back to prompt optimization |
| Domain templates | Prompt templates tuned per vertical (healthcare, financial, legal) |

### 1.4 Verification: Phase 1 Complete When
- [ ] Every stage transition involves at least a 2-agent debate
- [ ] Reasoning steps are individually verified with confidence scores
- [ ] Prompts improve measurably over 10+ pipeline runs
- [ ] Domain-specific templates produce noticeably better output than generic

---

## Phase 2: Self-Improvement Engine (Sprint 3-4)

**Goal:** Aragora can improve itself using its own pipeline infrastructure.

### 2.1 Meta-Pipeline (Self-Improving Aragora)
The pipeline improves the pipeline.

| Task | What |
|------|------|
| Self-scan → ideas | `MetaPlanner.scan()` generates improvement ideas from codebase signals, test failures, user feedback |
| Ideas → goals via debate | Run debate about which improvements are most impactful |
| Goals → implementation | `TaskDecomposer` + `AutonomousOrchestrator` execute changes |
| Verify → learn | Test results, user feedback, and performance metrics feed back into next cycle |

**AlphaEvolve patterns to apply:**
- Evolutionary candidate generation: generate multiple implementation candidates per goal
- Automated evaluation: run test suites + performance benchmarks as fitness functions
- Cross-pollination: successful patterns from one module inform improvements in others

### 2.2 External Signal Injection
The self-improvement system needs to know what matters.

| Task | What |
|------|------|
| User feedback ingest | Ratings, feature requests, usage patterns → prioritization signals |
| Business metrics | Pipeline completion rates, user retention, feature adoption → priority weighting |
| Market signals | Competitor analysis, research papers, ecosystem changes → opportunity detection |
| Obsidian-sourced goals | User notes tagged `#aragora-improve` → auto-create improvement goals |

### 2.3 Harness Generalization
Enable the pipeline to execute against any codebase, not just Aragora.

| Task | What |
|------|------|
| Claude Code harness | Already exists — `aragora/harnesses/claude_code.py` with CLAUDE.md injection |
| Codex harness | Already exists — `aragora/harnesses/codex.py` |
| Generic Git harness | `aragora/harnesses/git_repo.py` (new) — clone, branch, execute, PR |
| MCP tool integration | `aragora/harnesses/mcp_bridge.py` (new) — access 10,000+ MCP tool servers |

### 2.4 Verification: Phase 2 Complete When
- [ ] Aragora can run `self-improve "improve error handling"` and produce a working PR
- [ ] Self-improvement prioritizes based on user feedback, not just code metrics
- [ ] The pipeline can execute against an external GitHub repository
- [ ] Each self-improvement cycle produces a Decision Receipt

---

## Phase 3: User Experience Polish (Sprint 4-5)

**Goal:** Non-technical users can use the full pipeline without documentation.

### 3.1 Onboarding Wizard
Guide users from "I have a vague idea" to "agents are executing."

| Task | What |
|------|------|
| Welcome flow | "What are you trying to decide?" → category selection → guided pipeline creation |
| Template library | 20+ pre-built templates: hiring decisions, product launches, market entry, compliance audits, code reviews, architecture decisions |
| Persona configuration | "How technical are you?" → adjust UI complexity, terminology, default autonomy level |
| Interactive tutorial | Walk through a sample pipeline with real AI agents |

### 3.2 Real-Time Execution Dashboard
Show what agents are doing in real-time.

| Task | What |
|------|------|
| Agent activity feed | Live stream of agent proposals, critiques, revisions |
| Cost tracker | Running cost display per pipeline, per stage, per agent |
| Confidence dashboard | Aggregate confidence scores across pipeline stages |
| Intervention points | "Agents disagree here — your input needed" notification system |

### 3.3 Obsidian Plugin (Community)
Native Obsidian integration beyond file sync.

| Task | What |
|------|------|
| Sidebar panel | Show active pipelines, recent decisions, pending approvals |
| Inline commands | `{{aragora: debate this}}` in notes triggers debate |
| Graph view integration | Overlay pipeline provenance on Obsidian's graph view |
| Template insertion | Insert decision receipts as formatted notes |

### 3.4 Verification: Phase 3 Complete When
- [ ] New user can go from signup to first pipeline result in < 5 minutes
- [ ] Non-technical user can understand and intervene in agent debates
- [ ] Obsidian users can trigger and monitor pipelines without leaving the vault
- [ ] Cost is visible and controllable at every stage

---

## Phase 4: Competitive Differentiation (Sprint 5-6)

**Goal:** Build features that are structurally difficult for large AI labs to replicate.

### 4.1 Heterogeneous Debate as a Service
Package the debate engine as an API product.

| Task | What |
|------|------|
| Debate API | `POST /api/v1/debate/run` with configurable agents, rounds, consensus threshold |
| Embeddable widget | `<aragora-debate topic="..." />` web component |
| Webhook integration | Zapier/n8n triggers on debate completion |
| SDK methods | `aragora.debate("Should we launch?", agents=["claude", "gpt4", "gemini"])` |

### 4.2 Calibration Data Flywheel
Every debate makes the system smarter.

| Task | What |
|------|------|
| ELO + outcome tracking | Track which agent combinations produce verified-correct outputs |
| Automatic team selection | Best agents auto-selected per domain based on historical performance |
| Calibration export | Organizations can export their calibration data as competitive advantage |
| Cross-org learning | Anonymized calibration patterns shared across tenants (opt-in) |

### 4.3 Decision Receipt Marketplace
Cryptographic accountability as a feature.

| Task | What |
|------|------|
| Receipt templates | Industry-specific receipt formats (SOX, HIPAA, EU AI Act, ISO 27001) |
| Third-party verification | External auditors can verify receipt integrity without access to full data |
| Receipt chain | Link related decisions into a provenance chain |
| Export formats | PDF, JSON-LD, blockchain-anchored, API-queryable |

### 4.4 Institutional Memory Compound Interest
The Knowledge Mound grows more valuable over time.

| Task | What |
|------|------|
| Pattern extraction | Auto-detect recurring decision patterns across pipelines |
| Precedent search | "Has this org faced a similar decision before?" with semantic search |
| Contradiction detection | Flag when new decisions conflict with past decisions |
| Learning curves | Show how decision quality improves over time |

### 4.5 Verification: Phase 4 Complete When
- [ ] Debate API serves external consumers
- [ ] Agent team selection is measurably better after 100+ debates
- [ ] Decision receipts pass third-party audit verification
- [ ] Knowledge Mound demonstrates measurable compound improvement

---

## Phase 5: Ecosystem and Scale (Sprint 6+)

**Goal:** Platform effects that create sustainable competitive advantage.

### 5.1 Connector Ecosystem
Every tool integration is a switching cost.

| Task | What |
|------|------|
| MCP server | Aragora as MCP tool server (expose debate, pipeline, receipts to any MCP client) |
| MCP client | Aragora consumes MCP tools (file systems, databases, APIs, browsers) |
| CRM connectors | HubSpot, Salesforce — decision context from customer data |
| BI connectors | Tableau, Looker — decision context from analytics |

### 5.2 Multi-Tenant Self-Improvement
Each organization's usage improves the platform.

| Task | What |
|------|------|
| Federated learning | Cross-tenant calibration without data sharing |
| Vertical specialization | Healthcare orgs contribute to healthcare-specific improvements |
| Community templates | User-created pipeline templates in marketplace |
| Open-source agents | Community-contributed agent configurations |

### 5.3 Regulatory Positioning
Position Aragora as the compliance layer for AI decisions.

| Task | What |
|------|------|
| EU AI Act toolkit | Auto-generate required documentation from decision receipts |
| SOX compliance | Audit trail generation for financial decisions |
| HIPAA decision log | Healthcare-specific decision accountability |
| ISO 42001 | AI management system standard alignment |

---

## Implementation Priority Matrix

```
                    High Impact
                        │
         Phase 0.1      │     Phase 1.1
         (Intake)       │     (Debate transitions)
                        │
         Phase 0.2      │     Phase 2.1
         (Canvas wiring)│     (Meta-pipeline)
                        │
Low Effort ─────────────┼──────────────── High Effort
                        │
         Phase 0.3      │     Phase 3.3
         (Obsidian sync)│     (Obsidian plugin)
                        │
         Phase 4.2      │     Phase 5.2
         (Calibration)  │     (Multi-tenant)
                        │
                    Low Impact
```

**Recommended execution order:**
1. **Phase 0** (Foundation) — Unlock all existing infrastructure for end users
2. **Phase 1.1-1.2** (Debate transitions + verification) — Quality that builds trust
3. **Phase 2.1-2.2** (Self-improvement + signals) — Aragora improves itself
4. **Phase 3.1-3.2** (Onboarding + dashboard) — User adoption
5. **Phase 0.3 + 3.3** (Obsidian) — Differentiated integration
6. **Phase 4** (Competitive moats) — Lock in advantages
7. **Phase 5** (Ecosystem) — Platform effects

---

## Key Technical Decisions

### 1. Autonomy Levels Are First-Class
Every pipeline operation checks `autonomy_level` before proceeding. Default is level 2 (propose-and-approve). The setting is per-pipeline, overridable per-stage, and stored in user preferences.

### 2. Debate at Every Stage Transition
Not just at the final decision. This is the core differentiator — no other tool runs adversarial AI debate at every stage of idea→execution.

### 3. Process-Level Verification
Verify each reasoning step (ThinkPRM pattern), not just the final output. This catches hallucinations and weak reasoning before they cascade.

### 4. Obsidian as Knowledge Layer, Not UI Layer
Obsidian syncs with the Knowledge Mound. The web app is the primary UI. This avoids building an Obsidian plugin as a critical-path dependency.

### 5. Self-Improvement Starts with Self
Aragora improves itself first (dogfooding). Once proven, the same infrastructure generalizes to external codebases. This ensures the pipeline is battle-tested before external users rely on it.

### 6. Receipts Are Not Optional
Every pipeline execution produces a Decision Receipt. This is the accountability layer that makes enterprise adoption viable and regulatory compliance automatic.

---

## Metrics

### Phase 0 Success
- Pipeline start-to-finish in < 30 seconds for simple prompts
- 100% of existing tests passing
- Obsidian sync latency < 5 seconds

### Phase 1 Success
- Stage transition quality rated > 4/5 by users
- Reasoning step verification catches > 80% of hallucinated steps
- Prompt optimization shows > 10% improvement over 50 runs

### Phase 2 Success
- Self-improvement produces merged PRs with zero human code edits
- External codebase execution works on 3+ open-source repos
- User feedback signals influence > 50% of self-improvement priorities

### Phase 3 Success
- New user to first result in < 5 minutes
- Non-technical user satisfaction score > 4/5
- Obsidian users can complete full pipeline without web app

### Phase 4 Success
- Debate API serves > 100 external requests/day
- Agent team selection outperforms random by > 20%
- Decision receipts pass third-party audit

---

## What This Is NOT

- **Not another LangGraph/CrewAI.** Those are orchestration frameworks for developers. This is a decision platform for anyone.
- **Not another Obsidian plugin.** Obsidian is one knowledge source among many. The platform is web-first.
- **Not another AI code generator.** Code generation is one execution modality. The platform handles any kind of decision→execution.
- **Not enterprise-only.** Default configuration targets SMBs (10-200 people). Enterprise features are additive.
