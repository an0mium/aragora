# Prompt-to-Specification Market Analysis

## Date: 2026-02-27

## Context

This analysis synthesizes three inputs:
1. **4 weeks of user prompt patterns** extracted from Claude Code sessions (415 scored prompts, 37 high-intent)
2. **Nate's "Prompting just split into 4 different skills" essay and video** (Feb 27, 2026) — a framework for post-autonomous-agent prompting
3. **Aragora's product definition and evolution roadmap** — the platform's existing architecture and strategic direction

The goal: identify Aragora's market position at the intersection of what users actually need, what the industry is discovering, and what Aragora uniquely provides.

---

## Part 1: What Users Actually Do (Prompt Pattern Analysis)

### Methodology

Extracted 211,287 user messages from 28 days of Claude Code sessions. After filtering system-generated content, low-content messages, deduplication, and LLM-based intent scoring, 191 substantive prompts remained (18 score-5, 19 score-4, 154 score-3).

### The Eight Recurring Themes

**1. "What are the best next steps" — Strategic Self-Direction**
The most frequent pattern. The user repeatedly asks the AI system to assess the project holistically and determine highest-leverage next steps. This isn't indecision — it's attempting to make the system itself do strategic planning. The user wants a system that can answer "what should I do next?" better than a single agent can.

**2. Integration / Coordination / Synergy / Cross-Pollination**
This cluster of words appears in nearly every strategic prompt. The core need: hundreds of sophisticated subsystems exist but are not wired together into a coherent product experience. The user wants stranded features surfaced, connected, and leveraged — not more new features.

**3. Vague Prompt → Spec → Execution Pipeline**
The clearest articulation (Feb 15): "My goal is for the aragora codebase to accept vague underspecified input such as 'Maximize utility of this codebase to small and medium scale enterprises'..." This is both the product vision AND how the user wants to use Aragora personally.

**4. Self-Improvement / Nomic Loop / Dogfooding**
The user wants Aragora to improve itself, and wants proof it works by dogfooding. Key quote (Feb 25): "we haven't actually demonstrated by dogfooding that aragora works for anything useful." Trust through demonstration, not architecture.

**5. aragora.ai Working End-to-End**
Frequent deployment/infrastructure prompts (AWS, Vercel, Cloudflare, secrets management). The goal: a live site that actually works for visitors.

**6. Business-Grade Quality / SME Focus**
Landing page should be professional, debate answers business-oriented (not essay-focused), Oracle is a separate personality. The user wants a product, not a research demo.

**7. Security Hardening Before Public Exposure**
Automated pentesting, secrets management, rate limiting — do this BEFORE inviting users.

**8. Multi-Agent Orchestration That Actually Delivers**
Key quote (Feb 26): "can we test the swarm — you ask me questions about what aragora should do next, assign a task to the swarm, and see if aragora can execute." Demonstrated capability, not theoretical architecture.

### The Meta-Pattern

The prompts form a spiral: **assess → prioritize → execute → assess again**. The user is trying to get Aragora to do this spiral autonomously. The product IS the process being used to build it.

---

## Part 2: Nate's Four-Discipline Framework

### Source
"Prompting just split into 4 different skills. You're probably practicing 1 of them" — Nate's Substack, Feb 27, 2026 (paid article + video transcript)

### The Framework

Nate argues that "prompting" now hides four distinct disciplines, each operating at a different altitude and time horizon:

**Discipline 1: Prompt Craft** (Table Stakes)
- Synchronous, session-based, individual
- Clear instructions, examples, constraints, output format
- Was the whole game in 2024-2025
- Now table stakes — "the person in 1998 who couldn't send an email"

**Discipline 2: Context Engineering** (Where Industry Attention Is Now)
- Curating the entire information environment an agent operates within
- System prompts, tool definitions, retrieved documents, memory, MCP connections
- Your 200-token prompt is 0.02% of what the model sees; the other 99.98% is context
- Produces CLAUDE.md files, RAG pipelines, memory architectures
- "People who are 10x more effective aren't writing 10x better prompts — they've built 10x better context infrastructure"

**Discipline 3: Intent Engineering** (Emerging)
- Encoding organizational purpose: goals, values, tradeoff hierarchies, decision boundaries
- Context tells agents what to know; intent tells agents what to want
- The Klarna case: 2.3M conversations resolved, $40M savings projected, but customer satisfaction cratered because the agent optimized for speed when the org valued relationships
- "You can have perfect context and terrible intent alignment"

**Discipline 4: Specification Engineering** (Newest, Almost Nobody Talking About It)
- Writing documents that autonomous agents can execute against over extended time horizons
- Complete, structured, internally consistent descriptions of outcomes, quality measures, constraints, tradeoffs, and "done"
- Anthropic's own discovery: Opus 4.5 fails to build a production app from "build a clone of claude.ai" — the fix was specification patterns, not a better model
- "The specification became the scaffolding that let multiple agents produce coherent output over days"
- Mirrors the verbal-instructions-to-blueprints transition in human engineering

### Nate's Five Primitives

1. **Self-Contained Problem Statements** — Lütke's insight: state a problem with enough context that it's plausibly solvable without additional information
2. **Acceptance Criteria** — If you can't describe what "done" looks like, the agent stops at whatever its heuristics say is complete (the 80% problem)
3. **Constraint Architecture** — Musts, must-nots, preferences, and escalation triggers
4. **Decomposition** — Break into independently executable, testable, integratable components
5. **Evaluation Design** — Build evals with known-good outputs; run them systematically

### Nate's Recommended Learning Sequence

- Month 1: Close prompt craft gaps (reread docs, build baselines)
- Month 2: Build personal context layer (CLAUDE.md for your work)
- Month 3: Practice specification engineering (real project, full spec before touching AI)
- Month 4+: Build intent infrastructure (organizational decision frameworks)

### The Lütke Connection

Shopify CEO Tobi Lütke's key insight: the discipline of context engineering — stating problems with enough context that they're plausibly solvable without additional information — made him a better CEO, not just a better AI user. What companies call "politics" is often bad context engineering — buried disagreements about assumptions that nobody surfaced explicitly.

---

## Part 3: Aragora's Market Position

### The Core Insight

Nate's prescription: **people need to learn these four disciplines.** Four-month roadmap. Practice specification engineering. Build constraint architecture.

Aragora's thesis is the opposite and more commercially interesting: **most people won't learn these disciplines.** They are lazy, impatient, inarticulate, lacking domain knowledge. They have some idea of what they want but cannot express it with the precision autonomous agents require.

The market opportunity: **automate the ascent through all four layers.**

### How Aragora Maps to Nate's Stack

| Nate's Discipline | Nate's Prescription | What Aragora Does Instead |
|---|---|---|
| Prompt Craft | Learn to write clear prompts | Accept that users won't; take vague input as-is |
| Context Engineering | Build CLAUDE.md, load context manually | Auto-build from Obsidian, business data, Knowledge Mound, memory tiers, enterprise integrations |
| Intent Engineering | Manually encode org values and tradeoff hierarchies | **Interrogation engine** — extract intent from vague humans via adversarial questioning |
| Specification Engineering | Write complete specs before agents start | **Pipeline** — auto-generate adversarially-validated specs from extracted intent + context |

Nate tells people to climb the stack. **Aragora IS the stack.**

### What Nate Gets Right That Validates Aragora's Direction

**1. The interrogation pattern is already Anthropic's recommendation.**
Nate quotes Anthropic's best practice: "Interview me in detail. Ask about technical implementation, UI/UX, edge cases, concerns, and tradeoffs... Keep interviewing until we've covered everything, then write a complete spec." This is literally Aragora's Idea-to-Execution pipeline — the interrogation → crystallization → specification flow.

**2. The 80% problem is real and it's a spec problem.**
66% of developers cite "AI solutions that are almost right but not quite" as their top frustration. Aragora's pipeline addresses this by refusing to execute until the spec is good enough — the interrogation phase forces specification quality before agents start working.

**3. Self-contained problem statements are the bottleneck.**
Lütke's core insight — "state a problem with enough context that it's plausibly solvable without additional information" — is exactly what most users can't do. Aragora's interrogation engine is the mechanism that forces this to happen even when the user won't do it themselves.

**4. The enterprise specification engineering opportunity is massive.**
Nate's vision of "your entire organizational document corpus should be agent-readable" is exactly what Aragora's Knowledge Mound + 34 adapters + Obsidian sync is building toward. The one-person business advantage Nate describes ("just convert your Notion to be agent-readable and you're off to the races") is precisely Aragora's SME value proposition.

### What Nate Misses That Aragora Should Exploit

**1. Single-author specification has blind spots.**
Nate's framework is individual — one person learning to write better specs. Aragora's multi-agent adversarial debate means the spec itself gets stress-tested before execution. Six agents arguing about the spec surface blind spots that a single human writing alone would miss.

**2. Truth-seeking is absent from Nate's framework.**
Nate's framework is about productivity — getting work done faster. He doesn't address the fundamental problem that specifications can be well-formed but wrong. Aragora's adversarial debate protocol addresses epistemic quality, not just specification completeness. Prover-Estimator protocols, cross-verification, persuasion-vs-truth scoring — these have no equivalent in Nate's framework.

**3. Evaluation is treated as a human responsibility.**
Nate's Primitive Five (Evaluation Design) is "build test cases, run them periodically." Aragora's Gauntlet + settlement hooks + calibration tracking + Brier scoring is that system made autonomous and continuous.

**4. There's no feedback loop.**
Nate's four-month roadmap is linear: learn craft → build context → practice specs → encode intent. No feedback mechanism. Aragora's Nomic Loop closes this — the system evaluates its own output, debates improvements, implements them, verifies. The evaluation primitive made autonomous and self-correcting.

**5. He assumes individual adoption, not platform delivery.**
Nate teaches individuals to build their own context infrastructure, intent frameworks, and specification skills. Aragora can deliver this as a platform — the interrogation engine, the knowledge integration, the adversarial spec validation, and the orchestrated execution are all product features, not personal skills the user needs to develop.

---

## Part 4: Aragora's Product Definition (Revised)

### What Aragora Is

Aragora is a Decision Intelligence Platform — it orchestrates 43 AI agent types in adversarial multi-model debates to vet decisions, then produces cryptographically signed audit trails.

Key differentiators:
- **Multi-agent adversarial debate**: Different AI models argue for/against proposals, with consensus detection, dissent preservation, and calibration tracking
- **Idea-to-execution pipeline**: Vague prompts → goals → workflows → orchestrated agent execution
- **Self-improvement loop (Nomic Loop)**: The system debates improvements to itself, implements them, verifies, and commits
- **Decision accountability**: Cryptographic receipts, provenance chains, EU AI Act compliance artifacts
- **Knowledge integration**: 34 Knowledge Mound adapters, Obsidian connector, memory tiers, Bayesian belief networks

### What Aragora Should Become (Market-Aligned)

**Aragora is the specification engineering layer for people who won't do specification engineering themselves.**

The full pipeline:
1. **Accept vague input** — what real users actually produce (lazy, inarticulate, impatient)
2. **Interrogate to extract intent** — adversarial questioning surfaces hidden assumptions, constraints, and priorities the user didn't articulate
3. **Build context automatically** — from Obsidian, business data, Knowledge Mound, memory tiers, enterprise integrations, the user's prompt history and behavioral patterns
4. **Generate adversarially-validated specifications** — multi-agent debate stress-tests the spec before execution; truth-seeking protocols catch specs that are well-formed but wrong
5. **Execute against the spec with orchestrated agent teams** — the swarm/orchestrator with human observability and control plane
6. **Evaluate and improve continuously** — Nomic Loop, Gauntlet, settlement hooks, calibration tracking

### The User's Vision Statement

> "I want aragora to be able to accept a vague broad prompt like this, break it down, research and refine, interrogate me ask me all relevant questions, explain all I need to know to answer, then extend the inputs, turn into a well defined software spec and implement it"

This maps perfectly onto Nate's framework — but instead of teaching the user to do all four disciplines, Aragora does them automatically:
- "accept a vague broad prompt" = accepting Discipline 1 input quality
- "interrogate me, ask me all relevant questions" = automating Discipline 3 (Intent Engineering)
- "research and refine, extend the inputs" = automating Discipline 2 (Context Engineering)
- "turn into a well defined software spec" = automating Discipline 4 (Specification Engineering)
- "and implement it" = orchestrated execution against the spec

### Autonomy Configuration (User-Selected)

| Level | Default | Description |
|---|---|---|
| Fully Autonomous | ✓ (for self-improvement) | Identifies weaknesses, debates improvements, implements and ships |
| Propose-and-Approve | ✓ (for new features) | Proposes with reasoning, user approves/rejects |
| Human-Guided | | User sets goals and constraints, Aragora finds path |
| Metrics-Driven | | Auto-fixes regressions, new features require approval |

---

## Part 5: Strategic Implications

### Immediate Priority: Dogfood the Pipeline

The highest-leverage thing Aragora can do right now: **make the interrogation → spec → execution pipeline actually work end-to-end and demonstrate it by dogfooding.** Not more infrastructure. Not more adapters.

The user's most honest prompt (Feb 25): "we haven't actually demonstrated by dogfooding that aragora works for anything useful."

Nate's essay gives the vocabulary. The prompt history gives proof of demand. The pipeline exists in partial form. Ship it.

### Competitive Moat

What existing AI labs, large software companies, and open source projects are NOT building in the next 6-12 months:

1. **Adversarial specification validation** — nobody else debates the spec before executing it
2. **Truth-seeking protocols integrated into the pipeline** — Prover-Estimator, cross-verification, persuasion-vs-truth scoring
3. **Self-improving specification quality** — the Nomic Loop applied to the pipeline itself, not just the codebase
4. **Decision accountability with cryptographic receipts** — audit trails for every decision the pipeline makes
5. **Multi-model adversarial consensus** — using model disagreement as signal, not noise

### The Marc Schluper Objection

A commenter on Nate's essay raised the real-world objection: "Nobody can write a specification upfront — we humans also have a limited context window... We need incremental development."

Aragora's interrogation pipeline answers this directly. The spec IS developed incrementally, through adversarial questioning and iterative refinement. The user doesn't need to produce a complete specification from nothing — they need to answer questions, make choices, and provide feedback while the system builds the spec around their intent.

### What One-Person Businesses Need (Nate's Insight)

> "One-person businesses have the greatest advantage right now because if you are a one-person business and you can just convert your Notion to be agent-readable, you're off to the races today."

Aragora's Obsidian sync + Knowledge Mound + interrogation pipeline is exactly this — but with adversarial validation that Notion-to-agent pipelines lack.

---

## Appendix: Source Materials

- **Prompt extraction tool**: `scripts/extract_and_rank_prompts.py`
- **Scoring output**: 191 prompts scored 3+ on 1-5 intent scale
- **Essay source**: Nate's Substack, "Prompting just split into 4 different skills" (Feb 27, 2026, paid)
- **Video source**: Accompanying YouTube transcript (~40 min)
- **Aragora definition**: From prior strategic planning session
- **User Q&A**: From Aragora Evolution Roadmap planning session (see `docs/plans/ARAGORA_EVOLUTION_ROADMAP.md`)
- **Key references from essay**: Anthropic Feb 2026 agent autonomy study, Tobi Lütke on Acquired podcast (Sep 2025), Klarna AI agent deployment, TELUS 13,000 AI solutions, Zapier 800+ internal agents
