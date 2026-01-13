# Nomic Loop Documentation

The Nomic Loop is aragora's autonomous self-improvement system—a society of AI agents that debates and implements improvements to its own codebase.

Note: The nomic loop is experimental. Run it in a sandbox and review changes before auto-commit.

## Overview

The name "nomic" comes from the game of Nomic, where players modify the rules of the game as part of gameplay. Similarly, the nomic loop allows AI agents to propose, debate, and implement changes to aragora itself.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NOMIC LOOP                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Phase 0: Context Gathering                                        │
│   ├── Claude (Claude Code) ──┐                                      │
│   ├── Codex (Codex CLI)      ├── Explore codebase, read docs        │
│   ├── Gemini (Kilo Code)     │                                      │
│   └── Grok (Kilo Code) ──────┘                                      │
│                                                                      │
│   Phase 1: Proposal                                                 │
│   └── All agents propose improvements based on context              │
│                                                                      │
│   Phase 2: Debate                                                   │
│   └── Arena orchestrates critique rounds                            │
│                                                                      │
│   Phase 3: Voting                                                   │
│   └── Agents vote on proposals (weighted by Elo rating)             │
│                                                                      │
│   Phase 4: Implementation                                           │
│   └── Winning agent implements via CLI tool                         │
│                                                                      │
│   Phase 5: Verification                                             │
│   ├── Syntax check (python -m py_compile)                           │
│   ├── Import check (python -c "import aragora")                     │
│   └── Test suite (pytest)                                           │
│                                                                      │
│   Phase 6: Commit (if all checks pass)                              │
│   └── Auto-commit with detailed message                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Running the Nomic Loop

### Basic Usage

```bash
# Run with streaming dashboard
NOMIC_AUTO_COMMIT=1 python scripts/run_nomic_with_stream.py run --cycles 24 --auto

# Run without streaming
NOMIC_AUTO_COMMIT=1 python scripts/nomic_loop.py --cycles 10 --auto
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--cycles N` | Number of improvement cycles | 10 |
| `--auto` | Auto-commit successful changes (requires `NOMIC_AUTO_COMMIT=1`) | False |
| `--human-approval` | Require human approval for commits | False |
| `--skip-genesis` | Skip codebase analysis | False |

### Safety Gates (Environment)

| Variable | Description | Default |
|----------|-------------|---------|
| `NOMIC_AUTO_COMMIT` | Allow `--auto` commits (must be `1` to enable) | `0` |
| `NOMIC_AUTO_CONTINUE` | Auto-continue non-interactive runs | `1` |
| `ARAGORA_ENABLE_FORKING` | Enable parallel forked debates in the nomic loop | `0` |

### Live Dashboard

The streaming version provides real-time visibility:

- **HTTP Dashboard**: `http://localhost:8080`
- **WebSocket Stream**: `ws://localhost:8765/ws`
- **Public View**: `https://aragora.ai`

## Cycle Phases

### Phase 0: Context Gathering

All 4 agents explore the codebase using their respective CLI tools:

- **Claude**: Uses Claude Code for deep analysis
- **Codex**: Uses OpenAI Codex CLI
- **Gemini**: Uses Google's Kilo Code
- **Grok**: Uses xAI's Grok CLI (via Kilo Code)

Each agent reads key files (nomic_loop.py, core.py, etc.) and identifies potential improvements.

### Phase 1: Proposal

Agents submit structured proposals:

```json
{
  "title": "Add semantic caching for debate context",
  "description": "Implement vector-based caching to reduce redundant LLM calls",
  "impact": "HIGH",
  "complexity": "MEDIUM",
  "files_affected": ["aragora/debate/orchestrator.py", "aragora/memory/cache.py"]
}
```

### Phase 2: Debate

The Arena orchestrates structured critique:

1. Each agent critiques other proposals
2. Severity scores (0-1) indicate issue importance
3. Suggestions are concrete and actionable
4. Multiple rounds until convergence or max rounds

### Phase 3: Voting

Agents vote on the best proposal:

- Votes are weighted by Elo rating
- Confidence scores (0-1) indicate certainty
- Reasoning is recorded for learning

### Phase 4: Implementation

The winning agent implements via CLI:

```bash
# Example: Claude implements
claude-code --task "Implement the approved proposal" --context proposal.json
```

Implementation is sandboxed and limited to approved file changes.

### Phase 5: Verification

Three-stage verification:

1. **Syntax Check**: `python -m py_compile` on all modified files
2. **Import Check**: `python -c "import aragora"` ensures no broken imports
3. **Test Suite**: `pytest tests/` must pass (currently 12,349 tests)

### Phase 6: Commit

If all checks pass:

```bash
git add -A
git commit -m "feat(nomic): <title>

<description>

Proposed by: <agent>
Approved by: <voters>
Cycle: N

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

## State Management

### State File

Located at `.nomic/nomic_state.json`:

```json
{
  "phase": "verify",
  "stage": "complete",
  "cycle": 3,
  "all_passed": true,
  "features_integrated": {
    "Phase 1": ["ContinuumMemory", "ReplayRecorder", ...],
    "Phase 2": ["ConsensusMemory", "InsightExtractor"],
    ...
  }
}
```

### Checkpoints

The CheckpointManager saves state for crash recovery:

```python
checkpoint.save("debate-123", {
    "phase": "debate",
    "round": 2,
    "messages": [...],
    "votes": [...]
})
```

Resume after crash:
```bash
python scripts/run_nomic_with_stream.py run --resume
```

### Backups

Before each cycle, critical files are backed up:

```
.nomic/backups/
├── backup_cycle_1_20260103_120000/
│   ├── scripts/nomic_loop.py
│   ├── aragora/core.py
│   └── ...
└── backup_cycle_2_20260103_130000/
    └── ...
```

## Agent Specialization

Each agent has a persona that evolves based on success:

| Agent | Role | Specialization |
|-------|------|----------------|
| **Claude** | Visionary | Architecture, design patterns |
| **Codex** | Engineer | Implementation, performance |
| **Gemini** | Strategist | Product vision, viral growth |
| **Grok** | Lateral Thinker | Creative solutions, edge cases |

Personas are managed by PersonaManager and evolve via PersonaLaboratory.

### Structured Thinking Protocols

Each agent uses a specialized thinking protocol to ensure high-quality proposals:

#### Claude (Visionary Architect)
```
1. EXPLORE: Understand current state - read files, trace code paths
2. PLAN: Design approach before implementing - consider alternatives
3. REASON: Show thinking step-by-step - explain tradeoffs
4. PROPOSE: Make concrete, actionable proposals with clear impact
```
Uses Claude Code's Explore/Plan modes for deep codebase understanding.

#### Codex (Pragmatic Engineer)
```
1. TRACE: Follow code paths to understand dependencies and data flow
2. ANALYZE: Identify patterns, anti-patterns, and improvement opportunities
3. DESIGN: Consider multiple implementation approaches with pros/cons
4. VALIDATE: Think about edge cases, tests, and failure modes
```
Shows reasoning chains: "I observed X → which implies Y → so we should Z"

#### Gemini (Product Strategist)
```
1. EXPLORE: Understand current state - what exists, what's missing
2. ENVISION: Imagine the ideal outcome - what would success look like
3. REASON: Show thinking step-by-step - explain tradeoffs
4. PROPOSE: Make concrete, actionable proposals with clear impact
```
Focuses on viral growth potential and developer excitement.

#### Grok (Lateral Synthesizer)
```
1. DIVERGE: Generate multiple unconventional perspectives
2. CONNECT: Find surprising links between disparate ideas
3. SYNTHESIZE: Combine insights into novel, coherent proposals
4. GROUND: Anchor creative ideas in practical implementation
```
Shows lateral thinking: "Others see X, but what if Y..."

## Integrated Features

The nomic loop integrates all 30+ features:

### Memory Systems
- ContinuumMemory (multi-timescale)
- MemoryStream (per-agent)
- ConsensusMemory (topic tracking)
- SemanticRetriever (embeddings)

### Evolution
- PersonaManager (traits)
- PromptEvolver (system prompts)
- PersonaLaboratory (A/B testing)
- EloSystem (skill tracking)

### Verification
- ProofExecutor (code verification)
- FormalVerificationManager (Z3)
- ReliabilityScorer (confidence)

### Resilience
- CheckpointManager (crash recovery)
- BreakpointManager (human intervention)
- DebateTracer (audit logs)

## Monitoring

### Logs

Main log at `.nomic/nomic_loop.log`:

```
=== NOMIC LOOP STARTED: 2026-01-03T13:52:49 ===
[13:52:50] NOMIC CYCLE 1
[13:52:50]   [replay] Recording cycle 1
[13:52:50]   [persona] claude-visionary: balanced
[13:52:54]   claude (Claude Code): exploring codebase...
```

### Metrics

Track via the dashboard or logs:

- Consensus rate per cycle
- Average rounds to convergence
- Test pass rate
- Feature implementation success rate

## Troubleshooting

### Loop Stuck in Phase

Check if an agent CLI tool is hung:
```bash
ps aux | grep -E "claude|codex|kilo"
```

Kill and resume:
```bash
pkill -f "codex exec"
python scripts/run_nomic_with_stream.py run --resume
```

### Tests Failing

1. Check the specific failure in `.nomic/nomic_loop.log`
2. Rollback to last backup if needed
3. Run tests manually: `pytest tests/ -v`

### Checkpoint Corruption

Reset state:
```bash
rm .nomic/nomic_state.json
rm .nomic/checkpoints.db
```

## Best Practices

1. **Run with `--auto`** for autonomous operation
2. **Monitor the live dashboard** for real-time visibility
3. **Review commits** even with auto-commit (can revert)
4. **Set cycle limits** appropriate for the task
5. **Use human-approval** for critical codebases
